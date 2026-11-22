"""
GEPA optimization for the bullet-point summarizer.

Objective (weighted sum, maximize):
  + semantic_preservation : bullets vs. post semantic cosine sim
  - style_leakage         : bullets vs. post style cosine sim  (we want this LOW)
  - length_penalty        : bullets length / post length       (we want this LOW)

The style-stripping term is the key non-obvious one — it directly measures
whether the bullets carry the student's writing style, which is exactly what
we DON'T want.  If the bullets encode style, the cluster ID conditioning
becomes redundant at inference time.

Training data: Stack Exchange posts (we can't train on student data because
the released artifact — the prompt — would otherwise encode student-specific
patterns).  Validation on a small set of student posts to confirm the
optimized prompt generalizes.

We optimize the prompt for a local open-source model (Qwen-2.5-7B-Instruct
by default) using GPT-5 as the reflection LM, mirroring the rewriter setup.
The final released artifact is the optimized prompt + base model weights —
no API dependency at inference time, and no memorization of student data by
construction.
"""

import argparse
import json
import os
import pickle
import random

import dotenv
import dspy
import numpy as np
import pandas as pd
from dspy import GEPA
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()
random.seed(42)


# ---------------------------------------------------------------------------
# Metric weights — tune these based on what spot-checks look like.
#
#   sim_weight:    how much to reward semantic content preservation
#   style_weight:  how much to penalize stylistic leakage (higher = stricter)
#   length_weight: how much to penalize verbosity
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "sim_weight": 1.0,
    "recon_weight": 0.5,
    "style_weight": 0.6,
    "length_weight": 0.4,
}

TARGET_COMPRESSION = 0.30   # bullets ~30% of post length; anything above penalizes


# ---------------------------------------------------------------------------
# Signature + module
# ---------------------------------------------------------------------------
class BulletSummarizerSignature(dspy.Signature):
    """Summarize the given discussion forum post into bullet points that \
    capture what the poster is asking or doing. Strip stylistic phrasing \
    (hedging, filler, personal voice) while preserving the core content. \
    Preserve code snippets and formulas exactly."""
    post: str = dspy.InputField(desc="The original discussion forum post.")
    bullets: str = dspy.OutputField(
        desc="Bullet points capturing the post's content in neutral language.")


# ---------------------------------------------------------------------------
# In-context demos — one per failure mode identified in spot-checks:
#
#  1. Comparison question  → capture the question itself, not just the topic
#  2. Opinion / affirmation → declarative bullets, NOT imperatives
#  3. Technical help-request → describe the problem then "Requests X"
#  4. Rebuttal / argument  → flag the argumentative stance explicitly
#  5. Social / encouragement → report what was said; no invented requests
# ---------------------------------------------------------------------------
IN_CONTEXT_DEMOS = [
    # 1. Comparison question — the ask must appear in the bullets
    {
        "post": (
            "My question is about the difference between standard deviation "
            "and standard error. Standard deviation measures the spread of "
            "the data, but I've seen standard error described as the standard "
            "deviation of the sampling distribution. Are these the same thing "
            "or are they measuring different things?"
        ),
        "bullets": (
            "- Asks: whether standard deviation and standard error are the "
            "same measure or distinct\n"
            "- Standard deviation: spread of the data\n"
            "- Standard error: described as standard deviation of the "
            "sampling distribution"
        ),
    },
    # 2. Opinion / affirmation — use declarative present tense, never imperatives
    {
        "post": (
            "This lecture really helped clarify things for me! The "
            "visualization in particular made the abstract concepts much more "
            "concrete. I feel like I finally understand why regularization "
            "prevents overfitting."
        ),
        "bullets": (
            "- Found the lecture helpful, especially the visualization\n"
            "- Now understands why regularization prevents overfitting"
        ),
    },
    # 3. Technical help-request — describe problem then state the ask
    {
        "post": (
            "The videos are constantly buffering for me. I don't know whether "
            "it's my internet connection or a server issue. It's taken me twice "
            "as long to get through the material. I'm worried I won't finish "
            "before the deadline. Any suggestions?"
        ),
        "bullets": (
            "- Videos constantly buffer; unclear if internet or server issue\n"
            "- Progress delayed; may not finish before deadline\n"
            "- Requests suggestions"
        ),
    },
    # 4. Rebuttal / argument — preserve the counter-claim framing
    {
        "post": (
            "I disagree with the earlier post. Correlation does not imply "
            "causation. Just because two variables move together does not mean "
            "one causes the other — there are always potential confounders. "
            "You cannot draw causal conclusions from observational data alone."
        ),
        "bullets": (
            "- Argues, contra earlier post, that correlation does not imply "
            "causation\n"
            "- Co-movement between variables does not establish causality; "
            "confounders must be considered\n"
            "- Causal conclusions require more than observational data"
        ),
    },
    # 5. Social / encouragement — report what was said; do not invent a request
    # (Public-release note: the paper used a real Stanford MOOCPosts post here; it is
    # replaced by a synthetic post of the same type because that dataset cannot be
    # redistributed.)
    {
        "post": (
            "My daughter is the same way with math worksheets. "
            "She gets SO upset over one wrong answer. We started keeping a "
            "'favorite mistake' jar at dinner which helps a bit, and I'm "
            "hoping this course gives us more ideas. Hang in there!"
        ),
        "bullets": (
            "- Has a child who gets very upset over math mistakes\n"
            "- Already uses a 'favorite mistake' jar; hopes the course adds "
            "more ideas\n"
            "- Offers encouragement"
        ),
    },
    # 6. Direct-address rebuttal — preserve the counter-claim framing even
    #    when the post names the person being corrected
    {
        "post": (
            "That's not quite right. In Python 2, dictionaries do not "
            "preserve insertion order — that was only guaranteed starting in "
            "Python 3.7. You can verify this in the official documentation. "
            "So your example will behave differently depending on which "
            "Python version you're using."
        ),
        "bullets": (
            "- Argues, contra previous post, that Python 2 dicts do not "
            "preserve insertion order\n"
            "- Insertion order only guaranteed since Python 3.7 (per official "
            "docs)\n"
            "- Warns that behavior differs across Python versions"
        ),
    },
    # 7. Short comment / reaction — one compact bullet; do not over-expand
    {
        "post": (
            "This is such an important point! I've noticed the same thing "
            "with my own students — they get so anxious about being wrong "
            "that they stop taking risks at all."
        ),
        "bullets": (
            "- Agrees; observes that student anxiety about being wrong "
            "suppresses risk-taking"
        ),
    },
    # 8. Single clear question — one bullet only; do not restate as two
    {
        "post": (
            "I know a few colleagues who would love access to these "
            "resources. Can I share the link with them, or is enrollment "
            "required to view them?"
        ),
        "bullets": (
            "- Asks whether course resources can be shared with "
            "non-enrolled colleagues"
        ),
    },
]


class BulletSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(BulletSummarizerSignature)

    def forward(self, post):
        return self.predict(post=post, demos=IN_CONTEXT_DEMOS)

class ReconstructorSignature(dspy.Signature):
    """Given the following bullet summary of a student forum post, write what the original post might have looked like in natural prose. Aim to faithfully reflect all content in the bullets."""
    bullets: str = dspy.InputField()
    reconstruction: str = dspy.OutputField()


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
# Models are module-level so the metric (called by GEPA from threads) doesn't
# re-instantiate them per call.  Initialized in main().
SEMANTIC_MODEL = None
STYLE_MODEL = None
ROUGE_SCORER = None
WEIGHTS = DEFAULT_WEIGHTS

round_trip_recon = dspy.Predict(ReconstructorSignature)


def compute_score(post: str, bullets: str):
    """Return (score, components_dict) so GEPA can rank candidates and we can
    also log what drove each score."""
    if not bullets or not bullets.strip():
        return 0.0, {"sim": 0.0, "style_sim": 1.0, "compression": 10.0,
                     "reconstruction_rouge": 0.0, "length_penalty": 1.0}

    post_sem = SEMANTIC_MODEL.encode(post)
    bullets_sem = SEMANTIC_MODEL.encode(bullets)
    post_style = STYLE_MODEL.encode(post)
    bullets_style = STYLE_MODEL.encode(bullets)

    sim = float(cosine_similarity([post_sem], [bullets_sem])[0, 0])
    style_sim = float(cosine_similarity([post_style], [bullets_style])[0, 0])
    compression = len(bullets) / max(len(post), 1)

    reconstructed = round_trip_recon(bullets=bullets).reconstruction
    rouge_recon = ROUGE_SCORER.score(post, reconstructed)["rouge1"].fmeasure

    # Length penalty: zero when compression <= target, grows linearly beyond.
    # Capped to avoid dominating the signal.
    length_penalty = max(0.0, compression - TARGET_COMPRESSION)
    length_penalty = min(length_penalty, 1.0)

    score = (
        WEIGHTS["sim_weight"] * sim + WEIGHTS["recon_weight"] * rouge_recon
        - WEIGHTS["style_weight"] * style_sim
        - WEIGHTS["length_weight"] * length_penalty
    )
    return score, {
        "sim": sim,
        "style_sim": style_sim,
        "compression": compression,
        "reconstruction_rouge": rouge_recon,
        "length_penalty": length_penalty,
    }


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric.  Returns score, or (score, feedback) if trace is set for
    reflection."""
    post = gold.post
    bullets = pred.bullets if hasattr(pred, "bullets") else str(pred)
    score, components = compute_score(post, bullets)

    if trace is None:
        return score

    # Reflection feedback — only generated when GEPA is exploring
    feedback = []
    if components["sim"] < 0.5:
        feedback.append(
            f"Bullets lost too much semantic content (sim={components['sim']:.2f}). "
            "Preserve the core question or task being expressed.")
    if components["style_sim"] > 0.7:
        feedback.append(
            f"Bullets still carry the original writing style (style_sim="
            f"{components['style_sim']:.2f}). Paraphrase more aggressively in "
            "plain, neutral language; strip hedging, filler, and personal voice.")
    if components["compression"] > TARGET_COMPRESSION * 1.5:
        feedback.append(
            f"Bullets are too long (compression={components['compression']:.2f}, "
            f"target <= {TARGET_COMPRESSION}). Cut to the essentials.")
    if components["reconstruction_rouge"] < 0.3:
        feedback.append(
            f"Round-trip reconstruction is poor (ROUGE-1={components['reconstruction_rouge']:.2f}). "
            "The bullets are missing key specifics needed to reconstruct the original question — "
            "include concrete details, key terms, and the main ask.")
    if not feedback:
        feedback.append("Bullets look good; minor tuning only.")

    return score, "\n".join(feedback)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_stackexchange_examples(csv_path: str, n_train: int, n_val: int,
                                 post_col: str = "question_post",
                                 min_post_chars: int = 100):
    """Load forum posts for training/validation.

    ``post_col`` lets callers adapt to different CSV schemas (e.g.
    ``original_post`` for the Stanford MOOC dataset).
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[post_col])
    df = df[df[post_col].str.len() >= min_post_chars]
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train = [dspy.Example(post=row[post_col]).with_inputs("post")
             for _, row in df.iloc[:n_train].iterrows()]
    val = [dspy.Example(post=row[post_col]).with_inputs("post")
           for _, row in df.iloc[n_train:n_train + n_val].iterrows()]
    return train, val


def load_student_validation(csv_path: str, n: int):
    """Held-out student posts for sanity-checking generalization.  NOT used
    during GEPA optimization — only to spot-check the final prompt."""
    df = pd.read_csv(csv_path)
    df = df.sample(n=min(n, len(df)), random_state=42)
    return [dspy.Example(post=row["original_post"]).with_inputs("post")
            for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
def evaluate_on_examples(program, examples, label: str):
    """Run the program on a list of examples and print aggregate metrics."""
    per_example = []
    for ex in examples:
        try:
            pred = program(post=ex.post)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] eval failed: {exc}")
            continue
        bullets = pred.bullets if hasattr(pred, "bullets") else str(pred)
        score, components = compute_score(ex.post, bullets)
        components["score"] = score
        per_example.append(components)

    if not per_example:
        print(f"[{label}] no successful evals")
        return {}
    df = pd.DataFrame(per_example)
    agg = df.mean().to_dict()
    print(f"[{label}] n={len(per_example)}")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}")
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stackexchange",
                        default="stack_exchange_sft_dataset.csv",
                        help="CSV with 'question_post' column.")
    parser.add_argument("--student-data",
                        default="../../pipeline/final_data/stack_exchange_v2.csv",
                        help="Student data CSV for validation spot-check only.")
    parser.add_argument("--output", default="gepa_bullet_summarizer.json",
                        help="Where to save the optimized program.")
    parser.add_argument("--task-lm", default="openai/Qwen/Qwen2.5-7B-Instruct",
                        help="Local model that will run the summarizer. Uses "
                             "dspy.LM format — assumes a vLLM server for Qwen. "
                             "Adjust --task-lm-api-base as needed.")
    parser.add_argument("--task-lm-api-base", default="http://localhost:8002/v1")
    parser.add_argument("--reflection-lm", default="gpt-5",
                        help="Reflection LM for GEPA. GPT-5 mirrors the "
                             "rewriter setup.")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=150)
    parser.add_argument("--n-student-val", type=int, default=100,
                        help="Student posts used ONLY for post-hoc spot check, "
                             "never during GEPA optimization.")
    parser.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sim-weight", type=float, default=DEFAULT_WEIGHTS["sim_weight"])
    parser.add_argument("--recon-weight", type=float, default=DEFAULT_WEIGHTS["recon_weight"])
    parser.add_argument("--style-weight", type=float, default=DEFAULT_WEIGHTS["style_weight"])
    parser.add_argument("--length-weight", type=float, default=DEFAULT_WEIGHTS["length_weight"])
    parser.add_argument("--post-column", default="question_post",
                        help="Column name for the post text in the training CSV "
                             "(e.g. 'original_post' for Stanford MOOC).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Bind module-level state used by the metric
    # ------------------------------------------------------------------
    global SEMANTIC_MODEL, STYLE_MODEL, ROUGE_SCORER, WEIGHTS
    print("Loading scoring models...")
    SEMANTIC_MODEL = SentenceTransformer("all-mpnet-base-v2")
    STYLE_MODEL = SentenceTransformer("StyleDistance/styledistance")
    ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    WEIGHTS = {
        "sim_weight": args.sim_weight,
        "recon_weight": args.recon_weight,
        "style_weight": args.style_weight,
        "length_weight": args.length_weight,
    }
    print(f"Metric weights: {WEIGHTS}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading training data...")
    train, val = load_stackexchange_examples(
        args.stackexchange, args.n_train, args.n_val, post_col=args.post_column)
    print(f"  train: {len(train)}  val: {len(val)}")

    student_val = None
    if os.path.exists(args.student_data):
        print("Loading student validation set (spot-check only)...")
        student_val = load_student_validation(args.student_data, args.n_student_val)
        print(f"  student_val: {len(student_val)}")

    # ------------------------------------------------------------------
    # LMs
    # ------------------------------------------------------------------
    if "localhost" in args.task_lm_api_base or "127.0.0.1" in args.task_lm_api_base:
        task_lm = dspy.LM(args.task_lm, api_base=args.task_lm_api_base, api_key="dummy")
    else:
        task_lm = dspy.LM(args.task_lm)
    reflection_lm = dspy.LM(args.reflection_lm)
    dspy.configure(lm=task_lm)

    summarizer = BulletSummarizer()

    # ------------------------------------------------------------------
    # Baseline eval (before optimization)
    # ------------------------------------------------------------------
    print("\n=== Baseline (un-optimized prompt) ===")
    evaluate_on_examples(summarizer, val, "SE val")
    if student_val is not None:
        evaluate_on_examples(summarizer, student_val, "student val")

    # ------------------------------------------------------------------
    # GEPA optimization
    # ------------------------------------------------------------------
    print("\n=== Running GEPA optimization ===")
    gepa = GEPA(
        metric=metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        num_threads=args.workers,
        track_stats=True,
    )
    optimized = gepa.compile(summarizer, trainset=train, valset=val)
    optimized.save(args.output)
    print(f"Saved optimized program -> {args.output}")

    # ------------------------------------------------------------------
    # Post-optimization eval
    # ------------------------------------------------------------------
    print("\n=== Optimized ===")
    se_agg = evaluate_on_examples(optimized, val, "SE val")
    student_agg = None
    if student_val is not None:
        student_agg = evaluate_on_examples(optimized, student_val, "student val")

    # Persist aggregates alongside the program
    meta_path = args.output.replace(".json", "_eval.json")
    with open(meta_path, "w") as f:
        json.dump({
            "weights": WEIGHTS,
            "target_compression": TARGET_COMPRESSION,
            "stackexchange_val": se_agg,
            "student_val": student_agg,
        }, f, indent=2)
    print(f"Saved eval metrics -> {meta_path}")


if __name__ == "__main__":
    main()