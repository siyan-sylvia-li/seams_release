import dspy

import dotenv
dotenv.load_dotenv()

from sklearn.linear_model import LogisticRegression
import numpy

import pandas
import re
import random
import os
import json
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

# GEPA optimization data: 50 posts with human-written structural templates.
# In the paper these were drawn from the proprietary dataset and cannot be released;
# point SEAMS_MANUAL_REWRITES at your own CSV (same columns) to re-run the optimization.
# Only needed when running this file as a script; importing `Rewriter` does not read it.
MANUAL_REWRITES_CSV = os.environ.get("SEAMS_MANUAL_REWRITES", "manual_rewrites.csv")
manual_rewrites = None
all_depts = {}
embedder = None

MARKER_RE = re.compile(r'[\*\#\^\%]+')

# Per-example accuracy log used by metric_all()
METRIC_ACC_LOG = []

# Per-example rewrite/class dicts used by metric_all_lr()
METRIC_REWRITE_DICT = {}
METRIC_CLASS_DICT = {}


def get_masked_chars(original, rewritten):
    """
    Returns a boolean list of length len(original) where True means
    the character at that position was masked in the rewrite.
    """
    preserved_parts = MARKER_RE.split(rewritten)
    masked = [True] * len(original)
    pos = 0
    for part in preserved_parts:
        if not part:
            continue
        idx = original.find(part, pos)
        if idx == -1:
            continue
        for j in range(idx, idx + len(part)):
            masked[j] = False
        pos = idx + len(part)
    return masked


def rewrite_accuracy(original, model_rewrite, human_rewrite):
    """
    Character-level accuracy: fraction of characters where the model
    and human agree on masked vs. not-masked.
    """
    if not original:
        return 1.0
    model_masked = get_masked_chars(original, model_rewrite)
    human_masked = get_masked_chars(original, human_rewrite)
    agree = sum(m == h for m, h in zip(model_masked, human_masked))
    return agree / len(original)


def compute_acc(embeddings, labels):
    Xs = numpy.array(embeddings)
    Ys = [all_depts[y] for y in labels]
    lr = LogisticRegression(max_iter=1000, tol=0.0).fit(Xs, Ys)
    return lr.score(Xs, Ys)


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    original = gold.post
    human_rw = gold.human_rewrite
    model_rw = pred.rewrite

    if original not in METRIC_CLASS_DICT and hasattr(gold, "course_dept"):
        METRIC_CLASS_DICT[original] = gold.course_dept
    METRIC_REWRITE_DICT[original] = model_rw

    acc = rewrite_accuracy(original, model_rw, human_rw)
    METRIC_ACC_LOG.append(acc)

    if acc >= 0.90:
        feedback = "Excellent: your masking closely matches the human reference at the character level."
    elif acc >= 0.80:
        feedback = "Good: most masking decisions agree with the human reference, but some spans are over- or under-masked."
    elif acc >= 0.65:
        feedback = "Partial: noticeable disagreement with the human reference. Check whether you are masking the right spans and using the correct markers."
    else:
        feedback = "Poor: your masking strongly disagrees with the human reference. Re-read the marker rules and try to match which characters should be hidden."

    return ScoreWithFeedback(score=acc, feedback=feedback)


def metric_all():
    """Global metric: mean character-level accuracy across all logged examples."""
    if not METRIC_ACC_LOG:
        return 0.0
    return sum(METRIC_ACC_LOG) / len(METRIC_ACC_LOG)


def metric_all_lr():
    """Global metric: logistic-regression accuracy on embeddings of model rewrites."""
    if not METRIC_CLASS_DICT or not METRIC_REWRITE_DICT:
        return None
    all_rewrites = [METRIC_REWRITE_DICT[og] for og in METRIC_CLASS_DICT]
    all_classes = [METRIC_CLASS_DICT[og] for og in METRIC_CLASS_DICT]
    embeds = embedder(all_rewrites, batch_size=50)
    return compute_acc(embeds, all_classes)


class RewriterSignature(dspy.Signature):
    """You are a text rewriter. Your task is to mask domain-specific content from student discussion forum posts while preserving the student's writing style and all non-domain elements.

Use the following markers to replace domain-specific content:
- ### : the specific knowledge, concept, or term the student is asking about (named concepts, theories, formulas, terminology)
- *** : the question body directly related to the target concept (what the student is explicitly asking)
- %%% : explanation of the student's understanding of the concept
- ^^^ : context or condition of the question (a specific scenario, given information, or described situation)

Rules:
1. Each masked span must start and end exactly at the boundaries of the domain-specific content.
2. If two consecutive sentences belong to the same marker type, separate them with a period (e.g., %%%.%%% not %%%%%%).
3. Preserve all sentence structure, connectors, hedging language, punctuation, filler words, opinion markers, and other style-only elements.
"""
    post: str = dspy.InputField()
    rewrite: str = dspy.OutputField()

# In-context demos for the template rewriter.
# The paper used six posts (with human-written templates) from the proprietary dataset;
# those cannot be released. By default we load SYNTHETIC stand-ins written for this
# release (in_context_demos.synthetic.json), so templates produced with them will differ
# somewhat from the released *_v2.csv templates. Set SEAMS_TEMPLATE_DEMOS to use your own.
_DEMO_PATH = os.environ.get(
    "SEAMS_TEMPLATE_DEMOS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "in_context_demos.synthetic.json"))
IN_CONTEXT_DEMOS = json.load(open(_DEMO_PATH))

class Rewriter(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self.rewriter = dspy.ChainOfThought(RewriterSignature)

    def forward(self, post):
        return self.rewriter(post=post, demos=IN_CONTEXT_DEMOS)


if __name__ == "__main__":
    manual_rewrites = pandas.read_csv(MANUAL_REWRITES_CSV)
    if "dept" in manual_rewrites.columns:
        all_depts.update({x: i for i, x in enumerate(manual_rewrites["dept"].dropna().unique())})
    embedder = dspy.Embedder("openai/text-embedding-3-small")
    random.seed(42)

    all_data = []
    for i, row in manual_rewrites.iterrows():
        post = row["discussion_post_content_clean"]
        human_rw = row["human_rewrite"]
        if pandas.isna(post) or pandas.isna(human_rw):
            continue
        dept = row.get("dept") if "dept" in row.index else None
        example_fields = {"post": post, "human_rewrite": human_rw}
        if dept and not pandas.isna(dept):
            example_fields["course_dept"] = dept
        all_data.append(dspy.Example(example_fields).with_inputs("post"))

    random.shuffle(all_data)
    train = all_data[:40]
    val = all_data  # evaluate all available examples against human rewrites

    task_lm = dspy.LM("openai/gpt-5-mini")
    prompt_lm = dspy.LM("openai/gpt-5")

    dspy.configure(lm=task_lm)

    rewrite_program = Rewriter()

    if not os.path.exists("gepa_rewriter_char_acc.json"):

        gepa = GEPA(
            metric=metric,
            auto="light",
            reflection_lm=prompt_lm,
            num_threads=8,
            track_stats=True
        )

        optimized_program = gepa.compile(
            rewrite_program,
            trainset=train,
            valset=val
        )

        # all_candidates = optimized_program.detailed_results.candidates
        # json.dump([c. for c in all_candidates], open("gepa_char_acc_candidates.json", "w+"), indent=2)

        optimized_program.save("gepa_rewriter_char_acc.json")
    else:
        optimized_program = Rewriter()
        optimized_program.load("gepa_rewriter_char_acc.json")

    METRIC_ACC_LOG.clear()
    METRIC_REWRITE_DICT.clear()
    METRIC_CLASS_DICT.clear()

    print("EVALUATING ON FULL")
    evaluate = dspy.Evaluate(devset=all_data, metric=metric, provide_traceback=True, num_threads=8)
    per_example_score = evaluate(optimized_program).score
    global_score = metric_all()
    lr_score = metric_all_lr()
    print(f"Final char_acc (per-example mean from dspy.Evaluate): {per_example_score:.4f}")
    print(f"Final char_acc (global mean from metric_all):         {global_score:.4f}")
    if lr_score is not None:
        print(f"Final lr_acc   (logistic regression on rewrites):     {lr_score:.4f}")

    scores = {"char_acc_per_example": per_example_score, "char_acc_global": global_score}
    if lr_score is not None:
        scores["lr_acc"] = lr_score
    json.dump(scores, open("gepa_rewriter_char_acc_scores.json", "w+"))
