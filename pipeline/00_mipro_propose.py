"""
Direct-style Top-K RAG with MIPRO prompt candidates.

Simplified from the DP-ICL-RAG/MIPRO script:

  1. No HDBSCAN clusters, UMAP reducers, meta-clusters, Pass A/B sweeps, or DP.
  2. Embed all train posts with StyleDistance/styledistance.
  3. For each eval post, retrieve the global top-K most style-similar train posts
     by cosine similarity, optionally excluding posts from the same student.
  4. Pass those exemplars directly to a single DSPy rewriter call.
  5. Evaluate semantic similarity, style similarity, and ROUGE-1 F1.
  6. Keep the MIPRO instruction-candidate loop from the original script.

The output is one result JSON per proposed instruction candidate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import os
import pickle
import random
from typing import Any

import dotenv
import dspy
import numpy as np
import pandas as pd
import tqdm
from dspy.teleprompt import MIPROv2
from rouge_score import rouge_scorer as rouge_scorer_lib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


dotenv.load_dotenv()

DEFAULT_TOP_K = 20
DEFAULT_NUM_INSTRUCT_CANDIDATES = 10

ROUGE_SCORER = rouge_scorer_lib.RougeScorer(["rouge1"], use_stemmer=True)

# These globals are used by the MIPRO metric, matching the original script's style.
SEMANTIC_MODEL = SentenceTransformer("all-mpnet-base-v2")
STYLE_MODEL = SentenceTransformer("StyleDistance/styledistance")


def metric(gold, pred, trace=None):
    """MIPRO scoring metric: style + semantic + lexical overlap."""
    original_post = gold.original_post

    # DSPy may pass a Prediction object depending on how the metric is invoked.
    if hasattr(pred, "rewritten_post"):
        pred_text = pred.rewritten_post
    else:
        pred_text = str(pred)

    og_style = STYLE_MODEL.encode(original_post)
    gen_style = STYLE_MODEL.encode(pred_text)
    style_sim = float(cosine_similarity([og_style], [gen_style])[0, 0])

    og_sem = SEMANTIC_MODEL.encode(original_post)
    gen_sem = SEMANTIC_MODEL.encode(pred_text)
    sem_sim = float(cosine_similarity([og_sem], [gen_sem])[0, 0])

    rouge1_f = ROUGE_SCORER.score(original_post, pred_text)["rouge1"].fmeasure
    score = (0.33 * style_sim) + (0.33 * sem_sim) + (0.33 * rouge1_f)

    return score


def replace_symbols(rewrite: str) -> str:
    rewrite = rewrite.replace("###", "[concept or knowledge]")
    rewrite = rewrite.replace("***", "[question related to concept]")
    rewrite = rewrite.replace("%%%", "[student's explanation of understanding]")
    rewrite = rewrite.replace("^^^", "[question context or condition]")
    return rewrite



# ---------------------------------------------------------------------------
# DSPy rewriter module
# ---------------------------------------------------------------------------
class RewriteSignature(dspy.Signature):
    """Generate a student discussion forum post that covers the given content bullets and closely follows the style and structure demonstrated by the exemplar posts. Additionally, the post template further details the structure that the final rewritten post should follow."""

    bullet_summary: str = dspy.InputField(
        desc="Bullet points that the rewritten post must cover."
    )
    style_context: str = dspy.InputField(
        desc="Numbered peer posts demonstrating the target writing style, tone, formatting, and register."
    )
    post_template: str = dspy.InputField(
        desc="Description of the general structure of what the rewritten post should look like."
    )
    rewritten_post: str = dspy.OutputField(
        desc=(
            "The rewritten post, content-preserving and register-matched. "
            "Mirror the exemplars' style and structure without copying their content. Follow the structural information in the post template."
        )
    )


class Rewriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RewriteSignature)

    def forward(self, bullet_summary: str, style_context: str, post_template: str) -> str:
        return self.predict(
            bullet_summary=bullet_summary,
            style_context=style_context,
            post_template=post_template
        ).rewritten_post


# ---------------------------------------------------------------------------
# Direct style retrieval
# ---------------------------------------------------------------------------
def retrieve_top_k_by_style(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_texts: list[str],
    corpus_student_ids: list[str],
    top_k: int,
    rng: random.Random,
    exclude_student_id: str | None = None,
) -> tuple[list[str], list[str], list[float]]:
    """Return the global top-k most style-similar corpus posts.

    Ties are broken by deterministic shuffle for reproducibility. If
    exclude_student_id is provided, posts by that student are omitted before
    ranking.
    """
    if len(corpus_embs) == 0:
        return [], [], []

    keep = list(range(len(corpus_embs)))
    if exclude_student_id is not None:
        keep = [i for i in keep if corpus_student_ids[i] != exclude_student_id]

    if not keep:
        return [], [], []

    kept_embs = corpus_embs[keep]
    sims = cosine_similarity([query_emb], kept_embs)[0]

    order = list(range(len(sims)))
    rng.shuffle(order)
    order.sort(key=lambda i: sims[i], reverse=True)
    chosen_local = order[:top_k]
    chosen_global = [keep[i] for i in chosen_local]

    return (
        [corpus_texts[i] for i in chosen_global],
        [corpus_student_ids[i] for i in chosen_global],
        [float(sims[i]) for i in chosen_local],
    )


def format_style_context(exemplars: list[str]) -> str:
    if not exemplars:
        return "(no exemplars available)"
    return "\n\n".join(f"Example {i + 1}:\n{text}" for i, text in enumerate(exemplars))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_one_direct_style(
    example: dict[str, Any],
    rewriter: Rewriter,
    resources: dict[str, Any],
    rng_seed: int,
    top_k: int,
    exclude_same_student: bool = True,
) -> dict[str, Any] | None:
    rng = random.Random(rng_seed)

    target_sid = example.get("student_id", "__unknown__") if exclude_same_student else None
    nbr_texts, nbr_sids, nbr_sims = retrieve_top_k_by_style(
        query_emb=example["style_emb"],
        corpus_embs=resources["train_style_embs"],
        corpus_texts=resources["train_texts"],
        corpus_student_ids=resources["train_student_ids"],
        top_k=top_k,
        rng=rng,
        exclude_student_id=target_sid,
    )

    if not nbr_texts:
        print(f"[WARN] idx={example['index']} no style neighbors found")
        return None

    style_context = format_style_context(nbr_texts)

    try:
        generated = rewriter(
            bullet_summary=example["bullet_summary"],
            style_context=style_context,
            post_template=replace_symbols(example["rr_rewrite"])
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] idx={example['index']} generation failed: {exc}")
        return None

    gen_sem = resources["general_model"].encode(generated)
    gen_style = resources["style_model"].encode(generated)
    og_sem = resources["original_embs_sem"][example["row_pos"]]
    og_style = resources["original_embs_style"][example["row_pos"]]

    recon_sem = float(cosine_similarity([gen_sem], [og_sem])[0, 0])
    recon_style = float(cosine_similarity([gen_style], [og_style])[0, 0])
    rouge1_f = ROUGE_SCORER.score(example["original_post"], generated)["rouge1"].fmeasure

    return {
        "index": example["index"],
        "recon_semantic_sim": recon_sem,
        "recon_style_sim": recon_style,
        "rouge_f1": float(rouge1_f),
        "retrieved_style_sim_mean": float(np.mean(nbr_sims)) if nbr_sims else float("nan"),
        "retrieved_style_sim_max": float(np.max(nbr_sims)) if nbr_sims else float("nan"),
        "retrieved_count": len(nbr_texts),
        "retrieved_student_ids": nbr_sids,
        "retrieved_posts": nbr_texts,
        "generated": generated,
        "original": example["original_post"],
    }


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    return {
        "n": len(results),
        "recon_sem_mean": float(np.mean([r["recon_semantic_sim"] for r in results])),
        "recon_style_mean": float(np.mean([r["recon_style_sim"] for r in results])),
        "rouge_f1_mean": float(np.mean([r["rouge_f1"] for r in results])),
        "retrieved_style_sim_mean": float(np.mean([r["retrieved_style_sim_mean"] for r in results])),
        "retrieved_count_mean": float(np.mean([r["retrieved_count"] for r in results])),
    }


def run_instruction_candidate(
    candidate_idx: int,
    examples: list[dict[str, Any]],
    rewriter: Rewriter,
    resources: dict[str, Any],
    workers: int,
    output_path: str,
    eval_fn,
) -> dict[str, Any]:
    per_example = []
    config_name = f"instruction_candidate::{candidate_idx}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ex = {
            executor.submit(
                eval_fn,
                ex,
                rewriter,
                resources,
                rng_seed=hash((config_name, ex["index"])) & 0xFFFFFFFF,
            ): ex
            for ex in examples
        }
        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_ex),
            total=len(future_to_ex),
            desc=f"  candidate {candidate_idx}",
        ):
            r = fut.result()
            if r is not None:
                per_example.append(r)

    agg = summarize_scores(per_example)
    print(
        f"  candidate={candidate_idx} n={agg.get('n')} "
        f"sem={agg.get('recon_sem_mean', 0):.3f} "
        f"style={agg.get('recon_style_mean', 0):.3f} "
        f"rouge={agg.get('rouge_f1_mean', 0):.3f}"
    )

    bundle = {
        "candidate_idx": candidate_idx,
        "instruction": rewriter.predict.signature.instructions,
        "aggregate": agg,
        "per_example": per_example,
    }
    with open(output_path, "w") as f:
        json.dump(bundle, f, indent=2)
    return bundle


# ---------------------------------------------------------------------------
# Resource loading and example construction
# ---------------------------------------------------------------------------
def build_train_corpus(
    student_data: pd.DataFrame,
    splits: dict[str, list[int]],
    student_id_col: str,
) -> tuple[list[str], list[str]]:
    """Build train texts in the same spirit as the original script.

    This keeps that behavior so the train corpus matches the rewrite artifact.
    """
    train_texts = student_data.loc[splits["train"], "original_post"].astype(str).tolist()

    if student_id_col in student_data.columns:
        text_to_student_id = dict(
            zip(
                student_data["original_post"].astype(str),
                student_data[student_id_col].astype(str),
            )
        )
        train_student_ids = [text_to_student_id.get(t, "__unknown__") for t in train_texts]
    else:
        print(
            f"[WARN] --student-id-col '{student_id_col}' not found; "
            "same-student exclusion will be ineffective."
        )
        train_student_ids = [f"__unknown_{i}__" for i in range(len(train_texts))]

    return train_texts, train_student_ids


def load_resources(train_texts: list[str], train_student_ids: list[str]) -> dict[str, Any]:
    print("Loading embedding models...")
    general_model = SentenceTransformer("all-mpnet-base-v2")
    style_model = SentenceTransformer("StyleDistance/styledistance")

    print("Encoding train posts with style model for direct retrieval...")
    train_style_embs = np.array(style_model.encode(train_texts, batch_size=64))

    return {
        "general_model": general_model,
        "style_model": style_model,
        "train_texts": train_texts,
        "train_student_ids": train_student_ids,
        "train_style_embs": train_style_embs,
    }


def build_examples(
    eval_df: pd.DataFrame,
    resources: dict[str, Any],
    student_id_col: str = "student_id",
) -> list[dict[str, Any]]:
    summary_col = "summary" if "summary" in eval_df.columns else "bullets"

    examples = []
    print("Encoding eval posts with style model for direct retrieval...")
    eval_posts = eval_df["original_post"].astype(str).tolist()
    eval_style_embs = np.array(resources["style_model"].encode(eval_posts, batch_size=64))

    for pos, (row_idx, row) in enumerate(eval_df.iterrows()):
        post_text = str(row["original_post"])
        examples.append(
            {
                "index": int(row_idx),
                "row_pos": pos,
                "original_post": post_text,
                "bullet_summary": row[summary_col],
                "rr_rewrite": row["rewrites"],
                "student_id": str(row[student_id_col]) if student_id_col in row.index else "__unknown__",
                "style_emb": eval_style_embs[pos],
            }
        )
    return examples


def load_or_propose_instructions(
    rewriter: Rewriter,
    candidates_file: str,
    num_instruct_candidates: int,
    task_lm,
    prompt_lm,
) -> list[str]:
    mipro = MIPROv2(
        metric=metric,
        task_model=task_lm,
        prompt_model=prompt_lm,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        verbose=True,
    )

    if not os.path.exists(candidates_file):
        candidates = mipro._propose_instructions(
            rewriter,
            trainset=None,
            demo_candidates=None,
            view_data_batch_size=0,
            program_aware_proposer=True,
            data_aware_proposer=False,
            fewshot_aware_proposer=False,
            tip_aware_proposer=True,
            num_instruct_candidates=num_instruct_candidates,
        )
        candidates = {str(k): v for k, v in candidates.items()}
        with open(candidates_file, "w") as f:
            json.dump(candidates, f, indent=2)
    else:
        with open(candidates_file) as f:
            candidates = json.load(f)

    # Original code reads candidates["0"]. Keep that convention, but fall back
    # to flattening if DSPy changes the returned shape.
    if "0" in candidates and isinstance(candidates["0"], list):
        return candidates["0"]

    flattened = []
    for value in candidates.values():
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-data", default="student_post_bullets.csv")
    parser.add_argument("--splits", default="final_data/stack_exchange/eval_splits_seed0.json")
    parser.add_argument("--output-dir", default="direct_style_topk_mipro_output")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--prompt-model", default=None)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--max-posts", type=int, default=None)
    parser.add_argument("--sample-posts", type=int, default=100)
    parser.add_argument("--sample-random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--student-id-col", default="student_id")
    parser.add_argument(
        "--include-same-student",
        action="store_true",
        help="Allow retrieved exemplars from the same student. Default excludes them when student IDs are available.",
    )
    parser.add_argument(
        "--candidates-file",
        default="direct_style_topk_mipro_prompt_candidates.json",
    )
    parser.add_argument(
        "--num-instruct-candidates",
        type=int,
        default=DEFAULT_NUM_INSTRUCT_CANDIDATES,
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible API base URL (e.g. http://localhost:8001/v1 for vLLM)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data and splits
    # ------------------------------------------------------------------
    student_data = pd.read_csv(args.student_data).reset_index(drop=True)
    with open(args.splits) as f:
        splits = json.load(f)

    eval_indices = splits[args.eval_split]
    if args.max_posts:
        eval_indices = eval_indices[: args.max_posts]
    eval_df_full = student_data.loc[eval_indices]
    print(f"Loaded {len(eval_df_full)} posts from '{args.eval_split}' split")

    if args.sample_posts is not None and len(eval_df_full) > args.sample_posts:
        eval_df_full = eval_df_full.sample(
            n=args.sample_posts,
            random_state=args.sample_random_state,
        )
        print(f"Sampled {len(eval_df_full)} eval posts")

    # ------------------------------------------------------------------
    # Build train retrieval corpus and resources
    # ------------------------------------------------------------------
    print("Building train corpus...")
    train_texts, train_student_ids = build_train_corpus(
        student_data=student_data,
        splits=splits,
        student_id_col=args.student_id_col,
    )
    print(f"Loaded {len(train_texts)} train posts for direct style retrieval")

    resources = load_resources(train_texts, train_student_ids)

    print("Precomputing eval-set embeddings for metrics...")
    eval_posts = eval_df_full["original_post"].astype(str).tolist()
    resources["original_embs_sem"] = np.array(
        resources["general_model"].encode(eval_posts, batch_size=64)
    )
    resources["original_embs_style"] = np.array(
        resources["style_model"].encode(eval_posts, batch_size=64)
    )

    examples_full = build_examples(
        eval_df_full,
        resources,
        student_id_col=args.student_id_col,
    )

    # ------------------------------------------------------------------
    # Rewriter + MIPRO instruction candidates
    # ------------------------------------------------------------------
    lm_kwargs = {"api_base": args.api_base, "api_key": "none"} if args.api_base else {}
    task_lm = dspy.LM(args.model, cache=True, **lm_kwargs)
    prompt_lm = dspy.LM(args.prompt_model or args.model, cache=True, **lm_kwargs)
    dspy.configure(lm=task_lm)

    rewriter = Rewriter()
    instructions = load_or_propose_instructions(
        rewriter=rewriter,
        candidates_file=args.candidates_file,
        num_instruct_candidates=args.num_instruct_candidates,
        task_lm=task_lm,
        prompt_lm=prompt_lm,
    )

    eval_fn = functools.partial(
        eval_one_direct_style,
        top_k=args.top_k,
        exclude_same_student=not args.include_same_student,
    )

    # ------------------------------------------------------------------
    # Evaluate every proposed instruction candidate
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"DIRECT STYLE TOP-K MIPRO EVAL  (top_k={args.top_k})")
    print("=" * 70)

    all_results = {}
    for i, instruction in enumerate(tqdm.tqdm(instructions, desc="Candidates")):
        rewriter.predict.signature.instructions = instruction
        output_path = os.path.join(args.output_dir, f"{i}_direct_style_topk_results.json")
        bundle = run_instruction_candidate(
            candidate_idx=i,
            examples=examples_full,
            rewriter=rewriter,
            resources=resources,
            workers=args.workers,
            output_path=output_path,
            eval_fn=eval_fn,
        )
        all_results[str(i)] = {
            "instruction": instruction,
            "aggregate": bundle["aggregate"],
            "output_path": output_path,
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    ranking = sorted(
        all_results.items(),
        key=lambda kv: kv[1]["aggregate"].get("recon_sem_mean", 0.0),
        reverse=True,
    )
    with open(summary_path, "w") as f:
        json.dump({"ranking": ranking, "results": all_results}, f, indent=2)

    print("\n=== Top instruction candidates by recon_sem ===")
    for name, bundle in ranking[:10]:
        agg = bundle["aggregate"]
        print(
            f"  candidate {name:>3s}  "
            f"sem={agg.get('recon_sem_mean', 0):.3f}  "
            f"style={agg.get('recon_style_mean', 0):.3f}  "
            f"rouge={agg.get('rouge_f1_mean', 0):.3f}  "
            f"n={agg.get('n')}"
        )
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
