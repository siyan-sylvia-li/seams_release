"""
05_dp_prompt_eval.py

DP-Prompt baseline (document-level, no logit clipping).
Ref: Utpala et al., EMNLP 2023 Findings — "Locally Differentially Private
Document Generation Using Zero Shot Prompting."
https://github.com/SaitejaUtpala/dp_prompt

Mechanism: temperature-controlled sampling over the FULL vocabulary
  (top_k=0, top_p=1.0). This matches the paper's main experiments, which
  do NOT use logit clipping (Section 4.4). The DP guarantee is loose but
  finite; privacy is evaluated empirically via authorship attacks.

Temperatures swept (matching paper's open-source model sweep):
  {1.0, 1.25, 1.5, 1.75, 2.0}

Each temperature produces one config in the output JSON named
"dp_prompt_t{temp}", matching the multi-config format of 05_dp_icl_config_eval.py.

Usage
-----
python 05_dp_prompt_eval.py \\
    --student-data  final_data/stack_exchange_v2.csv \\
    --splits        final_data/stack_exchange/eval_splits_seed123.json \\
    --output-dir    final_data/stack_exchange/dp_prompt_outputs/seed_123 \\
    --temperatures  1.0 1.25 1.5 1.75 2.0 \\
    --model         Qwen/Qwen2.5-72B-Instruct \\
    --api-base      http://localhost:8001/v1 \\
    --workers       32
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time

import numpy as np
import openai
import pandas as pd
import tqdm
from rouge_score import rouge_scorer as rouge_scorer_lib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROUGE_SCORER = rouge_scorer_lib.RougeScorer(["rouge1"], use_stemmer=True)


# ---------------------------------------------------------------------------
# Prompt — adapted from paper ("Review" → "Post")
# ---------------------------------------------------------------------------

def build_prompt(post: str) -> str:
    return f"Post: {post}\nParaphrase of the post:"


# ---------------------------------------------------------------------------
# Generation only (no embedding — runs in parallel threads)
# ---------------------------------------------------------------------------

def generate_one(
    example: pd.Series,
    temperature: float,
    client: openai.OpenAI,
    model: str,
) -> dict:
    post = str(example["original_post"])
    prompt = build_prompt(post)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1.0,   # full vocabulary — required for DP guarantee
                max_tokens=512,
            )
            return {"index": int(example.name), "original": post,
                    "generated": resp.choices[0].message.content.strip()}
        except Exception as e:
            if attempt == 2:
                return {"index": int(example.name), "original": post,
                        "generated": None, "error": str(e)}
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Run one temperature config
# ---------------------------------------------------------------------------

def run_temperature(
    eval_df: pd.DataFrame,
    temperature: float,
    client: openai.OpenAI,
    model: str,
    resources: dict,
    workers: int,
) -> dict:
    examples = [row for _, row in eval_df.iterrows()]

    # Step 1: generate all in parallel
    raw: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, ex, temperature, client, model): ex
                   for ex in examples}
        for fut in tqdm.tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc=f"  t={temperature} generate"):
            r = fut.result()
            if r and r.get("generated"):
                raw.append(r)

    if not raw:
        return {"aggregate": {"n": 0}, "per_example": []}

    originals  = [r["original"]  for r in raw]
    generateds = [r["generated"] for r in raw]

    # Step 2: batch encode
    sem_model: SentenceTransformer   = resources["sem_model"]
    style_model: SentenceTransformer = resources["style_model"]
    print(f"  Encoding {len(raw)} pairs...")
    og_sem    = sem_model.encode(originals,  batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    gen_sem   = sem_model.encode(generateds, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    og_style  = style_model.encode(originals,  batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    gen_style = style_model.encode(generateds, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

    per_example = []
    for i, r in enumerate(raw):
        recon_sem   = float(cosine_similarity([gen_sem[i]],   [og_sem[i]])[0, 0])
        recon_style = float(cosine_similarity([gen_style[i]], [og_style[i]])[0, 0])
        rouge1_f    = ROUGE_SCORER.score(r["original"], r["generated"])["rouge1"].fmeasure
        per_example.append({
            "index":              r["index"],
            "original":           r["original"],
            "generated":          r["generated"],
            "recon_semantic_sim": round(recon_sem,   4),
            "recon_style_sim":    round(recon_style, 4),
            "rouge_f1":           round(rouge1_f,    4),
        })

    aggregate = {
        "n":                len(per_example),
        "recon_sem_mean":   round(float(np.mean([r["recon_semantic_sim"] for r in per_example])), 4),
        "recon_style_mean": round(float(np.mean([r["recon_style_sim"]    for r in per_example])), 4),
        "rouge_f1_mean":    round(float(np.mean([r["rouge_f1"]           for r in per_example])), 4),
    }
    print(f"  t={temperature}  n={aggregate['n']}"
          f"  sem={aggregate['recon_sem_mean']:.3f}"
          f"  style={aggregate['recon_style_mean']:.3f}"
          f"  rouge={aggregate['rouge_f1_mean']:.3f}")
    return {"aggregate": aggregate, "per_example": per_example}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-data", required=True)
    parser.add_argument("--splits",       required=True,
                        help="eval_splits_seed<N>.json")
    parser.add_argument("--eval-split",   default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--post-col",     default="original_post")
    parser.add_argument("--max-posts",    type=int, default=None)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[1.0, 1.25, 1.5, 1.75, 2.0],
                        help="Sampling temperatures (default: paper's open-source sweep)")
    parser.add_argument("--model",        default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--api-base",     default="http://localhost:8001/v1")
    parser.add_argument("--workers",      type=int, default=32)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--sem-model",    default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--style-model",  default="StyleDistance/styledistance")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.student_data).reset_index(drop=True)
    with open(args.splits) as f:
        splits = json.load(f)
    eval_indices = splits[args.eval_split]
    if args.max_posts:
        eval_indices = eval_indices[: args.max_posts]
    eval_df = df.loc[eval_indices]
    print(f"Loaded {len(eval_df)} posts from '{args.eval_split}' split")

    # OpenAI-compatible client (works with vLLM)
    client = openai.OpenAI(base_url=args.api_base, api_key="none")

    # Embedding models
    print(f"Loading semantic model: {args.sem_model}")
    sem_model = SentenceTransformer(args.sem_model)
    print(f"Loading style model:    {args.style_model}")
    style_model = SentenceTransformer(args.style_model)
    resources = {"sem_model": sem_model, "style_model": style_model}

    # Run each temperature
    all_results: dict = {}
    for temp in args.temperatures:
        config_name = f"dp_prompt_t{temp}"
        print(f"\n{'='*60}\nConfig: {config_name}")
        all_results[config_name] = run_temperature(
            eval_df, temp, client, args.model, resources, args.workers
        )

    out_path = os.path.join(args.output_dir, "pass_b_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
