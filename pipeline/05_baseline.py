"""
Baseline evaluation: generate student forum posts and measure semantic and style
similarity against the original posts.

Two baselines are run:
  baseline_br  -- conditioned on bullet_summary + post_template (structure hint)
  baseline_b   -- conditioned on bullet_summary only (no structural guidance)

Results are written to separate sub-directories under --output-dir.
"""

import argparse
import concurrent.futures
import json
import os

import dotenv
import dspy
import numpy as np
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()


# ---------------------------------------------------------------------------
# DSPy modules
# ---------------------------------------------------------------------------
class SimulatorBaselineBRSignature(dspy.Signature):
    """Generate a student discussion forum post that includes all the key points provided in the `bullet_summary`. Ensure that the post closely mirrors the writing style, tone, and format of a typical student discussion forum post. Additionally, follow the structural guidelines detailed in the `post_template` to ensure the rewritten post is well-organized and meets the specified criteria.\n\n1. **Content Coverage**: Make sure that every point listed in the `bullet_summary` is addressed in the rewritten post.\n2. **Style and Tone**: Adopt the same writing style, tone, and format as a typical student discussion forum post. This includes sentence structure, vocabulary, and any specific formatting elements.\n3. **Structure**: Follow the structure outlined in the `post_template`. This may include sections, paragraphs, headings, and other organizational elements that are important for the post's clarity and readability.\n\nBy adhering to these guidelines, the rewritten post should effectively communicate the required content while maintaining a consistent and appropriate style and structure."""

    bullet_summary: str = dspy.InputField(
        desc="Bullet points describing the content the post should cover.")
    post_template: str = dspy.InputField(
        desc="Description of the general structure of what the rewritten post should look like."
    )
    output_post: str = dspy.OutputField(
        desc="The generated student forum post.")


class SimulatorBaselineBR(dspy.Module):
    """Baseline with Bullets + Rewrite template (BR)."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SimulatorBaselineBRSignature)

    def forward(self, bullet_summary, post_template=None):
        return self.predict(bullet_summary=bullet_summary, post_template=post_template).output_post


class SimulatorBaselineBSignature(dspy.Signature):
    """Generate a student discussion forum post that includes all the key points provided in the `bullet_summary`. Ensure that the post closely mirrors the writing style, tone, and format of a typical student discussion forum post. Additionally, follow the structural guidelines detailed in the `post_template` to ensure the rewritten post is well-organized and meets the specified criteria.\n\n1. **Content Coverage**: Make sure that every point listed in the `bullet_summary` is addressed in the rewritten post.\n2. **Style and Tone**: Adopt the same writing style, tone, and format as a typical student discussion forum post. This includes sentence structure, vocabulary, and any specific formatting elements.\n\nBy adhering to these guidelines, the rewritten post should effectively communicate the required content while maintaining a consistent and appropriate style and structure."""

    bullet_summary: str = dspy.InputField(
        desc="Bullet points describing the content the post should cover.")
    output_post: str = dspy.OutputField(
        desc="The generated student forum post.")


class SimulatorBaselineB(dspy.Module):
    """Baseline with Bullets only (B)."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SimulatorBaselineBSignature)

    def forward(self, bullet_summary, post_template=None):
        return self.predict(bullet_summary=bullet_summary).output_post


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_one(example, simulator, resources):
    try:
        generated = simulator(
            bullet_summary=example["bullet_summary"],
            post_template=example.get("post_template"),
        )
    except Exception as exc:
        print(f"[WARN] idx={example['index']} generation failed: {exc}")
        return None

    gen_sem   = resources["general_model"].encode(generated)
    gen_style = resources["style_model"].encode(generated)
    og_sem    = resources["original_embs_sem"][example["row_pos"]]
    og_style  = resources["original_embs_style"][example["row_pos"]]

    return {
        "index":              example["index"],
        "recon_semantic_sim": float(cosine_similarity([gen_sem],   [og_sem])[0, 0]),
        "recon_style_sim":    float(cosine_similarity([gen_style], [og_style])[0, 0]),
        "generated":          generated,
    }


def summarize_scores(results):
    if not results:
        return {"n": 0}
    return {
        "n":                len(results),
        "recon_sem_mean":   float(np.mean([r["recon_semantic_sim"] for r in results])),
        "recon_style_mean": float(np.mean([r["recon_style_sim"]   for r in results])),
    }


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def load_resources():
    print("Loading embedding models...")
    return {
        "general_model": SentenceTransformer("all-mpnet-base-v2"),
        "style_model":   SentenceTransformer("StyleDistance/styledistance"),
    }


def build_examples(eval_df):
    summary_col = "summary" if "summary" in eval_df.columns else "bullets"
    return [
        {
            "index":          int(row_idx),
            "row_pos":        pos,
            "bullet_summary": row[summary_col],
            "post_template":  row["rewrites"]
        }
        for pos, (row_idx, row) in enumerate(eval_df.iterrows())
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_baseline(name, simulator, examples, resources, output_dir, workers):
    """Run one baseline variant and write results.json to output_dir/name/."""
    out = os.path.join(output_dir, name)
    os.makedirs(out, exist_ok=True)

    per_example = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(eval_one, ex, simulator, resources): ex
            for ex in examples
        }
        for fut in tqdm.tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc=name):
            r = fut.result()
            if r is not None:
                per_example.append(r)

    agg = summarize_scores(per_example)
    print(f"[{name}] n={agg['n']}  recon_sem={agg.get('recon_sem_mean', 0):.3f}  "
          f"recon_style={agg.get('recon_style_mean', 0):.3f}")

    output_path = os.path.join(out, "results.json")
    with open(output_path, "w") as f:
        json.dump({"aggregate": agg, "per_example": per_example}, f, indent=2)
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-data", default="student_post_bullets.csv")
    parser.add_argument("--splits",       default="final_data/stack_exchange/eval_splits_seed0.json")
    parser.add_argument("--output-dir",   default="baseline_output")
    parser.add_argument("--model",        default="openai/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--api-base",     default="http://localhost:8001/v1",
                        help="OpenAI-compatible API base URL (e.g. http://localhost:8001/v1 for vLLM)")
    parser.add_argument("--eval-split",   default="val", choices=["val", "test"])
    parser.add_argument("--max-posts",    type=int, default=None)
    parser.add_argument("--workers",      type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and splits
    student_data = pd.read_csv(args.student_data).reset_index(drop=True)
    with open(args.splits) as f:
        splits = json.load(f)
    eval_indices = splits[args.eval_split]
    if args.max_posts:
        eval_indices = eval_indices[:args.max_posts]
    eval_df = student_data.loc[eval_indices]
    print(f"Loaded {len(eval_df)} posts from '{args.eval_split}' split")

    # Load models and precompute original-post embeddings
    resources = load_resources()
    eval_posts = list(eval_df["original_post"])
    resources["original_embs_sem"]   = np.array(
        resources["general_model"].encode(eval_posts, batch_size=64))
    resources["original_embs_style"] = np.array(
        resources["style_model"].encode(eval_posts, batch_size=64))

    examples = build_examples(eval_df)

    lm_kwargs = {"api_base": args.api_base, "api_key": "none"} if args.api_base else {}
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))

    # baseline_br: bullets + rewrite template
    run_baseline("baseline_br", SimulatorBaselineBR(), examples, resources,
                 args.output_dir, args.workers)

    # baseline_b: bullets only
    run_baseline("baseline_b", SimulatorBaselineB(), examples, resources,
                 args.output_dir, args.workers)


if __name__ == "__main__":
    main()
