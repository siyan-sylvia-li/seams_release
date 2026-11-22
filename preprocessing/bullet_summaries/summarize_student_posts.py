"""
Run the optimized GEPA bullet summarizer over all student posts.

Loads gepa_bullet_summarizer.json and the student data CSV, runs inference
in batches, and writes a CSV with the original post and generated bullets.
"""

import argparse
import os

import dotenv
import dspy
import pandas as pd
from tqdm import tqdm

from gepa_bullet_summarizer import BulletSummarizer

dotenv.load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student-data",
        default="../../pipeline/final_data/stack_exchange_v2.csv",
        help="Student data CSV with 'original_post' column.",
    )
    parser.add_argument(
        "--program",
        default="gepa_bullet_summarizer.json",
        help="Path to the saved optimized DSPy program.",
    )
    parser.add_argument(
        "--output",
        default="student_post_bullets.csv",
        help="Where to write the output CSV.",
    )
    parser.add_argument(
        "--task-lm",
        default="openai/Qwen/Qwen2.5-7B-Instruct",
        help="Local model to run inference with.",
    )
    parser.add_argument("--task-lm-api-base", default="http://localhost:8002/v1")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Number of posts per batch."
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel threads for DSPy inference."
    )
    parser.add_argument(
        "--min-post-chars",
        type=int,
        default=0,
        help="Skip posts shorter than this many characters.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # LM + program
    # ------------------------------------------------------------------
    if "localhost" in args.task_lm_api_base or "127.0.0.1" in args.task_lm_api_base:
        lm = dspy.LM(args.task_lm, api_base=args.task_lm_api_base, api_key="dummy")
    else:
        lm = dspy.LM(args.task_lm)
    dspy.configure(lm=lm, num_threads=args.workers)

    summarizer = BulletSummarizer()
    summarizer.load(args.program)
    print(f"Loaded optimized program from {args.program}")

    # ------------------------------------------------------------------
    # Student data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.student_data)
    if args.min_post_chars > 0:
        before = len(df)
        df = df[df["original_post"].str.len() >= args.min_post_chars]
        print(f"Filtered {before - len(df)} short posts; {len(df)} remaining.")
    print(f"Summarizing {len(df)} posts...")

    # ------------------------------------------------------------------
    # Resume support: skip already-processed posts if output exists
    # ------------------------------------------------------------------
    done_ids: set = set()
    if os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        done_ids = set(existing["discussion_post_id"].tolist())
        print(f"Resuming — {len(done_ids)} posts already done.")
        df = df[~df["discussion_post_id"].isin(done_ids)]

    if df.empty:
        print("Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    results = []
    rows = df.to_dict("records")
    for i in tqdm(range(0, len(rows), args.batch_size), unit="batch"):
        batch = rows[i : i + args.batch_size]
        for row in batch:
            post = str(row["original_post"])
            try:
                pred = summarizer(post=post)
                bullets = pred.bullets if hasattr(pred, "bullets") else str(pred)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] failed on post {row.get('discussion_post_id')}: {exc}")
                bullets = ""
            results.append(
                {
                    "discussion_post_id": row.get("discussion_post_id"),
                    "student_id": row.get("student_id"),
                    "department": row.get("department"),
                    "original_post": post,
                    "summary": bullets,
                    "classification_1": row.get("classification_1"),
                    "classification_2": row.get("classification_2"),
                }
            )

        # Append-write after each batch so progress is never lost
        chunk = pd.DataFrame(results)
        write_header = not os.path.exists(args.output) or (
            i == 0 and not done_ids
        )
        chunk.to_csv(args.output, mode="a", header=write_header, index=False)
        results = []

    print(f"Done. Output written to {args.output}")


if __name__ == "__main__":
    main()
