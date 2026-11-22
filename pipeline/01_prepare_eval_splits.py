"""
Stage 0: Create fixed train/val/test splits BEFORE any clustering.

Key decision: we stratify by department and post-length bucket, both of which
are stable properties of posts that exist independent of any clustering choice.
This means the same held-out posts are used across every clustering configuration
we later evaluate — a necessary condition for fair comparison.

Run once. Subsequent clustering and evaluation scripts load split_indices.json.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LENGTH_BUCKETS = [0, 300, 500, float("inf")]  # char-length cuts


def bucket_length(n_chars: int) -> int:
    for i in range(len(LENGTH_BUCKETS) - 1):
        if LENGTH_BUCKETS[i] <= n_chars < LENGTH_BUCKETS[i + 1]:
            return i
    return len(LENGTH_BUCKETS) - 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_data/stack_exchange.csv")
    parser.add_argument("--output", default="final_data/stack_exchange/eval_splits.json")
    parser.add_argument("--eval-size", type=int, default=2000,
                        help="Held-out eval set size (across val+test combined)")
    parser.add_argument("--val-fraction", type=float, default=0.5,
                        help="Fraction of eval set that goes to val (rest to test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} posts from {args.input}")

    # Stratification key: department × length bucket.  Department comes from
    # the CSV column; length bucket is derived from original_post char count.
    assert "department" in df.columns, "expected a 'department' column"
    assert "original_post" in df.columns, "expected an 'original_post' column"

    df["length_bucket"] = df["original_post"].str.len().apply(bucket_length)
    df["strat_key"] = df["department"].astype(str) + "__" + df["length_bucket"].astype(str)

    # Drop strata with <2 members — train_test_split with stratify requires ≥2
    strat_counts = df["strat_key"].value_counts()
    singleton_strata = strat_counts[strat_counts < 4].index.tolist()
    if singleton_strata:
        print(f"[WARN] {len(singleton_strata)} strata have only 1 member; merging into 'RARE'")
        df.loc[df["strat_key"].isin(singleton_strata), "strat_key"] = "RARE"

    # First split: eval set aside
    eval_fraction = args.eval_size / len(df)
    train_df, eval_df = train_test_split(
        df,
        test_size=eval_fraction,
        stratify=df["strat_key"],
        random_state=args.seed,
    )
    # Second split: val vs test within eval set
    strat_counts = eval_df["strat_key"].value_counts()
    singleton_strata = strat_counts[strat_counts < 4].index.tolist()
    if singleton_strata:
        print(f"[WARN] {len(singleton_strata)} strata have only 1 member; merging into 'RARE'")
        eval_df.loc[eval_df["strat_key"].isin(singleton_strata), "strat_key"] = "RARE"

    val_df, test_df = train_test_split(
        eval_df,
        test_size=1 - args.val_fraction,
        stratify=eval_df["strat_key"],
        random_state=args.seed,
    )

    splits = {
        "train": train_df.index.tolist(),
        "val": val_df.index.tolist(),
        "test": test_df.index.tolist(),
        "meta": {
            "eval_size_requested": args.eval_size,
            "actual_train": len(train_df),
            "actual_val": len(eval_df),
            "actual_test": len(test_df),
            "length_buckets": LENGTH_BUCKETS[:-1] + ["inf"],
            "seed": args.seed,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits to {args.output}")
    print(f"  train: {len(train_df)} posts")
    print(f"  val:   {len(val_df)} posts")
    print(f"  test:  {len(test_df)} posts")


if __name__ == "__main__":
    main()
