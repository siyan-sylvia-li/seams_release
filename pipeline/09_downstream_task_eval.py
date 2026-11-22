"""
09_downstream_task_eval.py

Downstream task performance check: does training on rewrites generalise to originals?

Two classifiers are trained on the VAL split:
  A — trained on val set ORIGINAL posts        (upper bound)
  B — trained on val set REWRITE posts         (anonymized training signal)

Both are tested on the TEST split ORIGINAL posts.

If B ≈ A: the rewrites preserve the label signal for downstream use.

Regularisation is tuned on the training data only, identically for A and B and for both
embeddings: LogisticRegressionCV (C over logspace(-3, 3, 13), stratified 5-fold, macro-F1)
for classification; RidgeCV (alpha over logspace(-2, 4, 13), efficient leave-one-out) for
regression. The selected C / alpha is stored with each result.

Datasets / labels:
  og_students:    classification_1 = Bloom's taxonomy (6-class, LogReg)
  stack_exchange: department = subject (10 sites); pass --labels department
                  (see run_downstream_subject_se.sh)
  stanford_mooc:  classification_1 = urgency, classification_2 = confusion, 1-7 (Ridge)

Usage
-----
python 09_downstream_task_eval.py \\
    --student-data  final_data/stack_exchange_v2.csv \\
    --rewrites-json final_data/stack_exchange/dp_icl_outputs/meta/seed_123/pass_b_results.json \\
    --splits        final_data/stack_exchange/eval_splits_seed123.json \\
    --output        final_data/stack_exchange/downstream_eval_dpiclmeta_seed123.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_task(series: pd.Series) -> str:
    vals = series.dropna().unique()
    if set(vals) <= {True, False, "True", "False", 1, 0, 1.0, 0.0}:
        return "binary"
    try:
        [float(v) for v in vals]
        return "regression"
    except (ValueError, TypeError):
        return "multiclass"


def embed(texts: list[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True)


def make_clf():
    return LogisticRegressionCV(Cs=np.logspace(-3, 3, 13), cv=5, scoring="f1_macro",
                                max_iter=2000, random_state=0)


def make_reg():
    return RidgeCV(alphas=np.logspace(-2, 4, 13))


def eval_clf(clf, X: np.ndarray, y: np.ndarray) -> dict:
    preds = clf.predict(X)
    return {
        "C": float(clf.C_[0]),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "macro_f1": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
        "n": len(y),
    }


def eval_reg(reg, X: np.ndarray, y: np.ndarray) -> dict:
    preds = reg.predict(X)
    return {
        "alpha": float(reg.alpha_),
        "r2":  round(float(r2_score(y, preds)), 4),
        "mae": round(float(mean_absolute_error(y, preds)), 4),
        "n":   len(y),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-data",   required=True)
    parser.add_argument("--rewrites-json",  required=True,
                        help="pass_b_results.json from any 05_*.py eval.")
    parser.add_argument("--splits",         required=True,
                        help="eval_splits_seed<N>.json")
    parser.add_argument("--post-col",       default="original_post")
    parser.add_argument("--config",         default=None)
    parser.add_argument("--model",          default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--style-model",    default="StyleDistance/styledistance")
    parser.add_argument("--output",         default="downstream_eval.json")
    parser.add_argument("--labels",         default="classification_1,classification_2",
                        help="comma-separated label columns to evaluate")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ------------------------------------------------------------------
    # Load data and splits
    # ------------------------------------------------------------------
    df = pd.read_csv(args.student_data).reset_index(drop=True)
    with open(args.splits) as f:
        splits = json.load(f)

    val_df  = df.loc[splits["val"]].reset_index(drop=True)
    test_df = df.loc[splits["test"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Load rewrites (val split only)
    # ------------------------------------------------------------------
    with open(args.rewrites_json) as f:
        rdata = json.load(f)
    if "per_example" in rdata:
        examples = rdata["per_example"]
    else:
        config = args.config or next(iter(rdata))
        print(f"[info] Using config: '{config}'")
        examples = rdata[config]["per_example"]

    # Build rewrite map: original_text -> generated_text.
    # dp_icl results include "original" text; cluster results use "index" (df row index).
    rewrite_map: dict[str, str] = {}
    for ex in examples:
        if not ex.get("generated"):
            continue
        if ex.get("original"):
            rewrite_map[ex["original"]] = ex["generated"]
        elif ex.get("index") is not None:
            idx = int(ex["index"])
            if idx in df.index:
                orig_text = df.at[idx, args.post_col]
                if pd.notna(orig_text):
                    rewrite_map[str(orig_text)] = ex["generated"]
    print(f"[info] Loaded {len(rewrite_map)} rewrites")

    val_df["_rewrite"] = val_df[args.post_col].astype(str).map(rewrite_map)
    val_rw_mask = val_df["_rewrite"].notna()
    val_df_rw = val_df[val_rw_mask].reset_index(drop=True)
    print(f"[info] Val posts matched to rewrites: {len(val_df_rw)} / {len(val_df)}")

    # ------------------------------------------------------------------
    # Embed: val originals, val rewrites, test originals
    # ------------------------------------------------------------------
    print(f"\nLoading semantic model: {args.model}")
    sem_model = SentenceTransformer(args.model)
    print(f"Loading style model:   {args.style_model}")
    style_model = SentenceTransformer(args.style_model)

    texts_val_orig = val_df_rw[args.post_col].astype(str).tolist()
    texts_val_rw   = val_df_rw["_rewrite"].astype(str).tolist()
    texts_test     = test_df[args.post_col].astype(str).tolist()

    print("\nEmbedding with semantic model...")
    X_val_orig = embed(texts_val_orig, sem_model)
    X_val_rw   = embed(texts_val_rw,   sem_model)
    X_test     = embed(texts_test,     sem_model)

    print("\nEmbedding with style model...")
    S_val_orig = embed(texts_val_orig, style_model)
    S_val_rw   = embed(texts_val_rw,   style_model)
    S_test     = embed(texts_test,     style_model)

    # ------------------------------------------------------------------
    # Per-label evaluation
    # ------------------------------------------------------------------
    results = {}
    label_cols = [c for c in args.labels.split(",")
                  if c in df.columns and df[c].notna().sum() > 0]

    for label_col in label_cols:
        print(f"\n{'='*60}\nLabel: {label_col}")

        y_val_raw  = val_df_rw[label_col]
        y_test_raw = test_df[label_col]

        val_mask  = y_val_raw.notna().values
        test_mask = y_test_raw.notna().values

        if val_mask.sum() == 0 or test_mask.sum() == 0:
            print("  Skipping — no labels available.")
            continue

        task = detect_task(df[label_col].dropna())
        print(f"  Task: {task}  |  val n={val_mask.sum()}  test n={test_mask.sum()}")

        if task in ("binary", "multiclass"):
            le = LabelEncoder()
            le.fit(df[label_col].dropna().astype(str))
            y_val  = le.transform(y_val_raw[val_mask].astype(str))
            y_test = le.transform(
                pd.Series(y_test_raw[test_mask].astype(str))
                  .map(lambda x: x if x in le.classes_ else le.classes_[0])
            )

            # semantic
            clf_orig_sem = make_clf()
            clf_orig_sem.fit(X_val_orig[val_mask], y_val)
            clf_rw_sem = make_clf()
            clf_rw_sem.fit(X_val_rw[val_mask], y_val)
            res_orig_sem = eval_clf(clf_orig_sem, X_test[test_mask], y_test)
            res_rw_sem   = eval_clf(clf_rw_sem,   X_test[test_mask], y_test)

            # style
            clf_orig_sty = make_clf()
            clf_orig_sty.fit(S_val_orig[val_mask], y_val)
            clf_rw_sty = make_clf()
            clf_rw_sty.fit(S_val_rw[val_mask], y_val)
            res_orig_sty = eval_clf(clf_orig_sty, S_test[test_mask], y_test)
            res_rw_sty   = eval_clf(clf_rw_sty,   S_test[test_mask], y_test)

            print(f"  [sem]   orig acc={res_orig_sem['accuracy']:.3f} F1={res_orig_sem['macro_f1']:.3f} | "
                  f"rew acc={res_rw_sem['accuracy']:.3f} F1={res_rw_sem['macro_f1']:.3f} | "
                  f"Δacc={res_rw_sem['accuracy']-res_orig_sem['accuracy']:+.3f} ΔF1={res_rw_sem['macro_f1']-res_orig_sem['macro_f1']:+.3f}")
            print(f"  [style] orig acc={res_orig_sty['accuracy']:.3f} F1={res_orig_sty['macro_f1']:.3f} | "
                  f"rew acc={res_rw_sty['accuracy']:.3f} F1={res_rw_sty['macro_f1']:.3f} | "
                  f"Δacc={res_rw_sty['accuracy']-res_orig_sty['accuracy']:+.3f} ΔF1={res_rw_sty['macro_f1']-res_orig_sty['macro_f1']:+.3f}")

            results[label_col] = {
                "task": task, "classes": le.classes_.tolist(),
                "train_on_originals": res_orig_sem,
                "train_on_rewrites":  res_rw_sem,
                "style_train_on_originals": res_orig_sty,
                "style_train_on_rewrites":  res_rw_sty,
            }

        else:  # regression
            y_val  = y_val_raw[val_mask].astype(float).values
            y_test = y_test_raw[test_mask].astype(float).values

            # semantic
            reg_orig_sem = make_reg()
            reg_orig_sem.fit(X_val_orig[val_mask], y_val)
            reg_rw_sem = make_reg()
            reg_rw_sem.fit(X_val_rw[val_mask], y_val)
            res_orig_sem = eval_reg(reg_orig_sem, X_test[test_mask], y_test)
            res_rw_sem   = eval_reg(reg_rw_sem,   X_test[test_mask], y_test)

            # style
            reg_orig_sty = make_reg()
            reg_orig_sty.fit(S_val_orig[val_mask], y_val)
            reg_rw_sty = make_reg()
            reg_rw_sty.fit(S_val_rw[val_mask], y_val)
            res_orig_sty = eval_reg(reg_orig_sty, S_test[test_mask], y_test)
            res_rw_sty   = eval_reg(reg_rw_sty,   S_test[test_mask], y_test)

            print(f"  [sem]   orig R²={res_orig_sem['r2']:.3f} MAE={res_orig_sem['mae']:.3f} | "
                  f"rew R²={res_rw_sem['r2']:.3f} MAE={res_rw_sem['mae']:.3f} | "
                  f"Δ R²={res_rw_sem['r2']-res_orig_sem['r2']:+.3f} Δ MAE={res_rw_sem['mae']-res_orig_sem['mae']:+.3f}")
            print(f"  [style] orig R²={res_orig_sty['r2']:.3f} MAE={res_orig_sty['mae']:.3f} | "
                  f"rew R²={res_rw_sty['r2']:.3f} MAE={res_rw_sty['mae']:.3f} | "
                  f"Δ R²={res_rw_sty['r2']-res_orig_sty['r2']:+.3f} Δ MAE={res_rw_sty['mae']-res_orig_sty['mae']:+.3f}")

            results[label_col] = {
                "task": task,
                "train_on_originals": res_orig_sem,
                "train_on_rewrites":  res_rw_sem,
                "style_train_on_originals": res_orig_sty,
                "style_train_on_rewrites":  res_rw_sty,
            }

    output = {
        "rewrites_json": args.rewrites_json,
        "splits":        args.splits,
        "embed_model":   args.model,
        "style_model":   args.style_model,
        "results":       results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
