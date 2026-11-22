"""
06_aggregate_quality_metrics.py

Compute aggregate generation-quality and diversity metrics from eval-result
JSON files produced by 05_*.py scripts.

Metrics
-------
For generated posts (and originals when available):
  mattr                     -- Moving Average Type-Token Ratio (lexical
                               diversity; window = --mattr-window, default 50).
                               Computed per post, averaged across the set.
  self_bleu                 -- Corpus self-BLEU (each text as hypothesis, rest
                               as references; lower = more diverse).
  within_cluster_style_sim  -- Mean pairwise StyleDistance cosine similarity
                               among posts that share the same cluster label.
  between_cluster_style_sim -- Mean pairwise StyleDistance cosine similarity
                               across posts from different clusters.

Cluster grouping uses the axis named by --cluster-axis (default:
og_style_model).  Configs that lack per_axis cluster info (e.g. the simple
baseline) still get MATTR and self-BLEU; cluster metrics are reported as null.

Input JSON formats
------------------
Flat  (baseline): {"aggregate": {...}, "per_example": [...]}
Multi (sweep):    {"<config>": {"axis_settings": ...,
                                "aggregate":     ...,
                                "per_example":   [...]}, ...}
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# MATTR
# ---------------------------------------------------------------------------
def _mattr_one(text: str, window: int) -> float:
    tokens = text.lower().split()
    if len(tokens) < 2:
        return float("nan")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ttrs = [
        len(set(tokens[i : i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return float(np.mean(ttrs))


def mean_mattr(texts: list[str], window: int = 50) -> float:
    vals = [_mattr_one(t, window) for t in texts if t]
    valid = [v for v in vals if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Self-BLEU
# ---------------------------------------------------------------------------
def compute_self_bleu(texts: list[str], max_n: int = 500) -> float:
    """Corpus self-BLEU: each text as hypothesis, all others as references."""
    if len(texts) < 2:
        return float("nan")

    rng = random.Random(42)
    sample = texts if len(texts) <= max_n else rng.sample(texts, max_n)
    tok = [t.lower().split() for t in sample]

    refs_list = [[tok[j] for j in range(len(tok)) if j != i] for i in range(len(tok))]
    smooth = SmoothingFunction().method1
    return float(corpus_bleu(refs_list, tok, smoothing_function=smooth))


# ---------------------------------------------------------------------------
# Within / between cluster style similarity
# ---------------------------------------------------------------------------
def cluster_style_sims(
    texts: list[str],
    cluster_ids: list[int],
    style_model: SentenceTransformer,
) -> tuple[float, float]:
    """Return (within_mean, between_mean) cosine similarities."""
    embs = np.array(style_model.encode(texts, batch_size=64, show_progress_bar=False))
    cids = np.array(cluster_ids)
    unique = np.unique(cids)

    within: list[float] = []
    between: list[float] = []

    for cid in unique:
        mask = cids == cid
        in_e = embs[mask]
        out_e = embs[~mask]

        if in_e.shape[0] > 1:
            sim = cosine_similarity(in_e)
            iu = np.triu_indices(len(in_e), k=1)
            within.extend(sim[iu].tolist())

        if in_e.shape[0] > 0 and out_e.shape[0] > 0:
            cross = cosine_similarity(in_e, out_e)
            between.extend(cross.flatten().tolist())

    return (
        float(np.mean(within)) if within else float("nan"),
        float(np.mean(between)) if between else float("nan"),
    )


def mean_paired_sim(
    texts_a: list[str],
    texts_b: list[str],
    model: SentenceTransformer,
) -> float:
    """Mean cosine similarity between paired texts (e.g. generated vs original)."""
    assert len(texts_a) == len(texts_b)
    embs_a = np.array(model.encode(texts_a, batch_size=64, show_progress_bar=False))
    embs_b = np.array(model.encode(texts_b, batch_size=64, show_progress_bar=False))
    sims = [float(cosine_similarity([a], [b])[0, 0]) for a, b in zip(embs_a, embs_b)]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Bundle metrics for one set of texts
# ---------------------------------------------------------------------------
def compute_metrics(
    texts: list[str],
    cluster_ids: list[int] | None,
    style_model: SentenceTransformer | None,
    wegmann_model: SentenceTransformer | None,
    mattr_window: int,
    self_bleu_n: int,
    paired_texts: list[str] | None = None,
) -> dict:
    result: dict = {"n": len(texts)}
    result["mattr"] = mean_mattr(texts, mattr_window)
    result["self_bleu"] = compute_self_bleu(texts, self_bleu_n)

    if cluster_ids is not None and len(np.unique(cluster_ids)) > 1:
        if style_model is not None:
            within, between = cluster_style_sims(texts, cluster_ids, style_model)
            result["within_cluster_style_sim"] = within
            result["between_cluster_style_sim"] = between
        else:
            result["within_cluster_style_sim"] = None
            result["between_cluster_style_sim"] = None

        if wegmann_model is not None:
            within_w, between_w = cluster_style_sims(texts, cluster_ids, wegmann_model)
            result["within_cluster_wegmann_sim"] = within_w
            result["between_cluster_wegmann_sim"] = between_w
        else:
            result["within_cluster_wegmann_sim"] = None
            result["between_cluster_wegmann_sim"] = None
    else:
        result["within_cluster_style_sim"] = None
        result["between_cluster_style_sim"] = None
        result["within_cluster_wegmann_sim"] = None
        result["between_cluster_wegmann_sim"] = None

    # Paired similarity (generated vs original) using both models
    if paired_texts is not None and len(paired_texts) == len(texts):
        if style_model is not None:
            result["paired_style_sim"] = mean_paired_sim(texts, paired_texts, style_model)
        else:
            result["paired_style_sim"] = None
        if wegmann_model is not None:
            result["paired_wegmann_sim"] = mean_paired_sim(texts, paired_texts, wegmann_model)
        else:
            result["paired_wegmann_sim"] = None

    return result


# ---------------------------------------------------------------------------
# Extract records from per_example list
# ---------------------------------------------------------------------------
def extract_records(
    per_example: list[dict],
    cluster_axis: str,
    student_df: pd.DataFrame | None,
) -> tuple[list[str], list[str] | None, list[int] | None]:
    """Return (generated, originals, cluster_ids).

    All three lists are aligned: index i refers to the same eval post.
    originals / cluster_ids are None when the information is unavailable.
    """
    valid = [r for r in per_example if r.get("generated")]
    if not valid:
        return [], None, None

    generated = [r["generated"] for r in valid]

    # --- originals ---
    originals: list[str] | None = None
    if all("original" in r for r in valid):
        originals = [r["original"] for r in valid]
    elif student_df is not None:
        try:
            indices = [r["index"] for r in valid]
            originals = student_df.loc[indices, "original_post"].tolist()
        except Exception:
            originals = None

    # --- cluster IDs (target_cluster = original post's cluster label) ---
    cluster_ids: list[int] | None = None
    if valid and all("per_axis" in r for r in valid):
        first_axes = valid[0].get("per_axis", {})
        if cluster_axis in first_axes:
            try:
                cluster_ids = [
                    r["per_axis"][cluster_axis]["target_cluster"] for r in valid
                ]
            except Exception:
                cluster_ids = None

    return generated, originals, cluster_ids


# ---------------------------------------------------------------------------
# Load result file: detect flat vs multi-config format
# ---------------------------------------------------------------------------
def load_configs(path: str) -> dict[str, list[dict]]:
    """Return {config_name: per_example_list}."""
    with open(path) as f:
        data = json.load(f)

    if "per_example" in data:
        name = os.path.splitext(os.path.basename(path))[0]
        return {name: data["per_example"]}

    configs: dict[str, list[dict]] = {}
    for key, val in data.items():
        if isinstance(val, dict) and "per_example" in val:
            configs[key] = val["per_example"]
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate quality metrics: MATTR, self-BLEU, "
                    "within/between cluster style similarity."
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Result JSON files from 05_*.py scripts.")
    parser.add_argument("--student-data", default=None,
                        help="CSV with original_post column (used to look up "
                             "originals by index when not stored in the result file).")
    parser.add_argument("--cluster-axis", default="og_style_model",
                        help="Axis name for within/between cluster grouping.")
    parser.add_argument("--mattr-window", type=int, default=50)
    parser.add_argument("--self-bleu-n", type=int, default=500,
                        help="Max texts sampled for self-BLEU (for speed).")
    parser.add_argument("--no-style-emb", action="store_true",
                        help="Skip StyleDistance within/between cluster style sim.")
    parser.add_argument("--no-wegmann-emb", action="store_true",
                        help="Skip AnnaWegmann within/between cluster style sim.")
    parser.add_argument("--output", default="quality_metrics.json")
    args = parser.parse_args()

    student_df: pd.DataFrame | None = None
    if args.student_data:
        print(f"Loading student data from {args.student_data} ...")
        student_df = pd.read_csv(args.student_data).reset_index(drop=True)

    style_model: SentenceTransformer | None = None
    if not args.no_style_emb:
        print("Loading StyleDistance model ...")
        style_model = SentenceTransformer("StyleDistance/styledistance")

    wegmann_model: SentenceTransformer | None = None
    if not args.no_wegmann_emb:
        print("Loading AnnaWegmann/Style-Embedding model ...")
        wegmann_model = SentenceTransformer("AnnaWegmann/Style-Embedding")

    all_results: dict = {}

    for input_path in args.inputs:
        print(f"\n=== {input_path} ===")
        configs = load_configs(input_path)
        file_results: dict = {}

        for config_name, per_example in tqdm.tqdm(configs.items(), desc="Configs"):
            if not per_example:
                continue

            generated, originals, cluster_ids = extract_records(
                per_example, args.cluster_axis, student_df
            )
            if not generated:
                continue

            has_clusters = cluster_ids is not None
            effective_ids = cluster_ids if has_clusters else None

            print(f"\n  [{config_name}]  n={len(generated)}  "
                  f"has_originals={originals is not None}  "
                  f"has_clusters={has_clusters}  "
                  f"(axis={args.cluster_axis})")

            gen_metrics = compute_metrics(
                generated, effective_ids, style_model, wegmann_model,
                args.mattr_window, args.self_bleu_n,
                paired_texts=originals,
            )

            orig_metrics: dict | None = None
            if originals:
                orig_metrics = compute_metrics(
                    originals, effective_ids, style_model, wegmann_model,
                    args.mattr_window, args.self_bleu_n,
                )

            def _fmt(m: dict) -> str:
                w  = m.get("within_cluster_style_sim")
                b  = m.get("between_cluster_style_sim")
                ww = m.get("within_cluster_wegmann_sim")
                bw = m.get("between_cluster_wegmann_sim")
                ps = m.get("paired_style_sim")
                pw = m.get("paired_wegmann_sim")
                parts = [
                    f"mattr={m['mattr']:.4f}",
                    f"self_bleu={m['self_bleu']:.4f}",
                    f"within_sty={'n/a' if w is None else f'{w:.4f}'}",
                    f"between_sty={'n/a' if b is None else f'{b:.4f}'}",
                    f"within_weg={'n/a' if ww is None else f'{ww:.4f}'}",
                    f"between_weg={'n/a' if bw is None else f'{bw:.4f}'}",
                ]
                if ps is not None:
                    parts.append(f"paired_sty={ps:.4f}")
                if pw is not None:
                    parts.append(f"paired_weg={pw:.4f}")
                return "  ".join(parts)

            print(f"    generated  {_fmt(gen_metrics)}")
            if orig_metrics:
                print(f"    original   {_fmt(orig_metrics)}")

            file_results[config_name] = {
                "generated": gen_metrics,
                "original":  orig_metrics,
                "cluster_axis": args.cluster_axis if has_clusters else None,
            }

        all_results[input_path] = file_results

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
