"""
Stage 1: Clustering sweep.

Modified from the original all_embeddings_clustering.py in two important ways:

1. Drops the non-viable configurations (rws_*, rr_general) identified in the
   previous analysis — these never achieve <53% noise and we've decided not to
   spend further evaluation budget on them.

2. Saves the HDBSCAN clusterer AND the UMAP reducer for each configuration.
   Downstream scripts need both to project generated posts into cluster space
   via approximate_predict.  Without these, we couldn't measure cluster-match
   rate on reconstructed posts.

The HDBSCAN hyperparameter grid is kept unchanged (mcs × ms). If we later want
to explore finer granularity we can extend the grid selectively for top-ranked
configurations rather than globally.

Input: a CSV produced by rewrite_posts.py that has at minimum two columns:
    original_post  — the raw student post text
    rewrites       — the symbol-substituted rewrite of that post

Run after rewrite_posts.py.  Run time depends on your GPU for embedding;
clustering itself is CPU and fast.
"""

import argparse
import itertools
import json
import os
import pickle

import hdbscan
import numpy as np
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from umap import UMAP


HDBSCAN_GRID = list(itertools.product(
    [100, 150, 200],       # min_cluster_size
    [10, 15, 20, 30],     # min_samples
))

# Key: name used for output paths.
# Value: (text_column_key, embedding_model_name).
# text_column_key selects which column of the DataFrame to embed.
VIABLE_CONFIGS = {
    # "og_general_model":  ("original_post",  "all-mpnet-base-v2"),
    # "og_qa_model":       ("original_post",  "multi-qa-mpnet-base-dot-v1"),
    "og_style_model":    ("original_post",  "StyleDistance/styledistance"),
    # "rr_qa_model":       ("rewrites",       "multi-qa-mpnet-base-dot-v1"),
    # dropped: rr_general, rws_*  — never achieve viable noise
    # dropped: kw_*                — keyword expansion not well justified
}



def run_hdbscan_sweep(emb_umap: np.ndarray):
    """Run HDBSCAN across the hyperparameter grid.  Returns per-combo results
    AND the fitted clusterer objects (the originals don't keep these)."""
    results = {}
    clusterers = {}

    for mcs, ms in tqdm.tqdm(HDBSCAN_GRID, desc="HDBSCAN sweep"):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=True,  # <-- required for approximate_predict later
        )
        lbls = clusterer.fit_predict(emb_umap)
        mask = lbls >= 0
        noise_frac = float((~mask).mean())
        n_clusters = len(set(lbls[mask]))

        sil = float("nan")
        if n_clusters > 2 and mask.sum() > 1:
            sil = float(silhouette_score(
                emb_umap[mask], lbls[mask],
                sample_size=min(2000, mask.sum()), random_state=42,
            ))

        if n_clusters > 0:
            sizes = pd.Series(lbls[mask]).value_counts()
            min_size, median_size = int(sizes.min()), float(sizes.median())
        else:
            min_size, median_size = 0, 0.0

        key = f"mcs{mcs}_ms{ms}"
        results[key] = {
            "best_k": n_clusters,
            "silhouette": sil,
            "noise_frac": noise_frac,
            "min_cluster_size": min_size,
            "median_cluster_size": median_size,
            "labels_path": None,  # filled in by caller
        }
        clusterers[key] = clusterer

    return results, clusterers


def run_config(texts, model_name, output_prefix, umap_components=32, seed=42):
    print(f"\n=== {output_prefix} ({model_name}) ===")
    os.makedirs(output_prefix, exist_ok=True)

    model = SentenceTransformer(model_name)
    emb = np.array(model.encode(texts, batch_size=64, show_progress_bar=True))

    reducer = UMAP(n_components=umap_components, metric="cosine", random_state=seed)
    emb_umap = reducer.fit_transform(emb)

    results, clusterers = run_hdbscan_sweep(emb_umap)

    # Persist everything we'll need downstream
    np.save(os.path.join(output_prefix, "embeddings.npy"), emb)
    np.save(os.path.join(output_prefix, "embeddings_umap.npy"), emb_umap)
    with open(os.path.join(output_prefix, "umap_reducer.pkl"), "wb") as f:
        pickle.dump(reducer, f)

    for key, clusterer in clusterers.items():
        lbls = clusterer.labels_
        labels_path = os.path.join(output_prefix, f"labels_{key}.npy")
        np.save(labels_path, lbls)
        clusterer_path = os.path.join(output_prefix, f"clusterer_{key}.pkl")
        with open(clusterer_path, "wb") as f:
            pickle.dump(clusterer, f)
        results[key]["labels_path"] = labels_path
        results[key]["clusterer_path"] = clusterer_path

    with open(os.path.join(output_prefix, "hdbscan_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def load_dataframe(student_data: str, splits: str | None, split_name: str) -> pd.DataFrame:
    """Load the post+rewrite CSV and optionally filter to a train/val/test split."""
    df = pd.read_csv(student_data).reset_index(drop=True)
    if splits is not None:
        with open(splits) as f:
            split_indices = json.load(f)[split_name]
        before = len(df)
        df = df.loc[split_indices].reset_index(drop=True)
        print(f"Filtered corpus to '{split_name}' split: {len(df)}/{before} posts")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student-data", required=True,
        help="CSV produced by rewrite_posts.py; must have 'original_post' and "
             "'rewrites' columns.",
    )
    parser.add_argument(
        "--output-root",
        default="final_data/stack_exchange/hdbscan_results/",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for UMAP projection.")
    parser.add_argument("--splits", default=None,
                        help="Path to eval_splits.json. When provided, clustering "
                             "runs only on the selected split.")
    parser.add_argument("--split-name", default="train",
                        help="Which split key to use as the clustering corpus.")
    args = parser.parse_args()

    df = load_dataframe(args.student_data, args.splits, args.split_name)

    # Apply symbol substitution to the rewrites column in place for clustering.
    df = df.copy()
    # df["rewrites"] = df["rewrites"].astype(str).apply(replace_symbols)

    for config_name, (text_col, model_name) in VIABLE_CONFIGS.items():
        texts = df[text_col].astype(str).tolist()
        output_prefix = os.path.join(args.output_root, config_name)
        run_config(texts, model_name, output_prefix, seed=args.seed)


if __name__ == "__main__":
    main()