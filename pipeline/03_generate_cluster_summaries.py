"""
Stage 1b: Generate natural-language summaries for each cluster.

Run after 02_clustering_sweep.py, before the cluster-based eval scripts.
Writes cluster_summaries.json to each axis directory under --clustering-root.

Each summary is contrastive: the LM sees focal posts from the target cluster
alongside contrast posts sampled from other clusters, producing descriptions
that highlight what makes each cluster distinctive.  An ensemble of
--n-summaries candidates is generated and the one closest to their embedding
centroid (via text-embedding-3-small) is kept.

Sampling strategy: centroid-based for focal posts (posts closest to the
cluster centroid in UMAP-reduced space), random for contrast posts.

Only HDBSCAN settings whose labels_<key>.npy files exist are processed;
missing settings are silently skipped.
"""

import argparse
import concurrent.futures
import itertools
import json
import os
import random

import dotenv
import dspy
import numpy as np
import openai
import pandas as pd
import tqdm

dotenv.load_dotenv()

# Must match 02_clustering_sweep.py.
VIABLE_CONFIGS = {
    "og_style_model": "original_post",
}

_HDBSCAN_GRID_KEYS = [
    f"mcs{mcs}_ms{ms}"
    for mcs, ms in itertools.product([100, 150, 200], [10, 15, 20, 30])
]

CLUSTER_SUMMARY_PROMPTS = {
    "og_general_model": """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts \
from other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the topical or semantic content these focal posts share \
with student discussion posts in general (the baseline subject matter).
Sentence 2: describe specifically what TOPICAL or SEMANTIC content distinguishes \
the focal posts from the contrast posts — what makes this cluster's subject matter unique?
Be concrete; avoid phrases like "various topics". Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:""",

    "og_qa_model": """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts \
from other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the question structure and topical content these focal posts \
share with student discussion posts in general (the baseline: what kind of questions, about what).
Sentence 2: describe specifically what QUESTION STRUCTURE or TOPICAL angle distinguishes \
the focal posts from the contrast posts — what makes this cluster's questions unique?
Be concrete. Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:""",

    "og_style_model": """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts \
from other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the writing style and discourse features these focal posts \
share with student discussion posts in general (the baseline register: informal, first-person, etc.).
Sentence 2: describe specifically what WRITING STYLE or DISCOURSE features distinguish \
the focal posts from the contrast posts — what makes this cluster's style unique?
Focus on style and discourse patterns, not topic. Be concrete. Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:""",

    "rr_qa_model": """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts \
from other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the structural templates and interaction patterns these focal posts \
share with student discussion posts in general (the baseline format and move sequence).
Sentence 2: describe specifically what STRUCTURAL TEMPLATE or INTERACTION PATTERN \
distinguishes the focal posts from the contrast posts — what makes this cluster's structure unique?
Be concrete. Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:""",
}


def make_embed_fn(model="text-embedding-3-small"):
    client = openai.OpenAI()
    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
    return embed


def centroid_sample_indices(emb_umap, labels, cluster_id, n):
    cluster_indices = np.where(labels == cluster_id)[0]
    if len(cluster_indices) <= n:
        return cluster_indices.tolist()
    cluster_embs = emb_umap[cluster_indices]
    dists = np.linalg.norm(cluster_embs - cluster_embs.mean(axis=0), axis=1)
    return cluster_indices[np.argsort(dists)[:n]].tolist()


def summarize_cluster(lm, embed_fn, examples, contrast_examples, cluster_id, axis, n_summaries):
    """Generate n_summaries candidates; return the one closest to their embedding centroid."""
    prompt_template = CLUSTER_SUMMARY_PROMPTS.get(axis, CLUSTER_SUMMARY_PROMPTS["og_style_model"])
    formatted = "\n\n".join(f"[Focal Post {i+1}]\n{p}" for i, p in enumerate(examples))
    formatted_contrast = "\n\n".join(
        f"[Contrast Post {i+1}]\n{p}" for i, p in enumerate(contrast_examples)
    )
    prompt = prompt_template.format(
        n=len(examples), examples=formatted,
        n_contrast=len(contrast_examples), contrast_examples=formatted_contrast,
    )

    candidates: list[str] = []
    for _ in range(n_summaries):
        try:
            text = lm(messages=[{"role": "user", "content": prompt}])[0].strip()
            if text:
                candidates.append(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] cluster {cluster_id} summary attempt failed: {exc}")

    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    try:
        embs = np.array(embed_fn(candidates))
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_normed = embs / np.where(norms == 0, 1, norms)
        centroid = embs_normed.mean(axis=0)
        sims = embs_normed @ centroid
        return candidates[int(np.argmax(sims))]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] cluster {cluster_id} ensemble selection failed: {exc}")
        return candidates[0]


def summarize_setting(config_dir, texts, hdbscan_key, axis, lm, embed_fn,
                      n_samples, n_contrast, n_summaries, workers, rng):
    labels_path = os.path.join(config_dir, f"labels_{hdbscan_key}.npy")
    if not os.path.exists(labels_path):
        return None  # silently skip — not all grid keys will be present

    labels = np.load(labels_path)
    emb_umap = np.load(os.path.join(config_dir, "embeddings_umap.npy"))
    unique_clusters = sorted(set(labels[labels >= 0].tolist()))
    if not unique_clusters:
        return {}

    # Build per-cluster text lists for contrast sampling.
    cluster_texts: dict[int, list[str]] = {
        cid: [texts[i] for i in np.where(labels == cid)[0] if i < len(texts)]
        for cid in unique_clusters
    }

    tasks: dict[int, tuple[list[str], list[str]]] = {}
    for cid in unique_clusters:
        focal_idx = centroid_sample_indices(emb_umap, labels, cid, n_samples)
        focal = [texts[i] for i in focal_idx]

        other_texts = [t for oid, ts in cluster_texts.items() if oid != cid for t in ts]
        contrast = rng.sample(other_texts, min(n_contrast, len(other_texts))) if other_texts else []
        tasks[cid] = (focal, contrast)

    summaries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_cid = {
            executor.submit(
                summarize_cluster, lm, embed_fn, focal, contrast, cid, axis, n_summaries,
            ): cid
            for cid, (focal, contrast) in tasks.items()
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_cid),
            total=len(future_to_cid),
            desc=f"  {hdbscan_key}",
        ):
            summaries[int(future_to_cid[future])] = future.result()
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive cluster summaries for all HDBSCAN settings."
    )
    parser.add_argument("--student-data", required=True,
                        help="CSV with original_post column.")
    parser.add_argument("--splits", required=True,
                        help="eval_splits_seed<N>.json; summaries use the train split.")
    parser.add_argument("--clustering-root", required=True,
                        help="Per-seed clustering root, e.g. hdbscan_results/seed_42/.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embed-model", default="text-embedding-3-small",
                        help="OpenAI embedding model used for ensemble candidate selection.")
    parser.add_argument("--n-samples", type=int, default=15,
                        help="Focal posts per cluster (centroid-closest).")
    parser.add_argument("--n-contrast", type=int, default=5,
                        help="Contrast posts sampled from other clusters.")
    parser.add_argument("--n-summaries", type=int, default=10,
                        help="Summary candidates generated per cluster; best is kept.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for contrast post sampling.")
    parser.add_argument("--axis", default=None,
                        help="Process only this axis. Omit to process all axes in VIABLE_CONFIGS.")
    args = parser.parse_args()

    student_data = pd.read_csv(args.student_data).reset_index(drop=True)
    with open(args.splits) as f:
        splits = json.load(f)
    train_df = student_data.loc[splits["train"]]

    lm = dspy.LM(args.model)
    embed_fn = make_embed_fn(args.embed_model)
    rng = random.Random(args.seed)

    for axis, text_col in VIABLE_CONFIGS.items():
        if args.axis and axis != args.axis:
            continue
        config_dir = os.path.join(args.clustering_root, axis)
        if not os.path.isdir(config_dir):
            print(f"[skip] {axis} - directory not found")
            continue

        texts = train_df[text_col].astype(str).tolist()
        print(f"\n=== {axis} ({len(texts)} train posts) ===")

        out_path = os.path.join(config_dir, "cluster_summaries.json")
        # Load existing summaries so we only generate missing keys.
        if os.path.exists(out_path):
            with open(out_path) as f:
                all_summaries = json.load(f)
            print(f"  loaded {len(all_summaries)} existing summaries from {out_path}")
        else:
            all_summaries = {}

        missing_keys = [k for k in _HDBSCAN_GRID_KEYS if k not in all_summaries]
        print(f"  generating summaries for {len(missing_keys)} missing key(s): {missing_keys}")
        for hdbscan_key in missing_keys:
            result = summarize_setting(
                config_dir, texts, hdbscan_key, axis, lm, embed_fn,
                args.n_samples, args.n_contrast, args.n_summaries, args.workers, rng,
            )
            if result is not None:
                all_summaries[hdbscan_key] = result

        with open(out_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
