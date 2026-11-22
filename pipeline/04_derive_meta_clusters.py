"""
Stage 4: Derive meta-clusters from cluster centers across seeds.

For each axis, loads the cluster-member embeddings produced by the winning
HDBSCAN setting (from Stage 3 / pass_a_winners.json).  Each cluster is
represented by its center: the L2-normalised mean of its member raw post
embeddings.  All-But-the-Top (ABTT) is applied to the matrix of cluster
centers to strip the dominant shared directions (the generic "student post"
signal) that otherwise compress all pairwise similarities into a tight band
and make community detection unreliable.

The (ABTT-D, k) pair is then chosen by average-linkage hierarchical clustering
to jointly maximise:
  1. n_multi_seed  — clusters represented in ≥ min-seeds-multi seeds (primary)
  2. n_clusters    — total number of meta-clusters (secondary)
  3. silhouette    — cluster quality in ABTT space (tiebreaker)

After clustering, a fresh natural-language summary is generated for each
meta-cluster via an LLM ensemble: 10 candidates at temperature=0.8, with the
one closest to their embedding centroid kept as the representative.

Outputs per axis in --output-dir/<axis>/:
    merge_map.json        {seed: {orig_cid: meta_id}}
    merged_summaries.json {meta_id: representative_summary_text}
    audit.json            n_meta_clusters, coverage stats, per-cluster detail

Run as:
    python 04_derive_meta_clusters.py \\
        --student-data final_data/stack_exchange_v2.csv \\
        --splits final_data/stack_exchange/eval_splits_seed0.json
    python 04_derive_meta_clusters.py --winners-file sweep_output/pass_a_winners.json
    python 04_derive_meta_clusters.py --abtt-d 3 5 7 10 --max-k 20
"""

import argparse
import concurrent.futures
import json
import os
from collections import defaultdict
from glob import glob

import dotenv
import dspy
import numpy as np
import openai
import pandas as pd
import tqdm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEEDS = ["0", "7", "42", "99", "123"]

OPTIMAL = {
    "og_style_model": "mcs100_ms15",
}

ABTT_D_VALUES = [0, 1, 3, 5, 7, 11, 13]
MAX_K         = 25

# ---------------------------------------------------------------------------
# Axis-specific summarization prompts (mirrors 03_generate_cluster_summaries.py)
# ---------------------------------------------------------------------------
_PROMPT_SEMANTIC = """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts from \
other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the topical or semantic content these focal posts share with \
student discussion posts in general (the baseline subject matter).
Sentence 2: describe specifically what TOPICAL or SEMANTIC content distinguishes \
the focal posts from the contrast posts — what makes this cluster's subject matter unique?
Be concrete; avoid phrases like "various topics". Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:"""

_PROMPT_STYLE = """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts from \
other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the writing style and discourse features these focal posts share \
with student discussion posts in general (the baseline register: informal, first-person, etc.).
Sentence 2: describe specifically what WRITING STYLE or DISCOURSE features distinguish \
the focal posts from the contrast posts — what makes this cluster's style unique?
Focus on style and discourse patterns, not topic. Be concrete. Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:"""

_PROMPT_QA = """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts from \
other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the question structure and topical content these focal posts share \
with student discussion posts in general (the baseline: what kind of questions, about what).
Sentence 2: describe specifically what QUESTION STRUCTURE or TOPICAL angle distinguishes \
the focal posts from the contrast posts — what makes this cluster's questions unique?
Be concrete. Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:"""

_PROMPT_REWRITE = """\
You will see {n} FOCAL posts from one cluster and {n_contrast} CONTRAST posts from \
other clusters. Write a 2-sentence summary of the focal cluster.
Sentence 1: describe the structural templates and interaction patterns these focal posts \
share with student discussion posts in general (the baseline format and move sequence).
Sentence 2: describe specifically what STRUCTURAL TEMPLATE or INTERACTION PATTERN \
distinguishes the focal posts from the contrast posts — what makes this cluster's structure unique?
Be concrete. Output the summary only, no preamble.

Focal posts:
{examples}

Contrast posts (from other clusters):
{contrast_examples}

Summary:"""

_AXIS_PROMPT = {
    "og_general_model": _PROMPT_SEMANTIC,
    "og_qa_model":      _PROMPT_QA,
    "og_style_model":   _PROMPT_STYLE,
    "rr_qa_model":      _PROMPT_REWRITE,
}


# ---------------------------------------------------------------------------
# Ensemble summary generation
# ---------------------------------------------------------------------------
def make_embed_fn(model="text-embedding-3-small"):
    """Return a callable that embeds a list of strings → list of float vectors."""
    client = openai.OpenAI()
    def embed(texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(model=model, input=texts)
        return [item.embedding for item in response.data]
    return embed


def summarize_cluster(lm, embed_fn, examples, contrast_examples, cluster_id, axis, n_summaries=10):
    """Generate n_summaries candidates, then return the one closest to their embedding mean."""
    prompt_template = _AXIS_PROMPT.get(axis, _PROMPT_SEMANTIC)
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
            print(f"[WARN] meta-cluster {cluster_id} summary attempt failed: {exc}")

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
        print(f"[WARN] meta-cluster {cluster_id} ensemble selection failed: {exc}")
        return candidates[0]


# ---------------------------------------------------------------------------
# Cross-seed HDBSCAN winner selection
# ---------------------------------------------------------------------------
def pick_winners_from_sweep(sweep_outputs_dir, axes):
    """Average pass_a_scores.json across all seed subdirs and return best key per axis."""
    scores_per_axis = {a: {} for a in axes}
    paths = sorted(glob(os.path.join(sweep_outputs_dir, "seed_*", "pass_a_scores.json")))
    if not paths:
        print(f"  [warn] no pass_a_scores.json found under {sweep_outputs_dir}")
        return {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for axis in axes:
            for key, score in data.get(axis, {}).items():
                if "mcs50" not in key:
                    scores_per_axis[axis].setdefault(key, []).append(score)
    winners = {}
    for axis, key_scores in scores_per_axis.items():
        if not key_scores:
            continue
        best = max(key_scores, key=lambda k: np.mean(key_scores[k]))
        mean_score = float(np.mean(key_scores[best]))
        print(f"  {axis} winner (avg over {len(paths)} seeds): {best}  memsim={mean_score:.4f}")
        winners[axis] = best
    return winners


# ---------------------------------------------------------------------------
# Load cluster centers from raw post embeddings
# ---------------------------------------------------------------------------
def load_cluster_centers(axis, hdbscan_key, seeds, clustering_root):
    """Return one node per (seed, cluster_id) with its L2-normed center embedding."""
    nodes = []
    for seed in seeds:
        seed_dir    = os.path.join(clustering_root, f"seed_{seed}", axis)
        emb_path    = os.path.join(seed_dir, "embeddings.npy")
        labels_path = os.path.join(seed_dir, f"labels_{hdbscan_key}.npy")
        if not os.path.exists(emb_path) or not os.path.exists(labels_path):
            print(f"  [skip] missing artifacts for seed={seed}")
            continue
        embs   = np.load(emb_path)
        labels = np.load(labels_path)
        for cid in sorted(set(labels[labels >= 0].tolist())):
            member_embs = embs[labels == cid]
            center = member_embs.mean(axis=0)
            norm   = np.linalg.norm(center)
            center = center / norm if norm > 0 else center
            nodes.append({
                "seed":      seed,
                "cid":       int(cid),
                "center":    center,
                "n_members": int((labels == cid).sum()),
            })
    return nodes


# ---------------------------------------------------------------------------
# ABTT + hierarchical clustering
# ---------------------------------------------------------------------------
def abtt(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """All-But-the-Top (Mu et al., 2018): remove mean and top-D PCs, re-normalise."""
    emb = embeddings - embeddings.mean(axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(emb)
    emb = emb - (emb @ pca.components_.T) @ pca.components_
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1, norms)


def _cluster_stats(labels, nodes, min_seeds_multi):
    """Return (n_multi_seed, n_clusters, n_singletons) for a flat label array."""
    meta_seeds: dict[int, set] = {}
    for i, node in enumerate(nodes):
        meta_seeds.setdefault(int(labels[i]), set()).add(node["seed"])
    n_clusters   = len(meta_seeds)
    n_multi      = sum(1 for s in meta_seeds.values() if len(s) >= min_seeds_multi)
    n_singletons = sum(1 for s in meta_seeds.values() if len(s) == 1)
    return n_multi, n_clusters, n_singletons


def best_k_by_objective(emb_abtt, nodes, max_k, min_seeds_multi=3):
    """Average-linkage hierarchical clustering; return (labels, k, silhouette).

    Objective: maximise n_multi_seed (primary), then n_clusters (secondary),
    then silhouette (tiebreaker).
    """
    dists = pdist(emb_abtt, metric="cosine")
    Z     = linkage(dists, method="average")

    candidates = []
    for k in range(2, min(max_k + 1, len(emb_abtt))):
        raw_labels = fcluster(Z, t=k, criterion="maxclust") - 1
        if len(set(raw_labels.tolist())) < 2:
            continue
        n_multi, n_clusters, _ = _cluster_stats(raw_labels, nodes, min_seeds_multi)
        try:
            sil = float(silhouette_score(emb_abtt, raw_labels, metric="cosine"))
        except ValueError:
            sil = -1.0
        candidates.append((-n_multi, -n_clusters, -sil, k, raw_labels.copy()))

    if not candidates:
        return fcluster(Z, t=2, criterion="maxclust") - 1, 2, 0.0

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    best = candidates[0]
    return best[4], best[3], -best[2]


def sweep_abtt_and_k(centers, nodes, d_values, max_k, min_seeds_multi=3, verbose=True):
    """Sweep ABTT-D values; return (labels, k, silhouette, best_D)."""
    candidates = []
    for D in d_values:
        if D >= len(centers):
            continue
        emb_abtt = abtt(centers, n_components=D)
        labels, k, sil = best_k_by_objective(emb_abtt, nodes, max_k, min_seeds_multi)
        n_multi, n_clusters, n_singletons = _cluster_stats(labels, nodes, min_seeds_multi)
        if verbose:
            print(f"      D={D:2d}  k={k:2d}  sil={sil:.4f}  "
                  f"multi_seed={n_multi}  clusters={n_clusters}  singletons={n_singletons}")
        candidates.append((-n_multi, -n_clusters, -sil, D, k, labels))

    if not candidates:
        raise ValueError("No valid ABTT/k combinations found.")

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    best = candidates[0]
    return best[5], best[4], -best[2], best[3]


# ---------------------------------------------------------------------------
# Meta-cluster summary generation
# ---------------------------------------------------------------------------
def load_corpus_texts(student_data_path, splits_path):
    """Return the ordered list of original_post texts matching embeddings.npy row order."""
    df = pd.read_csv(student_data_path).reset_index(drop=True)
    with open(splits_path) as f:
        train_indices = json.load(f)["train"]
    return df.loc[train_indices].reset_index(drop=True)["original_post"].astype(str).tolist()


def generate_meta_summaries(nodes, labels, axis, hdbscan_key,
                             clustering_root, seed_texts, lm, embed_fn,
                             n_samples, n_summaries, n_contrast, workers):
    """Generate a summary for each meta-cluster and return per-cluster stats.

    Meta-centroids are computed in the original post embedding space (cluster
    centers), which is used both for representative node selection and for
    sampling the closest posts to pass to the LLM.

    Returns {meta_id: {"summary", "n_members", "representative_seed",
                        "representative_cid", "centroid_sim"}}.
    """
    seeds = sorted({n["seed"] for n in nodes})

    # Build text → L2-normed post embedding map across all seeds
    text_to_emb: dict[str, np.ndarray] = {}
    for seed in seeds:
        emb_path = os.path.join(clustering_root, f"seed_{seed}", axis, "embeddings.npy")
        if not os.path.exists(emb_path):
            continue
        embs = np.load(emb_path)
        for i, text in enumerate(seed_texts.get(seed, [])):
            if text not in text_to_emb and i < len(embs):
                text_to_emb[text] = embs[i]
    if not text_to_emb:
        print("  [warn] no post embeddings found; skipping summary generation")
        return {}

    # Pre-load per-(seed, cid) post indices
    seed_cid_indices: dict[tuple, np.ndarray] = {}
    for seed in seeds:
        seed_dir  = os.path.join(clustering_root, f"seed_{seed}", axis)
        lbls_path = os.path.join(seed_dir, f"labels_{hdbscan_key}.npy")
        if not os.path.exists(lbls_path):
            continue
        lbls = np.load(lbls_path)
        for cid in set(lbls[lbls >= 0].tolist()):
            seed_cid_indices[(seed, int(cid))] = np.where(lbls == cid)[0]

    # Pre-build the full text set per meta-cluster for contrast sampling
    meta_texts: dict[int, set[str]] = defaultdict(set)
    for i, node in enumerate(nodes):
        mid    = int(labels[i])
        stexts = seed_texts.get(node["seed"], [])
        for idx in seed_cid_indices.get((node["seed"], node["cid"]), []):
            if idx < len(stexts):
                meta_texts[mid].add(stexts[idx])

    rng = np.random.default_rng(42)

    tasks: dict[int, tuple[list[str], list[str]]] = {}  # meta_id -> (focal, contrast)
    representative_info: dict[int, dict] = {}

    for meta_id in sorted(set(labels.tolist())):
        mask       = np.where(labels == meta_id)[0]
        meta_nodes = [nodes[i] for i in mask]

        # Meta-centroid: mean of member cluster centers (already L2-normed)
        raw_center  = np.mean([n["center"] for n in meta_nodes], axis=0)
        norm        = np.linalg.norm(raw_center)
        meta_center = raw_center / norm if norm > 0 else raw_center

        # Representative node: cluster center closest to meta-centroid
        center_sims = np.array([float(n["center"] @ meta_center) for n in meta_nodes])
        best_node   = meta_nodes[int(np.argmax(center_sims))]

        representative_info[meta_id] = {
            "n_members":           sum(n["n_members"] for n in meta_nodes),
            "representative_seed": best_node["seed"],
            "representative_cid":  best_node["cid"],
            "centroid_sim":        float(center_sims.max()),
        }

        # Union of post texts across all member clusters, deduplicated
        all_texts_set: set[str] = set()
        for mnode in meta_nodes:
            stexts = seed_texts.get(mnode["seed"], [])
            for idx in seed_cid_indices.get((mnode["seed"], mnode["cid"]), []):
                if idx < len(stexts):
                    all_texts_set.add(stexts[idx])
        all_texts = sorted(all_texts_set)

        if not all_texts:
            tasks[meta_id] = ([], [])
            continue

        # Sample n_samples posts closest to meta-centroid in post embedding space
        emb_list = [text_to_emb[t] for t in all_texts if t in text_to_emb]
        if not emb_list:
            focal = all_texts[:n_samples]
        else:
            member_embs = np.array(emb_list)
            norms = np.linalg.norm(member_embs, axis=1, keepdims=True)
            member_embs_normed = member_embs / np.where(norms == 0, 1, norms)
            cosine_sims = member_embs_normed @ meta_center
            top = np.argsort(-cosine_sims)[:min(n_samples, len(all_texts))]
            focal = [all_texts[i] for i in top]

        # Sample n_contrast posts from other meta-clusters
        other_texts = sorted(
            set().union(*(texts for mid, texts in meta_texts.items() if mid != meta_id))
            - set(all_texts)
        )
        if other_texts and n_contrast > 0:
            idx = rng.choice(len(other_texts), size=min(n_contrast, len(other_texts)), replace=False)
            contrast = [other_texts[i] for i in sorted(idx)]
        else:
            contrast = []

        tasks[meta_id] = (focal, contrast)

    # Generate summaries in parallel (each job: n_summaries LLM calls + 1 embed call)
    summaries: dict[int, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {
            executor.submit(
                summarize_cluster, lm, embed_fn, focal, contrast, meta_id, axis, n_summaries,
            ): meta_id
            for meta_id, (focal, contrast) in tasks.items()
            if focal
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(future_to_meta),
            total=len(future_to_meta),
            desc="  meta-cluster summaries",
        ):
            summaries[future_to_meta[future]] = future.result()

    return {
        meta_id: {"summary": summaries.get(meta_id, ""), **representative_info[meta_id]}
        for meta_id in representative_info
    }


# ---------------------------------------------------------------------------
# Build outputs
# ---------------------------------------------------------------------------
def build_merge_map(nodes, labels, seeds):
    """Build {seed_str: {orig_cid_str: meta_id_int}}."""
    merge_map = {s: {} for s in seeds}
    for i, node in enumerate(nodes):
        merge_map[node["seed"]][str(node["cid"])] = int(labels[i])
    return merge_map


def build_audit(nodes, labels, meta_summaries, min_seeds_for_multi=3):
    """Build the audit dict saved to audit.json."""
    meta_to_seeds  = defaultdict(list)
    meta_to_cids   = defaultdict(list)
    seed_cid_count = defaultdict(int)

    for i, node in enumerate(nodes):
        mid = int(labels[i])
        if mid < 0:
            continue
        meta_to_seeds[mid].append(node["seed"])
        meta_to_cids[mid].append(node["cid"])
        seed_cid_count[(mid, node["seed"])] += 1

    same_seed_merges = [
        {"meta": mid, "seed": seed, "count": cnt}
        for (mid, seed), cnt in seed_cid_count.items()
        if cnt > 1
    ]

    per_meta = {}
    n_singletons = 0
    n_multi_seed = 0
    for mid in sorted(meta_to_seeds):
        seeds_in = sorted(set(meta_to_seeds[mid]))
        n_seeds  = len(seeds_in)
        size     = len(meta_to_seeds[mid])
        if size == 1:
            n_singletons += 1
        if n_seeds >= min_seeds_for_multi:
            n_multi_seed += 1
        info = meta_summaries.get(mid, {})
        per_meta[str(mid)] = {
            "size":                size,
            "n_seeds":             n_seeds,
            "seeds":               seeds_in,
            "orig_cids":           meta_to_cids[mid],
            "n_members":           info.get("n_members"),
            "representative_seed": info.get("representative_seed"),
            "representative_cid":  info.get("representative_cid"),
            "centroid_sim":        info.get("centroid_sim"),
        }

    return {
        "n_meta_clusters":  len(meta_to_seeds),
        "n_singletons":     n_singletons,
        "n_multi_seed":     n_multi_seed,
        "same_seed_merges": same_seed_merges,
        "per_meta_cluster": per_meta,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Derive meta-clusters via ABTT + hierarchical clustering on cluster centers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clustering-root",
        default="final_data/stack_exchange/hdbscan_results/",
    )
    parser.add_argument("--output-dir", default="meta_clustering_output")
    parser.add_argument(
        "--student-data",
        default="final_data/stack_exchange_v2.csv",
        help="CSV with original_post column; used for meta-cluster summarization.",
    )
    parser.add_argument(
        "--splits",
        default="final_data/stack_exchange/eval_splits_seed0.json",
        help="Fallback splits file when --splits-dir is not provided.",
    )
    parser.add_argument(
        "--splits-dir", default=None,
        help="Directory containing eval_splits_seed{N}.json files, one per seed.",
    )
    parser.add_argument(
        "--winners-file", default=None,
        help="Path to pass_a_winners.json from Stage 3; overrides OPTIMAL per axis.",
    )
    parser.add_argument(
        "--sweep-outputs-dir", default=None,
        help="Base directory containing seed_*/pass_a_scores.json files from Stage 3.",
    )
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    parser.add_argument("--axes",  nargs="+", default=list(OPTIMAL.keys()))
    parser.add_argument(
        "--abtt-d", nargs="+", type=int, default=ABTT_D_VALUES, dest="abtt_d",
        help="ABTT top-D values to sweep.",
    )
    parser.add_argument(
        "--max-k", type=int, default=MAX_K,
        help="Maximum number of meta-clusters to consider.",
    )
    parser.add_argument(
        "--model", default="gpt-5-mini",
        help="LLM for meta-cluster summarization.",
    )
    parser.add_argument("--n-samples",   type=int, default=15,
                        help="Focal posts to sample per meta-cluster for summarization.")
    parser.add_argument("--n-contrast",  type=int, default=5,
                        help="Contrast posts sampled from other meta-clusters for the prompt.")
    parser.add_argument("--n-summaries", type=int, default=10,
                        help="Candidate summaries to generate per meta-cluster; "
                             "the one closest to their embedding centroid is kept.")
    parser.add_argument("--embed-model", default="text-embedding-3-small",
                        help="OpenAI embedding model used for ensemble summary selection.")
    parser.add_argument("--workers",     type=int, default=8)
    parser.add_argument(
        "--min-seeds-multi", type=int, default=3,
        help="Minimum seeds for a meta-cluster to count as 'multi-seed' in audit.",
    )
    parser.add_argument(
        "--multi-seed-only", action="store_true",
        help="Drop non-multi-seed clusters from outputs; their nodes are marked -1 in merge_map.",
    )
    args = parser.parse_args()

    optimal = OPTIMAL.copy()
    if args.winners_file:
        with open(args.winners_file) as f:
            optimal.update(json.load(f))
    elif args.sweep_outputs_dir:
        print(f"Aggregating HDBSCAN scores across seeds in '{args.sweep_outputs_dir}'...")
        optimal.update(pick_winners_from_sweep(args.sweep_outputs_dir, args.axes))

    print("Loading per-seed corpus texts...")
    seed_texts: dict[str, list[str]] = {}
    for seed in args.seeds:
        sp = (os.path.join(args.splits_dir, f"eval_splits_seed{seed}.json")
              if args.splits_dir else args.splits)
        seed_texts[seed] = load_corpus_texts(args.student_data, sp)
        print(f"  seed {seed}: {len(seed_texts[seed])} train posts")

    lm       = dspy.LM(args.model)
    embed_fn = make_embed_fn(args.embed_model)

    os.makedirs(args.output_dir, exist_ok=True)

    for axis in args.axes:
        hdbscan_key = optimal.get(axis)
        if hdbscan_key is None:
            print(f"\n[skip] {axis}: no HDBSCAN key available")
            continue

        print(f"\n{'='*60}")
        print(f"Axis: {axis}  (hdbscan_key={hdbscan_key})")
        print(f"{'='*60}")

        nodes = load_cluster_centers(axis, hdbscan_key, args.seeds, args.clustering_root)
        if len(nodes) < 4:
            print(f"  [skip] too few clusters ({len(nodes)})")
            continue
        print(f"  Loaded {len(nodes)} cluster centers from {len(args.seeds)} seeds")

        centers = np.stack([n["center"] for n in nodes])  # already L2-normed

        print(f"  Sweeping ABTT D ∈ {args.abtt_d}, k ∈ [2, {args.max_k}]:")
        labels, best_k, best_sil, best_D = sweep_abtt_and_k(
            centers, nodes, args.abtt_d, args.max_k, args.min_seeds_multi, verbose=True,
        )
        n_multi, n_clusters, n_singletons = _cluster_stats(labels, nodes, args.min_seeds_multi)
        print(f"  → Best: D={best_D}  k={best_k}  sil={best_sil:.4f}  "
              f"multi_seed={n_multi}  clusters={n_clusters}  singletons={n_singletons}")

        if args.multi_seed_only:
            meta_seeds: dict[int, set] = defaultdict(set)
            for i, node in enumerate(nodes):
                meta_seeds[int(labels[i])].add(node["seed"])
            valid = {mid for mid, s in meta_seeds.items() if len(s) >= args.min_seeds_multi}
            labels = np.array([l if l in valid else -1 for l in labels.tolist()], dtype=int)
            kept = int((labels >= 0).sum())
            print(f"  multi-seed-only: kept {len(valid)} clusters "
                  f"({kept}/{len(nodes)} nodes assigned, {len(nodes)-kept} marked -1)")

        print(f"  Generating meta-cluster summaries ({args.n_summaries} candidates each)...")
        meta_summaries = generate_meta_summaries(
            nodes, labels, axis, hdbscan_key,
            args.clustering_root, seed_texts, lm, embed_fn,
            args.n_samples, args.n_summaries, args.n_contrast, args.workers,
        )

        merge_map = build_merge_map(nodes, labels, args.seeds)
        audit     = build_audit(nodes, labels, meta_summaries, args.min_seeds_multi)

        print(f"  Meta-clusters: {audit['n_meta_clusters']}  "
              f"singletons: {audit['n_singletons']}  "
              f"multi-seed (≥{args.min_seeds_multi}): {audit['n_multi_seed']}")

        axis_dir = os.path.join(args.output_dir, axis)
        os.makedirs(axis_dir, exist_ok=True)

        with open(os.path.join(axis_dir, "merge_map.json"), "w") as f:
            json.dump(merge_map, f, indent=2)

        merged_text_only = {
            str(mid): info["summary"] for mid, info in meta_summaries.items()
        }
        with open(os.path.join(axis_dir, "merged_summaries.json"), "w") as f:
            json.dump(merged_text_only, f, indent=2)

        with open(os.path.join(axis_dir, "audit.json"), "w") as f:
            json.dump(audit, f, indent=2)

        print(f"  Saved → {axis_dir}/")


if __name__ == "__main__":
    main()
