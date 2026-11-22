"""
Stage 3: HDBSCAN hyperparameter selection via membership similarity.

Pass A (per-axis HDBSCAN selection):
  For each (axis, candidate HDBSCAN setting), compute the average membership
  similarity of validation-set posts against their assigned cluster members.
  The setting with the highest average membership similarity wins and is saved
  to pass_a_winners.json for use in downstream stages.

  No generation is performed; membership similarity is computed directly from
  original post embeddings against the training-set cluster member embeddings.

Noise handling: noise points are soft-assigned to their most probable cluster
via HDBSCAN's membership_vector, with a nearest-exemplar fallback for truly
isolated points.
"""

import argparse
import itertools
import json
import os
import pickle

import dotenv
import hdbscan.prediction
import numpy as np
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Axes and candidate settings (kept in sync with Stage 2).
# ---------------------------------------------------------------------------
_HDBSCAN_GRID_KEYS = [
    f"mcs{mcs}_ms{ms}"
    for mcs, ms in itertools.product([100, 150, 200], [10, 15, 20, 30])
]
CANDIDATE_SETTINGS = {axis: list(_HDBSCAN_GRID_KEYS) for axis in [
    "og_style_model",
]}


EMB_MODEL_KEY = {
    "og_style_model":   "style_model",
}

# Maps axis -> precomputed original-post embedding key in resources dict
EMB_RESOURCE_KEY = {
    "og_style_model":   "original_embs_style"
}


def soft_assign_cluster(emb, reducer, clusterer):
    """For a noise point, find the most probable cluster via soft membership.

    Returns (assigned_cluster_id, was_originally_noise).
    Non-noise points are returned as-is with was_originally_noise=False.
    """
    umap_emb = reducer.transform(emb.reshape(1, -1))
    hard_labels, _ = hdbscan.prediction.approximate_predict(clusterer, umap_emb)
    hard_label = int(hard_labels[0])
    if hard_label >= 0:
        return hard_label, False

    membership = hdbscan.prediction.membership_vector(clusterer, umap_emb)[0]
    if membership.sum() > 0:
        return int(np.argmax(membership)), True

    # Truly isolated — no cluster gives nonzero membership.
    # Fall back to nearest cluster by exemplar centroid distance.
    best_cid, best_dist = -1, float("inf")
    for cid, ex_points in enumerate(clusterer.exemplars_):
        if len(ex_points) == 0:
            continue
        d = np.linalg.norm(ex_points - umap_emb[0], axis=1).min()
        if d < best_dist:
            best_dist, best_cid = d, cid
    return best_cid, True


def score_membership_sim(axis, hdbscan_key, examples, resources):
    """Average cosine similarity of each eval post against its assigned cluster members."""
    emb_key = EMB_RESOURCE_KEY[axis]
    sims = []
    for ex in examples:
        post_emb = resources[emb_key][ex["row_pos"]]
        cid = ex[f"{axis}__{hdbscan_key}_assigned_label"]
        member_embs = resources["cluster_member_embs"][(axis, hdbscan_key)].get(cid)
        if member_embs is not None and len(member_embs) > 0:
            sims.append(float(cosine_similarity([post_emb], member_embs).mean()))
    return float(np.mean(sims)) if sims else float("nan")


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
def load_resources(clustering_root, axes, active_settings):
    print("Loading embedding models...")
    style_model = SentenceTransformer("StyleDistance/styledistance")

    reducers, clusterers = {}, {}
    cluster_member_embs, stored_embeddings, stored_labels = {}, {}, {}
    print("Loading clustering artifacts + summaries...")
    for axis in axes:
        config_dir = os.path.join(clustering_root, axis)
        with open(os.path.join(config_dir, "umap_reducer.pkl"), "rb") as f:
            reducers[axis] = pickle.load(f)
        embs = np.load(os.path.join(config_dir, "embeddings.npy"))
        stored_embeddings[axis] = embs
        for hdbscan_key in active_settings[axis]:
            clusterer_path = os.path.join(config_dir, f"clusterer_{hdbscan_key}.pkl")
            if not os.path.exists(clusterer_path):
                print(f"  [warn] {axis}/{hdbscan_key}: clusterer missing")
                continue
            with open(clusterer_path, "rb") as f:
                clusterers[(axis, hdbscan_key)] = pickle.load(f)
            labels = np.load(os.path.join(config_dir, f"labels_{hdbscan_key}.npy"))
            stored_labels[(axis, hdbscan_key)] = labels
            member_map = {}
            for cid in set(labels[labels >= 0]):
                member_map[int(cid)] = embs[labels == cid]
            cluster_member_embs[(axis, hdbscan_key)] = member_map


    return {
        "style_model": style_model,
        "reducers": reducers,
        "clusterers": clusterers,
        "cluster_member_embs": cluster_member_embs,
        "stored_embeddings": stored_embeddings,
        "stored_labels": stored_labels,
    }


def build_examples(eval_df, axes, resources, post_text_to_npy_idx, active_settings):
    """One record per eval post with soft-assigned cluster labels for every
    (axis, setting) combination.

    Noise points (hard label -1) are soft-assigned to their most probable
    cluster via membership_vector, with a nearest-exemplar fallback.
    post_text_to_npy_idx maps original_post text -> row index in the corpus
    arrays (built from post_rewrite_dict insertion order in clustering).
    """

    examples = []
    print("Resolving cluster assignments (soft-assigning noise points)...")
    for pos, (row_idx, row) in enumerate(tqdm.tqdm(eval_df.iterrows(), total=len(eval_df))):
        post_text = str(row["original_post"])
        corpus_idx = post_text_to_npy_idx.get(post_text)

        ex = {
            "index": int(row_idx),
            "row_pos": pos
        }
        # Cache per-axis live embeddings for eval posts (not in corpus).
        # Computed once per axis across all hdbscan_keys for that axis.
        live_emb_cache: dict = {}

        for axis in axes:
            reducer = resources["reducers"][axis]
            if corpus_idx is not None:
                post_emb = resources["stored_embeddings"][axis][corpus_idx]
            else:
                if axis not in live_emb_cache:
                    model = resources[EMB_MODEL_KEY[axis]]
                    live_emb_cache[axis] = model.encode(post_text)
                post_emb = live_emb_cache[axis]

            for hdbscan_key in active_settings[axis]:
                key = (axis, hdbscan_key)
                if key not in resources["clusterers"]:
                    continue
                if corpus_idx is not None:
                    hard_label = int(resources["stored_labels"][key][corpus_idx])
                    if hard_label >= 0:
                        assigned, was_noise = hard_label, False
                    else:
                        assigned, was_noise = soft_assign_cluster(
                            post_emb, reducer, resources["clusterers"][key],
                        )
                else:
                    # Eval post: never in the train corpus, always soft-assigned.
                    assigned, was_noise = soft_assign_cluster(
                        post_emb, reducer, resources["clusterers"][key],
                    )
                ex[f"{axis}__{hdbscan_key}_assigned_label"] = int(assigned)
                ex[f"{axis}__{hdbscan_key}_was_noise"] = bool(was_noise)
        examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-data",
                        default="final_data/stack_exchange_v2.csv")
    parser.add_argument("--splits", default="final_data/stack_exchange/eval_splits_seed0.json")
    parser.add_argument("--clustering-root",
                        default="final_data/stack_exchange/hdbscan_results/")
    parser.add_argument("--output-dir", default="sweep_output")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--max-posts", type=int, default=None)
    parser.add_argument("--axes", nargs="+", default=list(CANDIDATE_SETTINGS.keys()))
    parser.add_argument("--pass-a-max-posts", type=int, default=200,
                        help="Eval set cap for Pass A (HDBSCAN-setting selection).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    winners_path = os.path.join(args.output_dir, "pass_a_winners.json")

    # Determine which HDBSCAN settings to load up front.
    # When skipping Pass A we only need the one winner per axis.
    active_settings = {axis: list(CANDIDATE_SETTINGS[axis]) for axis in args.axes}

    # --------------------------------------------------------------
    # Load data and splits
    # --------------------------------------------------------------
    student_data = pd.read_csv(args.student_data).reset_index(drop=True)
    with open(args.splits) as f:
        splits = json.load(f)
    eval_indices = splits[args.eval_split]
    if args.max_posts:
        eval_indices = eval_indices[:args.max_posts]
    eval_df_full = student_data.loc[eval_indices]
    print(f"Loaded {len(eval_df_full)} posts from '{args.eval_split}' split")

    # Limit to pass_a_max_posts up front — no need to embed or assign the rest.
    pass_a_df = eval_df_full.iloc[:args.pass_a_max_posts] if args.pass_a_max_posts else eval_df_full
    print(f"Pass A will use {len(pass_a_df)} posts")

    # --------------------------------------------------------------
    # Load resources + precompute original-post embeddings once
    # --------------------------------------------------------------
    resources = load_resources(args.clustering_root, args.axes, active_settings)

    print("Precomputing eval-set embeddings...")
    eval_posts = list(pass_a_df["original_post"])
    resources["original_embs_style"] = np.array(
        resources["style_model"].encode(eval_posts, batch_size=64))

    print("Building post -> corpus index mapping...")
    # Mirror the same train-only filter used in 02_clustering_sweep.py so that
    # indices into post_rewrite_dict align with the .npy arrays.
    train_texts = set(
        student_data.loc[splits["train"], "original_post"].astype(str).tolist()
    )
    post_text_to_npy_idx = {text: i for i, text in enumerate(train_texts)}

    # Cache soft-assignments keyed by clustering root, splits, settings, and n_posts,
    # so repeated runs skip the slow UMAP transform + membership_vector loop.
    cache_path = os.path.join(args.output_dir, "pass_a_examples_cache.pkl")
    cache_meta = {
        "clustering_root": args.clustering_root,
        "splits": args.splits,
        "eval_split": args.eval_split,
        "active_settings": {k: sorted(v) for k, v in active_settings.items()},
        "n_posts": len(pass_a_df),
    }
    pass_a_examples = None
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("meta") == cache_meta:
            pass_a_examples = cached["examples"]
            print(f"Loaded {len(pass_a_examples)} cached cluster assignments from {cache_path}")
    if pass_a_examples is None:
        pass_a_examples = build_examples(
            pass_a_df, args.axes, resources, post_text_to_npy_idx, active_settings,
        )
        with open(cache_path, "wb") as f:
            pickle.dump({"meta": cache_meta, "examples": pass_a_examples}, f)
        print(f"Cached cluster assignments -> {cache_path}")

    # ==============================================================
    # Pass A: per-axis HDBSCAN selection by membership similarity
    # ==============================================================
    print("\n" + "=" * 70)
    print("PASS A: per-axis HDBSCAN-setting selection (membership similarity)")
    print("=" * 70)
    pass_a_scores = {}
    winner_per_axis = {}
    for axis in args.axes:
        scores = {}
        for hdbscan_key in active_settings[axis]:
            if (axis, hdbscan_key) not in resources["clusterers"]:
                continue
            s = score_membership_sim(axis, hdbscan_key, pass_a_examples, resources)
            scores[hdbscan_key] = s
            print(f"  {axis}/{hdbscan_key}: memsim={s:.4f}")
        if not scores:
            continue
        best = max(scores, key=lambda k: scores[k])
        winner_per_axis[axis] = best
        pass_a_scores[axis] = scores
        print(f"  -> {axis} winner: {best} (memsim={scores[best]:.4f})")
    with open(winners_path, "w") as f:
        json.dump(winner_per_axis, f, indent=2)
    scores_path = os.path.join(args.output_dir, "pass_a_scores.json")
    with open(scores_path, "w") as f:
        json.dump(pass_a_scores, f, indent=2)
    print(f"Pass A winners saved -> {winners_path}")


if __name__ == "__main__":
    main()