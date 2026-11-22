"""
Stage 3: Zero-shot reconstruction evaluation.

Two passes, sharing the same eval set:

  Pass A (per-axis HDBSCAN selection):
    For each (axis, candidate HDBSCAN setting), run the simulator conditioned
    on that axis alone.  Rank by reconstruction quality.  The winning setting
    per axis is used in Pass B.  This replaces silhouette-based selection of
    HDBSCAN hyperparameters with task-grounded selection.

  Pass B (axis combinations):
    Enumerate subsets of axes at sizes {1, 2, 3, 4}, each axis using the
    HDBSCAN setting that won Pass A.  Rank configurations by reconstruction
    quality to pick the top candidates for full GEPA optimization later.

Metrics (priority order):
  1. recon_semantic_sim, recon_style_sim - generated vs. ORIGINAL post
  2. cluster-match rate per axis - does the generated post re-cluster to the
     target cluster?
  3. cluster-membership similarity per axis - mean cosine sim with cluster
     members (softer version of #2; matches existing GEPA metric)

Noise handling: noise points are soft-assigned to their most probable cluster
via HDBSCAN's membership_vector, with a nearest-exemplar fallback for truly
isolated points.  Every post is scored under every configuration; `n` equals
the full eval set size.  `soft_assign_rate_{axis}` in the aggregate tracks
what fraction of posts required soft assignment under each axis.
"""

import argparse
import concurrent.futures
import itertools
import json
import os
import pickle
import random

import dotenv
import dspy
import hdbscan.prediction
import numpy as np
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()

random.seed(42)

# ---------------------------------------------------------------------------
# Axes and candidate settings (kept in sync with Stage 2).
# Pass A picks the best setting per axis; Pass B uses those winners.
# ---------------------------------------------------------------------------
_HDBSCAN_GRID_KEYS = [
    f"mcs{mcs}_ms{ms}"
    for mcs, ms in itertools.product([100, 150, 200], [10, 15, 20, 30])
]
CANDIDATE_SETTINGS = {axis: list(_HDBSCAN_GRID_KEYS) for axis in [
    "og_style_model"
]}

OPTIMAL = {
    "og_style_model":   "mcs100_ms15",
}

AXIS_LABELS = {
    "og_general_model": "Semantic Content",
    "og_qa_model":      "Semantic Content / Question Type",
    "og_style_model":   "Writing Style",
    "rr_qa_model":      "Interaction Pattern / Post Structure",
}

EMB_MODEL_KEY = {
    "og_general_model": "general_model",
    "og_qa_model":      "qa_model",
    "og_style_model":   "style_model",
    "rr_qa_model":      "qa_model",
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class SimulatorSignature(dspy.Signature):
    """Generate a student discussion forum post that includes all the key points provided in the `bullet_summary`. Ensure that the post closely mirrors the writing style, tone, and format specified by the `requirements`. Additionally, follow the structural guidelines detailed in the `post_template` to ensure the rewritten post is well-organized and meets the specified criteria.\n\n1. **Content Coverage**: Make sure that every point listed in the `bullet_summary` is addressed in the rewritten post.\n2. **Style and Tone**: Adopt the same writing style, tone, and format as the described by the `requirements`. This includes descriptions of sentence structure, vocabulary, and any specific formatting elements.\n3. **Structure**: Follow the structure outlined in the `post_template`. This may include sections, paragraphs, headings, and other organizational elements that are important for the post's clarity and readability.\n\nBy adhering to these guidelines, the rewritten post should effectively communicate the required content while maintaining a consistent and appropriate style and structure."""
    bullet_summary: str = dspy.InputField(
        desc="Bullet points describing the content the post should cover.")
    post_template: str = dspy.InputField(
        desc="Description of the general structure of what the rewritten post should look like."
    )
    requirements: str = dspy.InputField(
        desc="Stylistic and structural requirements the post must satisfy.")
    output_post: str = dspy.OutputField(
        desc="The generated student forum post.")


class Simulator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SimulatorSignature)

    def forward(self, bullet_summary, post_template, requirements):
        return self.predict(bullet_summary=bullet_summary,
                            post_template=post_template,
                            requirements=requirements).output_post


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


def build_requirements(axis_settings, axis_to_cluster, axis_was_noise, summaries, rng):
    """Assemble the conditioning string by concatenating cluster summaries,
    in a randomized order to defuse positional bias.

    Every post gets a real cluster in every axis (noise points are soft-assigned
    upstream in build_examples), so this never returns None.
    """
    pieces = []
    for axis, hdbscan_key in axis_settings.items():
        cid = axis_to_cluster[axis]
        summary = summaries[axis][hdbscan_key].get(str(cid),
                                                    "(no summary available for assigned cluster)")
        pieces.append(f"{AXIS_LABELS[axis]}: {summary}")
    rng.shuffle(pieces)
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def cluster_predict(emb, reducer, clusterer):
    """Project embedding through UMAP and predict its HDBSCAN cluster."""
    umap_emb = reducer.transform(emb.reshape(1, -1))
    labels, _ = hdbscan.prediction.approximate_predict(clusterer, umap_emb)
    return int(labels[0])


def eval_one(example, axis_settings, simulator, summaries, resources, rng_seed):
    rng = random.Random(rng_seed)
    axis_to_cluster = {
        axis: example[f"{axis}__{hdbscan_key}_assigned_label"]
        for axis, hdbscan_key in axis_settings.items()
    }
    axis_was_noise = {
        axis: example[f"{axis}__{hdbscan_key}_was_noise"]
        for axis, hdbscan_key in axis_settings.items()
    }

    requirements = build_requirements(
        axis_settings, axis_to_cluster, axis_was_noise, summaries, rng,
    )

    try:
        generated = simulator(bullet_summary=example["bullet_summary"],
                              post_template=example["post_template"],
                              requirements=requirements)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] idx={example['index']} generation failed: {exc}")
        return None

    # Reconstruction sim vs. ORIGINAL post (precomputed)
    gen_sem = resources["general_model"].encode(generated)
    gen_style = resources["style_model"].encode(generated)
    og_sem = resources["original_embs_sem"][example["row_pos"]]
    og_style = resources["original_embs_style"][example["row_pos"]]
    recon_sem = float(cosine_similarity([gen_sem], [og_sem])[0, 0])
    recon_style = float(cosine_similarity([gen_style], [og_style])[0, 0])

    # Per-axis cluster-match + membership similarity.
    # target_cid is always >= 0 (soft-assigned if originally noise).
    per_axis = {}
    for axis, hdbscan_key in axis_settings.items():
        target_cid = axis_to_cluster[axis]
        reducer = resources["reducers"][axis]
        clusterer = resources["clusterers"][(axis, hdbscan_key)]
        gen_emb = resources[EMB_MODEL_KEY[axis]].encode(generated)
        pred_cid = cluster_predict(gen_emb, reducer, clusterer)
        member_embs = resources["cluster_member_embs"][(axis, hdbscan_key)].get(target_cid)
        mem_sim = float("nan")
        if member_embs is not None and len(member_embs) > 0:
            mem_sim = float(cosine_similarity([gen_emb], member_embs).mean())
        per_axis[axis] = {
            "hdbscan_key": hdbscan_key,
            "target_cluster": int(target_cid),
            "predicted_cluster": pred_cid,
            "match": bool(pred_cid == target_cid),
            "membership_similarity": mem_sim,
            "was_soft_assigned": bool(axis_was_noise[axis]),
        }

    return {
        "index": example["index"],
        "recon_semantic_sim": recon_sem,
        "recon_style_sim": recon_style,
        "per_axis": per_axis,
        "generated": generated,
    }


def summarize_config_scores(results, axes):
    if not results:
        return {"n": 0}

    agg = {
        "n": len(results),
        "recon_sem_mean": float(np.mean([r["recon_semantic_sim"] for r in results])),
        "recon_style_mean": float(np.mean([r["recon_style_sim"] for r in results])),
    }
    for axis in axes:
        matches = [r["per_axis"][axis]["match"] for r in results]
        memsims = [r["per_axis"][axis]["membership_similarity"] for r in results
                   if not np.isnan(r["per_axis"][axis]["membership_similarity"])]
        n_soft = sum(1 for r in results if r["per_axis"][axis]["was_soft_assigned"])
        agg[f"match_{axis}_mean"] = float(np.mean(matches))
        agg[f"memsim_{axis}_mean"] = float(np.mean(memsims)) if memsims else float("nan")
        agg[f"soft_assign_rate_{axis}"] = n_soft / len(results)
    return agg


def run_configuration(config_name, axis_settings, examples, simulator,
                      summaries, resources, workers, output_path, all_results):
    """Run the simulator across the eval set for one configuration."""
    per_example = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ex = {
            executor.submit(
                eval_one, ex, axis_settings, simulator, summaries, resources,
                rng_seed=hash((config_name, ex["index"])) & 0xffffffff,
            ): ex
            for ex in examples
        }
        for fut in tqdm.tqdm(concurrent.futures.as_completed(future_to_ex),
                              total=len(future_to_ex), desc=f"  {config_name}"):
            r = fut.result()
            if r is not None:
                per_example.append(r)

    axes = list(axis_settings.keys())
    agg = summarize_config_scores(per_example, axes)
    print(f"  n={agg.get('n')}  recon_sem={agg.get('recon_sem_mean', 0):.3f}  "
          f"recon_style={agg.get('recon_style_mean', 0):.3f}")

    all_results[config_name] = {
        "axis_settings": axis_settings,
        "aggregate": agg,
        "per_example": per_example,
    }
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return agg


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
def load_resources(clustering_root, axes, active_settings, use_meta=False, seed=None, meta_output_dir=None):
    print("Loading embedding models...")
    general_model = SentenceTransformer("all-mpnet-base-v2")
    qa_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    style_model = SentenceTransformer("StyleDistance/styledistance")

    reducers, clusterers, summaries = {}, {}, {}
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

        with open(os.path.join(config_dir, "cluster_summaries.json")) as f:
            summaries[axis] = json.load(f)

        if use_meta:
            meta_dir = os.path.join(meta_output_dir, axis)
            merge_map = json.load(open(os.path.join(meta_dir, "merge_map.json")))[str(seed)]
            merged_summs = json.load(open(os.path.join(meta_dir, "merged_summaries.json")))
            for hdbscan_key in summaries[axis]:
                summaries[axis][hdbscan_key] = {
                    cid: merged_summs[str(merge_map[cid])]
                    for cid in summaries[axis][hdbscan_key]
                    if cid in merge_map
                }

    return {
        "general_model": general_model,
        "qa_model": qa_model,
        "style_model": style_model,
        "reducers": reducers,
        "clusterers": clusterers,
        "cluster_member_embs": cluster_member_embs,
        "stored_embeddings": stored_embeddings,
        "stored_labels": stored_labels,
    }, summaries


def build_examples(eval_df, axes, resources, post_text_to_npy_idx, active_settings):
    """One record per eval post with soft-assigned cluster labels for every
    (axis, setting) combination.

    Noise points (hard label -1) are soft-assigned to their most probable
    cluster via membership_vector, with a nearest-exemplar fallback.
    post_text_to_npy_idx maps original_post text -> row index in the corpus
    arrays (matches the train-split ordering from 02_clustering_sweep.py).
    """
    summary_col = "summary" if "summary" in eval_df.columns else "bullets"

    examples = []
    print("Resolving cluster assignments (soft-assigning noise points)...")
    for pos, (row_idx, row) in enumerate(tqdm.tqdm(eval_df.iterrows(), total=len(eval_df))):
        post_text = str(row["original_post"])
        corpus_idx = post_text_to_npy_idx.get(post_text)

        ex = {
            "index": int(row_idx),
            "row_pos": pos,
            "bullet_summary": row[summary_col],
            "post_template": row["rewrites"],
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
# Helpers
# ---------------------------------------------------------------------------
def _derive_winners_from_sweep_scores(sweep_outputs_dir: str, axes: list) -> dict:
    """Average pass_a_scores.json across all seed dirs and return best setting per axis."""
    import glob
    score_files = glob.glob(os.path.join(sweep_outputs_dir, "seed_*", "pass_a_scores.json"))
    if not score_files:
        raise FileNotFoundError(f"No pass_a_scores.json files found under {sweep_outputs_dir}/seed_*/")

    accumulated: dict = {}  # axis -> {setting -> [scores]}
    for path in score_files:
        with open(path) as f:
            data = json.load(f)
        for axis, scores in data.items():
            if axis not in accumulated:
                accumulated[axis] = {}
            for setting, score in scores.items():
                accumulated[axis].setdefault(setting, []).append(score)

    winner_per_axis = {}
    for axis in axes:
        if axis not in accumulated:
            raise KeyError(f"Axis '{axis}' not found in any pass_a_scores.json under {sweep_outputs_dir}")
        avg_scores = {s: sum(v) / len(v) for s, v in accumulated[axis].items()}
        best = max(avg_scores, key=lambda k: avg_scores[k])
        winner_per_axis[axis] = best
        print(f"  [skip-pass-a] {axis} winner: {best} (avg memsim={avg_scores[best]:.4f})")
    return winner_per_axis


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
    parser.add_argument("--model", default="openai/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--api-base", default="http://localhost:8001/v1",
                        help="OpenAI-compatible API base URL (e.g. http://localhost:8001/v1 for vLLM)")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--max-posts", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--axes", nargs="+", default=list(CANDIDATE_SETTINGS.keys()))
    parser.add_argument("--combo-sizes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--pass-a-max-posts", type=int, default=100,
                        help="Eval set cap for Pass A (HDBSCAN-setting selection). "
                             "Smaller than Pass B because Pass A just picks a winner.")
    parser.add_argument("--skip-pass-a", action="store_true",
                        help="Derive Pass A winners from sweep_outputs/seed*/pass_a_scores.json "
                             "(average across seeds) instead of running the sweep.")
    parser.add_argument("--sweep-outputs-dir", default=None,
                        help="Directory containing seed*/pass_a_scores.json files used when "
                             "--skip-pass-a is set. Required when --skip-pass-a is used.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed identifying this pipeline run (used to locate "
                             "meta-cluster summaries in meta_clustering_output/).")
    parser.add_argument("--use-meta", action="store_true")
    parser.add_argument("--meta-output-dir", default=None,
                        help="Directory containing meta_clustering_output/<axis>/ subdirs. "
                             "Defaults to <clustering-root>/../meta_clustering_output.")
    args = parser.parse_args()

    if args.use_meta and args.seed is None:
        parser.error("--seed is required when --use-meta is set")

    if args.skip_pass_a and args.sweep_outputs_dir is None:
        parser.error("--sweep-outputs-dir is required when --skip-pass-a is set")

    if args.use_meta:
        args.skip_pass_a = True

    os.makedirs(args.output_dir, exist_ok=True)

    winners_path = os.path.join(args.output_dir, "pass_a_winners.json")

    # Determine which HDBSCAN settings to load up front.
    # When skipping Pass A we only need the one winner per axis.
    if args.use_meta:
        active_settings = {axis: [OPTIMAL[axis]] for axis in args.axes}
    elif args.skip_pass_a:
        winner_per_axis = _derive_winners_from_sweep_scores(
            args.sweep_outputs_dir, args.axes
        )
        with open(winners_path, "w") as f:
            json.dump(winner_per_axis, f, indent=2)
        active_settings = {axis: [winner_per_axis[axis]] for axis in args.axes}
    else:
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

    # --------------------------------------------------------------
    # Load resources + precompute original-post embeddings once
    # --------------------------------------------------------------
    meta_output_dir = args.meta_output_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.clustering_root)), "meta_clustering_output"
    )
    resources, summaries = load_resources(args.clustering_root, args.axes, active_settings,
                                           use_meta=args.use_meta, seed=args.seed,
                                           meta_output_dir=meta_output_dir)

    print("Precomputing eval-set embeddings...")
    eval_posts = list(eval_df_full["original_post"])
    resources["original_embs_sem"] = np.array(
        resources["general_model"].encode(eval_posts, batch_size=64))
    resources["original_embs_style"] = np.array(
        resources["style_model"].encode(eval_posts, batch_size=64))

    print("Building post -> corpus index mapping...")
    # Mirror the train-only ordering used in 02_clustering_sweep.py so that
    # indices into the .npy arrays align with the CSV's train split order.
    train_posts = (
        student_data.loc[splits["train"], "original_post"]
        .astype(str)
        .tolist()
    )
    post_text_to_npy_idx = {text: i for i, text in enumerate(train_posts)}

    examples_full = build_examples(
        eval_df_full, args.axes, resources, post_text_to_npy_idx, active_settings,
    )

    # --------------------------------------------------------------
    # Simulator
    # --------------------------------------------------------------
    lm_kwargs = {"api_base": args.api_base, "api_key": "none"} if args.api_base else {}
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))
    simulator = Simulator()

    # ==============================================================
    # Pass A: per-axis HDBSCAN selection
    # ==============================================================
    if not args.skip_pass_a:
        print("\n" + "=" * 70)
        print("PASS A: per-axis HDBSCAN-setting selection")
        print("=" * 70)
        pass_a_results = {}
        pass_a_path = os.path.join(args.output_dir, "pass_a_results.json")
        pass_a_examples = examples_full[:args.pass_a_max_posts]
        winner_per_axis = {}
        for axis in args.axes:
            best_setting, best_score = None, -float("inf")
            for hdbscan_key in active_settings[axis]:
                config_name = f"A::{axis}::{hdbscan_key}"
                axis_settings = {axis: hdbscan_key}
                agg = run_configuration(
                    config_name, axis_settings, pass_a_examples, simulator,
                    summaries, resources, args.workers, pass_a_path, pass_a_results,
                )
                score = agg.get("recon_sem_mean", 0.0)
                if score > best_score:
                    best_score, best_setting = score, hdbscan_key
            winner_per_axis[axis] = best_setting
            print(f"  -> {axis} winner: {best_setting} (recon_sem={best_score:.3f})")
        with open(winners_path, "w") as f:
            json.dump(winner_per_axis, f, indent=2)
        print(f"Pass A winners saved -> {winners_path}")

    # ==============================================================
    # Pass B: axis-subset combinations at winning settings
    # ==============================================================
    print("\n" + "=" * 70)
    print("PASS B: axis-combination sweep")
    print("=" * 70)
    pass_b_results = {}
    pass_b_path = os.path.join(args.output_dir, "pass_b_results.json")
    for size in args.combo_sizes:
        for subset in itertools.combinations(args.axes, size):
            config_name = "B::" + "+".join(subset)
            if "og_style" not in config_name:
                continue
            axis_settings = {axis: (OPTIMAL[axis] if args.use_meta else winner_per_axis[axis])
                             for axis in subset}
            run_configuration(
                config_name, axis_settings, examples_full, simulator,
                summaries, resources, args.workers, pass_b_path, pass_b_results,
            )

    ranking = sorted(
        pass_b_results.items(),
        key=lambda kv: kv[1]["aggregate"].get("recon_sem_mean", 0.0),
        reverse=True,
    )
    print("\n=== Top 10 Pass-B configurations (by recon_sem) ===")
    for name, bundle in ranking[:10]:
        agg = bundle["aggregate"]
        print(f"  {name:50s}  sem={agg.get('recon_sem_mean', 0):.3f}  "
            f"style={agg.get('recon_style_mean', 0):.3f}  n={agg.get('n')}")


if __name__ == "__main__":
    main()