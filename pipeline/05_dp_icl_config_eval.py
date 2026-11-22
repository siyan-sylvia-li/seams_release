"""
DP-ICL-RAG: RAG evaluation with DP-ICL-style generation.

Merges 04_rag_config_eval.py with 04_dp_icl_rewrites.py.  The evaluation
protocol (Pass A HDBSCAN sweep + Pass B axis-combination sweep) is identical
to the RAG variant.  The generation step is replaced with the DP-ICL
mechanism:

  1. Retrieve top-K within-cluster neighbors from the rewriting axis.
  2. Student-disjoint partition into M groups.
  3. M parallel LM calls — one per partition.
  4. Clip candidate embeddings to clip_norm, take mean, add Gaussian noise.
  5. Return the candidate nearest the noisy mean.

The rewriting axis used for retrieval is selected from axis_settings as
follows: dp_config.rewriting_axis if present in the config, otherwise the
first axis.  Evaluation metrics still cover every axis in the config.

New CLI flags vs. 04_rag_config_eval.py:
  --dp-m, --dp-k, --dp-clip-norm, --dp-noise-scale,
  --dp-embed-model, --rewriting-axis, --student-id-col
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import itertools
import json
import os
import pickle
import random
from dataclasses import dataclass, replace as dataclass_replace

import dotenv
import dspy
import hdbscan.prediction
import numpy as np
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer as rouge_scorer_lib

dotenv.load_dotenv()


# ---------------------------------------------------------------------------
# Axes and candidate settings (kept in sync with Stage 2)
# ---------------------------------------------------------------------------
_HDBSCAN_GRID_KEYS = [
    f"mcs{mcs}_ms{ms}"
    for mcs, ms in itertools.product([100, 150, 200], [10, 15, 20, 30])
]
CANDIDATE_SETTINGS = {axis: list(_HDBSCAN_GRID_KEYS) for axis in [
    "og_general_model", "og_qa_model", "og_style_model", "rr_qa_model",
]}

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

DEFAULT_TOP_K = 20
DEFAULT_REWRITING_AXIS = "og_style_model"

OPTIMAL = {
    "og_general_model": "mcs200_ms10",
    "og_qa_model":      "mcs200_ms15",
    "og_style_model":   "mcs100_ms15",
    "rr_qa_model":      "mcs100_ms30",
}

ROUGE_SCORER   = rouge_scorer_lib.RougeScorer(["rouge1"], use_stemmer=True)



# ---------------------------------------------------------------------------
# DP-ICL configuration
# ---------------------------------------------------------------------------
@dataclass
class DPICLConfig:
    K: int = DEFAULT_TOP_K   # neighbors retrieved from the rewriting axis
    M: int = 5               # disjoint partitions / candidate rewrites
    clip_norm: float = 1.0   # L2 clip applied to candidate embeddings
    noise_scale: float = 0.1 # Gaussian noise stddev (relative to clip_norm/M)
    rewriting_axis: str = DEFAULT_REWRITING_AXIS
    embed_model_key: str = "style_model"


# ---------------------------------------------------------------------------
# DSPy rewriter module
# ---------------------------------------------------------------------------
class RewriteSignature(dspy.Signature):
    """Generate a student discussion forum post that includes all the key points provided in the `bullet_summary`. Ensure that the post closely mirrors the writing style, tone, and format of the `style_context` exemplar posts. Additionally, follow the structural guidelines detailed in the `post_template` to ensure the rewritten post is well-organized and meets the specified criteria.\n\n1. **Content Coverage**: Make sure that every point listed in the `bullet_summary` is addressed in the rewritten post.\n2. **Style and Tone**: Adopt the same writing style, tone, and format as the exemplar posts provided in `style_context`. This includes sentence structure, vocabulary, and any specific formatting elements.\n3. **Structure**: Follow the structure outlined in the `post_template`. This may include sections, paragraphs, headings, and other organizational elements that are important for the post's clarity and readability.\n\nBy adhering to these guidelines, the rewritten post should effectively communicate the required content while maintaining a consistent and appropriate style and structure."""

    bullet_summary: str = dspy.InputField(
        desc="Bullet points that the rewritten post must cover.")
    post_template: str = dspy.InputField(
        desc="Description of the general structure of what the rewritten post should look like."
    )
    style_context: str = dspy.InputField(
        desc="Numbered peer posts demonstrating the target register.")
    rewritten_post: str = dspy.OutputField(
        desc="The rewritten post, content-preserving and register-matched, "
             "with all identifying surface details replaced.")


class Rewriter(dspy.Module):
    """Single rewrite call per partition.  Uses Predict (not ChainOfThought)
    to keep candidates length-comparable for cleaner embedding aggregation."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(RewriteSignature)

    def forward(self, bullet_summary: str, post_template: str, style_context: str) -> str:
        return self.predict(
            bullet_summary=bullet_summary,
            post_template=post_template,
            style_context=style_context,
        ).rewritten_post


# ---------------------------------------------------------------------------
# Cluster-assignment helpers
# ---------------------------------------------------------------------------
def soft_assign_cluster(emb, reducer, clusterer):
    umap_emb = reducer.transform(emb.reshape(1, -1))
    hard_labels, _ = hdbscan.prediction.approximate_predict(clusterer, umap_emb)
    hard_label = int(hard_labels[0])
    if hard_label >= 0:
        return hard_label, False

    membership = hdbscan.prediction.membership_vector(clusterer, umap_emb)[0]
    if membership.sum() > 0:
        return int(np.argmax(membership)), True

    best_cid, best_dist = -1, float("inf")
    for cid, ex_points in enumerate(clusterer.exemplars_):
        if len(ex_points) == 0:
            continue
        d = np.linalg.norm(ex_points - umap_emb[0], axis=1).min()
        if d < best_dist:
            best_dist, best_cid = d, cid
    return best_cid, True


def cluster_predict(emb, reducer, clusterer):
    umap_emb = reducer.transform(emb.reshape(1, -1))
    labels, _ = hdbscan.prediction.approximate_predict(clusterer, umap_emb)
    return int(labels[0])


# ---------------------------------------------------------------------------
# DP-ICL core: retrieval, partitioning, generation, aggregation
# ---------------------------------------------------------------------------
def retrieve_top_K(
    query_emb: np.ndarray,
    member_embs: np.ndarray,
    member_texts: list[str],
    member_student_ids: list[str],
    K: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Return the top-K cluster members most similar to query_emb.

    Ties broken by deterministic shuffle for reproducibility.
    """
    if len(member_embs) == 0:
        return [], []

    sims = cosine_similarity([query_emb], member_embs)[0]
    order = list(range(len(sims)))
    rng.shuffle(order)
    order.sort(key=lambda i: sims[i], reverse=True)
    chosen = order[:K]

    return ([member_texts[i] for i in chosen],
            [member_student_ids[i] for i in chosen])


def partition_disjoint_by_student(
    neighbor_texts: list[str],
    neighbor_student_ids: list[str],
    M: int,
) -> list[list[str]]:
    """Partition neighbors into M subsets so no student appears in more than
    one subset.

    Greedy round-robin by student group: bounds per-neighbor influence on the
    aggregate to 1/M candidates.  Empty partitions (when M > distinct students)
    are dropped.
    """
    by_student: dict[str, list[str]] = {}
    for text, sid in zip(neighbor_texts, neighbor_student_ids):
        by_student.setdefault(sid, []).append(text)

    student_groups = sorted(by_student.values(), key=len, reverse=True)
    partitions: list[list[str]] = [[] for _ in range(M)]
    for group in student_groups:
        smallest = min(range(M), key=lambda i: len(partitions[i]))
        partitions[smallest].extend(group)

    return [p for p in partitions if p]


def _format_style_context(partition: list[str]) -> str:
    return "\n\n".join(f"Example {i+1}:\n{text}" for i, text in enumerate(partition))


def generate_candidates(
    bullets: str,
    post_template: str,
    partitions: list[list[str]],
    rewriter: Rewriter,
) -> list[str]:
    candidates = []
    for part in partitions:
        ctx = _format_style_context(part)
        try:
            cand = rewriter(bullet_summary=bullets, post_template=post_template, style_context=ctx)
            candidates.append(cand)
        except Exception as exc:
            print(f"[WARN] candidate generation failed: {exc}")
    return candidates


def aggregate_with_noise(
    candidates: list[str],
    embed_fn,
    config: DPICLConfig,
    np_rng: np.random.Generator,
) -> np.ndarray:
    """Clip candidate embeddings, average, add Gaussian noise.

    Clip-then-average bounds each candidate's influence on the aggregate to
    clip_norm/M.  noise_scale=0 disables noise (aggregation-only baseline).
    """
    if not candidates:
        raise ValueError("No candidates to aggregate")

    embs = np.stack([embed_fn(c) for c in candidates])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    scale = np.minimum(1.0, config.clip_norm / (norms + 1e-12))
    clipped = embs * scale

    M = len(candidates)
    mean = clipped.mean(axis=0)

    if config.noise_scale > 0:
        noise_std = config.noise_scale * config.clip_norm / M
        return mean + np_rng.normal(0, noise_std, size=mean.shape)
    return mean


def select_nearest_candidate(
    candidates: list[str],
    aggregate: np.ndarray,
    embed_fn,
    config: DPICLConfig,
) -> str:
    cand_embs = np.stack([embed_fn(c) for c in candidates])
    norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    scale = np.minimum(1.0, config.clip_norm / (norms + 1e-12))
    clipped = cand_embs * scale

    dists = np.linalg.norm(clipped - aggregate, axis=1)
    return candidates[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _pick_rewriting_axis(axis_settings: dict, dp_config: DPICLConfig) -> str:
    """Return the axis to use for DP-ICL retrieval.

    Prefers dp_config.rewriting_axis when present in axis_settings; falls
    back to the first axis so single-axis Pass-A configs work naturally.
    """
    if dp_config.rewriting_axis in axis_settings:
        return dp_config.rewriting_axis
    return next(iter(axis_settings))


def eval_one_dp_icl(example, axis_settings, rewriter, resources, rng_seed,
                    dp_config: DPICLConfig, use_meta: bool = False):
    rng    = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    axis_to_cluster = {
        axis: example[f"{axis}__{hdbscan_key}_assigned_label"]
        for axis, hdbscan_key in axis_settings.items()
    }

    rewriting_axis = _pick_rewriting_axis(axis_settings, dp_config)
    hdbscan_key    = axis_settings[rewriting_axis]
    target_cid     = axis_to_cluster[rewriting_axis]
    target_sid     = example.get("student_id", "__unknown__")

    if use_meta and resources.get("merge_maps"):
        target_cid = resources["merge_maps"].get(rewriting_axis, {}).get(target_cid, target_cid)

    member_key  = (rewriting_axis, hdbscan_key)
    member_embs = resources["cluster_member_embs"].get(member_key, {}).get(target_cid)
    member_texts = resources["cluster_member_texts"].get(member_key, {}).get(target_cid, [])
    member_sids  = resources["cluster_member_student_ids"].get(member_key, {}).get(target_cid, [])

    if member_embs is None or len(member_embs) == 0:
        print(f"[WARN] idx={example['index']} no cluster members for {rewriting_axis}/{hdbscan_key} cid={target_cid}")
        return None

    # Exclude target's own posts from the neighbor pool.
    keep = [i for i, sid in enumerate(member_sids) if sid != target_sid]
    if not keep:
        return None
    member_embs  = member_embs[keep]
    member_texts = [member_texts[i] for i in keep]
    member_sids  = [member_sids[i]  for i in keep]

    # Step 1: top-K retrieval
    query_emb = example[f"{rewriting_axis}__emb"]
    nbr_texts, nbr_sids = retrieve_top_K(
        query_emb, member_embs, member_texts, member_sids, dp_config.K, rng,
    )
    if not nbr_texts:
        return None

    # Step 2: student-disjoint partition
    partitions = partition_disjoint_by_student(nbr_texts, nbr_sids, dp_config.M)
    if not partitions:
        return None

    # Step 3: M candidate rewrites
    candidates = generate_candidates(example["bullet_summary"], example["post_template"], partitions, rewriter)
    if not candidates:
        return None

    # Step 4+5: aggregate, select nearest
    embed_model = resources[dp_config.embed_model_key]
    embed_fn = lambda text: embed_model.encode(text)
    try:
        aggregate = aggregate_with_noise(candidates, embed_fn, dp_config, np_rng)
        generated = select_nearest_candidate(candidates, aggregate, embed_fn, dp_config)
    except Exception as exc:
        print(f"[WARN] idx={example['index']} aggregation failed: {exc}")
        return None

    gen_sem   = resources["general_model"].encode(generated)
    gen_style = resources["style_model"].encode(generated)
    og_sem    = resources["original_embs_sem"][example["row_pos"]]
    og_style  = resources["original_embs_style"][example["row_pos"]]
    recon_sem   = float(cosine_similarity([gen_sem],   [og_sem])[0, 0])
    recon_style = float(cosine_similarity([gen_style], [og_style])[0, 0])

    # ROUGE-1 F1: lexical overlap between generated and original post
    rouge1_f = ROUGE_SCORER.score(example["original_post"], generated)["rouge1"].fmeasure

    per_axis = {}
    for axis, hk in axis_settings.items():
        target_cluster_id = axis_to_cluster[axis]
        reducer    = resources["reducers"][axis]
        clusterer  = resources["clusterers"][(axis, hk)]
        gen_emb    = resources[EMB_MODEL_KEY[axis]].encode(generated)
        pred_cid   = cluster_predict(gen_emb, reducer, clusterer)
        mem_embs_a = resources["cluster_member_embs"][(axis, hk)].get(target_cluster_id)
        mem_sim    = float("nan")
        if mem_embs_a is not None and len(mem_embs_a) > 0:
            mem_sim = float(cosine_similarity([gen_emb], mem_embs_a).mean())
        per_axis[axis] = {
            "hdbscan_key":           hk,
            "target_cluster":        int(target_cluster_id),
            "predicted_cluster":     pred_cid,
            "match":                 bool(pred_cid == target_cluster_id),
            "membership_similarity": mem_sim,
            "was_soft_assigned":     bool(example[f"{axis}__{hk}_was_noise"]),
        }

    return {
        "index":              example["index"],
        "recon_semantic_sim": recon_sem,
        "recon_style_sim":    recon_style,
        "rouge_f1":           rouge1_f,
        "per_axis":           per_axis,
        "generated":          generated,
        "original":           example["original_post"],
        "rewriting_axis":     rewriting_axis,
        "n_partitions":       len(partitions),
        "n_candidates":       len(candidates),
    }


def summarize_config_scores(results, axes):
    if not results:
        return {"n": 0}
    agg = {
        "n":                len(results),
        "recon_sem_mean":   float(np.mean([r["recon_semantic_sim"] for r in results])),
        "recon_style_mean": float(np.mean([r["recon_style_sim"]    for r in results])),
        "rouge_f1_mean": float(np.mean([r["rouge_f1"]    for r in results])),
    }
    for axis in axes:
        matches = [r["per_axis"][axis]["match"] for r in results]
        memsims = [r["per_axis"][axis]["membership_similarity"] for r in results
                   if not np.isnan(r["per_axis"][axis]["membership_similarity"])]
        n_soft  = sum(1 for r in results if r["per_axis"][axis]["was_soft_assigned"])
        agg[f"match_{axis}_mean"]       = float(np.mean(matches))
        agg[f"memsim_{axis}_mean"]      = float(np.mean(memsims)) if memsims else float("nan")
        agg[f"soft_assign_rate_{axis}"] = n_soft / len(results)
    return agg


def run_configuration(config_name, axis_settings, examples, rewriter,
                      resources, workers, output_path, all_results, eval_fn):
    per_example = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ex = {
            executor.submit(
                eval_fn,
                ex, axis_settings, rewriter, resources,
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
    agg  = summarize_config_scores(per_example, axes)
    print(f"  n={agg.get('n')}  recon_sem={agg.get('recon_sem_mean', 0):.3f}  "
          f"recon_style={agg.get('recon_style_mean', 0):.3f}   rouge={agg.get('rouge_f1_mean', 0):.3f}")

    all_results[config_name] = {
        "axis_settings": axis_settings,
        "aggregate":     agg,
        "per_example":   per_example,
    }
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return agg


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
def add_student_ids_to_resources(
    resources: dict,
    text_to_student_id: dict[str, str],
) -> None:
    """Patch resources in-place to add cluster_member_student_ids."""
    cluster_member_sids: dict = {}
    for (axis, hdbscan_key), texts_map in resources["cluster_member_texts"].items():
        sids_map = {}
        for cid, texts in texts_map.items():
            sids_map[cid] = [
                text_to_student_id.get(t, f"__unknown_{i}__")
                for i, t in enumerate(texts)
            ]
        cluster_member_sids[(axis, hdbscan_key)] = sids_map
    resources["cluster_member_student_ids"] = cluster_member_sids


def load_resources(clustering_root, axes, corpus_texts, text_to_student_id,
                   use_meta: bool = False, seed: int = None, meta_output_dir: str = None):
    """Load embeddings, reducers, clusterers, and cluster member maps.

    Extends 04_rag_config_eval.load_resources with cluster_member_student_ids
    required for DP-ICL student-disjoint partitioning.
    """
    print("Loading embedding models...")
    general_model = SentenceTransformer("all-mpnet-base-v2")
    qa_model      = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    style_model   = SentenceTransformer("StyleDistance/styledistance")

    reducers, clusterers = {}, {}
    cluster_member_embs, cluster_member_texts = {}, {}
    stored_embeddings, stored_labels = {}, {}
    merge_maps = {}

    print("Loading clustering artifacts...")
    for axis in axes:
        config_dir = os.path.join(clustering_root, axis)
        with open(os.path.join(config_dir, "umap_reducer.pkl"), "rb") as f:
            reducers[axis] = pickle.load(f)
        embs = np.load(os.path.join(config_dir, "embeddings.npy"))
        stored_embeddings[axis] = embs

        if use_meta:
            meta_dir = os.path.join(meta_output_dir, axis)
            raw_map  = json.load(open(os.path.join(meta_dir, "merge_map.json")))[str(seed)]
            merge_maps[axis] = {int(k): v for k, v in raw_map.items()}

        for hdbscan_key in CANDIDATE_SETTINGS[axis]:
            clusterer_path = os.path.join(config_dir, f"clusterer_{hdbscan_key}.pkl")
            if not os.path.exists(clusterer_path):
                print(f"  [warn] {axis}/{hdbscan_key}: clusterer missing")
                continue
            with open(clusterer_path, "rb") as f:
                clusterers[(axis, hdbscan_key)] = pickle.load(f)
            labels = np.load(os.path.join(config_dir, f"labels_{hdbscan_key}.npy"))
            stored_labels[(axis, hdbscan_key)] = labels

            if use_meta and axis in merge_maps:
                axis_merge = merge_maps[axis]
                meta_emb_lists, meta_text_lists = {}, {}
                for cid in set(labels[labels >= 0]):
                    cid      = int(cid)
                    meta_cid = axis_merge.get(cid)
                    if meta_cid is None:
                        continue
                    valid_idxs = [int(i) for i in np.where(labels == cid)[0]
                                  if i < len(corpus_texts)]
                    meta_emb_lists.setdefault(meta_cid, []).extend(
                        embs[valid_idxs].tolist() if valid_idxs else []
                    )
                    meta_text_lists.setdefault(meta_cid, []).extend(
                        corpus_texts[i] for i in valid_idxs
                    )
                emb_map  = {mc: np.array(v) for mc, v in meta_emb_lists.items()}
                text_map = meta_text_lists
            else:
                emb_map, text_map = {}, {}
                for cid in set(labels[labels >= 0]):
                    cid        = int(cid)
                    valid_idxs = [int(i) for i in np.where(labels == cid)[0]
                                  if i < len(corpus_texts)]
                    emb_map[cid]  = embs[valid_idxs] if valid_idxs else np.empty((0, embs.shape[1]))
                    text_map[cid] = [corpus_texts[i] for i in valid_idxs]

            cluster_member_embs[(axis, hdbscan_key)]  = emb_map
            cluster_member_texts[(axis, hdbscan_key)] = text_map

    resources = {
        "general_model":        general_model,
        "qa_model":             qa_model,
        "style_model":          style_model,
        "reducers":             reducers,
        "clusterers":           clusterers,
        "cluster_member_embs":  cluster_member_embs,
        "cluster_member_texts": cluster_member_texts,
        "stored_embeddings":    stored_embeddings,
        "stored_labels":        stored_labels,
        "merge_maps":           merge_maps,
    }
    add_student_ids_to_resources(resources, text_to_student_id)
    return resources


def build_examples(eval_df, axes, resources, post_text_to_npy_idx,
                   student_id_col: str = "student_id"):
    summary_col = "summary" if "summary" in eval_df.columns else "bullets"

    examples = []
    print("Resolving cluster assignments (soft-assigning noise points)...")
    for pos, (row_idx, row) in enumerate(tqdm.tqdm(eval_df.iterrows(),
                                                    total=len(eval_df))):
        post_text  = str(row["original_post"])
        corpus_idx = post_text_to_npy_idx.get(post_text)

        ex = {
            "index":          int(row_idx),
            "row_pos":        pos,
            "original_post":  post_text,
            "bullet_summary": row[summary_col],
            "post_template":  row["rewrites"],
            "student_id":     str(row[student_id_col]) if student_id_col in row.index else "__unknown__",
        }

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

            ex[f"{axis}__emb"] = post_emb

            for hdbscan_key in CANDIDATE_SETTINGS[axis]:
                key = (axis, hdbscan_key)
                if key not in resources["clusterers"]:
                    continue
                if corpus_idx is not None:
                    hard_label = int(resources["stored_labels"][key][corpus_idx])
                    if hard_label >= 0:
                        assigned, was_noise = hard_label, False
                    else:
                        assigned, was_noise = soft_assign_cluster(
                            post_emb, reducer, resources["clusterers"][key])
                else:
                    assigned, was_noise = soft_assign_cluster(
                        post_emb, reducer, resources["clusterers"][key])

                ex[f"{axis}__{hdbscan_key}_assigned_label"] = int(assigned)
                ex[f"{axis}__{hdbscan_key}_was_noise"]      = bool(was_noise)

        examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-data",     default="student_post_bullets.csv")
    parser.add_argument("--splits",           default="final_data/stack_exchange/eval_splits_seed0.json")
    parser.add_argument("--clustering-root",
                        default="final_data/stack_exchange/hdbscan_results/")
    parser.add_argument("--output-dir",       default="sweep_output_dp_icl")
    parser.add_argument("--model",            default="openai/Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--api-base",         default="http://localhost:8001/v1",
                        help="OpenAI-compatible API base URL (e.g. http://localhost:8001/v1 for vLLM)")
    parser.add_argument("--eval-split",       default="val", choices=["val", "test"])
    parser.add_argument("--max-posts",        type=int, default=None)
    parser.add_argument("--workers",          type=int, default=8)
    parser.add_argument("--axes",             nargs="+", default=list(CANDIDATE_SETTINGS.keys()))
    parser.add_argument("--combo-sizes",      nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--pass-a-max-posts", type=int, default=100)
    parser.add_argument("--seed",             type=int, default=None)
    parser.add_argument("--use-meta",         action="store_true")
    parser.add_argument("--meta-output-dir",  default=None,
                        help="Directory containing meta_clustering_output/<axis>/ subdirs. "
                             "Defaults to <clustering-root>/../meta_clustering_output.")
    parser.add_argument("--student-id-col",   default="student_id",
                        help="Column in --student-data that holds the author/student ID.")
    # DP-ICL flags
    parser.add_argument("--dp-k",           type=int,   default=DEFAULT_TOP_K,
                        help="Neighbors retrieved from the rewriting axis cluster.")
    parser.add_argument("--dp-m",           type=int,   default=5,
                        help="Number of student-disjoint partitions / candidate rewrites.")
    parser.add_argument("--pass-a-dp-m",   type=int,   default=None,
                        help="Override --dp-m for Pass A only (e.g., 1 for faster HDBSCAN sweep). "
                             "Pass B always uses --dp-m.")
    parser.add_argument("--dp-clip-norm",   type=float, default=1.0,
                        help="L2 clip norm applied to candidate embeddings before averaging.")
    parser.add_argument("--dp-noise-scale", type=float, default=0.1,
                        help="Gaussian noise stddev as a multiple of clip_norm/M. 0 disables noise. "
                             "Used as the sole noise scale unless --noise-scales is provided.")
    parser.add_argument("--noise-scales", nargs="+", type=float, default=None,
                        help="Sweep these noise scales in Pass A and pick the best per axis. "
                             "Overrides --dp-noise-scale for the sweep. "
                             "Example: --noise-scales 0.0 0.05 0.1 0.2 0.5 1.0")
    parser.add_argument("--dp-embed-model", default="style_model",
                        choices=["style_model", "general_model", "qa_model"],
                        help="Embedding model used for candidate aggregation.")
    parser.add_argument("--rewriting-axis", default=DEFAULT_REWRITING_AXIS,
                        choices=list(CANDIDATE_SETTINGS.keys()),
                        help="Axis used for DP-ICL neighbor retrieval and partitioning.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.use_meta and args.seed is None:
        parser.error("--seed is required when --use-meta is set")

    os.makedirs(args.output_dir, exist_ok=True)

    dp_config = DPICLConfig(
        K=args.dp_k,
        M=args.dp_m,
        clip_norm=args.dp_clip_norm,
        noise_scale=args.dp_noise_scale,
        rewriting_axis=args.rewriting_axis,
        embed_model_key=args.dp_embed_model,
    )

    # ------------------------------------------------------------------
    # Load data and splits
    # ------------------------------------------------------------------
    student_data = pd.read_csv(args.student_data).reset_index(drop=True)
    with open(args.splits) as f:
        splits = json.load(f)
    eval_indices = splits[args.eval_split]
    if args.max_posts:
        eval_indices = eval_indices[:args.max_posts]
    eval_df_full = student_data.loc[eval_indices]
    print(f"Loaded {len(eval_df_full)} posts from '{args.eval_split}' split")

    if args.debug:
        eval_df_full = eval_df_full.sample(n=20, random_state=42)

    # ------------------------------------------------------------------
    # Build corpus text list and student-ID map
    # ------------------------------------------------------------------
    print("Building corpus text list...")
    corpus_texts         = student_data.loc[splits["train"], "original_post"].astype(str).tolist()
    post_text_to_npy_idx = {text: i for i, text in enumerate(corpus_texts)}

    text_to_student_id = {}
    if args.student_id_col in student_data.columns:
        text_to_student_id = dict(zip(
            student_data["original_post"].astype(str),
            student_data[args.student_id_col].astype(str),
        ))
    else:
        print(f"[WARN] --student-id-col '{args.student_id_col}' not found; "
              "student isolation will be skipped.")

    # ------------------------------------------------------------------
    # Load resources (includes cluster_member_student_ids for DP-ICL)
    # ------------------------------------------------------------------
    meta_output_dir = args.meta_output_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.clustering_root)), "meta_clustering_output"
    )
    resources = load_resources(
        args.clustering_root, args.axes, corpus_texts, text_to_student_id,
        use_meta=args.use_meta, seed=args.seed, meta_output_dir=meta_output_dir,
    )

    print("Precomputing eval-set embeddings...")
    eval_posts = list(eval_df_full["original_post"])
    resources["original_embs_sem"]   = np.array(
        resources["general_model"].encode(eval_posts, batch_size=64))
    resources["original_embs_style"] = np.array(
        resources["style_model"].encode(eval_posts, batch_size=64))

    examples_full = build_examples(
        eval_df_full, args.axes, resources, post_text_to_npy_idx,
        student_id_col=args.student_id_col,
    )

    # ------------------------------------------------------------------
    # Rewriter + eval function
    # ------------------------------------------------------------------
    lm_kwargs = {"api_base": args.api_base, "api_key": "none"} if args.api_base else {}
    dspy.configure(lm=dspy.LM(args.model, **lm_kwargs))
    rewriter = Rewriter()
    eval_fn  = functools.partial(eval_one_dp_icl, dp_config=dp_config, use_meta=args.use_meta)

    # ==================================================================
    # Pass A: per-axis HDBSCAN setting selection
    # ==================================================================
    pass_a_dp_config = (
        dataclass_replace(dp_config, M=args.pass_a_dp_m)
        if args.pass_a_dp_m is not None else dp_config
    )

    print("\n" + "=" * 70)
    print(f"PASS A: per-axis HDBSCAN-setting selection  "
          f"(dp_k={pass_a_dp_config.K}, dp_m={pass_a_dp_config.M}, "
          f"rewriting_axis={pass_a_dp_config.rewriting_axis})")
    print("=" * 70)
    winners_path = os.path.join(args.output_dir, "pass_a_winners.json")

    noise_scales_to_sweep = args.noise_scales if args.noise_scales else [args.dp_noise_scale]
    sweep_noise = len(noise_scales_to_sweep) > 1

    if args.use_meta:
        winner_per_axis = {axis: OPTIMAL[axis] for axis in args.axes}
        print("  --use-meta active: skipping Pass A sweep, using OPTIMAL settings.")
        for axis, setting in winner_per_axis.items():
            print(f"  -> {axis}: {setting}")
        with open(winners_path, "w") as f:
            json.dump(winner_per_axis, f, indent=2)
        print(f"Pass A winners saved -> {winners_path}")
    else:
        if sweep_noise:
            print(f"  Sweeping noise scales: {noise_scales_to_sweep}")
        pass_a_results  = {}
        pass_a_path     = os.path.join(args.output_dir, "pass_a_results.json")
        pass_a_examples = examples_full[:args.pass_a_max_posts]
        winner_per_axis = {}
        for axis in args.axes:
            best_hdbscan, best_noise, best_score = None, None, -float("inf")
            for noise_scale in noise_scales_to_sweep:
                ns_config = dataclass_replace(pass_a_dp_config, noise_scale=noise_scale)
                ns_eval_fn = functools.partial(
                    eval_one_dp_icl, dp_config=ns_config, use_meta=args.use_meta
                )
                for hdbscan_key in CANDIDATE_SETTINGS[axis]:
                    ns_suffix     = f"::ns{noise_scale}" if sweep_noise else ""
                    config_name   = f"A::{axis}::{hdbscan_key}{ns_suffix}"
                    agg = run_configuration(
                        config_name, {axis: hdbscan_key}, pass_a_examples, rewriter,
                        resources, args.workers, pass_a_path, pass_a_results, ns_eval_fn,
                    )
                    score = agg.get("recon_sem_mean", 0.0)
                    if score > best_score:
                        best_score, best_hdbscan, best_noise = score, hdbscan_key, noise_scale
            if sweep_noise:
                winner_per_axis[axis] = {"hdbscan_key": best_hdbscan, "noise_scale": best_noise}
                print(f"  -> {axis} winner: {best_hdbscan} @ noise_scale={best_noise} "
                      f"(recon_sem={best_score:.3f})")
            else:
                winner_per_axis[axis] = best_hdbscan
                print(f"  -> {axis} winner: {best_hdbscan} (recon_sem={best_score:.3f})")
        with open(winners_path, "w") as f:
            json.dump(winner_per_axis, f, indent=2)
        print(f"Pass A winners saved -> {winners_path}")

    # ==================================================================
    # Pass B: axis-subset combinations at winning settings
    # ==================================================================

    # If noise was swept in Pass A, apply the rewriting axis winner's noise_scale
    # to dp_config and rebind eval_fn so Pass B uses the optimal noise level.
    rewriting_winner = winner_per_axis.get(dp_config.rewriting_axis)
    if isinstance(rewriting_winner, dict):
        dp_config = dataclass_replace(dp_config, noise_scale=rewriting_winner["noise_scale"])
        eval_fn   = functools.partial(eval_one_dp_icl, dp_config=dp_config, use_meta=args.use_meta)
        print(f"  Pass B noise_scale set to winner: {dp_config.noise_scale}")

    def _hdbscan_key(winner):
        return winner["hdbscan_key"] if isinstance(winner, dict) else winner

    print("\n" + "=" * 70)
    print(f"PASS B: axis-combination sweep  "
          f"(rewriting_axis={dp_config.rewriting_axis}, noise_scale={dp_config.noise_scale})")
    print("=" * 70)
    pass_b_results = {}
    pass_b_path    = os.path.join(args.output_dir, "pass_b_results.json")
    for size in args.combo_sizes:
        for subset in itertools.combinations(args.axes, size):
            config_name   = "B::" + "+".join(subset)
            axis_settings = {axis: _hdbscan_key(winner_per_axis[axis]) for axis in subset}
            run_configuration(
                config_name, axis_settings, examples_full, rewriter,
                resources, args.workers, pass_b_path, pass_b_results, eval_fn,
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
