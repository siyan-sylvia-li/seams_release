"""
10_authorship_attack_v2.py

Within-cluster authorship attribution attack implementing four attackers:

Cluster assignment strategy (--author-home-cluster, default on):
  For each eval post the target cluster is the non-noise cluster in which the
  true author has the MOST training posts ("author home cluster").  This
  maximises coverage: any eval post whose author has training data anywhere is
  evaluatable, rather than only posts whose rewrite happened to land in the
  same cluster as the author's training posts.  Pass --no-author-home-cluster
  to revert to the original behaviour (use the cluster the rewrite was
  assigned to).

  1. AnnaWegmann embedding-diff + logistic regression (style classifier)
  2. AnnaWegmann max-sim retrieval                    (style retrieval)
  3. Stylometric LR: char n-grams + function words + punct + sent_len + TTR
  4. all-mpnet-base-v2 max-sim retrieval              (content retrieval)

Metrics (all chance-normalised per cluster, then averaged across posts):
  - Top-1 accuracy / chance(k)          where chance(k) = 1/k
  - MRR / chance-MRR(k)                 where chance-MRR(k) = H_k / k

Usage
-----
# SEAMS M=5 (cluster-json = rewrites-json by default)
python 10_authorship_attack_v2.py \\
    --student-data  final_data/og_students_v2.csv \\
    --rewrites-json final_data/og_students/dp_icl_outputs/meta/seed_123/pass_b_results.json \\
    --splits        final_data/og_students/eval_splits_seed123.json \\
    --hdbscan-root  final_data/og_students/hdbscan_results \\
    --merge-map     final_data/og_students/meta_clustering_output/.../merge_map.json \\
    --cluster-seed  123 \\
    --output        results/seams_m5_seed123.json

# Real posts ceiling (no rewrites-json; needs --cluster-json from any SEAMS run)
python 10_authorship_attack_v2.py \\
    --student-data  final_data/og_students_v2.csv \\
    --use-original-as-query \\
    --cluster-json  final_data/og_students/dp_icl_outputs/meta/seed_123/pass_b_results.json \\
    --splits        final_data/og_students/eval_splits_seed123.json \\
    ...

# Baseline (cluster-json needed separately; original resolved by CSV index)
python 10_authorship_attack_v2.py \\
    --student-data  final_data/og_students_v2.csv \\
    --rewrites-json final_data/og_students/baselines/seed_123/baseline_br/results.json \\
    --cluster-json  final_data/og_students/dp_icl_outputs/meta/seed_123/pass_b_results.json \\
    --splits        final_data/og_students/eval_splits_seed123.json \\
    ...

Known limitation: candidate reference posts are drawn from the same training-split pool
that SEAMS uses for ICL conditioning. DP noise mitigates direct memorisation but this
is documented as a caveat.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("HUGGINGFACE_HUB_CACHE",
                      os.path.expanduser("~/.cache/huggingface/hub"))

AXIS = "og_style_model"

# ---------------------------------------------------------------------------
# Stylometric helpers
# ---------------------------------------------------------------------------
_FUNCTION_WORDS = [
    "a","about","above","after","again","against","all","also","am","an","and",
    "any","are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","could","did","didn't","do","does","doesn't",
    "doing","don't","down","during","each","even","few","for","from","further","get",
    "got","had","hadn't","has","hasn't","have","haven't","having","he","her","here",
    "him","himself","his","how","i","if","in","into","is","isn't","it","its","itself",
    "just","let","like","may","me","might","more","most","much","must","my","myself",
    "no","nor","not","now","of","off","on","once","only","or","other","our","out",
    "over","own","same","she","should","shouldn't","so","some","such","than","that",
    "the","their","them","then","there","these","they","this","those","though",
    "through","to","too","under","until","up","us","very","was","wasn't","we","were",
    "weren't","what","when","where","which","while","who","whom","why","will","with",
    "won't","would","wouldn't","you","your",
]
_PUNCT = list('.,!?;:-()"\'')


def _base_stylo_vec(text: str) -> np.ndarray:
    """Fixed-length stylometric features: funcwords + punct rates + sent_len + TTR."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    n = max(len(tokens), 1)
    # function word frequencies
    fw = np.array([tokens.count(w) / n for w in _FUNCTION_WORDS], dtype=np.float32)
    # punctuation rates
    punct = np.array([text.count(p) / max(len(text), 1) for p in _PUNCT], dtype=np.float32)
    # sentence length stats (in words)
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_lens = [len(re.findall(r"\b\w+\b", s)) for s in sents] if sents else [n]
    sl = np.array([
        np.mean(sent_lens), np.std(sent_lens), float(max(sent_lens))
    ], dtype=np.float32)
    # TTR
    ttr = np.array([len(set(tokens)) / n], dtype=np.float32)
    return np.concatenate([fw, punct, sl, ttr])


# ---------------------------------------------------------------------------
# Pair feature helpers
# ---------------------------------------------------------------------------
def _pair_emb(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    cos = float(cosine_similarity([ea], [eb])[0, 0])
    return np.concatenate([np.abs(ea - eb), [cos]], dtype=np.float32)


def _pair_stylo(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    cos = float(cosine_similarity([va], [vb])[0, 0])
    return np.concatenate([np.abs(va - vb), [cos]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Chance-MRR for cluster of size k
# ---------------------------------------------------------------------------
def _chance_mrr(k: int) -> float:
    return sum(1.0 / i for i in range(1, k + 1)) / k


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _resolve_examples(data: dict, config: str | None) -> list[dict]:
    if "per_example" in data:
        return data["per_example"]
    if config is None:
        config = next(iter(data))
    return data[config]["per_example"]


def load_cluster_map(
    cluster_json: str,
    config: str | None,
    hdbscan_root: str,
    merge_map_path: str,
    cluster_seed: str,
    train_indices: list[int],
    student_data: pd.DataFrame,
    post_col: str,
) -> tuple[dict[int, int], dict[int, dict[str, list[str]]]]:
    """Return (idx_to_meta, meta_to_train).

    idx_to_meta  : CSV row index -> meta cluster id (for eval posts)
    meta_to_train: meta cluster id -> {student_id -> [post texts]} (training posts)
    """
    with open(cluster_json) as f:
        data = json.load(f)
    examples = _resolve_examples(data, config)

    with open(merge_map_path) as f:
        merge_map = json.load(f)
    seed_map = {int(k): v for k, v in merge_map.get(cluster_seed, {}).items()}

    # Map CSV row index → meta cluster from per_axis
    idx_to_meta: dict[int, int] = {}
    hdbscan_key = None
    for ex in examples:
        pa = ex.get("per_axis", {}).get(AXIS, {})
        per_seed_cid = pa.get("target_cluster")
        if hdbscan_key is None:
            hdbscan_key = pa.get("hdbscan_key")
        idx = ex.get("index")
        if idx is not None and per_seed_cid is not None:
            idx_to_meta[int(idx)] = seed_map.get(per_seed_cid, -1)

    print(f"[cluster] hdbscan_key={hdbscan_key}  seed={cluster_seed}  "
          f"eval indices mapped={len(idx_to_meta)}  "
          f"non-noise meta clusters={len([c for c in set(idx_to_meta.values()) if c >= 0])}")

    if hdbscan_key is None:
        return idx_to_meta, {}

    labels_path = os.path.join(
        hdbscan_root, f"seed_{cluster_seed}", AXIS, f"labels_{hdbscan_key}.npy"
    )
    labels = np.load(labels_path)
    meta_to_train: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for i, idx in enumerate(train_indices):
        meta_cid = seed_map.get(int(labels[i]), -1)
        row = student_data.iloc[idx]
        meta_to_train[meta_cid][str(row["student_id"])].append(str(row[post_col]))

    return idx_to_meta, dict(meta_to_train)


def load_eval_pairs(
    rewrites_json: str | None,
    config: str | None,
    student_data: pd.DataFrame,
    post_col: str,
    use_original_as_query: bool,
) -> list[dict]:
    """Return list of {idx, original_text, query_text} for eval examples.

    query_text = generated post (normal mode) or original post (ceiling mode).
    """
    if use_original_as_query or rewrites_json is None:
        # Will be filled in main using idx_to_meta keys
        return []

    with open(rewrites_json) as f:
        data = json.load(f)
    examples = _resolve_examples(data, config)

    records = []
    for ex in examples:
        idx = ex.get("index")
        gen = ex.get("generated", "")
        if not gen or idx is None:
            continue
        orig = ex.get("original", "")
        if not orig:
            try:
                orig = str(student_data.iloc[int(idx)][post_col])
            except (IndexError, KeyError):
                continue
        records.append({"idx": int(idx), "original": orig, "query": gen})
    return records


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
def embed_all(texts: list[str], model: SentenceTransformer) -> dict[str, np.ndarray]:
    unique = list(dict.fromkeys(texts))
    embs = model.encode(unique, batch_size=64, show_progress_bar=False)
    return {t: embs[i] for i, t in enumerate(unique)}


# ---------------------------------------------------------------------------
# Training pair construction (author hold-out: exclude eval authors)
# ---------------------------------------------------------------------------
def build_train_pairs(
    train_posts: dict[str, list[str]],
    eval_sids: set[str],
    n_same: int,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], list[int]]:
    eligible = {sid: posts for sid, posts in train_posts.items()
                if sid not in eval_sids and len(posts) >= 2}
    same_pool: list[tuple[str, str]] = []
    for posts in eligible.values():
        for a, b in combinations(posts, 2):
            same_pool.append((a, b))
    rng.shuffle(same_pool)
    same = same_pool[:n_same]

    all_sids = list(eligible.keys())
    diff: list[tuple[str, str]] = []
    attempts = 0
    while len(diff) < len(same) and attempts < len(same) * 20:
        attempts += 1
        sa, sb = rng.sample(all_sids, 2)
        diff.append((rng.choice(eligible[sa]), rng.choice(eligible[sb])))

    pairs = same + diff
    labels = [1] * len(same) + [0] * len(diff)
    combined = list(zip(pairs, labels))
    rng.shuffle(combined)
    pairs, labels = zip(*combined)
    return list(pairs), list(labels)


# ---------------------------------------------------------------------------
# Attackers
# ---------------------------------------------------------------------------
class EmbeddingClassifier:
    """Attacker 1: pair-feature LR classifier on embedding space."""
    name = "emb_clf"

    def __init__(self, emb_cache: dict[str, np.ndarray]):
        self._cache = emb_cache
        self._clf: Pipeline | None = None

    def fit(self, pairs: list[tuple[str, str]], labels: list[int]) -> None:
        X = np.array([_pair_emb(self._cache[a], self._cache[b])
                      for a, b in pairs if a in self._cache and b in self._cache])
        y = np.array(labels[:len(X)])
        self._clf = Pipeline([("sc", StandardScaler()),
                               ("lr", LogisticRegression(C=1.0, max_iter=1000,
                                                         random_state=0))])
        self._clf.fit(X, y)
        print(f"  [{self.name}] trained on {len(X)} pairs")

    def score(self, query: str, candidate_posts: list[str]) -> float:
        if self._clf is None or query not in self._cache:
            return 0.0
        eq = self._cache[query]
        probs = []
        for p in candidate_posts:
            if p in self._cache:
                feat = _pair_emb(eq, self._cache[p])
                probs.append(float(self._clf.predict_proba([feat])[0, 1]))
        return max(probs) if probs else 0.0


class MaxSimRetriever:
    """Attackers 2 & 4: max cosine similarity over candidate's held-out posts."""
    def __init__(self, name: str, emb_cache: dict[str, np.ndarray]):
        self.name = name
        self._cache = emb_cache

    def fit(self, *args, **kwargs) -> None:
        pass  # no training

    def score(self, query: str, candidate_posts: list[str]) -> float:
        if query not in self._cache:
            return 0.0
        eq = self._cache[query]
        sims = [float(cosine_similarity([eq], [self._cache[p]])[0, 0])
                for p in candidate_posts if p in self._cache]
        return max(sims) if sims else 0.0


class StylometricClassifier:
    """Attacker 3: char n-gram + funcwords + punct + sent_len + TTR LR."""
    name = "stylo_clf"

    def __init__(self, max_ngram_features: int = 2000):
        self._ngram_max = max_ngram_features
        self._tfidf: TfidfVectorizer | None = None
        self._clf: Pipeline | None = None
        self._full_cache: dict[str, np.ndarray] = {}  # full vector cache post-fit

    def _vec(self, text: str) -> np.ndarray:
        if text in self._full_cache:
            return self._full_cache[text]
        base = _base_stylo_vec(text)
        if self._tfidf is None:
            return base
        ng = self._tfidf.transform([text]).toarray()[0].astype(np.float32)
        result = np.concatenate([base, ng])
        self._full_cache[text] = result
        return result

    def prime_cache(self, texts: list[str]) -> None:
        """Batch-vectorize texts to populate the full cache."""
        if self._tfidf is None:
            return
        unique = [t for t in dict.fromkeys(texts) if t not in self._full_cache]
        if not unique:
            return
        ng_matrix = self._tfidf.transform(unique).toarray().astype(np.float32)
        for text, ng_row in zip(unique, ng_matrix):
            base = _base_stylo_vec(text)
            self._full_cache[text] = np.concatenate([base, ng_row])

    def fit(self, pairs: list[tuple[str, str]], labels: list[int],
            all_texts: list[str]) -> None:
        self._tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=self._ngram_max, sublinear_tf=True,
        )
        self._tfidf.fit(all_texts)
        # Batch-vectorize all training texts at once before building pair features
        self.prime_cache(all_texts)
        pair_texts = [t for pair in pairs for t in pair]
        self.prime_cache(pair_texts)
        X = np.array([_pair_stylo(self._vec(a), self._vec(b)) for a, b in pairs])
        y = np.array(labels)
        self._clf = Pipeline([("sc", StandardScaler()),
                               ("lr", LogisticRegression(C=1.0, max_iter=1000,
                                                         random_state=0))])
        self._clf.fit(X, y)
        print(f"  [{self.name}] trained on {len(X)} pairs  "
              f"feat_dim={X.shape[1]}")

    def score(self, query: str, candidate_posts: list[str]) -> float:
        if self._clf is None:
            return 0.0
        vq = self._vec(query)
        probs = [float(self._clf.predict_proba([_pair_stylo(vq, self._vec(p))])[0, 1])
                 for p in candidate_posts]
        return max(probs) if probs else 0.0


# ---------------------------------------------------------------------------
# Per-post ranking
# ---------------------------------------------------------------------------
def rank_one(
    query: str,
    true_sid: str,
    candidates: list[tuple[str, list[str]]],  # (sid, posts)
    attackers: list,
) -> dict:
    K = len(candidates)
    result: dict = {"K": K, "true_sid": true_sid}
    for att in attackers:
        scores = sorted(
            [(sid, att.score(query, posts)) for sid, posts in candidates],
            key=lambda x: x[1], reverse=True,
        )
        rank = next((i + 1 for i, (sid, _) in enumerate(scores) if sid == true_sid), K + 1)
        result[att.name] = {
            "rank": rank,
            "top1": int(rank == 1),
            "rr": round(1.0 / rank, 6),
            "chance_top1": round(1.0 / K, 6),
            "chance_mrr": round(_chance_mrr(K), 6),
        }
    return result


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def aggregate(per_example: list[dict], attacker_names: list[str]) -> dict:
    out = {}
    for name in attacker_names:
        rows = [r[name] for r in per_example if name in r]
        if not rows:
            continue
        top1       = np.mean([r["top1"]       for r in rows])
        mrr        = np.mean([r["rr"]         for r in rows])
        ch_top1    = np.mean([r["chance_top1"] for r in rows])
        ch_mrr     = np.mean([r["chance_mrr"] for r in rows])
        out[name] = {
            "n":               len(rows),
            "top1":            round(float(top1), 4),
            "mrr":             round(float(mrr), 4),
            "chance_top1":     round(float(ch_top1), 4),
            "chance_mrr":      round(float(ch_mrr), 4),
            "top1_over_chance": round(float(top1 / ch_top1) if ch_top1 > 0 else 0, 4),
            "mrr_over_chance":  round(float(mrr  / ch_mrr)  if ch_mrr  > 0 else 0, 4),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-data",          required=True)
    parser.add_argument("--rewrites-json",          default=None,
                        help="Generated posts JSON. Omit with --use-original-as-query.")
    parser.add_argument("--cluster-json",           default=None,
                        help="SEAMS pass_b_results.json for cluster assignments. "
                             "Defaults to --rewrites-json for SEAMS outputs.")
    parser.add_argument("--splits",                 required=True)
    parser.add_argument("--hdbscan-root",           required=True)
    parser.add_argument("--merge-map",              required=True)
    parser.add_argument("--cluster-seed",           required=True)
    parser.add_argument("--eval-split",             default="val",
                        choices=["val", "test"])
    parser.add_argument("--config",                 default=None)
    parser.add_argument("--post-col",               default="original_post")
    parser.add_argument("--sid-col",                default="student_id")
    parser.add_argument("--use-original-as-query",  action="store_true",
                        help="Query = original post (real-posts ceiling).")
    parser.add_argument("--n-train-pairs",          type=int, default=5000)
    parser.add_argument("--seed",                   type=int, default=42)
    parser.add_argument("--output",                 required=True)
    parser.add_argument("--no-author-home-cluster", action="store_true",
                        help="Revert to original behaviour: use the cluster "
                             "the eval post's rewrite was assigned to.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    cluster_json = args.cluster_json or args.rewrites_json
    if cluster_json is None:
        parser.error("Provide --cluster-json (or --rewrites-json for SEAMS outputs).")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading student data...")
    df = pd.read_csv(args.student_data).reset_index(drop=True)
    splits = json.load(open(args.splits))
    train_idx = splits["train"]
    eval_idx  = splits[args.eval_split]

    eval_sids = set(str(df.iloc[i][args.sid_col]) for i in eval_idx)
    train_posts_all: dict[str, list[str]] = defaultdict(list)
    for i in train_idx:
        row = df.iloc[i]
        train_posts_all[str(row[args.sid_col])].append(str(row[args.post_col]))

    print(f"  train posts: {sum(len(v) for v in train_posts_all.values())}  "
          f"eval posts: {len(eval_idx)}  eval authors: {len(eval_sids)}")

    # ------------------------------------------------------------------
    # Cluster map
    # ------------------------------------------------------------------
    print("\nBuilding cluster map...")
    merge_map_path = args.merge_map
    idx_to_meta, meta_to_train = load_cluster_map(
        cluster_json, args.config, args.hdbscan_root, merge_map_path,
        args.cluster_seed, train_idx, df, args.post_col,
    )

    # ------------------------------------------------------------------
    # Author home cluster: sid -> meta_cid with most training posts
    # ------------------------------------------------------------------
    author_home: dict[str, int] = {}
    for meta_cid, students in meta_to_train.items():
        if meta_cid < 0:
            continue
        for sid, posts in students.items():
            prev = author_home.get(sid)
            if prev is None or len(posts) > len(meta_to_train[prev].get(sid, [])):
                author_home[sid] = meta_cid
    print(f"  authors with a home cluster: {len(author_home)}")

    # ------------------------------------------------------------------
    # Eval pairs
    # ------------------------------------------------------------------
    if args.use_original_as_query:
        eval_records = [
            {"idx": i,
             "original": str(df.iloc[i][args.post_col]),
             "query":    str(df.iloc[i][args.post_col])}
            for i in eval_idx if i in idx_to_meta
        ]
    else:
        eval_records = load_eval_pairs(
            args.rewrites_json, args.config, df, args.post_col, False,
        )
        # Filter to only eval indices that have cluster assignments
        eval_records = [r for r in eval_records if r["idx"] in idx_to_meta]

    print(f"  eval records with cluster assignment: {len(eval_records)}")

    # ------------------------------------------------------------------
    # Gather all texts for embedding
    # ------------------------------------------------------------------
    print("\nCollecting texts for embedding...")
    all_texts: list[str] = []
    for r in eval_records:
        all_texts.append(r["query"])
        all_texts.append(r["original"])
    for posts in train_posts_all.values():
        all_texts.extend(posts)
    all_texts = list(dict.fromkeys(all_texts))
    print(f"  unique texts: {len(all_texts)}")

    # ------------------------------------------------------------------
    # Load embedding models
    # ------------------------------------------------------------------
    print("\nLoading AnnaWegmann/Style-Embedding...")
    wegmann = SentenceTransformer("AnnaWegmann/Style-Embedding")
    print("Loading all-mpnet-base-v2...")
    mpnet   = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("Encoding with AnnaWegmann...")
    weg_cache = embed_all(all_texts, wegmann)
    print("Encoding with all-mpnet...")
    mpn_cache = embed_all(all_texts, mpnet)

    # ------------------------------------------------------------------
    # Build training pairs (excluding eval authors)
    # ------------------------------------------------------------------
    print("\nBuilding training pairs (excluding eval authors)...")
    pairs, labels = build_train_pairs(
        train_posts_all, eval_sids, args.n_train_pairs, rng,
    )
    print(f"  pairs: {len(pairs)}  (same={sum(labels)}  diff={len(labels)-sum(labels)})")

    # ------------------------------------------------------------------
    # Train / prepare attackers
    # ------------------------------------------------------------------
    print("\nPreparing attackers...")
    att1 = EmbeddingClassifier(weg_cache)
    att1.name = "wegmann_clf"
    att1.fit(pairs, labels)

    att2 = MaxSimRetriever("wegmann_ret", weg_cache)

    att3 = StylometricClassifier()
    all_train_texts = [t for sid, posts in train_posts_all.items()
                       if sid not in eval_sids for t in posts]
    att3.fit(pairs, labels, all_train_texts)
    # Pre-compute vectors for all texts (including eval-author candidate posts)
    print(f"  [{att3.name}] priming cache for all {len(all_texts)} texts...")
    att3.prime_cache(all_texts)
    print(f"  [{att3.name}] cache size: {len(att3._full_cache)}")

    att4 = MaxSimRetriever("mpnet_ret", mpn_cache)

    attackers = [att1, att2, att3, att4]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"\nRunning within-cluster attack on {len(eval_records)} eval posts...")
    per_example = []
    skipped = 0

    use_home = not args.no_author_home_cluster

    for r in eval_records:
        idx      = r["idx"]
        query    = r["query"]
        orig     = r["original"]
        sid      = str(df.iloc[idx][args.sid_col])

        if use_home:
            meta_cid = author_home.get(sid)
            if meta_cid is None:
                skipped += 1
                continue
        else:
            meta_cid = idx_to_meta.get(idx, -1)
            if meta_cid < 0 or meta_cid not in meta_to_train:
                skipped += 1
                continue

        cluster_students = meta_to_train[meta_cid]

        # True author needs at least one held-out training post ≠ original
        true_posts = [p for p in cluster_students.get(sid, []) if p != orig]
        if not true_posts:
            skipped += 1
            continue

        # Build candidate list: (sid, held-out posts)
        candidates: list[tuple[str, list[str]]] = []
        for cand_sid, posts in cluster_students.items():
            if cand_sid == sid:
                candidates.append((cand_sid, true_posts))
            else:
                if posts:
                    candidates.append((cand_sid, posts))

        if len(candidates) < 2:
            skipped += 1
            continue

        rec = rank_one(query, sid, candidates, attackers)
        rec["idx"]  = idx
        rec["meta_cluster"] = meta_cid
        per_example.append(rec)

    print(f"  evaluated: {len(per_example)}  skipped: {skipped}")

    # ------------------------------------------------------------------
    # Aggregate and report
    # ------------------------------------------------------------------
    att_names = [a.name for a in attackers]
    agg = aggregate(per_example, att_names)

    print(f"\n{'='*65}")
    print(f"{'Attacker':<16} {'top1':>6} {'chance':>6} {'ratio':>6}  "
          f"{'MRR':>6} {'chMRR':>6} {'ratio':>6}")
    print("-" * 65)
    for name, m in agg.items():
        print(f"{name:<16} {m['top1']:>6.3f} {m['chance_top1']:>6.3f} "
              f"{m['top1_over_chance']:>6.3f}  "
              f"{m['mrr']:>6.3f} {m['chance_mrr']:>6.3f} "
              f"{m['mrr_over_chance']:>6.3f}")

    # Cluster size distribution
    K_counts: dict[int, int] = defaultdict(int)
    for r in per_example:
        K_counts[r["K"]] += 1

    result = {
        "args": vars(args),
        "author_home_cluster": use_home,
        "n_eval": len(per_example),
        "n_skipped": skipped,
        "aggregate": agg,
        "cluster_size_distribution": dict(sorted(K_counts.items())),
        "per_example": per_example,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
