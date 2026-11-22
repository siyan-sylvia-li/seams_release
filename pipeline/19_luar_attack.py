"""
19_luar_attack.py

LUAR retrieval attacker (Rivera-Soto et al., 2021; rrivera1849/LUAR-MUD), added to the
within-cluster authorship attack of 10_authorship_attack_v2.py without rerunning it.

Protocol is identical to 10_authorship_attack_v2.py (default author-home-cluster mode):
same eval posts, same candidate pools (students with training posts in the true author's
home meta-cluster, from the SEAMS cluster file), same chance normalisation. Each post is
embedded with LUAR as a one-document episode; a candidate's score is the max cosine
similarity between the query and any of the candidate's training posts (as Weg-ret).

Training posts are embedded once per (dataset, seed) and reused for every method.

Usage:
  python 19_luar_attack.py --dataset stack_exchange [--methods seams_m5,real] [--seeds 0,7]
Output: final_data/<ds>/authorship_luar_outputs/<method>_seed<S>.json
"""

import argparse
import importlib.util
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

PIPE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(PIPE, "final_data")
_spec = importlib.util.spec_from_file_location("atk", os.path.join(PIPE, "10_authorship_attack_v2.py"))
atk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atk)

LUAR = "rrivera1849/LUAR-MUD"
MAXLEN = 512
METHODS = {  # name -> rewrites path (None = real posts as queries)
    "seams_m5":                  "dp_icl_outputs/meta/seed_{S}/pass_b_results.json",
    "seams_m1":                  "dp_icl_outputs/meta_m1/seed_{S}/pass_b_results.json",
    "baseline_esa_zscs":         "dp_icl_outputs/baseline_esa_zscs/seed_{S}/pass_b_results.json",
    "baseline_br":               "baselines/seed_{S}/baseline_br/results.json",
    "dp_prompt":                 "dp_prompt_outputs/seed_{S}/pass_b_results.json",  # first config = t=1.0
    "baseline_semantic_knn":     "dp_icl_outputs/baseline_semantic_knn/seed_{S}/pass_b_results.json",
    "real":                      None,
    "ablation_unconstrained":    "dp_icl_outputs/ablation_unconstrained/seed_{S}/pass_b_results.json",
    "ablation_deterministic":    "dp_icl_outputs/ablation_deterministic/seed_{S}/pass_b_results.json",
    "ablation_random_selection": "dp_icl_outputs/ablation_random_selection/seed_{S}/pass_b_results.json",
}


class LuarEmbedder:
    def __init__(self, dev):
        self.tok = AutoTokenizer.from_pretrained(LUAR, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(LUAR, trust_remote_code=True).to(dev).eval()
        self.dev = dev
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed(self, texts, bs=64):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            enc = self.tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            ids = enc["input_ids"].unsqueeze(1).to(self.dev)        # (B, episode=1, L)
            mask = enc["attention_mask"].unsqueeze(1).to(self.dev)
            out = self.model(ids, mask).float().cpu().numpy()
            for t, e in zip(batch, out):
                self.cache[t] = e
        return self.cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default=None, help="comma-separated (default: all with rewrites present)")
    ap.add_argument("--seeds", default="0,7,42,99,123")
    args = ap.parse_args()
    ds = args.dataset
    base = os.path.join(FD, ds)
    out_dir = os.path.join(base, "authorship_luar_outputs")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(FD, f"{ds}_v2.csv")).reset_index(drop=True)
    merge_map = os.path.join(base, "meta_clustering_output", "og_style_model", "merge_map.json")
    emb = LuarEmbedder("cuda" if torch.cuda.is_available() else "cpu")
    methods = args.methods.split(",") if args.methods else list(METHODS)

    for S in [int(s) for s in args.seeds.split(",")]:
        splits = json.load(open(os.path.join(base, f"eval_splits_seed{S}.json")))
        train_idx, eval_idx = splits["train"], splits["val"]
        seams_json = os.path.join(base, f"dp_icl_outputs/meta/seed_{S}/pass_b_results.json")
        idx_to_meta, meta_to_train = atk.load_cluster_map(
            seams_json, None, os.path.join(base, "hdbscan_results"), merge_map, str(S),
            train_idx, df, "original_post")
        author_home = {}
        for cid, students in meta_to_train.items():
            if cid < 0:
                continue
            for sid, posts in students.items():
                prev = author_home.get(sid)
                if prev is None or len(posts) > len(meta_to_train[prev].get(sid, [])):
                    author_home[sid] = cid
        train_texts = [str(df.iloc[i]["original_post"]) for i in train_idx]
        print(f"[{ds} seed {S}] embedding {len(train_texts)} training posts", flush=True)
        cache = emb.embed(train_texts)
        retr = atk.MaxSimRetriever("luar_ret", cache)

        for name in methods:
            out = os.path.join(out_dir, f"{name}_seed{S}.json")
            if os.path.exists(out):
                continue
            tpl = METHODS[name]
            if tpl is None:
                recs = [{"idx": i, "original": str(df.iloc[i]["original_post"]),
                         "query": str(df.iloc[i]["original_post"])} for i in eval_idx if i in idx_to_meta]
            else:
                rp = os.path.join(base, tpl.format(S=S))
                if not os.path.exists(rp):
                    continue
                recs = [r for r in atk.load_eval_pairs(rp, None, df, "original_post", False) if r["idx"] in idx_to_meta]
            emb.embed([r["query"] for r in recs])
            per_example, skipped = [], 0
            for r in recs:
                sid = str(df.iloc[r["idx"]]["student_id"])
                cid = author_home.get(sid)
                if cid is None:
                    skipped += 1
                    continue
                cluster_students = meta_to_train[cid]
                true_posts = [p for p in cluster_students.get(sid, []) if p != r["original"]]
                if not true_posts:
                    skipped += 1
                    continue
                cands = [(c, true_posts if c == sid else posts) for c, posts in cluster_students.items()
                         if c == sid or posts]
                if len(cands) < 2:
                    skipped += 1
                    continue
                rec = atk.rank_one(r["query"], sid, cands, [retr])
                rec["idx"], rec["meta_cluster"] = r["idx"], cid
                per_example.append(rec)
            agg = atk.aggregate(per_example, ["luar_ret"])
            print(f"  {name:26s} n={len(per_example)}  MRR ratio={agg['luar_ret']['mrr_over_chance']:.3f}", flush=True)
            json.dump({"dataset": ds, "seed": S, "method": name, "model": LUAR, "n_eval": len(per_example),
                       "n_skipped": skipped, "aggregate": agg, "per_example": per_example}, open(out, "w"))
    print(f"done {ds}", flush=True)


if __name__ == "__main__":
    main()
