"""
23_global_attack.py

Global (cluster-agnostic) authorship attack: the attacker ranks ALL training authors, with no
knowledge of the true author's style cluster. Complements the within-cluster attack
(10_authorship_attack_v2.py / 19_luar_attack.py), whose candidate pool is the true author's home
cluster.

Queries: rewritten (or, for "real", original) validation posts whose author has >= 1 training post.
Candidates: every author with >= 1 training post; a candidate's score is the max cosine similarity
between the query and any of the candidate's training posts. Attackers: LUAR (luar_ret) and
Wegmann style embedding (wegmann_ret). Metric: MRR / chance-MRR (H_K / K), as in the other scripts.

Usage: python 23_global_attack.py --dataset stack_exchange [--methods ...] [--seeds 0,7]
Output: final_data/<ds>/authorship_global_outputs/<method>_seed<S>.json
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PIPE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(PIPE, "final_data")
_spec = importlib.util.spec_from_file_location("luar", os.path.join(PIPE, "19_luar_attack.py"))
luar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(luar)
atk = luar.atk  # 10_authorship_attack_v2 helpers (load_eval_pairs)

METHODS = {k: v for k, v in luar.METHODS.items() if not k.startswith("ablation_")}


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default="real,seams_m5,seams_m1,baseline_esa_zscs,baseline_br,dp_prompt")
    ap.add_argument("--seeds", default="0,7,42,99,123")
    args = ap.parse_args()
    ds = args.dataset
    base = os.path.join(FD, ds)
    out_dir = os.path.join(base, "authorship_global_outputs")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(FD, f"{ds}_v2.csv")).reset_index(drop=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    luar_emb = luar.LuarEmbedder(dev)
    weg = SentenceTransformer("AnnaWegmann/Style-Embedding", device=dev)

    for S in [int(s) for s in args.seeds.split(",")]:
        splits = json.load(open(os.path.join(base, f"eval_splits_seed{S}.json")))
        tr = df.loc[splits["train"]]
        sids = tr["student_id"].astype(str).values
        texts = tr["original_post"].astype(str).tolist()
        order = np.argsort(sids, kind="stable")
        sids, texts = sids[order], [texts[i] for i in order]
        authors, starts = np.unique(sids, return_index=True)
        a_index = {a: i for i, a in enumerate(authors)}
        K = len(authors)
        chance = float(np.sum(1.0 / np.arange(1, K + 1)) / K)
        print(f"[{ds} seed {S}] {len(texts)} training posts, {K} candidate authors", flush=True)
        T = {"luar_ret": unit([luar_emb.embed(texts)[t] for t in texts]),
             "wegmann_ret": unit(weg.encode(texts, batch_size=64))}

        for name in args.methods.split(","):
            out = os.path.join(out_dir, f"{name}_seed{S}.json")
            if os.path.exists(out):
                continue
            tpl = METHODS[name]
            if tpl is None:
                recs = [{"idx": i, "query": str(df.iloc[i]["original_post"])} for i in splits["val"]]
            else:
                rp = os.path.join(base, tpl.format(S=S))
                if not os.path.exists(rp):
                    continue
                recs = atk.load_eval_pairs(rp, None, df, "original_post", False)
            recs = [r for r in recs if str(df.iloc[r["idx"]]["student_id"]) in a_index]
            true = np.array([a_index[str(df.iloc[r["idx"]]["student_id"])] for r in recs])
            qtexts = [r["query"] for r in recs]
            Q = {"luar_ret": unit([luar_emb.embed(qtexts)[q] for q in qtexts]),
                 "wegmann_ret": unit(weg.encode(qtexts, batch_size=64))}
            agg = {}
            for att in Q:
                sims = Q[att] @ T[att].T                                   # (queries, training posts)
                per_author = np.maximum.reduceat(sims, starts, axis=1)     # max over each author's posts
                true_score = per_author[np.arange(len(recs)), true][:, None]
                ranks = (per_author > true_score).sum(1) + 1               # ties favor the true author
                rr = 1.0 / ranks
                agg[att] = {"n": len(recs), "K": K, "mrr": float(rr.mean()), "chance_mrr": chance,
                            "mrr_over_chance": float(rr.mean() / chance),
                            "top1": float((ranks == 1).mean()), "top10": float((ranks <= 10).mean())}
            print(f"  {name:20s} n={len(recs)}  " + "  ".join(f"{a}={v['mrr_over_chance']:.2f}" for a, v in agg.items()),
                  flush=True)
            json.dump({"dataset": ds, "seed": S, "method": name, "candidates": "all training authors",
                       "aggregate": agg}, open(out, "w"), indent=1)
    print(f"done {ds}", flush=True)


if __name__ == "__main__":
    main()
