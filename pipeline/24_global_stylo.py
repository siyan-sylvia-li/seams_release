"""
24_global_stylo.py

Stylometric classifier attacker (Stylo-clf) for the global, cluster-agnostic attack
(23_global_attack.py), with GPU-batched scoring.

Training is identical to 10_authorship_attack_v2.py: same pair sampler (5,000 same-author pairs
from non-evaluation authors, random.Random(42)), same features (char 2-4-gram TF-IDF + function
words, punctuation, sentence length, TTR), same StandardScaler + LogisticRegression on
[|v_a - v_b|, cos(v_a, v_b)]. Only scoring is vectorized: the classifier is linear, so ranking by
predicted probability equals ranking by the logit w.(|v_q - v_t| - mu)/sd + w_cos (cos - mu)/sd + b,
computed for all (query, training post) pairs at once. A candidate's score is the max over their posts.

--check reproduces the stored WITHIN-cluster Stylo-clf result for one dataset/method/seed with this
scorer (same home-cluster candidate pools), to confirm the vectorized scorer matches the original.

Usage:
  python 24_global_stylo.py --dataset stack_exchange --check --methods seams_m5 --seeds 0
  python 24_global_stylo.py --dataset stack_exchange
Output: adds "stylo_clf" to final_data/<ds>/authorship_global_outputs/<method>_seed<S>.json
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

PIPE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(PIPE, "final_data")
_spec = importlib.util.spec_from_file_location("atk", os.path.join(PIPE, "10_authorship_attack_v2.py"))
atk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atk)
METHODS = {"real": None,
           "seams_m5": "dp_icl_outputs/meta/seed_{S}/pass_b_results.json",
           "seams_m1": "dp_icl_outputs/meta_m1/seed_{S}/pass_b_results.json",
           "baseline_esa_zscs": "dp_icl_outputs/baseline_esa_zscs/seed_{S}/pass_b_results.json",
           "baseline_br": "baselines/seed_{S}/baseline_br/results.json",
           "dp_prompt": "dp_prompt_outputs/seed_{S}/pass_b_results.json",
           "baseline_semantic_knn": "dp_icl_outputs/baseline_semantic_knn/seed_{S}/pass_b_results.json"}


def fit_stylo(df, splits):
    """Train exactly as 10_authorship_attack_v2.py main() does."""
    rng = random.Random(42)
    eval_sids = set(str(df.iloc[i]["student_id"]) for i in splits["val"])
    train_posts_all = defaultdict(list)
    for i in splits["train"]:
        row = df.iloc[i]
        train_posts_all[str(row["student_id"])].append(str(row["original_post"]))
    pairs, labels = atk.build_train_pairs(train_posts_all, eval_sids, 5000, rng)
    clf = atk.StylometricClassifier()
    all_train_texts = [t for sid, posts in train_posts_all.items() if sid not in eval_sids for t in posts]
    with contextlib.redirect_stdout(io.StringIO()):
        clf.fit(pairs, labels, all_train_texts)
    sc, lr = clf._clf.named_steps["sc"], clf._clf.named_steps["lr"]
    w = lr.coef_[0] / sc.scale_                       # fold the scaler into the linear model
    b = lr.intercept_[0] - float(np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))
    return clf, w, b


def vecs(clf, texts):
    clf.prime_cache(texts)
    return np.stack([clf._vec(t) for t in texts]).astype(np.float32)


def logits(Vq, Vt, w, b, dev, block=4):
    """(queries x training posts) logits of the pair classifier, computed on the GPU in blocks."""
    D = Vq.shape[1]
    wd = torch.tensor(w[:D], dtype=torch.float32, device=dev); wc = float(w[D]); bb = float(b)
    T = torch.tensor(Vt, device=dev)
    Tn = T / T.norm(dim=1, keepdim=True).clamp(min=1e-12)
    out = []
    for i in range(0, len(Vq), block):
        Q = torch.tensor(Vq[i:i + block], device=dev)
        diff = (Q[:, None, :] - T[None, :, :]).abs()          # (B, T, D)
        cos = (Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-12)) @ Tn.T
        out.append((diff @ wd + wc * cos + bb).cpu().numpy())
    return np.concatenate(out)


def rank_metrics(scores_by_author, true_idx):
    true_score = scores_by_author[np.arange(len(true_idx)), true_idx][:, None]
    ranks = (scores_by_author > true_score).sum(1) + 1
    return 1.0 / ranks, ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="0,7,42,99,123")
    ap.add_argument("--check", action="store_true", help="reproduce the stored within-cluster Stylo-clf result")
    args = ap.parse_args()
    ds = args.dataset
    base = os.path.join(FD, ds)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(os.path.join(FD, f"{ds}_v2.csv")).reset_index(drop=True)

    for S in [int(s) for s in args.seeds.split(",")]:
        splits = json.load(open(os.path.join(base, f"eval_splits_seed{S}.json")))
        clf, w, b = fit_stylo(df, splits)
        tr = df.loc[splits["train"]]
        sids = tr["student_id"].astype(str).values
        texts = tr["original_post"].astype(str).tolist()
        order = np.argsort(sids, kind="stable")
        sids, texts = sids[order], [texts[i] for i in order]
        authors, starts = np.unique(sids, return_index=True)
        a_index = {a: i for i, a in enumerate(authors)}
        K = len(authors)
        chance = float(np.sum(1.0 / np.arange(1, K + 1)) / K)
        Vt = vecs(clf, texts)
        print(f"[{ds} seed {S}] stylo features {Vt.shape}, {K} candidate authors", flush=True)

        for name in args.methods.split(","):
            tpl = METHODS[name]
            if tpl is None:
                recs = [{"idx": i, "original": str(df.iloc[i]["original_post"]),
                         "query": str(df.iloc[i]["original_post"])} for i in splits["val"]]
            else:
                recs = atk.load_eval_pairs(os.path.join(base, tpl.format(S=S)), None, df, "original_post", False)

            if args.check:
                stored = json.load(open(os.path.join(base, "authorship_v2b_outputs", f"{name}_seed{S}.json")))
                keep = {e["idx"] for e in stored["per_example"]}
                recs_c = [r for r in recs if r["idx"] in keep]
                L = logits(vecs(clf, [r["query"] for r in recs_c]), Vt, w, b, dev)
                rr, chs = [], []
                with contextlib.redirect_stdout(io.StringIO()):
                    i2m, m2t = atk.load_cluster_map(os.path.join(base, f"dp_icl_outputs/meta/seed_{S}/pass_b_results.json"),
                                                    None, os.path.join(base, "hdbscan_results"),
                                                    os.path.join(base, "meta_clustering_output/og_style_model/merge_map.json"),
                                                    str(S), splits["train"], df, "original_post")
                home = {}
                for c, st in m2t.items():
                    if c < 0:
                        continue
                    for sid, p in st.items():
                        if sid not in home or len(p) > len(m2t[home[sid]].get(sid, [])):
                            home[sid] = c
                text_pos = defaultdict(list)
                for j, t in enumerate(texts):
                    text_pos[t].append(j)
                for q, r in enumerate(recs_c):
                    sid = str(df.iloc[r["idx"]]["student_id"]); cl = m2t[home[sid]]
                    cand = [(c, [p for p in posts if c != sid or p != r["original"]]) for c, posts in cl.items()]
                    cand = [(c, p) for c, p in cand if p]
                    sc = {c: max(L[q, j] for t in p for j in text_pos[t]) for c, p in cand}
                    rank = 1 + sum(v > sc[sid] for c, v in sc.items() if c != sid)
                    rr.append(1 / rank); Kc = len(cand); chs.append(np.sum(1 / np.arange(1, Kc + 1)) / Kc)
                mine = np.mean(rr) / np.mean(chs)
                print(f"  CHECK {name} seed {S}: vectorized within-cluster Stylo-clf = {mine:.4f}  "
                      f"stored = {stored['aggregate']['stylo_clf']['mrr_over_chance']:.4f}", flush=True)
                continue

            out = os.path.join(base, "authorship_global_outputs", f"{name}_seed{S}.json")
            res = json.load(open(out))
            if "stylo_clf" in res["aggregate"]:
                continue
            recs = [r for r in recs if str(df.iloc[r["idx"]]["student_id"]) in a_index]
            true = np.array([a_index[str(df.iloc[r["idx"]]["student_id"])] for r in recs])
            L = logits(vecs(clf, [r["query"] for r in recs]), Vt, w, b, dev)
            per_author = np.maximum.reduceat(L, starts, axis=1)
            rr, ranks = rank_metrics(per_author, true)
            res["aggregate"]["stylo_clf"] = {"n": len(recs), "K": K, "mrr": float(rr.mean()), "chance_mrr": chance,
                                             "mrr_over_chance": float(rr.mean() / chance),
                                             "top1": float((ranks == 1).mean()), "top10": float((ranks <= 10).mean())}
            json.dump(res, open(out, "w"), indent=1)
            print(f"  {name:20s} n={len(recs)}  stylo_clf={rr.mean() / chance:.2f}", flush=True)
    print(f"done {ds}", flush=True)


if __name__ == "__main__":
    main()
