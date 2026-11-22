"""
18_downstream_finetune.py

Downstream utility with a fine-tuned classifier instead of frozen-embedding probes
(09_downstream_task_eval.py). Same splits and pairing as 09:

  A: DistilBERT fine-tuned on VAL ORIGINAL posts   (the posts that have a rewrite)
  B: DistilBERT fine-tuned on VAL REWRITES of those posts
  Both tested on TEST ORIGINAL posts.  Delta = B - A.

Fixed hyperparameters for every run (no tuning): distilbert-base-uncased, 3 epochs,
lr 5e-5, batch 16, max length 256, linear schedule with 10% warmup, weight decay 0.01.
Regression targets are standardized on the training set. The training seed is the
data seed, so A and B share initialization and data order.

The model trained on originals depends only on (dataset, seed, training posts), so it is
cached and reused across methods whose rewrites cover the same val posts.

Labels (as reported in the paper): og_students classification_1 (Bloom, macro-F1),
stanford_mooc classification_2 (confusion, R^2), stack_exchange department
(subject, 10 sites, macro-F1; output files tagged 'subject_').

Usage:
  python 18_downstream_finetune.py --dataset stack_exchange [--methods dpiclmeta,...] [--seeds 0,7]
Output: final_data/<ds>/downstream_ft_<method>_seed<S>.json (skips existing)
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PIPE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(PIPE, "final_data")
MODEL = "distilbert-base-uncased"
EPOCHS, LR, BS, MAXLEN, WARMUP, WD = 3, 5e-5, 16, 256, 0.1, 0.01

LABELS = {"og_students": "classification_1", "stanford_mooc": "classification_2",
          "stack_exchange": "department"}
DEFAULT_TAGS = {"stack_exchange": "subject"}  # SE subject files are named downstream_ft_subject_*
# name -> (rewrites path relative to dataset dir, config)
METHODS = {
    "dpiclmeta":   ("dp_icl_outputs/meta/seed_{S}/pass_b_results.json", None),
    "seams_m1":    ("dp_icl_outputs/meta_m1/seed_{S}/pass_b_results.json", None),
    "esa_zscs":    ("dp_icl_outputs/baseline_esa_zscs/seed_{S}/pass_b_results.json", None),
    "clustermeta": ("cluster_eval_outputs/meta/seed_{S}/pass_b_results.json", None),
    "baseline_br": ("baselines/seed_{S}/baseline_br/results.json", None),
    "baseline_b":  ("baselines/seed_{S}/baseline_b/results.json", None),
    "dp_prompt":   ("dp_prompt_outputs/seed_{S}/pass_b_results.json", "dp_prompt_t1.0"),
    "semantic_knn": ("dp_icl_outputs/baseline_semantic_knn/seed_{S}/pass_b_results.json", None),
}


def load_rewrites(path, cfg, df, post_col="original_post"):
    """original text -> rewrite, exactly as in 09_downstream_task_eval.py."""
    d = json.load(open(path))
    ex = d["per_example"] if "per_example" in d else d[cfg or next(iter(d))]["per_example"]
    m = {}
    for e in ex:
        if not e.get("generated"):
            continue
        if e.get("original"):
            m[e["original"]] = e["generated"]
        elif e.get("index") is not None and int(e["index"]) in df.index:
            o = df.at[int(e["index"]), post_col]
            if pd.notna(o):
                m[str(o)] = e["generated"]
    return m


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_and_test(train_texts, y_train, test_texts, y_test, task, n_classes, seed, tok, dev):
    set_seed(seed)
    reg = task == "regression"
    if reg:
        mu, sd = float(np.mean(y_train)), float(np.std(y_train)) or 1.0
        y_train = (np.asarray(y_train) - mu) / sd
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1 if reg else n_classes).to(dev)
    enc = tok(list(train_texts), truncation=True, max_length=MAXLEN)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              "labels": float(y_train[i]) if reg else int(y_train[i])} for i in range(len(train_texts))]
    g = torch.Generator().manual_seed(seed)
    collate = lambda b: tok.pad(b, return_tensors="pt")
    dl = DataLoader(items, batch_size=BS, shuffle=True, generator=g, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    model.train()
    for _ in range(EPOCHS):
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            if reg:
                b["labels"] = b["labels"].float()
            loss = model(**b).loss
            loss.backward()
            opt.step(); sch.step(); opt.zero_grad()
    model.eval()
    preds = []
    tenc = tok(list(test_texts), truncation=True, max_length=MAXLEN)
    titems = [{"input_ids": tenc["input_ids"][i], "attention_mask": tenc["attention_mask"][i]}
              for i in range(len(test_texts))]
    with torch.no_grad():
        for b in DataLoader(titems, batch_size=64, collate_fn=collate):
            out = model(**{k: v.to(dev) for k, v in b.items()}).logits
            preds.append(out.squeeze(-1).cpu().numpy() if reg else out.argmax(-1).cpu().numpy())
    p = np.concatenate(preds)
    del model
    torch.cuda.empty_cache()
    if reg:
        p = p * sd + mu
        return {"r2": round(float(r2_score(y_test, p)), 4), "mae": round(float(mean_absolute_error(y_test, p)), 4),
                "n": len(y_test)}
    return {"accuracy": round(float(accuracy_score(y_test, p)), 4),
            "macro_f1": round(float(f1_score(y_test, p, average="macro", zero_division=0)), 4), "n": len(y_test)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LABELS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="0,7,42,99,123")
    ap.add_argument("--label", default=None, help="label column (default: the paper's label for the dataset)")
    ap.add_argument("--tag", default="", help="output name tag, e.g. 'subject' -> downstream_ft_subject_<method>_seed<S>.json")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds, label = args.dataset, args.label or LABELS[args.dataset]
    tag_name = args.tag or (DEFAULT_TAGS.get(ds, "") if not args.label else "")
    tag = f"{tag_name}_" if tag_name else ""
    df = pd.read_csv(os.path.join(FD, f"{ds}_v2.csv")).reset_index(drop=True)
    lab = df[label].dropna()
    task = "binary" if set(lab.unique()) <= {True, False, 1, 0, "True", "False"} else (
        "regression" if pd.api.types.is_numeric_dtype(lab) else "multiclass")
    le = None
    if task != "regression":
        le = LabelEncoder().fit(lab.astype(str))
    tok = AutoTokenizer.from_pretrained(MODEL)
    cache_dir = os.path.join(FD, ds, f"downstream_ft_{tag}orig_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for S in [int(s) for s in args.seeds.split(",")]:
        splits = json.load(open(os.path.join(FD, ds, f"eval_splits_seed{S}.json")))
        val = df.loc[splits["val"]].reset_index(drop=True)
        test = df.loc[splits["test"]].reset_index(drop=True)
        test = test[test[label].notna()].reset_index(drop=True)
        y_test = test[label].astype(float).values if task == "regression" else le.transform(test[label].astype(str))
        for name in args.methods.split(","):
            out = os.path.join(FD, ds, f"downstream_ft_{tag}{name}_seed{S}.json")
            if os.path.exists(out):
                continue
            tpl, cfg = METHODS[name]
            rw_path = os.path.join(FD, ds, tpl.format(S=S))
            if not os.path.exists(rw_path):
                print(f"[missing rewrites] {ds} {name} seed {S}", flush=True)
                continue
            rmap = load_rewrites(rw_path, cfg, df)
            v = val.copy()
            v["_rw"] = v["original_post"].astype(str).map(rmap)
            v = v[v["_rw"].notna() & v[label].notna()].reset_index(drop=True)
            y_tr = v[label].astype(float).values if task == "regression" else le.transform(v[label].astype(str))
            ncls = 1 if task == "regression" else len(le.classes_)

            key = hashlib.md5("|".join(v["original_post"].astype(str)).encode()).hexdigest()[:12]
            cpath = os.path.join(cache_dir, f"seed{S}_{key}.json")
            if os.path.exists(cpath):
                res_o = json.load(open(cpath))
            else:
                res_o = train_and_test(v["original_post"].astype(str), y_tr, test["original_post"].astype(str),
                                       y_test, task, ncls, S, tok, dev)
                json.dump(res_o, open(cpath, "w"))
            res_r = train_and_test(v["_rw"].astype(str), y_tr, test["original_post"].astype(str),
                                   y_test, task, ncls, S, tok, dev)
            m = "r2" if task == "regression" else "macro_f1"
            print(f"{ds} {name} seed {S} n={len(v)}  orig {m}={res_o[m]:.3f}  rw {m}={res_r[m]:.3f}  "
                  f"delta={res_r[m] - res_o[m]:+.3f}", flush=True)
            json.dump({"rewrites_json": rw_path, "model": MODEL,
                       "hparams": {"epochs": EPOCHS, "lr": LR, "batch": BS, "max_len": MAXLEN,
                                   "warmup": WARMUP, "weight_decay": WD},
                       "results": {label: {"task": task, "n_train": len(v),
                                           "train_on_originals": res_o, "train_on_rewrites": res_r}}},
                      open(out, "w"), indent=2)
    print(f"done {ds}", flush=True)


if __name__ == "__main__":
    main()
