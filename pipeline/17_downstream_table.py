"""
Build the downstream-utility table (tab:downstream_utility) from
final_data/<ds>/downstream_eval_<name>_seed<S>.json and compare with the paper.

Delta (x100) = metric when training on rewrites - metric when training on originals,
tested on original test posts. Semantic column: all-mpnet-base-v2 embeddings; Style
column: StyleDistance embeddings. Metric: macro-F1 (classification) or R^2 (regression).
Labels used (as in the paper): og_students classification_1 (Bloom, 6-class),
stanford_mooc classification_2 (rating, Ridge), stack_exchange department (subject, 10 sites; files tagged 'subject_').
Mean +/- population std over available seeds.

Usage: python 17_downstream_table.py   -> prints table + paper comparison, writes audit/downstream_table.json\n       python 17_downstream_table.py --finetune   -> fine-tuned DistilBERT table (18_downstream_finetune.py)
"""

import json
import os
import re

import numpy as np

PIPE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(PIPE, "final_data")
PAPER = os.path.join(PIPE, "..", "seams_paper", "latex", "acl_latex.tex")
SEEDS = [0, 7, 42, 99, 123]
# (dataset, paper name, label column, metric, output-file tag)
DATASETS = [("og_students", "Proprietary", "classification_1", "macro_f1", ""),
            ("stanford_mooc", "Stanford MOOC", "classification_2", "r2", ""),
            ("stack_exchange", "Stack Exchange", "department", "macro_f1", "subject_")]
METHODS = {"SEAMS": "dpiclmeta", "SEAMS-1": "seams_m1", "ESA": "esa_zscs", "CS": "clustermeta",
           "DP-Prompt": "dp_prompt", "ZS-CS": "baseline_br", "ZS-C": "baseline_b"}


APPENDIX = os.path.join(PIPE, "..", "seams_paper", "latex", "appendix.tex")


def paper_rows(label):
    """{method: [(mean, std), ...]} for each method row of the table with this label (either .tex file)."""
    for f in (PAPER, APPENDIX):
        tex = "\n".join(l for l in open(f).read().split("\n") if not l.lstrip().startswith("%"))
        if r"\label{" + label + "}" in tex:
            i = tex.index(r"\label{" + label + "}")
            t = tex[tex.rfind(r"\begin{table", 0, i):i]
            rows = {}
            for line in t.split("\\\\"):  # one table row per \\ (rows may span source lines)
                name = line.split("&")[0].split("\n")[-1].replace("\\quad", "").strip()
                if "&" in line and name in METHODS:
                    rows[name] = [(float(m), float(sd)) for m, sd in
                                  re.findall(r"([+-]?\d+\.\d+)\s*\\pm\s*(\d+\.\d+)", line)]
            return rows
    return {}


def check(name, got, paper_vals):
    """got: list of (mean, std) in table column order."""
    if not paper_vals:
        return "(not in paper)"
    ok = len(got) == len(paper_vals) and all(abs(a - c) < 0.06 and abs(b - d) < 0.06
                                             for (a, b), (c, d) in zip(got, paper_vals))
    return "match" if ok else f"MISMATCH paper={paper_vals}"


def finetune_table():
    """Fine-tuned DistilBERT deltas (18_downstream_finetune.py): one column per dataset."""
    out = {}
    for ds, pname, label, metric, tag in DATASETS[:3]:
        out[pname] = {}
        for meth, fname in METHODS.items():
            d, lvl = [], []
            for s in SEEDS:
                p = os.path.join(FD, ds, f"downstream_ft_{tag}{fname}_seed{s}.json")
                if not os.path.exists(p):
                    continue
                r = json.load(open(p))["results"][label]
                lvl.append(100 * r["train_on_originals"][metric])
                d.append(100 * (r["train_on_rewrites"][metric] - r["train_on_originals"][metric]))
            if d:
                out[pname][meth] = {"delta": [float(np.mean(d)), float(np.std(d))], "n": len(d),
                                    "orig_level": float(np.mean(lvl))}
    for pname, rows in out.items():
        if rows:
            print(f"train-on-originals level x100 [{pname}]: {next(iter(rows.values()))['orig_level']:.1f}")
    print(f"{'method':10s} " + " ".join(f"{p:>22s}" for p in out))
    for meth in METHODS:
        cells = [f"{out[p][meth]['delta'][0]:+6.1f} ± {out[p][meth]['delta'][1]:4.1f} (n={out[p][meth]['n']})"
                 if meth in out[p] else "-" for p in out]
        print(f"{meth:10s} " + " ".join(f"{c:>22s}" for c in cells))
    paper = paper_rows("tab:downstream_utility")
    print("\npaper check (tab:downstream_utility):")
    for meth in METHODS:
        got = [tuple(out[p][meth]["delta"]) for p in out if meth in out[p]]
        print(f"  {meth:10s} {check(meth, got, paper.get(meth))}")
    os.makedirs(os.path.join(PIPE, "audit"), exist_ok=True)
    json.dump(out, open(os.path.join(PIPE, "audit", "downstream_finetune_table.json"), "w"), indent=1)


def main():
    if "--finetune" in __import__("sys").argv:
        return finetune_table()
    out = {}
    for ds, pname, label, metric, tag in DATASETS:
        out[pname] = {}
        for meth, fname in METHODS.items():
            sem, sty, o_sem, o_sty = [], [], [], []
            for s in SEEDS:
                p = os.path.join(FD, ds, f"downstream_eval_{tag}{fname}_seed{s}.json")
                if not os.path.exists(p):
                    continue
                res = json.load(open(p))["results"]
                if label not in res:
                    continue
                r = res[label]
                o_sem.append(100 * r["train_on_originals"][metric])
                o_sty.append(100 * r["style_train_on_originals"][metric])
                sem.append(100 * (r["train_on_rewrites"][metric] - r["train_on_originals"][metric]))
                sty.append(100 * (r["style_train_on_rewrites"][metric] - r["style_train_on_originals"][metric]))
            if sem:
                out[pname][meth] = {"sem": [float(np.mean(sem)), float(np.std(sem))],
                                    "sty": [float(np.mean(sty)), float(np.std(sty))], "n": len(sem),
                                    "orig_level": [float(np.mean(o_sem)), float(np.mean(o_sty))]}

    paper = paper_rows("tab:downstream_probe")
    print(f"{'dataset':26s} {'method':10s} {'n':>2s} {'semantic':>14s} {'style':>16s}")
    for pname, rows in out.items():
        for meth, r in rows.items():
            print(f"{pname:26s} {meth:10s} {r['n']:2d} {r['sem'][0]:+6.1f} ± {r['sem'][1]:4.1f} "
                  f"{r['sty'][0]:+7.1f} ± {r['sty'][1]:5.1f}")
    print("\npaper check (tab:downstream_probe):")
    for meth in METHODS:
        got = []
        for pname in ["Proprietary", "Stanford MOOC", "Stack Exchange"]:
            if meth in out[pname]:
                got += [tuple(out[pname][meth]["sem"]), tuple(out[pname][meth]["sty"])]
        print(f"  {meth:10s} {check(meth, got, paper.get(meth))}")
    os.makedirs(os.path.join(PIPE, "audit"), exist_ok=True)
    json.dump(out, open(os.path.join(PIPE, "audit", "downstream_table.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
