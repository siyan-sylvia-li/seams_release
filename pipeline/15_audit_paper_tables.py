"""
Audit: recompute the paper's result tables from raw per-post outputs and
compare them with the numbers in the LaTeX source.

Recomputed independently (5-seed mean +/- population std, values x100 except
MRR ratios):
  Table 2 (tab:generation_quality), all datasets:
    Sem, Sty    mean of stored per-post recon_semantic_sim / recon_style_sim
    Weg         AnnaWegmann/Style-Embedding cos(generated, original)
    R-1         rouge_score ROUGE-1 F1, stemmed, score(original, generated)
    MATTR, Self-BLEU   06_aggregate_quality_metrics.mean_mattr / compute_self_bleu
  Table 5 (tab:authorship): mrr_over_chance from authorship_v2b_outputs
  Appendix ablation tables (tab:ablation-quality, tab:ablation-attack)

A cell is flagged if |paper - recomputed| exceeds half a unit in the paper's
last printed digit plus 0.05 of slack (rounding).

Usage: CUDA_VISIBLE_DEVICES=3 python 15_audit_paper_tables.py
Writes audit/audit_report.md and audit/recomputed.json
"""

import importlib
import json
import os
import re

import numpy as np
import pandas as pd

PIPE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(PIPE, "..", "seams_paper", "latex")
OUT = os.path.join(PIPE, "audit")
FD = os.path.join(PIPE, "final_data")
SEEDS = [0, 7, 42, 99, 123]
DATASETS = {"Proprietary": "og_students", "Stanford MOOC": "stanford_mooc", "Stack Exchange": "stack_exchange"}

METHODS = {  # paper name -> (path template relative to dataset dir, config)
    "SEAMS":     ("dp_icl_outputs/meta/seed_{s}/pass_b_results.json", None),
    "SEAMS-1":   ("dp_icl_outputs/meta_m1/seed_{s}/pass_b_results.json", None),
    "ESA":       ("dp_icl_outputs/baseline_esa_zscs/seed_{s}/pass_b_results.json", None),  # proprietary only
    "CS":        ("cluster_eval_outputs/meta/seed_{s}/pass_b_results.json", None),
    "DP-Prompt": ("dp_prompt_outputs/seed_{s}/pass_b_results.json", "dp_prompt_t1.0"),
    "ZS-CS":     ("baselines/seed_{s}/baseline_br/results.json", None),
    "ZS-C":      ("baselines/seed_{s}/baseline_b/results.json", None),
}
ABLATIONS = {
    "SEAMS ($M{=}5$)": "dp_icl_outputs/meta/seed_{s}/pass_b_results.json",
    "Unconstrained":   "dp_icl_outputs/ablation_unconstrained/seed_{s}/pass_b_results.json",
    "Deterministic":   "dp_icl_outputs/ablation_deterministic/seed_{s}/pass_b_results.json",
    "Random":          "dp_icl_outputs/ablation_random_selection/seed_{s}/pass_b_results.json",
}
ATTACK_FILES = {"SEAMS": "seams_m5", "SEAMS-1": "seams_m1", "ESA": "baseline_esa_zscs", "ZS-CS": "baseline_br",
                "DP-Prompt": "dp_prompt", "Real (unmodified)": "real",
                "Unconstrained": "ablation_unconstrained", "Deterministic": "ablation_deterministic",
                "Random": "ablation_random_selection"}
ATTACKERS = ["wegmann_clf", "wegmann_ret", "stylo_clf", "mpnet_ret"]


def per_example(path, cfg=None):
    d = json.load(open(path))
    return d["per_example"] if "per_example" in d else d[cfg or next(iter(d))]["per_example"]


class Metrics:
    def __init__(self):
        from rouge_score import rouge_scorer
        from sentence_transformers import SentenceTransformer
        self.agg = importlib.import_module("06_aggregate_quality_metrics")
        self.rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        self.weg = SentenceTransformer("AnnaWegmann/Style-Embedding")

    def seed_metrics(self, recs, originals):
        gens = [r["generated"] for r in recs]
        origs = [originals[r["index"]] for r in recs]
        a = self.weg.encode(gens, batch_size=64, show_progress_bar=False)
        b = self.weg.encode(origs, batch_size=64, show_progress_bar=False)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        return {
            "Sem": float(np.mean([r["recon_semantic_sim"] for r in recs])),
            "Sty": float(np.mean([r["recon_style_sim"] for r in recs])),
            "Weg": float((a * b).sum(1).mean()),
            "R-1": float(np.mean([self.rouge.score(o, g)["rouge1"].fmeasure for o, g in zip(origs, gens)])),
            "MATTR": self.agg.mean_mattr(gens, 50),
            "Self-BLEU": self.agg.compute_self_bleu(gens, 500),
        }


def agg_seeds(per_seed):
    keys = per_seed[0].keys()
    return {k: (100 * float(np.mean([p[k] for p in per_seed])),
                100 * float(np.std([p[k] for p in per_seed])), len(per_seed)) for k in keys}


NUM = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*(?:\$?\\pm\$?|±)\s*(\d+(?:\.\d+)?))?")


def parse_cells(row):
    """Split a LaTeX table row into cells; return list of (mean, std, decimals) or None."""
    cells = [c for c in row.split("&")]
    out = []
    for c in cells:
        c = re.sub(r"\\textbf\{|\\bm\{|\\mathbf\{|\\phantom\{[^}]*\}|[{}$\\]", " ", c)
        c = c.replace("pm", "±")
        m = NUM.search(c)
        if not m or "--" in c:
            out.append(None)
            continue
        dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        out.append((float(m.group(1)), float(m.group(2)) if m.group(2) else None, dec))
    return out


def find_table(tex, label):
    i = tex.index(r"\label{" + label + "}")
    j = tex.rfind(r"\begin{table", 0, i)
    return tex[j:i]


def compare(rows, recomputed, cols, report, title):
    report.append(f"\n## {title}\n\n| Row | Column | Paper | Recomputed | Δ | Flag |\n|---|---|---|---|---|---|")
    n_flag = 0
    for name, cells in rows:
        if name not in recomputed:
            report.append(f"| {name} | — | (row in paper) | not recomputed | | ⚠️ no source |")
            continue
        for col, cell in zip(cols, cells):
            if cell is None or col not in recomputed[name]:
                continue
            m, s, dec = cell
            rm, rs, ns = recomputed[name][col]
            tol = 0.5 * 10 ** (-dec) + 0.05 * (10 ** (-dec) if dec else 1)
            bad = abs(m - rm) > tol or (s is not None and abs(s - rs) > tol + 0.05)
            n_flag += bad
            if bad:
                report.append(f"| {name} | {col} | {m}{'' if s is None else f' ± {s}'} | "
                              f"{rm:.{max(dec,1)}f} ± {rs:.{max(dec,1)}f} (n={ns}) | {m - rm:+.{max(dec,1)}f} | ❌ |")
    report.append(f"\n{n_flag} cell(s) flagged.")
    return n_flag


def main():
    os.makedirs(OUT, exist_ok=True)
    strip = lambda t: "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("%"))  # skip commented-out tables
    tex = strip(open(os.path.join(PAPER, "acl_latex.tex")).read())
    app = strip(open(os.path.join(PAPER, "appendix.tex")).read())
    M = Metrics()
    recomputed, report = {}, ["# Paper table audit", ""]

    # ---- Table 2 ----
    t2 = find_table(tex, "tab:generation_quality")
    cols2 = ["Sem", "Sty", "Weg", "R-1", "MATTR", "Self-BLEU"]
    total = 0
    for ds_name, ds in DATASETS.items():
        df = pd.read_csv(os.path.join(FD, f"{ds}_v2.csv")).reset_index(drop=True)
        originals = df["original_post"].astype(str).to_dict()
        rec = {}
        for meth, (tpl, cfg) in METHODS.items():
            seeds = []
            for s in SEEDS:
                p = os.path.join(FD, ds, tpl.format(s=s))
                if os.path.exists(p):
                    recs = [r for r in per_example(p, cfg) if r.get("generated")]
                    seeds.append(M.seed_metrics(recs, originals))
            if seeds:
                rec[meth] = agg_seeds(seeds)
                print(f"[{ds}] {meth}: {len(seeds)} seeds", flush=True)
        recomputed[ds] = rec
        # paper rows for this dataset block
        block = t2.split(r"\textsc{" + ds_name + "}")[1].split(r"\midrule")[0].split(r"\bottomrule")[0]
        rows = []
        for line in block.split("\n"):
            if "&" not in line:
                continue
            parts = line.split("&")
            name = re.sub(r"\(.*?\)|\$.*?\$", "", parts[1]).strip()
            rows.append((name, parse_cells("&".join(parts[2:]).split("\\\\")[0])))
        total += compare(rows, rec, cols2, report, f"Table 2 — {ds_name}")

    # ---- Table 5 + ablation attack table ----
    AUTH = os.path.join(FD, "og_students", "authorship_v2b_outputs")
    att = {}
    for name, f in ATTACK_FILES.items():
        vals = [json.load(open(os.path.join(AUTH, f"{f}_seed{s}.json")))["aggregate"] for s in SEEDS
                if os.path.exists(os.path.join(AUTH, f"{f}_seed{s}.json"))]
        if vals:
            att[name] = {a: (float(np.mean([v[a]["mrr_over_chance"] for v in vals])),
                             float(np.std([v[a]["mrr_over_chance"] for v in vals])), len(vals)) for a in ATTACKERS}
    att["SEAMS ($M{=}5$)"] = att["SEAMS"]
    recomputed["attacks"] = att
    for label, title in [("tab:authorship", "Table 5 — authorship attacks"),
                         ("tab:ablation-attack", "Appendix — ablation attacks")]:
        src = tex if label in tex else app
        t = find_table(src, label)
        rows = []
        for line in t.split("\n"):
            if "&" in line and "Method" not in line and "Weg" not in line:
                parts = line.split("&")
                rows.append((re.sub(r"\s+", " ", parts[0]).strip(), parse_cells("&".join(parts[1:]).split("\\\\")[0])))
        rows = [(n.replace("\\", "").strip() if "M{=}5" not in n else n.strip(), c) for n, c in rows]
        total += compare(rows, att, ATTACKERS, report, title)

    # ---- Ablation quality table ----
    df = pd.read_csv(os.path.join(FD, "og_students_v2.csv")).reset_index(drop=True)
    originals = df["original_post"].astype(str).to_dict()
    abl = {}
    for name, tpl in ABLATIONS.items():
        seeds = [M.seed_metrics([r for r in per_example(os.path.join(FD, "og_students", tpl.format(s=s))) if r.get("generated")], originals)
                 for s in SEEDS if os.path.exists(os.path.join(FD, "og_students", tpl.format(s=s)))]
        abl[name] = agg_seeds(seeds)
    recomputed["ablations"] = abl
    t = find_table(app, "tab:ablation-quality")
    rows = []
    for line in t.split("\n"):
        if "&" in line and "Method" not in line:
            parts = line.split("&")
            rows.append((parts[0].strip(), parse_cells("&".join(parts[1:]).split("\\\\")[0])))
    q_rows = [(n, c) for n, c in rows if len(c) == 4]
    d_rows = [(n, c) for n, c in rows if len(c) == 2]
    total += compare(q_rows, abl, ["Sem", "Sty", "Weg", "R-1"], report, "Appendix — ablation quality")
    total += compare(d_rows, abl, ["MATTR", "Self-BLEU"], report, "Appendix — ablation diversity")

    report.insert(1, f"**Total flagged cells: {total}**")
    open(os.path.join(OUT, "audit_report.md"), "w").write("\n".join(report) + "\n")
    json.dump(recomputed, open(os.path.join(OUT, "recomputed.json"), "w"), indent=1)
    print(f"total flagged: {total}; wrote {OUT}/audit_report.md")


if __name__ == "__main__":
    main()
