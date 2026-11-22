"""
20_audit_attack_tables.py

Checks every authorship-attack table in the paper against the raw attack outputs
(authorship_v2b_outputs/ for Weg/Stylo/MPNet, authorship_luar_outputs/ for LUAR).
Fast: reads stored aggregates only (no embedding).

Tables: tab:authorship_global (appendix: global attack, 3 dataset blocks x 3 attackers),
tab:authorship_full (appendix: within-cluster, 3 dataset blocks x 5 attackers), tab:ablation-attack
(appendix: proprietary ablations x 5 attackers).

Usage: python 20_audit_attack_tables.py
"""

import json
import os
import re

import numpy as np

PIPE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(PIPE, "final_data")
TEX = [os.path.join(PIPE, "..", "seams_paper", "latex", f) for f in ("acl_latex.tex", "appendix.tex")]
SEEDS = [0, 7, 42, 99, 123]
ROWS = {"Real (unmodified)": "real", "SEAMS": "seams_m5", "SEAMS ($M{=}5$)": "seams_m5", "SEAMS-1": "seams_m1",
        "ESA": "baseline_esa_zscs", "ZS-CS": "baseline_br", "DP-Prompt": "dp_prompt",
        "Unconstrained": "ablation_unconstrained", "Deterministic": "ablation_deterministic",
        "Random": "ablation_random_selection"}
COLS = {"Weg-ret": "wegmann_ret", "Stylo-clf": "stylo_clf", "LUAR": "luar_ret",
        "Weg-clf": "wegmann_clf", "MPNet-ret": "mpnet_ret"}
DS = {"Proprietary": "og_students", "Stanford MOOC": "stanford_mooc", "Stack Exchange": "stack_exchange"}


GLOBAL_LABELS = {"tab:authorship_global"}  # tables drawn from 23/24 (global attack) outputs
CURRENT = {"label": None}


def raw(ds, method, attacker):
    if CURRENT["label"] in GLOBAL_LABELS:
        sub = "authorship_global_outputs"
    else:
        sub = "authorship_luar_outputs" if attacker == "luar_ret" else "authorship_v2b_outputs"
    v = [json.load(open(os.path.join(FD, ds, sub, f"{method}_seed{s}.json")))["aggregate"][attacker]["mrr_over_chance"]
         for s in SEEDS]
    return float(np.mean(v)), float(np.std(v))


def table(label):
    for f in TEX:
        tex = "\n".join(l for l in open(f).read().split("\n") if not l.lstrip().startswith("%"))
        if r"\label{" + label + "}" in tex:
            i = tex.index(r"\label{" + label + "}")
            return tex[tex.rfind(r"\begin{table", 0, i):i]
    raise KeyError(label)


def check(label, layout):
    """layout(cells_header, block) -> list of (dataset, attacker) per numeric cell."""
    CURRENT["label"] = label
    t = table(label)
    header = re.search(r"Method\}?\s*&(.*?)\\\\", t, re.S).group(1)
    attackers = [COLS[c] for c in re.findall(r"(Weg-ret|Stylo-clf|LUAR|Weg-clf|MPNet-ret)", header)]
    block, n_bad, n = None, 0, 0
    for row in t.split("\\\\"):
        m = re.search(r"\\textsc\{(Proprietary|Stanford MOOC|Stack Exchange)\}", row)
        if m and "&" not in row.split(m.group(0))[1][:5]:
            block = DS[m.group(1)]
        name = row.split("&")[0].split("\n")[-1].replace("\\quad", "").strip()
        if "&" not in row or name not in ROWS:
            continue
        nums = [(float(a), float(b)) for a, b in re.findall(r"(\d+\.\d+)\s*\$?\s*\\pm\s*\$?\s*(\d+\.\d+)", row)]
        for (ds, att), (pm, ps) in zip(layout(attackers, block), nums):
            rm, rs = raw(ds, ROWS[name], att)
            n += 1
            if abs(pm - rm) > 0.006 or abs(ps - rs) > 0.006:
                n_bad += 1
                print(f"  MISMATCH {label} {name} {ds} {att}: paper {pm:.2f}±{ps:.2f}  raw {rm:.3f}±{rs:.3f}")
    print(f"{label}: {n} cells checked, {n_bad} mismatches")
    return n_bad


def main():
    bad = 0
    # main-text attack results are a chart (21_authorship_bars.py, drawn directly from the raw outputs)
    bad += check("tab:authorship_global", lambda att, blk: [(blk, a) for a in att])
    bad += check("tab:authorship_full", lambda att, blk: [(blk, a) for a in att])
    bad += check("tab:ablation-attack", lambda att, _: [("og_students", a) for a in att])
    print(f"total mismatches: {bad}")


if __name__ == "__main__":
    main()
