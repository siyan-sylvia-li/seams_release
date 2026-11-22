"""
Style-fidelity vs. attribution-risk trade-off figure (proprietary dataset).

x: style fidelity (StyleDistance cos to original, x100), y: attack MRR ratio.
Main figure: two panels — a style attacker (Wegmann retrieval) and the strongest
attacker, LUAR (authorship representations; 19_luar_attack.py). --all-attackers draws all five.
Points: 5-seed mean, error bars = population std over seeds (as in the tables).
SEAMS is the only colored series; other methods are gray and directly labeled;
real posts are a dashed reference line (they have style fidelity 100 by definition).

Usage: python 16_tradeoff_figure.py [--all-attackers] [--out figures/tradeoff]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PIPE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(PIPE, "final_data", "og_students")
SEEDS = [0, 7, 42, 99, 123]
AUTH = os.path.join(B, "authorship_v2b_outputs")
AUTH_LUAR = os.path.join(B, "authorship_luar_outputs")
AUTH_GLOBAL = os.path.join(B, "authorship_global_outputs")
GLOBAL = False  # --global: LUAR and Weg-ret from the cluster-agnostic attack
XMIN = 45  # --xmin

# label -> (rewrite file template, config, attack file prefix)
METHODS = {
    "SEAMS":     ("dp_icl_outputs/meta/seed_{s}/pass_b_results.json", None, "seams_m5"),
    "SEAMS-1":   ("dp_icl_outputs/meta_m1/seed_{s}/pass_b_results.json", None, "seams_m1"),
    "ESA":       ("dp_icl_outputs/baseline_esa_zscs/seed_{s}/pass_b_results.json", None, "baseline_esa_zscs"),
    "ZS-CS":     ("baselines/seed_{s}/baseline_br/results.json", None, "baseline_br"),
    "DP-Prompt": ("dp_prompt_outputs/seed_{s}/pass_b_results.json", "dp_prompt_t1.0", "dp_prompt"),
}
ATTACKERS = {"wegmann_ret": "Weg-ret (style)", "luar_ret": "LUAR (authorship)",
             "stylo_clf": "Stylo-clf (style)", "wegmann_clf": "Weg-clf (style)", "mpnet_ret": "MPNet-ret (semantic)"}
# label placement: (dx, dy, ha) in points; default is right of the point
LABEL_POS = {"SEAMS": (-6, -9, "right"), "SEAMS-1": (-6, 3, "right"),
             "ZS-CS": (5, 3, "left"), "DP-Prompt": (5, -9, "left"), "ESA": (5, 3, "left")}
# in the crowded low-fidelity corner, place labels at the end of the method's own error bar
BAR_LABEL = {"wegmann_ret": {"ZS-CS": "above", "DP-Prompt": ("below", "right")},
             "mpnet_ret": {"ZS-CS": "below", "DP-Prompt": "above"},
             "luar_ret": {"ZS-CS": ("above", "right"), "DP-Prompt": ("below", "right"), "SEAMS": "below"},
             "wegmann_clf": {"ZS-CS": "above", "DP-Prompt": "below"},
             "stylo_clf": {"ZS-CS": "above", "DP-Prompt": "below"}}

BLUE, GRAY, INK, MUTED, RED = "#2a78d6", "#8a8a85", "#1a1a19", "#5f5f5a", "#e34948"
FS = 6.5  # base font size; set from --width in main()


def per_example(path, cfg=None):
    d = json.load(open(path))
    return d["per_example"] if "per_example" in d else d[cfg or next(iter(d))]["per_example"]


def mean_std(xs):
    return float(np.mean(xs)), float(np.std(xs))


def attack_file(attacker, atk, s):
    if GLOBAL:
        return os.path.join(AUTH_GLOBAL, f"{atk}_seed{s}.json")
    return os.path.join(AUTH_LUAR if attacker == "luar_ret" else AUTH, f"{atk}_seed{s}.json")


def load():
    pts = {}
    for name, (tpl, cfg, atk) in METHODS.items():
        sty, att = [], {a: [] for a in ATTACKERS}
        for s in SEEDS:
            p = os.path.join(B, tpl.format(s=s))
            files = {k: attack_file(k, atk, s) for k in ATTACKERS}
            if not (os.path.exists(p) and all(os.path.exists(f) for f in files.values())):
                continue
            sty.append(100 * np.mean([r["recon_style_sim"] for r in per_example(p, cfg) if r.get("generated")]))
            for k, f in files.items():
                att[k].append(json.load(open(f))["aggregate"][k]["mrr_over_chance"])
        if sty:
            pts[name] = {"sty": mean_std(sty), "n": len(sty), **{k: mean_std(v) for k, v in att.items()}}
    real = {k: mean_std([json.load(open(attack_file(k, "real", s)))["aggregate"][k]["mrr_over_chance"]
                         for s in SEEDS]) for k in ATTACKERS}
    return pts, real


def panel(ax, pts, real, attacker):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#c9c8c0")
    ax.tick_params(colors=MUTED, labelsize=FS - 1)
    rm, _ = real[attacker]
    ax.axhline(rm, color=RED, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    # label in the empty middle of the panel (Sty ~70-90), clear of all error bars
    ax.annotate("Real posts", (0.55, rm), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=FS - 0.5, color=RED, ha="center", va="bottom")
    for name, p in pts.items():
        x, xs = p["sty"]
        y, ys = p[attacker]
        hi = name == "SEAMS"
        c = BLUE if hi else GRAY
        ax.errorbar(x, y, xerr=xs, yerr=ys, fmt="o", color=c, ecolor=c, elinewidth=0.8, capsize=0,
                    markersize=5 if hi else 4, markeredgecolor="white", markeredgewidth=0.8, zorder=4 if hi else 3)
        where = BAR_LABEL.get(attacker, {}).get(name)
        if where:
            where, bha = where if isinstance(where, tuple) else (where, "center")
            yb = y + ys if where == "above" else y - ys
            dxb = {"center": 0, "right": 3, "left": -3}[bha]
            ax.annotate(name, (x, yb), xytext=(dxb, 2 if where == "above" else -2), textcoords="offset points",
                        fontsize=FS - 0.5, color=BLUE if hi else MUTED, ha=bha,
                        va="bottom" if where == "above" else "top", fontweight="bold" if hi else "normal")
        else:
            dx, dy, ha = LABEL_POS.get(name, (5, 3, "left"))
            ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=FS - 0.5,
                        color=BLUE if hi else MUTED, ha=ha, fontweight="bold" if hi else "normal")
    ax.set_title(ATTACKERS[attacker], fontsize=FS + 0.5, color=INK, loc="center", fontweight="bold")
    ax.set_xlim(XMIN, 100)
    ax.margins(y=0.12)
    ax.set_xlabel("Style fidelity (Sty) →", fontsize=FS, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-attackers", action="store_true")
    ap.add_argument("--attackers", default=None, help="comma-separated subset, e.g. luar_ret")
    ap.add_argument("--global", dest="global_attack", action="store_true")
    ap.add_argument("--xmin", type=float, default=45)
    ap.add_argument("--out", default=os.path.join(PIPE, "figures", "tradeoff"))
    ap.add_argument("--width", type=float, default=3.3,
                    help="Figure width in inches: 3.3 = one ACL column, 6.8 = full page (figure*).")
    ap.add_argument("--height", type=float, default=None, help="Figure height in inches (default: by width).")
    args = ap.parse_args()
    global GLOBAL, XMIN
    GLOBAL = args.global_attack
    XMIN = args.xmin
    if GLOBAL:  # the global attack has retrieval attackers only
        for k in [k for k in ATTACKERS if k not in ("luar_ret", "wegmann_ret")]:
            ATTACKERS.pop(k)
    global FS
    FS = 6.5 if args.width < 5 else 8
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pts, real = load()
    missing = [m for m in METHODS if m not in pts]
    if missing:
        print(f"[warn] no complete data yet for: {missing}")
    attackers = list(ATTACKERS) if args.all_attackers else (args.attackers.split(",") if args.attackers else ["wegmann_ret", "luar_ret"])
    ncol = min(2, len(attackers))
    nrow = int(np.ceil(len(attackers) / ncol))
    h = args.height or (1.9 if args.width < 5 else 2.3) * nrow + 0.3
    fig, axes = plt.subplots(nrow, ncol, figsize=(args.width, h), squeeze=False)
    for ax, a in zip(axes.flat, attackers):
        panel(ax, pts, real, a)
    for ax in axes[:, 0]:
        ax.set_ylabel("MRR ratio (↓ better)", fontsize=FS, color=MUTED)
    fig.tight_layout(pad=0.4, w_pad=1.0 if args.width < 5 else 2.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300 if ext == "png" else None)
    json.dump({"points": pts, "real": real}, open(f"{args.out}.json", "w"), indent=1)
    print(f"wrote {args.out}.pdf/.png/.json; methods: {list(pts)}")


if __name__ == "__main__":
    main()
