"""
Authorship-attack bar chart (main-text figure; the full MRR-ratio table is in the appendix).

One panel per dataset, grouped bars per method (Real posts first), MRR ratio mean +/- population std
over 5 seeds. Weg-ret and Stylo-clf use the left y-axis; LUAR, whose ratios are larger, uses the
right y-axis. Lower is better. Styled to match latex/generation_quality_sem_weg.pdf.

Usage: python 21_authorship_bars.py [--out figures/authorship_bars] [--height 2.6]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

PIPE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(PIPE, "final_data")
SEEDS = [0, 7, 42, 99, 123]
DATASETS = {"og_students": "Proprietary", "stanford_mooc": "Stanford MOOC", "stack_exchange": "Stack Exchange"}
METHODS = {"real": "Real", "seams_m5": "SEAMS", "seams_m1": "SEAMS-1", "baseline_br": "ZS-CS",
           "dp_prompt": "DP-Prompt", "baseline_esa_zscs": "ESA"}
LEFT = {"wegmann_ret": ("Weg-ret", "#4C72B0"), "stylo_clf": ("Stylo-clf", "#DD8452")}
RIGHT = {"luar_ret": ("LUAR (right axis)", "#55A868")}
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
REAL_COLOR = "#c0392b"
RIGHT_AXIS_COLOR = "#1e6b34"  # darker than the LUAR bars, for contrast on white
GAP = 0.6  # extra space between Real and the methods


def ratios(ds, method, attacker):
    sub = "authorship_luar_outputs" if attacker == "luar_ret" else "authorship_v2b_outputs"
    v = [json.load(open(os.path.join(FD, ds, sub, f"{method}_seed{s}.json")))["aggregate"][attacker]["mrr_over_chance"]
         for s in SEEDS]
    return float(np.mean(v)), float(np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PIPE, "figures", "authorship_bars"))
    ap.add_argument("--width", type=float, default=8.0)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--fontsize", type=float, default=10)
    ap.add_argument("--label-rotation", type=float, default=40)
    ap.add_argument("--no-stylo", action="store_true", help="drop Stylo-clf (kept in the appendix table)")
    args = ap.parse_args()
    if args.no_stylo:
        LEFT.pop("stylo_clf", None)
    if os.path.exists(FONT):
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = "Liberation Sans"
    plt.rcParams["font.size"] = args.fontsize

    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height))
    x = np.arange(len(METHODS)) + np.r_[0, np.full(len(METHODS) - 1, GAP)]
    w = 0.26
    err = {"elinewidth": 0.8, "capsize": 1.5}
    out = {}
    for i, (ax, (ds, title)) in enumerate(zip(axes, DATASETS.items())):
        ax2 = ax.twinx()
        for k, (a, (label, c)) in enumerate(list(LEFT.items()) + list(RIGHT.items())):
            vals = [ratios(ds, m, a) for m in METHODS]
            out.update({f"{ds}|{m}|{a}": v for m, v in zip(METHODS, vals)})
            target = ax2 if a in RIGHT else ax
            off = (k - 1) * w if len(LEFT) + len(RIGHT) == 3 else (k - 0.5) * w
            target.bar(x + off, [v[0] for v in vals], w, yerr=[v[1] for v in vals], color=c,
                       edgecolor="black", linewidth=0.5, error_kw=err, label=label, zorder=3)
        ax.set_title(title, fontsize=args.fontsize + 1)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS.values(), rotation=args.label_rotation, ha="right" if args.label_rotation < 90 else "center",
                           rotation_mode="anchor" if args.label_rotation < 90 else "default")
        ax.get_xticklabels()[0].set_color(REAL_COLOR)  # "Real" is the reference, not a method
        ax.axvline(0.5 + GAP / 2, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
        ax.yaxis.grid(True, linestyle=":", color="#bbbbbb", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", colors=RIGHT_AXIS_COLOR)
        if i == 0:
            ax.set_ylabel("MRR ratio")
        if i == 2:
            ax2.set_ylabel("MRR ratio (LUAR)", color=RIGHT_AXIS_COLOR)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[0].figure.axes[3].get_legend_handles_labels()  # first twin axis
    fig.legend(h1 + h2, [f"{s} ↓" for s in l1 + l2], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    json.dump(out, open(f"{args.out}.json", "w"), indent=1)
    print(f"wrote {args.out}.pdf/.png/.json")


if __name__ == "__main__":
    main()
