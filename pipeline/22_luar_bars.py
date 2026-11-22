"""
Compact LUAR-only authorship chart: datasets as groups, one bar per method (Real first), MRR ratio
mean +/- population std over 5 seeds. SEAMS blue, Real red, other methods gray (as in the trade-off
figure). All attackers are in the appendix table.

Usage: python 22_luar_bars.py [--out figures/luar_bars] [--width 4.25] [--height 2.1]
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
COLORS = {"real": "#c0392b", "seams_m5": "#2a78d6", "seams_m1": "#8fb8e8", "baseline_br": "#9a9a95",
          "dp_prompt": "#c4c4bf", "baseline_esa_zscs": "#6f6f6a"}
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


SUB = "authorship_luar_outputs"  # set to authorship_global_outputs by --global


def luar(ds, m):
    v = [json.load(open(os.path.join(FD, ds, SUB, f"{m}_seed{s}.json")))["aggregate"]["luar_ret"]
         ["mrr_over_chance"] for s in SEEDS]
    return float(np.mean(v)), float(np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PIPE, "figures", "luar_bars"))
    ap.add_argument("--width", type=float, default=4.25)
    ap.add_argument("--height", type=float, default=2.1)
    ap.add_argument("--fontsize", type=float, default=7.5)
    ap.add_argument("--global", dest="global_attack", action="store_true",
                    help="use the cluster-agnostic attack (23_global_attack.py) instead of within-cluster")
    args = ap.parse_args()
    global SUB
    if args.global_attack:
        SUB = "authorship_global_outputs"
    if os.path.exists(FONT):
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = "Liberation Sans"
    plt.rcParams["font.size"] = args.fontsize

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    n = len(METHODS)
    w = 0.13
    for k, (m, label) in enumerate(METHODS.items()):
        vals = [luar(ds, m) for ds in DATASETS]
        x = np.arange(len(DATASETS)) + (k - (n - 1) / 2) * w + (0.03 if k > 0 else -0.03)
        ax.bar(x, [v[0] for v in vals], w, yerr=[v[1] for v in vals], color=COLORS[m], edgecolor="black",
               linewidth=0.4, error_kw={"elinewidth": 0.7, "capsize": 1.2}, label=label, zorder=3)
    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(DATASETS.values())
    ax.set_ylabel("LUAR MRR ratio ↓")
    ax.yaxis.grid(True, linestyle=":", color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    leg = ax.legend(ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=args.fontsize - 0.5,
                    handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
    leg.get_texts()[0].set_color(COLORS["real"])
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
