"""
Sweep ABTT-D and max-k to find the combination that maximises multi-seed
meta-clusters while maximising the total number of clusters.

Run as:
    python 04b_sweep_meta_params.py
    python 04b_sweep_meta_params.py --abtt-d 3 5 7 10 --max-k-values 10 15 20
    python 04b_sweep_meta_params.py --output sweep_results.json
"""

import argparse
import importlib.util
import json
import pathlib

import dotenv
import numpy as np

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Import from the numeric-prefixed main script
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "derive_meta", _HERE / "04_derive_meta_clusters.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_cluster_centers = _mod.load_cluster_centers
abtt                 = _mod.abtt
_cluster_stats       = _mod._cluster_stats
best_k_by_objective  = _mod.best_k_by_objective
SEEDS                = _mod.SEEDS
OPTIMAL              = _mod.OPTIMAL
ABTT_D_VALUES        = _mod.ABTT_D_VALUES


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sweep ABTT-D and max-k to maximise multi-seed meta-clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clustering-root",
        default="final_data/stack_exchange/hdbscan_results/",
    )
    parser.add_argument("--axes",         nargs="+", default=list(OPTIMAL.keys()))
    parser.add_argument("--seeds",        nargs="+", default=SEEDS)
    parser.add_argument("--winners-file", default=None)
    parser.add_argument(
        "--abtt-d", nargs="+", type=int, default=ABTT_D_VALUES, dest="abtt_d",
    )
    parser.add_argument(
        "--max-k-values", nargs="+", type=int, default=[10, 15, 20, 25, 30],
        help="max-k values to sweep.",
    )
    parser.add_argument("--min-seeds-multi", type=int, default=3)
    parser.add_argument("--top-n",           type=int, default=10)
    parser.add_argument("--output",          default=None)
    args = parser.parse_args()

    optimal = OPTIMAL.copy()
    if args.winners_file:
        with open(args.winners_file) as f:
            optimal.update(json.load(f))

    all_results: dict = {}

    for axis in args.axes:
        hdbscan_key = optimal.get(axis)
        if hdbscan_key is None:
            print(f"\n[skip] {axis}: no HDBSCAN key")
            continue

        print(f"\n{'='*60}")
        print(f"Axis: {axis}  (hdbscan_key={hdbscan_key})")
        print(f"{'='*60}")

        nodes = load_cluster_centers(axis, hdbscan_key, args.seeds, args.clustering_root)
        if len(nodes) < 4:
            print(f"  [skip] too few clusters ({len(nodes)})")
            continue
        centers = np.stack([n["center"] for n in nodes])
        print(f"  {len(nodes)} cluster centers from {len(args.seeds)} seeds")

        n_combos = len(args.abtt_d) * len(args.max_k_values)
        print(f"  Sweeping {n_combos} combinations (D × max_k)...")

        axis_results = []
        for D in args.abtt_d:
            if D >= len(centers):
                continue
            emb_abtt = abtt(centers, n_components=D)
            for max_k in args.max_k_values:
                labels, k, sil = best_k_by_objective(
                    emb_abtt, nodes, max_k, args.min_seeds_multi,
                )
                n_multi, n_clusters, n_singletons = _cluster_stats(
                    labels, nodes, args.min_seeds_multi,
                )
                axis_results.append({
                    "D":          D,
                    "max_k":      max_k,
                    "k":          k,
                    "silhouette": round(sil, 4),
                    "n_clusters":   n_clusters,
                    "n_multi_seed": n_multi,
                    "n_singletons": n_singletons,
                })

        axis_results.sort(
            key=lambda x: (-x["n_multi_seed"], -x["n_clusters"], -x["silhouette"])
        )

        header = (f"  {'D':<5} {'max_k':<7} {'k':<5} {'sil':<7} "
                  f"{'clusters':<10} {'multi_seed':<12} singletons")
        print(f"\n  Top {args.top_n} (min_seeds≥{args.min_seeds_multi}):")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in axis_results[:args.top_n]:
            marker = " ★" if r is axis_results[0] else ""
            print(f"  {r['D']:<5} {r['max_k']:<7} {r['k']:<5} {r['silhouette']:<7} "
                  f"{r['n_clusters']:<10} {r['n_multi_seed']:<12} {r['n_singletons']}{marker}")

        if axis_results:
            best = axis_results[0]
            print(f"\n  Best config: --abtt-d {best['D']} --max-k {best['max_k']}")

        all_results[axis] = axis_results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
