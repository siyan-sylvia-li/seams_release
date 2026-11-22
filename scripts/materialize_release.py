"""Unpack the released rewrites into the pipeline's on-disk layout, so the evaluation
scripts (06_aggregate_quality_metrics, 09/18 downstream, 10/19/23/24 attacks, 15/17/20
table audits) can run on them without regenerating anything.

  python scripts/materialize_release.py [--datasets stack_exchange,stanford_mooc]

Writes pipeline/final_data/<ds>/... with the same file names the run_*.sh scripts
produce. Stack Exchange is fully self-contained. For Stanford MOOC, first run
scripts/rebuild_stanford_csv.py (needs your own copy of MOOCPosts); withheld rewrites
(rows that reproduce a Stanford post, see data/README.md) are skipped, so Stanford
numbers are computed on the released subset and will not match the paper exactly.
"""
import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FD = os.path.join(ROOT, "pipeline", "final_data")
SEEDS = [0, 7, 42, 99, 123]
AXIS = "B::og_style_model"  # result-file key used by the SEAMS scripts
# release method -> (pipeline path template, wrapper)
LAYOUT = {
    "seams":     ("dp_icl_outputs/meta/seed_{s}/pass_b_results.json", "axis"),
    "seams_1":   ("dp_icl_outputs/meta_m1/seed_{s}/pass_b_results.json", "axis"),
    "cs":        ("cluster_eval_outputs/meta/seed_{s}/pass_b_results.json", "axis"),
    "esa":       ("dp_icl_outputs/baseline_esa_zscs/seed_{s}/pass_b_results.json", "axis"),
    "dp_prompt": ("dp_prompt_outputs/seed_{s}/pass_b_results.json", "dp_prompt_t1.0"),
    "zs_cs":     ("baselines/seed_{s}/baseline_br/results.json", "flat"),
    "zs_c":      ("baselines/seed_{s}/baseline_b/results.json", "flat"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="stack_exchange,stanford_mooc")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        src = os.path.join(DATA, ds)
        dst = os.path.join(FD, ds)
        os.makedirs(dst, exist_ok=True)
        if ds == "stack_exchange":
            shutil.copy(os.path.join(src, "stack_exchange_v2.csv"), os.path.join(FD, "stack_exchange_v2.csv"))
        elif not os.path.exists(os.path.join(FD, "stanford_mooc_v2.csv")):
            print("[note] pipeline/final_data/stanford_mooc_v2.csv not found; "
                  "run scripts/rebuild_stanford_csv.py before evaluating Stanford rewrites")
        for s in SEEDS:
            shutil.copy(os.path.join(src, "splits", f"eval_splits_seed{s}.json"),
                        os.path.join(dst, f"eval_splits_seed{s}.json"))
        for m, (tmpl, wrap) in LAYOUT.items():
            for s in SEEDS:
                recs = []
                for line in open(os.path.join(src, "rewrites", m, f"seed_{s}.jsonl")):
                    r = json.loads(line)
                    if r["rewrite"] is None:
                        continue
                    recs.append({"index": r["row_index"], "generated": r["rewrite"],
                                 "recon_semantic_sim": r["recon_semantic_sim"],
                                 "recon_style_sim": r["recon_style_sim"]})
                body = {"per_example": recs}
                out = body if wrap == "flat" else {AXIS if wrap == "axis" else wrap: body}
                p = os.path.join(dst, tmpl.format(s=s))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(out, open(p, "w"), ensure_ascii=False)
            print(f"{ds}/{m}: 5 seeds -> {os.path.join(dst, tmpl.split('/')[0])}")


if __name__ == "__main__":
    main()
