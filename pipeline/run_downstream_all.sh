#!/usr/bin/env bash
# run_downstream_all.sh
# Downstream utility eval (09_downstream_task_eval.py) for every method x dataset x seed
# whose rewrites exist; skips outputs already present. Embedding models only (no LLM).
# Output: final_data/<ds>/downstream_eval_<name>_seed<S>.json
#
#   CUDA_VISIBLE_DEVICES=3 setsid nohup bash run_downstream_all.sh > downstream_all.log 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export PYTHONUNBUFFERED=1

SEEDS=(0 7 42 99 123)
# name | rewrites path (relative to dataset dir) | config ("-" = none)
METHODS=(
  "dpiclmeta|dp_icl_outputs/meta/seed_{S}/pass_b_results.json|-"
  "seams_m1|dp_icl_outputs/meta_m1/seed_{S}/pass_b_results.json|-"
  "esa_zscs|dp_icl_outputs/baseline_esa_zscs/seed_{S}/pass_b_results.json|-"
  "clustermeta|cluster_eval_outputs/meta/seed_{S}/pass_b_results.json|-"
  "baseline_br|baselines/seed_{S}/baseline_br/results.json|-"
  "baseline_b|baselines/seed_{S}/baseline_b/results.json|-"
  "dp_prompt|dp_prompt_outputs/seed_{S}/pass_b_results.json|dp_prompt_t1.0"
)

# Stack Exchange (subject labels, --labels department) is run by run_downstream_subject_se.sh.
for DS in ${DATASETS:-og_students stanford_mooc}; do
  for spec in "${METHODS[@]}"; do
    IFS="|" read -r NAME TPL CFG <<< "$spec"
    for S in "${SEEDS[@]}"; do
      RW="final_data/$DS/${TPL//\{S\}/$S}"
      OUT="final_data/$DS/downstream_eval_${NAME}_seed${S}.json"
      [ -f "$OUT" ] && continue
      [ -f "$RW" ] || { echo "[missing rewrites] $DS $NAME seed $S"; continue; }
      echo "$(date +%T) $DS $NAME seed $S"
      CFG_ARG=(); [ "$CFG" != "-" ] && CFG_ARG=(--config "$CFG")
      python 09_downstream_task_eval.py --student-data "final_data/${DS}_v2.csv" \
          --rewrites-json "$RW" --splits "final_data/$DS/eval_splits_seed${S}.json" \
          "${CFG_ARG[@]}" --output "$OUT" || echo "[FAILED] $DS $NAME seed $S"
    done
  done
done
echo "$(date) downstream pass done"
