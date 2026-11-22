#!/usr/bin/env bash
# run_downstream_subject_se.sh
# Stack Exchange subject classification (10 sites, the `department` column) as a separate
# downstream task; outputs are tagged "subject" so the paper's runs are untouched.
#   fine-tuned DistilBERT : final_data/stack_exchange/downstream_ft_subject_<method>_seed<S>.json  (GPU 0)
#   embedding probes      : final_data/stack_exchange/downstream_eval_subject_<method>_seed<S>.json (GPU 1)
#
#   setsid nohup bash run_downstream_subject_se.sh > downstream_subject_se.log 2>&1 < /dev/null &
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/
set -uo pipefail
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export PYTHONUNBUFFERED=1
DS=stack_exchange

CUDA_VISIBLE_DEVICES=0 python 18_downstream_finetune.py --dataset $DS --label department --tag subject \
    > downstream_subject_se_ft.log 2>&1 &

(
SEEDS=(0 7 42 99 123)
METHODS=(
  "dpiclmeta|dp_icl_outputs/meta/seed_{S}/pass_b_results.json|-"
  "seams_m1|dp_icl_outputs/meta_m1/seed_{S}/pass_b_results.json|-"
  "esa_zscs|dp_icl_outputs/baseline_esa_zscs/seed_{S}/pass_b_results.json|-"
  "clustermeta|cluster_eval_outputs/meta/seed_{S}/pass_b_results.json|-"
  "baseline_br|baselines/seed_{S}/baseline_br/results.json|-"
  "baseline_b|baselines/seed_{S}/baseline_b/results.json|-"
  "dp_prompt|dp_prompt_outputs/seed_{S}/pass_b_results.json|dp_prompt_t1.0"
)
for spec in "${METHODS[@]}"; do
  IFS="|" read -r NAME TPL CFG <<< "$spec"
  for S in "${SEEDS[@]}"; do
    OUT="final_data/$DS/downstream_eval_subject_${NAME}_seed${S}.json"
    [ -f "$OUT" ] && continue
    CFG_ARG=(); [ "$CFG" != "-" ] && CFG_ARG=(--config "$CFG")
    CUDA_VISIBLE_DEVICES=1 python 09_downstream_task_eval.py --student-data final_data/${DS}_v2.csv \
        --rewrites-json "final_data/$DS/${TPL//\{S\}/$S}" --splits final_data/$DS/eval_splits_seed${S}.json \
        "${CFG_ARG[@]}" --labels department --output "$OUT" || echo "[FAILED] probe $NAME $S"
  done
done
) > downstream_subject_se_probe.log 2>&1 &
wait
echo "$(date) subject runs done"
