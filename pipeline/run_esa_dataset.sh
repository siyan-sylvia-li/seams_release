#!/usr/bin/env bash
# run_esa_dataset.sh DATASET
# DP-ICL ESA baseline (ZS-CS public candidates) on any dataset, 5 seeds, then
# quality metrics. Same settings as the SEAMS meta runs (run_dp_icl_eval_*.sh).
# No authorship attacks (the paper evaluates attacks on og_students only;
# use run_esa_og.sh there).
#
#   setsid nohup bash run_esa_dataset.sh stanford_mooc > final_data/stanford_mooc/dp_icl_outputs/baseline_esa_zscs/run.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

DATASET="$1"
export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$HOME/.dspy_cache}"
export DSPY_CACHE_LIMIT=30000000000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-openai/Qwen/Qwen2.5-72B-Instruct}"
API_BASE="${API_BASE:-http://localhost:8001/v1}"
WORKERS="${WORKERS:-16}"
read -r -a SEEDS <<< "${SEEDS:-0 7 42 99 123}"

BASE="final_data/$DATASET"
OUT_ROOT="$BASE/dp_icl_outputs/baseline_esa_zscs"
mkdir -p "$OUT_ROOT"

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="$OUT_ROOT/seed_${SEED}"
    if [ -f "$OUT_DIR/pass_b_results.json" ]; then echo "[SKIP] $DATASET seed=$SEED"; continue; fi
    mkdir -p "$OUT_DIR"
    echo "--- ESA $DATASET (seed=$SEED) ---"
    python 05_dp_icl_esa_selection.py \
        --student-data    "final_data/${DATASET}_v2.csv" \
        --splits          "$BASE/eval_splits_seed${SEED}.json" \
        --clustering-root "$BASE/hdbscan_results/seed_${SEED}" \
        --meta-output-dir "$BASE/meta_clustering_output" \
        --output-dir      "$OUT_DIR" \
        --eval-split      val \
        --axes            og_style_model \
        --student-id-col  student_id \
        --seed            "$SEED" \
        --use-meta \
        --dp-m            5 \
        --public-source   zscs \
        --esa-n-public    5 \
        --workers         "$WORKERS" \
        --model           "$MODEL" \
        --api-base        "$API_BASE"
done

INPUTS=""
for SEED in "${SEEDS[@]}"; do
    f="$OUT_ROOT/seed_${SEED}/pass_b_results.json"
    [ -f "$f" ] && INPUTS="$INPUTS $f"
done
python 06_aggregate_quality_metrics.py --inputs $INPUTS --student-data "final_data/${DATASET}_v2.csv" \
    --output "$BASE/dp_icl_outputs/quality_metrics_baseline_esa_zscs.json"
echo "=== All done ($DATASET) ==="
