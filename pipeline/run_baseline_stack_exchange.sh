#!/usr/bin/env bash
# Baseline eval pipeline for stack_exchange.
# Mirrors run_baseline_og.sh — runs baseline_br and baseline_b per seed,
# then aggregates quality metrics.
#
# Usage:
#   ./run_baseline_stack_exchange.sh
#   SEEDS="42 123" ./run_baseline_stack_exchange.sh
#   EVAL_SPLIT=test ./run_baseline_stack_exchange.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$HOME/.dspy_cache}"
export DSPY_CACHE_LIMIT=30000000000  # 30 GB

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL="${MODEL:-openai/Qwen/Qwen2.5-72B-Instruct}"
API_BASE="${API_BASE:-http://localhost:8001/v1}"

SEEDS="${SEEDS:-123 42 7 0 99}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"

EVAL_ROOT="final_data/stack_exchange"
STUDENT_DATA="final_data/stack_exchange_v2.csv"
BASELINES_ROOT="${EVAL_ROOT}/baselines"

for SEED in $SEEDS; do
    echo ""
    echo "============================================================"
    echo "Seed $SEED"
    echo "============================================================"

    SPLITS_FILE="${EVAL_ROOT}/eval_splits_seed${SEED}.json"
    OUTPUT_DIR="${BASELINES_ROOT}/seed_${SEED}"

    echo "--- Baselines (seed=$SEED) ---"
    python 05_baseline.py \
        --student-data "$STUDENT_DATA" \
        --splits "$SPLITS_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --workers 32 \
            --model "$MODEL" --api-base "$API_BASE"

done

echo ""
echo "--- Aggregate quality metrics: baseline_br ---"
python 06_aggregate_quality_metrics.py \
    --inputs \
        "${BASELINES_ROOT}/seed_123/baseline_br/results.json" \
        "${BASELINES_ROOT}/seed_42/baseline_br/results.json" \
        "${BASELINES_ROOT}/seed_7/baseline_br/results.json" \
        "${BASELINES_ROOT}/seed_0/baseline_br/results.json" \
        "${BASELINES_ROOT}/seed_99/baseline_br/results.json" \
    --student-data "$STUDENT_DATA" \
    --output "${BASELINES_ROOT}/quality_metrics_baseline_br.json"

echo ""
echo "--- Aggregate quality metrics: baseline_b ---"
python 06_aggregate_quality_metrics.py \
    --inputs \
        "${BASELINES_ROOT}/seed_123/baseline_b/results.json" \
        "${BASELINES_ROOT}/seed_42/baseline_b/results.json" \
        "${BASELINES_ROOT}/seed_7/baseline_b/results.json" \
        "${BASELINES_ROOT}/seed_0/baseline_b/results.json" \
        "${BASELINES_ROOT}/seed_99/baseline_b/results.json" \
    --student-data "$STUDENT_DATA" \
    --output "${BASELINES_ROOT}/quality_metrics_baseline_b.json"

echo ""
echo "Done. Baselines written to ${BASELINES_ROOT}/"
