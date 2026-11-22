#!/usr/bin/env bash
# Cluster-based eval pipeline for stack_exchange.
# Mirrors run_cluster_eval_og.sh.
#
# Usage:
#   ./run_cluster_eval_stack_exchange.sh
#   SEEDS="42 123" ./run_cluster_eval_stack_exchange.sh
#   EVAL_SPLIT=test ./run_cluster_eval_stack_exchange.sh
#   SKIP_PASS_A=1 ./run_cluster_eval_stack_exchange.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$HOME/.dspy_cache}"
export DSPY_CACHE_LIMIT=30000000000  # 30 GB

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL="${MODEL:-openai/Qwen/Qwen2.5-72B-Instruct}"
API_BASE="${API_BASE:-http://localhost:8001/v1}"

SEEDS="${SEEDS:-123 42 7 0 99}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
SKIP_PASS_A="${SKIP_PASS_A:-0}"

EVAL_ROOT="final_data/stack_exchange"
STUDENT_DATA="final_data/stack_exchange_v2.csv"
CLUSTERING_ROOT_BASE="${EVAL_ROOT}/hdbscan_results"
META_OUTPUT_DIR="${EVAL_ROOT}/meta_clustering_output"
CLUSTER_EVAL_ROOT="${EVAL_ROOT}/cluster_eval_outputs"

for SEED in $SEEDS; do
    echo ""
    echo "============================================================"
    echo "Seed $SEED"
    echo "============================================================"

    SPLITS_FILE="${EVAL_ROOT}/eval_splits_seed${SEED}.json"
    CLUSTERING_ROOT="${CLUSTERING_ROOT_BASE}/seed_${SEED}"

    PER_SEED_OUTPUT_DIR="${CLUSTER_EVAL_ROOT}/per_seed/seed_${SEED}"
    mkdir -p "$PER_SEED_OUTPUT_DIR"

    echo "--- Per-seed eval (seed=$SEED) ---"
    PASS_A_FLAG=""
    SWEEP_OUTPUTS_FLAG=""
    if [ "$SKIP_PASS_A" = "1" ]; then
        PASS_A_FLAG="--skip-pass-a"
        SWEEP_OUTPUTS_FLAG="--sweep-outputs-dir ${EVAL_ROOT}/sweep_outputs"
    fi

    python 05_cluster_based_approach_eval.py \
        --student-data "$STUDENT_DATA" \
        --splits "$SPLITS_FILE" \
        --clustering-root "$CLUSTERING_ROOT" \
        --output-dir "$PER_SEED_OUTPUT_DIR" \
        --eval-split "$EVAL_SPLIT" \
        --seed "$SEED" \
        $PASS_A_FLAG $SWEEP_OUTPUTS_FLAG \
        --workers 32 \
            --model "$MODEL" --api-base "$API_BASE"

    META_SEED_OUTPUT_DIR="${CLUSTER_EVAL_ROOT}/meta/seed_${SEED}"
    mkdir -p "$META_SEED_OUTPUT_DIR"

    echo "--- Meta-cluster eval (seed=$SEED) ---"
    python 05_cluster_based_approach_eval.py \
        --student-data "$STUDENT_DATA" \
        --splits "$SPLITS_FILE" \
        --clustering-root "$CLUSTERING_ROOT" \
        --meta-output-dir "$META_OUTPUT_DIR" \
        --output-dir "$META_SEED_OUTPUT_DIR" \
        --eval-split "$EVAL_SPLIT" \
        --seed "$SEED" \
        --use-meta \
        --workers 32 \
            --model "$MODEL" --api-base "$API_BASE"

done

echo ""
echo "All seeds complete: $SEEDS"

collect_pass_b_inputs() {
    local subdir="$1"
    local inputs=""
    for SEED in $SEEDS; do
        f="${CLUSTER_EVAL_ROOT}/${subdir}/seed_${SEED}/pass_b_results.json"
        if [ -f "$f" ]; then
            inputs="$inputs $f"
        else
            echo "[warn] missing: $f" >&2
        fi
    done
    echo "$inputs"
}

echo ""
echo "--- Aggregate quality metrics: per-seed ---"
PER_SEED_INPUTS=$(collect_pass_b_inputs "per_seed")
python 06_aggregate_quality_metrics.py \
    --inputs $PER_SEED_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${CLUSTER_EVAL_ROOT}/quality_metrics_per_seed.json"

echo ""
echo "--- Aggregate quality metrics: meta ---"
META_INPUTS=$(collect_pass_b_inputs "meta")
python 06_aggregate_quality_metrics.py \
    --inputs $META_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${CLUSTER_EVAL_ROOT}/quality_metrics_meta.json"

echo ""
echo "Done. Quality metrics written to ${CLUSTER_EVAL_ROOT}/"
