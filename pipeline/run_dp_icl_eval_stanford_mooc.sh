#!/usr/bin/env bash
# DP-ICL eval pipeline for stanford_mooc.
# Mirrors run_dp_icl_eval_og.sh.
#
# Usage:
#   ./run_dp_icl_eval_stanford_mooc.sh
#   SEEDS="42 123" ./run_dp_icl_eval_stanford_mooc.sh
#   EVAL_SPLIT=test ./run_dp_icl_eval_stanford_mooc.sh
#   NOISE_SCALES="0.0 0.05 0.1 0.2 0.5 1.0" ./run_dp_icl_eval_stanford_mooc.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$HOME/.dspy_cache}"
export DSPY_CACHE_LIMIT=30000000000  # 30 GB

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL="${MODEL:-openai/Qwen/Qwen2.5-72B-Instruct}"
API_BASE="${API_BASE:-http://localhost:8001/v1}"

SEEDS="${SEEDS:-123 42 7 0 99}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
NOISE_SCALES="${NOISE_SCALES:-}"

EVAL_ROOT="final_data/stanford_mooc"
STUDENT_DATA="final_data/stanford_mooc_v2.csv"
CLUSTERING_ROOT_BASE="${EVAL_ROOT}/hdbscan_results"
META_OUTPUT_DIR="${EVAL_ROOT}/meta_clustering_output"
DP_ICL_ROOT="${EVAL_ROOT}/dp_icl_outputs"

for SEED in $SEEDS; do
    echo ""
    echo "============================================================"
    echo "Seed $SEED"
    echo "============================================================"

    SPLITS_FILE="${EVAL_ROOT}/eval_splits_seed${SEED}.json"
    CLUSTERING_ROOT="${CLUSTERING_ROOT_BASE}/seed_${SEED}"

    PER_SEED_OUTPUT_DIR="${DP_ICL_ROOT}/per_seed/seed_${SEED}"
    mkdir -p "$PER_SEED_OUTPUT_DIR"

    NOISE_SCALES_FLAG=""
    if [ -n "$NOISE_SCALES" ]; then
        NOISE_SCALES_FLAG="--noise-scales $NOISE_SCALES"
    fi

    echo "--- Per-seed DP-ICL eval (seed=$SEED) ---"
    python 05_dp_icl_config_eval.py \
        --student-data "$STUDENT_DATA" \
        --splits "$SPLITS_FILE" \
        --clustering-root "$CLUSTERING_ROOT" \
        --output-dir "$PER_SEED_OUTPUT_DIR" \
        --eval-split "$EVAL_SPLIT" \
        --axes og_style_model \
        --student-id-col student_id \
        --seed "$SEED" \
        $NOISE_SCALES_FLAG \
        --pass-a-dp-m 1 \
        --workers 32 \
            --model "$MODEL" --api-base "$API_BASE"

    META_SEED_OUTPUT_DIR="${DP_ICL_ROOT}/meta/seed_${SEED}"
    mkdir -p "$META_SEED_OUTPUT_DIR"

    echo "--- Meta-cluster DP-ICL eval (seed=$SEED) ---"
    python 05_dp_icl_config_eval.py \
        --student-data "$STUDENT_DATA" \
        --splits "$SPLITS_FILE" \
        --clustering-root "$CLUSTERING_ROOT" \
        --meta-output-dir "$META_OUTPUT_DIR" \
        --output-dir "$META_SEED_OUTPUT_DIR" \
        --eval-split "$EVAL_SPLIT" \
        --axes og_style_model \
        --student-id-col student_id \
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
        f="${DP_ICL_ROOT}/${subdir}/seed_${SEED}/pass_b_results.json"
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
    --output "${DP_ICL_ROOT}/quality_metrics_per_seed.json"

echo ""
echo "--- Aggregate quality metrics: meta ---"
META_INPUTS=$(collect_pass_b_inputs "meta")
python 06_aggregate_quality_metrics.py \
    --inputs $META_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${DP_ICL_ROOT}/quality_metrics_meta.json"

echo ""
echo "Done. Quality metrics written to ${DP_ICL_ROOT}/"
