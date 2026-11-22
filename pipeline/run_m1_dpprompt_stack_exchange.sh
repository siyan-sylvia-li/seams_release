#!/usr/bin/env bash
# run_m1_dpprompt_stack_exchange.sh
# Runs SEAMS M=1 (Pass B only, reusing existing Pass A winners) and DP-Prompt
# for stack_exchange across 5 seeds. Writes quality metrics at the end.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$HOME/.dspy_cache}"
export DSPY_CACHE_LIMIT=30000000000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-openai/Qwen/Qwen2.5-72B-Instruct}"
DP_PROMPT_MODEL="${DP_PROMPT_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
API_BASE="${API_BASE:-http://localhost:8001/v1}"

EVAL_ROOT="final_data/stack_exchange"
STUDENT_DATA="final_data/stack_exchange_v2.csv"
CLUSTERING_ROOT_BASE="${EVAL_ROOT}/hdbscan_results"
META_OUTPUT_DIR="${EVAL_ROOT}/meta_clustering_output"
DP_ICL_ROOT="${EVAL_ROOT}/dp_icl_outputs"
DP_PROMPT_ROOT="${EVAL_ROOT}/dp_prompt_outputs"
M1_ROOT="${DP_ICL_ROOT}/meta_m1"

SEEDS=(0 7 42 99 123)

mkdir -p "$M1_ROOT" "$DP_PROMPT_ROOT"

# ---- SEAMS M=1 ----
echo "=== SEAMS M=1 for stack_exchange ==="
for SEED in "${SEEDS[@]}"; do
    PASS_A_WINNERS="${DP_ICL_ROOT}/meta/seed_${SEED}/pass_a_winners.json"
    OUT_DIR="${M1_ROOT}/seed_${SEED}"
    if [ -f "${OUT_DIR}/pass_b_results.json" ]; then
        echo "[SKIP] M=1 seed=$SEED already done"
        continue
    fi
    if [ ! -f "$PASS_A_WINNERS" ]; then
        echo "[SKIP] M=1 seed=$SEED — pass_a_winners.json not found"
        continue
    fi
    mkdir -p "$OUT_DIR"
    cp "$PASS_A_WINNERS" "$OUT_DIR/pass_a_winners.json"
    echo "--- SEAMS M=1 Pass B (seed=$SEED) ---"
    python 05_dp_icl_config_eval.py \
        --student-data      "$STUDENT_DATA" \
        --splits            "${EVAL_ROOT}/eval_splits_seed${SEED}.json" \
        --clustering-root   "${CLUSTERING_ROOT_BASE}/seed_${SEED}" \
        --meta-output-dir   "$META_OUTPUT_DIR" \
        --output-dir        "$OUT_DIR" \
        --eval-split        val \
        --axes              og_style_model \
        --student-id-col    student_id \
        --seed              "$SEED" \
        --use-meta \
        --dp-m              1 \
        --pass-a-dp-m       1 \
        --workers           32 \
        --model             "$MODEL" \
        --api-base          "$API_BASE"
done

# ---- DP-Prompt ----
echo ""
echo "=== DP-Prompt for stack_exchange ==="
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${DP_PROMPT_ROOT}/seed_${SEED}"
    if [ -f "${OUT_DIR}/pass_b_results.json" ]; then
        echo "[SKIP] DP-Prompt seed=$SEED already done"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "--- DP-Prompt (seed=$SEED) ---"
    python 05_dp_prompt_eval.py \
        --student-data  "$STUDENT_DATA" \
        --splits        "${EVAL_ROOT}/eval_splits_seed${SEED}.json" \
        --eval-split    val \
        --temperatures  1.0 1.25 1.5 1.75 2.0 \
        --model         "$DP_PROMPT_MODEL" \
        --api-base      "$API_BASE" \
        --workers       32 \
        --output-dir    "$OUT_DIR"
done

# ---- Aggregate quality metrics ----
echo ""
echo "=== Aggregate quality metrics: M=1 ==="
M1_INPUTS=""
for SEED in "${SEEDS[@]}"; do
    f="${M1_ROOT}/seed_${SEED}/pass_b_results.json"
    [ -f "$f" ] && M1_INPUTS="$M1_INPUTS $f" || echo "[warn] missing $f" >&2
done
python 06_aggregate_quality_metrics.py \
    --inputs $M1_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${DP_ICL_ROOT}/quality_metrics_meta_m1.json"

echo ""
echo "=== Aggregate quality metrics: DP-Prompt ==="
DP_INPUTS=""
for SEED in "${SEEDS[@]}"; do
    f="${DP_PROMPT_ROOT}/seed_${SEED}/pass_b_results.json"
    [ -f "$f" ] && DP_INPUTS="$DP_INPUTS $f" || echo "[warn] missing $f" >&2
done
python 06_aggregate_quality_metrics.py \
    --inputs $DP_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${DP_PROMPT_ROOT}/quality_metrics.json"

echo ""
echo "=== All done. ==="
