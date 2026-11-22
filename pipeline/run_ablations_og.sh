#!/usr/bin/env bash
# run_ablations_og.sh
# Runs three ablations on og_students across 5 seeds:
#   1. Unconstrained retrieval  (05_dp_icl_unconstrained_retrieval.py)
#   2. Deterministic selection  (05_dp_icl_deterministic_selection.py)
#   3. Random selection         (05_dp_icl_random_selection.py)
# All reuse OPTIMAL HDBSCAN config via --use-meta (no Pass A sweep).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

export DSPY_CACHEDIR="${DSPY_CACHEDIR:-$HOME/.dspy_cache}"
export DSPY_CACHE_LIMIT=30000000000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

MODEL="${MODEL:-openai/Qwen/Qwen2.5-72B-Instruct}"
API_BASE="${API_BASE:-http://localhost:8001/v1}"

EVAL_ROOT="final_data/og_students"
STUDENT_DATA="final_data/og_students_v2.csv"
CLUSTERING_ROOT_BASE="${EVAL_ROOT}/hdbscan_results"
META_OUTPUT_DIR="${EVAL_ROOT}/meta_clustering_output"
DP_ICL_ROOT="${EVAL_ROOT}/dp_icl_outputs"

UNCONSTRAINED_ROOT="${DP_ICL_ROOT}/ablation_unconstrained"
DETERMINISTIC_ROOT="${DP_ICL_ROOT}/ablation_deterministic"
RANDOM_ROOT="${DP_ICL_ROOT}/ablation_random_selection"

SEEDS=(0 7 42 99 123)

mkdir -p "$UNCONSTRAINED_ROOT" "$DETERMINISTIC_ROOT" "$RANDOM_ROOT"

# ---- Ablation 1: Unconstrained retrieval ----
echo "=== Ablation: Unconstrained Retrieval (og_students) ==="
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${UNCONSTRAINED_ROOT}/seed_${SEED}"
    if [ -f "${OUT_DIR}/pass_b_results.json" ]; then
        echo "[SKIP] unconstrained seed=$SEED already done"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "--- Unconstrained (seed=$SEED) ---"
    python 05_dp_icl_unconstrained_retrieval.py \
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
        --dp-m              5 \
        --pass-a-dp-m       1 \
        --workers           32 \
        --model             "$MODEL" \
        --api-base          "$API_BASE"
done

# ---- Ablation 2: Deterministic selection (no DP noise) ----
echo ""
echo "=== Ablation: Deterministic Selection (og_students) ==="
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${DETERMINISTIC_ROOT}/seed_${SEED}"
    if [ -f "${OUT_DIR}/pass_b_results.json" ]; then
        echo "[SKIP] deterministic seed=$SEED already done"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "--- Deterministic (seed=$SEED) ---"
    python 05_dp_icl_deterministic_selection.py \
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
        --dp-m              5 \
        --pass-a-dp-m       1 \
        --workers           32 \
        --model             "$MODEL" \
        --api-base          "$API_BASE"
done

# ---- Ablation 3: Random selection ----
echo ""
echo "=== Ablation: Random Selection (og_students) ==="
for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${RANDOM_ROOT}/seed_${SEED}"
    if [ -f "${OUT_DIR}/pass_b_results.json" ]; then
        echo "[SKIP] random_selection seed=$SEED already done"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "--- Random Selection (seed=$SEED) ---"
    python 05_dp_icl_random_selection.py \
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
        --dp-m              5 \
        --pass-a-dp-m       1 \
        --workers           32 \
        --model             "$MODEL" \
        --api-base          "$API_BASE"
done

# ---- Aggregate quality metrics ----
echo ""
echo "=== Aggregate quality metrics: Unconstrained ==="
UC_INPUTS=""
for SEED in "${SEEDS[@]}"; do
    f="${UNCONSTRAINED_ROOT}/seed_${SEED}/pass_b_results.json"
    [ -f "$f" ] && UC_INPUTS="$UC_INPUTS $f" || echo "[warn] missing $f" >&2
done
python 06_aggregate_quality_metrics.py \
    --inputs $UC_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${DP_ICL_ROOT}/quality_metrics_ablation_unconstrained.json"

echo ""
echo "=== Aggregate quality metrics: Deterministic ==="
DT_INPUTS=""
for SEED in "${SEEDS[@]}"; do
    f="${DETERMINISTIC_ROOT}/seed_${SEED}/pass_b_results.json"
    [ -f "$f" ] && DT_INPUTS="$DT_INPUTS $f" || echo "[warn] missing $f" >&2
done
python 06_aggregate_quality_metrics.py \
    --inputs $DT_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${DP_ICL_ROOT}/quality_metrics_ablation_deterministic.json"

echo ""
echo "=== Aggregate quality metrics: Random Selection ==="
RS_INPUTS=""
for SEED in "${SEEDS[@]}"; do
    f="${RANDOM_ROOT}/seed_${SEED}/pass_b_results.json"
    [ -f "$f" ] && RS_INPUTS="$RS_INPUTS $f" || echo "[warn] missing $f" >&2
done
python 06_aggregate_quality_metrics.py \
    --inputs $RS_INPUTS \
    --student-data "$STUDENT_DATA" \
    --output "${DP_ICL_ROOT}/quality_metrics_ablation_random_selection.json"

echo ""
echo "=== All done. ==="
