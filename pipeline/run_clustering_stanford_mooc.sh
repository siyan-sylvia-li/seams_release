#!/usr/bin/env bash
# Clustering pipeline for stanford_mooc.
# Mirrors run_clustering_og.sh — runs stages 0-3 per seed, then meta-clustering.
#
# Usage:
#   ./run_clustering_stanford_mooc.sh
#   SEEDS="42 123" ./run_clustering_stanford_mooc.sh
#   EVAL_SPLIT=test ./run_clustering_stanford_mooc.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEEDS="${SEEDS:-123 42 7 0 99}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"

EVAL_ROOT="final_data/stanford_mooc"
STUDENT_DATA="final_data/stanford_mooc_v2.csv"
CLUSTERING_ROOT_BASE="${EVAL_ROOT}/hdbscan_results"
SWEEP_OUTPUTS_DIR="${EVAL_ROOT}/sweep_outputs"
META_OUTPUT_DIR="${EVAL_ROOT}/meta_clustering_output"

mkdir -p "$EVAL_ROOT"

for SEED in $SEEDS; do
    echo ""
    echo "============================================================"
    echo "Seed $SEED"
    echo "============================================================"

    SPLITS_FILE="${EVAL_ROOT}/eval_splits_seed${SEED}.json"
    CLUSTERING_ROOT="${CLUSTERING_ROOT_BASE}/seed_${SEED}"
    OUTPUT_DIR="${SWEEP_OUTPUTS_DIR}/seed_${SEED}"

    echo "--- Stage 0: eval splits (seed=$SEED) ---"
    python 01_prepare_eval_splits.py \
        --input "$STUDENT_DATA" \
        --output "$SPLITS_FILE" \
        --seed "$SEED"

    echo "--- Stage 1: clustering sweep (seed=$SEED, train only) ---"
    python 02_clustering_sweep.py \
        --output-root "$CLUSTERING_ROOT" \
        --splits "$SPLITS_FILE" \
        --student-data "$STUDENT_DATA" \
        --seed "$SEED"

    echo "--- Stage 1b: cluster summaries (seed=$SEED) ---"
    python 03_generate_cluster_summaries.py \
        --student-data "$STUDENT_DATA" \
        --splits "$SPLITS_FILE" \
        --clustering-root "$CLUSTERING_ROOT"

    echo "--- Stage 2: identifying cluster performance (seed=$SEED) ---"
    python 03_pick_best_clustering.py \
        --student-data "$STUDENT_DATA" \
        --splits "$SPLITS_FILE" \
        --clustering-root "$CLUSTERING_ROOT" \
        --output-dir "$OUTPUT_DIR"

done

echo ""
echo "All seeds complete: $SEEDS"

echo ""
echo "--- Stage 3: meta-clustering (across seeds) ---"
python 04_derive_meta_clusters.py \
    --student-data "$STUDENT_DATA" \
    --splits-dir "$EVAL_ROOT" \
    --clustering-root "$CLUSTERING_ROOT_BASE" \
    --sweep-outputs-dir "$SWEEP_OUTPUTS_DIR" \
    --output-dir "$META_OUTPUT_DIR" \
    --multi-seed-only

echo ""
echo "Meta-clustering complete -> $META_OUTPUT_DIR"
