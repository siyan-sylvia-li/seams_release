#!/usr/bin/env bash
# run_authorship_v2b_dataset.sh  (DATASET env; default stanford_mooc)
# Within-cluster authorship attack v2b on any dataset with SEAMS cluster outputs.
# Methods: seams_m5, seams_m1, real, baseline_br (ZS-CS), dp_prompt (t=1.0), baseline_esa_zscs (ESA)
# Select a subset with METHODS="baseline_br dp_prompt" (default: all).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"  # always run from pipeline/

DATASET="${DATASET:-stanford_mooc}"
BASE="final_data/$DATASET"
OUT="$BASE/authorship_v2b_outputs"
STUDENT_DATA="final_data/${DATASET}_v2.csv"
HDBSCAN="$BASE/hdbscan_results"
MERGE_MAP="$BASE/meta_clustering_output/og_style_model/merge_map.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export PYTHONUNBUFFERED=1

SEEDS=(0 7 42 99 123)

mkdir -p "$OUT"

run_one() {
    local METHOD=$1 SEED=$2 EXTRA_REWRITES=$3 EXTRA_ARGS=$4
    local SEAMS_JSON="$BASE/dp_icl_outputs/meta/seed_${SEED}/pass_b_results.json"
    local OUTFILE="$OUT/${METHOD}_seed${SEED}.json"
    if [[ -f "$OUTFILE" ]]; then
        echo "[SKIP] $OUTFILE already exists"
        return 0
    fi
    echo "=== $METHOD seed=$SEED ==="
    python 10_authorship_attack_v2.py \
        --student-data  "$STUDENT_DATA" \
        $EXTRA_REWRITES \
        --cluster-json  "$SEAMS_JSON" \
        --splits        "$BASE/eval_splits_seed${SEED}.json" \
        --hdbscan-root  "$HDBSCAN" \
        --merge-map     "$MERGE_MAP" \
        --cluster-seed  "$SEED" \
        --output        "$OUTFILE" \
        $EXTRA_ARGS
}

read -r -a METHODS <<< "${METHODS:-seams_m5 seams_m1 real baseline_br dp_prompt baseline_esa_zscs}"
for SEED in "${SEEDS[@]}"; do
  for M in "${METHODS[@]}"; do
    case $M in
      seams_m5)          run_one seams_m5 "$SEED" "--rewrites-json $BASE/dp_icl_outputs/meta/seed_${SEED}/pass_b_results.json" "" ;;
      seams_m1)          run_one seams_m1 "$SEED" "--rewrites-json $BASE/dp_icl_outputs/meta_m1/seed_${SEED}/pass_b_results.json" "" ;;
      real)              run_one real "$SEED" "" "--use-original-as-query" ;;
      baseline_br)       run_one baseline_br "$SEED" "--rewrites-json $BASE/baselines/seed_${SEED}/baseline_br/results.json" "" ;;
      dp_prompt)         run_one dp_prompt "$SEED" "--rewrites-json $BASE/dp_prompt_outputs/seed_${SEED}/pass_b_results.json" "" ;;  # first config = t=1.0
      baseline_esa_zscs) run_one baseline_esa_zscs "$SEED" "--rewrites-json $BASE/dp_icl_outputs/baseline_esa_zscs/seed_${SEED}/pass_b_results.json" "" ;;
    esac
  done
done

echo ""
echo "=== All done. Outputs in $OUT ==="
