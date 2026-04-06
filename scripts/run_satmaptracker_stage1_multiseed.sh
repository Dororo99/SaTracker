#!/bin/bash
# Multi-seed runner for SatMapTracker stage1 training script.
#
# Example:
#   bash scripts/run_satmaptracker_stage1_multiseed.sh \
#     --seeds 0,1,2 \
#     --wandb-base sat_a1_prior \
#     --prior-only
#
# Any unknown args are forwarded to:
#   scripts/train_satmaptracker_stage1_skeleton.sh

set -euo pipefail

SEEDS="0,1,2"
WANDB_BASE="SatMapTracker_stage1"
FORWARD_ARGS=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_satmaptracker_stage1_skeleton.sh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"
            shift 2 ;;
        --wandb-base)
            WANDB_BASE="$2"
            shift 2 ;;
        *)
            FORWARD_ARGS+=("$1")
            shift ;;
    esac
done

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

for seed in "${SEED_ARR[@]}"; do
    seed="$(echo "$seed" | xargs)"
    run_name="${WANDB_BASE}_seed${seed}"
    echo "=================================================="
    echo "[MultiSeed] Running seed=${seed} (wandb=${run_name})"
    echo "=================================================="
    bash "${TRAIN_SCRIPT}" \
        --seed "${seed}" \
        --deterministic \
        --wandb-name "${run_name}" \
        "${FORWARD_ARGS[@]}"
done

echo "[MultiSeed] Completed seeds: ${SEEDS}"
