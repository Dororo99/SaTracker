#!/usr/bin/env bash
# Run inference/evaluation for a checkpoint, then export metrics json.
#
# Example:
#   bash scripts/eval_satmaptracker_stage1.sh \
#     --config plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py \
#     --checkpoint work_dirs/satmaptracker_stage1_bev_pretrain/iter_1.pth \
#     --gpus 2 \
#     --work-dir work_dirs/eval_a3_b1_seed0 \
#     --output-json work_dirs/eval_a3_b1_seed0/metrics.json \
#     -- --cfg-options model.use_sat_prior=True model.use_sat_backbone_fusion=False

set -euo pipefail

CONFIG="plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py"
CHECKPOINT=""
GPUS=2
WORK_DIR=""
OUTPUT_JSON=""
SEED=0
DETERMINISTIC=true
DRY_RUN=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"; shift 2 ;;
        --checkpoint)
            CHECKPOINT="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --work-dir)
            WORK_DIR="$2"; shift 2 ;;
        --output-json)
            OUTPUT_JSON="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --no-deterministic)
            DETERMINISTIC=false; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break ;;
        *)
            EXTRA_ARGS+=("$1")
            shift ;;
    esac
done

if [ -z "${CHECKPOINT}" ]; then
    echo "Error: --checkpoint is required."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR="${ROOT_DIR}/work_dirs/eval_$(basename "${CHECKPOINT}" .pth)"
fi
if [ -z "${OUTPUT_JSON}" ]; then
    OUTPUT_JSON="${WORK_DIR}/metrics.json"
fi

TEST_CMD=(
    bash "${ROOT_DIR}/tools/dist_test.sh"
    "${CONFIG}"
    "${CHECKPOINT}"
    "${GPUS}"
    --eval
    --eval-options save_semantic=True
    --work-dir "${WORK_DIR}"
    --seed "${SEED}"
)
if [ "${DETERMINISTIC}" = true ]; then
    TEST_CMD+=(--deterministic)
fi
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    TEST_CMD+=("${EXTRA_ARGS[@]}")
fi

EVAL_CMD=(
    python "${SCRIPT_DIR}/evaluate_submission_metrics.py"
    --config "${CONFIG}"
    --result-path "${WORK_DIR}/submission_vector.json"
    --output "${OUTPUT_JSON}"
    --eval-semantic
)

echo "[Eval] Inference command:"
printf '  %q ' "${TEST_CMD[@]}"; echo
echo "[Eval] Metric export command:"
printf '  %q ' "${EVAL_CMD[@]}"; echo

if [ "${DRY_RUN}" = true ]; then
    echo "[Dry Run] Exiting without execution."
    exit 0
fi

cd "${ROOT_DIR}"
"${TEST_CMD[@]}"
"${EVAL_CMD[@]}"
echo "[Eval] Metrics saved to ${OUTPUT_JSON}"
