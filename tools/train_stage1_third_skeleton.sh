#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/train_stage1_third_skeleton.sh
#   RESUME=1 bash tools/train_stage1_third_skeleton.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 bash tools/train_stage1_third_skeleton.sh

CONFIG=${CONFIG:-plugin/configs/satmaptracker/nuscenes_newsplit/satmaptracker_stage1_third.py}
WORKDIR=${WORKDIR:-work_dirs/satmaptracker_stage1_third_skel_10_1_1}
GPUS=${GPUS:-2}
PORT=${PORT:-29503}
SEED=${SEED:-0}
RESUME=${RESUME:-0}
DETERMINISTIC=${DETERMINISTIC:-1}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=2,3
fi

EXTRA_ARGS=()
if [[ "${DETERMINISTIC}" == "1" ]]; then
  EXTRA_ARGS+=(--deterministic)
fi

if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume-from "${WORKDIR}/latest.pth")
fi

echo "[train] CONFIG=${CONFIG}"
echo "[train] WORKDIR=${WORKDIR}"
echo "[train] GPUS=${GPUS}"
echo "[train] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train] PORT=${PORT}"
echo "[train] RESUME=${RESUME}"

PORT=${PORT} \
bash tools/dist_train.sh "${CONFIG}" "${GPUS}" \
  --work-dir "${WORKDIR}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

