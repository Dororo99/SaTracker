#!/bin/bash
# SatMapTracker Stage1 BEV Pretrain with Skeleton-Recall Loss
#
# Usage:
#   bash scripts/train_satmaptracker_stage1_skeleton.sh                    # default (skeleton ON)
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --no-skeleton      # skeleton OFF
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --skel-weight 2.0  # custom weight
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --gpus 4,5,6,7     # custom GPUs
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --wandb-name my_exp # custom wandb name

set -e

# ============================================================
# Default settings
# ============================================================
GPUS="2,3"
NUM_GPUS=2
MASTER_PORT=29570
USE_SKELETON=true
SKEL_WEIGHT=1.0
SKEL_CLASSES="[1,2]"
BEV_VIS_INTERVAL=500
WANDB_ENTITY="IRCV_Mapping"
WANDB_PROJECT="Third-SatMAE_MapTracker-AID4AD-seonghyun"
WANDB_NAME="Third-SatMAE_MapTracker_sig_skeleton-AID4AD-dohyun"
CONFIG="plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py"

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
            shift 2 ;;
        --port)
            MASTER_PORT="$2"
            shift 2 ;;
        --no-skeleton)
            USE_SKELETON=false
            shift ;;
        --skel-weight)
            SKEL_WEIGHT="$2"
            shift 2 ;;
        --bev-vis-interval)
            BEV_VIS_INTERVAL="$2"
            shift 2 ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift 2 ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2 ;;
        --config)
            CONFIG="$2"
            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

# ============================================================
# Build cfg-options string
# ============================================================
CFG_OPTIONS=""

# Skeleton loss control
if [ "$USE_SKELETON" = true ]; then
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel.type=SkelRecallLoss"
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel.loss_weight=${SKEL_WEIGHT}"
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel.skel_classes=${SKEL_CLASSES}"
    echo "[Config] Skeleton-Recall Loss: ON (weight=${SKEL_WEIGHT}, classes=${SKEL_CLASSES})"
else
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel=None"
    echo "[Config] Skeleton-Recall Loss: OFF"
fi

# WandB settings
CFG_OPTIONS+=" --cfg-options log_config.hooks.1.init_kwargs.entity=${WANDB_ENTITY}"
CFG_OPTIONS+=" --cfg-options log_config.hooks.1.init_kwargs.project=${WANDB_PROJECT}"
CFG_OPTIONS+=" --cfg-options log_config.hooks.1.init_kwargs.name=${WANDB_NAME}"

# BEV visualization interval
CFG_OPTIONS+=" --cfg-options custom_hooks.0.interval=${BEV_VIS_INTERVAL}"

# ============================================================
# Print settings
# ============================================================
echo "============================================"
echo " SatMapTracker Stage1 BEV Pretrain"
echo "============================================"
echo " GPUs:          ${GPUS} (${NUM_GPUS} devices)"
echo " Master Port:   ${MASTER_PORT}"
echo " Config:        ${CONFIG}"
echo " WandB:         ${WANDB_PROJECT} / ${WANDB_NAME}"
echo " BEV Vis:       every ${BEV_VIS_INTERVAL} iters"
echo "============================================"
echo ""

# ============================================================
# Run training
# ============================================================
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

cd $(dirname $(dirname $(realpath $0)))

/venv/maptracker/bin/python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    ${CFG_OPTIONS}
