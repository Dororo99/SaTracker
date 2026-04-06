#!/bin/bash
# SatMapTracker Stage1 BEV Pretrain with Skeleton-Recall Loss
#
# Usage:
#   bash scripts/train_satmaptracker_stage1_skeleton.sh                    # default (cross-attn ON, conv-fusion OFF, prior-only OFF)
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --no-skeleton      # skeleton OFF
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --no-cross-attn    # early fusion OFF
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --no-conv-fusion   # late fusion OFF
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --sat-encoder resnet_fpn  # ResNet50+FPN encoder
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --sat-gate -1.0    # cross-attn gate init
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --fusion-gate -1.0 # conv fusion gate init
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --sat-pretrained none
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --prior-only        # A1 mode: prior-only residual logit correction
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --prior-only --prior-class-range-gate --prior-class-logits "[-2.0,0.0,0.0]"
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --prior-only --prior-warp --prior-warp-offset-scale 0.05
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --use-lovasz --use-cldice --use-abl
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --dry-run
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --seed 0 --deterministic
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --gpus 4,5,6,7     # custom GPUs
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --wandb-name my_exp # custom wandb name
#   MAPTRACKER_PYTHON=$(which python) bash scripts/train_satmaptracker_stage1_skeleton.sh ...  # custom python bin

set -e

# ============================================================
# Default settings
# ============================================================
GPUS="0,1"
NUM_GPUS=2
MASTER_PORT=29572
USE_SKELETON=true
SKEL_WEIGHT=1.0
SKEL_CLASSES="[2]"
SAT_ENCODER=satmae
SAT_PRETRAINED_OVERRIDE=""
USE_CROSS_ATTN=true
SAT_FUSION_MODE=gate
SAT_GATE_INIT=-1.0
USE_CONV_FUSION=false
FUSION_GATE_INIT=-1.0
PRIOR_ONLY=false
USE_PRIOR_CLASS_RANGE_GATE=false
PRIOR_CLASS_LOGITS="[-2.0,0.0,0.0]"
PRIOR_RANGE_CENTER=12.0
PRIOR_RANGE_SCALE=4.0
USE_PRIOR_WARP=false
PRIOR_WARP_HIDDEN=64
PRIOR_WARP_OFFSET_SCALE=0.05
PRIOR_WARP_REG_WEIGHT=0.01
USE_LOVASZ=false
LOVASZ_WEIGHT=0.5
USE_CLDICE=false
CLDICE_WEIGHT=0.15
CLDICE_CLASSES="[1]"
CLDICE_ITERS=10
USE_ABL=false
ABL_WEIGHT=0.08
ABL_CLASSES="[2]"
DRY_RUN=false
SEED=""
DETERMINISTIC=false
NO_VALIDATE=false
MAX_ITERS_OVERRIDE=""
WORKERS_PER_GPU_OVERRIDE=""
BEV_VIS_INTERVAL=500
WANDB_ENTITY="IRCV_Mapping"
WANDB_PROJECT="Third-SaTracker-dohyun"
WANDB_NAME="prior+class/range_gate+warp"
CONFIG="plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py"
DATASET_TRAIN_SAMPLES=9274
CFG_BATCH_SIZE=3
CFG_NUM_EPOCHS=18
CFG_NUM_EPOCH_INTERVAL_DIVISOR=6

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
        --sat-encoder)
            SAT_ENCODER="$2"
            shift 2 ;;
        --sat-pretrained)
            SAT_PRETRAINED_OVERRIDE="$2"
            shift 2 ;;
        --no-cross-attn)
            USE_CROSS_ATTN=false
            shift ;;
        --sat-fusion-mode)
            SAT_FUSION_MODE="$2"
            shift 2 ;;
        --sat-gate)
            SAT_GATE_INIT="$2"
            shift 2 ;;
        --no-conv-fusion)
            USE_CONV_FUSION=false
            shift ;;
        --fusion-gate)
            FUSION_GATE_INIT="$2"
            shift 2 ;;
        --prior-only)
            PRIOR_ONLY=true
            USE_CROSS_ATTN=false
            USE_CONV_FUSION=false
            shift ;;
        --prior-class-range-gate)
            USE_PRIOR_CLASS_RANGE_GATE=true
            shift ;;
        --prior-class-logits)
            PRIOR_CLASS_LOGITS="$2"
            shift 2 ;;
        --prior-range-center)
            PRIOR_RANGE_CENTER="$2"
            shift 2 ;;
        --prior-range-scale)
            PRIOR_RANGE_SCALE="$2"
            shift 2 ;;
        --prior-warp)
            USE_PRIOR_WARP=true
            shift ;;
        --prior-warp-hidden)
            PRIOR_WARP_HIDDEN="$2"
            shift 2 ;;
        --prior-warp-offset-scale)
            PRIOR_WARP_OFFSET_SCALE="$2"
            shift 2 ;;
        --prior-warp-reg-weight)
            PRIOR_WARP_REG_WEIGHT="$2"
            shift 2 ;;
        --use-lovasz)
            USE_LOVASZ=true
            shift ;;
        --lovasz-weight)
            LOVASZ_WEIGHT="$2"
            shift 2 ;;
        --use-cldice)
            USE_CLDICE=true
            shift ;;
        --cldice-weight)
            CLDICE_WEIGHT="$2"
            shift 2 ;;
        --cldice-classes)
            CLDICE_CLASSES="$2"
            shift 2 ;;
        --cldice-iters)
            CLDICE_ITERS="$2"
            shift 2 ;;
        --use-abl)
            USE_ABL=true
            shift ;;
        --abl-weight)
            ABL_WEIGHT="$2"
            shift 2 ;;
        --abl-classes)
            ABL_CLASSES="$2"
            shift 2 ;;
        --dry-run)
            DRY_RUN=true
            shift ;;
        --no-validate)
            NO_VALIDATE=true
            shift ;;
        --max-iters)
            MAX_ITERS_OVERRIDE="$2"
            shift 2 ;;
        --workers-per-gpu)
            WORKERS_PER_GPU_OVERRIDE="$2"
            shift 2 ;;
        --seed)
            SEED="$2"
            shift 2 ;;
        --deterministic)
            DETERMINISTIC=true
            shift ;;
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

if [ "$SAT_ENCODER" != "satmae" ] && [ "$SAT_ENCODER" != "resnet_fpn" ]; then
    echo "Invalid --sat-encoder: ${SAT_ENCODER} (allowed: satmae, resnet_fpn)"
    exit 1
fi

if [ "$SAT_FUSION_MODE" != "gate" ] && [ "$SAT_FUSION_MODE" != "add" ]; then
    echo "Invalid --sat-fusion-mode: ${SAT_FUSION_MODE} (allowed: gate, add)"
    exit 1
fi

if [ "$USE_PRIOR_CLASS_RANGE_GATE" = true ] && [ "$PRIOR_ONLY" != true ]; then
    PRIOR_ONLY=true
    USE_CROSS_ATTN=false
    USE_CONV_FUSION=false
    echo "[Config] --prior-class-range-gate requested without --prior-only; enabling prior-only automatically."
fi

if [ "$USE_PRIOR_WARP" = true ] && [ "$PRIOR_ONLY" != true ]; then
    PRIOR_ONLY=true
    USE_CROSS_ATTN=false
    USE_CONV_FUSION=false
    echo "[Config] --prior-warp requested without --prior-only; enabling prior-only automatically."
fi

# ============================================================
# Build cfg-options list (single --cfg-options invocation)
# ============================================================
CFG_OPTIONS=()

# Skeleton loss control
if [ "$USE_SKELETON" = true ]; then
    CFG_OPTIONS+=("model.seg_cfg.loss_skel.type=SkelRecallLoss")
    CFG_OPTIONS+=("model.seg_cfg.loss_skel.loss_weight=${SKEL_WEIGHT}")
    CFG_OPTIONS+=("model.seg_cfg.loss_skel.skel_classes=${SKEL_CLASSES}")
    echo "[Config] Skeleton-Recall Loss: ON (weight=${SKEL_WEIGHT}, classes=${SKEL_CLASSES})"
else
    CFG_OPTIONS+=("model.seg_cfg.loss_skel=None")
    echo "[Config] Skeleton-Recall Loss: OFF"
fi

# Satellite encoder selection
if [ "$USE_CROSS_ATTN" = true ] || [ "$PRIOR_ONLY" = true ]; then
    if [ "$SAT_ENCODER" = "resnet_fpn" ]; then
        # Replace SatMAE config fully to prevent stale keys (img_size/patch/pretrained).
        CFG_OPTIONS+=("model.sat_encoder_cfg._delete_=True")
        CFG_OPTIONS+=("model.sat_encoder_cfg.type=SatelliteEncoder")
        CFG_OPTIONS+=("model.sat_encoder_cfg.out_channels=256")
        CFG_OPTIONS+=("model.sat_encoder_cfg.token_grid_size=14")
        CFG_OPTIONS+=("model.sat_encoder_cfg.frozen=False")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.type=ResNet")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.depth=50")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.num_stages=4")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.out_indices=[0,1,2,3]")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.frozen_stages=-1")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.norm_cfg.type=BN2d")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.norm_eval=True")
        CFG_OPTIONS+=("model.sat_encoder_cfg.backbone_cfg.style=pytorch")
        echo "[Config] Satellite Encoder: ResNet50+FPN (trainable, grid=14×14)"
    else
        echo "[Config] Satellite Encoder: SatMAE ViT-L (frozen)"
    fi

    if [ -n "${SAT_PRETRAINED_OVERRIDE}" ]; then
        if [ "${SAT_PRETRAINED_OVERRIDE}" = "none" ] || [ "${SAT_PRETRAINED_OVERRIDE}" = "None" ]; then
            CFG_OPTIONS+=("model.sat_encoder_cfg.pretrained=None")
            echo "[Config] Sat pretrained: OFF (None)"
        else
            CFG_OPTIONS+=("model.sat_encoder_cfg.pretrained=${SAT_PRETRAINED_OVERRIDE}")
            echo "[Config] Sat pretrained: ${SAT_PRETRAINED_OVERRIDE}"
        fi
    fi

else
    CFG_OPTIONS+=("model.sat_encoder_cfg=None")
    echo "[Config] Satellite Encoder: OFF"
fi

# Cross-attention fusion mode
if [ "$USE_CROSS_ATTN" = true ]; then
    CFG_OPTIONS+=("model.use_sat_backbone_fusion=True")
    CFG_OPTIONS+=("model.backbone_cfg.transformer.encoder.transformerlayers.sat_fusion_mode=${SAT_FUSION_MODE}")
    if [ "$SAT_FUSION_MODE" = "gate" ]; then
        CFG_OPTIONS+=("model.backbone_cfg.transformer.encoder.transformerlayers.sat_gate_init=${SAT_GATE_INIT}")
        echo "[Config] Cross-Attention (Early Fusion): ON (mode=gate, init=${SAT_GATE_INIT}, sigmoid≈$(python3 -c "import math; print(f'{1/(1+math.exp(-${SAT_GATE_INIT})):.0%}')"))"
    else
        echo "[Config] Cross-Attention (Early Fusion): ON (mode=add)"
    fi
else
    CFG_OPTIONS+=("model.use_sat_backbone_fusion=False")
    echo "[Config] Cross-Attention (Early Fusion): OFF"
fi

# Conv fusion (late fusion) control
if [ "$USE_CONV_FUSION" = true ]; then
    CFG_OPTIONS+=("model.conv_fusion_cfg.gate_init=${FUSION_GATE_INIT}")
    echo "[Config] Conv Fusion (Late Fusion): ON (gate_init=${FUSION_GATE_INIT}, sigmoid≈$(python3 -c "import math; print(f'{1/(1+math.exp(-${FUSION_GATE_INIT})):.0%}')"))"
else
    CFG_OPTIONS+=("model.conv_fusion_cfg=None")
    echo "[Config] Conv Fusion (Late Fusion): OFF"
fi

# A1 mode: prior-only residual logit correction
if [ "$PRIOR_ONLY" = true ]; then
    CFG_OPTIONS+=("model.use_sat_prior=True")
    echo "[Config] Prior-only A1: ON (S_final = S_cam + U_sat * DeltaS_sat)"
else
    CFG_OPTIONS+=("model.use_sat_prior=False")
fi

# A2 mode: class-aware + range-aware prior gate
if [ "$USE_PRIOR_CLASS_RANGE_GATE" = true ]; then
    CFG_OPTIONS+=("model.use_sat_class_range_gate=True")
    CFG_OPTIONS+=("model.prior_gate_cfg.class_logit_init=${PRIOR_CLASS_LOGITS}")
    CFG_OPTIONS+=("model.prior_gate_cfg.range_center=${PRIOR_RANGE_CENTER}")
    CFG_OPTIONS+=("model.prior_gate_cfg.range_scale=${PRIOR_RANGE_SCALE}")
    echo "[Config] Prior class/range gate: ON (class_logits=${PRIOR_CLASS_LOGITS}, center=${PRIOR_RANGE_CENTER}, scale=${PRIOR_RANGE_SCALE})"
else
    CFG_OPTIONS+=("model.use_sat_class_range_gate=False")
fi

# A3 mode: alignment-aware warp head on satellite prior
if [ "$USE_PRIOR_WARP" = true ]; then
    CFG_OPTIONS+=("model.use_sat_prior_warp=True")
    CFG_OPTIONS+=("model.sat_prior_warp_cfg.hidden_channels=${PRIOR_WARP_HIDDEN}")
    CFG_OPTIONS+=("model.sat_prior_warp_cfg.offset_scale=${PRIOR_WARP_OFFSET_SCALE}")
    CFG_OPTIONS+=("model.sat_prior_warp_cfg.offset_reg_weight=${PRIOR_WARP_REG_WEIGHT}")
    echo "[Config] Prior warp: ON (hidden=${PRIOR_WARP_HIDDEN}, offset_scale=${PRIOR_WARP_OFFSET_SCALE}, reg_weight=${PRIOR_WARP_REG_WEIGHT})"
else
    CFG_OPTIONS+=("model.use_sat_prior_warp=False")
fi

# B1 losses
if [ "$USE_LOVASZ" = true ]; then
    CFG_OPTIONS+=("model.seg_cfg.loss_lovasz.loss_weight=${LOVASZ_WEIGHT}")
    echo "[Config] Lovasz Loss: ON (weight=${LOVASZ_WEIGHT})"
else
    CFG_OPTIONS+=("model.seg_cfg.loss_lovasz.loss_weight=0.0")
fi

if [ "$USE_CLDICE" = true ]; then
    CFG_OPTIONS+=("model.seg_cfg.loss_cldice.loss_weight=${CLDICE_WEIGHT}")
    CFG_OPTIONS+=("model.seg_cfg.loss_cldice.classes=${CLDICE_CLASSES}")
    CFG_OPTIONS+=("model.seg_cfg.loss_cldice.iterations=${CLDICE_ITERS}")
    echo "[Config] clDice Loss: ON (weight=${CLDICE_WEIGHT}, classes=${CLDICE_CLASSES}, iters=${CLDICE_ITERS})"
else
    CFG_OPTIONS+=("model.seg_cfg.loss_cldice.loss_weight=0.0")
fi

if [ "$USE_ABL" = true ]; then
    CFG_OPTIONS+=("model.seg_cfg.loss_abl.loss_weight=${ABL_WEIGHT}")
    CFG_OPTIONS+=("model.seg_cfg.loss_abl.classes=${ABL_CLASSES}")
    echo "[Config] ABL Loss: ON (weight=${ABL_WEIGHT}, classes=${ABL_CLASSES})"
else
    CFG_OPTIONS+=("model.seg_cfg.loss_abl.loss_weight=0.0")
fi

# WandB settings
CFG_OPTIONS+=("log_config.hooks.1.init_kwargs.entity=${WANDB_ENTITY}")
CFG_OPTIONS+=("log_config.hooks.1.init_kwargs.project=${WANDB_PROJECT}")
CFG_OPTIONS+=("log_config.hooks.1.init_kwargs.name=${WANDB_NAME}")

# BEV visualization interval
CFG_OPTIONS+=("custom_hooks.0.interval=${BEV_VIS_INTERVAL}")

# Keep schedule/LR consistent when NUM_GPUS changes from config default.
NUM_ITERS_PER_EPOCH=$((DATASET_TRAIN_SAMPLES / (NUM_GPUS * CFG_BATCH_SIZE)))
if [ "${NUM_ITERS_PER_EPOCH}" -lt 1 ]; then
    NUM_ITERS_PER_EPOCH=1
fi
NUM_EPOCHS_INTERVAL=$((CFG_NUM_EPOCHS / CFG_NUM_EPOCH_INTERVAL_DIVISOR))
if [ "${NUM_EPOCHS_INTERVAL}" -lt 1 ]; then
    NUM_EPOCHS_INTERVAL=1
fi
MAX_ITERS=$((CFG_NUM_EPOCHS * NUM_ITERS_PER_EPOCH))
EVAL_INTERVAL=$((NUM_EPOCHS_INTERVAL * NUM_ITERS_PER_EPOCH))
OPT_LR=$(python3 - <<PY
num_gpus = int("${NUM_GPUS}")
print(5e-4 * (num_gpus / 8.0))
PY
)
CFG_OPTIONS+=("optimizer.lr=${OPT_LR}")
CFG_OPTIONS+=("runner.max_iters=${MAX_ITERS}")
CFG_OPTIONS+=("evaluation.interval=${EVAL_INTERVAL}")
CFG_OPTIONS+=("checkpoint_config.interval=${EVAL_INTERVAL}")
if [ -n "${WORKERS_PER_GPU_OVERRIDE}" ]; then
    CFG_OPTIONS+=("data.workers_per_gpu=${WORKERS_PER_GPU_OVERRIDE}")
fi

if [ -n "${MAX_ITERS_OVERRIDE}" ]; then
    CFG_OPTIONS+=("runner.max_iters=${MAX_ITERS_OVERRIDE}")
    CFG_OPTIONS+=("evaluation.interval=1000000")
    CFG_OPTIONS+=("checkpoint_config.interval=1000000")
    CFG_OPTIONS+=("log_config.interval=1")
    MAX_ITERS="${MAX_ITERS_OVERRIDE}"
    EVAL_INTERVAL=1000000
fi

# ============================================================
# Print settings
# ============================================================
echo "============================================"
echo " SatMapTracker Stage1 BEV Pretrain"
echo "============================================"
echo " GPUs:          ${GPUS} (${NUM_GPUS} devices)"
echo " Master Port:   ${MASTER_PORT}"
echo " Config:        ${CONFIG}"
echo " Sat Encoder:   ${SAT_ENCODER}"
echo " Cross-Attn:    ${USE_CROSS_ATTN} (mode=${SAT_FUSION_MODE}, gate=${SAT_GATE_INIT})"
echo " Conv Fusion:   ${USE_CONV_FUSION} (gate=${FUSION_GATE_INIT})"
echo " Prior-only:    ${PRIOR_ONLY}"
echo " Prior gate:    ${USE_PRIOR_CLASS_RANGE_GATE} (class_logits=${PRIOR_CLASS_LOGITS}, center=${PRIOR_RANGE_CENTER}, scale=${PRIOR_RANGE_SCALE})"
echo " Prior warp:    ${USE_PRIOR_WARP} (hidden=${PRIOR_WARP_HIDDEN}, offset_scale=${PRIOR_WARP_OFFSET_SCALE}, reg_weight=${PRIOR_WARP_REG_WEIGHT})"
echo " Lovasz:        ${USE_LOVASZ} (weight=${LOVASZ_WEIGHT})"
echo " clDice:        ${USE_CLDICE} (weight=${CLDICE_WEIGHT}, classes=${CLDICE_CLASSES}, iters=${CLDICE_ITERS})"
echo " ABL:           ${USE_ABL} (weight=${ABL_WEIGHT}, classes=${ABL_CLASSES})"
echo " Skeleton:      ${USE_SKELETON} (weight=${SKEL_WEIGHT})"
echo " Seed:          ${SEED:-default(0)}"
echo " Deterministic: ${DETERMINISTIC}"
echo " No-validate:   ${NO_VALIDATE}"
echo " Sched it/ep:   ${NUM_ITERS_PER_EPOCH}"
echo " Sched max_it:  ${MAX_ITERS}"
echo " Sched eval_it: ${EVAL_INTERVAL}"
echo " Optimizer lr:  ${OPT_LR}"
echo " WandB:         ${WANDB_PROJECT} / ${WANDB_NAME}"
echo " BEV Vis:       every ${BEV_VIS_INTERVAL} iters"
echo "============================================"
echo ""

if [ "${DRY_RUN}" = true ]; then
    echo "[Dry Run] Training command generation complete. Exiting without launch."
    exit 0
fi

# ============================================================
# Run training
# ============================================================
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

cd $(dirname $(dirname $(realpath $0)))

PYTHON_BIN="${MAPTRACKER_PYTHON:-/venv/maptracker/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "Error: No usable python executable found."
        echo "Hint: activate your environment, or run with MAPTRACKER_PYTHON=/path/to/python bash scripts/train_satmaptracker_stage1_skeleton.sh ..."
        exit 1
    fi
fi

echo "[Config] Python: ${PYTHON_BIN}"

if ! "${PYTHON_BIN}" - <<'PY'
import importlib
for m in ("mmcv", "mmdet", "mmdet3d"):
    importlib.import_module(m)
print("ok")
PY
then
    echo "Error: Required training modules (mmcv/mmdet/mmdet3d) are not importable in ${PYTHON_BIN}."
    echo "Hint: activate/install the MapTracker environment, or set MAPTRACKER_PYTHON to a valid interpreter."
    exit 1
fi

TRAIN_ARGS=()
if [ -n "${SEED}" ]; then
    TRAIN_ARGS+=(--seed "${SEED}")
fi
if [ "${DETERMINISTIC}" = true ]; then
    TRAIN_ARGS+=(--deterministic)
fi
if [ "${NO_VALIDATE}" = true ]; then
    TRAIN_ARGS+=(--no-validate)
fi

PYTHONPATH="$(pwd):${PYTHONPATH}" "${PYTHON_BIN}" -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tools/train.py \
    ${CONFIG} \
    "${TRAIN_ARGS[@]}" \
    --launcher pytorch \
    --cfg-options "${CFG_OPTIONS[@]}"
