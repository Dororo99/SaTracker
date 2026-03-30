#!/usr/bin/env bash
# SatMapTracker Stage 1: Independent BEV Pretraining
# Camera + Satellite, each with separate seg head
# GPU 0,1

CONFIG=plugin/configs/satmaptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain_third.py
GPUS=2
PORT=${PORT:-29600}

CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:1}
