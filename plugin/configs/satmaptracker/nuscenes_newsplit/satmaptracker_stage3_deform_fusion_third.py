"""
SatMapTracker Stage 3: Deformable Attention Fusion (1/3 dataset)

Dual-Path Deformable Attention Fusion:
  - cam→sat cross-attention: camera가 satellite에서 보완 정보 획득 (ped_crossing 등)
  - sat→cam cross-attention: satellite이 camera에서 fine structure 획득 (divider 등)
  - spatial gate: dual-path 결과를 pixel-wise로 조합
"""

_base_ = ['./satmaptracker_stage3_joint_finetune.py']

# ── Override fusion module ──
model = dict(
    sat_fusion_cfg=dict(
        _delete_=True,      # fully replace base config's sat_fusion_cfg
        type='SatCamDeformAttnFusion',
        in_channels=256,
        num_heads=8,
        num_points=4,       # 4 sampling points per query — cost-efficient
        num_classes=3,      # ped_crossing, divider, boundary
        ffn_dim=512,
        dropout=0.1,
    ),
    model_name='SatMapTracker_Stage3_DeformFusion',
)

# ── 1/3 dataset split ──
num_gpus = 2
batch_size = 3
num_iters_per_epoch = 9274 // (num_gpus * batch_size)
num_epochs = 36
num_epochs_interval = num_epochs // 6
total_iters = num_epochs * num_iters_per_epoch

data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        ann_file='./datasets/nuscenes/nuscenes_map_infos_train_newsplit_third.pkl',
    ),
    val=dict(
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit_third.pkl',
        eval_config=dict(
            ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit_third.pkl',
        ),
    ),
    test=dict(
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit_third.pkl',
        eval_config=dict(
            ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit_third.pkl',
        ),
    ),
)

evaluation = dict(interval=num_epochs_interval * num_iters_per_epoch)
checkpoint_config = dict(interval=num_epochs_interval * num_iters_per_epoch)

runner = dict(
    type='MyRunnerWrapper', max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 entity='IRCV_Mapping',
                 project='Third-SatMapTracker-Seperate-AID4AD-Kyungmin',
                 name='stage3-deform-fusion',
             )),
    ])

load_from = "work_dirs/satmaptracker_stage2_warmup_third/latest.pth"
