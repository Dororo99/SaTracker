_base_ = ['./satmaptracker_stage1.py']

# ============================================================================
# SatMapTracker Stage 1: SatelliteConvFuser + Triple Supervision (1/3 dataset)
# ============================================================================

num_gpus = 4
batch_size = 3
num_iters_per_epoch = 9274 // (num_gpus * batch_size)  # 1/3 dataset: 9274 samples
num_epochs = 18
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

# Add skeleton regularization to main seg loss only (cam branch).
model = dict(
    skeleton_main_only=True,
    seg_cfg=dict(
        loss_skeleton=dict(
            type='MaskSkeletonLoss',
            loss_weight=1.0,  # target ratio (config-level): focal:dice:skeleton = 10:1:1
            num_dilation=1,
            class_indices=[1, 2],  # divider, boundary
            class_weights=[2.0, 1.0],  # divider:boundary = 2:1
            ignore_empty_targets=True,
        ),
    ),
)

# Model config inherited from base (history_mode='cam', residual fusion, etc.)
# Optimizer config inherited from base (both backbones at lr_mult=0.1).

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 entity='IRCV_Mapping',
                 project='Third-SaTracker-dohyun-v2',
                 name='satracker-stage1-third-skel',
             )),
    ])
