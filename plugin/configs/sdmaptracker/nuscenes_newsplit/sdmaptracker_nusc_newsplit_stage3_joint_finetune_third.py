_base_ = ['./sdmaptracker_nusc_newsplit_stage3_joint_finetune.py']

# ============================================================================
# SDMapTracker Stage 3: Joint Finetune (1/3 dataset for fast iteration)
# ============================================================================

num_gpus = 4
batch_size = 2
num_iters_per_epoch = 9274 // (num_gpus * batch_size)  # 1/3 dataset: 9274 samples
num_epochs = 36
num_epochs_interval = num_epochs // 6
total_iters = num_epochs * num_iters_per_epoch

# Override dataset to use 1/3 split
data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        ann_file='./datasets/nuscenes/nuscenes_map_infos_train_newsplit_third.pkl',
        sd_prior_cache_path='./datasets/nuscenes/sd_prior_cache_train_newsplit.pkl',
    ),
)

# Adjust schedule for 1/3 dataset
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=170,  # ~500 / 3
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

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
                 project='sdmaptracker',
                 name='newsplit-sdmap-stage3_joint_finetune_third',
             )),
    ])

# with_cp=False: original MapTracker uses False, gradient checkpointing was causing hang
model = dict(
    backbone_cfg=dict(
        img_backbone=dict(with_cp=False),
    ),
)
find_unused_parameters = True

load_from = "work_dirs/sdmaptracker_nusc_newsplit_stage2_warmup_third/latest.pth"
