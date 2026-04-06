_base_ = ['./satmaptracker_stage1.py']

# ============================================================================
# SatMapTracker Stage 1: sat_encoder lr_mult=0.5 비교 실험 (1/3 dataset)
# ============================================================================

num_gpus = 2
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

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sat_encoder': dict(lr_mult=0.5),
        }),
    weight_decay=1e-2)

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
                 project='Third-SatFuse-AID4AD-Kyungmin',
                 name='satmaptracker-stage1-third-satlr05',
             )),
    ])
