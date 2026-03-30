_base_ = ['./satmaptracker_stage2_warmup.py']

# ============================================================================
# SatMapTracker Stage 2: Tracking Warmup (1/3 dataset)
# ============================================================================

num_gpus = 2
batch_size = 4
num_iters_per_epoch = 9274 // (num_gpus * batch_size)  # 1/3 dataset
num_epochs = 4
num_epochs_interval = num_epochs
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
                 name='stage2',
             )),
    ])

load_from = "work_dirs/satmaptracker_stage1_bev_pretrain_third/latest.pth"
