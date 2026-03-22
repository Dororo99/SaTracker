_base_ = ['./sdmaptracker_nusc_newsplit_stage2_warmup.py']

# ============================================================================
# SDMapTracker Stage 2: SD+Free Warmup (1/3 dataset for fast iteration)
# ============================================================================

num_gpus = 4
batch_size = 4  # reduced from 8 due to OOM with SD queries (100→150 queries)
num_iters_per_epoch = 9274 // (num_gpus * batch_size)  # 1/3 dataset: 9274 samples
num_epochs = 4
num_epochs_interval = num_epochs
total_iters = num_epochs * num_iters_per_epoch

# Override dataset to use 1/3 split
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,  # reduced from 10 to save RAM (each worker loads SD cache)
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
    min_lr_ratio=0.95)

model = dict(
    mem_warmup_iters=170,  # ~500 / 3
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
                 project='sdmaptracker',
                 name='newsplit-sdmap-stage2_warmup_third',
             )),
    ])
