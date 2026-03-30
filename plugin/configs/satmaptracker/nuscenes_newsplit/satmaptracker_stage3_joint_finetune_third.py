_base_ = ['./satmaptracker_stage3_joint_finetune.py']

# ============================================================================
# SatMapTracker Stage 3: Joint Finetuning (1/3 dataset)
# ============================================================================

num_gpus = 4
batch_size = 3
num_iters_per_epoch = 9274 // (num_gpus * batch_size)  # 1/3 dataset
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
                 name='stage3',
             )),
    ])

# Override fusion: ConvFusion handles misaligned feature spaces
# (Stage1 independent heads → cam/sat BEV in different spaces)
model = dict(
    sat_fusion_cfg=dict(
        _delete_=True,
        type='SatCamConvFusion',
        in_channels=256,
        hidden_channels=256,
    ),
)

load_from = "work_dirs/satmaptracker_stage2_warmup_third/latest.pth"
