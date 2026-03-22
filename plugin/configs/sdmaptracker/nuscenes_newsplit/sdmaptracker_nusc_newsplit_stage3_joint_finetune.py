_base_ = ['../../maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage3_joint_finetune.py']

# ============================================================================
# SDMapTracker Stage 3: Joint Finetune
# - SD queries enabled
# - Full model training (backbone + head)
# - Memory enabled
# - Track + SD + Free queries
# ============================================================================

# SD map query settings
num_free_queries = 50
max_sd_queries = 50

model = dict(
    head_cfg=dict(
        num_queries=100,  # kept for backward compat
        num_free_queries=num_free_queries,
        max_sd_queries=max_sd_queries,
        sd_attr_dim=4,
        sd_tag_dim=2,
        use_sd_queries=True,
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=5.0),
                reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=True),
                sd_prior_cost=dict(type='SDPriorCost', weight=5.0),
            ),
        ),
    ),
)

# Override train pipeline to include sd_priors in Collect3D keys
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_h = 480
img_w = 800
img_size = (img_h, img_w)
roi_size = (60, 30)
coords_dim = 2
num_points = 20
permute = True
canvas_size = (200, 100)
thickness = 3

train_pipeline = [
    dict(
        type='VectorizeMap',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
    ),
    dict(
        type='RasterizeMap',
        roi_size=roi_size,
        coords_dim=coords_dim,
        canvas_size=canvas_size,
        thickness=thickness,
        semantic_mask=True,
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size,
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors', 'semantic_mask', 'sd_priors'], meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name'))
]

# Add SD prior cache path to dataset
data = dict(
    train=dict(
        sd_prior_cache_path='./datasets/nuscenes/sd_prior_cache_train_newsplit.pkl',
        pipeline=train_pipeline,
    ),
    val=dict(
        sd_prior_cache_path='./datasets/nuscenes/sd_prior_cache_val_newsplit.pkl',
    ),
    test=dict(
        sd_prior_cache_path='./datasets/nuscenes/sd_prior_cache_val_newsplit.pkl',
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 entity='IRCV_Mapping',
                 project='sdmaptracker',
                 name='newsplit-sdmap-stage3_joint_finetune',
             )),
    ])

load_from = "work_dirs/sdmaptracker_nusc_newsplit_stage2_warmup/latest.pth"
