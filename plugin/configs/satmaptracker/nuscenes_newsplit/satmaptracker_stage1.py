"""
SatMapTracker Stage 1: BEV Pretrain with Satellite Fusion

Fusion: SatelliteConvFuser (concat + conv, no residual)
Triple supervision: cam_bev, sat_bev, fused_bev through shared seg_decoder
History: fused_bev (per-frame aligned satellite)
Encoder: SatelliteEncoder (ResNet50 + FPN)
"""

_base_ = [
    '../../_base_/default_runtime.py'
]

type = 'Mapper'
plugin = True
plugin_dir = 'plugin/'

# ── Image configs ──
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_h = 480
img_w = 800
img_size = (img_h, img_w)
num_cams = 6

num_gpus = 2
batch_size = 3
num_iters_per_epoch = 27846 // (num_gpus * batch_size)
num_epochs = 18
num_epochs_interval = num_epochs // 6
total_iters = num_epochs * num_iters_per_epoch
num_queries = 100

# ── Category configs ──
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1

# ── BEV configs ──
roi_size = (60, 30)
bev_h = 50
bev_w = 100
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]

# ── Vector / raster params ──
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True
canvas_size = (200, 100)
thickness = 3

meta = dict(
    use_lidar=False, use_camera=True, use_radar=False,
    use_map=False, use_external=False, output_format='vector')

# ── Model dims ──
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_points = 20
permute = True

# ── AID4AD satellite config ──
aid4ad_root = '/workspace/sumin/2026.Online-Mapping.for-NeurIPS-2026/datasets/nuscenes/AID4AD_frames'

model = dict(
    type='SatMapTracker',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    history_steps=4,
    test_time_history_steps=20,
    mem_select_dist_ranges=[1, 5, 10, 15],
    skip_vector_head=True,      # Stage 1: BEV pretrain only
    freeze_bev=False,
    track_fp_aug=False,
    use_memory=False,
    mem_len=4,
    mem_warmup_iters=500,
    # ── Satellite fusion ──
    # Temporal history carries pure cam_bev; fusion is applied only at the
    # current frame via seg_decoder_fused. 'fused' mode is deprecated because
    # fused_bev in the history caused TemporalSelfAttention to drift along
    # with the fusion layer during training.
    history_mode='cam',
    freeze_sat_encoder=False,
    sat_encoder_cfg=dict(
        type='SatelliteEncoder',
        backbone_cfg=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d'),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained',
                          checkpoint='torchvision://resnet50'),
        ),
        out_channels=bev_embed_dims,
        bev_size=(bev_h, bev_w),
    ),
    sat_fusion_cfg=dict(
        type='SatelliteConvFuser',
        in_channels=bev_embed_dims,
        hidden_channels=bev_embed_dims,
        use_residual=True,
    ),
    # ── Camera backbone (same as MapTracker) ──
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        use_grid_mask=True,
        history_steps=4,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=bev_embed_dims,
            start_level=0,
            add_extra_convs=True,
            num_outs=num_feat_levels,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention',
                             embed_dims=bev_embed_dims, num_levels=1),
                        dict(type='SpatialCrossAttention',
                             deformable_attention=dict(
                                 type='MSDeformableAttention3D',
                                 embed_dims=bev_embed_dims,
                                 num_points=8,
                                 num_levels=num_feat_levels),
                             embed_dims=bev_embed_dims),
                    ],
                    feedforward_channels=bev_embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims//2,
            row_num_embed=bev_h,
            col_num_embed=bev_w),
    ),
    head_cfg=dict(
        type='MapDetectorHead',
        num_queries=num_queries,
        embed_dims=embed_dims,
        num_classes=num_class,
        in_channels=bev_embed_dims,
        num_points=num_points,
        roi_size=roi_size,
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        trans_loss_weight=0.1,
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=num_points,
            coord_dim=2,
            encoder=dict(type='PlaceHolderEncoder', embed_dims=embed_dims),
            decoder=dict(
                type='MapTransformerDecoder_new',
                num_layers=6,
                prop_add_stage=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=embed_dims,
                             num_heads=8, attn_drop=0.1, proj_drop=0.1),
                        dict(type='CustomMSDeformableAttention', embed_dims=embed_dims,
                             num_heads=8, num_levels=1, num_points=num_points, dropout=0.1),
                        dict(type='MultiheadAttention', embed_dims=embed_dims,
                             num_heads=8, attn_drop=0.1, proj_drop=0.1),
                    ],
                    ffn_cfgs=dict(type='FFN', embed_dims=embed_dims,
                                  feedforward_channels=embed_dims*2, num_fcs=2,
                                  ffn_drop=0.1, act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=embed_dims*2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'cross_attn', 'norm', 'ffn', 'norm')))),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                      alpha=0.25, loss_weight=5.0),
        loss_reg=dict(type='LinesL1Loss', loss_weight=50.0, beta=0.01),
        assigner=dict(
            type='HungarianLinesAssigner',
            cost=dict(type='MapQueriesCost',
                      cls_cost=dict(type='FocalLossCost', weight=5.0),
                      reg_cost=dict(type='LinesL1Cost', weight=50.0,
                                    beta=0.01, permute=permute))),
    ),
    # ── Seg decoder (fused=parent seg_decoder, cam/sat=separate decoders) ──
    seg_cfg=dict(
        type='MapSegHead',
        num_classes=num_class,
        in_channels=bev_embed_dims,
        embed_dims=bev_embed_dims,
        bev_size=(bev_w, bev_h),
        canvas_size=canvas_size,
        loss_seg=dict(type='MaskFocalLoss', use_sigmoid=True, loss_weight=10.0),
        loss_dice=dict(type='MaskDiceLoss', loss_weight=1.0),
    ),
    model_name='SatMapTracker_Stage1',
)

# ── Data pipeline with AID4AD satellite loading ──
train_pipeline = [
    dict(type='VectorizeMap', coords_dim=coords_dim, roi_size=roi_size,
         sample_num=num_points, normalize=True, permute=permute),
    dict(type='RasterizeMap', roi_size=roi_size, coords_dim=coords_dim,
         canvas_size=canvas_size, thickness=thickness, semantic_mask=True),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='LoadAID4ADSatelliteImage',
         data_root=aid4ad_root, canvas_size=None, normalize=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D',
         keys=['img', 'vectors', 'semantic_mask', 'sat_img'],
         meta_keys=('token', 'ego2img', 'sample_idx',
                    'ego2global_translation', 'ego2global_rotation',
                    'img_shape', 'scene_name')),
]

test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='LoadAID4ADSatelliteImage',
         data_root=aid4ad_root, canvas_size=None, normalize=True),
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'sat_img'],
         meta_keys=('token', 'ego2img', 'sample_idx',
                    'ego2global_translation', 'ego2global_rotation',
                    'img_shape', 'scene_name')),
]

eval_config = dict(
    type='NuscDataset',
    data_root='./datasets/nuscenes',
    ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit.pkl',
    meta=meta, roi_size=roi_size, cat2id=cat2id,
    pipeline=[
        dict(type='VectorizeMap', coords_dim=coords_dim, simplify=True,
             normalize=False, roi_size=roi_size),
        dict(type='RasterizeMap', roi_size=roi_size, coords_dim=coords_dim,
             canvas_size=canvas_size, thickness=thickness, semantic_mask=True),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors', 'semantic_mask'],
             meta_keys=['token', 'ego2img', 'sample_idx',
                        'ego2global_translation', 'ego2global_rotation',
                        'img_shape', 'scene_name']),
    ],
    interval=1,
)

match_config = dict(
    type='NuscDataset',
    data_root='./datasets/nuscenes',
    ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit.pkl',
    meta=meta, roi_size=roi_size, cat2id=cat2id,
    pipeline=[
        dict(type='VectorizeMap', coords_dim=coords_dim, simplify=False,
             normalize=True, roi_size=roi_size, sample_num=num_points),
        dict(type='RasterizeMap', roi_size=roi_size, coords_dim=coords_dim,
             canvas_size=canvas_size, thickness=thickness),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors', 'semantic_mask'],
             meta_keys=['token', 'ego2img', 'ego2cam', 'sample_idx',
                        'ego2global_translation', 'ego2global_rotation',
                        'img_shape', 'scene_name', 'img_filenames',
                        'cam_intrinsics', 'cam_extrinsics',
                        'lidar2ego_translation', 'lidar2ego_rotation']),
    ],
    interval=1,
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=8,
    train=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_train_newsplit.pkl',
        meta=meta, roi_size=roi_size, cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-2, matching=True,
        multi_frame=5, sampling_span=10),
    val=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit.pkl',
        meta=meta, roi_size=roi_size, cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True, seq_split_num=1, eval_semantic=True),
    test=dict(
        type='NuscDataset',
        data_root='./datasets/nuscenes',
        ann_file='./datasets/nuscenes/nuscenes_map_infos_val_newsplit.pkl',
        meta=meta, roi_size=roi_size, cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True, seq_split_num=1, eval_semantic=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# ── Optimizer ──
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=5e-2)

evaluation = dict(interval=num_epochs_interval*num_iters_per_epoch)
find_unused_parameters = True
checkpoint_config = dict(interval=num_epochs_interval*num_iters_per_epoch)

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
                 name='satmaptracker-stage1-full',
             )),
    ])

SyncBN = True
