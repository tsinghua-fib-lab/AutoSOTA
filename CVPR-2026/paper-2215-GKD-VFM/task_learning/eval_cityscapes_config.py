# Evaluation config for Cityscapes only
crop_size = (512, 512)
num_classes = 19
default_scope = 'mmseg'
launcher = 'none'

env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

log_level = 'INFO'
log_processor = dict(by_epoch=False)

model = dict(
    type='MultiScaleModel',
    ms_scales=[0.75, 1.0, 1.25],
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=crop_size,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
    backbone=dict(
        type='DinoVisionTransformer',
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        img_size=512,
        ffn_layer='mlp',
        init_values=None,
        block_chunks=0,
        out_indices=[3, 5, 7, 11],
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[384, 384, 384, 384],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(
                num_feats=128, normalize=True
            ),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            num_feats=128, normalize=True
        ),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0]*num_classes + [0.1],
        ),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type='mmdet.MaskPseudoSampler'),
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        crop_size=(512, 512),
        stride=(341, 341),
    ),
)

# Cityscapes-only evaluation
cityscapes_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val',
            seg_map_path='gtFine/val',
        ),
        pipeline=cityscapes_test_pipeline,
    ),
)

val_dataloader = test_dataloader
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
)
test_evaluator = val_evaluator
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

work_dir = './work_dirs/eval_cityscapes'
load_from = None
resume = False
randomness = dict(seed=42)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    name='visualizer',
    vis_backends=vis_backends)
