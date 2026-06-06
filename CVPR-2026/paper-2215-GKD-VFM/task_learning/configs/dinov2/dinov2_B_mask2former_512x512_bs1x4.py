# dataset config
_base_ = [
    "../_base_/datasets/dg_gta_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/dinov2_mask2former.py"
]

model = dict(backbone=dict(
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                img_size=512,
                ffn_layer="mlp",
                init_values=1e-05,
                block_chunks=0,
                out_indices=[3, 5, 7, 11],
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_cfg=None
            ),
            decode_head=dict(in_channels=[768, 768, 768, 768])
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size={{_base_.crop_size}}, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0),
            backbone=dict(lr_mult=0.1)),
        norm_decay_mult=0.0)
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=10000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
