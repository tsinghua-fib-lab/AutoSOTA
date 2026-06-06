# dataset config
_base_ = [
    "../_base_/datasets/dg_gta_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/clip-L_mask2former.py",
]
model = dict(type="FrozenBackboneEncoderDecoder")
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
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "query_embed": embed_multi,
            "query_feat": embed_multi,
            "level_embed": embed_multi,
            "norm": dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
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
'''
2024/02/05 07:45:55 - mmengine - INFO - Iter(val) [3500/3500]    citys_aAcc: 89.0100  citys_mIoU: 51.4900  citys_mAcc: 67.1600  bdd_aAcc: 88.5600  bdd_mIoU: 49.5100  bdd_mAcc: 61.6800  map_aAcc: 89.5800  map_mIoU: 55.5500  map_mAcc: 69.1900  mean_aAcc: 89.0500  mean_mIoU: 52.1833  mean_mAcc: 66.0100  data_time: 0.0032  time: 0.3971
'''