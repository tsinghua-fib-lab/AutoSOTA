# dataset config
_base_ = [
    "../_base_/datasets/cityscapes-c_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/models/rein_dinov2_mask2former.py",
]
crop_size = (1024, 1024)
model = dict(
    backbone=dict(
        img_size=1024,
        init_cfg=dict(
            checkpoint="checkpoints/dinov2_converted_1024x1024.pth",
        ),
    ),
    data_preprocessor=dict(
        size=crop_size,
    ),
    test_cfg=dict(
        crop_size=(1024, 1024),
        stride=(683, 683),
    ),
)
# training schedule for 160k
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
find_unused_parameters = True
