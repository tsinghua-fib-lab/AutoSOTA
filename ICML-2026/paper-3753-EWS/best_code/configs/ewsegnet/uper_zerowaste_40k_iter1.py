_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/zero_waste.py',
    '../_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/model_best.pth.tar', # pretrain (imagenet) weight path 
    backbone=dict(
        type='EWSegNet',
        style='pytorch',
        drop_path_rate=0.2),
    decode_head=dict(num_classes=5,
                     in_channels=[80, 160, 320, 640],
                     channels=256,
                     in_index=[0, 1, 2, 3],
                     norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=5,
                        in_channels=320,
                        channels=256,
                        in_index=4,
                        norm_cfg=norm_cfg)
    )


gpu_multiples = 1  # we used 1 gpu

# optimizer
optimizer = dict(type='AdamW', lr=0.00005*gpu_multiples, betas=(0.9, 0.999), weight_decay=0.009)
optimizer_config = dict()

# learning policy
lr_config = dict(policy='poly', warmup='linear', warmup_iters=1500,
                 warmup_ratio=1e-6, power=0.95, min_lr=1e-7, by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=2000//gpu_multiples)
evaluation = dict(interval=4000//gpu_multiples, metric=['mIoU', 'mFscore'], save_best='mIoU')
