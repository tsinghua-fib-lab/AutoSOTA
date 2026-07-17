norm_cfg = dict(type='GN', requires_grad=True, num_groups=32)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/model_best.pth.tar',
    backbone=dict(
        type='EWSegNet',
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        drop_path_rate=0.2),
    decode_head=dict(
        type='UPerHead',
        in_channels=[80, 160, 320, 640],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.2,
        num_classes=5,
        norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=4,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.2,
        num_classes=5,
        norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'ZeroWasteDataset'
data_root = '/datasets/zerowaste-f-final/splits_final_deblurred/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='AlignResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type='ZeroWasteDataset',
            data_root='/datasets/zerowaste-f-final/splits_final_deblurred/',
            img_dir='train/data',
            ann_dir='train/sem_seg',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(
                    type='Resize',
                    img_scale=(2048, 512),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='ZeroWasteDataset',
        data_root='/datasets/zerowaste-f-final/splits_final_deblurred/',
        img_dir='val/data',
        ann_dir='val/sem_seg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ZeroWasteDataset',
        data_root='/datasets/zerowaste-f-final/splits_final_deblurred/',
        img_dir='test/data',
        ann_dir='test/sem_seg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ''
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
gpu_multiples = 1
optimizer = dict(
    type='AdamW', lr=5e-05, betas=(0.9, 0.999), weight_decay=0.009)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=0.95,
    min_lr=1e-07,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=4000, metric=['mIoU', 'mFscore'], save_best='mIoU')
work_dir = 'work_dirs/iter1_gn_droppath'
gpu_ids = range(0, 1)
