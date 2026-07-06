_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_ade20k.txt',
    text_template='seg_template',
    prob_thd=0.07,
    background=True,
    tau=5.0,
    tem=5.0
)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
