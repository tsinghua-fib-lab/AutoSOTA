_base_ = "./base_config.py"

model = dict(
    name_path="./configs/cls_voc21.txt",
    text_template="seg_template",
    prob_thd=0.28, 
    background=True,
    tau=3.0,
    tem=0.3,
    pamr_steps=3,
    pamr_stride=(8, 16),
    slide_stride=84,
    slide_crop=336
)

dataset_type = "PascalVOCDataset"
data_root = "/datasets/VOCdevkit/VOC2012"

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 336), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs")
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path="JPEGImages", seg_map_path="SegmentationClass"),
        ann_file="ImageSets/Segmentation/val.txt",
        pipeline=test_pipeline))
