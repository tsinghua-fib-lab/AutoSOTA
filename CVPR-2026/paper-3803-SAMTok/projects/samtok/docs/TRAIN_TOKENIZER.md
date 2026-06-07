# Training Tokenizer

## Environment Setup
You need to install xtuner:
```
pip install -U 'xtuner[deepspeed]'
```

## Dataset Preparation
We collect corresponding masks from the following 11 datasets to train the mask tokenizer:
<!-- | Dataset | Number of sampled masks | Folder structure | Download Link |
|:-:|:-:|:-:|:-:|
| SA1B | 200M | [data/sam_full](projects/samtok/data/sam_full) | [here](https://ai.meta.com/datasets/segment-anything-downloads/) |
| COCONut | 3.5M | [data/object365](projects/samtok/data/object365), [data/coco](projects/samtok/data/coco) | [here](https://cocodataset.org/#download) |
| EntitySeg | 1.4M | [data/entity_lr](projects/samtok/data/entity_lr) | [here](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/images_lr) |
| PixelWeb | 1.8M | [data/pixelweb_100k](projects/samtok/data/pixelweb_100k) | [here](https://huggingface.co/datasets/cyberalchemist/PixelWeb/tree/main/pixelweb_100k/annotated) |
| ADE | 197K | []() | [here]() | -->
- SA1B (200M)
- COCONut (3.5M)
- EntitySeg (1.4M)
- PixelWeb (1.8M)
- ADE (197K)
- Cityscape (223K)
- LISA++ (114K)
- SAV (2.0M)
- MeVIS (7K)
- ReVOS (2K)
- YTRefVOS (16K)

## Launch Training
Here is an example to launch to training of the mask tokenizer using SA1B data: 
```
bash tools/dist.sh train projects/samtok/configs/vq_sam2_256x2_unshare.py 8
```
Note: You can first download the SA1B dataset from [here](https://ai.meta.com/datasets/segment-anything-downloads/), then organize it into [this folder structure](../data/sam_full). Then download the data information from [here](https://huggingface.co/datasets/zhouyik/SAMTok_Training_Data/blob/main/sam_info.json) and place it in the [./data](data) folder. During actual training, we use all the aforementioned data and set the global batch size to 1024.
