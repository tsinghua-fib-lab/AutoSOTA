## Model Code For Pixel-SAIL

## Prepare Data
1. Load the sa2va data from `https://huggingface.co/datasets/Dense-World/Sa2VA-Training/tree/main`.
2. Load the COCO 2017 dataset.
3. Load the dataset from `https://huggingface.co/datasets/zhangtao-whu/st_train/tree/main`.

Data structure should be like:
```
data/
├── glamm_data
|   ├── images
|   ├── annotations
├── llava_data
|   ├── llava_images
|   ├── LLaVA-Instruct-150K
|   ├── LLaVA-Pretrain
├── coco
|   ├── annotations
|       panoptic_train2017.json
|       panoptic_val2017.json
|       panoptic_train2017
|   ├── train2017
|   ├── val2017
|   ├── test2017
├── ref_seg
|   ├── refcoco
|   ├── refcoco+
|   ├── refcocog
├── muse
|   ├── MUSE_train.json
|   ├── MUSE_val.json    
|   ├── MUSE_test_many.json
|   ├── MUSE_test_less.json
├── pixel2cap
|   ├── pix2cap_coco_train.json
|   ├── pix2cap_coco_val.json  
```

