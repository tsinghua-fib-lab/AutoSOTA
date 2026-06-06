# Task Learning

## Dataset Preparation

The Preparation is similar as [Rein](https://github.com/w1oves/Rein/tree/train).

**UrbanSyn**: Download all image and label packages from [UrbanSyn](http://www.urbansyn.org/#loaded) and extract them to `data/urbansyn`.

**Mapillary:** Download MAPILLARY v1.2 from [Mapillary Research](https://research.mapillary.com/) and extract it to `data/mapillary`.

**ACDC**: Download all image and label packages from [ACDC](https://acdc.vision.ee.ethz.ch/) and extract them to `data/acdc`.

**BDD100K**: Download all image and label packages from [BDD100K](http://bdd-data.berkeley.edu/) and extract them to `data/bdd100k`.


Prepare datasets with these commands:
```shell
# Convert data for validation if preparing for the first time
python tools/convert_datasets/gta.py data/gta # Source domain
python tools/convert_datasets/cityscapes.py data/cityscapes
# Convert Mapillary to Cityscapes format and resize for validation
python tools/convert_datasets/mapillary2cityscape.py data/mapillary data/mapillary/cityscapes_trainIdLabel --train_id
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images data/mapillary/cityscapes_trainIdLabel/val/label data/mapillary/half/val_img data/mapillary/half/val_label
```

## Convert Weights
Convert distiled weights for training:

```bash
python tools/convert_models/convert_gkd.py checkpoints/checkpoint_100.pth checkpoints/distillation_converted.pth
```

## Training
Start training in single GPU:
```
python tools/train.py configs/frozen_vfms/deit_vit-B_mask2former.py
```
Start training in multiple GPU:
```
PORT=12345 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/frozen_vfms/deit_vit-B_mask2former.py NUM_GPUS
```


## Acknowledgment

This part is heavily build upon [Rein](https://github.com/w1oves/Rein/tree/train). We gratefully thank the authors for their wonderful works.