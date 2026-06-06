# Domain-general Distillation

## Dataset Preparation
### Proxy Dataset
Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
### Task Dataset
**Cityscapes:** Download `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from [Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/) and extract them to `/path/to/cityscapes`.

**GTA:** Download all image and label packages from [TU Darmstadt](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `/path/to/gta`.

## Pretraining Weights
Download the model file [facebookresearch_dinov2_main](https://github.com/facebookresearch/dinov2), Download the weights [DINOv2](https://github.com/facebookresearch/dinov2), [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md), [DINO](https://github.com/facebookresearch/dino), [EVA02](https://github.com/baaivision/EVA/tree/master/EVA-02)

## Training
### Task-agnotic Distillation
To train DeiT-small with teacher DINOv2-base on ImageNet on a single node with 4 gpus for 100 epochs run:
```shell
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 128 --epochs 100 \
    --data-set IMNET --data-path imagenet/images \
    --teacher-model vit_base --target_model vit_small --model distillation_models_deit \
    --teacher-path facebookresearch_dinov2_main \
    --student-path weights/deit_small_distilled_patch16_224-649709d9.pth \
    --patch_size 14 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log
```
or run training script:

```shell
bash train_script\dinov2_B_distill_deit_S.sh
```
Specify the directory of datasets with `data-path`, the weight path of teacher `teacher-path`, the weight path of teacher `student-path`.

### Domain-agnotic Distillation
To train DeiT-small with teacher DINOv2-base on GTA on a single node with 4 gpus for 300 epochs run:
```shell
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --batch-size 32 --epochs 300 --input-size 512 --global_crops_size 512 \
    --data-set URBAN --data-path GTA5/images/ --domain-distillation \
    --teacher-model vit_base --target_model vit_small --model distillation_models_deit \
    --teacher-path dinov2_vitb14_pretrain.pth \
    --student-path  \
    --patch_size 16 --mask_probability 0.5 --mask_ratio 0.5 --mask_first_n \
    --lambda_token 1.0 --lambda_fea 1.0 --lambda_patch 1.0 \
    --output_dir log
```
or run training script:

```shell
bash train_script\dinov2_B_distill_deit_S.sh
```
Specify the directory of datasets with `data-path`, the weight path of teacher `teacher-path`, the weight path of teacher `student-path`.


## Acknowledgment

This part is heavily build upon [Proteus](https://github.com/BeSpontaneous/Proteus-pytorch/tree/main), [DINOv2](https://github.com/facebookresearch/dinov2) and [EVA02](https://github.com/baaivision/EVA/tree/master/EVA-02). We gratefully thank the authors for their wonderful works.