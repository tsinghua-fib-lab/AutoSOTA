# Data
## Data Overview

|Data name| Json file | Json_Size | Images_Size|
| --- | ---| ---| ---|
| [VersaD](https://huggingface.co/datasets/FitzPC/VHM_VersaD)|[list_pretrain.json](https://huggingface.co/datasets/FitzPC/VHM_dataset_pretrain/resolve/main/list_pretrain.json) | 1.02 GB | 140 GB
|[VHM_SFT](https://huggingface.co/datasets/FitzPC/VHM_dataset_sft/tree/main)| [list_sft.json](https://huggingface.co/datasets/FitzPC/VHM_dataset_sft/blob/main/list_sft.json) | 1.5 GB | 124 GB
|[VHM_Eval](https://huggingface.co/datasets/FitzPC/VHM_eval_dataset/tree/main)| - |- | 7.35 GB

### VersaD Dataset
This dataset is curated from crowdai, cvact, cvusa, fmow, loveda, millionAID, etc, resulting in total 140M high-quality image-text pairs with the help of powerful Gemini-Vision. The dataset is used for the pretraining of VHM.

### SFT Dataset
This dataset contains VersaD-Instruct，HnstD and VariousRS-Instruct sub datasets. The images in these datasets come from public datasets such as BANDON,DOIR,DOTA,FBP,METER-ML,MSAR,Mts-WH,NWPU-RESISC45,RSITMD,RSVQA,UCM,crowdAI,deepglobe,fair1M and fmow, the instruction portion is based on their original labels.
### VHM_Eval Dataset 
This dataset is a collection of all evaluation data in the [paper](https://arxiv.org/pdf/2403.20213), including Table 5-9 and Table 11. The specific relationship between tasks and corresponding files is as follows:
|Task name |Question type| Json file 
| --- | --- |---|
|___Honst Tasks___|||
|Honst-presence|open-end|presence_mo_dota.json|
|Honst-object absolute position|multi-choice|abspos_dota-test_mc.json|
|Honst-object absolute position-false|multi-choice|abspos_false_dota-test_adversarial_mc.json<br>abspos_false_dota-test_popular_mc.json<br>abspos_false_dota-test_random_mc.json|
|Honst-object relative position|multi-choice|relpos_dota-test_mc.json|
|Honst-object relative position-false|multi-choice|relpos_false_dota-test_adversarial_mc.json<br>relpos_false_dota-test_popula_mc.json<br>relpos_false_dota-test_random_mc.json|
|Honst-clolor| open-end |color_dota-test_fair1m-val_open.json|
|Honst-color-false| open-end |color_false_dota-test_random_open.json <br>color_false_dota-test_popular_open.json<br>   color_false_dota-test_adversarial_open.json<br>|
|Honst-clolor-pan false| open-end |color_false_pan_dota-test.json|
|___VariousRS Tasks___|||
|Scene Classification|open-end|cls_WHU_RS19.json <br>cls_SIRI_WHU.json<br>cls_NWPU_RESISC45.json<br>cls_METER_ML.json<br>cls_AID.json|
|Building footprint vectorization |open-end|bfv_crowdai_val.json|
|Counting|open-end|counting_dota-test_open.json|
|Image Resolution|open-end|gsd_dota_fbp.json|
|Image Modality|multi-choice|imgType_mcq.json|
|Multi-label Classification|open-end|mlc_fbp_test.json<br>mlc_gid_test.json|
|Geometric Measurement|open-end|obj_meas_dota_test.json|
|RSVQA-HR*|open-end|RSVQA_HR-comp_RSVQA.json<br>RSVQA_HR-presence_RSVQA.json|
|RSVQA-LR*|open-end|RSVQA_LR-presence_RSVQA.json<br>RSVQA_LR-presence_RSVQA.json<br>RSVQA_LR-rural_urban_RSVQA.json|
|Visual Grounding| open-end|VG_DOIR_RSVG_test.json|

\* means that the dataset is a randomly sampled subset; you need to download the entire dataset yourself.

## Data preparation
### Pretrain stage dataset preparation

1. Please download the [VersaD](https://huggingface.co/datasets/FitzPC/VHM_VersaD) dataset.
2. Prepare the datasets according to the file structure shown below, where `pretrain_base` denotes the root directory of the entire pretrain dataset.

```
{pretrain_base}/
    # image dirs
    crowdai/
        image0.jpg
        image1.jpg
        ...
        imagexx.jpg
    cvusa/
    ...

    # json files
    list_pretrain.json
```

### SFT stage dataset preparation
1. Please download the  [VHM_SFT](https://huggingface.co/datasets/FitzPC/VHM_dataset_sft/tree/main) dataset.
2. Prepare the datasets according to the file structure shown below, where `sft_base` denotes the root directory of the entire SFT dataset.

```
{sft_base}/
    # image dirs
    BANDON/
        image0.jpg
        image1.jpg
        ...
        imagexx.jpg
    DOTA-train/
    ...

    # json files
    list_sft.json
```


**Important notice**: For the convenience, we provide a zip file for web data. These images must be used for academic purpose.