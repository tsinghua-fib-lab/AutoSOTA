<p align="center">
   <h1 align="center">
      <img src="images/pet_dino.png" width="60" style="vertical-align: middle; margin-top: 6px;">
      PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training
    </h1>
   <h3 align="center">CVPR 2026 <span style="color: red;">Highlight</span> 🔥</h3>
  
  <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=xG8B1vsAAAAJ&hl=zh-CN&oi=ao">Weifu Fu</a><sup>1,*,†</sup>,
    <a href="https://scholar.google.com.hk/citations?user=4H2mSI0AAAAJ&hl=zh-CN&authuser=1&oi=sra">Jinyang Li</a><sup>2,*</sup>,
    <a href="https://csgaobb.github.io/">Bin-Bin Gao</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=QuMf688AAAAJ&hl=en">Jialin Li</a><sup>3</sup>
    <br>
    <a href="https://scholar.google.com.hk/citations?user=pfyGbuEAAAAJ&hl=zh-CN&oi=ao">Yuhuan Lin</a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?user=nmNQjgIAAAAJ&hl=zh-CN">Hanqiu Deng</a><sup>1</sup>,
    <a href="https://scholar.google.com.hk/citations?user=jRDPE2AAAAAJ&hl=zh-CN&oi=ao">Wenbing Tao</a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=aqvFa1EAAAAJ&hl=en">Yong Liu</a><sup>1</sup>,
    <a href="https://scholar.google.com/citations?user=fqte5H4AAAAJ&hl=en">Chengjie Wang</a><sup>1</sup>
    <br>
    <br>
    <sup>1</sup>YouTu Lab, Tencent &nbsp;&nbsp;
    <sup>2</sup>Huazhong University of Science and Technology &nbsp;&nbsp;
    <sup>3</sup>Kling Team, Kuaishou Technology
    <br>
    <br>
    <sup>*</sup>Equal Contribution. &nbsp;&nbsp; <sup>†</sup>Corresponding Author.
  </p>

  <p align="center">
    <a href="https://arxiv.org/pdf/2604.00503"><img src="https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
    &nbsp;
    <a href="https://fuweifuvtoo.github.io/pet-dino"><img src="https://img.shields.io/badge/Project-Website-green.svg" alt="Home Page"></a>
  </p>
</p>

## 💡 News
* **[2026.04.09]** 🏆 **PET-DINO** is selected as a **CVPR 2026 Highlight 🔥**!
* **[2026.04.02]** 🚀 Code and pre-trained models have been released.
* **[2026.04.01]** 📄 The [PET-DINO](https://arxiv.org/pdf/2604.00503) paper is now available on arXiv.
* **[2026.02.21]** 🎉 PET-DINO is accepted by **CVPR 2026**!

## 📖 Introduction

PET-DINO is a universal detector supporting both text and visual prompts. 

- Alignment-Friendly Visual Prompt Generation (**AFVPG**): PET-DINO efficiently integrates visual cues while reducing development costs.  
- **The first training strategies for open-set detection**: Intra-Batch Parallel Prompting (**IBP**), Dynamic Memory-Driven Prompting (**DMD**).  
- While enhancing the model’s ability to detect complex and domain-specific objects, PET-DINO preserves the original capability of the text prompt pathway and adapts well to **diverse real-world scenarios** with strong **open-set classification** ability.

**Text Prompt** enables detection of objects described by text labels (e.g., "zebra . giraffe . bird"), while **Visual Prompt** enables detection using visual cues and can be evaluated in two modes: **Visual-I** and **Visual-G**.

## 🛠️ Environment

Please first install following the instructions in the [get_started](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/get_started.md) section, then you need to install additional dependency packages:

```bash
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

> **Note**: The LVIS third-party library does not currently support numpy >= 1.24. Please ensure your numpy version meets the requirements. It is recommended to install `numpy==1.23`.

## 📂 Data Preparation

### Pretrained Weights

Download the following pretrained models and place them under `pretrained/`:

| Model | Path |
| --- | --- |
| Swin-T | `pretrained/swin/swin_tiny_patch4_window7_224.pth` |
| Swin-L | `pretrained/swin/swin_large_patch4_window7_224.pth` |
| BERT | `pretrained/bert-base-uncased/` |
| [MM-Grounding-DINO Swin-T](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth) | `pretrained/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth` |
| [MM-Grounding-DINO Swin-L](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth) | `pretrained/grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth` |

For downloading BERT weights, please refer to the [MM-Grounding-DINO usage guide](configs/mm_grounding_dino/usage.md#download-bert-weight).

### Datasets

For detailed data preparation instructions, please refer to the [MM-Grounding-DINO dataset preparation guide](configs/mm_grounding_dino/dataset_prepare.md).

**Training data**: [Objects365v1](configs/mm_grounding_dino/dataset_prepare.md#1-objects365v1)

**Evaluation datasets**:
- [COCO 2017](configs/mm_grounding_dino/dataset_prepare.md#1-coco-2017)
- [LVIS 1.0 (minival)](configs/mm_grounding_dino/dataset_prepare.md#2-lvis-10)
- [ODinW35](configs/mm_grounding_dino/dataset_prepare.md#3-odinw)

The overall data directory structure should look like this:

```text
data/
├── objects365v1/
│   ├── objects365_train.json
│   ├── objects365_val.json
│   ├── o365v1_train_od.json
│   ├── o365v1_label_map.json
│   ├── train/
│   └── val/
├── coco/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   ├── train2017/
│   └── val2017/
├── lvis/
│   ├── annotations/
│   │   ├── lvis_v1_train.json
│   │   ├── lvis_v1_val.json
│   │   ├── lvis_v1_minival_inserted_image_name.json
│   ├── train2017/
│   └── val2017/
└── odinw/
    ├── AerialMaritimeDrone/
    │   ├── large/
    │   │   ├── test/
    │   │   ├── train/
    │   │   └── valid/
    │   └── tiled/
    ├── AmericanSignLanguageLetters/
    ├── Aquarium/
    └── ...  (35 datasets in total)
```

## 🚀 Train

```bash
bash tools/dist_train.sh configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py 8 --auto-scale-lr
```

## 📊 Evaluation

Checkpoints for quick-start Evaluation and Inference can be downloaded from [HuggingFace](https://huggingface.co/fuweifu/PET-DINO/tree/main).

### COCO

```bash
# Visual-I
CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual'

# Text
CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Text'

# Visual-G (extract embedding first, then evaluate)
python scripts/image_demo.py data/coco/train2017 \
    $CONFIG --weights $CHECKPOINT \
    --prompt_type 'Visual' \
    --input_od_json 'images/cocoTrain_odvg_16.json' \
    --extract-visual-embedding --out-dir ./outputs/visual_embedding_coco/

CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual' \
    test_cfg.type='CustomTestLoop' \
    test_cfg.prompt_visual_embedding_path='./outputs/visual_embedding_coco/'
```

### LVIS minival

```bash
# Visual-I
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_lvis_minival_vi.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual'

# Text
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_lvis_minival_vg_text.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Text'

# Visual-G
python scripts/image_demo.py data/lvis \
    $CONFIG --weights $CHECKPOINT \
    --prompt_type 'Visual' \
    --input_od_json 'images/lvisTrain_odvg_16.json' \
    --extract-visual-embedding --out-dir ./outputs/visual_embedding_lvis/

CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_lvis_minival_vg_text.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual' \
    test_cfg.type='CustomTestLoop' \
    test_cfg.prompt_visual_embedding_path='./outputs/visual_embedding_lvis/'
```

### ODinW35

```bash
# Visual-I
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_odinw35.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Visual'
# Manually copy the output to the `sample_text` variable in `scripts/extract_bbox_map_from_odinw_str_results.py`, then run:
python scripts/extract_bbox_map_from_odinw_str_results.py

# Text
CONFIG=configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_odinw35.py
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
    --cfg-options model.test_cfg.prompt_type='Text'
# Manually copy the output to the `sample_text` variable in `scripts/extract_bbox_map_from_odinw_str_results.py`, then run:
python scripts/extract_bbox_map_from_odinw_str_results.py

# Visual-G
CONFIG=./configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
ODinW35_CONFIG=./configs/pet_dino/val/pet_dino_swin-t_8xb4_12e_obj365_evaluate_odinw35.py
bash scripts/evaluate_visual-G_of_odinw35.sh $CONFIG $ODinW35_CONFIG $CHECKPOINT 8
# Calculate the average mAP across all ODINW datasets
python scripts/get_avg_map_of_odinw.py
```

## ⚡ Inference

### Text Prompt

```bash
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT --pred-score-thr 0.3 \
    --texts 'zebra. giraffe. bird' -c
```

### Visual Prompt (Bounding Boxes)

```bash
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT --pred-score-thr 0.3 \
    --prompt_type 'Visual' \
    --prompt_bboxes '[[1291.6, 679.9, 1536.8, 840.0], [845.0, 181.9, 1210.2, 723.5], [638.0, 470.9, 815.0, 704.4]]' \
    --prompt_bboxes_labels '[30, 2, 46]'
```

### Visual Prompt (Visual Embeddings)

```bash
# Extract visual embedding (single image)
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT \
    --prompt_type 'Visual' \
    --prompt_bboxes '[[1291.6, 679.9, 1536.8, 840.0], [845.0, 181.9, 1210.2, 723.5], [638.0, 470.9, 815.0, 704.4]]' \
    --prompt_bboxes_labels '[30, 2, 46]' \
    --extract-visual-embedding --out-dir ./outputs/visual_embedding_animals/

# Inference with visual embedding
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CHECKPOINT --pred-score-thr 0.3 \
    --prompt_type 'Visual' \
    --prompt_visual_embedding_path outputs/visual_embedding_animals/30.pt
```

## 📜 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{fu2026pet,
  title={PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training},
  author={Fu, Weifu and Li, Jinyang and Gao, Bin-Bin and Li, Jialin and Lin, Yuhuan and Deng, Hanqiu and Tao, Wenbing and Liu, Yong and Wang, Chengjie},
  journal={arXiv preprint arXiv:2604.00503},
  year={2026}
}
```

## 🤝 Acknowledgement

This project is built upon the following excellent works:

- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MM-Grounding-DINO](https://arxiv.org/abs/2401.02361): An open and comprehensive pipeline for unified object grounding and detection.

## License

This project is released under the [Apache 2.0 license](LICENSE).
