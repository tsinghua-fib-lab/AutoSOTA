<div align="center">
<h3>
Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment (ICLR 2026 Oral)
</h3>

Shijie Zhao*, Xuanyu Zhang*, Weiqi Li, Junlin Li, Li Zhang, Tianfan Xue, Jian Zhang

Bytedance Inc.

  </a>
    <a href="https://arxiv.org/abs/2510.11369">
    <img
      src="https://img.shields.io/badge/RALI-Paper-red?logo=arxiv&logoColor=red"
      alt="RALI Paper on arXiv"
    />
  </a>
<a href="https://huggingface.co/ByteDance/Q-Insight">
    <img 
        src="https://img.shields.io/badge/RALI-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="Q-Insight Family Model"
    />
</a>


</div>

## 🚩 Updates
- 2026.02.14 The inference code of RALI is released!
- 2026.01.26 RALI has been accepted at ICLR 2026 as an **oral** presentation!

## 🔥 Introduction
We revisit the reasoning mechanism in MLLM-based IQA model (such as Q-Insight) and propose a CLIP-based lightweight image scorer RALI. We verifies that through RL training, MLLMs leverage their reasoning capability to convert redundant visual representations into compact, cross-domain aligned text representations. This conversion is the source of the generalization exhibited by these reasoning-based IQA models. RALI uses only about 4% of Q-Insight’s parameters and inference time, while achieving comparable accuracy.

<p align="center">
  <img src="assets/teaser_rali.png">
</p>


## 🔧 Dependencies and Installation
```bash
git clone https://github.com/xuanyuzhang21/RALI.git
bash setup.sh
```

## ⚡ Quick Inference
Please download the **RALI** pretrained weights from the [link](https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI). After downloading, place the checkpoint under `./checkpoints`, so that the directory structure becomes:

```text
RALI/
├── checkpoints/
│   ├── ckpt.pt
│   ├── pca.pkl
│   ├── basis.npz
│   └── best/
│       ├── config.json
│       ├── pytorch_model.bin (or *.safetensors)
│       ├── preprocessor_config.json
│       └── ...
```
Then run the following code: 
```bash
python demo_rali_score.py
```

## 📖 Dataset Preparation
Download meta files and source images from [Data-DeQA-Score](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main) and arrange the folders as follows:
```
|-- RALI 
    |-- Data-DeQA-Score
        |-- KONIQ
            |-- images/*.jpg
            |-- metas
        |-- KADID10K
            |-- images/*.png
            |-- metas
        |-- SPAQ
            |-- images/*.jpg
            |-- metas     
        ... 
```

## Evaluation
Run the following code to reproduce the results of our paper. Change the `--test_json` to the path of your testing json.

```bash
bash eval_json.sh
```

## Acknowledgement
We appreciate the releasing codes and data of [Q-Insight](https://github.com/bytedance/Q-Insight)  and [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score).

## Citation
```
@article{zhao2025reasoning,
  title={Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment},
  author={Zhao, Shijie and Zhang, Xuanyu and Li, Weiqi and Li, Junlin and Zhang, Li and Xue, Tianfan and Zhang, Jian},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
