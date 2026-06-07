# Visual Reasoning Tracer (VRT)

[\[🏠 Project Page\]](https://harboryuan.github.io/visual-reasoning-tracer/) [\[📄 Paper\]](https://arxiv.org/pdf/2512.05091) [\[🤗 VRT-80k\]](https://huggingface.co/datasets/HarborYuan/VisualReasoningTracer) [\[🤗 VRT-Bench\]](https://huggingface.co/datasets/HarborYuan/VRT-Eval)

## Overview

Current Multimodal Large Language Models (MLLMs) often lack transparency in their reasoning processes. To bridge this gap, we introduce **Visual Reasoning Tracer (VRT)**, a task requiring models to explicitly predict intermediate objects in a reasoning path. We present **VRT-Bench** for evaluation, a new metric for reasoning quality, and **VRT-80k**, a large-scale training dataset. Models trained on VRT-80k demonstrate significant improvements in grounded reasoning.

The VRT project is built upon the Sa2VA framework and fully supports the Sa2VA training and evaluation pipeline. For more information about Sa2VA, please refer to the [Sa2VA README](../../README.md).

## Demos

### VRT-Bench Examples

<div align="center">
  <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/fig_bench1.jpg" width="100%"/>
  <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/fig_bench2.jpg" width="100%"/>
  <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/fig_bench3.jpg" width="100%"/>
  <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/fig_bench4.jpg" width="100%"/>
</div>

### Comparison

| GT | R-Sa2VA (Ours) |
| :---: | :---: |
| <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/comp1_gt.jpg" width="100%"/> | <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/comp1_rs2va.jpg" width="100%"/> |

| Gemini 2.5 Pro | Qwen3VL |
| :---: | :---: |
| <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/comp1_gemini.jpg" width="100%"/> | <img src="https://harboryuan.github.io/visual-reasoning-tracer/static/images/comp1_qwen3.jpg" width="100%"/> |

## Model Training

To train the model, use the following command: (same as training other Sa2VA models)

```bash
bash tools/dist.sh train projects/vrt_sa2va/configs_sa2va/vrt_sa2va_4b_qwen3_sft.py 8
```

## Models

We provide the following pretrained models:

| Model | Training | Link |
| :--- | :--- | :--- |
| **R-Sa2VA-Qwen3VL-4B-SFT** | SFT | [HuggingFace](https://huggingface.co/HarborYuan/R-Sa2VA-Qwen3VL-4B-SFT) |
| **R-Sa2VA-Qwen3VL-4B-RL** | SFT + RL | [HuggingFace](https://huggingface.co/HarborYuan/R-Sa2VA-Qwen3VL-4B-RL) |

## Data Preparation

Download the VRT-80k dataset to `data/VRT-Training`:

```bash
uv run huggingface-cli download HarborYuan/VisualReasoningTracer --repo-type dataset --local-dir data/VRT-Training
```

Download the VRT-Bench dataset to `data/VRT-Eval`:

```bash
uv run huggingface-cli download HarborYuan/VRT-Eval --repo-type dataset --local-dir data/VRT-Eval
```

## Evaluation
To evaluate the model on VRT-Bench, use the following command:

```bash
uv run projects/vrt_sa2va/evaluation/vrt_eval_single.py HarborYuan/R-Sa2VA-Qwen3VL-4B-RL --gpus 8
```

## Evaluation Results

We provide the evaluation results on VRT-Bench below.

| Model | R-LQ | R-SQ | A-mIoU |
| :--- | :---: | :---: | :---:  |
| **R-Sa2VA-Qwen3VL-4B-SFT** | 66.3 | 87.3 | 59.5 |
| **R-Sa2VA-Qwen3VL-4B-RL** |  67.0 | 86.7 | 62.1 |

Please refer to the [paper](https://arxiv.org/pdf/2512.05091) for detailed evaluation results.

## Directory Structure

- `configs_sa2va/`: Configuration files for Sa2VA models SFT on VRT-80k.
- `configs_sa2va_rl/`: Configuration files for Sa2VA models RL.
- `data_loader/`: Data loading scripts and utilities.
- `evaluation/`: Evaluation scripts and metrics for assessing reasoning capabilities.
- `models/`: Model definitions and architectures.
- `prompt/`: Prompt templates and engineering for visual reasoning tasks.
- `utils/`: Utility functions and helper scripts.

## Getting Started

Please refer to the [Project Page](https://harboryuan.github.io/visual-reasoning-tracer/) for more detailed information, including the paper, dataset, and additional resources.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yuan2025vrt,
  author    = {Haobo Yuan and Yueyi Sun and Yanwei Li and Tao Zhang and Xueqing Deng and Henghui Ding and Lu Qi and Anran Wang and Xiangtai Li and Ming-Hsuan Yang},
  title     = {Visual Reasoning Tracer: Object-Level Grounded Reasoning Benchmark},
  journal   = {arXiv pre-print},
  year      = {2025},
}
```
