# VDOT: Efficient Unified Video Creation via Optimal Transport Distillation
<a href="https://arxiv.org/abs/2512.06802"><img src='https://img.shields.io/badge/VDOT-arXiv-red' alt='Paper PDF'></a>
<a href="https://vdot-page.github.io/"><img src='https://img.shields.io/badge/VDOT-Project_Page-green' alt='Project Page'></a>
<a href="https://huggingface.co/yutongwang1012/VDOT"><img src='https://img.shields.io/badge/VDOT-HuggingFace_Model-yellow'></a>
<a href="https://huggingface.co/datasets/yutongwang1012/UVCBench"><img src='https://img.shields.io/badge/UVCBench-HuggingFace_Dataset-yellow'></a>

[Yutong Wang](https://yutongwang1012.github.io/)<sup>1</sup>, [Haiyu Zhang](https://github.com/aejion)<sup>3,2</sup>, [Tianfan Xue](https://tianfan.info/)<sup>4,2</sup>, [Yu Qiao](https://mmlab.siat.ac.cn/yuqiao/)<sup>2</sup>, [Yaohui Wang](https://wyhsirius.github.io/)<sup>2*</sup>, [Chang Xu](http://changxu.xyz/)<sup>1*</sup>, [Xinyuan Chen](https://scholar.google.com/citations?user=3fWSC8YAAAAJ&hl=zh-CN)<sup>2*</sup>

<sup>1</sup>USYD, <sup>2</sup>Shanghai AI Laboratory, <sup>3</sup>BUAA, <sup>4</sup>CUHK

## Introduction
<strong>VDOT</strong> is an efficient, unified video creation model that achieves high-quality results in just 4 denoising steps. 
By employing Computational Optimal Transport (OT) within the distillation process, VDOT ensures training stability and enhances both training and inference efficiency.
VDOT unifies a wide range of capabilities, such as <strong>Reference-to-Video (R2V)</strong>, <strong>Video-to-Video (V2V)</strong>, <strong>Masked Video Editing (MV2V)</strong>, and arbitrary <strong>composite tasks</strong>, matching the versatility of VACE with significantly reduced inference costs.

https://github.com/user-attachments/assets/e474c322-b5d2-4617-a198-e7cbde138004

## 🎉 News
- [x] Mar 15, 2026: 🔥Release code of model training, inference, and gradio demos. 
- [x] Mar 14, 2026: 🔥VDOT-14B is now avaiable at [HuggingFace](https://huggingface.co/yutongwang1012/VDOT). 
- [x] Mar 14, 2026: 🔥UVCBench is now avaiable at [HuggingFace](https://huggingface.co/datasets/yutongwang1012/UVCBench). 
- [x] Feb 21, 2026: [VDOT](https://vdot-page.github.io/) is accepted by CVPR 2026.
- [x] Dec 7, 2025: We propose [VDOT](https://vdot-page.github.io/), a 4-step unified video creation model based on VACE.


## ⚙️ Installation
The codebase was tested with Python 3.10.13, CUDA version 12.4, and PyTorch >= 2.5.1.

```bash
git clone https://github.com/hhhh1138/VDOT.git && cd VDOT
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124  # If PyTorch is not installed.
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1
```

### Setup for Preprocess Tools 
For the preprocessing tools, such as extracting the source video and mask video of **classic video creation tasks (such as V2V and MV2V tasks)**, we recommend using the annotator tools in the [VACE](https://github.com/ali-vilab/VACE) repository. 

For the **more complicated creation tasks**, such as video character replacement and video try-on tasks, they typically involve more complex input conditions: a restricted pose source video, a mask video for the target area, and images of the person or garments to be replaced. 
For details on the processing methods, please refer to our [scripts]() folder.

### Local Directories Setup
We recommend to organize local directories as:
```angular2html
VDOT
├── ...
├── benchmarks
│   ├── VACE-Benchmark
│   └── UVCBench
├── models
│   ├── VACE-Annotators
│   ├── VACE-Wan2.1-14B
│   └── VDOT ### (download from [huggingface](https://huggingface.co/yutongwang1012/VDOT))
│       └── google
│       ├── vdot-weights
│       ├   └── vdot_14b.pt
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       └── Wan2.1_VAE.pth
├── inference
└── training
```

## 🚀 Usage
In VDOT, users can generate videos based on any combination of input conditions in just four denoising steps. 

### Inference 
```bash
# See the commands in ``run_vdot.sh'', we recommend using 4 GPUs for the inference of VDOT-14B.
torchrun --nproc_per_node=4 --nnodes=1 inference/vace_wan_inference.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \
    --ring_size 1 \
    --size 480p \
    --sample_guide_scale 1 \
    --sample_steps 4 \
    --src_video example_video_1.mp4 \
    --src_mask example_video_2.mp4 \
    --src_ref_images example_image_1.png,example_image_2.png \
    --prompt "xxx"
```

### Gradio Demo 
```bash
torchrun --nproc_per_node=4 --nnodes=1 inference/vdot_gradio.py
```

### Training 
```bash
cd training
bash train_vdot.sh
```

## Acknowledgement
We are grateful for the following awesome projects, including [VACE](https://github.com/ali-vilab/VACE), [Wan](https://github.com/Wan-Video/Wan2.1), and [Self-Forcing](https://github.com/guandeh17/Self-Forcing).


## BibTeX

```bibtex
@article{wang2025vdot,
  title={VDOT: Efficient Unified Video Creation via Optimal Transport Distillation},
  author={Wang, Yutong and Zhang, Haiyu and Xue, Tianfan and Qiao, Yu and Wang, Yaohui and Xu, Chang and Chen, Xinyuan},
  journal={arXiv preprint arXiv:2512.06802},
  year={2025}
}
```
