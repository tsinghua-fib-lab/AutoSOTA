<h1 align="center">SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing
</h1>
<p align="center">
  <a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/115001">
    <img src="https://img.shields.io/badge/NeurIPS%202025-Oral-7D3CFF?style=flat-square"></a>
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2506.05414">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2506.05414-B31B1B?style=flat-square&logo=arxiv&logoColor=white">
  </a>
  <a href="https://zijuncui02.github.io/SAVVY/">
    <img src="https://img.shields.io/badge/Project%20Page-3A6EA5?style=flat-square&logo=googlechrome&logoColor=white">
  </a>
</p>
<h4 align="center" style="color:gray">
  <a href="https://www.mingfeichen.com/" target="_blank">Mingfei Chen*</a>,
  <a href="https://zijuncui.com/" target="_blank"> Zijun Cui*</a>,
    <a href="https://dragonliu1995.github.io/" target="_blank">Xiulong Liu*</a>,
  <a href="https://xiangjinlin.com/" target="_blank"> Jinlin Xiang</a>,
  <a href="https://www.linkedin.com/in/yang-caleb-zheng/" target="_blank"> Caleb Zheng</a>,
  <a href="https://christincha.github.io/" target="_blank"> Jingyuan Li</a>,
  <a href="https://faculty.washington.edu/shlizee/NW/team.html" target="_blank"> Eli Shlizerman</a> <br>
  University of Washington
  * Equal contribution
</h4>

## Overview

3D spatial reasoning in dynamic audio-visual environments remains largely unexplored by current Audio-Visual LLMs (AV-LLMs). **SAVVY** introduces a training-free reasoning pipeline that enhances AV-LLMs by recovering object trajectories and constructing unified global 3D maps for spatial question answering. Alongside the algorithm, we present **SAVVY-Bench**, the first benchmark for evaluating dynamic 3D spatial reasoning in audio-visual environments.

⭐ If this project helps your research, a star is appreciated!



## Project Components

This project consists of three main components, each hosted in separate repositories:


### Quick Links
| Component | Folder | Description |
|-----------|------------|-------------|
| **SAVVY Algorithm** | [SAVVY](https://github.com/shlizee/savvy/tree/main/SAVVY/) | Training-free spatial reasoning pipeline; data preprocessing tools, benchmark data, annotations |
| **SAVVY-Bench Dataset** | [HuggingFace](https://huggingface.co/datasets/uwneuroai/SAVVY-Bench) | Preview QA pairs in the Hugging Face Data Studio|
| **Evaluation Code** | [SAVVY-Bench](https://github.com/shlizee/savvy/tree/main/SAVVY-Bench/) | Multi-model benchmarking framework |
---

### [SAVVY Algorithm](https://github.com/shlizee/savvy/tree/main/SAVVY/)
**Main Algorithm Repository** - Core SAVVY pipeline implementation

The SAVVY algorithm employs a multi-stage reasoning approach:
- **Snapshot Descriptor (SD)**: Prompts AV-LLMs to extract structured descriptions and initial object trajectories
- **Text-Guided Segmentation (Seg)**: Refines trajectories using visual foundation models (CLIPSeg, SAM2, ZoeDepth)
- **Spatial Audio & Track Aggregation**: Fuses multi-modal cues into a unified 3D map for final predictions

**Get Started**: Visit the [SAVVY repository](https://github.com/shlizee/savvy/tree/main/SAVVY) for installation, usage instructions, and algorithm details.

---

### [SAVVY-Bench Dataset](https://huggingface.co/datasets/uwneuroai/SAVVY-Bench)
**Hugging Face Repository** - Benchmark dataset, annotations, and data preprocessing tools

SAVVY-Bench is constructed through a four-stage pipeline combining automated tools with human validation:
1. **Data Preprocessing**: Undistorts videos, aligns multi-stream recordings, processes sensor data
2. **Annotation**: Objects and sounding events with 3D spatial annotations
3. **QA Synthesis**: Template-based generation of structured question-answer pairs
4. **Quality Review**: Human verification ensuring precision

**Access Dataset**: 
- Download from [Hugging Face](https://huggingface.co/datasets/uwneuroai/SAVVY-Bench)
- Data preprocessing scripts available in the [SAVVY repository](https://github.com/shlizee/savvy/tree/main/SAVVY) under `data_utils/` folder

---

### [AV-LLM Evaluation Code on SAVVY-Bench](https://github.com/shlizee/savvy/tree/main/SAVVY-Bench)
**Evaluation Repository** - We currently support the evaluation of the following AV-LLMs on SAVVY-Bench:
- [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
- [MiniCPM-o2.6](https://github.com/OpenBMB/MiniCPM-o)
- [Video-SALMONN (video-salmonn-13b)](https://github.com/bytedance/SALMONN)
- Gemini 2.5 Flash / Pro
- [Ola](https://github.com/Ola-Omni/Ola)
- [EgoGPT](https://github.com/EvolvingLMMs-Lab/EgoLife)
- [Longvale](https://github.com/ttgeng233/LongVALE)


**Run Benchmarks**: Visit the [SAVVY-Bench evaluation repository](https://github.com/shlizee/savvy/tree/main/SAVVY-Bench) for setup instructions and evaluation scripts.




## Citation

If you use SAVVY or SAVVY-Bench in your research, please cite:

```bibtex
@article{chen2025savvy,
  title={SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing},
  author={Chen, Mingfei and Cui, Zijun and Liu, Xiulong and Xiang, Jinlin and Zheng, Caleb and Li, Jingyuan and Shlizerman, Eli},
  journal={arXiv preprint arXiv:2506.05414},
  year={2025}
}
% This paper will appear in NeurIPS 2025
```

## License

Please refer to individual component licenses:
- SAVVY Algorithm: See [LICENSE](https://github.com/shlizee/savvy/tree/main/SAVVY/LICENSE) 
- Original Aria Dataset: [Project Aria License Agreement](https://www.projectaria.com/datasets/aea/license/)
- [EgoLifter submodule](https://github.com/facebookresearch/egolifter): [Apache License 2.0](https://github.com/facebookresearch/egolifter/blob/main/LICENSE)

---
