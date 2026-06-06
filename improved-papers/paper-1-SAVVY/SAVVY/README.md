<h1 align="center">SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing
</h1>
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/NeurIPS%202025-Oral-7D3CFF?style=flat-square"></a>
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
  * Equal contribution
</h4>



## Overview

3D spatial reasoning in dynamic audio-visual environments remains largely unexplored by current Audio-Visual LLMs (AV-LLMs). SAVVY is a training-free reasoning pipeline that enhances AV-LLMs by recovering object trajectories and constructing a unified global 3D map for spatial question answering.

This repository contains only the code for the SAVVY algorithm. The benchmark dataset, annotations, and evaluation data for SAVVY-Bench are provided separately on [HuggingFace](https://huggingface.co/datasets/uwneuroai/SAVVY-Bench). In addition, we also host another repository for SAVVY-Bench evaluation code to support a variety of existing AV-LLMs, please refer to [repo](https://github.com/shlizee/savvy/tree/main/SAVVY-Bench) for further details.

If this repo or benchmark helps your research, a star ⭐ is appreciated.

## Environment Setup
To install all required Python dependencies and set up the environment used by the SAVVY pipeline, run:
```
sh install.sh
```
## Dataset and Benchmark setup
To obtain the video and audio data used by SAVVY, please follow the README.md under `data_utils` folder to perform preprocessing, and then save the processed video data under `data/videos` folder. For other additional modality data, you can follow also the scripts under that folder to obtain. Before preprocessing, please follow [Project Aria's official license agreement](https://www.projectaria.com/datasets/aea/license/) and go through their official process to download the original data. The original dataset is very large in storage (hundreds GB+), but only a small fraction of the data is needed for SAVVY, therefore we additionally prepare two QA demo samples (set `RUN_DEMO = True` in both `get_sd_gemini.py` and `get_seg.py`)  if you want to skip the data preparation stage and try out SAVVY pipeline in an end2end manner.


## SD — Snapshot Descriptor

The **Snapshot Descriptor (SD)** module is the first step of the SAVVY pipeline. Given a query \(Q\) and video, SD prompts the AV-LLM **once** to produce a structured description containing:

- Estimate the relevant time span of the queried event. 
- Determine whether the question is **egocentric** or **allocentric**.
- Identify up to 3 object roles:
  - **Target object** (sound source / queried object)  
  - **Reference object** (anchor point for allocentric frames)  
  - **Facing object** (defines orientation for third-person coordinate frame)

Each object is represented by:
- A textual description.  
- An **egocentric trajectory** estimated from AV-LLM outputs.

To obtain the Snapshot Descriptor of the query, run: 

`python scripts/get_sd_gemini.py`

## Seg — Text-Guided Snapshot Segmentation

The Seg module refines object trajectories by using visual foundation models to fill in missing observations, particularly for dynamic or briefly visible objects. Using the textual phrases from the SD module, **Seg** samples **N** frames from the video and applies **text-guided segmentation** (CLIPSeg, SAM2) to extract object masks. From each mask, it computes the centroid to estimate **azimuth** (θ) and applies **monocular depth estimation** (e.g., ZoeDepth) to obtain **distance** (r). These multi-modal cues are then fused into **refined egocentric trajectories** with up to N points per object.

To obtain refined trajectories using Seg, run the following commands:
```
mkdir third_party
cd third_party
git clone https://github.com/isl-org/ZoeDepth.git
git clone https://github.com/mit-han-lab/efficientvit.git
cd ../scripts
python get_seg.py
```


## SAVVY QA
After obtaining the above QA-related audio-visual cues for each query, we can run the final step (including spatial audio estimation + track aggregation + final prediction) to obtain all predictions of the SAVVY pipeline on SAVVY Bench:

```
python main.py
```
The prediction results will be saved at data/output/predavmap.json

To test SAVVY result on SAVVY-Bench QA, please run:
```
python eval_savvy_qa.py
```

## Citation

If you use SAVVY or SAVVY-Bench in your research, please cite:

```bibtex
@article{chen2025savvy,
  title={SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing},
  author={Chen, Mingfei and Cui, Zijun and Liu, Xiulong and Xiang, Jinlin and Zheng, Caleb and Li, Jingyuan and Shlizerman, Eli},
  journal={arXiv preprint arXiv:2506.05414},
  year={2025}
}
# This paper will appear in NeurIPS 2025
```
