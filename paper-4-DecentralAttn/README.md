<div align="center">
  <h2><b> Code for ICLR26 (Oral):</b></h2>
  <h2><b> Decentralized Attention Fails Centralized Signals: Rethink Transformers for Medical Time Series </b></h2>
</div>

<div align="center">

**[<a href="https://openreview.net/forum?id=oZJFY2BQt2">Paper</a>]**
**[<a href="https://zhuanlan.zhihu.com/p/1979237006845441203">中文解读1</a>]**
**[<a href="https://paper.dou.ac/p/2602.18473">中文解读2</a>]**

</div>

## Introduction

### 1. Mismatch between centralized MedTS signals and decentralized Attention.

<p align="center">
<img width="1153" height="518" alt="image" src="https://github.com/user-attachments/assets/807c224c-1f43-45d9-906e-2848aed4e993" />
</p>

### 2. Reprogram decentralized Attention into the centralized CoTAR block.

<p align="center">
<img width="1166" height="420" alt="image" src="https://github.com/user-attachments/assets/f20f535d-1eba-4858-8637-abfca6c0fb0a" />
</p>

### 3. TeCh: a unified CoTAR-based framework that captures temporal, channel, or both via adaptive tokenization.

<p align="center">
<img width="1156" height="543" alt="image" src="https://github.com/user-attachments/assets/89b07193-142b-4241-8ca4-894172db9341" />
</p>

### 4. Improved effectiveness, higher efficiency, and stronger robustness.

<p align="center">
<img width="1146" height="599" alt="image" src="https://github.com/user-attachments/assets/c6bc4a55-9aa1-46a6-90b9-eafe2ac60840" />
</p>

## Usage

1. Install requirements.
```
pip install -r requirements.txt
```

2. Prepare data. You can download all datasets from [**Medformer**](https://github.com/DL4mHealth/Medformer). **All the datasets are well pre-processed** *(except for the TDBrain dataset, which requires permission first)* and can be used easily thanks to their efforts. Then, place all datasets under the folder
```
./dataset
```

4. Train the model. We provide the experiment scripts of all benchmarks under the folder
```
./scripts
```
5. For example, you can use the command line  below to get the result of  **APAVA**. The whole training history is under the ***'./logs'*** folder.
```
bash ./scripts/APAVA.sh
```

## Citation
If you find this repo helpful, please cite our paper.

```
@inproceedings{
yu2026tech,
title={Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series},
author={Guoqi Yu and Juncheng Wang and Chen Yang and Jing Qin and Angelica I Aviles-Rivero and Shujun Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=oZJFY2BQt2}
}
```

## Acknowledgement

This project is built on the code in the repo [**Medformer**](https://github.com/DL4mHealth/Medformer).
**Thanks a lot for their amazing work!**

***Please also star their project and cite their paper if you find this repo useful.***
```
@article{wang2024medformer,
  title={Medformer: A multi-granularity patching transformer for medical time-series classification},
  author={Wang, Yihe and Huang, Nan and Li, Taida and Yan, Yujun and Zhang, Xiang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={36314--36341},
  year={2024}
}
```

