<div align="center">
  <img src="fig/logo.png" width="300px" height="300px">
</div>

# [CVPR2026] Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation

[![arXiv](https://img.shields.io/badge/Paper-arXiv:2603.02554-Green)](https://arxiv.org/pdf/2603.02554) [![BibTex](https://img.shields.io/badge/Paper-BibTex-yellow)](#bibtex)

Official PyTorch implementation of **GKD**, as presented in our paper:

[Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](https://arxiv.org/pdf/2603.02554)

by [Chonghua Lv](https://scholar.google.com/citations?user=VCdfb0gAAAAJ&hl=zh-CN)<sup>1</sup>, [Dong Zhao](https://scholar.google.com/citations?user=j_UjUUUAAAAJ&hl=zh-CN)<sup>2</sup>, Shuang Wang<sup>1</sup>, Dou Quan<sup>1</sup>, Ning Huyan<sup>3</sup>, Nicu Sebe<sup>2</sup>, Zhun Zhong<sup>4</sup>

<sup>1</sup>Xidian University,  <sup>2</sup>University of Trento,  <sup>3</sup>Tsinghua University, <sup>4</sup>Hefei University of Technology

---

:bell: **News:**
-

* [2026-03-02] :zap::zap::zap:We release the training code and model weights.

* [2026-02-21] :fire::fire::fire:GKD has been accepted by **CVPR26**.

---

## :pushpin:Overview
- `Motivation`: Conventional knowledge distillation approaches primarily preserve in-domain accuracy while neglecting out-of-domain generalization, which is essential under distribution shifts. This limitation becomes more severe with the emergence of vision foundation models (VFMs): although VFMs exhibit strong robustness on unseen data, distilling them with conventional KD often compromises this ability.

- `Methodology`: GKD decouples representation learning from task learning. In the first stage, the student acquires domain-agnostic representations through selective feature distillation, and in the second stage, these representations are frozen for task adaptation, thereby mitigating overfitting to visible domains. To further support transfer, we introduce a query-based soft distillation mechanism, where student features act as queries to teacher representations to selectively retrieve transferable spatial knowledge from VFMs. Extensive experiments on five domain generalization benchmarks demonstrate that GKD consistently outperforms existing KD methods, achieving average gains of +1.9% in foundation-to-foundation (F2F) and +10.6% in foundation-to-local (F2L) distillation.

<div align="center">
  <img src="fig/GKD_Fig2.png" width="100%" height="100%">
</div>

### Comparison with State-of-the-Art KD

<div align="center">
  <img src="fig/table_1.png" width="" height="">
</div>

### Comparison with State-of-the-Art Domain Generalization (DG)

<div align="center">
  <img src="fig/table_2.png" width="80%" height="80%">
</div>

---

We provide the models distilled from DINOv2 and task learning:
| Model | Backbone | GTA5 |  cityscapes  | Potsdam-RGB |
| --------------- | --------------- | ------------- |------------- | ------------- |
| DeiT  | ViT-B | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) |
| DeiT  | ViT-S | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) |
| DINO  | ViT-B | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) |
| DINO  | ViT-S | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) | [Baidu Netdisk](https://pan.baidu.com/s/1OiPn1Wc9-s23WkbVE8CnTw?pwd=habv) <br> [Hugging Face](https://huggingface.co/yongers/GKD) |

## :wrench:Setup Environment

For this project, we used python 3.8.13. Create a Python virtual environment using e.g. Conda:

```shell
conda create -n GKD python=3.8.13 && conda activate GKD
```

install PyTorch `v2.0.1`. the requirements can be installed with:

```shell
pip install -r requirements.txt
```

## :ok_hand:Get Started
We provide a comprehensive codebase which contains the implementation of [Task-agnotic Distillation](general_distillation/),  [Domain-agnotic Distillation](general_distillation/) and [Task Learning](task_learning/) and Please go to the folders for specific docs.
## :loud_sound:Acknowledgment

Our codebase is heavily build upon [Proteus](https://github.com/BeSpontaneous/Proteus-pytorch/tree/main), [DINOv2](https://github.com/facebookresearch/dinov2), [EVA02](https://github.com/baaivision/EVA/tree/master/EVA-02) and [Rein](https://github.com/w1oves/Rein/tree/train). We gratefully thank the authors for their wonderful works.


## :memo:Contact
If you have any questions or feedback, feel free to reach out:
- Chonghua Lv: youngerlv@stu.xidian.edu.cn

## :chart_with_upwards_trend:Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Younger-hua/GKD&type=Date)](https://star-history.com/#Younger-hua/GKD&Date)