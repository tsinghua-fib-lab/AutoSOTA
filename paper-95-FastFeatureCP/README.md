# FastFeatureCP(FFCP)

> [**Accelerating Feature Conformal Prediction via Taylor Approximation**]([https://](https://arxiv.org/abs/2412.00653)) 
>
>[Zihao Tang](https://tangzihao1997.github.io/), [Boyuan Wang](https://github.com/ElvisWang1111), [Chuan Wen](https://alvinwen428.github.io/), [Jiaye Teng](https://www.tengjiaye.com/)

<div align="center">
  <img src="./main_concept.jpg" alt="main_concept" width="80%"/>
</div>

## News

***10/08/2025***
  1. Our work is accepted at [NeurIPS2025](https://neurips.cc/virtual/2025/poster/115011).

***12/03/2024***
  1. Our paper is updated on [arXiv](https://arxiv.org/abs/2412.00653).


## Introduction

Conformal prediction is widely adopted in uncertainty quantification, due to its post-hoc, distribution-free, and model-agnostic properties. In the realm of modern deep learning, researchers have proposed Feature Conformal Prediction (FCP), which deploys conformal prediction in a feature space, yielding reduced band lengths. However, the practical utility of FCP is limited due to the time-consuming non-linear operations required to transform confidence bands from feature space to output space. In this paper, we present Fast Feature Conformal Prediction (FFCP), a method that accelerates FCP by leveraging a first-order Taylor expansion to approximate these non-linear operations. The proposed FFCP introduces a novel non-conformity score that is both effective and efficient for real-world applications. Empirical validations showcase that FFCP performs comparably with FCP (both outperforming the Split CP version) while achieving a significant reduction in computational time by approximately 50x in both regression and classification tasks. 

* Feature Conformal Prediction (FCP) : [Paper](https://arxiv.org/pdf/2210.00173) & [Code](https://github.com/AlvinWen428/FeatureCP)

* Conformalized Quantile Regression (CQR) : [Paper](https://arxiv.org/pdf/1905.03222) & [Code](https://github.com/yromano/cqr?utm_source=catalyzex.com)

*  Regularized Adaptive Prediction Sets (RAPS) : [Paper](https://arxiv.org/abs/2009.14193) & [Code](https://github.com/aangelopoulos/conformal_classification/tree/master)

* Localized Conformal Prediction (LCP) : [Paper](https://arxiv.org/pdf/2106.08460)

## Installation

We use Python 3.8, and other packages can be installed by:
```
pip install -r requirements.txt
```

## Usage

### Fast Feature Conformal Prediction (FFCP)

#### One-dim regression
```
cd FastFeatureCP
python main.py --data com
```
#### High-dimensional regression
##### Synthetic dataset:
```
cd FastFeatureCP
python main.py --data x100-y10-reg
```
##### Cityscapes
```
export $CITYSCAPES_PATH = 'your path to the cityscapes'
cd FastFeatureCP_seg
python main_fcn.py --dataset-dir $CITYSCAPES_PATH
```
### Fast Feature Conformalized Quantile Regression (FFCQR)
```
cd FastFeatureCQR
python main_CQR.py --data com
```
### Fast Feature Localized Conformal Prediction (FFLCP)
```
cd FastFeatureLCP
python main_LCP.py --data com
```
### Fast Feature Regularized Adaptive Prediction Sets (FFRAPS)
```
export $imagenet_val_PATH = 'your path to the imagenet_val'
cd FastFeatureLCP
python main_FFRaps.py $imagenet_val_PATH
```
## 3. Citation
If you find our work is helpful to you, please cite our paper:

```
@article{tang2024predictive,
  title={Predictive Inference With Fast Feature Conformal Prediction},
  author={Tang, Zihao and Wang, Boyuan and Wen, Chuan and Teng, Jiaye},
  journal={arXiv preprint arXiv:2412.00653},
  year={2024}
}
```
