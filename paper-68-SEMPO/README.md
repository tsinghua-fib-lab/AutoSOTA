<div align="center">
  <h2><b>SEMPO: Lightweight Foundation Models for Time Series Forecasting </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/mala-lab/SEMPO?color=green)
![](https://img.shields.io/github/stars/mala-lab/SEMPO?color=yellow)
![](https://img.shields.io/github/forks/mala-lab/SEMPO?color=lightblue)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

SEMPO is a novel time series foundation model with significantly reduced model size and pre-training scale, yet demonstrating superior generalization ability on diverse downstream forecasting tasks.

## Updates/News

* Oct 2025: Release of SEMPO library, along with SEMPO preprint now on [arXiv](https://arxiv.org/pdf/2510.19710).
  
* Sep 2025: The [SEMPO Paper](https://arxiv.org/pdf/2510.19710) has been accepted to NeurIPS 2025 as a Poster presentation!

## Introduction

> This work proposes SEMPO, a novel lightweight foundation model that requires pretraining on relatively small-scale data, yet exhibits strong general time series forecasting. SEMPO comprises two key modules: 1) energy-aware SpEctral decomposition module, that substantially improves the utilization of pre-training data by modeling not only the high-energy frequency signals but also the low-energy yet informative frequency signals that are ignored in current methods; and 2) Mixture-of-PrOmpts enabled Transformer, that learns heterogeneous temporal patterns through small dataset-specific prompts and adaptively route time series tokens to these prompt-based experts for parameter-efficient model adaptation across different datasets and domains. Equipped with these modules, SEMPO significantly reduces both pre-training data scale and model size, while achieving strong generalization. 


<p align="center">
    <img src="figures/framework.png" alt="" align="center" width="700px" />
</p>

## 📚 Pretraining Data

All pretraining datasets, except PEMS04 and PEMS07, use the [numpy format UTSD](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/). You can access the datasets from [Google Drive](https://drive.google.com/drive/my-drive?dmr=1&ec=wgc-drive-hero-goto), then place the downloaded contents under `./dataset`. 

## ⚙️ Installation

Install Pytorch 2.1.2+cu118 with Python 3.10, and then install the dependencies:

```setup
pip install -r requirements.txt
```

## 🏋️ Pretraining

To pretrain the model(s) in the paper, run the follow command, which supports both single-GPU and multi-GPU execution on a single node. For convenience, we provide a single-GPU pretrained model in the folder `./checkpoints/`.

```pre-training
bash ./scripts/time_series_forecasting/pretrain/sempo_utsd.sh
```

## 🔥 Fine-tuning

To fine-tune the model(s) in the paper, use the few-shot examples in the folder `./scripts/time_series_forecasting/few_shot`. Run with --is_pretraining 0, --is_training 1, and --is_zeroshot 0, using two configurations of 5% and 10%. For example:

```fine-tuning
bash ./scripts/time_series_forecasting/few_shot/sempo_ETTh1.sh
```

## 🧊 Evaluation

To evaluate the model(s) in the paper, use the zero-shot examples in the folder `./scripts/time_series_forecasting/zero_shot`. Run with --is_pretraining 0, --is_training 0, and --is_zeroshot 1. For example:

```eval
bash ./scripts/time_series_forecasting/zero_shot/sempo_weather.sh
```

To evaluate other advanced foundation models such as Chronos-Bolt, first download their pretrained weights from [HuggingFace](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444) and place the downloaded weights under `./models`. Then use the zero-shot examples above and run with: 

```eval
--task_name long_term_forecast_chronos --model Chronos
```

## Citation

If you're using this repo in your research or applications, please cite our paper:

```
@misc{he2025sempolightweightfoundationmodels,
      title={SEMPO: Lightweight Foundation Models for Time Series Forecasting}, 
      author={Hui He and Kun Yi and Yuanchi Ma and Qi Zhang and Zhendong Niu and Guansong Pang},
      year={2025},
      eprint={2510.19710},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.19710}, 
}
```

## Acknowledgement

We appreciate the following GitHub repos for providing valuable code bases and efforts.

- Time-MoE [\[repo\]](https://github.com/Time-MoE/Time-MoE)
- TTM [\[repo\]](https://github.com/glehet/TTM1)
- uni2ts [\[repo\]](https://github.com/SalesforceAIResearch/uni2ts)
- chronos-forecasting [\[repo\]](https://github.com/amazon-science/chronos-forecasting)
- Large-Time-Series-Model [\[repo\]](https://github.com/thuml/Large-Time-Series-Model)
- gift-eval [\[repo\]](https://github.com/SalesforceAIResearch/gift-eval)
- FilterNet [\[repo\]](https://github.com/aikunyi/FilterNet)
- TSLANet [\[repo\]](https://github.com/emadeldeen24/TSLANet)

## License

This project is licensed under the Apache-2.0 License.

