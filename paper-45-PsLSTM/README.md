# Unlocking the Power of LSTM for Long Term Time Series Forecasting (P-sLSTM)

This repository contains the official implementation for the paper: **"Unlocking the Power of LSTM for Long Term Time Series Forecasting" (AAAI-25)**. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33303)


## 📜 Abstract

Traditional recurrent neural network architectures, such as long short-term memory neural networks (LSTM), have historically held a prominent role in time series forecasting (TSF) tasks. While the recently introduced sLSTM for Natural Language Processing (NLP) introduces exponential gating and memory mixing that are beneficial for long term sequential learning, its potential short memory issue is a barrier to applying sLSTM directly in TSF. To address this, we propose a simple yet efficient algorithm named **P-sLSTM**, which is built upon sLSTM by incorporating **patching** and **channel independence**. These modifications substantially enhance sLSTM's performance in TSF, achieving state-of-the-art results. Furthermore, we provide theoretical justifications for our design, and conduct extensive comparative and analytical experiments to fully validate the efficiency and superior performance of our model.


## 🏗️ Model Architecture

P-sLSTM enhances the sLSTM architecture for long-term time series forecasting by integrating two key techniques:

* **Patching:** The input time series is segmented into patches. This allows the sLSTM backbone to capture local temporal information within each patch, overcoming the potential short memory limitations of standard RNNs when processing very long sequences.
* **Channel Independence (CI):** Each channel (variate) of the multivariate time series is processed independently with a shared sLSTM backbone. This strategy has been shown to prevent overfitting and improve generalization in time series models.

<img width="1418" height="456" alt="PsLSTM_Overview_CMYK" src="https://github.com/user-attachments/assets/ab270c14-749f-4933-855b-a584343f1179" />


## ⚙️ Environment

The model implementation requires a specific Python and PyTorch environment.

* **Python:** Version `3.9` or newer is required.
* **PyTorch/CUDA:** Ensure your PyTorch version is compatible with your installed CUDA version.

The following setups are confirmed to work:
* PyTorch 2.1.0, Python 3.10, and CUDA 12.1
* PyTorch 2.3.0, Python 3.12, and CUDA 12.1

Additionally, please ensure `ninja` is installed:
```bash
pip install ninja
```

## 🚀 Setup and Data Preparation

### 1. Clone Repository
```bash
git clone https://github.com/Eleanorkong/P-sLSTM.git
cd P-sLSTM
````

### 2\. Dependency Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3\. Data Acquisition

The experiments are conducted on publicly available datasets benchmarked in Time-Series-Library. Datasets can be downloaded following the instructions at:

  * [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
  * [https://github.com/thuml/iTransformer](https://github.com/thuml/iTransformer)

### 4\. Data Configuration

Downloaded datasets must be placed in the `./dataset` directory. The expected directory structure is as follows:

```
P-sLSTM/
├── dataset/
│   ├── weather.csv
│   ├── electricity.csv
│   └── ...
├── scripts/
└── ...
```

## 📊 Experiment Replication

Example scripts are provided for replicating the experimental results presented in the paper. For detailed guidance on script usage and structure, please refer to the conventions established in the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) repository.

**Example Usage:**

To execute the P-sLSTM model for long-term forecasting on the Weather dataset, run the provided script:

```sh
sh scripts/EXP-LongForecasting/P_sLSTM/weather.sh
```

This script can be adapted for other datasets and forecasting horizons.


## 🙏 Acknowledgements

The implementation of P-sLSTM builds upon and utilizes components from the following outstanding repositories. We gratefully acknowledge the authors of these repositories for making their work publicly available.

  * **Time-Series-Library:** [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
  * **iTransformer:** [https://github.com/thuml/iTransformer](https://github.com/thuml/iTransformer)
  * **PatchTST:** [https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) 
  * **xLSTM (sLSTM):** [https://github.com/NX-AI/xlstm](https://github.com/NX-AI/xlstm)


## Citation

If you utilize this code or find our work beneficial to your research, please consider citing the original paper:

```bibtex
@inproceedings{kong2025unlocking,
  title={Unlocking the power of lstm for long term time series forecasting},
  author={Kong, Yaxuan and Wang, Zepu and Nie, Yuqi and Zhou, Tian and Zohren, Stefan and Liang, Yuxuan and Sun, Peng and Wen, Qingsong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={11},
  pages={11968--11976},
  year={2025}
}
```
