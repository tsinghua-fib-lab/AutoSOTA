# 🕵️ BARO: Robust Root Cause Analysis for Microservice Systems

[![DOI](https://zenodo.org/badge/787200147.svg)](https://zenodo.org/doi/10.5281/zenodo.11063695)
[![pypi package](https://img.shields.io/pypi/v/fse-baro.svg)](https://pypi.org/project/fse-baro)
[![Downloads](https://static.pepy.tech/badge/fse-baro)](https://pepy.tech/project/fse-baro)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/phamquiluan/baro/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/phamquiluan/baro/tree/main)
[![Build and test](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml/badge.svg?branch=main)](https://github.com/phamquiluan/baro/actions/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/phamquiluan/baro/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phamquiluan/baro/actions/workflows/python-publish.yml)

**BARO** is an end-to-end anomaly detection and root cause analysis approach for microservices failures. This repository includes artifacts for reuse and reproduction of experimental results presented in our FSE'24 paper titled _"BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection"_.

**Table of Contents**
  * [Installation](#installation)
  * [How-to-use](#how-to-use)
    + [Data format](#data-format)
    + [Basic usage example](#basic-usage-example)
  * [Reproducibility](#reproducibility)
    + [Reproduce RQ1 - Anomaly Detection Effectiveness](#reproduce-rq1---anomaly-detection-effectiveness)
    + [Reproduce RQ2 - Root Cause Analysis Effectiveness](#reproduce-rq2---root-cause-analysis-effectiveness)
    + [Reproduce RQ3 - Components of BARO](#reproduce-rq3---components-of-baro)
    + [Reproduce RQ4 - Sensitivity Analysis](#reproduce-rq4---sensitivity-analysis)
  * [Download Paper](#download-paper)
  * [Download Datasets](#download-datasets)
  * [Running Time and Instrumentation Cost](#running-time-and-instrumentation-cost)
  * [Supplementary materials](#supplementary-materials)
  * [Citation](#citation)
  * [Contact](#contact)

## Installation

Clone BARO from GitHub

```bash
git clone https://github.com/phamquiluan/baro.git && cd baro
```

Install BARO from PyPI

```bash
# install BARO from PyPI
pip install fse-baro
```

OR, build BARO from source

```
# build BARO from source
pip install -e .
```

BARO has been tested on Linux and Windows, with different Python versions. More details are in [INSTALL.md](./INSTALL.md).

## How-to-use

### Data format

The data must be a `pandas.DataFrame` that consists of multivariate time series metrics data. We require the data to have a column named `time` that stores the timestep. Each other column stores a time series for metrics data with the name format of `<service>_<metric>`. For example, the column `cart_cpu` stores the CPU utilization of service `cart`. A sample of valid data could be downloaded using the `download_data()` method that we will demonstrate shortly below.


### Basic usage example

BARO consists of two modules, namely MultivariateBOCPD (implemented in `baro.anomaly_detection.bocpd`) and RobustScorer (implemented in `baro.root_cause_analysis.robust_scorer`). We expose these two modules for users/researchers to reuse them more conveniently. The basic sample commands to run BARO are presented as follows,

```python
# You can put the code here to a file named test.py
from baro.anomaly_detection import bocpd
from baro.root_cause_analysis import robust_scorer
from baro.utility import download_data, read_data

# download a sample data to data.csv
download_data()

# read data from data.csv
data = read_data("data.csv")

# perform anomaly detection 
anomalies = bocpd(data) 
print("Anomalies are detected at timestep:", anomalies[0])

# perform root cause analysis
root_causes = robust_scorer(data, anomalies=anomalies)["ranks"]

# print the top 5 root causes
print("Top 5 root causes:", root_causes[:5])
```

<details>
<summary>Expected output after running the above code (it takes around 1 minute)</summary>

```
python test.py
Downloading data.csv..: 100%|████████████████████████████████| 570k/570k [00:00<00:00, 17.1MiB/s]
Anomalies are detected at timestep: 243
Top 5 root causes: ['checkoutservice_latency', 'cartservice_mem', 'cartservice_latency', 'cartservice_cpu', 'main_mem']
```
</details>

👉 For more detailed tutorials, you can also check this [tutorials/how-to-use-baro.ipynb](tutorials/how-to-use-baro.ipynb).


## Reproducibility

We have provided a file named `main.py` to assist in reproducing the results of our paper, which can be run using Python with the following syntax:

```bash
python main.py [-h] [--anomaly-detection] [--saved] [--dataset DATASET] [--fault-type FAULT_TYPE] [--rq4] [--eval-metric EVAL_METRIC]
```

The description for the arguments/options of the file `main.py` are as follows:
```bash
options:
  -h, --help            show this help message and exit
  --anomaly-detection   Reproduce anomaly detection results.
  --saved               Use saved anomaly detection results to reproduce the
                        presented results without rerunning anomaly detection.
  --dataset DATASET     Choose a dataset to analyze. Options:
                        ['OnlineBoutique', 'SockShop', and 'TrainTicket'].
  --fault-type FAULT_TYPE
                        Specify the fault type for root cause analysis.
                        Options: ['cpu', 'mem', 'delay', 'loss', and 'all'].
                        If 'all' is selected, the program will run the root
                        cause analysis for all fault types.
  --rq4                 Reproduce RQ4 results.
  --eval-metric EVAL_METRIC
                        Evaluation metric for RQ4. Options: ['top1', 'top3',
                        'avg5']. Default: 'avg5'.
```


### Reproduce RQ1 - Anomaly Detection Effectiveness

To reproduce the anomaly detection performance of BARO, as presented in Table 2. You can run the following commands (the corresponding dataset will be automatically downloaded and extracted to folder `./data`):

```bash
python main.py --dataset OnlineBoutique --anomaly-detection
```

<details>
<summary>Expected output after running the above code (it takes around two hours)</summary>

<br />

The results are a bit better than the numbers presented in the paper (Table 2).
```
Downloading fse-ob.zip..: 100% 151M/151M [04:12<00:00, 597kiB/s]
Running:   2% 2/100 [02:51<2:20:31, 86.03s/it]
====== Reproduce BOCPD =====
Dataset: fse-ob
Precision: 0.76
Recall   : 1.00
F1       : 0.87
```
</details>

👉 You can also checked the [tutorials/reproduce_multivariate_bocpd.ipynb](tutorials/reproduce_multivariate_bocpd.ipynb) to reproduce the saved anomalies in our datasets.

### Reproduce RQ2 - Root Cause Analysis Effectiveness

As presented in Table 3, BARO achieves Avg@5 of 0.91, 0.96, 0.95, 0.62, and 0.86 for CPU, MEM, DELAY, LOSS, and ALL fault types on the Online Boutique dataset. To reproduce the RCA performance of our BARO as presented in the Table 3. You can run the following commands:

```bash
# For Linux users
python main.py --dataset OnlineBoutique --fault-type cpu \
  && python main.py --dataset OnlineBoutique --fault-type mem \
  && python main.py --dataset OnlineBoutique --fault-type delay \
  && python main.py --dataset OnlineBoutique --fault-type loss \
  && python main.py --dataset OnlineBoutique --fault-type all \
```

```bash
# For Windows users
python main.py --dataset OnlineBoutique --fault-type cpu && python main.py --dataset OnlineBoutique --fault-type mem && python main.py --dataset OnlineBoutique --fault-type delay && python main.py --dataset OnlineBoutique --fault-type loss && python main.py --dataset OnlineBoutique --fault-type all 
```

<details>
<summary>Expected output after running the above code (it takes few seconds)</summary>

```
Running: 100%|███████████████████████████████████████████████████| 25/25 [00:02<00:00, 11.94it/s]
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: cpu
Avg@5 Acc : 0.91

Running: 100%|███████████████████████████████████████████████████| 25/25 [00:02<00:00, 12.10it/s]
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: mem
Avg@5 Acc : 0.96

Running: 100%|███████████████████████████████████████████████████| 25/25 [00:01<00:00, 12.73it/s]
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: delay
Avg@5 Acc : 0.95

Running: 100%|███████████████████████████████████████████████████| 25/25 [00:02<00:00, 12.35it/s]
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: loss
Avg@5 Acc : 0.62

Running: 100%|█████████████████████████████████████████████████| 100/100 [00:06<00:00, 15.82it/s]
====== Reproduce BARO =====
Dataset   : fse-ob
Fault type: all
Avg@5 Acc : 0.86
```
</details>


### Reproduce RQ3 - Components of BARO

Our RQ3 relies on the experimental results of RQ2, which we reproduced above. 

### Reproduce RQ4 - Sensitivity Analysis

As presented in Figure 5, BARO maintains stable accuracy on the Online Boutique dataset when we vary `t_bias` from `-40` to `40`. To reproduce these results, for example, you can run the following command to obtain the Avg@5 scores on the Online Boutique dataset:

```bash
python main.py --dataset OnlineBoutique --rq4 --eval-metric avg5 
```

<details>
<summary>Expected output after running the above code (it takes few minutes)</summary>

<br />

The output list presents the `Avg@5` scores when we vary `t_bias`. You can see that BARO can maintain a stable performance.


```
Running: 100%|███████████████████████████████████████████████████| 40/40 [04:00<00:00,  6.02s/it]
[0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.86, 0.86, 0.87, 0.87, 0.86, 0.87, 0.87, 0.86, 0.86, 0.86, 0.86, 0.85, 0.85, 0.85, 0.85, 0.84, 0.85, 0.85, 0.85, 0.85]
```
</details>




## Download Paper

Our paper could be downloaded at [docs/paper.pdf](docs/paper.pdf) or at [doi/10.1145/3660805](https://dl.acm.org/doi/10.1145/3660805).

## Download Datasets

Our datasets and their description are publicly available in Zenodo repository with the following information:

- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11046533.svg)](https://doi.org/10.5281/zenodo.11046533)
- Dataset URL: https://zenodo.org/records/11046533

We also provide utility functions to download our datasets using Python. The downloaded datasets will be available at directory `data`.

```python
from baro.utility import (
    download_online_boutique_dataset,
    download_sock_shop_dataset,
    download_train_ticket_dataset,
)
download_online_boutique_dataset()
download_sock_shop_dataset()
download_train_ticket_dataset()
```
<details>
<summary>Expected output after running the above code (it takes few minutes to download our datasets)</summary>

```
$ python test.py
Downloading fse-ob.zip..: 100%|██████████| 151M/151M [01:03<00:00, 2.38MiB/s]
Downloading fse-ss.zip..: 100%|██████████| 127M/127M [00:23<00:00, 5.49MiB/s]
Downloading fse-tt.zip..: 100%|██████████| 286M/286M [00:56<00:00, 5.10MiB/s]
```
</details>


## Running Time and Instrumentation Cost

Please refer to our [docs/running_time_and_instrumentation_cost.md](docs/running_time_and_instrumentation_cost.md) document.


## Supplementary materials

You can download our supplementary materials from [docs/fse_baro_supplementary_material.pdf](docs/fse_baro_supplementary_material.pdf)

## Citation

```
@article{pham2024baro,
author = {Pham, Luan and Ha, Huong and Zhang, Hongyu},
title = {BARO: Robust Root Cause Analysis for Microservices via Multivariate Bayesian Online Change Point Detection},
year = {2024},
issue_date = {July 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {1},
number = {FSE},
url = {https://doi.org/10.1145/3660805},
doi = {10.1145/3660805},
journal = {Proc. ACM Softw. Eng.},
month = {jul},
articleno = {98},
numpages = {24},
keywords = {Anomaly Detection, Microservice Systems, Root Cause Analysis}
}
```

## Contact

[phamquiluan\@gmail.com](mailto:phamquiluan@gmail.com?subject=BARO)
