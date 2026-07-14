# 🕵️ Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?

[![DOI](https://zenodo.org/badge/840137303.svg)](https://zenodo.org/doi/10.5281/zenodo.13294048)
[![pypi package](https://img.shields.io/pypi/v/RCAEval.svg)](https://pypi.org/project/RCAEval)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/phamquiluan/RCAEval/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/phamquiluan/RCAEval/tree/main)
[![Build and test](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml)

This repository includes artifacts for reuse and reproduction of experimental results presented in our ASE'24 paper titled "Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?".

**Table of Contents** 
  * [Installation](#installation)
  * [How-to-use](#how-to-use)
    + [Data format](#data-format)
    + [Basic usage example](#basic-usage-example)
  * [Reproducibility](#reproducibility)
    + [Reproduce RQ1 - Causal Discovery Performance](#reproduce-rq1---causal-discovery-performance)
    + [Reproduce RQ2 - Root Cause Analysis Performance](#reproduce-rq2---root-cause-analysis-performance)
    + [Reproduce RQ3 - Efficiency](#reproduce-rq3---efficiency)
    + [Reproduce RQ4 - Data lengths](#reproduce-rq4---data-lengths)
  * [Download Datasets](#download-datasets)
  * [Download Paper and Supplementary Material](#download-paper-and-supplementary-material)
  * [Licensing](#licensing)
  * [Acknowledgments](#acknowledgments)

## Prerequisites

We recommend using machines equipped with at least 8 cores, 16GB RAM, and ~30GB available disk space with Ubuntu 22.04 or Ubuntu 20.04, and Python3.10 installed to stably reproduce the experimental results in our paper.

## Installation

The `default` environment, which is used for most methods, can be easily installed as follows. We require Ubuntu and Python 3.10 to reproduce our results stably. Detailed installation instructions for all methods are in [SETUP.md](SETUP.md).


Open your terminal and run the following commands

```bash
sudo apt update -y
sudo apt install -y build-essential \
  libxml2 libxml2-dev zlib1g-dev \
  python3-tk graphviz
```

Clone RCAEval from GitHub

```bash
git clone https://github.com/phamquiluan/RCAEval.git && cd RCAEval
```

Create virtual environment with Python 3.10 (refer [SETUP.md](SETUP.md) to see how to install Python3.10 on Linux)

```bash
python3.10 -m venv env
. env/bin/activate
```

Install RCAEval using pip

```bash
pip install pip==20.0.2
pip install -e .[default]
```

Or, install RCAEval from PyPI

```bash
# Install RCAEval from PyPI
pip install pip==20.0.2
pip install RCAEval[default]
```

Test the installation

```bash
python -m pytest tests/test.py::test_basic
```

Expected output after running the above command (it takes less than 1 minute)

```bash 
$ pytest tests/test.py::test_basic
============================== test session starts ===============================
platform linux -- Python 3.10.12, pytest-7.3.1, pluggy-1.0.0
rootdir: /home/ubuntu/RCAEval
plugins: dvc-2.57.3, hydra-core-1.3.2
collected 1 item                                                                 

tests/test.py .                                                            [100%]

=============================== 1 passed in 3.16s ================================
```


## How-to-use

### Data format

The data must be a `pandas.DataFrame` that consists of multivariate time series metrics data. We require the data to have a column named `time` that stores the timestep. Each other column stores a time series for metrics data with the name format of `<service>_<metric>`. For example, the column `cart_cpu` stores the CPU utilization of service `cart`. A sample of valid data could be downloaded using the `download_data()` method that we will demonstrate shortly below.



### Basic usage example

RCAEval stores all the RCA methods in the `e2e` module (implemented in `RCAEval.e2e`). Available methods are: pc_pagerank, pc_randomwalk, fci_pagerank, fci_randomwalk, granger_pagerank, granger_randomwalk, lingam_pagerank, lingam_randomwalk, fges_pagerank, fges_randomwalk, ntlr_pagerank, ntlr_randomwalk, causalrca, causalai, run, microcause, e_diagnosis, baro, rcd, nsigma, and circa.

A basic example to use BARO, an RCA baseline, to perform RCA are presented as follows,

```python
# You can put the code here to a file named test.py
from RCAEval.e2e import baro
from RCAEval.utility import download_data, read_data

# download a sample data to data.csv
download_data()

# read data from data.csv
data = read_data("data.csv")
anomaly_detected_timestamp = 1692569339

# perform root cause analysis
root_causes = baro(data, anomaly_detected_timestamp)["ranks"]

# print the top 5 root causes
print("Top 5 root causes:", root_causes[:5])
```

Expected output after running the above code (it takes around 1 minute)

```
$ python test.py
Downloading data.csv..: 100%|████████████████████| 570k/570k [00:00<00:00, 19.8MiB/s]
Top 5 root causes: ['emailservice_mem', 'recommendationservice_mem', 'cartservice_mem', 'checkoutservice_latency', 'cartservice_latency']
```


## Reproducibility

### Reproduce RQ1 - Causal Discovery Performance

We provide a script named `rq1.py` to assist in reproducing the RQ1 results from our paper. This script can be executed using Python with the following syntax: 

```
python rq1.py [-h] [--dataset DATASET] [--method METHOD] [--length LENGTH] [--test]
```

The available options and their descriptions are as follows:

```
options:
  -h, --help            Show this help message and exit
  --dataset DATASET     Choose a dataset. Valid options:
                        [circa10, circa50, rcd10, rcd50, causil10, causil50]
  --method METHOD       Choose a method (e.g. `pc`, `fci`, `granger`, `ICALiNGAM`, `DirectLiNGAM`, `ges`, `fges`, `pcmci`, `ntlr`.)
  --length LENGTH       Specify the length of the time series (used for RQ4)
  --test                Perform smoke test on certain methods without fully run
```

For example, in Table 3, PC achieves F1, F1-S, and SHD scores of 0.49, 0.65, and 16 on the CIRCA 10 dataset. To reproduce these results, you can run the following commands:

```bash
python rq1.py --dataset circa10 --method pc
```

The expected output should be exactly as presented in the paper (it takes less than 1 minute to run the code)

```
F1:   0.49
F1-S: 0.65
SHD:  16
Avg speed: 0.08
```

We can replace the pc method with other methods (e.g., fci, granger) and substitute circa10 with other datasets to replicate the corresponding results shown in Table 3. This reproduction process is also integrated into our Continuous Integration (CI) setup. For more details, refer to the [.github/workflows/reproduce-rq1.yml](.github/workflows/reproduce-rq1.yml) file.


### Reproduce RQ2 - Root Cause Analysis Performance

We provide a script named `rq2.py` to assist in reproducing the RQ2 results from our paper. This script can be executed using Python with the following syntax: 

```
python rq2.py [-h] [--dataset DATASET] [--method METHOD] [--tdelta TDELTA] [--length LENGTH] [--test] 
```

The available options and their descriptions are as follows:

```
options:
  -h, --help            Show this help message and exit
  --dataset DATASET     Choose a dataset. Valid options:
                        [online-boutique, sock-shop-1, sock-shop-2, train-ticket,
                         circa10, circa50, rcd10, rcd50, causil10, causil50]
  --method METHOD       Choose a method (`pc_pagerank`, `pc_randomwalk`, `fci_pagerank`, `fci_randomwalk`, `granger_pagerank`, `granger_randomwalk`, `lingam_pagerank`, `lingam_randomwalk`, `fges_pagerank`, `fges_randomwalk`, `ntlr_pagerank`, `ntlr_randomwalk`, `causalrca`, `causalai`, `run`, `microcause`, `e_diagnosis`, `baro`, `rcd`, `nsigma`, and `circa`)
  --tdelta TDELTA       Specify $t_delta$ to simulate delay in anomaly detection (e.g.`--tdelta 60`)
  --length LENGTH       Specify the length of the time series (used for RQ4)
  --test                Perform smoke test on certain methods without fully run
```

For example, in Table 5, BARO [ $t_\Delta = 0$ ] achieves Avg@5 of 0.97, 1, 0.91, 0.98, and 0.67 for CPU, MEM, DISK, DELAY, and LOSS fault types on the Online Boutique dataset. To reproduce these results, you can run the following commands:

```bash
python rq2.py --dataset online-boutique --method baro 
```

The expected output should be exactly as presented in the paper (it takes less than 1 minute to run the code)

```
--- Evaluation results ---
Avg@5-CPU:   0.97
Avg@5-MEM:   1.0
Avg@5-DISK:  0.91
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.67
---
Avg speed: 0.07
```

As presented in Table 5, BARO [ $t_\Delta = 60$ ] achieves Avg@5 of 0.94, 0.99, 0.87, 0.99, and 0.6 for CPU, MEM, DISK, DELAY, and LOSS fault types on the Online Boutique dataset. To reproduce these results, you can run the following commands:

```bash
python rq2.py --dataset online-boutique --method baro --tdelta 60
```

The expected output should be exactly as presented in the paper (it takes less than 1 minute to run the code)

```
--- Evaluation results ---
Avg@5-CPU:   0.94
Avg@5-MEM:   0.99
Avg@5-DISK:  0.87
Avg@5-DELAY: 0.99
Avg@5-LOSS:  0.6
---
Avg speed: 0.07
```

We can replace the baro method with other methods (e.g., nsigma, fci_randomwalk) and substitute online-boutique with other datasets to replicate the corresponding results shown in Table 5. This reproduction process is also integrated into our Continuous Integration (CI) setup. For more details, refer to the [.github/workflows/reproduce-rq2.yml](.github/workflows/reproduce-rq2.yml) file.


### Reproduce RQ3 - Efficiency

The running time of each method is recorded in the scripts of RQ1 and RQ2 as we described and presented above.

### Reproduce RQ4 - Data lengths

Our RQ4 relies on the scripts of RQ1 and RQ2 as we described and presented above, with the option `--length`. 

As presented in Figure 3, BARO maintains stable accuracy on the Online Boutique dataset when we vary the data length from 60 to 600. To reproduce these results, for example, you can run the following Bash script:

```bash
# You can put the code here to a file named tmp.sh, and run the script by `bash tmp.sh`
for length in 60 120 180 240 300 360 420 480 540 600; do
    python rq2.py --dataset online-boutique --method baro --length $length
done
```


<details>
<summary>Expected output after running the above code (it takes few minutes)</summary>

<br />

The output list presents the `Avg@5` scores when we vary the data length. You can see that BARO can maintain a stable performance.


```
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.64it/s]
--- Evaluation results ---
Avg@5-CPU:   0.84
Avg@5-MEM:   0.98
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 120 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.37it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 180 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.46it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 240 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.47it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 300 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.40it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 360 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.42it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 420 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.53it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 480 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.37it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 540 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.50it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
=== VARY LENGTH: 600 ===
100%|█████████████████████████████████████████████████████████████████████| 125/125 [00:08<00:00, 14.49it/s]
--- Evaluation results ---
Avg@5-CPU:   0.82
Avg@5-MEM:   0.99
Avg@5-DISK:  0.94
Avg@5-DELAY: 0.98
Avg@5-LOSS:  0.66
---
Avg speed: 0.07
```
</details>


## Download Datasets

Our datasets and their description are publicly available in Zenodo repository with the following information:
- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13305663.svg)](https://doi.org/10.5281/zenodo.13305663)
- Dataset URL: https://zenodo.org/records/13305663

We also provide utility functions to download our datasets using Python. The downloaded datasets will be available at directory `data`.

```python
from RCAEval.utility import (
    download_syn_rcd_dataset,
    download_syn_circa_dataset,
    download_syn_causil_dataset,
    download_rca_circa_dataset,
    download_rca_rcd_dataset,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
)

download_syn_rcd_dataset()
download_syn_circa_dataset()
download_syn_causil_dataset()
download_rca_circa_dataset()
download_rca_rcd_dataset()
download_online_boutique_dataset()
download_sock_shop_1_dataset()
download_sock_shop_2_dataset()
download_train_ticket_dataset()
```
<details>
<summary>Expected output after running the above code (it takes few minutes to download our datasets)</summary>

```
$ python test.py
Downloading syn_rcd.zip..: 100%|██████████| 11.2M/11.2M [00:03<00:00, 3.56MiB/s]
Downloading syn_circa.zip..: 100%|████████| 52.5M/52.5M [00:06<00:00, 7.51MiB/s]
Downloading syn_causil.zip..: 100%|███████| 73.8M/73.8M [00:09<00:00, 8.05MiB/s]
Downloading rca_circa.zip..: 100%|████████| 52.7M/52.7M [00:06<00:00, 7.95MiB/s]
Downloading rca_rcd.zip..: 100%|██████████| 22.7M/22.7M [00:04<00:00, 5.16MiB/s]
Downloading online-boutique.zip..: 100%|██| 31.0M/31.0M [00:04<00:00, 6.49MiB/s]
Downloading sock-shop-1.zip..: 100%|██████| 3.54M/3.54M [00:02<00:00, 1.68MiB/s]
Downloading sock-shop-2.zip..: 100%|██████| 79.1M/79.1M [00:09<00:00, 8.38MiB/s]
Downloading train-ticket.zip..: 100%|███████| 280M/280M [00:27<00:00, 10.1MiB/s]
```
</details>


## Download Paper and Supplementary Material

- Our paper can be downloaded at https://dl.acm.org/doi/10.1145/3691620.3695065 or  [docs/ase-paper.pdf](docs/ase-paper.pdf)
- Our supplementary material can be downloaded at [docs/ase-paper-supplementary-material.pdf](docs/ase-paper-supplementary-material.pdf)

## Licensing

This repository includes code from various sources with different licenses. We have included their corresponding LICENSE into the [LICENSES](LICENSES) directory:

- **BARO**: Licensed under the [MIT License](LICENSES/LICENSE-BARO). Original source: [BARO GitHub Repository](https://github.com/phamquiluan/baro/blob/main/LICENSE).
- **CIRCA**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE-CIRCA). Original source: [CIRCA GitHub Repository](https://github.com/NetManAIOps/CIRCA/blob/master/LICENSE).
- **RCD**: Licensed under the [MIT License](LICENSES/LICENSE-RCD). Original source: [RCD GitHub Repository](https://github.com/azamikram/rcd).
- **E-Diagnosis**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE-E-Diagnosis). Original source: [PyRCA GitHub Repository](https://github.com/salesforce/PyRCA/blob/main/LICENSE).
- **CausalAI**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE-CausalAI). Original source: [CausalAI GitHub Repository](https://github.com/salesforce/causalai/blob/main/LICENSE).
- **NSigma**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE-NSigma). Original source: [Scikit-learn GitHub Repository](https://github.com/scikit-learn/scikit-learn?tab=BSD-3-Clause-1-ov-file).
- **MicroCause**: Licensed under the [Apache License 2.0](LICENSES/LICENSE-MicroCause). Original source: [MicroCause GitHub Repository](https://github.com/PanYicheng/dycause_rca/blob/main/LICENSE).
- **PC/FCI/LiNGAM**: Licensed under the [MIT License](LICENSES/LICENSE-CausalLearn). Original source: [Causal-learn GitHub Repository](https://github.com/py-why/causal-learn/blob/main/LICENSE).
- **Granger**: Licensed under the [BSD 3-Clause "New" or "Revised" License](LICENSES/LICENSE-Granger). Original source: [Statsmodels GitHub Repository](https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt).
- **CausalRCA**: No License. Original source: [CausalRCA GitHub Repository](https://github.com/AXinx/CausalRCA_code).
- **RUN**: No License. Original source: [RUN GitHub Repository](https://github.com/zmlin1998/RUN).

**For the code implemented by us and for our datasets, we distribute them under the [MIT LICENSE](LICENSE)**.

## Acknowledgments

We would like to express our sincere gratitude to the researchers and developers who created the baselines used in our study. Their work has been instrumental in making this project possible. We deeply appreciate the time, effort, and expertise that have gone into developing and maintaining these resources. This project would not have been feasible without their contributions.


## Contact

[phamquiluan\@gmail.com](mailto:phamquiluan@gmail.com?subject=RCAEval)
