# $\oslash$ Source Models Leak What They Shouldn’t $\nrightarrow$: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization
## [📄 CVPR 2026](https://arxiv.org/abs/2604.08238)

## Setup

### Dataset
Download the **OfficeHome** dataset from [this link](https://www.hemanthdv.org/officeHomeDataset.html) and place it in the `data/OfficeHome` directory.

Alternatively, if the dataset is stored elsewhere, update the `data_path` in the configuration dictionary in `main.py`.

### Environment
This repository uses **Python 3.12.2**

```bash
conda create -n scada python=3.12.2 -y
conda activate scada
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch 2.4.1 and torchvision 0.19.1 from the [official website](https://pytorch.org)

---

## Running SCADA Unlearning

### Single Task
To run the proposed method for forget classes `{1,2,3}` on a single OfficeHome Task (e.g. **(Art → Product)**):

```bash
python main.py -d OfficeHome -s Art -t Product -m minimax -fc 1,2,3
```

### All Tasks
To run the proposed method for classes `{1,2,3}` across all OfficeHome domain pairs:

```bash
bash minimax.sh
```
