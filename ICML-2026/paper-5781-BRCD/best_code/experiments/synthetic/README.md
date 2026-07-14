# BRCD: Bayesian Root Cause Discovery

This repository contains the implementation of BRCD (Bayesian Root Cause Discovery), a novel approach for identifying root causes in causal systems using Bayesian inference.

## Environment Setup

To replicate the experiments in this repository, you need to set up a conda environment.

### Prerequisites

- Python 3.8+
- Conda package manager
- CUDA (optional, for GPU acceleration)

### Creating the Environment

1. Create a new conda environment:
```bash
conda create -n brcd python=3.8
conda activate brcd
```

2. Install core dependencies:
```bash
conda install numpy scipy pandas scikit-learn matplotlib seaborn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install additional required packages:
```bash
pip install causal-learn==0.1.2.3
pip install pyagrum==0.21.0
pip install networkx==2.6.3
pip install tqdm==4.67.1
pip install igraph==0.11.8
pip install numba==0.61.0
pip install graphviz==0.19.1
pip install cairosvg==2.5.2
pip install pydot==1.4.2
pip install pydotplus==2.0.2
pip install scikit-network==0.24.0
pip install dowhy
pip install pyAgrum
pip install xges
pip install graphical_models
pip install cliquepicking
```

4. Install additional dependencies for specific models:
```bash
# For AERCA
pip install requests==2.32.3

# For RCD
pip install python-igraph==0.9.9

# For CausalRCA
pip install requests==2.26.0
```

5. Verify installation by running:
```bash
python -c "import torch; import causal_learn; import pyagrum; print('Environment setup successful!')"
```

## Experiment Replication

This section provides instructions for replicating the various experiments described in the paper.

### Synthetic Experiments

#### 1. Synthetic Experiment: Small to Large Graphs

This experiment evaluates performance across different graph sizes.

**Data Generation:**
```bash
cd /path/to/brcd
python data_generation.py --n 10 20 50 75 100 1000 --obs_sample_sizes 10000 --int_sample_sizes 100 200 500 750 1000 10000 --output_dir <YOUR_DIR_TO_STORE_DATA>
```

**Run Experiment:**
```bash
python experiment_synthetic.py --data_dir <YOUR_DIR_WHERE_DATA_WAS_SAVED> --output_dir <DIR_WHERE_YOU_TO_SAVE_THE_OUTPUT> --xaxis n
```

#### 2. Synthetic Experiment: Interventional Sample Convergence

This experiment shows how performance improves with increasing interventional sample sizes.

**Data Generation:**
```bash
python data_generation.py --n 50 --obs_sample_sizes 10000 --int_sample_sizes 5 10 100 500 1000 --output_dir <YOUR_DIR_TO_STORE_DATA>
```

**Run Experiment:**
```bash
python experiment_synthetic.py --data_dir <YOUR_DIR_WHERE_DATA_WAS_SAVED> --output_dir <DIR_WHERE_YOU_TO_SAVE_THE_OUTPUT> --xaxis intSampleSize
```

#### 3. Synthetic Experiment: Multiple Root Causes

This experiment evaluates performance when multiple root causes are present.

**Data Generation:**
```bash
python data_generation.py --n 50 --obs_sample_sizes 10000 --int_sample_sizes 5000 --output_dir <YOUR_DIR_TO_STORE_DATA> --multiple_rootcauses 2
```

**Run Experiment:**
```bash
python experiment_multiple_rootcause.py --data_dir <YOUR_DIR_WHERE_DATA_WAS_SAVED> --output_dir <DIR_WHERE_YOU_TO_SAVE_THE_OUTPUT> --xaxis rc
```

## Project Structure

```
brcd/
├── data_generation.py              # Synthetic data generation
├── experiment_synthetic.py         # Synthetic experiments
├── experiment_multiple_rootcause.py # Multiple root cause experiments
├── experiment_ob.py               # Online Boutique experiments
├── experiment_sockshop.py          # SockShop experiments
├── models/
│   ├── brcd_k.py                  # Main BRCD implementation for multiple root causes
│   ├── boss.py                    # BOSS algorithm for causal discovery
│   ├── rcd/                       # RCD baseline implementation
│   ├── CausalRCA_code/           # CausalRCA baseline implementation
│   └── ...                        # Other baseline methods
├── utils.py                       # Utility functions
└── README.md                      # This file
```

