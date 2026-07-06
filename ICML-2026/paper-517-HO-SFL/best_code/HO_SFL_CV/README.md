## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (for GPU training)
- `uv` package manager (recommended) or `pip`

### Using `uv` (Recommended)

This project uses `uv` for dependency management with support for multiple CUDA versions.

```bash
uv sync --extra cu121
```


These commands will:
- Create/update a virtual environment
- Install all project dependencies including PyTorch with the specified CUDA support
- Lock dependencies in `uv.lock` for reproducibility


## Supported Models and Datasets

### Computer Vision Models


| Model Name | Description |
|-----------|-------------|
| `ResNet18` | ResNet-18 (default, pretrained available) |

### Datasets

| Dataset | Type | Classes | Description |
|---------|------|---------|-------------|
| `CIFAR10` | Image Classification | 10 | 32x32 colored images |
| `CIFAR100` | Image Classification | 100 | 32x32 colored images |


## Main Entry Point

The project uses **Hydra** for configuration management. All parameters are defined in YAML config files under `conf/` and can be overridden via command line.

### Entry Script: `main_v2.py`

**Basic Usage:**
```bash
python main_v2.py --config-name <config_file> [overrides...]
```

### Available Algorithms (Config Files)

1. **`base.yaml`** - HO-SFL (Hybrid-Order Split Federated Learning)
2. **`sfl.yaml`** - SFL (Standard Split Federated Learning with gradients)
3. **`sfl_zo.yaml`** - SFL-ZO (Split Federated Learning with zeroth-order optimization)
4. **`mu_splitfed.yaml`** - MU-SplitFed
5. **`fsl_sage.yaml`** - FSL-SAGE


## Example Usage

### 1. HO-SFL (Hybrid-Order)

```bash
python main_v2.py \
  --config-name base \
  seed=42 \
  system.device=cuda:0 \
  data.dataset=CIFAR10 \
  data.partition.algo=iid \
  model.model_name=ResNet18 \
  algo.split_point=6 \
  algo.lr=0.001 \
  algo.zo_p=10 \
  algo.zo_mu=1e-4 \
  runner.num_clients=100 \
  runner.sampled_clients=10 \
  runner.communication_rounds=1000 \
  runner.total_samples=160000 \
  logging.use_wandb=True \
  logging.project_name=ICML_Experiment \
  logging.log_dir=./logs/ho_sfl
```

**Key HO-SFL Parameters:**
- `algo.zo_p`: Number of perturbations for zeroth-order gradient estimation
- `algo.zo_mu`: Perturbation magnitude

### 2. Standard SFL (Gradient-based)

```bash
python main_v2.py \
  --config-name sfl \
  seed=42 \
  system.device=cuda:0 \
  data.dataset=CIFAR10 \
  algo.split_point=6 \
  algo.lr=0.001 \
  runner.communication_rounds=50 \
  runner.local_epochs=1 \
  logging.use_wandb=True
```

### 3. SFL-ZO (Zeroth-Order)

```bash
python main_v2.py \
  --config-name sfl_zo \
  seed=42 \
  data.dataset=CIFAR100 \
  algo.zo_p=1 \
  algo.zo_mu=1e-3 \
  runner.communication_rounds=50
```

### 4. Non-IID Data Partitioning

To use Dirichlet distribution for non-IID data split:

```bash
python main_v2.py \
  --config-name base \
  data.partition.algo=dirichlet \
  data.partition.alpha=0.5
```

**Partition Parameters:**
- `data.partition.algo`: `iid` or `dirichlet`
- `data.partition.alpha`: Dirichlet concentration parameter (lower = more non-IID)


## Common Parameters

### System
- `seed`: Random seed for reproducibility
- `system.device`: CUDA device (e.g., `cuda:0`)

### Data
- `data.dataset`: Dataset name (`CIFAR10`, `CIFAR100`, `FashionMNIST`)
- `data.train_batch_size`: Training batch size
- `data.test_batch_size`: Test batch size
- `data.partition.algo`: Data partitioning strategy

### Model
- `model.model_name`: Model architecture
- `model.use_pretrained`: Use pretrained weights (boolean)
- `model.freeze_bn`: Freeze batch normalization layers

### Algorithm
- `algo.split_point`: Layer index where to split the model (client/server)
- `algo.optimizer`: Optimizer type (`adamw`, `sgd`)
- `algo.lr`: Learning rate

### Runner
- `runner.communication_rounds`: Total number of federated rounds
- `runner.num_clients`: Total number of clients
- `runner.sampled_clients`: Clients sampled per round
- `runner.total_samples`: Maximum total samples to process (training stops when reached)
- `runner.evaluation_interval`: Evaluate every N rounds

### Logging
- `logging.use_wandb`: Enable Weights & Biases logging
- `logging.project_name`: W&B project name
- `logging.log_dir`: Local log directory


## Batch Experiments

Use the provided shell scripts to run multiple experiments:

```bash
# CIFAR-10 IID experiments
bash cifar10iid.sh

# CIFAR-10 Non-IID experiments
bash cifar10niid.sh

# CIFAR-100 experiments
bash cifar100iid.sh
bash cifar100niid.sh
```

These scripts will run multiple algorithms with different seeds for comprehensive comparison.
