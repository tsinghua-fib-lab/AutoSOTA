## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (for GPU training)
- `uv` package manager (recommended) or `pip`

### Using `uv` (Recommended)

This project uses `uv` for dependency management with support for multiple CUDA versions.

#### Option 1: CUDA 12.1 (Default)
```bash
uv sync --extra cu121
```

#### Option 2: CUDA 12.8
```bash
uv sync --extra cu128
```

These commands will:
- Create/update a virtual environment
- Install all project dependencies including PyTorch with the specified CUDA support
- Lock dependencies in `uv.lock` for reproducibility


## Supported Models and Datasets

### Large Language Models

The project supports the following LLMs:

| Model Name | HuggingFace Model ID | Recommended Split Point |
|-----------|----------------------|------------------------|
| `llama-3.2-1b` | `meta-llama/Llama-3.2-1b` | 8 |
| `opt-125m` | `facebook/opt-125m` | 3 |
| `gemma-3-270m` | `google/gemma-3-270m` | 5 |

**Note**: Specify model using the `--large-model` flag (e.g., `--large-model=llama-3.2-1b`)

### Datasets

The project supports NLU (Natural Language Understanding) tasks:

| Dataset | Type | Description |
|---------|------|-------------|
| `sst2` | Classification | Sentiment classification (typically 5 rounds) |
| `wsc` | Coreference | Word sense disambiguation (typically 560 rounds) |
| `rte` | Inference | Recognizing textual entailment (typically 125 rounds) |

**Note**: Configure using `--dataset` flag (e.g., `--dataset=sst2`)

## Main Entry Points

### 1. Standard SFL Training (`main_sfl.py`)

Standard Split Federated Learning with gradient-based client updates.

**Usage:**
```bash
python main_sfl.py \
    --seed=42 \
    --cuda \
    --device=cuda:0 \
    --large-model=llama-3.2-1b \
    --lora \
    --split \
    --split-point=8 \
    --optimizer=adamw \
    --lr=1e-5 \
    --num-clients=10 \
    --sampled-client-num=3 \
    --total-rounds=5 \
    --local_epochs=1 \
    --evaluation-interval=1 \
    --dataset=sst2 \
    --max-length=128 \
    --train-batch-size=32 \
    --test-batch-size=200 \
    --iid \
    --log-to-wandb \
    --project-name=ICML_LLM_Baseline3 \
    --log-dir="./logs/baseline3/llama-3.2-1b/sst2" \
    --total-samples=80000
```

**Explanation:**
- `--total-samples=80000`: Maximum total number of samples to process (highest priority - training stops when this limit is reached)
- `--seed=42`: Random seed for reproducibility
- `--large-model=llama-3.2-1b`: Base LLM to use
- `--lora`: Apply LoRA fine-tuning (reduces trainable parameters)
- `--split`: Enable model splitting into client/server
- `--split-point=8`: Split at layer 8 of Llama (client gets first 8 layers, server gets remaining)
- `--num-clients=10`: Total number of clients in the federation
- `--sampled-client-num=3`: Number of clients sampled per round
- `--total-rounds=5`: Total training rounds
- `--iid`: Use IID (independent and identically distributed) data split (vs. Dirichlet)

### 2. SFL with Zeroth-Order Optimization (`main_zo.py`)

Split Federated Learning using zeroth-order gradient estimation (derivative-free).

**Usage:**
```bash
python main_zo_sfl.py \
    --seed=42 \
    --cuda \
    --device=cuda:0 \
    --large-model=llama-3.2-1b \
    --lora \
    --split \
    --split-point=8 \
    --optimizer=adamw \
    --lr=1e-5 \
    --num-clients=10 \
    --sampled-client-num=3 \
    --total-rounds=5 \
    --local_epochs=1 \
    --evaluation-interval=1 \
    --dataset=sst2 \
    --max-length=128 \
    --train-batch-size=32 \
    --test-batch-size=200 \
    --iid \
    --num-pert=1 \
    --mu=1e-3 \
    --log-to-wandb \
    --project-name=ICML_LLM_Baseline3 \
    --log-dir="./logs/baseline3/llama-3.2-1b/sst2" \
    --total-samples=80000
```

**Additional ZO-specific Parameters:**
- `--num-pert=1`: Number of perturbations for zeroth-order gradient estimation
- `--mu=1e-3`: Perturbation magnitude for ZO estimation

### 3. Hybrid-Order SFL (`main_ho_sfl.py`)

Advanced variant with hybrid-order gradient estimation using hybrid approaches.

**Usage:**
```bash
python main_ho_sfl.py \
    --seed=42 \
    --cuda \
    --device=cuda:0 \
    --large-model=llama-3.2-1b \
    --lora \
    --split \
    --split-point=5 \
    --optimizer=adamw \
    --lr=1e-5 \
    --mu=1e-3 \
    --num-pert=2 \
    --num-clients=10 \
    --sampled-client-num=3 \
    --total-steps=1000 \
    --evaluation-interval=10 \
    --dataset=sst2 \
    --max-length=128 \
    --train-batch-size=32 \
    --test-batch-size=200 \
    --iid \
    --log-to-wandb \
    --project-name=ICML_LLM_Baseline_New \
    --log-dir="./logs/baseline_new/llama-3.2-1b/sst2" \
    --total-samples=80000
```

**Note:** HO-SFL uses `--total-steps` instead of `--total-rounds`



### 4. Automated Memory Profiling with Monitoring (`mock.sh`)

The `mock.sh` script is a convenience script that combines GPU memory monitoring with memory profiling for different frameworks. It's designed to quickly profile and compare GPU memory usage across multiple frameworks in one go.

**What it does:**
- Runs `monitor_gpu.py` + `main_mock.py` for each framework
- Collects GPU memory statistics for: HO-SFL, SFL, Centralized, and Inference modes
- Generates detailed memory usage reports for comparison

**Usage:**
```bash
bash mock.sh
```

**What happens when you run it:**

The script will sequentially execute memory profiling for 4 different frameworks:

1. **HO-SFL Framework Profiling**
   ```bash
   python ./monitor_gpu.py main_mock.py \
       --framework=HO-SFL \
       --large-model=llama-3.2-1b \
       --dataset=sst2 \
       --total-steps=10 \
       --total-rounds=3 \
       ...
   ```

2. **Standard SFL Framework Profiling**
   ```bash
   python ./monitor_gpu.py main_mock.py \
       --framework=SFL \
       ...
   ```

3. **Centralized Training Profiling**
   ```bash
   python ./monitor_gpu.py main_mock.py \
       --framework=Centralized \
       ...
   ```

4. **Inference-Only Profiling**
   ```bash
   python ./monitor_gpu.py main_mock.py \
       --framework=Inference \
       ...
   ```

**Output:**

Each run will generate two files in `./gpu_memory_results/`:
- `gpu_log_<pid>.csv`: Detailed timestamped memory usage
- `gpu_summary_<pid>.txt`: Summary showing maximum memory usage

**Example Output:**
```
GPU Maximum Memory Usage Statistics
====================================
Command: python main_mock.py --framework=SFL ...

GPU: NVIDIA GeForce RTX 5090 (GPU-f15c3518-9e2d-c0bd-fbe9-02794b2a80b4)
  Max Memory: 12345 MiB (12.05 GB)
  Time: 2026-01-29 14:30:15.123456
```

**Customizing the Script:**

You can edit `mock.sh` to change:
- **Model**: Change `--large-model=llama-3.2-1b` to `opt-125m` or `gemma-3-270m`
- **Dataset**: Change `--dataset=sst2` to `wsc` or `rte`
- **Steps**: Adjust `--total-steps=10` for longer profiling
- **Batch size**: Modify `--train-batch-size=32` to test different memory loads
- **Framework list**: Comment out frameworks you don't want to profile

---
