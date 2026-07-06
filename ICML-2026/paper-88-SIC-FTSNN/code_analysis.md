# Code Analysis — SOTA Preparation Repair for Paper 88

## Original Preparation Failure

The preparation step failed because the FMNIST dataset directory `/datasets/fmnist` was missing in the SOTA container `autosota_sota_paper_88`. The eval command creates a symlink `propdata/FMNIST -> /datasets/fmnist` and then calls `datasets.FashionMNIST(data_path, download=True)`. Torchvision's `FashionMNIST.__init__` calls `os.makedirs(self.raw_folder)` where `raw_folder = data_path/FashionMNIST/raw`. Since the symlink target `/datasets/fmnist` didn't exist, `os.makedirs` raised `FileNotFoundError`.

**Root cause**: The FMNIST dataset was downloaded during the reproduction run into the `/datasets` NFS mount, but the dataset directory was subsequently cleaned up. The NFS mount `/datasets` is present and writable; only the `fmnist/` subdirectory was missing.

## Repair Applied

1. Created `/datasets/fmnist/` directory in the container
2. Re-ran the standard eval command from `/repo`
3. FMNIST re-downloaded automatically by torchvision via `download=True`

## Corrected Evaluation Command

```bash
cd /repo
mkdir -p propdata && ln -sfn /datasets/fmnist propdata/FMNIST
python3 simple_snn.py \
  --data_path propdata/FMNIST \
  --Fault True \
  --fault_type stuck \
  --fault_ratio 0.3 \
  --Dynamic True \
  --num_steps 8 \
  --num_epochs 50 \
  --batch_size 100 \
  --learning_rate 0.001 \
  --gpu_num 0 \
  --plot False
```

All paths are container-local. No Docker exec wrappers needed.

## Baseline Verification

Expected baseline from reproduction manifest: Accuracy = 87.16%
GPU: NVIDIA A100-SXM4-80GB (gpu_num=0, first of 2 visible GPUs)
PyTorch: 2.1.0 with CUDA 12.1

## Container State

- Container: `autosota_sota_paper_88` (running, image `autosota/paper-88:reproduced`)
- Mounts: `/datasets`, `/models`, `/autosota_cache`, `/autosota_artifacts`
- Git: repo at `/repo`, tags: `_baseline` (commit `8f96a86`)
- Tools: `/tools/record_score.sh` exists and is executable

## Safe Optimization Targets

### 1. Training hyperparameters (simple_snn.py)
- `--num_epochs` (line ~47): extend from 50
- `--learning_rate` (line ~48): scheduler type (currently StepLR gamma=0.75)
- `--batch_size` (line ~43): currently 100
- Optimizer: Adam at line 541

### 2. LIF Neuron parameters (simple_snn.py lines 231-242)
- `surrogate_function=surrogate.ATan()` → can change to `Sigmoid(alpha=4.0)`
- `v_threshold=1.0` → can make learnable
- `tau=2.0` → membrane time constant

### 3. Fragmentation config (simple_snn.py lines 508-536)
- `gumbel_tau=1.0` → annealing schedule (learnable_fragmentation.py line 1476)
- `warmup_iters=500` → entropy-gated warmup
- `balance_weight=0.01`, `line_sep_weight=1e-3`, `line_cross_weight=1e-3`
- `init_logit_bias=4.0` in DynamicGlobalMultiLineFragsMoE

### 4. Training loop (simple_snn.py lines 640-690)
- Loss computation: add firing rate regularization
- Gradient clipping before optimizer.step()
- Label smoothing: soften target_onehot

### 5. Architecture (simple_snn.py lines 227-241)
- Add FragNorm between hidden Linear→LIF layers
- Currently only input layer has FragNorm

### 6. Fault injection (simple_snn.py lines 284-296)
- Per-epoch fault seed variation for robustness

## Reusable Resources
- `/datasets/fmnist` — FMNIST dataset (auto-downloaded)
- No `/paper_data` mount available
