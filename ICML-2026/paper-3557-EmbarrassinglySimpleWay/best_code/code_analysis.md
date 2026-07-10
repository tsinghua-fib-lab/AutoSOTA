# Code Analysis — Paper 3557 SOTA Optimization

## Preparation Failure Diagnosis

The preparation script failed for two reasons:
1. **Proxy connectivity**: The container proxy `172.17.0.1:17890` returned 502 Bad Gateway for some `archive.ubuntu.com` packages, causing `apt-get install git` to fail.
2. **Missing git**: The container image `autosota/paper-3557:reproduced` does not have git pre-installed, and the preparation script exits with code 127 when `git` is not found.

**Fix**: Install git without proxy (`unset HTTPS_PROXY HTTP_PROXY ...`) then run `apt-get install -y git`. The proxy is not needed for archive.ubuntu.com.

## Corrected In-Container Evaluation Command

```bash
cd /repo
python3 orthogonal_cnn_cifar10.py --epochs 100 --runs 5 --seed 42 --output results_pogo_cifar10.json
```

This command produces:
- stdout: training log with per-epoch metrics per run
- stdout `SUMMARY` block: mean accuracy, std accuracy, mean time, std time across runs
- `results_pogo_cifar10.json`: JSON output with per-run results and summary

## Baseline Verification

Existing `results_pogo_cifar10.json` matches the manifest:
- Accuracy (TTA): 0.9145 (91.45%) — within [91.0%, 92.2%]
- Training Time: 1.267 min/run

## Code Architecture

### Key files
- `/repo/orthogonal_cnn_cifar10.py` (548 lines): Self-contained experiment script
- `/repo/pogo/`: POGO optimizer library
  - `pogo.py`: POGO Stiefel manifold optimizer
  - `base.py`: Base Euclidean optimizers (SGD, VectorAdam, Muon)

### Key Optimization Surfaces

1. **`train_one_run()` function** (lines ~280-480):
   - POGO wrapped around `base.VectorAdam()` for orthogonal conv weights
   - Separate SGD optimizer for non-orthogonal params (whiten bias, head)
   - Linear LR decay scheduler
   - Cross-entropy with label_smoothing=0.2
   - Augmentation: flip + translate(2)

2. **`main()` function** (lines ~483-548):
   - Runs 5 independent runs with incremental seeds

3. **POGO optimizer** (in `pogo/pogo.py`):
   - `base_optimizer`: VAdam (Vector Adam with scalar second moment)
   - `lambda_every=-1`: Fixed λ=0.5 (no polynomial root-finding)
   - `rows=True`: Row-orthogonal constraint (O < I*k*k)

### Safe Optimization Targets

- LR schedule: Replace linear decay with cosine annealing + warmup
- Base optimizer: Available options are SGD, VectorAdam, Muon (in base.py)
- Custom base optimizer: Can implement Adam (per-element moments) in base.py
- Label smoothing: Can use cosine schedule
- Data augmentation: Can add RandAugment, Mixup
- EMA: Can add weight averaging for evaluation

### Constraints
- All orthogonal conv weights MUST remain float32 (POGO requirement)
- Training must use POGO or derived orthoptimizer
- Evaluation protocol (TTA level 2) must remain unchanged
