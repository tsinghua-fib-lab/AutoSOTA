# Code Analysis & Preparation Repair — Paper 5441

## Original Preparation Failure

The orchestrator's preparation step failed because:
1. **No git in container**: The `autosota/paper-5441:reproduced` image (based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`) does not include git.
2. **apt-get blocked by proxy**: The proxy configuration caused 502 errors from `archive.ubuntu.com` and `security.ubuntu.com` when attempting `apt-get install git`.
3. **conda also blocked**: `conda install git` also failed with proxy connection errors.

## Repair Applied

1. Copied host git binary (`/usr/bin/git`) and its shared libraries (libc, libpcre2-8, libpthread, libz) into `/usr/local/git/lib/`.
2. Created wrapper script `/usr/local/bin/git` that sets `LD_LIBRARY_PATH` to include those libs.
3. Initialized git repo in `/repo` with baseline commit and `_baseline` tag.
4. Copied `/tools/record_score.sh` from host.
5. Created `/autosota_artifacts/paper-5441/sota/` directory structure.

## Corrected Evaluation Command

**In-container command** (from `/repo`):
```
python3 reproduce_threshold.py
```

## Baseline Verification

| Metric | Paper | Reproduced | Match |
|--------|-------|------------|-------|
| Mean r  | 0.968  | 0.9685 | PASS  |
| Std r   | 0.003  | 0.0031 | PASS  |
| Min r   | 0.961  | 0.9608 | PASS  |
| Max r   | 0.973  | 0.9731 | PASS  |

15 runs on 81-point grid, threshold mechanism. Matches paper Table 5 results.

## Code Architecture

- `NAVAR.py`: MLP-based NAVAR and LSTM-based NAVARLSTM models
- `train_NAVAR.py`: Training loop with MSE + L1 penalty, batch sampling
- `reproduce_threshold.py`: Main evaluation script: data generation, training, ICE estimation, Pearson r

Key config (baseline): maxlags=1, hidden_nodes=16, hidden_layers=1, dropout=0.10, epochs=2000, lr=3e-4, lambda1=0.15, weight_decay=0.001, batch_size=128

## Safe Optimization Targets

- `train_NAVAR.py:71`: Loss function (MSE to Huber)
- `train_NAVAR.py:134-144`: Add early stopping with best-model checkpointing
- `train_NAVAR.py:72`: Add LR scheduler
- `train_NAVAR.py:131`: Add gradient clipping
- `train_NAVAR.py:125-126`: Regularization (L1 to group lasso)
- `train_NAVAR.py:96-104`: Batch sampling strategy
- `reproduce_threshold.py:74-87`: MC dropout at inference
- `reproduce_threshold.py:101-105`: Vectorized GPU-batched ICE
- `reproduce_threshold.py:126-141`: Model capacity parameters
