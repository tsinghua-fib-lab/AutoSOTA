# Code Analysis: GFFMERGE (Paper 4790) — SOTA Preparation Repair

## Original Preparation Failure

**Root Cause:** Disk space exhaustion on the Docker overlay filesystem (200G capacity, 100% used). The `git add -A` command failed when trying to index `M3GNet/runs/repro_eval/merged_ft.ckpt` (3.4MB) because the overlay had zero bytes available.

**Secondary Issue:** The container had accumulated 191G of data from cache mounts and previous experiments. No `.gitignore` existed, so git attempted to track all files including large data files (aspirin.extxyz: 484MB, uracil.extxyz: 180MB) and checkpoint files.

## Repair Actions

1. **Cleaned pip/conda caches and temp files** — freed ~9.5GB
2. **Reinitialized git repository** — original git objects were corrupted from the partial `git add`; reinitialized from scratch since the original commit history was lost
3. **Added `.gitignore`** — excludes `*.extxyz`, `*.ckpt`, `*.pt`, `*.npz`, `runs/`, `tmp/`, `__pycache__/`, `*.log`
4. **Created clean baseline commit** and `_baseline` tag
5. **Copied `/tools/record_score.sh`** into container
6. **Verified baseline evaluation** produces metrics matching the reproduction manifest

## Corrected In-Container Evaluation Command

```bash
cd /repo/M3GNet && export TMPDIR=/repo/tmp && mkdir -p /repo/tmp && python3 eval_gffmerge.py \
  --ckpt-a runs/aspirin_seed42/best_chk.ckpt \
  --ckpt-b runs/uracil_seed42/best_chk.ckpt \
  --config-a runs/aspirin_seed42/config.yaml \
  --config-b runs/uracil_seed42/config.yaml \
  --label-a aspirin --label-b uracil \
  --output-dir runs/<experiment_name> --seed 42
```

Additional flags for optimization: `--epochs-ft`, `--limit-ft`, `--lr-ft`, `--force-weight`, `--energy-weight`, `--last-n-blocks`, `--batch-size`

## Baseline Verification Evidence

| Metric | Manifest Baseline | Reproduced (eV) | Reproduced (kcal/mol) | Match |
|--------|------------------|-----------------|----------------------|-------|
| Energy MAE | 0.0293 kcal/mol | 0.001271 eV | 0.02931 kcal/mol | ✓ (Δ=0.00001) |
| Force MAE | 0.915 kcal/mol/Å | 0.03966 eV/Å | 0.9146 kcal/mol/Å | ✓ (Δ=0.0004) |

Conversion: divide eV by 0.0433641153 to get kcal/mol (per manifest eval_output_format).

## Pipeline Structure

The `eval_gffmerge.py` script runs three stages:
1. **GFFMERGE closed-form merge** (`scripts/merge_closed_form_individual_m3gnet.py`) — merges two single-task checkpoints into one multi-task checkpoint. Time: ~6s.
2. **Fine-tuning** (`scripts/switch_finetune_energy_readout_last_block_m3gnet.py`) — fine-tunes last N blocks (default 3) on combined data with switch embedding. Time: ~5min (10 epochs).
3. **Switch embedding evaluation** (`scripts/evaluate_switch_embeddings_m3gnet.py`) — evaluates on test set using per-sample embedding switching.

## Safe Optimization Targets

### Merge stage (`merge_closed_form_individual_m3gnet.py`)
- `--regularization` (default 1e-6): controls diagonal loading in closed-form solve
- Per-layer adaptive regularization based on condition number diagnostics (already logged)
- Sequential layer merging with propagated activations

### Fine-tuning stage (`switch_finetune_energy_readout_last_block_m3gnet.py`)
- Loss function: Huber loss instead of MSE for forces
- Energy-force consistency regularization
- Dropout regularization before trainable layers
- Gradient clipping in optimization loop
- Early stopping based on validation loss

### Hyperparameters (`eval_gffmerge.py`)
- `--force-weight` (default 0.1): weight for force loss term
- `--energy-weight` (default 1.0): weight for energy loss term
- `--last-n-blocks` (default 3): number of graph layers to fine-tune
- `--epochs-ft` (default 10): fine-tuning epochs
- `--lr-ft` (default 1e-4): learning rate
- Cosine LR schedule with extended training

## Constraints
- DGL CUDA backend unavailable; training runs on CPU
- TMPDIR must be /repo/tmp for checkpoint writes
- GPU devices 2,3 available for evaluation
- Overlay disk space: ~9.5GB free after cleanup; large checkpoints must use /repo/tmp
- Checkpoints: ~3.4MB each; fine to store in runs/ subdirectories (gitignored)

## Key Files
- `/repo/M3GNet/eval_gffmerge.py` — main evaluation orchestrator
- `/repo/M3GNet/scripts/merge_closed_form_individual_m3gnet.py` — GFFMERGE merge
- `/repo/M3GNet/scripts/switch_finetune_energy_readout_last_block_m3gnet.py` — fine-tuning
- `/repo/M3GNet/scripts/evaluate_switch_embeddings_m3gnet.py` — evaluation
- `/repo/M3GNet/configs/` — YAML configs for training/eval
- `/repo/M3GNet/data/cache/` — preprocessed DGL graph caches
