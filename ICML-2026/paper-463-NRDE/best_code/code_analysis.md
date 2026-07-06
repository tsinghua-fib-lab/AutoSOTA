# Code Analysis — Paper 463: Noise-Robust Density Estimation (NRDE)

## Evaluation Path
- **Entry point:** `eval_mammography.py` → calls `NRDE_run()` from `NRDE.py` with best hyperparams
- **Data loading:** `read_data()` in NRDE.py (z-score normalization, 50:50 normal-only train/test split)
- **Metric computation:** `testing()` in NRDE.py uses `torchmetrics.AUROC` and `torchmetrics.AveragePrecision`
- **Output format:** JSON stdout with `auroc_mean`, `auroc_std`, `auprc_mean`, `auprc_std`, `individual_runs`

## Training/Inference Path
- `NRDE_run()` orchestrates:
  1. Normalize data (z-score)
  2. Create `_RealNVP` model
  3. Per epoch: `train_one_epoch()` → `contribution_calculation()` → `build_contributed_testset()` → `testing()`
  4. Returns best AUROC/AUPRC across epochs

## Model Architecture (`model.py`)
- `_RealNVP` with 2 coupling layers (more are commented out at lines 687–703)
- `CouplingLayer` uses checkerboard masks, `ResidualBlock` for s/t network
- `ResidualBlock`: `cat(x, -x)` → `fc1` → (fc2, fc3 commented out) → `fc4`
- Key params: `input_dim=6`, `mid_dim=2048`, `masktype=0` (checkerboard)

## Config Path
- Hardcoded in `eval_mammography.py`: lr=0.005, grad_pun=1.0, epochs=100, bs=512, mid_dim=2048, act=2 (LeakyReLU), PNAL='L_1sq', n_runs=5

## Metric Parser
- JSON stdout from eval_mammography.py

## Reusable Resources
- Dataset: `/datasets/23_mammography.npz` (11183 samples, 6 features, 260 anomalies)
- No `/paper_data` mount

## Risky Files (DO NOT MODIFY)
- `eval_mammography.py` — metric definitions, data split logic
- `NRDE.py` `read_data()` — train/test split
- `testing()` — AUROC/AUPRC computation
- `first_i_by_max_gap_ratio_1d()` — contribution calculation algorithm
- `build_contributed_testset()` — test set construction

## Safe Modification Targets
- `model.py` `_RealNVP.__init__` — uncomment coupling layers
- `model.py` `ResidualBlock.__init__` — add spectral_norm, modify feature augmentation
- `NRDE.py` `train_one_epoch()` — add KL loss, center loss, noise injection
- `NRDE.py` `NRDE_run()` — extend epochs, add LR scheduler, grad_pun warmup
- `eval_mammography.py` — ensemble score averaging (post-hoc aggregation, not metric change)
