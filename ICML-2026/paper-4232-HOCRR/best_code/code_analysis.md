# Code Analysis for Paper 4232: Higher-Order Certified Robustness for Regression

## Evaluation Path

- **Eval script:** `/repo/eval_certification.py`
- **Eval command:** `python3 eval_certification.py --skip_estimation --skip_radii`
- **Input files:**
  - `outputs/mnist_sigma0p75_estimation.json` — pre-computed variance/gradient estimates
  - `outputs/mnist_ecg_radii_sigma0p75.json` — pre-computed certified radii
- **Output:** stdout table + `outputs/reproduction_summary.json`
- **Metric names:** `AbsAcc_R{radius}`, `CondAcc_R{radius}`, `MeanDist_R{radius}`
- **Parser:** `compute_metrics()` function reads estimation JSON for clean preds and true angles, radii JSON for certified radii, computes all metrics in-memory

## Full Pipeline (3 steps)

### Step 1: Estimation
```
python3 experiments/mnist_rotation/mnist_rotation_full_certification.py \
  --model_path models/e2cnn_rotation_model.pth --use_rotation_dataset \
  --sigma 0.75 --n_test 100 --N_values 10000 --n_trials 1 \
  --confidence 0.90 --device cuda --stratified --seed 42 --skip_bootstrap \
  --output outputs/mnist_sigma0p75_estimation.json
```
- Uses `BoundedCertifierConvergenceValidator` from `bounded_certifier_convergence_analysis.py`
- Computes variance (C) UCB, gradient norm (G) UCB, theta estimates

### Step 2: Radii
```
python3 experiments/mnist_rotation/compute_ecg_radii_from_estimates.py \
  --estimation_file outputs/mnist_sigma0p75_estimation.json \
  --eps_y_deg 10.0 --N 10000 --trial 0 --ci_type analytical \
  --confidence 0.90 --output outputs/mnist_ecg_radii_sigma0p75.json
```
- Uses `BoundedCertifierWithMean` from `bounded_fn_certifier_with_mean.py`
- Reads pre-computed UCBs, computes certified radius per sample

### Step 3: Eval
```
python3 eval_certification.py --skip_estimation --skip_radii
```
- Parses both JSONs, prints table, saves summary

## Key Source Files

### Training
- `experiments/mnist_rotation/train_e2cnn_rotation.py` — training script (30 epochs, StepLR, MSE loss)
- `experiments/mnist_rotation/e2cnn_rotation_model.py` — RotationEquivariantCNN_Simple (N=8, ~50K params)
- `experiments/mnist_rotation/dataset_generator.py` — MNISTRotationDataset (augmentation_factor=1)

### Certification Estimation
- `experiments/mnist_rotation/bounded_certifier_convergence_analysis.py` — BoundedCertifierConvergenceValidator with U-statistic estimators (variance, gradient norm, theta)
- `src/regression_certifiers/certify/variance_gradient_certifier.py` — VarianceGradientCertifier (used for legacy U-statistic estimates)

### Radius Computation
- `src/regression_certifiers/certify/bounded_fn_certifier_with_mean.py` — BoundedCertifierWithMean ((E,C,G)+M formulation)
- `experiments/mnist_rotation/compute_ecg_radii_from_estimates.py` — CLI wrapper

### Evaluation
- `eval_certification.py` — metric computation and table output

## CI Code Pattern (Target for CODE-01)

All certifier/validator files use the same pattern:
```python
alpha_total = 1 - self.confidence
alpha_split = alpha_total / K  # K = 2 or 3 (number of estimators)
z_critical = norm.ppf(1 - alpha_split / 2)  # TWO-sided → overly conservative for UCB
```
Fix: Change to `norm.ppf(1 - alpha_split)` for one-sided upper confidence bounds.

Files to fix:
1. `bounded_certifier_convergence_analysis.py` — lines 92-96 (variance), 421-425, 500-504, 870-945 (gradient), 1638-1689 (theta CI)
2. `variance_gradient_certifier.py` — lines with `norm.ppf(1 - alpha_split / 2)`
3. `bounded_fn_certifier_with_mean.py` — lines 90-92, 135-137, 166-168
4. `bounded_fn_certifier_variance_mean.py` — lines 103-105, 131-133

## Safe Modification Targets
- Training script (model architecture, loss, LR schedule, epochs)
- Dataset augmentation factor
- Model N parameter (equivariance group order)
- CI computation (z-critical, t-critical)
- n_trials parameter
- MC sample count (N_values)
- Bootstrap CI enable/disable

## Risky/Do-Not-Modify
- `eval_certification.py` — metric definitions, formula (except at your own risk)
- Test data/splits
- `/tools/record_score.sh`
- `outputs/reproduction_summary.json` (read-only for baseline)

## Known Issues
1. CI double-splitting: `alpha_split/2` in `norm.ppf` gives two-sided CI; certification only needs one-sided UCB
2. No best-validation checkpointing: training saves only final epoch
3. DataLoader not reproducible (no generator seed)
4. n_trials=1 vs paper n_trials=5
5. Model trained only 30 epochs with StepLR
6. augmentation_factor=1 (minimal data diversity)

## GPU Configuration
- Devices: 4,5 (CUDA_VISIBLE_DEVICES should be set)
- Container has CUDA access

## Cache Mounts
- `/autosota_cache` — shared cache
- `/datasets` — pre-downloaded datasets
- `/models` — pre-downloaded models
