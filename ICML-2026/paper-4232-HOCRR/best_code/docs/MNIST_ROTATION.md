# MNIST Rotation Experiments

This experiment trains or loads an E2CNN regressor for rotated MNIST angle prediction, then compares `(E,C)+M`, `(E,C,G)+M`, and alpha-smoothing certificates.

## Train the Model

```bash
python experiments/mnist_rotation/train_e2cnn_rotation.py
```

This creates a local checkpoint. Keep checkpoints under `models/` or another ignored directory.

## Full Certification Estimates

Example for one sigma:

```bash
python experiments/mnist_rotation/mnist_rotation_full_certification.py \
  --model_path models/e2cnn_rotation_model.pth \
  --use_rotation_dataset \
  --sigma 0.75 \
  --n_test 100 \
  --N_values 10000 \
  --n_trials 5 \
  --confidence 0.90 \
  --output outputs/mnist_sigma0p75_conf90.json
```

Repeat for `sigma in {0.06,0.12,0.25,0.5,0.75}`.

## Alpha-Smoothing Baseline

```bash
python experiments/mnist_rotation/mnist_alpha_trimming_certification.py \
  --model_path models/e2cnn_rotation_model.pth \
  --sigma 0.06 \
  --alpha 0.49 \
  --n_test 100 \
  --n_tr 10000 \
  --n_sample 500 \
  --P 0.9 \
  --output outputs/mnist_alpha_sigma0p06_alpha0p49.json
```

## Figures and Tables

Use the plotting/postprocessing scripts in `experiments/mnist_rotation/`, especially:

- `compute_ec_radii_from_estimates.py`
- `compute_ecg_radii_from_estimates.py`
- `plot_cdf_best_sigma_updated.py`
- `compute_combined_certified_metrics_table.py`
- `plot_theta_convergence_analysis.py`
