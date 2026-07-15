# Real-data experiments for RSA-CP: Efficient Conformal Prediction in Small-Sample Regimes via Random Score Alignment

This folder is the GitHub-ready real-data artifact for the RSA-CP paper.  It
organizes the ImageNet classification and MEPS regression experiments while
preserving the original SPI repository data-loading/config style.

The original SPI repository is:

```text
https://github.com/Meshiba/spi
```

## Folder Structure

```text
real_data/
  README.md
  requirements.txt
  run_all_real_data.py
  run_all_real_data.ipynb
  real_data_pipeline.py
  plotting_boxplots.py
  plotting_paper_style.py
  methods/
    score_only_methods.py
    score_ot_utils.py
    score_dist_method.py
    rsa_cp_core.py
  experiments/
    figure4_imagenet_main.py
    figure5_imagenet_ncal.py
    figure8_imagenet_nsyn.py
    figure9_meps_age_groups.py
    figure10_alignment_mismatch.py
  configs/
    imagenet_clip_marginal.yml
    meps_regression_ages_0_to_20.yml
    meps_regression_ages_20_to_40.yml
    meps_regression_ages_40_to_60.yml
    meps_regression_ages_60_to_100.yml
  outputs/
    raw/
    summary/
    figures/
    logs/
  notebooks/
    RSA_CP_MEPS_score_only_clean.ipynb
    RSA_CP_ImageNet_score_only_clean.ipynb
  checks/
    method_correctness_check.py
    old_method_check.py
```

## Requirements

Python 3.10+ is recommended. Install packages with:

```bash
pip install -r requirements.txt
```

The core scripts use `numpy`, `pandas`, and `Pillow`. `jupyter`/`nbformat` are
only needed for the clean notebooks.

## Data Requirements

Large real-data score/prediction files are not included in this release zip.
Place the data using the same relative layout as the original SPI repository:

```text
data/
  imagenet/models/CLIP-ViT-B-32-laion2B-s34B-b79K/imagenet_train/
  imagenet/models/CLIP-ViT-B-32-laion2B-s34B-b79K/sdv5/
  meps/models/quantile_regression/alpha_0.1/meps_21/...
  meps/models/quantile_regression/alpha_0.1/meps_20/...
```

The expected file names are the existing SPI-style arrays:

- `pred.npy`
- `true.npy`

By default, paths in `configs/*.yml` are resolved relative to:

1. `RSA_CP_DATA_ROOT`, if the environment variable is set;
2. this `real_data/` folder;
3. the parent repository root.

For example:

```bash
set RSA_CP_DATA_ROOT=.
python run_all_real_data.py
```

On macOS/Linux:

```bash
export RSA_CP_DATA_ROOT=.
python run_all_real_data.py
```

## Reproduce All Real-data Figures

From this `real_data/` folder:

```bash
python run_all_real_data.py
```

If raw CSVs already exist and you only want to regenerate figures:

```bash
python run_all_real_data.py --plots-only
```

The main runner regenerates:

- raw CSV files under `outputs/raw/`
- summary CSV files under `outputs/summary/`
- PNG/PDF figures under `outputs/figures/`
- logs under `outputs/logs/`
- `outputs/real_data_figures_manifest.csv`

## Reproduce Individual Figures

```bash
python experiments/figure4_imagenet_main.py
python experiments/figure5_imagenet_ncal.py
python experiments/figure8_imagenet_nsyn.py
python experiments/figure9_meps_age_groups.py
```

`experiments/figure10_alignment_mismatch.py` is included as an optional helper
for the score-level alignment mismatch stress test. The current real-data
release focuses on Figures 4, 5, 8, and 9.

## Expected Outputs

Figure 4:

- `outputs/raw/figure4_imagenet_main_raw.csv`
- `outputs/summary/figure4_imagenet_main_summary.csv`
- `outputs/figures/figure4_imagenet_main_boxplot.png`
- `outputs/figures/figure4_imagenet_main_boxplot.pdf`

Figure 5:

- `outputs/raw/figure5_imagenet_ncal_raw.csv`
- `outputs/summary/figure5_imagenet_ncal_summary.csv`
- `outputs/figures/figure5_imagenet_ncal.png`
- `outputs/figures/figure5_imagenet_ncal.pdf`

Figure 8:

- `outputs/raw/figure8_imagenet_nsyn_raw.csv`
- `outputs/summary/figure8_imagenet_nsyn_summary.csv`
- `outputs/figures/figure8_imagenet_nsyn.png`
- `outputs/figures/figure8_imagenet_nsyn.pdf`

Figure 9:

- `outputs/raw/figure9_meps_age_groups_raw.csv`
- `outputs/summary/figure9_meps_age_groups_summary.csv`
- `outputs/figures/figure9_meps_age_groups_boxplot.png`
- `outputs/figures/figure9_meps_age_groups_boxplot.pdf`

## Method Notes

`SCP` is standard split conformal prediction using only real/minority
calibration scores.

`Synthetic-only` calibrates using reference/synthetic scores only.

`SPI` uses the true fast-form SPI score threshold implemented in
`methods/score_ot_utils.py`. If replacing it with pooled conformal in another
experiment, rename the method to `Pooled CP`.

`RSA-CP (OT) (Ours)` uses score-level Random Score Alignment with barycentric
OT and the Beta-Binomial rank window.

## Important Implementation Notes

MEPS uses CQR endpoints. If a MEPS prediction input has shape `(n, 1000)`, the
runner converts the quantile grid into lower/upper endpoints:

```text
q_lower = alpha / 2
q_upper = 1 - alpha / 2
S = max(q_lower(x) - y, y - q_upper(x))
```

ImageNet uses APS scores. ImageNet probability/logit matrices are never treated
as CQR endpoints.

RSA-CP (OT) uses barycentric OT:

```text
T(S_(i)) = sum_j P_ij S_ref_(j) / sum_j P_ij
```

and candidate-specific Beta-Binomial rank-window calibration:

```text
k = 1 + sum(Z_real <= Z_new)
B | k ~ BetaBin(N, k, m + 2 - k)
include iff Z_new <= q_rsa(k)
```

RSA-CP here is not OT-score augmentation plus a standard conformal quantile. It
does not use mean scaling or weighted quantiles.

## Checks

Run:

```bash
python checks/method_correctness_check.py
python checks/old_method_check.py
```

These write:

- `outputs/logs/method_correctness_check.txt`
- `outputs/logs/old_method_check.txt`

The full runner also writes:

- `outputs/logs/cqr_endpoint_check.txt`
- `outputs/logs/imagenet_aps_check.txt`
- `outputs/logs/run_log.txt`

## Troubleshooting

If a config is missing, check the `configs/` folder and make sure you run from
the `real_data/` folder.

If data are missing, either place them under the SPI-style `data/` layout or set
`RSA_CP_DATA_ROOT` to the folder containing `data/`.

If runtime is long, first run `python run_all_real_data.py --plots-only` to
verify plotting from included CSVs, then run the full pipeline when ready.

If a method named `SPI` is replaced by pooled real+synthetic conformal in future
experiments, rename it to `Pooled CP` in tables and figures.
