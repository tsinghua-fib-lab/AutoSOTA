# Code for RSA-CP: Efficient Conformal Prediction in Small-Sample Regimes via Random Score Alignment

This folder contains the reproducible simulation code for the RSA-CP paper figures.
All scripts use relative paths and write outputs under `outputs/`.

## How to Run

From this `simulations/` folder:

```sh
Rscript run_all_simulations.R
```

Or render the R Markdown runner:

```r
rmarkdown::render("run_all_simulations.Rmd")
```

Individual figures can be regenerated with:

```sh
Rscript experiments/figure3_reference_score_size.R
Rscript experiments/figure6_noise_stability.R
Rscript experiments/figure7_calibration_size.R
Rscript experiments/shock_probability_experiment.R
```

Required R packages are listed in `requirements.txt`.

## Folder Structure

- `R/`: method implementations, DGPs, simulation helpers, plotting helpers.
- `experiments/`: one script per simulation experiment.
- `checks/`: method and output reproducibility checks.
- `outputs/raw/`: trial-level CSV files.
- `outputs/summary/`: grouped summary CSV files.
- `outputs/figures/`: PDF and PNG figures.
- `outputs/logs/`: run logs, method checks, session info.

## How the Four Figures Are Generated

### Figure 3: reference/synthetic score size

Script: `experiments/figure3_reference_score_size.R`

This varies the reference or synthetic score size `N` over
`c(50, 100, 250, 500, 1000, 2000, 3000)` in the LogNormal and Student-t
regimes. It compares `SCP`, `RSA-CP (OT) (Ours)`, `SPI`, and
`Synthetic-only` on coverage, average width, and computation time.

Outputs:

- `outputs/raw/figure3_reference_score_size_raw.csv`
- `outputs/summary/figure3_reference_score_size_summary.csv`
- `outputs/figures/figure3_reference_score_size.pdf`
- `outputs/figures/figure3_reference_score_size.png`

### Figure 6: generator-noise stability

Script: `experiments/figure6_noise_stability.R`

This keeps the original simulation generator idea and varies the synthetic
generator noise level. `SPI` and `Synthetic-only` use full synthetic data and
include synthetic data generation plus score computation in their timing.
`RSA-CP (OT) (Ours)` uses the generated scores as reference scores for this
stress test, but its timing counts only score-level RSA-CP calibration, not
full synthetic `X, Y` generation.

Outputs:

- `outputs/raw/figure6_noise_stability_raw.csv`
- `outputs/summary/figure6_noise_stability_summary.csv`
- `outputs/figures/figure6_noise_stability.pdf`
- `outputs/figures/figure6_noise_stability.png`

### Figure 7: calibration size and score-distribution sensitivity

Script: `experiments/figure7_calibration_size.R`

The left half varies calibration size `m` over `c(20, 30, 40, 80, 160)`.
The right half is a boxplot sensitivity check over reference-score
distribution assumptions: `SCP`, `RSA-CP Gamma`, `RSA-CP Normal`, and
`RSA-CP Beta`.

Outputs:

- `outputs/raw/figure7_calibration_size_raw.csv`
- `outputs/summary/figure7_calibration_size_summary.csv`
- `outputs/raw/figure7_score_distribution_boxplot_raw.csv`
- `outputs/summary/figure7_score_distribution_boxplot_summary.csv`
- `outputs/figures/figure7_calibration_size_and_score_distribution.pdf`
- `outputs/figures/figure7_calibration_size_and_score_distribution.png`

### Shock probability drift experiment

Script: `experiments/shock_probability_experiment.R`

At shock probability `p = 0`, real calibration scores and reference scores come
from the same held-out score distribution. For `p > 0`, only the reference
scores receive positive tail shocks, so the reference distribution drifts away
from the real/test score distribution. The plot compares only `SCP` and
`RSA-CP (OT) (Ours)`.

Outputs:

- `outputs/raw/shock_probability_raw.csv`
- `outputs/summary/shock_probability_summary.csv`
- `outputs/figures/shock_probability.pdf`
- `outputs/figures/shock_probability.png`

## Method Notes

`SCP` is standard split conformal prediction using only real calibration scores.

`Synthetic-only` calibrates directly on synthetic scores.

`SPI` uses the SPI fast-form score threshold with rank windows. It is not a
pooled conformal heuristic.

`RSA-CP (OT) (Ours)` uses reference scores, barycentric OT alignment from real
calibration scores to the reference-score scale, candidate ranks after
alignment, and Beta-Binomial rank-window calibration.

