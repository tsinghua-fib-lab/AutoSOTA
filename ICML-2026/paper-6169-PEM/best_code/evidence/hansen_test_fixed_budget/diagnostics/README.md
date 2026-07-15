# Residual-Pool Diagnostics (BERW-Hetero) — Hansen Fixed-Budget Slice

Goal: make the theory’s “mismatch decomposition” *measurable* on real runs, rather than
an unverifiable narrative.

This folder contains **internal state traces** from BERW-Hetero runs on a small bbob-noisy
fixed-budget slice (the same budget scale as the Hansen money-plot), plus a summary table/plot
for compact reporting.

These diagnostics are designed to make the underlying assumptions *refutable* by exposing internal state traces.

## What’s inside

- `state_index.csv`: index of per-problem state trace CSV files.
- `traces/*.csv`: per-problem per-generation state traces, including:
  - pool size (`noise_z_pool_size`)
  - clipping saturation (`noise_z_clip_frac`)
  - within-generation shape-shift (`noise_shape_w1`, `noise_shape_ks`)
  - drift vs previous generation (`noise_drift_w1`, `noise_drift_ks`)
  - scale-model fit quality (`noise_scale_fit_r2`) and predicted-scale spread (`noise_scale_pred_cv`)
  - split-median centering stability (`noise_center_split_rel`, `noise_center_split_cv`)
- `diagnostics_summary.csv`: per-(function,instance) summary (mean/max/final of the above).
- `diagnostics_summary.png`: boxplots grouping diagnostics by function index.
- `perf_vs_diagnostics.csv`: merges noisefree Hansen outcomes with diagnostics (per (function,instance)).
- `perf_vs_diagnostics.png`: scatter plots of performance ratio vs diagnostics (sanity; not a causal claim).
- `worst_cases.md`: a small table of the most BERW-worse pairs (boundary examples).

## How to interpret

These diagnostics are designed to make the theory *refutable*:
if any assumption is badly violated, the corresponding statistic should visibly degrade.

- **Pool size**: if `noise_z_pool_size` stays tiny, BERW is effectively operating without a usable residual pool.
- **Clipping saturation**: a large `noise_z_clip_frac` indicates the standardization+clipping is doing heavy lifting;
  in that case any guarantee should be interpreted as a guarantee for a *winsorized* noise model.
- **Shape shift / drift**: large `noise_shape_w1` or `noise_drift_w1` indicates distribution mismatch across
  heteroskedastic buckets or time; this flags the “adaptive pool mismatch” term in the theory’s error decomposition.
- **Scale / center stability**: low `noise_scale_fit_r2` or large `noise_scale_pred_cv` suggests the linear
  heteroscedastic scale model is strained; large `noise_center_split_rel` or `noise_center_split_cv` indicates the
  robust centering (median) is unstable given current reevaluations.

Notes:
- Several statistics are computed only from the (few) reevaluated points each generation; in very low-noise regimes
  reevaluations may be rare, so these fields can be `nan`. This is expected (and itself indicates “not the regime”).

## Setup

- Suite: `bbob-noisy`
- Dimension: `D=40`
- Instances: `1–15` (COCO standard)
- Functions (indices): the same high-misranking subset used by the Hansen fixed-budget package:
  `8,10,11,13,14,16,17,19,20,22,23,25,26,28,29`
- Budget: `B=100×D`
- Algorithm: `BERW-Hetero`

## Reproduce

```bash
python3 tools/run_hansen_fixed_budget_residual_diagnostics.py \
  --out-dir evidence/hansen_test_fixed_budget/diagnostics \
  --dims 40 --functions 8,10,11,13,14,16,17,19,20,22,23,25,26,28,29 --instances 1-15 --budget-mult 100

python3 tools/analyze_hansen_diagnostics_vs_performance.py \
  --out-dir evidence/hansen_test_fixed_budget/diagnostics
```
