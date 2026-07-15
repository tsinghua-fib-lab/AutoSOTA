# Probe ROC / Budget Sweep (bbob-noisy)

This evidence package quantifies **how reliable the misranking probe is as a classifier** when its budget changes.
It addresses the practical question: how much probe budget is needed to make reliable decisions?

## Setup

- Script: `tools/probe_budget_roc.py`
- Suite: COCO `bbob-noisy`
- Labels (noise-free): from `Results/bbob_noisy_d40_i1-15_probe_labels_B200/noisefree/bbob_summary.csv`
  - `label=berw` if `BERW-Hetero` beats `CMA-ES-sep`
  - `label=cma` otherwise (ties dropped; see `summary.json`)
- Dimension: `D=40`, budget `B=200×D`
- Functions: `1–30` (i.e., `101–130`), instances `1–5` → `n_labeled=150`
- Probe budget proxy: misranking probe candidate count `λ` (`lam_override`)
  - each probe uses two noisy draws → ≈ `2λ` function evaluations
- λ list: `{4, 8, 16, 32}`

## Outputs

- `evidence/bbob_noisy_probe_budget_roc/roc.csv`: per (λ, threshold) confusion + TPR/FPR + accuracy
- `evidence/bbob_noisy_probe_budget_roc/summary.json`: per-λ AUC + best threshold/accuracy and accuracy at `threshold=0.12`
- `evidence/bbob_noisy_probe_budget_roc/auc_vs_lam.png`: AUC and accuracy vs probe budget (λ)
- `evidence/bbob_noisy_probe_budget_roc/roc_curves.png`: ROC curves per λ

## Reading the results

For a compact summary (AUC, best-threshold accuracy, and accuracy at the default threshold `t=0.12` used in this repository),
use `evidence/bbob_noisy_probe_budget_roc/summary.json`.

Note: this evidence package uses instances `1–5` as a fast sweep to quantify the budget–reliability curve.
You can reproduce the same analysis on instances `1–15` by re-running the script with `--instances 1-15`.
