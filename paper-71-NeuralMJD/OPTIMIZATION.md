# Optimization Report: Neural MJD

**Paper**: Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction  
**Paper ID**: 71  
**Repository folder**: `paper-71-NeuralMJD`  
**Source**: AutoSota closed-loop scores (`optimizer/papers/paper-3356/runs/run_20260321_040008/results/scores.jsonl`); no standalone `final_report.md` was emitted for this run.  
**Synced to AutoSota_list**: 2026-03-22  

---

## Summary

- **Primary metric**: test **MAE** on S&P 500 config (`config/sp500/mjd/neural_mjd.yaml`), **lower is better**.  
- **Paper-reported baseline** (config): MAE **17.411**.  
- **Our run baseline** (iter 0): MAE **17.127** (already better than paper table).  
- **Best confirmed** (iter 11): MAE **14.084** at best epoch during search.  
- **Final evaluation** (iter `final`): MAE **14.096** — reported as the locked-in best configuration.  
- **Improvement vs paper baseline 17.411**: **≈ −19.0%** MAE.

## Baseline vs. best (primary line)

| Stage | MAE | Notes |
|-------|-----|--------|
| Paper baseline (config) | 17.411 | Reference from `config.yaml` |
| Iter 0 (repro baseline) | 17.127 | Best epoch 44 |
| **Best (iter 11)** | **14.084** | `n_runs=300`, `w_cond_mean_loss=2.0`, antithetic variates |
| Final eval (iter final) | 14.096 | Confirms config at epoch 39 |

## Key changes that moved the metric

1. **`n_runs` scaling** (Monte Carlo / sampling runs for jump-diffusion paths): 10 → 30 → 50 → 100 → 200 → **300** gave the largest drops; **500** overshot and hurt within the fixed epoch budget (rolled back).  
2. **Antithetic variates** in MJD sampling (paired ± noise for Brownian and jump Gaussians) reduced variance of gradient targets and beat `n_runs` scaling alone at equal cost order.  
3. **`w_cond_mean_loss=2.0`**: Up-weighting the conditional mean loss improved point forecasts vs the default 1.0; 3.0 was too aggressive and regressed.

## What did not help

- **EMA 0.999** with large `n_runs`: slowed effective convergence within 51 epochs.  
- **Finer SDE integration** (`steps_per_unit_time` 5→10) and **richer Poisson tail** (`max_n` 5→10): small NLL gains did not translate to better MAE.

## Artifacts in this mirror

- Large `data/` and `output/` trees were **stripped** from this listing to keep the monorepo lightweight; re-fetch S&P 500 CSVs and retrain per upstream `README` / Docker notes.

## Raw logs

- See `optimizer/papers/paper-3356/runs/run_20260321_040008/` for full `scores.jsonl`, idea library, and code analysis memos.
