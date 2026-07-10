# COLSA Code Analysis for SOTA Optimization (Paper 3554)

## Evaluation Path

- **Entry point**: `evaluate.R` — self-contained R script
- **Command**: `Rscript evaluate.R`
- **Timeout**: 30 minutes per evaluation
- **No command-line args needed** (defaults to simulation mode; `--real-tcga PATH` exists but requires real data)

## Evaluation Flow

1. Generates simulation data matching TCGA structure (n=7315, K=18 batches, d=23 genes)
2. Fits Oracle (full-data Cox model via `survival::coxph`)
3. Fits COLSA sequentially across 18 batches
4. Computes 3 metrics by comparing COLSA vs Oracle estimates

## Metric Parsing

Extracted from stdout lines:
- `Correctly Recovered Inference: N/23` → primary metric
- `Pearson r of Z-statistics: X.XXXX`
- `Mean Absolute Log-HR Difference: X.XXXXX`

Also saved to `/repo/eval_results.rds` as an R list.

## Core Source Files

| File | Purpose | Safe to Modify? |
|------|---------|-----------------|
| `evaluate.R` | Evaluation script | Yes — hyperparameters, basis selection, boundary calc |
| `R/colsa.R` | `colsa()`, `update.colsa()`, `vcov.colsa()`, `summary.colsa()`, `AIC.colsa()`, `BIC.colsa()` | Yes — initialization, regularization, numerical stability |
| `R/utils.R` | `objective()`, `int_basehaz()`, `prox_forward()`, `prox_reverse()` | Yes — quadrature nodes, Hessian conditioning |
| `R/sim.R` | Simulation utilities | Maybe — not used in evaluate.R |
| `R/COLSA-package.R` | Package metadata | No |

## Key Optimization Targets

### Parameters in evaluate.R
- `SEED` (42): affects data generation
- `PRE_ESTIMATION_FACTOR` (2): scaling for pre-estimation basis
- `NU` (0.2): growth exponent for auto-basis
- `SIG_THRESH` (1.96): significance threshold (DO NOT CHANGE — metric definition)
- `alpha_best`: derived from AIC on batch 1
- `boundary`: `c(0, max(df$time))`
- AIC search range: `1:5`

### Parameters in colsa.R
- `colsa(..., scale = 2.0)`: pre-estimation scaling
- `colsa(..., init = "zero")`: initialization method ("zero" or "flexsurv")
- `update.colsa(..., alpha = 1.0)`: basis growth rate
- `update.colsa(..., nu = 0.2)`: growth exponent
- `vcov.colsa()`: Cholesky fallback uses `solve()` without regularization

### Parameters in utils.R
- `int_basehaz(..., n_nodes = 5)`: Gaussian quadrature nodes
- `objective()`: Hessian computation

## Red-Line Boundaries

- **Do NOT change**: `SIG_THRESH`, `TARGET_GENES`, `ORACLE_LOGHR`, `N_TOTAL`, `K_BATCHES`, `N_GENES`
- **Do NOT change**: metric computation formulas, stdout format, `eval_results.rds` structure
- **Do NOT change**: data generation logic (except `SEED` for multi-seed testing)

## Paper Data

- No pre-downloaded paper data mount exists
- Simulation data is generated deterministically from `ORACLE_LOGHR` and `SEED`
- Real TCGA data is not available (network proxy blocks)

## Baseline Metrics (iteration 0)

```
Correctly Recovered Inference: 23/23
Pearson r of Z-statistics: 0.9999
Mean Absolute Log-HR Difference: 0.00042
```

Primary metric is at theoretical maximum (23/23). Optimization focuses on:
1. Verifying robustness across seeds
2. Improving guardrail metrics (Pearson r, MAD) through numerical precision
3. Better basis selection and initialization
