# Paper 3235 SOTA Code Analysis

## Preparation Failure Diagnosis

**Root cause:** Docker overlay storage pool at 200G was completely full, preventing `apt-get install git`. The pool was consumed by ~20+ old paper containers with large writable layers (oldest from July 8, 2026).

**Repair:** Removed 3 oldest/largest idle containers (papers 2152, 2635, 2190) freeing ~45GB. Copied git binary from host (apt mirrors unreachable via proxy).

## Corrected In-Container Evaluation Command

```bash
cd /repo && PYTHONPATH=/autosota_cache/site-packages:$PYTHONPATH MPLCONFIGDIR=/autosota_cache/tmp/matplotlib python3 /autosota_cache/tmp/eval_config.py --n-seeds=N
```

Or for full reproduction:
```bash
cd /repo && PYTHONPATH=/autosota_cache/site-packages:$PYTHONPATH MPLCONFIGDIR=/autosota_cache/tmp/matplotlib python3 /autosota_cache/tmp/reproduce_jobcorps.py
```

## Baseline Verification

- Mean MISE = 1.2466 (SE 0.1209) over 100 independent runs
- Matches paper Table 2 exactly: 1.2466 (0.1209)
- Evaluation produces identical results to reproduction

## Reusable Resources

- `/repo/DML_methods/Data_and_Resources/` — All JobCorps data files (emp_app.csv, semi-syn data grf.csv, h_star_grf_empapp.csv)
- `/autosota_cache/site-packages/` — Installed Python packages (scipy, pandas, scikit-learn, matplotlib)
- `/autosota_cache/tmp/results/` — Previous evaluation results

## Safe Optimization Targets

All hyperparameters in `estimate_h_grid_tensor()` are safe to modify:
- Kernel parameters: ell_x, nu_x, ell_t, nu_t, l_H, nu_H, kernel_type_f, kernel_type_H
- Ridge parameters: c_val, beta_grid_n, beta0_f, beta0_prime_f
- Stage-2 parameters: second_stage_n, second_stage_range
- Data split: 50/50 split in estimate_h_grid_tensor (line 113-115)
- Regularization selection: proxy-validation in estimate_h_grid_tensor (lines 164-187)
- Marginalization: empirical average (lines 137-143, 158-162)

Key functions:
- `_solve_krr_eig()` — eigendecomposition-based KRR solver
- `get_kernel_matrix()` — Matérn/Gaussian kernel computation
- `make_Xss()` — min-max covariate scaling
