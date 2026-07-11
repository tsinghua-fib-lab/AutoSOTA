# Code Analysis — Paper 3928 (LiMVAM)

## Evaluation Path

- **Command**: python3 experiments_meg/eval_pearson.py
- **Script**: experiments_meg/eval_pearson.py
- **What it loads**: B_total.npy from a results directory
- **Default dir**: experiments_meg/4_results/aparc_sub_30_random_subjects_50_runs_pairwise_limvam
- **Metric**: Average pairwise Pearson correlation between element-wise median B matrices across runs
- **Output**: Parse final line {pearson_correlation: X.XXXX, method: PairwiseLiMVAM}

## Data Availability

| Data | Available | Location |
|------|-----------|----------|
| MEG envelope data (Cam-CAN) | NO | Not in container |
| Pre-saved MEG results (B, T, P) | YES | experiments_meg/4_results/ |
| fMRI data (9 subjects, 9 regions) | YES | experiments_fmri/data/ |
| Synthetic experiment code | YES | experiments_synthetic/runs/ |

**Critical**: Cannot re-run run_camcan_multiple_times.py because raw MEG data (X.npz) is unavailable.
Optimization uses post-processing of pre-saved results and proxy evaluation on fMRI/synthetic data.

## Pre-saved Results Structure

- B_total.npy: shape (50, 30, 10, 10) — 50 runs x 30 subjects x 10x10 DAG matrices
- T_total.npy: shape (50, 30, 10, 10) — strictly lower triangular matrices per subject
- P_total.npy: shape (50, 10, 10) — permutation matrices per run

**Key finding**: All 50 runs have UNIQUE P matrices. Ordering instability is the dominant bottleneck.

## Optimization Strategy

### Primary: Post-process existing MEG results
- Extract consensus ordering from 50 P matrices (Borda count)
- Transform B matrices to use consensus ordering
- Apply robust aggregation (trimmed mean)
- Evaluate with same eval_pearson.py

### Secondary: Algorithm code changes + proxy validation
- Modify limvam/pairwise_limvam.py and limvam/utils.py
- Create fMRI-based stability experiment
- Validate improvements on other datasets

## Safe Modification Targets
1. limvam/utils.py:155-157 — Sigma_j rank check to SVD pseudoinverse
2. limvam/utils.py:161-162 — FGLS to ridge-regularized FGLS
3. limvam/pairwise_limvam.py:15-17 — Empirical covariance to shrinkage estimator
4. Post-processing: consensus ordering, trimmed mean aggregation

## Risky Files (DO NOT MODIFY)
- experiments_meg/eval_pearson.py
- experiments_meg/4_results/ (reference)
- /tools/record_score.sh
- /autosota_artifacts/
