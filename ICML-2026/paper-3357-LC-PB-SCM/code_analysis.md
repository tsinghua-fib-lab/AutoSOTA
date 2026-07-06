# SOTA Preparation Repair Analysis — Paper 3357

## Original Preparation Failure

The git binary was missing from the container at the time of the preparation script run. The orchestrator tried `apt-get install git` which failed due to 502 Bad Gateway from apt repositories through the proxy (172.17.0.1:17890). A second attempt with a freshly started container also failed.

### Root Cause
The `/usr/bin/git` binary was present in the container image but was missing runtime dependencies (libcurl3-gnutls, perl, liberror-perl, etc.). The apt-get install attempt to fix this failed due to proxy issues. Some dependencies were manually downloaded from mirrors.edge.kernel.org and installed via `dpkg -i`, which resolved the git functionality.

### Repair
- Manually downloaded missing git dependencies from kernel.org mirror
- Verified `/usr/bin/git` works: `git version 2.25.1`
- Initialized git repo, created baseline commit + `_baseline` tag
- Installed `/tools/record_score.sh`
- Created `/autosota_artifacts/paper-3357/sota/` directory

## Corrected In-Container Evaluation Command

```bash
cd /repo && /usr/bin/git config user.name optimizer && /usr/bin/git config user.email opt@local && python3 eval.py
```

The evaluation runs entirely on synthetic data (no external datasets needed), generating 50 PB-SCM parameter configurations with:
- `sample_size=10000`, `bootstrap_round=200`, `p_value=0.05`, `n_jobs=4`, `seed=42`

## Baseline Verification

- **F1 Score:** 0.7513 ± 0.3103
- **Manifest baseline:** 0.7513
- **Match:** Exact reproduction
- **Paper reported:** 0.72 ± 0.19
- **Rubric bounds:** [0.53, 0.91]

## Reusable /paper_data Resources

The `/paper_data` mount contains football events data (CSV files) for the real-world experiment. The Case 2 evaluation uses synthetic data generated from PB-SCM, so no `/paper_data` resources are needed.

## Safe Optimization Targets

### Primary Levers (from manifest + code analysis)

1. **bootstrap_round** (200 → 500+): More bootstrap samples → higher statistical power for hypothesis testing. Runtime scales linearly. Most promising.
2. **p_value** (0.05 → 0.10): Relax significance threshold for one-sided test (p-quantile > 0). More edges pass but may increase false positives.
3. **skeleton_threshold / r2_threshold** (1e-4): Lowering may increase sensitivity in edge detection. Currently not wired through eval.py to pgf_confounder_partial.
4. **Alternative hypothesis test**: Switch from `hypothesis_test_significance_one_side` to `hypothesis_test` with interval-based testing in `process_term()`.
5. **n_jobs** (4 → higher): More parallelism, but doesn't change metric — purely runtime optimization.

### Algorithm Structure

```
eval.py
  └─ pgf_confounder_partial(data, bootstrap_round, p_value, n_jobs, seed)
       ├─ learn_skeleton()        → Bootstrap tests for pairwise interactions
       ├─ learn_r2()              → Direction tests for skeleton edges  
       └─ interactionsBySkeleton() → Higher-order interaction tests
  └─ compute_f1(learned_mag, ground_truth_mag) → Mark-level F1 with circle compatibility
```

### Key Code Points

- `process_term()` in `pgf_confounder_partial.py` line 44: Uses `hypothesis_test_significance_one_side`
- `hypothesis_test()` in `util.py` line 176: Alternative two-sided interval test (commented out)
- `contradictory_check()` in `pgf_confounder_partial.py` line 78: Resolves R2 conflicts with fixed z-values
- `eval.py` hardcodes all parameters; optimization requires modifying eval.py or creating variant scripts
