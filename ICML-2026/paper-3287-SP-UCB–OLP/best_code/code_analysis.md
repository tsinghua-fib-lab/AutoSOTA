# Code Analysis: Paper 3287 - SOTA Preparation Repair

## Original Preparation Failure

The preparation failed because:
1. The container overlay FS was 100% full (200G/200G used), causing apt-get update to fail with "No space left on device"
2. git was not pre-installed in the environment image and could not be installed via apt
3. The orchestrator preparation script tried to install git via apt which failed, then could not create the git baseline

## Repair

1. Cleaned /var/lib/apt/lists/* and /var/cache/apt/archives/* in the container
2. Re-ran apt-get update which succeeded
3. Installed git via apt-get install -y git (succeeded after cleanup)
4. Initialized git repo at /repo and created baseline commit with _baseline tag
5. Copied /tools/record_score.sh into container

## Verified In-Container Evaluation Command

```
cd /repo
PYTHONPATH=/autosota_cache/tmp/python-packages:$PYTHONPATH MPLCONFIGDIR=/autosota_cache/tmp python3 /autosota_cache/tmp/eval_paper.py
```

Options:
- --seeds "0,1,2" to limit seeds for faster testing
- --data /datasets/alibaba/batch_task.csv (default)
- --results /autosota_cache/tmp/results_eval (default)

## Baseline Evidence (50 seeds, full reproduction)

```
SP-UCB-OLP (alpha=0.01): 96.67% +/- 3.11%, median=97.33%
Greedy (alpha=0):      83.97% +/- 12.44%
SP-UCB-OLP range: [75.89%, 97.82%]
Oracle: 99.97%, Random: 68.01%
```

This exactly matches the manifest baseline: competitive_ratio: 96.67.

## Key Source Files

- /repo/run_experiment.py -- Experiment runner with run_experiments() function
- /repo/algorithms/sp_ucb_olp.py -- SP-UCB-OLP algorithm implementation
- /repo/algorithms/base.py -- Base algorithm class with budget tracking (b_safe)
- /repo/data/alibaba_loader.py -- Alibaba trace data loader with noise injection
- /autosota_cache/tmp/eval_paper.py -- Evaluation harness

## Safe Optimization Targets

### Parameters (no code change needed)
- alpha (exploration rate, default 0.01): in eval_paper.py PAPER_CONFIG
- solve_frequency (default 10): in sp_ucb_olp.py constructor
- n_restarts (default 2): in sp_ucb_olp.py constructor
- noise_sigma (default 0.1): in alibaba_loader.py

### Code Modification Points
- algorithms/sp_ucb_olp.py:108-114 -- _compute_confidence_radii (UCB bonus)
- algorithms/sp_ucb_olp.py:148-159 -- _solve_saddle_point_scipy objective
- algorithms/sp_ucb_olp.py:176-181 -- scipy minimize call (maxiter, ftol)
- algorithms/sp_ucb_olp.py:241 -- solve frequency trigger
- algorithms/sp_ucb_olp.py:251 -- random choice for admission
- data/alibaba_loader.py:151 -- noise injection in reward
- algorithms/base.py:87 -- b_safe computation

### Reusable Artifacts
- Alibaba Cluster Trace v2018 at /datasets/alibaba/batch_task.csv (781MB, 5000 arrivals extracted)
- Python packages at /autosota_cache/tmp/python-packages/ (scipy, pandas, matplotlib)
- No /paper_data mount; all data from /datasets/

## Optimization Constraints
- Must use in-container eval command; no host-side modifications
- Metric: competitive_ratio (higher is better, 0-100%)
- Evaluation is CPU-only (no GPU needed)
- Each 50-seed eval takes ~21 minutes
- Quick testing with 10-15 seeds takes ~4-6 minutes
