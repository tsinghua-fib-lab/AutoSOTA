# Paper 5494 — SOTA Preparation Repair

## Original Failure

The orchestrator preparation script failed because:
1. `git` binary was not installed in the `autosota/paper-5494:reproduced` Docker image.
2. `apt-get install git` failed because the container has no outbound network access to `archive.ubuntu.com` or `security.ubuntu.com`.
3. Both reusable container reuse and fresh container attempts hit the same issue.

## Repair

- Copied `/usr/bin/git` from the host to the container at `/usr/bin/git`. The host git (2.25.1) has only standard library dependencies (libpcre2, libz, libpthread, libc) all available in the container.
- Initialized a git repo in `/repo` with baseline commit and `_baseline` tag.
- Created `/tools/record_score.sh` (copied from host script).
- Ensured `/autosota_artifacts/paper-5494/sota/` directory exists.

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 eval.py
```

This runs `aggregated_weights_power` for b=0.0 (Type I Error) and b=0.2 (Power) with:
- n=500, theta=2, K=4, transform=logquad, d=10, R_eval=500, alpha=0.05, seed=123

## Baseline Confirmation

| Metric | Reproduced | Manifest | Paper |
|--------|-----------|----------|-------|
| Type I Error | 0.060 | 0.06 | 0.056 |
| Power | 0.982 | 0.982 | 0.981 |

Reproduced values match the manifest exactly. Differences from paper are within MC noise (R=500, SE~0.01 for Type I, ~0.006 for Power).

## Safe Optimization Targets

The `aggregated_weights_power` function accepts these tunable parameters:
- `K` (int): binary expansion depth, default 4. Paper tests 3-6.
- `theta` (float): Clayton copula parameter, default 2.
- `R_eval` (int): Monte Carlo replications, default 500.
- `seed` (int): random seed, default 123.
- `unbiased_plugin` (bool): use unbiased variance, default True.

The function also has internal hardcoded levers:
- 10-fold SNR voting (in `_ten_folds_indices` and `aggregated_weights_power`)
- Binary hard voting between identity and J weights
- Only two candidate weight matrices (identity, J)

The `eval.py` can be modified to accept command-line arguments for parameter sweeps.
The `cobet/wa_dcobet.py` can be modified to change the voting mechanism, weight blending scheme, and candidate weight pool.

## No `/paper_data` Mount

No pre-downloaded data mount. All data is generated in-memory via Clayton copula + LogQuad transform. Pure CPU computation, no GPU needed.
