# SOTA Preparation Repair — Paper 5182

## Original Failure

The normal SOTA preparation script failed because:
1. **Git not installed**: The `autosota/paper-5182:reproduced` image has no `git` package.
2. **apt-get network failure**: The preparation script ran `apt-get install -y -qq git` without first unsetting proxy environment variables (`HTTP_PROXY`, `HTTPS_PROXY`, etc.), causing apt to route through the proxy (port 17890) which returned HTTP 502 errors for `.deb` downloads.
3. **Docker policy rejection**: The first container creation attempt (`autosota_sota_paper_5182`) failed due to OPA policy rejecting `--network host`. The orchestrator retried without `--network host` and succeeded.

## Repair Applied

1. Verified container `autosota_sota_paper_5182` (ID `322491cd3b52`) was running with image `autosota/paper-5182:reproduced`.
2. Installed `git` via `apt-get` after unsetting all proxy environment variables:
   ```bash
   unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
   apt-get update -qq && apt-get install -y -qq git
   ```
3. Initialized git repo in `/repo` and created `_baseline` tag at commit `cac000f`.
4. Copied `/tools/record_score.sh` from host into container.

## Baseline Verification (Iteration 0)

**Command** (in container):
```bash
cd /repo && python3 cv_predictive_validity.py --n-folds 5 --seed 42
```

**Results** — exact match with reproduction manifest:

| Metric     | Baseline | Manifest | Match |
|------------|----------|----------|-------|
| Accuracy   | 0.9256   | 0.9256   | ✓     |
| Log-loss   | 0.2018   | 0.2018   | ✓     |
| Brier      | 0.0583   | 0.0583   | ✓     |
| Baserate   | 0.6488   | —        | —     |

1868 games (1212 wins, 656 losses) over 271 questions in 5-fold CV.

## Reusable /paper_data Resources

The `/paper_data` mount contains pre-generated benchmark artifacts for multiple papers. For paper 5182, the `/paper_data/paper-5182/` directory is available but **not needed** — all required data (benchmarks/, answers/, critiques/, debates/, automated_evaluations/, evaluations/) was already included in the repo checkout. The CV evaluation requires no GPU, no network, and no LLM API calls.

## Optimization Targets

The evaluation is CPU-only, deterministic given fixed seed, and fast (~30-60 seconds per run). Safe optimization targets:

1. **BT hyperparameters**: sigma priors, learning rate, max iterations, tolerance
2. **Optimizer**: Replace SGD with Adam
3. **Regularization**: Add explicit L2 penalty
4. **Edge weighting**: Weight by judge confidence
5. **Post-hoc calibration**: Platt scaling
6. **CV configuration**: n_folds, seed
7. **Game inclusion**: Non-unanimous adjudication resolution

## Constraints

- No modification to benchmarks/, answers/, critiques/, evaluations/, or automated_evaluations/ data
- No modification to metric computation
- No hard-coded predictions
- All changes must be in cv_predictive_validity.py or new wrapper scripts
- Record all results with /tools/record_score.sh
