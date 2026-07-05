# Code Analysis: Paper 4915 SOTA Preparation Repair

## Preparation Failure Diagnosis

The SOTA preparation failed because:
1. The original reusable container (`autosota_repro_paper_4915`) had apt 502 errors (proxy misconfiguration), so `git` could not be installed.
2. The first Docker run for the SOTA container failed due to host networking being rejected by the Docker authorization plugin.
3. The second Docker run succeeded without `--network host`, but the proxy was set to `http://172.17.0.1:17890` inside the container, which caused apt to fail again with 502 errors.

### Repair Actions
- Started container `autosota_sota_paper_4915` from `autosota/paper-4915:reproduced` without `--network host`.
- Unset proxy environment variables (`HTTP_PROXY`, `HTTPS_PROXY`, etc.) inside the container.
- Successfully installed `git` via `apt-get`.
- Initialized git repo at `/repo`, created baseline commit and `_baseline` tag.
- Copied `/tools/record_score.sh` from host.
- Created `/autosota_artifacts/paper-4915/sota/` directory.

## Corrected Evaluation Command

```bash
cd /repo
python3 eval_reproduction.py
```

All data is already at `/repo/data/paper/` (the 4 JSON files from `/paper_data/paper/`).

## Baseline Evidence

The corrected evaluation ran successfully and reproduced the manifest metrics exactly:
- **Relative CI Half-Width (MBAR, min at peak)**: 0.003863 (manifest: 0.003863) ✓
- **Minimum Accessible Probability Density (at ARI≈14)**: 2.194954e-07 (manifest: 2.194954e-07) ✓
- **Direct Relative CI Half-Width**: -0.011471 (negative due to Wilson interval behavior at peak)

Runtime: ~23 minutes (20 min bootstrap, 3 min MBAR+preprocessing).

## Reusable /paper_data Resources

- `/paper_data/paper/`: The 4 JSON data files (ari_trajectories.json, ari_unbiased_samples.json, log_probs_trajectories.json, log_probs_unbiased_samples.json) — already copied to `/repo/data/paper/`.
- `/paper_data/TinyStories-8M/`: Full model weights (pytorch_model.bin, config.json, tokenizer files) — available if TPS re-sampling is needed.
- `/paper_data/gpt-neo-125M-tokenizer/`: Tokenizer files (merges.txt, tokenizer.json, tokenizer_config.json, vocab.json) — available for tokenization.

## Optimization Targets

The evaluation pipeline processes pre-computed TPS trajectory data through:
1. Burn-in removal (10%)
2. Gelman-Rubin convergence filtering (cutoff 1.1)
3. MBAR free energy computation
4. Importance weight computation
5. Histogram binning (80 bins, [-8, 15])
6. Bootstrap CI (100 iterations, 32 parallel workers)
7. Metric extraction

### Safe Optimization Levers
- Convergence diagnostic (split-R-hat instead of standard GR)
- Importance weight stabilization (winsorizing, numerical stability)
- Data quality (random subsampling, full unbiased data usage)
- Bootstrap quality (ESS filtering, block bootstrap)
- Binning strategy (Bayesian blocks adaptive binning)
- Parameter tuning (burnin, GR cutoff, bin count/range)
- Overlap-based state rejection

### Out of Scope
- Re-running TPS sampling (requires model training/inference pipeline not in container)
- Changing metric definitions or benchmark protocol
- Modifying the underlying data

## Git State
- Baseline commit: c052ef9f (tag: _baseline)
- Best commit: c1fdc1b4 (tag: _best, iter 0 baseline)
