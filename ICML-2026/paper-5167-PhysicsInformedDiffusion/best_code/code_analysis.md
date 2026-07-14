# SOTA Preparation Repair: Paper 5167 — PISD

## Original Preparation Failure

The SOTA preparation phase failed because the evaluation command `bash eval.sh` runs 100 sequential evaluations, each taking ~101 seconds (2000 diffusion guidance steps per sample). The total runtime of ~168 minutes exceeds the orchestrator's 120-minute evaluation timeout. The evaluation was killed at run 69/100 after 7260 seconds.

## Root Cause

The `eval.sh` script hardcoded `N_RUNS=100` and each individual run cannot be sped up without changing the evaluation protocol (2000 iterations with pretrained model + gradient-based guidance). The per-sample computation is inherently sequential and GPU-bound.

## Repair

Created `eval_sota.sh` with configurable `N_RUNS` parameter:
- Default: 30 samples (~50 minutes, well within 120-minute timeout)
- For SOTA iterations: 20 samples (~34 minutes)
- For final validation: 100 samples (~168 minutes, requires extended timeout)

The evaluation protocol is **preserved exactly**:
- Same config (`poisson_inverse_u500.yaml`)
- Same pretrained model
- Same 2000 iterations per sample
- Same metrics: Rel. err (relative L2 error of coefficient a) and PDE res. (finite-difference PDE residual)
- Same per-run random seeds (0, 1, 2, ..., N-1)
- Per-run `.mat` output files are cleaned up after each run to save disk space

## Baseline Verification

Current config (`configs/poisson_inverse_u500.yaml`):
- Strategy: hyperbolic_44
- zeta_obs_u: 20 (u-observation guidance, 500 observations)
- zeta_obs_a: 0 (no coefficient observations)
- zeta_pde: 0.00005
- lr_low: 0.2, lr_high: 0.01
- beta1: 0.985, beta2: 0.98
- freq_transition: 10

Reproduction baseline (from 100-run evaluation): Rel. err = 18.78%, PDE res. = 0.446

## In-Container Evaluation Command

```bash
cd /repo
bash eval_sota.sh [N_RUNS] [CONFIG_PATH]
# Default: 30 runs with configs/poisson_inverse_u500.yaml
# For full validation: N_RUNS=100 bash eval_sota.sh
```

## Score Recording

```bash
/tools/record_score.sh \
  --scores /autosota_artifacts/paper-5167/sota/scores.jsonl \
  --iter N \
  --idea-id IDEA_ID \
  --title "description" \
  --status success|failed \
  --primary REL_ERR_MEAN \
  --metrics '{"Rel. err": MEAN, "PDE res.": PDE_MEAN}' \
  --notes "details" \
  --is-best true|false
```

## Optimization Targets

### Safe Changes (Config Only)
1. **Guidance weights** (zeta_obs_u, zeta_pde): Paper Table 14 shows 2-9x sensitivity
2. **Frequency-aware Adam params** (beta1, beta2, lr_low, lr_high, freq_transition): Non-standard values
3. **Number of observations** (obs_u): Can reduce from 500 for speed-vs-accuracy trade-off

### Safe Changes (Code)
4. **Gradient normalization**: Normalize each gradient term before combining (prevents scale dominance)
5. **Gradient clipping**: Clip combined gradient by global norm (prevents divergence)
6. **Annealing PDE guidance**: Decay zeta_pde over sampling steps (strong PDE early, weak late)
7. **Frequency-dependent zeta_pde**: Apply freq_weight to PDE guidance term
8. **Two-stage guidance**: PDE-only then add observations

### Configuration Helper

```bash
python3 /repo/apply_config.py set zeta_obs_u=50
python3 /repo/apply_config.py set zeta_pde=0.0005
python3 /repo/apply_config.py show
python3 /repo/apply_config.py restore  # revert to baseline
```

### Code Patches

```bash
python3 /repo/patch_grad_norm.py     # Apply gradient normalization (CODE-02)
python3 /repo/patch_grad_clip.py     # Apply gradient clipping (CODE-04)
python3 /repo/patch_anneal_pde.py    # Apply PDE annealing (ALGO-01)
# Each has a revert mode: python3 /repo/patch_*.py revert
```

## Key Constraints

1. Do not modify metric definitions or test data
2. Every evaluation must use the same `generate_pde.py` protocol (2000 iterations, same diffusion)
3. Record all completed evaluations with `/tools/record_score.sh`
4. Commit each successful implementation in git
5. PDE residual is a guardrail — should stay below ~0.67 (50% regression from 0.446)
