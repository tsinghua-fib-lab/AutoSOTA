# Code Analysis: Paper 3470 SOTA Preparation Repair

## Preparation Failure

The SOTA preparation failed because:
1. **git not installed**: The container image `autosota/paper-3470:reproduced` does not include `git`.
2. **apt-get proxy failure**: The container was configured with proxy `http://172.17.0.1:17890` which returned 502 Bad Gateway errors, preventing `apt-get install git`.
3. **Fix**: Ran `apt-get install git` without proxy environment variables (direct connection), which succeeded.

## Corrected In-Container Evaluation Command

```bash
cd /repo
python3 reproduce_final.py --n 5 --x-n 1 --T 2 --N 5000 --graph-dense 2.0 --noise-type gauss --seeds 0 1 2 3 4 --num-steps 50000 --num-initializations 3 --save-dir /repo/outputs/sota_baseline
```

Note: Changed `--save-dir` from `/repo/outputs/reproduction` to avoid overwriting reproduction results.

## Baseline Metrics

| Metric | Mean | Std |
|--------|------|-----|
| d-MCC  | 0.99897 | 0.00023 |
| MCC    | 0.94007 | 0.00891 |
| Amari  | 0.11102 | 0.01992 |

These match the manifest baseline within numerical noise.

## Key Source Files

- `reproduce_final.py`: Standalone evaluation script with embedded optimization. This is the primary target for optimization.
- `main_numerical.py`: Standalone training script (not used by reproduce_final.py).
- `evaluation.py`: Contains `compute_mcc_g()` (d-MCC), `compute_mcc()` (MCC), and `amari_distance_rect()` (Amari).
- `numerical_data_generator.py`: Synthetic data generation.

## Optimization Targets

The optimization is in `run_opt()` within `reproduce_final.py`:
- Lines 19-54: `run_opt()` function — main optimization loop
- Parameters: `A_hat` (mixing matrix), `D_flat` (domain scaling), `sigma_vec` (noise std), `B_free` (causal graph)
- Loss: Frobenius norm of covariance difference
- Optimizer: Adam with default betas

## Reusable Resources

- No external datasets needed (synthetic data)
- No pre-downloaded models or checkpoints
- Cache paths: `/autosota_cache`, `/datasets`, `/models`

## Safe Optimization Targets

1. **A_hat initialization** (line 22): Replace `torch.rand()` with SVD-based init
2. **Loss function** (lines 35-39): Add L1 sparsity on B, replace inverse with solve
3. **Training loop** (lines 28-46): Add gradient clipping, NaN detection
4. **Optimizer parameters** (line 27): Tune Adam beta2
5. **Training schedule**: Staged optimization (freeze B first)
6. **Multiple initializations**: Increase from 3 to higher values
