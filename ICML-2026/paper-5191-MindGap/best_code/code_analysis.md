# Code Analysis for Paper 5191 SOTA Optimization

## Evaluation Path
- **Main script:** `/repo/eval_ecoli_dp_pcd.py`
- **Eval command:** `python3 eval_ecoli_dp_pcd.py`
- **Timeout:** 10 minutes
- **Output:** stdout (parsable metrics) and `/repo/ml_results.json`

## Key Files
| File | Purpose | Safe to modify? |
|------|---------|-----------------|
| `eval_ecoli_dp_pcd.py` | Main eval: loads ecoli, runs DP-PCD, reports metrics | YES - parameters, optimizer logic |
| `calibrate_for_ml.jl` | Julia calibration for DP mechanisms | YES - calibration approach |
| `calibrate_ml.jl` | Alternative calibration | YES |
| `multi_gaussian_high_epsilon.jl` | M-G mechanism calibration | YES - with care |
| `quasi_gaussian_default.jl` | Q-G mechanism calibration | YES - with care |
| `analytic_gaussian.jl` | A-G mechanism calibration | NO - standard reference |
| `mechanism_calibration.json` | Pre-computed calibration values | YES - regenerate |
| `ml_results.json` | Output results | NO - eval script output |

## Metric Parser
- stdout: parse lines matching `AG`, `MG`, `QG`, `PCD` for `in_sample_error_mean` values
- Also read from `/repo/ml_results.json` for structured output
- Format: `MG          14.85% ±6.04%      15.42% ±7.40%`

## Config/Parameters
- `T=100`: iterations
- `P=ceil(d/4)=2`: coordinates updated per iteration
- `LAM=1e-8`: l1 regularization
- `NOISE_SCALE=0.0385`: effective per-coordinate sensitivity scaling
- `N_SPLITS=500`: random train/test splits
- `SEED=42`: random seed
- `CALIB`: Julia-calibrated sigma values

## Step Size
- `L_j = 0.25 * mean(X[:,j]^2)` — per-coordinate Lipschitz constants
- `step_size = mean(1/L_j)` — scalar step size (should be per-coordinate for theoretical correctness)

## Known Levers
1. `NOISE_SCALE`: per-coordinate sensitivity scaling (currently 0.0385)
2. `step_size`: from Lipschitz constant (scalar vs per-coordinate)
3. `T`: iterations (currently 100)
4. `P`: proximal updates per iteration (currently 2)
5. `K`: M-G mixture components (currently 10)
6. `LAM`: l1 regularization (currently 1e-8)
7. `N_SPLITS`: evaluation splits (currently 500)
8. Julia calibration: `eps_per`, `delta_per`, noise sigmas

## Safe Modification Targets
1. `dp_pcd()` function: coordinate selection, noise application, step sizes
2. `main()` function: parameter configuration, noise sampler construction
3. Julia calibration scripts: privacy accounting approach
4. NEW: diagnostic functions, per-coordinate tracking

## Risky Files (do NOT modify)
- Dataset loading: `load_ecoli()`, `preprocess()` — evaluation protocol
- Metric computation: `accuracy_score` usage — must stay unchanged
- `stable_sigmoid()`, `soft_threshold()` — core math functions (can improve numerics)
- `analytic_gaussian.jl` — standard reference implementation
