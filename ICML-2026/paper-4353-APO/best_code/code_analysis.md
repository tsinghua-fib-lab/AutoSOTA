# Code Analysis for Paper 4353 — GVALID SOTA Optimization

## Evaluation Path
- **Entry**: `python3 eval.py --n_seeds 20 --gpu_id 0`
- **Flow**: eval.py → `src/experiment.run_single_seed()` → loops over seeds
- **Per seed**: Creates HardNonLinear8D dataset → GPModel → GVALIDSampler → LHS init (35 pts) → 63 AL rounds (batch 5) → N=350 total
- **Metrics**: `src/evaluator.evaluate_metrics()` at each round, returns `E2_error`, `policy_suboptimality`, `dose_error`
- **Output format**: `policy_suboptimality: mean=X.XXXXXX, std=Y.YYYYYY` after `FINAL RESULT`

## Train/Inference Path
- **GP training**: `GPModel._fit_model()` → 90-step Adam with jitter schedule `[1e-6, ..., 5e-3]`
- **GP inference**: `GPModel.predict()` (batch mode), `GPModel.__call__()` (torch mode)
- **t* optimization**: `src/utils.optimize_t_for_x_batch_torch()` — GPU grid search over t ∈ [0,1]
- **Normalization**: Supported in GPModel but disabled (`normalize_x=False, normalize_y=False`)

## Key Configuration Points
- `experiment.py:49`: `GPModel(dim_x=..., normalize_x=False, normalize_y=False)` ← SAFE
- `experiment.py:198-249`: Active learning loop, passes `validation_context` ← SAFE
- `gp_model.py:47-61`: `reset_hyperparameters()` — fixed lengthscale init ← SAFE
- `gp_model.py:96-99`: `training_steps=90, training_lr=0.06` ← SAFE
- `samplers.py:864-882`: GVALIDSampler.__init__ — `target_sample_size=32, cand_t_grid_size=10` ← SAFE
- `samplers.py:988`: beta=0.0 in t* estimation (already greedy) ← SAFE
- `samplers.py:1097-1140`: Greedy batch selection with Schur complement ← SAFE
- `samplers.py:1116-1138`: Schur update loop ← SAFE (but numerically sensitive)
- `eval.py:19`: `--gpu_id` flag, `CUDA_VISIBLE_DEVICES` setting

## Metric Parser
- `evaluator.py`: `E2_error = mean((f_pred_at_t_star_true - f_at_t_star_true)^2)`
- `evaluator.py`: `policy_suboptimality = v_star - v_hat` where v = mean(f(x, t*(x)))
- `evaluator.py`: `dose_error = mean(|t_pred - t_true|)`

## Safe Modification Targets
1. `gp_model.py` — enable normalization, change lengthscale init, add deep kernel, change kernel structure
2. `samplers.py` GVALIDSampler — batch selection logic, target params, Schur update fixes
3. `samplers.py` — add new sampler classes
4. `experiment.py` — GPModel constructor args, beta scheduling
5. `eval.py` — only add new CLI args (do NOT change metric computation)

## Risky / Do-Not-Touch Files
- `evaluator.py` — metric definitions (DO NOT CHANGE)
- `datasets.py` — dataset generation (DO NOT CHANGE)
- `eval.py` lines 60-85 — final result parsing (DO NOT CHANGE)

## Rollback Points
- `git tag _baseline` points to commit `34788f6`
- `git tag _best` currently at `aba3b4e` (iter-0 baseline)
- Each iteration commits before changes: `git commit -m "pre-iter-N snapshot"`
