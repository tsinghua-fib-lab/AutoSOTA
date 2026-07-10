# Code Analysis — Paper 3290: Treatment Risk Control

## Evaluation Path
- Entry: `experiments/main.py` → `main()` 
- For STAR dataset with `--guarantee p`: 
  1. `run_fft()` → builds FFT policies over lambda grid
  2. `calibrate_fft_bound()` → selects best policy per constraint level using Hoeffding-Bentkus bound
  3. True evaluation on held-out `df_beta` using `weighter_beta.get_obj_and_constr()`
- Output: prints "Mean true obj:" (Population Risk) and "Mean true constr:" (Treatment Risk)
- Also saves CSV at `result/<exp_folder_name>/<timestamp>/<name>.csv`

## Metric Parser
- Parse stdout for "Mean true obj:" and "Mean true constr:" arrays
- For tau=0.35, read index 5 (6th value) of each array
- Constraint values default order: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

## Key Files
- `experiments/main.py` — evaluation entry point, argparse, main loop
- `method/fft.py` — `run_fft()`, `calibrate_fft_bound()`, `calibrate_fft_risk()`
- `method/create_fft_constraint.py` — `FftPolicyConstraint.find_best_loss()` with tree penalties
- `method/create_fft_base.py` — `FftPolicyBase` with tree search logic
- `method/weights_p_bound.py` — `Weights`, `WeightsDecision`, `upper_confidence_bound()` (Hoeffding-Bentkus)
- `method/weights_risk_control.py` — Conformal risk control variant
- `data/star_data.py` — STAR dataset loading, `get_p_a_x()` returns constant 0.5
- `data/data_loader.py` — Dataset factory
- `utils/save_utils.py` — Output path generation
- `utils/results.py` — `save_results()`, CSV output

## Safe Modification Targets (Sorted by Impact)
1. **`method/weights_p_bound.py`** `upper_confidence_bound()` — hardcoded `delta=0.1`, linear scan step=0.001
2. **`method/create_fft_constraint.py`** `find_best_loss()` — `0.02*n` penalty (line ~"loss_i + 0.02 * n"), lookahead penalty `0.001`
3. **`method/fft.py`** `run_fft()` — `np.linspace()` for lambda grid
4. **`method/weights_p_bound.py`** — add empirical Bernstein bound function
5. **`experiments/main.py`** — n_splits, n_bins, lookahead, data_split, n_mc parameters

## Baseline (Iteration 0)
- Commit: 995cbfa
- Treatment Risk @ tau=0.35: 0.238
- Population Risk @ tau=0.35: 0.540
