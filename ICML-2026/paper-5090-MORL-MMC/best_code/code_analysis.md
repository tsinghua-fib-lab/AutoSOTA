# Code Analysis — Paper 5090: Constrained Max-Min MORL

## Evaluation Path
- Entry: `python3 /repo/tabular_experiment.py`
- Eval command: `python3 tabular_experiment.py --momdp_type bipartite --n_states 30 --n_actions 3 --K 2 --L 1 --gamma 0.8 --beta 0.03 --l_w 0.001 --ITER 3000 --conv_th 1e-4 --tightness 0.1 --seeds 0 1 2`
- Timeout: 10 min
- Output format: Parse SUMMARY section for `Constrained max-min : error=X.XXXXXX`

## Key Files
- `/repo/tabular_experiment.py` (347 lines) — main implementation
- `/repo/results_reproduction.json` — cached reproduction results
- `/repo/supplementary_ICML26/` — paper supplementary materials

## Architecture
- `generate_bipartite_momdp()` / `generate_hierarchical_momdp()` — MOMDP generation
- `constraint_range()` — LP to find min/max constraint values
- `solve_optimal_value()` — LP for ground-truth optimal max-min value
- `run_algorithm()` — Algorithm 1: dual optimization with:
  - Soft VI inner loop (fixed 10k max iter, conv_th=1e-4)
  - Projected gradient for dual variables u (constraint), w (objectives)
  - Euclidean simplex projection for w, clipping for u
- `softmax_policy()`, `evaluate_policy()` — policy extraction/evaluation
- `soft_vi_step()` — soft Bellman backup with entropy regularization

## Metric Parser
- Read stdout SUMMARY section: `Constrained max-min : error=X.XXXXXX`
- Or parse `--output` JSON: `.constrained_maxmin[].error`

## Safe Modification Targets
1. `run_algorithm()` — dual update logic, learning rates, momentum
2. `soft_vi_step()` — convergence criteria, beta schedule
3. `main()` — warm-start initialization, caching
4. Learning rate schedules, convergence thresholds

## Risky Files (do not modify)
- `fix_range.py`, `run_final.py`, `run_full_experiment.py`, `run_tighter.py` — historical scripts
- `results_*.json` — output artifacts
- `supplementary_ICML26/` — paper source materials

## Known Levers
1. `--tightness` (0.1): constraint tightness
2. `--beta` (0.03): entropy regularization
3. `--l_w` (0.001): dual learning rate
4. `--ITER` (3000): outer iterations
5. `--conv_th` (1e-4): inner loop convergence
