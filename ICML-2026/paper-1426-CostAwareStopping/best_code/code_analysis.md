# Code Analysis — Paper 1426 SOTA

## Evaluation Path
- **Entry point:** `/repo/evaluate.py --quick`
- **Data source:** Pre-computed pickle `/repo/notebooks/empirical_results/lcbench_known_cost_metrics_per_acq_updated.pkl`
  - Contains BO trajectories for 4 datasets: Fashion-MNIST, adult, higgs, volkert
  - Each trajectory has 50 seeds × up to 201 iterations (200 BO + initial)
  - Signals: PBGI(1e-3/1e-4/1e-5) acq, LogEIPC acq, exp min regret gap, regret upper bound, PRB, best observed, test error, cumulative cost
- **Alternative:** `/repo/compute_rankings.py` and `/repo/compute_rankings_v2.py` — standalone ranking scripts using same pickle

## Ranking Computation
- For each (dataset, acquisition, stopping_rule), compute cost-adjusted regret:
  - regret = test_error[stop_idx] - best_error + 1e-4 × cumulative_cost[stop_idx]
- Within each (dataset, acquisition) pair, rank 7 stopping rules by regret (lower = better)
- Top-1%: percentage of datasets where a stopping rule ranks #1

## Stopping Rules (evaluate.py lines 348-371)
1. **PBGI:** `PBGI(1e-4)_acq >= best_obs[k-1]` - stop when acquisition exceeds best
2. **LogEIPC-med:** `LogEIPC_acq <= log(0.01) + median(LogEIPC[1:21])` - fixed threshold
3. **SRGap-med:** `exp_min_regret_gap <= 0.1 × median(exp_min_regret_gap[1:21])`
4. **UCB-LCB:** `regret_ub <= 0.01`
5. **PRB:** `PRB >= 0.95`
6. **GSS:** IQR-based convergence check
7. **Convergence:** `best_obs[k] == best_obs[k-5]`

## Key Files
| File | Role | Safe to modify? |
|------|------|-----------------|
| `evaluate.py` | Main eval script with ranking computation | Yes — stopping rules, thresholds, window params |
| `compute_rankings.py` | Standalone ranking computation | Yes — same as evaluate.py |
| `pandora_automl/utils.py` | GP model fitting (`fit_gp_model`) | Yes — but effects only in full mode |
| `pandora_automl/acquisition/stable_gittins.py` | PBGI acquisition | Yes — but effects only in full mode |
| `pandora_automl/acquisition/log_ei_puc.py` | LogEIPC acquisition | Yes — but effects only in full mode |
| `run_lcbench_experiment.py` | Full BO experiment runner | Yes — but effects only in full mode |

## Reusable /paper_data
- `data_2k_lw.json` (929MB) — LCBench benchmark data, already symlinked

## Modification Strategy
- **Quick mode (post-hoc):** Can modify stopping rules, thresholds, windows, and ensemble logic in `evaluate.py` without re-running BO
- **Full mode (re-running BO):** Can modify GP model, acquisition functions, initial design, cost model
- **Mixed:** Can generate new stopping signals from existing pickle data (e.g., smoothed PBGI acq)

## Risk Assessment
- Changing stopping rule logic is low-risk: internal to ranking computation, no eval protocol change
- Changing GP model requires care: must not change metric definition or benchmark data
- Hard constraints: never modify test data, metric definitions, or scoring scripts

## Key Bounds
| Bound | Value |
|-------|-------|
| eval_command | `python3 evaluate.py --quick` |
| eval_timeout | 30 min |
| max_iterations | 12 |
| target_attempts | 6+ |
| baseline Top-1 | 75.0% |
| baseline Top-2 | 100.0% |
| baseline Top-3 | 100.0% |
