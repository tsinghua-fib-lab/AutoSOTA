# Code Analysis - Paper 448: MES-RET

## Evaluation Path
- `run_quick_reproduction.py` — main eval script. Runs MES-RET → CMA-ES → compares metrics.
- Stdout reports #Better, #Worse, #Tie, Friedman ranks, runtime.
- Saves full results to `/repo/reproduction_full.json`.
- `eval_results.py` — reads saved results and re-reports metrics (no re-computation).

## Core Algorithm
- `mes_ret_opt_v2.py` — MESRET class (lines 52-300+) and CMAES baseline (lines ~310-350).
- MESRET phases per generation:
  1. Phase 1: Self evolution (independent CMA-ES step per task with transfer injection)
  2. Phase 2: Reward calculation (improvement-based or diversity-based, stochastic switch)
  3. Phase 3: Knowledge aggregation (weighted by rewards)
  4. Phase 4: Reward-weighted evaluation (extra evaluations for high-reward tasks)
- Key parameters: sigma0=0.3, tau=1, popsize=100
- Transfer: Mean aggregation + Covariance aggregation (lines ~240-270)
- Reward: fit (improvement) or div (diversity) via Eq 8 stochastic switch

## Config Path
- `run_quick_reproduction.py` line 23: hardcoded config
- `mes_ret_opt_v2.py` MESRET class init: default parameters
- Budget: `max_fe = 3000 * 50 * n_funcs // 10` (= 420,000 for 28 funcs)

## Metric Parser
- Stdout lines: `#Better (MES-RET > CMA-ES): X/Y`
- `#Worse`, `#Tie`, `Friedman Rank MES-RET: X.XXXX`
- Runtime from `Total time: Xs` or individual timings
- Fallback: parse `/repo/reproduction_full.json`

## Known Bugs (CODE-01)
- `sigma0_val[0]` at line 77: assumes array but sigma0_val is scalar
- Seed: `self.seed + t * 10000` can produce collisions

## Safe Modification Targets
- `mes_ret_opt_v2.py` MESRET class methods
- `run_quick_reproduction.py`: parameterize for config search

## Risky Files (do not change)
- `eval_results.py` — metric definition
- `build_cec2017_tasks()` — task definitions
- `evaluate()` — evaluation protocol
- `CMAES` class — baseline implementation

## Container Resources
- GPU: 2x A100-80GB (not used, CPU-only workload)
- BLAS threads: 4
- Cache mounts: /autosota_cache, /datasets, /models
