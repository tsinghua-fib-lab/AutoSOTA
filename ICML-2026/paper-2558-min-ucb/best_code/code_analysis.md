# Code Analysis for Paper 2558 SOTA Optimization

## Evaluation Path
- `run_eval.py` → calls `APUB.solve_two_stage_apub()` and `SAA.solve_nf()`
- Parses metrics from stdout and writes to `reproduction_results.json`
- Metrics: Time(s) via `time.perf_counter()`, Iteration (num_optimal_cuts)

## Key Files
- `apub.py` — APUB-M L-shaped algorithm (master problem, feasibility cuts, optimality cuts)
- `saa.py` — SAA-M baseline L-shaped algorithm
- `evaluation.py` — Full sweep evaluation (alpha sweep, M sweep, parallel execution)
- `run_eval.py` — Reproduction/SOTA evaluation entry point
- `config.yaml` — Problem parameters (T matrix, c vector, generator settings)
- `utils.py` — Config loading and parameter sampling

## Core Algorithm (APUB-M)
1. `initialize_master_problem()` — builds master LP with x variables and eta
2. `solve_two_stage_apub()` — main loop: solve master → generate optimality cuts → repeat
3. `generate_optimality_cuts()` — for each of N=120 scenarios, solve second-stage LP, collect duals, bootstrap aggregate (M=5000), compute CVaR bound, add cut

## Bottlenecks (per iteration)
1. **N=120 sequential second-stage LP solves** (lines 86-109) — each creates new `gp.Model()`, solves, destroys
2. **`np.apply_along_axis` for bootstrap counts** (line 122) — Python-loop overhead
3. **Default Gurobi parameters** — no tuning for small repeated LPs
4. **Master problem grows** — each iteration adds one constraint

## Safe Modification Targets
- `apub.py:solve_master_problem` — add solver parameters (Method, Presolve, Cuts)
- `apub.py:generate_optimality_cuts` — vectorize bootstrap; reuse models
- `apub.py:solve_two_stage_apub` — add warm-start, multi-cut, adaptive M
- `saa.py:solve_nf` — solver parameter tuning

## Risky Files (do not modify)
- `run_eval.py` — evaluation protocol (data loading, metric computation, output format)
- `config.yaml` — problem definition (T, c, constraints)
- `120.pkl` — pre-generated data
- `/tools/record_score.sh` — scoring infrastructure

## Verified Baseline
- APUB-M Iteration: 9.17 ± 1.39 (30 runs, seed=1234)
- APUB-M Time: ~6.1s (varies by hardware load)
- SAA-M Iteration: 7.97 ± 1.38
- SAA-M Time: ~3.6s
