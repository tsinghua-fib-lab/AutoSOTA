# Code Analysis: Approximate Proportionality in Online Fair Division (Paper 310)

## Evaluation Path
- Main script: `online_fair_division_experiments.py`
- Eval command: `python3 online_fair_division_experiments.py --n 8 --m 40 --trials 500 --seed 1428 --outdir results`
- Output: stdout (print_summary) + results/summary.csv + stress_summary.csv + figures
- Metric parser: Parse stdout for "uniform" block → "Alg. 1" line → PROP1 and welfare values
- Baseline: PROP1=1.000, welfare=0.892 (uniform family, Alg. 1)
- Greedy-1 baseline: PROP1=0.775, welfare=0.946 (uniform, welfare upper bound)

## Algorithm Structure

### Algorithm 1 (`algorithm1_miv`, lines 164-217)
- MIV-normalizes values (each agent scaled so max=1)
- Online: for each good t, computes phi potential for each possible recipient
- phi: complex function based on PROP1 accounting terms (x, y)
- r_i: tracks first max-valued good for each agent (tol=1e-10 threshold)
- Selects agent minimizing sum of phi values
- Guarantees PROP1 ≥ 1.0 (theoretical result)

### Greedy Strategies
- Greedy-1: max_i v_i(g_t)/v_i(G^(t)) — best welfare, poor PROP1
- Greedy-2: min_i v_i(A_i^(t-1))/v_i(G^(t)) — PROP1=1.0, low welfare
- Greedy-3: min PROP1 slack — PROP1=1.0, moderate welfare

### Instance Families
- uniform: rng.random((n, m)) — main evaluation family
- dense: binary mask + value scaling — Alg 1 beats Greedy-1 on welfare!
- correlated: common signal + idiosyncratic noise
- specialist: each good has one specialist agent

## Key Insight
Algorithm 1 achieves PROP1=1.0 but leaves 5.4% welfare on the table vs Greedy-1 (uniform).
The optimization frontier: close this gap while maintaining PROP1=1.0.

## Safe Modification Targets
1. `algorithm1_miv`: Add post-processing (swap_improve, offline_optimize)
2. `phi_for_agent`: Add welfare-bias term
3. `run_random_families`: Add per-trial assertions, new algorithm variants
4. New functions: swap_improve, offline_optimize, best_of_k, etc.

## Risky Files (do not touch)
- `metrics()`: Defines PROP1 and welfare computation — DO NOT MODIFY
- `normalize_by_miv()`: Core normalization — DO NOT MODIFY
- `INSTANCE_FAMILIES`: Data generation — DO NOT MODIFY
- `greedy1/2/3`: Baseline algorithms — DO NOT MODIFY for comparison integrity

## Container Notes
- Container: autosota_repro_paper_310
- Image: autosota/paper-310:reproduced
- Base: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
- No GPU needed — pure CPU numpy computation
- No datasets/models — synthetic generation
- HF_TOKEN configured but not used

## Optimization Constraints
- Must maintain PROP1=1.0 (exact, no regression)
- Must not modify metrics, data generation, or evaluation protocol
- All changes must be within Algorithm 1 or post-processing
