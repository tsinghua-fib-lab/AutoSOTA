# Code Analysis for Paper 4384: MMD-Reg

## Evaluation Path
- experiments/benchmark.py -> main() -> loads data, runs algorithm, computes errors
- Eval command: uv run python -u experiments/benchmark.py ...

## Train/Inference Path
- No training; MMD-Reg is a non-learning registration method
- Inference: benchmark_utilities.py:mmd_reg() -> inner_objective_solution() for each W

## Config Path
- CLI args in benchmark.py:get_args()
- Key params: D (feature count), l (length scale), dist (Gaussian/Laplace)
- LM solver params hardcoded in benchmark_utilities.py:63: tol=2e-5, solver=cholesky, maxiter=100, damping=1.0

## Metric Parser
- Output JSON: results/pcpnet_cpu.json
- Keys: "Average Rotation Error" (deg), "Average Translation Error" (Euclidean), "Average Run Time" (seconds)
- scores.jsonl uses RRE (deg), RTE (Euclidean), Time (ms = RunTime*1000)

## Reusable Resources
- Dataset: /repo/datasets/processed/pcpnet_high_noise.hdf5 (1400 samples)

## Safe Modification Targets
- benchmark.py: CLI args, W generation (multi-scale, orthogonal RFF)
- benchmark_utilities.py: mmd_reg() iterative loop, LM params
- mmd_reg/mmd_unweighted.py: random_feature_map, MMD residual, inner_objective_solution

## Risky Files
- None that would change metrics or evaluation protocol
