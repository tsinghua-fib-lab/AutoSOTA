# Code Analysis for cuRegOT (Paper 6018) Optimization

## Repository Structure

- `/repo/eval_repro.py` — Main evaluation script (SPLR + BCD baseline, 10-run median)
- `/repo/setup.sh` — Container setup (cuDSS library fix)
- `/repo/curegot/` — Local source (not used at runtime; installed package used)
- `/repo/src/sinkhorn_splr.cu` — SPLR solver implementation (CUDA)
- `/repo/src/solvers_numpy.cpp` — Python bindings (pybind11)
- `/repo/src/solvers.h` — Solver function signatures
- `/repo/setup.py` — Build system

## Evaluation Path

1. `eval_repro.py` → `curegot.numpy.sinkhorn_splr()` → `cuda_sinkhorn_splr()` (C++)
2. Generates Synthetic I (iid): N(0,1) positions, absolute distance, normalized by max
3. Problem: 1600×1200, eta=0.001, S=10, tol=0.0, max_iter=5000
4. 10 independent runs (seeds 0-9), median metrics
5. Output format: `Marginal_Error_SPLR: <value>` and `Runtime_SPLR: <value>s`

## Config/Parameters

Parameters exposed via Python kwargs to `sinkhorn_splr()`:
- `density` (float): Max density for Hessian sparsification. Default: 10/min(n,m) ≈ 0.00833
- `shift` (float): Hessian regularization. Default: 0.001
- `sparsity_pattern_cycle` (int): Frequency of sparsity pattern updates. Default: 30; eval uses S=10
- `candidate_sinkhorn_iter` (int): Sinkhorn iterations for candidate generation. Default: 3
- `tol` (float): Convergence tolerance. Default: 0.0 (runs full max_iter)
- `max_iter` (int): Maximum iterations. Default: 5000
- `verbose` (int): Verbosity. Default: 0

Parameter flow to C++:
- `density` → `density_max` (clamped to [0,1])
  - `density_min` = 0.01 * density_max
  - Initial `density` = 0.1 * density_max
  - `Kmax` = density_max * n * (m-1)
- `shift` → `shift_max`
  - Actual shift = min(gnorm, shift_max)
- Parameter adaptation: density *= 1.1 (bad move) or 0.99 (good move) every sparsity_pattern_cycle iterations

## Algorithm Flow (per SPLR iteration)

1. Compute low-rank BFGS vectors (y, s)
2. Compute search direction (sparse Cholesky + BFGS update)
3. Wolfe line search
4. Compare with Sinkhorn candidate iterate (every sparsity_pattern_cycle iterations)
5. Update density (adaptive)
6. Recompute sparsified Hessian

## Metric Parser

Parses stdout lines matching:
- `Marginal_Error_SPLR: <float>` → primary metric
- `Runtime_SPLR: <float>s` → resource metric
- `Marginal_Error_BCD: <float>` → reference baseline
- `Runtime_BCD: <float>s` → reference baseline

## Red-Line Constraints (Safe Modification Targets)

**Safe to modify:**
- `eval_repro.py`: Add/change SPLR parameters (density, shift, candidate_sinkhorn_iter)
- `eval_repro.py`: Modify BCD parameters (baseline only, not scored)
- `src/sinkhorn_splr.cu`: Algorithm logic, density adaptation, line search parameters
- `src/solvers_numpy.cpp`: Default parameter values, new parameter exposure
- `setup.py`: Build configuration

**Do NOT modify:**
- Data generation (Synthetic I formula)
- Metric computation (compute_marginal_error)
- Test splits (seeds 0-9, 10 runs)
- Scoring scripts (`/tools/record_score.sh`)
- Evaluation protocol (n=1600, m=1200, eta=0.001, tol=0.0, cost normalization)

## No Pre-downloaded Data

All data generated synthetically from N(0,1) in eval_repro.py. No `/paper_data` mount needed.

## Build Notes

To rebuild after C++ changes:
```bash
cd /repo && pip install --no-build-isolation .
```
