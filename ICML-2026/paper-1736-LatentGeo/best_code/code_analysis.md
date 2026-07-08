# Code Analysis — Paper 1736: Geodesic Calculus on Implicitly Defined Latent Manifolds

## Evaluation Path
- Entry: `scripts/reproduce_torus_final.py::main()` → `run_trial()`
- Calls `GeodesicSolverNumpy.AugLagrangeMinimize()` from `src/latentgeodesics/geodesics/geodesic_solver_numpy.py`
- Metric parsing: stdout lines `Path Energy: <float>` and `Computation Time: <float>`

## Core Files
- `src/latentgeodesics/geodesics/geodesic_solver_numpy.py` — Main solver (AugLagrangeMinimize, Lagrangian, DLagrangian)
- `src/latentgeodesics/geodesics/metrics.py` — sqdist/gradsqdist definitions
- `scripts/reproduce_torus_final.py` — Evaluation script with torus definition, run_trial()

## Config/Parameters
- K=49 segments (resolution=48 interior nodes, 150 DoF)
- Torus: R=0.8, r=0.2
- Endpoints: (u=0,v=0) → (u=2.7658,v=1.1597)
- Solver: mu=100, alpha=100, maxmu=30000, tol=1e-5, tolConstraint=1e-6
- Inner: scipy BFGS, gtol=1/mu

## Metric Parser
- Path Energy: parse `Path Energy: <float>` from stdout
- Computation Time: parse `Computation Time: <float>` from stdout

## Reusable Resources
- No external data/models needed — torus is defined analytically
- GPU available but not used (scipy BFGS is CPU-only)

## Safe Modification Targets
1. `geodesic_solver_numpy.py: Lagrangian()` — vectorize energy loop (line 136)
2. `geodesic_solver_numpy.py: DLagrangian()` — vectorize gradient (lines 159-162)
3. `geodesic_solver_numpy.py: AugLagrangeMinimize()` — early stopping, solver method, maxiter
4. `scripts/reproduce_torus_final.py: compute_path_energy()` — vectorize (lines 57-61)
5. `scripts/reproduce_torus_final.py: run_trial()` — parameterize solver args, add x0 init

## Risky Files (do not modify)
- `src/latentgeodesics/__init__.py` — package init
- `pyproject.toml` — dependencies
- Evaluation protocol / metric definitions
