# Reproducibility guide

This document separates lightweight repository checks from paper-scale reproduction.
The smoke tests are data-free and are intended to verify that the package installs
and that the symbolic core is functional. Full experiments require the model
checkpoint and cluster database described in [`data/DOWNLOAD.md`](../data/DOWNLOAD.md).

## Quick checks

| Target | Command | Data needed | Expected runtime | Expected result |
|---|---|---:|---:|---|
| Package import and symbolic core | `python scripts/smoke_test.py` | None | < 1 min | `SIGS smoke test passed.` |
| Unit-style tests | `pytest -q` | None | < 1 min | All tests pass |

## Paper-level components

| Result family | Entry point | Data needed | What it checks |
|---|---|---|---|
| Classical scalar PDE benchmarks | `notebooks/demo_pdes.ipynb` | model checkpoint, cluster database | Burgers, diffusion, KdV, damped wave and related baselines |
| Shallow water system | `notebooks/demo_shallow_water.ipynb`, `scripts/discover.py`, `scripts/optimize.py` | model checkpoint, cluster database | Multi-field symbolic discovery and parameter refinement |
| Compressible Euler experiments | `scripts/euler_search.py`, `scripts/run_euler_four_fields.py`, `scripts/run_euler_same_series.py`, `scripts/plot_euler.py` | model checkpoint, cluster database | Coupled nonlinear system experiments and plotting |
| Baselines | `baselines/` | baseline-specific dependencies | Reference comparisons with PINN/FBPINN, HD-TLGP and FEniCS-style workflows |

## Recommended reviewer path

1. Install the repository in a fresh environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. Run the data-free checks.

   ```bash
   python scripts/smoke_test.py
   pytest -q
   ```

3. Download the model and cluster artifacts following [`data/DOWNLOAD.md`](../data/DOWNLOAD.md).

4. Run a small Stage I sample from the README quick start before launching larger experiments.

5. For paper-scale runs, use fixed seeds and record the command, commit SHA, hardware, wall-clock time and downloaded artifact versions.

## Reporting reproduction results

When reporting a reproduction attempt, please include:

- repository commit SHA;
- Python version and operating system;
- whether CPU or GPU was used;
- versions/checksums of downloaded model and cluster artifacts;
- command or notebook used;
- random seed;
- wall-clock time;
- final residual and, where available, relative L2 error.
