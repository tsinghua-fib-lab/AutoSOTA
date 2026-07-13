# Best-Policy Identification (BPI)

Python experiments for best-policy identification on tabular environments (e.g., RiverSwim variants).

## Requirements
To run the examples you need at least Python 3.10 and the following libraries installed: `numpy scipy cvxpy mosek torch matplotlib notebook tqdm seaborn pandas cython`

## Setup
- Dependencies are Python-based (NumPy/TQDM, etc.). If needed, install your environment packages via pip/conda.
- `run.py` attempts to use an optional Cython acceleration (`utils/cutils.pyx`) via `pyximport`; it falls back to pure Python if the build/import fails.

## Run
Run from within this folder:
- `python run.py`

This writes compressed result files under `bpi/data/<env>/` (e.g., `*.pkl.lzma`).

## Plotting
- `make_plots.ipynb` contains plotting/analysis utilities (notebook outputs are cleared for submission hygiene).

## Acknowledgements and license

Parts of the code in this folder are based on "Model Free Active Exploration in Reinforcement Learning" by Alessio Russo and Alexandre Proutiere (see the upstream project: https://github.com/rssalessio/ModelFreeActiveExplorationRL). The upstream repository is released under the MIT License (with some files originally from BSuite under Apache-2.0 — see upstream `LICENSE` and `LICENSE-APACHE`).

If you redistribute or publish derivative code that includes substantial portions of the upstream project, please retain the upstream copyright and permission notice as required by the MIT license.

Suggested citation for the upstream work:

Russo, A., & Proutiere, A. "Model-Free Active Exploration in Reinforcement Learning." NeurIPS 2023. Upstream code: https://github.com/rssalessio/ModelFreeActiveExplorationRL

