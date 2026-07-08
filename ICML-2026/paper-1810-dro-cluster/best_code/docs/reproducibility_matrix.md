# Reproducibility Matrix

This release keeps algorithms source-grounded in the research checkout and archived local source trees.
Raw third-party datasets and large intermediate matrices are not committed.

## Included Commands

| Paper component | Release status | Entry point |
| --- | --- | --- |
| Main synthetic simulation | Included for DRO, Lasso, MFC variants, ACC, k-medoids, and k-means | `scripts/run_synthetic_experiments.py --scenario main` |
| Common-factor sweep | Included for source-grounded methods | `scripts/run_synthetic_experiments.py --scenario common-factor` |
| Noise sweep | Included for source-grounded methods | `scripts/run_synthetic_experiments.py --scenario noise` |
| No-global-factor simulation | Included for source-grounded methods | `scripts/run_synthetic_experiments.py --scenario additional` |
| Ablation studies | Included | `scripts/run_ablation_studies.py --ablation all` |
| Extended Yale B face clustering | Included for DRO, Lasso, MFC variants, ACC, and k-medoids | `scripts/run_face_experiments.py --data-dir data/external/CroppedYalePNG` |
| Financial portfolio backtest | Documented, not repackaged as a standalone script | `docs/portfolio_experiment.md` |

## Source Provenance

- DRO solver and synthetic generator:
  `code/subspace_clustering/subspace_clustering.py` and `code/subspace_clustering/simulation.py`.
- Ablation studies:
  `code/ablation_studies.py`.
- ACC/CORD, Lasso nodewise regression, square-root Lasso, and k-medoids:
  archived `portfolio_backtesting.stock_selection.clustering` source and old simulation scripts.
- MFC:
  historical `code/portfolio-backtesting` git commit `8a30e26` (`Add multifactor clustering algorithm`).
- Extended Yale B preprocessing and split generation:
  `code/image_dataset/image_experiments.ipynb`.
- Portfolio backtest:
  `code/subspace_clustering/portfolio.py`, `portfolio_analysis.ipynb`,
  and archived `portfolio_backtesting` package code.

## Not Vendored

The manuscript also reports SSC, SSC-EnSC, SSC-OMP, LRR, and co-clustering.
The old tree contains wrappers for SSC, SSC-EnSC, SSC-OMP, LRR, and co-clustering, but they depend on
third-party code or packages. Those methods are documented rather than copied, to avoid redistributing
third-party code or inventing substitute algorithms.
