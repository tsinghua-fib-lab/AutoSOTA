# Code Analysis for Paper 6249

## Overview
Active timepoint selection for measure-valued trajectory reconstruction.
Synthetic oscillatory sequential branching data with OT-based tangent-space GP.

## Key Paths
- **Eval entry**: `experiments/active_sampling.py` → `main()` (hydra)
- **Config**: `conf/exp_sequential_branching.yaml`
- **Active loop**: `src/active_wasserstein/active/loop.py` → `ActiveLearningLoop.step()`
- **Surrogate**: `src/active_wasserstein/active/surrogate.py` → `LinearizedWassersteinGPSurrogate.fit()`
- **GP**: `src/active_wasserstein/inference/gpytorch_regression.py` → `GPyTorchHilbertRegressor.condition()`
- **Acquisition**: `src/active_wasserstein/acquisition/uncertainty.py` → `UncertaintySampler`
- **PCA/Basis**: `src/active_wasserstein/geometry/tangent.py` → `pca_vector_fields_with_components()`
- **Trajectory**: `src/active_wasserstein/data/synthetic_branching.py` → `OscillatorySequentialBranching`
- **Components**: `experiments/components.py` → strategy factories

## Metrics
- **Mean W2**: `uniform_metric` = mean Wasserstein-2 distance across eval times
- **Mean w-W2**: `velocity_metric` = velocity-weighted mean W2
- Computed in `experiments/active_sampling_utils.py` → `compute_weighted_metric()`
- Final stdout: `active:...: uniform=<Mean_W2>, velocity=<Mean_w-W2>`

## Safe Modification Targets
1. `src/active_wasserstein/acquisition/uncertainty.py` — modify scoring, add velocity weights
2. `src/active_wasserstein/inference/gpytorch_regression.py` — modify noise priors, GP config
3. `src/active_wasserstein/geometry/tangent.py` — modify PCA/SVD with trimming
4. `experiments/components.py` — add new acquisition strategies
5. `src/active_wasserstein/active/surrogate.py` — modify scaling, reference
6. Hydra config files — parameter tuning via CLI overrides

## Risk: No modification of
- Evaluation metric computation
- Test data/splits/labels
- Scoring scripts
- Trajectory definition (except config parameters)
