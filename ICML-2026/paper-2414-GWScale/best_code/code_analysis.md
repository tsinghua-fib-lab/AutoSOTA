# Code Analysis — Paper 2414: CNT-GW Solver

## Evaluation Path
- **Entry**: `eval.sh` → `python3 eval_horses.py`
- **Evaluation script**: `eval_horses.py` — loads meshes, runs CntGW solver, prints JSON with `time_seconds`, `gw_eps`
- **Metric parser**: JSON stdout keys `time_seconds` (TIME), `gw_eps` (GW_eps)
- **Timeout**: 10 minutes

## Key Source Files
- `solvers/eot/sinkhorn.py` — SinkhornSolver (inner OT loop, dominant runtime ~80%)
- `solvers/gromov_wasserstein/generic/generic_sinkhorngw.py` — SinkhornBasedGW (outer GW loop)
- `solvers/gromov_wasserstein/generic/embedding_based.py` — EmbeddingBasedGW
- `solvers/gromov_wasserstein/implementations/embedding_based/quadratic.py` — QuadraticGW (cost_matrix, update_potential)
- `solvers/gromov_wasserstein/implementations/embedding_based/cnt.py` — CntGW (kernel PCA, loss)
- `utils/implementation/kernels.py` — kernel_pca, reduce_kernel (CPU-based eigendecomp)
- `utils/math/dimension_reduction.py` — symmetric_pca (scipy eigsh)
- `utils/implementation/gw_losses.py` — gw_loss_from_points, gw_loss_euclidean
- `utils/implementation/initializations.py` — Potential initialization strategies

## Reusable Resources
- Mesh PLY files in `data/` (included in repo)
- `/autosota_cache`, `/datasets`, `/models` — available for caching

## Safe Modification Targets
1. `sinkhorn.py:237-261` — Sinkhorn solve loop (warm-start f,g, momentum)
2. `generic_sinkhorngw.py:348-362` — solver_step (skip clear, pass prev f,g)
3. `dimension_reduction.py:15-33` — symmetric_pca (GPU eigh)
4. `kernels.py:83-146` — kernel_pca, reduce_kernel (GPU path)
5. `cnt.py:105-111` — loss method (approx=False path)
6. `gw_losses.py:72-75` — gw_loss_from_points (hardcoded .to("cuda"))

## Risky Files (do not modify)
- `eval.sh`, `eval_horses.py` — evaluation protocol
- `data/` — test data
- `utils/math/costs.py`, `utils/math/functions.py` — core math (change only if correct)
