# InfoGlobe Code Analysis for SOTA Optimization

## Evaluation Path
- **Eval command**: `python3 eval_final.py`
- **Script**: `/repo/eval_final.py`
- **Data**: `/repo/sim_data/result/adata_17_res.h5ad` (240 cells x 1100 genes, 8 cell types)
- **Output format**: stdout lines matching `Trustworthiness|Continuity|Spearman Correlation`; JSON at `/repo/reproduction_final.json`

## Training Path
- **Model**: `InfoGlobe.infoglobe.GlobeEmbedding` (K=20 factors)
- **Primary training method**: `sparse_fit()` at `/repo/InfoGlobe/infoglobe.py` line ~290
- **Alternative method**: `fit()` (uses sampled pairs, GradScaler, learnable c — NOT used for eval)
- **Initialization**: `InfoGlobe.utils.initialize()` — random init, no seed
- **Normalization**: `InfoGlobe.utils.normalize()` — `torch.softmax`

## Config Parameters
- **K**: 20 (number of factors, 10-30)
- **max_iter**: 10000 (10000-50000)
- **l1_ratio**: 0.5 (reconstruction loss weight)
- **l2_ratio**: 0.5 (geometric MDS loss weight)
- **l3_ratio**: 0.1 (orthogonal regularization weight)
- **lr**: 1e-3 (fixed Adam learning rate, no scheduler)
- **c**: 1 (fixed scaling factor in sparse_fit; is learnable in fit())

## Metric Parser
- Trustworthiness: `sklearn.manifold.trustworthiness` at KNN=7,12
- Continuity: `trustworthiness(Q, P, ...)` (swapped args)
- Spearman Correlation: `scipy.stats.spearmanr` on all pairwise Fisher-Rao distances

## Key Files
- `/repo/eval_final.py` — Evaluation script (SAFE TO MODIFY: parameters, seed loop, model config)
- `/repo/InfoGlobe/infoglobe.py` — GlobeEmbedding class with `sparse_fit()` (SAFE TO MODIFY: optimizer, loss, scheduler)
- `/repo/InfoGlobe/utils.py` — `initialize()`, `normalize()`, `get_knn()` (SAFE TO MODIFY: add seed param)
- `/repo/InfoGlobe/metrics.py` — Loss functions: `angle_mse_loss`, `fisher_rao_dis`, etc. (SAFE TO MODIFY: add new loss functions)
- `/repo/sim_data/result/adata_17_res.h5ad` — Data file (DO NOT MODIFY)

## Red-Line Boundaries
- DO NOT modify: metric definitions, test data, splits, labels
- DO NOT hard-code predictions or metric values
- DO NOT change the evaluation protocol

## Key Differences fit() vs sparse_fit()
- `fit()` has learnable `c_raw`, uses sampled pairs (num_pairs), GradScaler
- `sparse_fit()` uses full distance matrix, no c scaling, no GradScaler
- `sparse_fit()` adds orthogonal_loss (loss3) which fit() lacks
- Both use `angle_mse_loss` for MDS loss

## Optimization Opportunities
1. Add learnable c to sparse_fit (present in fit() but missing in sparse_fit())
2. Add LR scheduler (fixed 1e-3 may be suboptimal)
3. Add gradient clipping (three loss terms at different scales)
4. Add seed control and ensemble (non-deterministic init)
5. Curriculum schedule for loss weights
6. Two-phase optimization (reconstruction warmup)
7. Tangent PCA initialization
