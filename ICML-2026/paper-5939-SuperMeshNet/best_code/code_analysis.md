# Code Analysis — SuperMeshNet (Paper 5939)

## Evaluation Path
- Entry: `reproduce.py` → `main()`
- Train: `train/train_GCN.py` → `train_GCN_comp()` / `train_GCN_sup()`
- Models: `models/GCN.py` (GCN_shared, GCNConv), `models/Common.py` (MLP, Decoder_F, Decoder_G)
- Data: `utils/dataset.py` (GraphDataset_paired, GraphDataset_unpaired)
- Interpolation: `utils/knn_interpolate.py` (knn_interpolate)
- Metric parser: `utils/analysis.py` → `RMSE(result_dir, model, learning, N_paired, N_total, ib, exp_list)`
  - Loads `.npy` loss curve, takes `np.mean(np.sqrt(loss_curve[-6:-1]))`
- Eval output: stdout line matching `SuperMeshNet GCN comp` with `RMSE = X.XXXXXX`

## Config Path
- Hidden size: 30 (hardcoded in train_GCN_comp default)
- Depth (GCN layers): 3 per processor (LR + HR)
- Learning rate: 1e-3 (Adam)
- AMP: Yes (GradScaler)
- ib_n (node-level centering): True for comp, False for sup
- Early stopping: test loss <1% improvement over last 15 epochs after epoch >15
- Max epochs: 5000 (but early-stop at ~30-35)
- Batch: 1 labeled pair + 2 labeled pair + 1 unpaired per step (3 sample effective batch)
- KNN interpolation: default k=3

## Metric Parser
- `RMSE()` computes sqrt of MSE test loss averaged over last 5 epochs. Already in RMSE scale.
- Primary metric: RMSE (lower is better). Baseline: 0.0411.

## Paper Data
- Data at `data/data_angle/` (1000 samples, subdirs 0-999)
- Each sample: L_mesh_geometry.npy, L_mesh_topology.npy, L_y.npy, H_mesh_geometry.npy, H_mesh_topology.npy, H_y.npy
- 333 LR nodes, 4053 HR nodes per sample

## Safe Modification Targets
1. `models/Common.py` — MLP.forward() (add BatchNorm calls), Decoder classes
2. `models/GCN.py` — GCN_shared (add GraphNorm, change architecture)
3. `train/train_GCN.py` — training loop (loss fn, optimizer, LR schedule, gradient clipping, EMA)
4. `utils/knn_interpolate.py` — k parameter
5. `reproduce.py` — data splitting determinism, config params

## Risky Files (do not modify logic)
- `utils/analysis.py` — RMSE metric computation
- `utils/dataset.py` — data loading and normalization
- Test data, labels, normalization constants
