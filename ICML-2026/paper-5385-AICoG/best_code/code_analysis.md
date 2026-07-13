# Code Analysis — Paper 5385: AICoG (Aitchison Embeddings for Compositional Graph Representations)

## Evaluation Path
- **eval.py** → calls main.py 5× → parses "ROC:" and "PR:" from stdout → prints SUMMARY line
- **main.py** → calls model.link_prediction() → uses held-out edges (sparse_i_rem/sparse_j_rem) + negative edges (non_sparse_i/non_sparse_j)
- **Metrics**: sklearn.metrics.roc_auc_score (AUC-ROC), sklearn.metrics.auc(tpr, precision) (PR-AUC)
- **Data**: Cora, D=8 (K=9), training edges from sparse_i.txt/sparse_j.txt, held-out from sparse_i_rem.txt/sparse_j_rem.txt, negatives from non_sparse_i.txt/non_sparse_j.txt

## Training Path
- Model: LSM (nn.Module + Spectral_clustering_init) in src/AICoG.py
- Init: spectral clustering (Normalized random-walk Laplacian, eigs with which="SR")
- Forward: softmax → ILR transform (learned Helmert basis, QR-orthonormalized) → D-dim Euclidean embeddings
- Loss: Bernoulli log-likelihood with 5×|E| uniform negative samples, plus scaling phase (first 500 epochs: softplus calibration)
- Optimizer: Adam with lr=0.01, 5000 epochs
- Bias terms: gamma (per-node random effects), bias (global), scaling_factor (for softplus calibration)

## Config Path
- All configuration via argparse in main.py
- Key params: K (9 for D=8), epochs (5000), scaling_epochs (500), lr (0.01), dataset (cora), LP (True)
- CUDA: default True, uses cuda:0

## Metric Parser
- eval.py regex: `ROC:\s+([\d.]+)` and `PR:\s+([\d.]+)`
- Parses both AUC-ROC and PR-AUC from each of 5 runs
- Reports mean ± std, final SUMMARY line

## Reusable Resources
- /datasets/ — pre-packaged Cora, CiteSeer, DBLP, ASTROPH, GRQC datasets
- /models/ — container cache (no pre-trained weights, all training from scratch)
- /autosota_cache/ — general cache

## Risky Files
- eval.py — DO NOT modify eval protocol, parsing, or number of runs
- Datasets under /repo/datasets/cora/ — DO NOT modify training/test splits
- src/AICoG.py link_prediction() — DO NOT modify AUC/PR computation or test data assembly

## Safe Modification Targets
1. **src/AICoG.py LSM_likelihood_bias()** — loss function (regularization, sampling, etc.)
2. **src/AICoG.py LSM.__init__()** — model architecture, initialization, parameter groups
3. **src/spectral_clustering.py** — initialization procedure (edge weights, eigenvector method)
4. **main.py** — training loop (optimizer, gradient clipping, hyperparameters, seeding)
5. **eval.py** — seed passing (NOT eval protocol), environment variables

## Baseline Commit (iter-0)
- AUC-ROC: 0.8407 (paper: 0.837, reproduce CI: [0.808, 0.8399])
- PR-AUC: 0.8713 (paper: 0.869, reproduce CI: [0.841, 0.8718])
- Tag: _baseline, _best → 0be828d
