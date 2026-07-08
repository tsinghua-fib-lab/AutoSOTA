# Code Analysis for Paper 2026 - STPGC Optimization

## Evaluation Path
- **Coarsening**: `python3 graph_coarsening.py --dataname Cora --ratio 0.5`
  - Output: `coarsened_graph/Cora_0.50.npy` (old_to_new mapping + res dict with new_x, new_edge_index, node_label)
- **Training+Evaluation**: `python3 train_gcn.py --dataset Cora --coarsening_ratio 0.5 --runs 10`
  - Loads coarsened graph from `coarsened_graph/Cora_0.50.npy`
  - Trains 2-layer GCN (Net class) on coarsened graph
  - Evaluates on original Cora test set
  - Output: `ave_acc: X.XXXX +/- X.XXXX` (average of 10 runs)

## Key Files
- `GNN/graph_coarsening.py`: STPGC coarsening algorithm (CoreAlgorithm class)
- `GNN/train_gcn.py`: GCN training and evaluation script
- `GNN/network.py`: Model architectures (Net, GCN_2, GCN, net_gcn, APPNP_Net)

## Key Code Points
- **Feature aggregation**: `graph_coarsening.py:641` - `torch.mean(data.x[new_to_old[v]], dim=0)` 
- **Label assignment**: `graph_coarsening.py:660-667` - majority vote over constituent nodes
- **Mis-percentage**: `graph_coarsening.py:675` - computed but not saved to output
- **Model**: `Net` class in `network.py:6-23` - 2-layer GCNConv, hidden=256, no BatchNorm
- **Training**: Adam(lr=0.01), nll_loss on log_softmax, 200 epochs, early_stopping=10
- **Coarsening params**: θ1=15 (deg1), del_edge=0.1 (heterophilic edge deletion), deg2=deg1
- **Strong collapse**: `graph_coarsening.py:317-402` - neighborhood subset collapse with tolerance

## Metric Parser
- Primary metric: Accuracy (from `ave_acc:` line in stdout)
- Format: `ave_acc: 0.8261 +/- 0.0039` → 82.61%
- Backup: per-run individual test_acc values printed before average

## Config Path
- Coarsening: hardcoded in `para_dict_num_edge_01` (line 755) and CLI args (lines 764-771)
- Training: CLI args with defaults (lines 93-97)

## Safe Modification Targets
- Feature aggregation function (line 641)
- GCN model architecture (network.py Net class)
- Training hyperparameters (lr, epochs, early_stopping, hidden dim)
- Loss function (nll_loss → cross_entropy with label smoothing)
- Optimizer setup (add LR scheduler)
- Coarsening parameters (θ1, del_edge)

## Risky Files (do not modify)
- Dataset loading (Planetoid, CitationFull)
- Train/val/test split creation (index_to_mask)
- Evaluation metric computation (test_acc calculation, ave_acc printing)
- Scoring script (/tools/record_score.sh)

## Red-Line Constraints
- Do not modify evaluation protocol
- Do not change test data, labels, or splits
- Do not hard-code predictions or metric values
- Report all metrics honestly
