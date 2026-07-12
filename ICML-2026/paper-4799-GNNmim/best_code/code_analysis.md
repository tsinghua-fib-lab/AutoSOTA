# Code Analysis for Paper 4799 — GNNmim SOTA Optimization

## Evaluation Path
- `eval_script.py` → loads `synthetic.pt` → applies FD-MNAR at mu=0.50 → prepares GNNmim features → calls `evaluate_gcn()` → parses F1/Acc/Loss
- The script replicates `main.py` code path for (model=gnnmim, mechanism=FDMNAR, prob=0.5)

## Key Files
- `/repo/eval_script.py` — main evaluation entry point
- `/repo/models.py` — GCNFull, GraphSAGEFull, GATFull, GINFull, GCNII, GCNmf model classes
- `/repo/utils.py` — evaluate_gcn(), fill_nan_with_col_mean_split(), produce_NA_ood(), evaluate_gnn()
- `/repo/data/synthetic.pt` — 1000 nodes, 5 features, 2 classes

## Train/Inference Path (evaluate_gcn in utils.py lines 275-376)
1. For each seed (5 seeds: [1, 43, 15, 118, 222]):
2. Fill NaN features with column mean (from train+val only)
3. Create GCNFull(in_channels, 128, num_classes, num_layers=2) → num_layers=2 hardcoded
4. Adam optimizer with lr=0.01 (NO weight_decay)
5. Train max 500 epochs with patience=50
6. Cross-entropy loss on training nodes
7. Best model selected by validation F1
8. Test F1, accuracy, loss computed

## Config Parameters (hardcoded in evaluate_gcn)
- lr=0.01 (line 306)
- hidden_channels=128 (line 307)
- num_layers=2 (line 309)
- dropout=0.5 (GCNFull default, line 35)
- max_epochs=500 (line 278)
- patience=50 (line 279)
- NO weight_decay (line 316)
- NO learning rate scheduler

## Metric Parser
- Output line: "F1 Score:     0.7508"
- Parser: extract float after "F1 Score:"

## Safe Modification Targets
- `/repo/models.py` GCNFull class: BatchNorm, residual connections, input dropout
- `/repo/utils.py` evaluate_gcn(): optimizer params, scheduler, loss function, training epochs, fill strategy
- `/repo/eval_script.py`: max_epochs, patience, hidden_channels, dropout, num_layers

## Risky Files (DO NOT TOUCH)
- Data loading: synthetic.pt
- produce_NA_ood(): missingness mechanism
- Dataset splits (train/val/test masks)
- Metric computation (f1_score, accuracy_score)
- Seeds list [1, 43, 15, 118, 222]

## Reusable Resources
- None external; synthetic.pt is in-repo
