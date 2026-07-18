# Code Analysis — Paper 233 (Full-Spectrum GNN)

## Evaluation Path
- Main script: `Heterophily/small/train.py`
- Evaluation command: `cd Heterophily/small && python train.py --net Cheb_GAT --dataset Squirrel --semi_rnd --hidden 64 --K 2 --lr 0.02 --wd 0.0005 --dropout 0.900 --prop_lr 0.01 --prop_wd 0.0001 --dprate 0.900 --heads 1 --alpha_init -2.000`
- 10 runs with different fixed seeds, outputs mean ± CI

## Train/Inference Path
- `train.py:RunExp()` — main training loop
- Single call to `train(model, optimizer, data)` per epoch (one forward/backward pass)
- `test(model, data)` computes accuracies on train/val/test masks
- Early stopping based on val_loss history (200 epoch patience)

## Model Architecture (Cheb_GAT)
- Input → lin1 (features→hidden) → ReLU → lin2 (hidden→num_classes) → ChebConv1 → h + sigmoid(α)·GAT(h) → ChebConv2 → log_softmax
- Key params: hidden=64, K=2 (Chebyshev order), heads=1, dropout=0.9, dprate=0.9, alpha_init=-2.0
- Model checkpointed at epoch with best validation LOSS

## Metric Parser
- Test Accuracy: `test acc mean = (\d+\.\d+) ± (\d+\.\d+)` — group 1 is mean % accuracy
- Runtime: `each run avg_time:(\d+\.\d+)s`
- GPU Memory: must be measured separately (nvidia-smi or torch.cuda)

## Dataset
- Squirrel: 2223 nodes, 65718 edges, 2089 features, 5 classes
- Bundled in repo: `Heterophily/data/squirrel_filtered_directed.npz`
- Random splits: 2.5% train, 2.5% val per class (~55 train, ~56 val)

## Safe Modification Targets
- `train.py:RunExp()` — optimizer setup, scheduler, checkpointing logic, weight averaging
- `train.py:train()` — gradient clipping, edge dropout
- `models.py:Cheb_GAT.__init__()` / `forward()` — architecture changes
- Training hyperparameters (lr, dropout, K, heads, alpha_init, etc.)

## Risky Files (DO NOT MODIFY)
- `dataset_loader.py` — data loading, splits
- `utils.py` — random_splits(), fixed_splits()
- Data files in `Heterophily/data/`
- Metric computation in `train.py:main` (test_acc_mean computation)
- Scoring script: `/tools/record_score.sh`

## Key Observations
1. `best_val_acc` is computed but NEVER used for checkpointing; only `best_val_loss` is used
2. No gradient clipping in training loop
3. No learning rate scheduler
4. Model outputs `log_softmax` and training uses `F.nll_loss`
5. SEEDS list is fixed for reproducibility: [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539, 3212139042, 2424918363]
