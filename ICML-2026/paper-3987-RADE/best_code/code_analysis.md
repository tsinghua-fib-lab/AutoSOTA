# Code Analysis — Paper 3987: RADE

## Evaluation path
- `main.py` runs training + evaluation end-to-end
- `eval.py` provides `evaluate()` which runs model on clean graph
- Output format: per-run "Final Test: XX.XX", aggregate "Final Test: XX.XX ± YY.YY"
- Results saved to `results/cora/MPNN_gcn.csv` and `single_dataset/results.csv`
- Metric parser in `logger.py` — `print_statistics()` computes mean/std of test accuracies

## Key files
- `main.py`: training loop, data loading, training orchestration
- `models.py`: MPNNs class — instantiates GCN/GIN/GAT convs, supports RADE/dropout/BN/LN
- `rade_convs.py`: RADEGCNConv, RADEGINConv, RADEGATConv with EP correction (delta-method)
- `augmentation.py`: BernoulliEdgeAugmentor — samples edge drops/adds
- `pq_gradnorm.py`: PQGradNormTuner — online p/q tuning via gradient norm matching
- `parse.py`: CLI arg definitions, auto-defaults for full-batch datasets
- `eval.py`: evaluate() — standard eval on clean graph
- `dataset.py`: data loading (Planetoid for Cora)

## Config path
- CLI args only, no config file (except `sweep.yaml` for W&B sweeps)
- Auto-defaults in `parse.py` via `FULL_BATCH_AUTO_DEFAULTS`

## Critical: deterministic mode
- `fix_seed()` sets `torch.use_deterministic_algorithms(True)` and `torch.set_deterministic_debug_mode("error")`
- Dropout tested and confirmed working under this mode
- All regularization must be compatible with deterministic mode

## Safe modification targets
1. **main.py training loop**: LR schedule, p/q warmup, loss function
2. **parse.py**: add new CLI args, change auto-defaults
3. **models.py**: architecture (already supports --bn, --dropout, --res, etc.)
4. **rade_convs.py**: EP correction formula (4th-order)
5. **eval.py**: TTA (test-time augmentation)
6. **dataset.py**: structural encoding features

## Risky files
- `pq_gradnorm.py`: complex gradient-based p/q tuning — modifying is risky
- `augmentation.py`: edge sampling core — must preserve paper consistency

## Eval command (corrected for in-container use)
```bash
cd /repo/RADE_Node_Classification/full_batch && python3 main.py --dataset cora --gnn gcn --aug_tech rade --rade_variant rade-of --ep_correction True --pq_gradnorm True --runs 5 --device 0 --data_dir /datasets --seed 42
```
