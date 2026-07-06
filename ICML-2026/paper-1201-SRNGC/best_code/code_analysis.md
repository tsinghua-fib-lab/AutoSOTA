# Code Analysis for SRNGC (Paper 1201) - SOTA Optimization

## Evaluation Path

1. Entry: `src/real_data.py` -- parses args, loads data, calls `grid_search()`.
2. Eval wrapper: `/repo/eval_traffic.sh` -- loops over 5 seeds (2025-2029), runs `real_data.py` with:
   - `--dataset CausalTime --series 3 --subject 1 --seed $seed`
   - `--num_workers 1 --exec_idx 1 --penalty_type Fast_Shap --use_best`
   - Parses stdout for `AUROC=X.XXXX | AUPRC=X.XXXX`
3. Train: `src/train.py` `grid_search()` → `train_model()` → `evaluate_model()`
4. Config: `utils/configs.py` `build_training_config()` + `get_best_hparams()`
5. Model: `model/models.py` `ResidualMLP`
6. Penalty: `model/penalty.py` `Shapley_Penalty` (with `num_proj=1` for Fast_Shap)
7. Importance: `model/imp_measure.py` `Shapley_Value` (computes importance matrix from Jacobian)
8. Metrics: `utils/cls_performance.py` `cls_metrics()` -- computes AUROC/AUPRC from importance matrix vs ground-truth graph

## Data Flow

```
eval_traffic.sh
  → python src/real_data.py --dataset CausalTime --series 3 --penalty_type Fast_Shap --use_best
    → Data("./data", "CausalTime", 3).load_data() → X [480, 40, 20], network [20, 20]
    → TimeSeriesDataset(X, lag=2, Norm=True, device) → lagged sequences
    → get_best_hparams("CausalTime", "Fast_Shap", 3) → {lag:2, lr:0.005, hidden_dim:60, dropout:0.2, layers:5, ind_lambda:1.0, int_lambda:0.01, weight_decay:1e-5}
    → build_training_config(use_best=True) → single-point param_grid
    → grid_search() → train_model() + evaluate_model()
    → cls_metrics(network, importance_matrix) → {auroc, auprc}
    → stdout: "Shapley (diag)   | AUROC=X.XXXX | AUPRC=X.XXXX"
  → grep -oP "AUROC=\K[0-9.]+" | head -1
```

## Key Files and Modification Safety

| File | Role | Safe to modify? |
|------|------|-----------------|
| `utils/configs.py` | Config builder, param_grid | YES - add params, modify grid |
| `src/train.py` | Training loop, grid search | YES - add schedulers, gradient clipping, bootstrap |
| `model/penalty.py` | Shapley penalty computation | YES - add num_proj variants, NaN guards |
| `model/imp_measure.py` | Importance score computation | YES - add NaN guards, alternative aggregation |
| `model/models.py` | Model architectures | YES - architecture changes |
| `utils/setups.py` | Model/penalty/importance builders | YES - new penalty/importance types |
| `data/Dataset.py` | Data loading, lagged sequences | YES - lag changes, adaptive lag |
| `utils/cls_performance.py` | AUROC/AUPRC computation | DO NOT MODIFY - metric definitions |
| `src/real_data.py` | Entry point | Caution - only arg additions |
| `eval_traffic.sh` | Eval wrapper | Caution - only if needed |
| `data/CausalTime/` | Test data | DO NOT MODIFY |
| `/tools/record_score.sh` | Score recording | DO NOT MODIFY |

## Baseline Metrics
- AUROC: 0.792 (paper: 0.795±0.017, within CI)
- AUPRC: 0.6291 (paper: 0.622±0.010, within CI)

## Optimization Objective
- Primary: Maximize AUROC (core metric)
- Secondary: Maintain AUPRC (do not regress >5% below 0.629 = 0.598)
- Success: AUROC > 0.792 AND AUPRC >= 0.598

## Known Levers (from manifest)
1. Model architecture: layers (0-5), hidden_dim (currently 60), dropout
2. Training hyperparams: lr (0.005), ind_lambda (1.0), int_lambda (0.01), weight_decay (1e-5)
3. Penalty variant: Fast_Shap (=1 proj) vs Fast_Shap_3/Fast_Shap_5
4. Lag selection (currently 2 for Traffic)
5. Batch size (512 for CausalTime)
6. MAX_EPOCHS (2000) and early_stopping_patience (50)

## Red-Line Constraints
- Do NOT modify: eval_traffic.sh metric extraction, cls_metrics(), test data/labels, dataset splits
- Do NOT hard-code predictions or metric values
- All changes must be in /repo within container
