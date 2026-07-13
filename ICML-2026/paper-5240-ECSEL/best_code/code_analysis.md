# ECSEL COMPAS Code Analysis (Paper 5240 SOTA)

## Evaluation Path
- **Script**: `eval_compas.py`
- **Command**: `python3 eval_compas.py`
- **Data loading**: `load_compas()` reads from `/datasets/compas-scores-two-years.csv`
- **Split**: 80/20 train/test, stratify=y, random_state=42
- **Model**: `SignomialClassifier(internal_scaling_range=(0.01, 10.01), **BEST_CONFIG)`
- **Metric parser**: stdout prints `MetricName: value` lines; parsed by orchestrator

## Key Files
- `ecsel.py` — SignomialClassifier + adam_minibatch (core model, ~1081 lines)
- `eval_compas.py` — Evaluation script with BEST_CONFIG (~150 lines)
- `Datasets.py` — Dataset loaders
- `run_compas_reproduction.py` — Full Optuna HP search pipeline

## BEST_CONFIG (baseline)
```python
{
    "K": 3,
    "l1_strength": 0.001623616273384131,
    "batch_size": 64,
    "lr": 0.0017516096124643734,
    "num_epochs": 869,
    "patience": 50,
    "use_sigmoid": True,
    "sigmoid_threshold": 0.5,
    "random_state": 42,
    "verbose": False,
}
```

## Safe Modification Targets
1. **BEST_CONFIG in eval_compas.py**: n_restarts, validation_split, weight_decay, num_epochs, lr, batch_size, sigmoid_threshold
2. **ecsel.py adam_minibatch()**: LR schedule (cosine warmup), Adam parameters
3. **ecsel.py _loss_function()**: Class-balanced loss weights
4. **ecsel.py predict_proba()**: Temperature scaling
5. **eval_compas.py main()**: Post-hoc calibration, bagging ensemble

## Risky Files (DO NOT MODIFY)
- `compute_metrics()` — metric definitions
- `load_compas()` — data loading and preprocessing
- `SCALE_RANGE`, `RANDOM_STATE`, `TEST_SIZE` — evaluation protocol constants
- `/tools/record_score.sh` — scoring script

## Reusable Container Resources
- `/datasets/compas-scores-two-years.csv` — COMPAS dataset
- No `/models` or `/paper_data` mount

## Baseline Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 68.19 |
| F1 | 67.77 |
| Precision | 68.48 |
| Recall | 68.19 |
| MinorityRecall | 56.91 |
| Training time (s) | 9.92 |

## Known Levers
- K (1-3 terms), l1_strength (1e-4 to 1e-2), learning_rate, batch_size, num_epochs
- patience, sigmoid_threshold, n_restarts, validation_split, weight_decay
- Higher K=3 with batch_size=64, sigmoid_threshold=0.5 consistently best

## Notes
- Paper reports 3518 samples; our preprocessing gives 4589 — likely additional charge-type filtering
- Full protocol: 5-fold stratified CV + Optuna TPE 30 trials + retrain on full training + test eval
