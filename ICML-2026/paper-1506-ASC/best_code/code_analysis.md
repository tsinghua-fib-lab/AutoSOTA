# Paper 1506 SOTA Preparation Repair — Code Analysis

## Preparation Failure Diagnosis

**Root cause:** The `reproduce.py` script trains 4 models sequentially (k=3 ambiguous, k=2 ambiguous, k=1 strategic baseline, naive hinge), each for 500 epochs. Total runtime exceeds 60 minutes. The SOTA preparation pipeline applied a 960-second (16-minute) timeout, which killed the script during the second model (k=2) training.

**Evidence:**
- k=3 model completed successfully at epoch 500/500 (accuracy 99.5%, saved in `reproduction_results.json`)
- Script was killed with `[TIMEOUT] command exceeded 960s` during k=2 training at epoch ~6/500
- The k=3 results in `reproduction_results.json` match the manifest baseline (99.5% accuracy, paper: 99.2%)

## Repair Applied

**Corrected evaluation command:** `python3 run_separable_exp.py --k 3 --epochs 500 --seed 1`

This runs only the target k=3 ambiguous strategic classifier experiment. Runtime: ~15.5 minutes on A100-80GB, well within the 20-minute evaluation timeout.

**Verification:**
- Baseline run confirmed: accuracy=99.50%, neg_recall=100.0%, avg_burden_to_AOP=0.792
- All metrics match the reproduction baseline within normal numerical noise

## Repository Structure

```
/repo/
├── model.py              # StrategicClassifierFiniteSet, StrategicClassifierForWarmup
├── model_utils.py         # Loss functions: AmbiguousStrategicHingeLoss, HingeLoss, etc.
├── trainer.py             # StrategicTrainer with fit/predict
├── dataloader.py          # SCPIDataset, create_dataloaders
├── data_utils.py          # Data generation utilities
├── run_separable_exp.py   # Single-experiment runner (CLI args for k, lr, epochs, etc.)
├── reproduce.py           # Full reproduction (all 4 models)
└── synthetic_exp.ipynb    # Original notebook
```

## Key Model Details

- **Model:** StrategicClassifierFiniteSet with k=3 classifiers (1 chosen + 2 disguise)
- **Data:** Synthetic 2D separable data, 1000 points, 50:10:40 split
- **Loss:** AmbiguousStrategicHingeLoss with cvxpylayers projections
- **Optimizer:** Adam(lr=0.006, weight_decay=0.001)
- **Regularization:** reg_classifier=0.001, reg_auxiliary=0.001
- **Temperature:** tau=0.15 (soft-min for ambiguity aggregation)
- **Disguise init:** dev=0.4 (perturbation scale from w_chosen)

## Known Code Issues

1. **trainer.py:103** — `early_stopping = None` unconditionally disables early stopping parameter
2. **run_separable_exp.py:107** — Adam weight_decay + manual reg_classifier double-counts L2 regularization
3. **model.py calc_raw_movement** — Sample-by-sample cvxpylayers projection (slow, ~2s/epoch for 1000 samples)
4. **dev=0.4** hardcoded as default in both model constructor and experiment script
5. **tau=0.15** fixed throughout training (no annealing)

## Safe Optimization Targets

All optimization targets modify only training hyperparameters or model architecture within the same evaluation protocol. No test data, metric definitions, or benchmark outputs are changed.

## Reusable Resources

- No external datasets needed (synthetic data generation)
- No pre-downloaded models or checkpoints
- Container has cvxpylayers, cvxpy, PyTorch 2.1.0, CUDA 12.1
