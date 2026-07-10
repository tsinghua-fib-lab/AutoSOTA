# Code Analysis for Paper 3112 SOTA Optimization

## Evaluation Path
- Main script: `/repo/eval_breastCancer.py`
- Entry: `python3 eval_breastCancer.py`
- Uses the breastCancer dataset (N=94, 1 binary feature "treat") from icensBKL R package
- 5-fold CV with seed=0, shuffle=True

## Train/Inference Paths
- **Model**: MLP2Layer class (in eval_breastCancer.py) — 3-layer MLP with dropout=0.5, ReLU, Softmax output
- **Training**: `train_one_fold()` — uses NegativeLogLikelihoodInterval loss, Adam optimizer, lr=0.01, 5000 epochs
- **Inference**: `evaluate_fold()` — forward pass, cumsum, CumulativeDist, compute SIC-Log + IC-Cal

## Configuration
- N_EPOCHS=5000, BATCH_SIZE=128, HIDDEN_SIZE=32, LEARNING_RATE=0.01
- Early stopping: patience=1000 (stops if no loss improvement for 1000 epochs)
- DEVICE: cuda:0 if available

## Metric Parser
- SIC-Log: `metric_cdf.negative_log_likelihood_interval(dist, lb_test, ub_test).mean()`
- IC-Cal: `metric_quantile.ic_calibration(dist, lb_test, ub_test, p=2.0)`
- Output: `print("SIC-Log: {:.4f} +/- {:.4f}".format(...))` and `print("IC-Cal:  {:.6f} +/- {:.6f}".format(...))`

## Dataset
- Loaded from `/autosota_cache/tmp/icensBKL/icensBKL/data/breastCancer.rda`
- 94 samples, 1 binary covariate (`treat`), interval-censored outcomes
- No /paper_data mount

## Safe Modification Targets
- `eval_breastCancer.py`: model architecture, optimizer, LR schedule, training loop
- `src/cenreg/pytorch/mlp.py`: MLP architecture parameters
- `src/cenreg/utils.py`: bin boundary creation
- `src/cenreg/pytorch/loss_cdf.py`: loss computation

## Red-Line (DO NOT MODIFY)
- `src/cenreg/metric/cdf.py`: metric definitions
- `src/cenreg/metric/quantile.py`: IC-Cal computation
- `src/cenreg/pytorch/distribution.py`: CDF distribution logic
- `src/cenreg/distribution/`: distribution logic
- Dataset loading/processing logic (test data split)
- Evaluation metrics definition

## Baseline
- Iter 0: SIC-Log=2.0068, IC-Cal=0.008259
- Paper: SIC-Log=1.5343, IC-Cal=0.008892
- Target: SIC-Log < 1.6914 (paper CI upper bound), IC-Cal ≤ 0.014304
