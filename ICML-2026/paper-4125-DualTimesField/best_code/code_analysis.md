# Code Analysis — DualTimesField (Paper 4125)

## Evaluation Path
- **Entry point**: `/repo/eval_reproduction.py`
- **Command**: `python eval_reproduction.py --dataset ETTh1 --epochs 300 --seed 42`
- **Output format**: JSON block after `REPRODUCTION RESULT` header with `test_mse`, `test_mae`
- **Key flow**: Train model → validate → select best → evaluate on test set → print JSON
- **Training**: AdamW optimizer, CosineAnnealingLR scheduler, 300 epochs, batch_size=32
- **Best model selection**: best_val_loss tracking during training

## Core Source Files
| File | Role |
|---|---|
| `src/dualfield/core.py` | Main model: FourierFeatures, ContinuousTimeField, GaborAtom, DiscreteGeometricField, ScaleAnnealingScheduler, DualTimesField, compute_loss |
| `src/dualfield/trainer.py` | DualFieldTrainer (alternate training path, not used by eval_reproduction.py) |
| `src/dualfield/losses.py` | Alternative loss classes (DualFieldLoss, GroupLassoLoss, FrequencyDomainLoss, MultiScaleLoss, CompositeLoss) — NOT used by eval_reproduction.py |
| `src/dualfield/metrics.py` | Metrics + ReconstructionEvaluator (has known bug: calls `self.model(t)` instead of `self.model(x, t)`) |
| `reconstruction/datasets.py` | TimeSeriesDataset, MultiDatasetLoader — data loading for ETTh1 |
| `reconstruction/train.py` | Original training entry point |
| `eval_reproduction.py` | Custom SOTA evaluation script |

## Known Bugs (Pre-SOTA Analysis)
1. **`core.py:29-31`**: `frequency_constraint_loss()` always returns zero (sigmoid×cutoff < cutoff → relu=0). Dead code.
2. **`core.py:320`**: `freq_constraint_loss` computed but never added to `total_loss`.
3. **`core.py:290`**: `ScaleAnnealingScheduler(total_epochs=1000)` hardcoded; training runs 300 epochs so annealing never starts.
4. **`metrics.py:85,112`**: Calls `self.model(t)` instead of `self.model(x, t)` — broken evaluation path.
5. **`trainer.py:58`**: Calls `self.model.ctf.compute_smoothness_loss(t)` instead of `(x, t)`.

## Config / Hyperparameters
- Model: num_frequencies=16, hidden_dim=64, num_layers=3, freq_cutoff=8.0, num_atoms=16, sigma_base=0.05
- Loss weights: sparsity_lambda=0.001, smoothness_lambda=0.001, DGF weight=0.1 (hardcoded)
- Training: lr=1e-3, weight_decay=1e-4, batch_size=32, epochs=300, grad_clip=1.0
- Scale annealing: total_epochs=1000 (BUG), warmup_ratio=0.3
- Parameters: ~29,534

## Safe Modification Targets
- `src/dualfield/core.py`: FourierFeatures.__init__, frequency initialization, compute_loss (loss terms), ScaleAnnealingScheduler params, GaborAtom regularization
- `eval_reproduction.py`: Passing additional params to DualTimesField (e.g., total_epochs)
- NOT safe: metric definitions, test data/splits, normalization, output parsing

## Red-Line Boundaries
- Evaluation command must remain: `python eval_reproduction.py --dataset ETTh1 --epochs 300 --seed 42`
- Metrics (MSE, MAE) computed by F.mse_loss, F.l1_loss on test data — unchanged
- Test split: last 20% of ETTh1 — unchanged
- No hard-coded metric values
