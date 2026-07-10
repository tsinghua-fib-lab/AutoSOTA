# Code Analysis — Paper 3623 (Prime Attention)

## Key files

| File | Role |
|------|------|
| `eval.sh` | Runs 4 training+eval passes (pred_len=96,192,336,720), averages MSE/MAE |
| `run.py` | Main entry point, arg parsing, calls Exp_Forecast |
| `experiments/exp_forecasting.py` | Training loop, loss functions, FreDFLoss class |
| `models/Transformer.py` | iTransformer with PrimeFilterAttention |
| `layers/attention_layers.py` | StandardAttention, PrimeFilterAttention, DifferentialAttention |
| `prime/PrimeFilters.py` | Frequency-domain filter feature computation |
| `prime/IdentityDropout.py` | Dropout that drops to 1.0 (identity) instead of 0.0 |
| `utils/tools.py` | adjust_learning_rate (step decay), EarlyStopping |
| `scripts/Weather.sh` | Original paper script (not used by eval.sh directly) |

## Important discovery: LR scheduler

The code defines `_select_scheduler()` creating `CosineAnnealingLR` but **NEVER instantiates or calls it**.
The actual LR schedule is `adjust_learning_rate()` in `utils/tools.py`, which uses `lradj='type1'`:
halving every epoch. With lr=0.0005, after 5 epochs LR is ~3e-5 — extremely aggressive.

Epoch 1: 0.0005 → Epoch 2: 0.00025 → Epoch 3: 0.000125 → Epoch 4: 6.25e-5 → Epoch 5: 3.125e-5

## Eval protocol
- `bash eval.sh` runs 4 independent training runs (pred_len=96,192,336,720)
- Each run: 10 epochs, batch_size=128, lr=0.0005, d_model=128, n_layers=3, dropout=0.4, seed=2025
- Metrics: average MSE and MAE across 4 prediction lengths
- Parse: grep "Prime Attention - Weather Average:" for final MSE/MAE

## Safe modification targets
- `run.py`: add new CLI args (grad clipping, lradj options, horizon weighting, l1_lambda)
- `experiments/exp_forecasting.py`: training loop modifications (grad clipping, loss weighting, EMA, scheduler)
- `models/Transformer.py`: architecture changes (learnable temperature)
- `layers/attention_layers.py`: attention mechanism changes (temperature, memory bank)
- `prime/PrimeFilters.py`: filter computation changes, L1 target
- `scripts/Weather.sh`: hyperparameter variants (not used directly by eval.sh)
- `eval.sh`: modified flag overrides (preserving eval protocol)

## Red-line boundaries
- Do NOT change: eval.sh metric parsing, data splits, test data, denormalization, metric definitions
- CAN change: training hyperparameters, loss functions, architecture internals, optimization procedure
