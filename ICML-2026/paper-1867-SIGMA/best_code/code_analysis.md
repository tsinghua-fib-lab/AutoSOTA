# SiGMA Code Analysis — Paper 1867

## Evaluation Path
- **Eval script**: `/repo/eval_electricity.sh`
- Runs 4 independent training+eval cycles for pred_len in {96, 192, 336, 720}
- Each cycle: train then test on same GPU
- Metrics parsed from stdout line: `mse:X, mae:Y`
- Average across 4 horizons computed via bash arithmetic
- **Command** (inside container): `cd /repo && bash eval_electricity.sh`

## Training Path
- `run.py` -> `Exp_Long_Term_Forecast.train()` -> `exp_long_term_forecasting.py`
- Optimizer: Adam with initial lr=0.01
- LR schedule: `adjust_learning_rate()` in `utils/tools.py`
  - `type1` (default): halving every epoch
  - `cosine`: cosine annealing from lr to 0
- Loss: `nn.MSELoss()` in `_select_criterion()` (line ~65 of exp_long_term_forecasting.py)
- EarlyStopping: patience=3, tracks val_loss via `vali()` method

## Config Path
- `script/SiGMA_ECL.sh` — baseline config for Electricity
- Key params: d_model=16, d_ff=32, e_layers=1, lr=0.01, batch_size=32, epochs=10, patience=3
- SiGMA-specific: scale_independence=0, feature_transformation=1, channel_independence=1

## Metric Parser
- `utils/metrics.py` — metric() returns (mae, mse, rmse, mape, mspe)
- MSE = np.mean((true - pred)^2) — standard implementation
- All horizons use same metric function

## Data Pipeline
- `data_provider/data_loader.py` — Dataset_Custom class
- 70/10/20 train/val/test split
- StandardScaler fitted on train data only
- Electricity CSV at `/repo/dataset/electricity/electricity.csv`

## Model Architecture
- `models/SiGMA.py` — SiGMA model
- MultiScaleGenerator: learnable discrete Gaussian kernel for decomposition
- ForecastingModule: MLP-based processing of decomposed scales
- With e_layers=1: single decomposition -> smooth + residual
- With channel_independence=1: treats each of 321 features independently

## Safe Modification Targets
1. `utils/tools.py` — LR scheduler (add warmup, modify cosine)
2. `exp/exp_long_term_forecasting.py` — criterion selection, gradient clipping, data augmentation
3. `script/SiGMA_ECL.sh` — all hyperparameters (d_model, d_ff, e_layers, lr, epochs, etc.)
4. `run.py` — CLI args (extend with new flags if needed)
5. `eval_electricity.sh` — only if fixing eval issues (add new args to pass through)

## Risky Files (do NOT modify)
- `utils/metrics.py` — metric definitions (MSE, MAE computation)
- `data_provider/data_loader.py` — data splits, scaling
- Dataset files in `dataset/electricity/`

## Reusable Resources
- No pre-downloaded checkpoints or weights
- Dataset: `/repo/dataset/electricity/electricity.csv` (146MB)
- Raw data cached at: `/autosota_cache/tmp/electricity_raw/`
