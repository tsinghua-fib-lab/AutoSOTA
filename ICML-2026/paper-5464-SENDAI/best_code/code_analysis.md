# SENDAI Jr. Code Analysis

## Evaluation Path
- Entry point: /repo/model/run_multi.py (multi-run wrapper)
- Config: Hardcoded in run_multi.py main(): hidden_size=32, decoder_layers=[256,256], n_sensors=64, lags=5, etc.
- Pipeline: load_data() -> fix_bad_frames() -> select_sensors() -> create_time_delay_dataset() -> SHRED training -> DASHRED training -> evaluation
- Model: SHRED (LSTM encoder + MLP decoder) -> DASHRED (SHRED weights + latent transform + GAN alignment)

## Key Files
- model/SENDAI/models.py — SHRED/DASHRED architectures (safe to modify decoder, LSTM, attention)
- model/SENDAI/training.py — train_shred(), train_dashred() (safe to modify loss functions, optimization)
- model/SENDAI/data.py — Data loading, dataset creation (DO NOT MODIFY — eval data/splits)
- model/SENDAI/metrics.py — SSIM, RMSE computation (DO NOT MODIFY — metric definitions)
- model/SENDAI/utils.py — Utilities: device, timing (safe to modify)
- model/run_multi.py — Multi-run evaluation (safe to modify config params only)

## Metric Parser
- stdout: "Seed N - Full RMSE: X.XXXX, Full SSIM: X.XXXX"
- Output file: multi_run_summary.json in output-dir with ssim.mean, rmse.mean

## Training Flow
1. Stage 1 (SHRED): LSTM encoder + MLP decoder trained on simulation data with MSE loss
2. Stage 2 (DA-SHRED):
   - 2a: GAN latent space alignment (sim->real)
   - 2b: Fine-tune with full state supervision on simulation + small transform regularization

## Config Levers (modifiable without red-line)
- hidden_size (32 -> 64)
- decoder_layers ([256,256] -> [512,512] or [512,256] etc.)
- shred_epochs, dashred_epochs, gan_epochs
- batch_size, lr
- n_sensors, lags, sensor_strategy
- dropout (hardcoded in model init, not in config)

## Risky Files (DO NOT MODIFY)
- metrics.py — metric computation
- data.py — data splits, scaling
- Test data at /repo/data/western_us/processed/

## Architecture Details
- SHRED.forward(): LSTM encode -> last hidden state -> MLP decoder -> output
- DASHRED.forward(): LSTM encode -> latent z -> z_t = z + scale * transform(z) -> decoder(z_t)
- SHRED decoder: [hidden_size] -> [256] -> [256] -> [state_dim] with LayerNorm+ReLU+Dropout
- No SSIM in loss function — MSE only
- No temporal attention — only last LSTM hidden state used
- No spatial awareness in decoder — flat MLP

## Container Info
- GPU: CUDA available (devices 6,7)
- Data: pre-downloaded at /repo/data/western_us/processed/
- Cache: /autosota_cache, /datasets, /models mounted
- No external data downloads needed
