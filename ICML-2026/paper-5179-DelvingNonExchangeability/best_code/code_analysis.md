# Code Analysis for Paper 5179 SOTA Optimization

## Evaluation Path
- Command: python3 run_scale_seeds.py --src-dir logs/base/la/stgnn/2026-07-12/19-39-45 --seeds 0 1 2 3 4 --output scale_results.json
- Reads Stage 1 residuals
- Produces scale_results.json with per-seed per-alpha metrics
- Key metrics for alpha=0.1: observed_coverage, pi_width, winkler

## Training Path
- experiments/run_scale.py -> run_experiment() -> Trainer.fit -> Trainer.test
- Config: conformal_model/scale/config/la.py
- Model: SpectralExchangeableQuantileNet from spec_decoupled_components.py
- Predictor: scalePredictor from basicts/runners/scale_predictor.py
- Loss: QuantileLoss from conformal_model/scale/arch/losses.py

## Key Architecture
- SGWT decoupling: low-freq (SpatioTemporal backbone) + high-freq (std+RMS stats -> MLP)
- Gating: sigmoid(Conv2d(low_features)) * high_out
- Quantiles: 39 levels [0.025, 0.050, ..., 0.975]
- Output: (B, H=1, N=207, Q=39) quantile predictions

## Config
- lr=8e-4, weight_decay=1e-4, batch_size=16, epochs=30, patience=3
- crossing_penalty=15.0, n_scales=4, kernel_type=mexican_hat
- val_len=0.1, grad_clip_val=5
- backbone: node_dim=128, embed_dim=128, num_layer=5

## Safe Modification Targets
- conformal_model/scale/config/la.py (hyperparameters)
- conformal_model/scale/arch/losses.py (loss function)
- conformal_model/scale/arch/spec_decoupled_components.py (architecture)
- experiments/run_scale.py (training loop, optimizer)

## Risky Files (do not modify)
- Test data/splits (handled by TSL MetrLA dataset)
- Metric computation (basicts/metrics/torch_metrics/*)
- run_scale_seeds.py (evaluation wrapper, unless seed determinism)

## Paper Data
- No pre-downloaded data. METR-LA downloads automatically via TSL.
- Stage 1 residuals at logs/base/la/stgnn/2026-07-12/19-39-45/

## Key Observations
1. run_scale.py uses torch.optim.Adam, not AdamW (CODE-1 fix)
2. _high_frequency_stats only uses std+RMS (ALGO-2 opportunity)
3. QuantileLoss has crossing_penalty but no smoothness term (ALGO-6)
4. val_len=0.1 holds out 10% of calibration data (CODE-3)
5. SEED is passed but deterministic settings not enforced (CODE-4)
