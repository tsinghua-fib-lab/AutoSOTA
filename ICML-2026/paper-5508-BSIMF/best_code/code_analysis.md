# Code Analysis — Paper 5508: BUQ-SIM

## Evaluation Path
- **Entry**: `ex_with_metrics.py` → `main()`
- Creates model (`ContentUncertaintyDAG`), data loaders (`get_train_val_loaders`)
- Trains via `train()` (epoch loop calling `_run_epoch` with optimizer)
- Post-training: `fit_x_pred_sigma_scale()` calibrates global sigma multiplier
- Final `validate()` with calibrated sigma scale writes metrics
- **Output**: `metrics_val.csv` (per-epoch val metrics), `metrics_train.csv`, `model_best.pt`

## Train/Inference Path
- `train.py`: `train()` orchestrates epochs, `_run_epoch()` runs one epoch (train or val)
- `_run_epoch` computes ELBO terms via `obj.compute_elbo()`, then X and Y reconstruction metrics
- For prediction (X|Y), it drops all X for some subjects when `drop_all_x_prob > 0`
- Prediction intervals computed at line 486-492 with hardcoded `z90 = 1.26`
- `fit_x_pred_sigma_scale()` calibrates sigma_scale on calibration split using correct z (1.645)
- EM update for GMM latent prior via `em_update_model_mixture_from_loader()`

## Config Path
- All hyperparameters passed via argparse in `ex_with_metrics.py`
- Model config in `ContentUncertaintyDAG.__init__` (model.py)
- ELBO weights in `obj.py:639`

## Metric Parser
- `metrics_val.csv` columns: `x_r2` (R_X_sq), `x_pred_r2` (R_X_pred_sq), `x_pred_picp90` (PICP90), `x_pred_mpiw90` (MPIW90), `x_rmse` (RMSE_X), `x_pred_rmse` (RMSE_X_pred), `y_rmse_all` (RMSE_Y)
- Scale-invariant metrics (R_sq) match paper values; scale-dependent metrics (MPIW90, RMSE) differ due to non-standardized scores

## Reusable Resources
- `/datasets/abide/abide_i_merged.pkl`: ABIDE I connectivity + behavioral scores (512 subjects, 116-region AAL, 12 scores)
- `/datasets/abide/abide_i_merged_std.pkl`: Standardized variant (not used by baseline)
- `/datasets/abide/rois_aal/`: AAL ROI files
- No pre-trained model checkpoints available

## Risky Files
- `train.py`: Core training logic; z90 bug at line 486, prediction metric computation
- `obj.py`: ELBO computation with extreme term weights at line 639
- `model.py`: Model architecture (ContentUncertaintyDAG); learnable mask prior disabled
- `ex_with_metrics.py`: Entry point; argparse defaults

## Safe Modification Targets
1. **train.py:486** — Fix z90 from 1.26 to `_z_for_central_coverage(0.90)` (1.645)
2. **obj.py:639** — Rebalance ELBO term weights
3. **ex_with_metrics.py** — Change default `drop_all_x_prob` from 0.3 to 0.5
4. **model.py:533-539** — Enable `learn_mask_prior` flag
5. **train.py:478-492** — Replace Gaussian with Student-t prediction intervals
6. **obj.py** — Add CRPS auxiliary loss
7. **train.py** — Sparsity annealing schedule
8. **model.py** — Add dropout regularization (vit_dropout, MLP dropout)

## Key Baseline Metrics
| Metric | Baseline | Paper | Direction |
|--------|----------|-------|-----------|
| R_X_sq | 0.994 | 0.976 | higher |
| R_X_pred_sq | 0.679 | 0.561 | higher |
| PICP90 | 0.809 | - | higher (target 0.90) |
| MPIW90 | 11.912 | - | lower |
| RMSE_X | 1.633 | - | lower |
| RMSE_Y | 0.336 | - | lower |

## Evaluation Command (container)
```bash
ABIDE_DATA_PKL=/datasets/abide/abide_i_merged.pkl python3 ex_with_metrics.py \
  --out_dir /repo/output/eval --batch_size 16 --epochs 100 --lr 1e-3 \
  --weight_decay 1e-5 --seed 42 --n_splits 0 --fold 0 --val_frac 0.2 \
  --num_workers 0 --z_dim 16 --u_dim -1 --num_components 4 \
  --vit_embed_dim 64 --vit_depth 2 --vit_num_heads 2 --vit_patch_size 4 \
  --rank_r 4 --mask_prior_rho 0.01 --sparse_m_lambda 0.1 \
  --sparse_m_target 0.3 --sparse_m_on content --mask_tv_lambda 0.0 \
  --mask_tv_samples 1 --drop_all_x_prob 0.3 --sigma_calib_frac 0.2 \
  --sigma_calib_every 10 --sigma_calib_max_points 200000 \
  --sigma_calib_target 0.90 --refresh_mixture_every 10
```
