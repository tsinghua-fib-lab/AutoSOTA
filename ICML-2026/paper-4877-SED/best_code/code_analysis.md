# Code Analysis for Paper 4877: Skipping the Zeros in Diffusion Models

## 1. Evaluation Path
- Script: /repo/evaluate_v2.py
- Entry point: main() function
- Flow: Load real data from h5ad, load SVAE + SED checkpoints, generate N samples, compute SCC/MMD/sparsity
- Key CLI args: --svae_ckpt, --sed_ckpt, --n_samples, --batch_size, --output

## 2. Train/Inference Path
- Reduced training: /repo/train_reduced.py (20K SVAE + 100K SED)
- Full training: /repo/train_pipeline.py (100K SVAE + 500K SED)
- SVAE-only: /repo/train_svae.py
- Lightning CLI: /repo/sed/svae_main.py, /repo/sed/sed_main.py
- SAVAE: Transformer-based VAE (d_model=256, 3 layers, 4 heads), Adam lr=1.0 + Noam-style warmup
- SED: DDPM on SAVAE latents, Adam lr=1e-4 (fixed), frozen SVAE encoder
- Inference: autoregressive SVAE decode from diffusion-generated latent

## 3. Config Path
- /repo/configs/sed/sed.yaml, sed_unet_*.yaml
- /repo/configs/vae/svae_medium.yaml, svae_small/large/xlarge.yaml
- /repo/configs/data/sparse_scrna.yaml
- /repo/sed_configs/sed_scrna_sedp.yaml (override)

## 4. Metric Parser (in evaluate_v2.py main())
- SCC: scipy.stats.spearmanr(real_mean_expression, gen_mean_expression) on log1p-normalized data
- MMD: RBF kernel MMD in 20-dim PCA space, subsampled to 5000 cells each
- Sparsity: (size - nonzero) / size
- Output: JSON with SCC, MMD, gen_sparsity, real_sparsity

## 5. Reusable Resources
- Dataset: /tmp/habermann_human_lung_pf.h5ad (also /datasets/)
- Baseline checkpoints: /repo/svae_output/svae_20k.pth, /repo/sed_output/sed_100k.pth
- Real data validation set used as reference for metrics

## 6. Risky Files (DO NOT MODIFY)
- evaluate_v2.py metric computation (SCC, MMD, sparsity formulas)
- /tools/record_score.sh
- Dataset: /tmp/habermann_human_lung_pf.h5ad
- Test split / data module setup in sed/data/scrna.py

## 7. Safe Modification Targets

### Eval-only (no retraining):
- evaluate_v2.py: Diffusion init params (use_ddim, timesteps, noise_schedule)
- evaluate_v2.py: Add --use_ddim, --ddim_steps, --time_difference CLI args
- Only sampling process changes, not metric computation

### Training (require retraining):
- train_pipeline.py / train_reduced.py: steps, batch_size, LR schedule, grad clip
- sed/models/diffusion/diffusion.py: forward pass, noise schedules, sampling
- sed/models/modules/unet.py: MLPUnet architecture, conditioning
- sed/models/callbacks/weight_averaging.py: EMA schedule
- sed/models/vae/svae.py: KL beta, architecture

## 8. Current Baseline
- SCC: 0.9503 (paper: 0.82), MMD: 0.1257 (paper: 0.54)
- Gen sparsity: 0.9686, Real sparsity: 0.9741
- Training: 20K SAVAE + 100K SED (partial budget)

## 9. GPU Resources
- 2x NVIDIA A100-SXM4-80GB (80GB each)
- Container eval uses GPU 0 (CUDA_VISIBLE_DEVICES=0)

## 10. Environment
- PyTorch 2.1.0, CUDA 12.1, Lightning 2.6.5, NumPy 1.26.4
- Wandb disabled (WANDB_MODE=disabled)
