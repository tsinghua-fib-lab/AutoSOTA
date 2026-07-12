# scChord Code Analysis (Paper 4532)

## Repository Overview
- **Container**: `autosota_repro_paper_4532`
- **Repo Path**: `/repo`
- **Paper**: scChord: A Probabilistic Manifold Rectification Framework for RNA-to-Protein Translation
- **Data**: GSE100866_CBMC.h5ad (CITE-seq CBMC, 1000 HVGs, 80/20 split, seed=0)

## Key Files
| File | Role |
|------|------|
| `eval.py` | Standalone evaluation script |
| `train_stage1_vae.py` | VAE training (ProteinVAE) |
| `train_stage2_cfm.py` | CFM training (RNAEncoder + FlowNet) |
| `models.py` | Model definitions (ProteinVAE, RNAEncoder, FlowNet, AdaLNBlock) |
| `metrics.py` | Metric computation (PCC, CMD, RMSE, MMD) |
| `data.py` | Data loading and preprocessing |

## Reproduction Checkpoints
- **VAE**: `outputs_stage1_gauss/vae_best.pt` (epoch 561/600, val_loss=-4.605, Gaussian, dz=32, beta_kl=0.8)
- **Flow**: `outputs_stage2_gauss/flow_best.pt` (epoch 196/200, val_loss=1.494, dc=512, flow_hidden_dim=256, n_blocks=4, lambda_cons=0.1)
- **Data info**: `outputs_stage1_gauss/data_info.pt`

## Baseline Metrics (eval cmd with cfg_scale=3.0)
- PCC-P: 0.8692, PCC-C: 0.9408, CMD-P: 0.0035, CMD-C: 0.0313, RMSE: 0.5114

## Evaluation Path
- `python3 eval.py --data_path ... --vae_path ... --flow_path ... --data_info_path ... --device cuda:0`
- Uses torchdiffeq odeint with dopri5, n_steps=50, cfg_scale=3.0
- Output: prints all metrics, saves predictions to --output_dir (defaults to outputs_eval/)

## Safe Modification Targets
1. `eval.py`: ODE solver params (rtol, atol, ode_method, n_steps, cfg_scale) — inference only
2. `train_stage2_cfm.py`: CFM loss weighting (line ~155), lambda_cons, epochs, lr schedule, contrastive loss addition
3. `train_stage1_vae.py`: beta_kl schedule, dist_type, dz, epochs, regularization
4. `models.py`: AdaLN blocks, FlowNet capacity, VAE distribution params, regularization terms

## Risky Files (do not modify)
- `metrics.py` — metric definitions (PCC, CMD, RMSE, MMD)
- `data.py` — data loading, splits
- `ComputeMetrics.ipynb` — reference notebook
- `pipeline/`, `scChord_scGPT/`, `gat_fm/`, `efficiency_benchmark/` — other components

## Key Configuration Parameters
- VAE: dz=32, hidden=[256,256], beta_kl=0.8, dist_type=Gaussian, lr=1e-3
- Flow: dc=512, rna_hidden=[1024,512], flow_hidden_dim=256, n_blocks=4, lambda_cons=0.1, p_uncond=0.15, lr=1e-3
- Training: batch_size=256, epochs_vae=600, epochs_flow=200, warmup=25, cosine decay
- Eval: cfg_scale=3.0, n_steps=50, rtol=1e-5, atol=1e-5

## Optimization Strategy
Priority order: CODE-01 (ODE solver) → CODE-04 (Flow epochs) → PARAM-01 (cfg sweep) → ALGO-01 (Weighted CFM) → ALGO-04 (KL annealing) → CODE-02 (Ensemble)

## Rollback Points
- `_baseline` tag at commit 4c73545
- Pre-iteration commits for each attempt

## Additional Ideas (generated during SOTA)

### Idea 13: Learning Rate Schedule Tuning for Flow Model
- **ID**: PARAM-02
- **Type**: PARAM
- **Priority**: P1
- **Hypothesis**: Current warmup=25 + cosine decay with lr=1e-4 is the reproduction setting. Testing higher lr (3e-4, 5e-4) with longer warmup (50 epochs) could accelerate convergence and improve final metrics.
- **Target**: train_stage2_cfm.py --lr and warmup_epochs
- **Expected Effect**: PCC-P +0.003-0.008
- **Cost**: Full retrain (200-400 epochs)

### Idea 14: Gene Mask Ratio Tuning
- **ID**: PARAM-03
- **Type**: PARAM
- **Priority**: P2
- **Hypothesis**: The mask_ratio_range of (0.2, 0.5) for RNA consistency training may be suboptimal. Testing (0.1, 0.3), (0.3, 0.6), (0.4, 0.7) could find a better regularization strength.
- **Target**: train_stage2_cfm.py train_epoch() mask_ratio_range
- **Expected Effect**: PCC-P +0.001-0.005
- **Cost**: Full retrain

### Idea 15: FlowNet Width Increase (hidden_dim from 256 to 512)
- **ID**: CODE-05
- **Type**: CODE
- **Priority**: P1
- **Hypothesis**: FlowNet hidden_dim=256 may underfit the conditional vector field. Increasing to 512 with n_blocks=6 gives the model more capacity to learn complex RNA→protein mappings.
- **Target**: train_stage2_cfm.py --flow_hidden_dim 512 --flow_n_blocks 6
- **Expected Effect**: PCC-P +0.005-0.015
- **Cost**: +60-80% training time; +100% parameters

### Idea 16: Gradient Accumulation for Effective Larger Batch
- **ID**: CODE-06
- **Type**: CODE
- **Priority**: P2
- **Hypothesis**: Batch size 512 with only 6894 training cells means very few batches per epoch (~14). Gradient accumulation (acc_steps=2 or 4) simulates larger batch training which could stabilize CFM loss.
- **Target**: train_stage2_cfm.py optimizer step logic
- **Expected Effect**: PCC-P +0.002-0.005
- **Cost**: Minimal

### Idea 17: Temperature Annealing for CFG
- **ID**: ALGO-08
- **Type**: ALGO
- **Priority**: P2
- **Hypothesis**: Static CFG scale throughout training may be suboptimal. Annealing cfg_scale from 1.0 to 3.0 over training encourages the model to first learn the base mapping before applying strong guidance.
- **Target**: train_stage2_cfm.py train_epoch(), validation, and inference
- **Expected Effect**: PCC-P +0.003-0.008
- **Cost**: Minimal code change
