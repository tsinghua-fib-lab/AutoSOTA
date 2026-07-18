# Code Analysis for Paper 3593 — SyNGLER SOTA Optimization

## Evaluation Path
- **Script**: `scripts/run_and_eval_polblogs.sh` — two-step: generate → evaluate
- **Step 1**: `experiments/real_data/run_syngler.py --dataset polblogs --r 2 --fitted_pkl data/real/polblogs/run/r=2/seed=0.pkl --output runs/syngler/polblogs_r2 --methods res --num_samples 200`
- **Step 2**: `python scripts/eval_polblogs.py --samples_dir runs/syngler/polblogs_r2/syngr/samples --ref_adj data/real/polblogs/generator/seed=0.npy --output runs/syngler/polblogs_r2/eval_results.json --device cpu`
- **Metrics parsed from**: JSON output + stdout

## Train/Inference Path
- **LSM Fitting** (Stage 1): `syngler.lsm.source.Model` — PGD with two-phase: `PGD_initialization` (fits G=ZZ^T directly) then `PGD` (refines Z, alpha)
- **SyNG-R Generation** (Stage 2): `syngler.res.bootstrap` — bootstrap-resamples (Z, alpha) rows, reconstructs adjacency via Bernoulli(sigmoid(Z @ Z.T + alpha + alpha.T + rho))
- **SyNG-D Forest** (Stage 2 alt): `syngler.diff.forest` — ForestDiffusion over X=[Z|alpha], NOT AVAILABLE (no ForestDiffusion module)
- **SyNG-D MLP** (Stage 2 alt): `syngler.diff.mlp` — GPU DDPM with residual MLP score network, AVAILABLE

## Config Path
- LSM fitting: `syngler/lsm/config/default.json` (alpha_enable, Z_enable, Z_standardize, sparsity_estimation)
- No separate config for SyNG-R generation (parameters in bootstrap.py)

## Metric Parser
- `scripts/eval_polblogs.py` computes: Tri_RMSE, Clus_RMSE, DegC_KS, Eig_MMD
- Tri/Clus: RMSE of (triangle_density, global_clustering_coefficient) across 200 samples vs reference
- DegC: KS test on pooled degree centrality vectors
- Eig: MMD on pooled absolute eigenvalues (subsampled to 5000)

## Pre-fit LSM Pickles
- `data/real/polblogs/run/r={2,3,4,5,6}/seed=0.pkl`
- Each contains: model_Z (n×r), model_alpha (n,), model_sparsity (float), converged (bool)
- Generated externally (fitting script not in repo)

## Reference Data
- `data/real/polblogs/generator/seed=0.npy`: 1222×1222 float32 adjacency, 16714 edges, density ~0.0224

## Risky Files
- `scripts/eval_polblogs.py`: DO NOT MODIFY — metric definitions
- `data/real/polblogs/generator/seed=0.npy`: DO NOT MODIFY — reference data
- `data/real/polblogs/run/r=*/seed=0.pkl`: Read-only pre-fit pickles

## Safe Modification Targets
1. `syngler/lsm/source.py`: PGD optimizer hyperparameters, initialization
2. `syngler/res/bootstrap.py`: Bootstrap seed, sampling parameters
3. `syngler/utils/source.py`: `reconstruct_adjacency` temperature/threshold
4. `syngler/diff/mlp.py`: MLP diffusion hyperparameters
5. `experiments/real_data/run_syngler.py`: num_samples, methods, seeds
6. New LSM fitting script: refit with better hyperparameters
7. `scripts/run_and_eval_polblogs.sh`: R value, num_samples, methods

## Optimization Levers (Priority-Ordered)
1. **Refit LSM with better PGD** (CODE): More iterations, tuned eta_0, better init
2. **Try r=3,4,5,6 pre-fit pickles** (PARAM): Higher latent dimension capacity
3. **SyNG-D MLP diffusion** (ALGO): Learn latent distribution via DDPM
4. **Temperature scaling** (CODE): Scale logits before sigmoid for sharper/softer edges
5. **Threshold reconstruction** (CODE): Deterministic P>0.5 instead of Bernoulli
6. **Increase num_samples** (PARAM): 500 or 1000 for more stable metrics
7. **Ensemble across r values** (ALGO): Average or select best r per metric
