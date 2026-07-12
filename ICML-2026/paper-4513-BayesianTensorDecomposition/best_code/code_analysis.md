# Paper 4513 — DiffBCP SOTA Preparation Repair

## Original Failure

The SOTA preparation step failed because:
1. `git` was not installed in the container
2. `apt-get install git` failed with a 502 Bad Gateway error from archive.ubuntu.com
3. Without git, the baseline commit and `_baseline` tag could not be created

## Repair Steps

1. **Re-ran `apt-get install git`** — the transient 502 error resolved; git 2.25.1 installed
2. **Created baseline commit and `_baseline` tag** — clean code state with no modifications
3. **Copied `/tools/record_score.sh`** — scoring infrastructure
4. **Created `/repo/run_eval.py`** — flexible evaluation wrapper supporting hydra overrides
5. **Created `/repo/record_result.py`** — helper for parsing and recording results
6. **Verified evaluation pipeline** — 1-image test successful (PSNR=32.76)

## Container State

- Container: `autosota_repro_paper_4513` (reusable, running)
- Base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- GPUs: 2× NVIDIA A100-SXM4-80GB
- Python: 3.10.13 (conda)
- Model: `/repo/model/ffhq_10m.pt` (FFHQ EDM checkpoint, 358 MB)
- Data: `/repo/data/ffhq_128.npy` (128 FFHQ images)

## Corrected Evaluation Command

The evaluation command from the reproduction manifest works from within the container at `/repo`:

```bash
cd /repo
python run_eval.py \
  gpu=0 seed=42 \
  +data=ffhq data.mask_name=random_mask_obs03 \
  +task=completion task.noise.sigma=0.05 \
  +model=edm_unet_adm_dps_ffhq \
  +sampler=pnp_edm sampler.mode=vp_sde \
  sampler.num_iters=100 sampler.num_burn_in_iters=40 \
  sampler.use_tau_to_anneal=true sampler.anneal_const=100.0 \
  sampler.rho=10 sampler.rho_decay_rate=0.9 sampler.rho_min=0.3 \
  sampler.decomposition.use=true sampler.decomposition.tau_beta=1e-3 \
  sampler.decomposition.init_rank=200 \
  sampler.decomposition.num_gibbs_iters=0 \
  sampler.decomposition.use_patch=true \
  sampler.decomposition.patch_size=16 sampler.decomposition.stride=8 \
  +num_val=N
```

## Baseline Verification

Single-image test confirms pipeline functional:
- PSNR (posterior mean): 32.76 dB (manifest: 32.55, within normal variation)
- SSIM: 87.78 (manifest: 88.51)
- LPIPS: 16.98 (manifest: 16.98 — exact match)

## Optimization Targets

Safe targets for optimization (no red-line violations):
1. `sampler.decomposition.num_gibbs_iters` — more Gibbs iterations for better CP mixing
2. `sampler.anneal_const`, `sampler.rho`, `sampler.rho_min` — coupling schedule parameters
3. `sampler.decomposition.tau_beta`, `sampler.decomposition.init_rank` — decomposition hyperparameters
4. CUSP CP zeta sampling fix (CODE-01) — enforces correct stick-breaking semantics
5. Hann window patch blending (ALGO-02) — smoother patch reconstruction
6. Decoupled annealing schedule (ALGO-01) — independent rho decay

## Reusable Resources

- `/repo/model/ffhq_10m.pt` — pre-trained EDM diffusion model
- `/repo/data/ffhq_128.npy` — 128 FFHQ images at 256×256
- `/repo/data/random_mask_obs03.npy` — 70% random missing pixel masks
