# DASBS Code Analysis

## Repository Structure
- `configs/ising4.yaml` — Main training config (Hydra/OmegaConf)
- `train.py` — Entry point for training, multi-GPU support
- `utils_train.py` — All training logic: loss functions, buffer management, sampling, annealing
- `sampling.py` — Tau-leaping sampler for CTMC
- `noise.py` — Noise schedule (const, log_linear)
- `model/sit_rope.py` — RopeSiT model architecture
- `model/__init__.py` — get_model dispatcher
- `model/ema.py` — EMA utilities
- `eval.py` — Evaluation script
- `eval_metrics.py` — Detailed evaluation script
- `energy/ising.py` — Ising model energy, magnetization, correlations
- `energy/mcmc.py` — MCMC (Wolff cluster) sampler
- `utils.py` — Logger, ESS, energy W2 distance

## Actual Training Config (from .hydra_0705072601/config.yaml)
- L=24, seq_len=576, beta=0.28
- model: ropeSiT, hidden_size=32, n_blocks=6, n_heads=4, dtype=bfloat16
- batch_size=64, buffer_size=512, resample_freq=20
- num_stages=5, controller_steps=400, corrector_steps=200
- loss: controller=tm, corrector=tm, breg_func: tlogt for both
- noise: log_linear, alpha=0.5, gamma=1.0
- sampling_steps=200, num_repl=8
- lr=0.001, clip_grad_norm=0
- EMA decay=0.9999
- anneal=false, use_log_rnd=false
- t_weight_func: unif for both
- seed=0

## Evaluation
- Command: `python3 eval.py --ckpt outputs/ising_L24_beta0.28/ckpt5.pth --gt_samples gt_samples_ising_L24_beta0.28.npy --n_samples 5000 --batch_size 512 --steps 200`
- Uses EMA weights from checkpoint
- Output format: `DeltaMag  = <float>`, `DeltaCorr = <float>`, `EW2   = <float>`
- Metrics computed against 51,200 MCMC Wolff reference samples

## Safe Modification Targets
- `configs/ising4.yaml` — All hyperparameters (model, training, noise, optim)
- `train.py` — LR scheduler addition (lines 77-78 area)
- `utils_train.py` — Loss functions, path weights, time weights (all config-controlled)

## Red-line Boundaries
- eval.py — DO NOT modify (metric computation, data loading)
- eval_metrics.py — DO NOT modify
- energy/ — DO NOT modify (energy functions, MCMC)
- noise.py — Config-controlled, safe to use as-is
- sampling.py — Core algorithm, safe to use as-is
- gt_samples_ising_L24_beta0.28.npy — DO NOT modify

## Manifest Command Recovery
- eval_command in manifest is correct in-container command
- No host-side commands needed
- GPUs 6,7 on host mapped to 0,1 in container

## Key Observations
1. sampling_steps=200 (not 25 as config default suggests)
2. n_blocks=6 (not 4 as config default suggests)
3. batch_size=64 (not 512 as config default suggests)
4. Training time: ~13 min for 3000 total steps (5 stages)
5. EW2=15.16 vs paper's ~5.4 — largest gap
6. DeltaCorr=0.00239 vs paper's 0.0023 — already very close
7. DeltaMag=0.00034 vs paper's 0.015 — already much better
