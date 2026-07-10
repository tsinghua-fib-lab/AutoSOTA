# Code Analysis for Paper 3301 SOTA Optimization

## Evaluation Path
-  — loads checkpoint, creates trainer, calls , computes COV/JSD/TVD
- Metrics computed in  (coverage, jsd_histogram_2d, tvd_histogram_2d)
- Output: stdout + 

## Training Path
- Entry:  with CLI args (--trainer=DDPM, --problem=smileyface_sphere, etc.)
- Trainer:  in 
- Model:  in  (input_dim=3, hidden_dim=64, time_concat=True)

## Key Configuration
- Baseline: DDPM, sphere task, sigma=0.05, 100K samples, 200 epochs, seed=42
- Noise schedule: linear  (line 106)
- Timesteps: T=250
- Batch size: 64
- LR: 1e-3
- Optimizer: Adam with grad clip max_norm=1.0
- Checkpoint: 

## Key Findings for Optimization

### EMA (Partially Implemented)
-  method exists at line 236 but  and  are NEVER initialized
- EMA not called in  — dead code

### Noise Schedule
- Linear schedule at line 106 — can replace with cosine

### Sampling Projection
- Only projects at t=0 (final step) at line 800-817
- Multi-step projection during reverse process would reduce off-manifold drift

### Time Embedding
- Currently uses  → effective_time_embed_dim=1 (raw scalar)
- driver.py supports  but not used in baseline
- Sinusoidal embedding would give richer temporal signal

### Determinism
- eval_reproduce.py sets seed but need to verify sampling uses controlled RNG

## Safe Modification Targets
1.  — noise schedule (betas)
2.  — EMA (add init + call in train loop)
3.  — sampling projection (add periodic projection)
4.  — add EMA model init
5.  — add EMA update call, gradient logging
6.  — sinusoidal time embedding
7.  — seed control, EMA weight loading

## Risky Files (Do Not Modify)
-  — metric computation
-  — data generation
-  — metric parsing/output format (can add seed fixes but not change parsing)
