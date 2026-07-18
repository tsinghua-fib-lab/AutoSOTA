# Code Analysis for Paper 3392 SOTA Optimization

## Evaluation Path
- **Script**: `/repo/run_reproduction.py` (7030 bytes)
- **Command**: `source /opt/conda/etc/profile.d/conda.sh && conda activate scoremd && cd /repo && XLA_PYTHON_CLIENT_PREALLOCATE=false python3 run_reproduction.py`
- **Output**: JSON at `/repo/bba_mep_results.json` with key `peak_energy_converged_mep_kbT`
- **Timeout**: 60 minutes (actual ~30 seconds)

## Key Configuration (run_reproduction.py)
- Line 34: `N_IMAGES = 51`
- Line 35: `N_MEP_ITERS = 3000`
- Line 36: `MEP_STEP_SIZE = 0.001`
- Line 37: `EVAL_T = 0.05` (ScoreMD time: 0=data, 1=noise)
- Line 131-137: Endpoint selection: frame 0 vs farthest among first 500
- Line 146-155: MEP loop: score gradient step + uniform reparam each iter
- Line 115-128: `uniform_reparam`: JAX linear arc-length reparam
- Line 103-112: JIT-compiled `score_fn` and `energy_fn` at fixed `EVAL_T`
- Line 163: Peak energy = max of negative log_q along string

## Metric Computation
- `energy_fn` uses `model.log_q` (log probability under ScoreMD potential)
- Peak energy = max of `-log_q` along string (higher log_q = lower energy = better)
- Reported as `peak_energy_converged_mep_kbT` in JSON output

## Model and Data
- **Checkpoint**: `/models/scoremd_models/models/bba/both/model/1800` (EMA params, Orbax)
- **Data**: BBA D.E. Shaw trajectories (56K frames, 28 alpha-carbons)
  - `/repo/storage/deshaw/bba-0_ca.h5`, `bba-1_ca.h5`
  - `/repo/storage/deshaw/bba.pdb`, `bba_tica.pic`
- **Model**: ScoreMD MixtureOfModels (transformer_large_score x2 + transformer_large_potential)
  - hidden_nf=128, n_layers=3, feature_embedding_dim=16
  - T ranges: m1 [0.6,1.0], m2 [0.1,0.6], m3 [0.0,0.1] (potential)

## Safe Modification Targets
1. `EVAL_T` (line 37): Sweep over candidate values
2. `N_MEP_ITERS` (line 35): Increase for better convergence
3. `MEP_STEP_SIZE` (line 36): Tune step size
4. `N_IMAGES` (line 34): Tune number of string images
5. Endpoint selection (lines 131-137): Try different pairs
6. `uniform_reparam` (lines 115-128): Add endpoint preservation, cubic spline
7. MEP loop (lines 146-155): Adaptive step, two-phase schedule, reparam frequency
8. Energy computation (lines 157-163): Ensemble across eval_t values

## Risky Files (do not modify)
- ScoreMD source at `/scoremd/src/`: external library
- Checkpoint at `/models/scoremd_models/`: immutable weights
- Dataset files in `/repo/storage/deshaw/`: immutable data
- Model architecture (lines 68-102): must match checkpoint

## GPU Resources
- 2x NVIDIA A100-SXM4-80GB (80GB each)
- JAX preallocation disabled (XLA_PYTHON_CLIENT_PREALLOCATE=false)
