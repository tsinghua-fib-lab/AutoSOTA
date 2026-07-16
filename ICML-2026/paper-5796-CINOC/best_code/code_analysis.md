# CINOC SOTA Preparation Repair - Code Analysis

## Preparation Failure

The orchestrator failed because:
1. `git` was not installed in container `autosota_repro_paper_5796`
2. `apt-get install git` failed with 502 Bad Gateway from `archive.ubuntu.com` (proxy issue)
3. The container could not initialize git repo, create baseline commit, or tag `_baseline`

## Repair Applied

1. Changed apt sources from `archive.ubuntu.com` to `mirrors.aliyun.com` in `/etc/apt/sources.list`
2. Fixed broken packages from partial dpkg install with `apt-get --fix-broken install`
3. Installed git successfully via apt after mirror change
4. Copied `record_score.sh` to `/tools/record_score.sh`
5. Initialized git repo, committed baseline, tagged `_baseline`
6. Created `/autosota_artifacts/paper-5796/sota/scores.jsonl`

## Corrected In-Container Evaluation Command

```bash
bash /repo/eval_fkpp_cinoc.sh
```

This script runs:
```bash
cd /repo/examples/fkpp1d/decentralized
python3 bench3.py
```

Output format: Table with "Method | Mean Track Error | 2-Sigma" columns.
CINOC row: `CINOC           | 0.000046             | ±0.000076`

## Baseline Verification

- **Reproduction run:** CINOC Tracking MSE = 4.6e-5, Uncontrolled = 0.102613
- **Paper value:** CINOC Tracking MSE = 4.6e-5 ± 7.5e-5
- **Match:** Exact
- **Pre-trained model:** `/repo/examples/fkpp1d/decentralized/decentralized_params.msgpack` (45KB)
- **Model architecture:** `DecentralizedControlNet(features=(64, 64))`
- **Evaluation:** 50 evaluation episodes, each 300 timesteps, 100 grid points, 20 agents

## Reusable Resources

- Pre-trained CINOC weights: `/repo/examples/fkpp1d/decentralized/decentralized_params.msgpack`
- No external datasets needed (on-the-fly GRF generation)
- No `/paper_data` mount

## Safe Optimization Targets

### Training Pipeline
- Training script: `/repo/examples/fkpp1d/decentralized/train.py`
- Policy model: `/repo/tesseracts/solverFKPP_decentralized/models/policy.py`
- Dynamics: `/repo/examples/fkpp1d/decentralized/dynamics_dual.py`
- Data generation: `/repo/examples/fkpp1d/decentralized/data_utils.py`
- Evaluation: `/repo/examples/fkpp1d/decentralized/bench3.py`

### Key Parameters
- Architecture: `features=(64, 64)` for branch net, trunk net `[32, 32]`
- Loss weights: track=5.0, effort=0.001, bound=100.0, coll=1.0, accel=0.1
- Training: 500 epochs, batch_size=32, n_pde=100, n_agents=20
- Learning rate: exponential_decay(1e-3, 2000, 0.5)
- Optimizer: Adam with clip_by_global_norm(1.0)
- Data: 5000 GRF samples, init length_scale=0.2, target length_scale=0.4
- Dynamics: nu=0.005 (diffusion), rho=3.0 (growth rate)
- Noise: noise_u=0.0, noise_z=0.0 (disabled by default)

### Implementation Notes
- Training takes ~500 epochs, roughly 5-10 minutes per full training run on A100
- Evaluation is fast (~30 seconds for 50 episodes)
- Changes to model architecture require retraining from scratch
- The loss function and optimizer are the safest modification points
- Noise parameters are already supported in `unroll_controlled()` via `noise_u`, `noise_z` kwargs

## Optimization Strategy

Priority order for iteration attempts:
1. Loss function modifications (ALGO-01, ALGO-03) - no architecture change, quick to test
2. Training hyperparameters (PARAM-01, PARAM-02, ALGO-05) - no code structure change
3. Architecture modifications (ALGO-02, ALGO-04) - requires retraining
4. Data augmentation (CODE-02) - one-time data generation change
5. Multi-seed selection (CODE-03) - training procedure change
