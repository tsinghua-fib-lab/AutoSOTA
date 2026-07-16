# Code Analysis for RFM Optimization (Paper 2787)

## Evaluation Path
- `configs/default.yaml` → `main.py` → `src/trainer.py` → `src/agent/rfm.py`
- Eval command: `MUJOCO_GL=osmesa python3 main.py --override env.name=walker-run --override logger.debug=true --override logger.log_dir=/autosota_cache/logs --override logger.wandb_dir=/autosota_cache/wandb --override seed=0`
- Metric parsing: wall-clock `real` time from bash `time` wrapper, final_episode_reward from stdout `episode_reward`
- Output format: Logger writes to JSONL in log_dir

## Train/Inference Path
- `Trainer.train()` loops max_steps (250K), each step:
  1. Sample action via flow integration (10 steps Euler)
  2. Environment step
  3. If buffer warm, `agent.update()`:
     a. Sample on-policy actions (flow integration, 32 particles)
     b. Sample next actions (flow integration, 32 particles)
     c. Critic update (2 Q-networks)
     d. Posterior estimation (100 MC samples × Q forward pass)
     e. Actor update (flow matching loss)
  4. Every 10K steps: evaluate with 25 envs, 32 particles deterministic

## Config Path
- Base: `configs/default.yaml`
- Overlay: `configs/rfm.yaml` (reward_scale=1.0, max_grad_norm=10.0)
- CLI overrides via `--override key=value`

## Metric Parser
- training_time_minutes: wall-clock time from `time` wrapper
- final_episode_reward: last eval log entry

## Reusable Resources
- JAX compilation cache: `/autosota_cache/jax_cache`
- Log directory: `/autosota_cache/logs`
- No pre-downloaded data; dm_control fetches models automatically

## Risky Files (do not modify)
- `/repo/src/envs/` - environment definitions
- `/repo/src/utils/logger.py` - metric logging
- `/repo/src/components/buffer.py` - replay buffer

## Safe Modification Targets
- `configs/default.yaml` - hyperparameters
- `configs/rfm.yaml` - RFM-specific overrides
- `src/agent/rfm.py` - RFM algorithm implementation
- `src/trainer.py` - training loop
- `main.py` - entry point and JIT warmup

## Baseline Metrics
- training_time_minutes: 26.33 (1580s)
- final_episode_reward: 725.8 at 250K steps
- Training speed: ~185 it/s steady-state on A100

## Key Bottlenecks (ordered by compute cost)
1. Posterior estimation: 100 MC samples × Q forward pass per update (~50% of compute)
2. Flow integration: 10×2=20 forward passes per update (~30% of compute)
3. Particle sampling: 32 particles for action selection (~15% of compute)
4. Evaluation: 25 envs × episode_length × 32 particles × 10 integration steps every 10K steps
