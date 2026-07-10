# SOTA Preparation Repair — Paper 2787 (Reverse Flow Matching)

## Original Failure

The preparation failed during git installation inside the container `autosota_repro_paper_2787`:
```
dpkg: error processing archive .../git_1%3a2.25.1-1ubuntu3.14_amd64.deb (--unpack):
 cannot copy extracted data for ./usr/lib/git-core/git-http-fetch to /usr/lib/git-core/git-http-fetch.dpkg-new: failed to write (No space left on device)
```

**Root cause**: Docker overlay filesystem was at 200G/200G (100% full), with only 6-93 MB free. The apt-get install of git failed during file copy.

## Repair Steps

1. **Cleaned apt caches**: `apt-get clean`, removed `/var/lib/apt/lists/*`, `/var/cache/apt/archives/*.deb`
2. **Configured partial packages**: `dpkg --configure -a` completed successfully for all partially unpacked packages (perl, openssh-client, libcurl3-gnutls, etc.)
3. **Cleaned conda/pip caches**: `conda clean --all -y` freed ~260MB of tarballs
4. **Re-ran apt-get update + install git**: Succeeded with 6M remaining on overlay
5. **Initialized git repo**: Baseline commit `98065583` with `_baseline` tag
6. **Installed record_score.sh**: Copied to `/tools/record_score.sh`

## Verified In-Container Evaluation Command

```bash
export PYTHONPATH=/autosota_cache/venv-paper2787/lib/python3.10/site-packages:$PYTHONPATH
export MUJOCO_GL=osmesa
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /repo
timeout 3600 python -c "
import time, subprocess, sys, json, glob, os
start = time.time()
r = subprocess.run(
    [sys.executable, main.py, --config, configs/rfm.yaml,
     --override, env.name=walker-run,
     --override, algo=rfm,
     --override, seed=1,
     --override, logger.wandb_project=,
     --override, logger.debug=true,
     --override, logger.log_dir=/autosota_cache/paper2787-logs],
    capture_output=True, text=True, timeout=3500
)
elapsed = time.time() - start
print(fTRAINING_WALL_TIME_MINUTES={elapsed/60:.2f})

# Find and parse metrics.jsonl for final episode reward
log_base = /autosota_cache/paper2787-logs/walker-run/rfm
run_dirs = sorted(glob.glob(f{log_base}/*/metrics.jsonl))
if run_dirs:
    with open(run_dirs[-1]) as f:
        lines = f.readlines()
        last = json.loads(lines[-1])
        print(fFINAL_EPISODE_REWARD={last["episode_reward"]:.2f})
        print(fFINAL_STEP={last["step"]})
"
```

## Baseline Metrics (Verified from Reproduction)

- `training_time_minutes`: 26.78
- `final_episode_reward`: 719.06
- Hardware: 2× NVIDIA A100-SXM4-80GB
- Environment: walker-run (DMControl)
- Steps: 250,000

## Key Code Architecture

### Main optimization target: `src/agent/rfm.py` (756 lines)
- `make_posterior_mean_fns_for_q()` — posterior estimation with MC samples (lines 81-190)
- `_make_fused_update()` — single JIT function for all updates (lines 523-688)
- `_sample_latent_actions_flow_fn()` — flow integration with configurable steps (lines 296-355)
- `_sample_actions_flow_argmax_fn()` — argmax action sampling (lines 408-448)
- Config-driven params: num_estimator_mc_samples, num_integration_steps, sampler_method, num_particles_training, init_temperature, final_temperature

### Supporting files
- `src/agent/critic.py` — ReturnFunction with Q-network and optimizer (line 168: `optax.adam(lr)`)
- `src/components/mlp.py` — ScoreMLP, Block class (line 35), no LayerNorm
- `src/trainer.py` — Trainer with batch_size config (line 30 default: 512 in Trainer, overridden by config)

## Config Override Paths

All tunable parameters accessible via `--override`:
| Parameter | Config Path | Baseline | Range |
|-----------|------------|----------|-------|
| MC samples | `rfm.num_estimator_mc_samples` | 100 | 10-100 |
| Integration steps | `rfm.num_integration_steps` | 10 | 3-10 |
| Sampler method | `rfm.sampler_method` | euler | euler/heun |
| Training particles | `rfm.num_particles_training` | 32 | 8-32 |
| Init temperature | `rfm.init_temperature` | 0.02 | 0.01-0.10 |
| Final temperature | `rfm.final_temperature` | 0.02 | 0.01-0.05 |
| Batch size | `trainer.batch_size` | 256 | 128-512 |
| Actor LR | `actor.lr` | 0.0003 | 1e-4 to 1e-3 |
| Critic LR | `critic.lr` | 0.001 | 3e-4 to 3e-3 |
| Hidden size | `actor.hidden_size` | 256 | 128-512 |
| Hidden layers | `actor.hidden_layers` | 2 | 1-3 |

## Optimization Strategy

P0 (config-only, low risk):
1. RFM-002: Reduce integration steps + euler sampler (num_integration_steps=5, sampler_method=euler)
2. RFM-001: Reduce MC samples from 100 to lower values (50, 25)
3. RFM-004: Reduce training particles from 32 to 16
4. RFM-005: Temperature annealing (0.05 → 0.01)

P1 (code changes, moderate risk):
5. RFM-011: AdamW optimizer
6. RFM-006: Cosine LR schedule
7. RFM-003: JAX linearize for posterior estimation
8. RFM-010: LayerNorm in ScoreMLP Block

P2 (code changes, higher risk):
9. RFM-012: Batch size increase
10. RFM-007: Reflow regularization
11. RFM-008: Split fused update
12. RFM-009: Adaptive MC samples

Target: ≥6 non-baseline iterations, improve training_time while preserving episode_reward ≥683.
