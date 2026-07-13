# CellBRIDGE Code Analysis for SOTA Optimization

## Evaluation Path
- **Command**: `bash eval_cellbridge.sh 42` (seed=42)
- **Step 1**: `sweep_alpha_align.py` — generates UOT-FGW coupling (deterministic, ~10 min)
- **Step 2**: `train_flow.py` — trains velocity field via flow matching (~3 min/seed on A100)
- **Step 3**: `sample_with_velocity.py` — samples pushforward using best checkpoint, computes W1/W2 (~1 min)

## Key Files
| File | Role |
|---|---|
| `conf/flow_matching.yaml` | Training hyperparams (LR, scheduler, batch_size, epochs, topk) |
| `conf/model/mlp.yaml` | Model architecture (hidden_dim, depth, norm, dropout) |
| `conf/sampling_velocity.yaml` | Inference config (ODE steps, method, t_final) |
| `conf/trainer/default.yaml` | Trainer config (gradient_clip_val, callbacks) |
| `src/cellbridge/dynamics/flow_matching.py` | WarmupCosine scheduler, FlowMatchingLitModule, FlowTrainer |
| `src/cellbridge/dynamics/models.py` | VelocityMLP with optional LayerNorm |
| `src/cellbridge/dynamics/sampling.py` | VelocitySampler, ODE integration, metric evaluation |
| `src/cellbridge/utils/extraction.py` | Best checkpoint selection by val/loss |

## Metric Parsing
- Metrics printed as Python dict to stdout: `{'pushforward': {'wasserstein_1': <W1>, 'wasserstein_2': <W2>}, ...}`
- Also saved to `metrics.json` under `alpha_<value>/fm/seed_<N>/sample_velocity/`
- Parse: `wasserstein_1` and `wasserstein_2` from the `pushforward` dict

## State of the Codebase
- **Already implemented**: ModelCheckpoint (monitors val/loss), best checkpoint loading (`use_best_checkpoint: true`), WarmupCosine scheduler class (but disabled with scheduler=none), LayerNorm support (but disabled with norm=none)
- **Baseline config**: scheduler=none, norm=none, hidden_dim=64, depth=3, steps=100, method=rk4, max_epochs=500, batch_size=128, topk=10
- **Coupling**: Deterministic UOT-FGW with numIterEMD=200000, alpha=1.0

## Safe Modification Targets
1. `conf/flow_matching.yaml` — optimizer config (scheduler, warmup), dataset config (topk), max_epochs, batch_size
2. `conf/model/mlp.yaml` — hidden_dim, depth, norm
3. `conf/sampling_velocity.yaml` — ODE steps, method
4. `conf/trainer/default.yaml` — gradient_clip_val

## Performance Optimization
- Coupling is deterministic — can skip Step 1 when coupling.npy already exists
- Training ~3 min/seed, sampling ~1 min
- Full pipeline: ~14 min per iteration

## Pre-existing Resources
- Data: `/repo/data/light_lite.h5ad` (23MB, V1 Light)
- Venv: `/autosota_cache/paper-5071/venv`
- UV cache: `/autosota_cache/uv`
- Baseline coupling: `/autosota_cache/paper-5071/experiments/light/align_sweep/eval_42/artifacts/alpha_1.000/coupling.npy`
