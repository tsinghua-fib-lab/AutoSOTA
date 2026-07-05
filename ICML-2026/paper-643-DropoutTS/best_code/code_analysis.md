# DropoutTS SOTA Optimization - Code Analysis

## Preparation Failure & Repair

**Original Failure:** The SOTA preparation step failed because:
1. `git` was not installed in the `autosota/paper-643:reproduced` container image
2. `apt-get install git` failed due to transient 502 Bad Gateway errors from the proxy (`http://172.17.0.1:17890`)
3. The first `docker run` attempt for `autosota_sota_paper_643` was rejected by Docker auth plugin (`--network host`)
4. The second `docker run` attempt succeeded (without `--network host`)

**Repair Applied:**
- Verified container `autosota_sota_paper_643` is running (ID: 5c403c78b22f)
- Successfully installed `git` via `apt-get install -y --fix-missing git` (network recovered)
- Initialized git repo at `/repo` with baseline commit and `_baseline` tag
- Copied `/tools/record_score.sh` into container
- Verified artifacts mount `/autosota_artifacts` is writable
- Smoke-tested training pipeline with 2-epoch run on H=96 (passed)
- Recorded iteration 0 baseline with record_score.sh

## Baseline Metrics (Verified)
| Metric | Reproduced | Paper | Gap |
|--------|-----------|-------|-----|
| MSE | 0.3877 | 0.380 | +2.0% |
| MAE | 0.4037 | 0.399 | +1.2% |
| MSE_H96 | 0.2931 | - | - |
| MSE_H192 | 0.3726 | - | - |
| MSE_H336 | 0.4393 | - | - |
| MSE_H720 | 0.4460 | - | - |

## Optimization Surface

### DropoutTS Parameters (Primary Levers)
- `init_sensitivity` (gamma): Baseline 1.0. Controls noise-to-dropout mapping aggressiveness. Higher = more differentiated dropout. Is a learnable parameter.
- `p_min`/`p_max`: Baseline [0.05, 0.5]. Dropout rate bounds.
- `init_alpha`: Baseline 10.0. RRF filter sigmoid sharpness. Is a learnable parameter.
- Learnable RRF params: `sfm_scale`, `sfm_bias`, `alpha`, `sensitivity` - all optimized via gradient descent

### Training Parameters
- `batch_size`: Baseline 64. Larger batches improve noise score batch-normalization
- `lr`: Baseline 2e-4. Adam optimizer
- `weight_decay`: Baseline 5e-4
- `num_epochs`: 100, but early stopping (patience=10) typically stops at 13-40

### Available Infrastructure (Not Used in Baseline)
- `GradientClipping` callback (exists in codebase, not used)
- `CosineWarmup` / `CosineWarmupRestarts` LR schedulers (exist, not used)
- `AdamWnanoGPT` optimizer (exists, not used)

### Where Dropout is Applied
- `FeatureEmbedding.dropout` (p=0.1 from TimeMixerConfig)
- `MLPLayer.dropout` (p=0.0 in mixing layers - effectively inactive)
- All `nn.Dropout` layers replaced by `SampleAdaptiveDropout` via `DropoutTSCallback`

### Architecture Details
- Model: TimeMixer backbone with DropoutTS
- Dataset: ETTh2, split 6:2:2 (8640/2880/2880)
- Input length: 96, Output horizons: [96, 192, 336, 720]
- 7 features, Channel-independent mode
- GPUs: 2x A100-80GB (using GPU 0 for training)
- AMP with float32, GradScaler enabled

## Current Evaluation Command (Repaired)
```bash
cd /repo && python3 run_optimization.py
```
Environment variables control hyperparameters: DROPOUTTS_GAMMA, DROPOUTTS_P_MIN, DROPOUTTS_P_MAX, DROPOUTTS_ALPHA, BATCH_SIZE, LR, WEIGHT_DECAY, NUM_EPOCHS, SEED, USE_CLIP_GRAD, CLIP_GRAD_MAX_NORM, USE_COSINE_WARMUP, WARMUP_EPOCHS, GPU
