# Code Analysis — Paper 2579 SOTA Preparation Repair

## Original Preparation Failure

The preparation phase failed because the SOTA container could not install `git` via `apt-get install`. The apt repositories were unreachable due to network/proxy configuration issues during the initial setup window.

The original preparation log shows:
1. First attempt in reusable container `autosota_repro_paper_2579`: apt-get failed with "502 Bad Gateway" for ubuntu archive packages, git not installed
2. Fallback to fresh container `autosota_sota_paper_2579` from `autosota/paper-2579:reproduced`: apt-get failed with "Connection failed" for multiple packages, git not installed

## Repair Actions

1. Verified container `autosota_sota_paper_2579` is running (Docker image: `autosota/paper-2579:reproduced`)
2. Re-ran `apt-get update && apt-get install -y git` — succeeded this time (network conditions improved)
3. Initialized git repo at `/repo` with baseline commit and `_baseline` tag
4. Created `/tools/` and copied `record_score.sh` from host into container
5. Created `/autosota_artifacts/paper-2579/sota/` directory for scores and reports
6. Verified baseline evaluation produces expected accuracy

## Corrected In-Container Evaluation Command

```bash
cd /repo && python3 eval_ucr.py
```

No Docker-level wrapper needed. The script is self-contained and uses:
- UCR datasets at `/datasets/ucr/` (121 dataset directories)
- Mantis-8M checkpoint auto-downloaded/cached via HF mirror at `/autosota_cache/hf/`
- GPU auto-detected via `DEVICE = "cuda"`

## Baseline Reproduction Evidence

```
UCR Average Accuracy: 0.8190
Paper value:          0.8195
Delta:                -0.0005
Within bounds:        True  (CI: [0.8029, 0.8212])
Datasets evaluated:   108/108 (13 skipped)
Elapsed time:         75.6s
```

Configuration: MantisV1, layer 3 (return_transf_layer=2), combined CLS+mean tokens (512-dim), RandomForest n=200, input length 512.

The baseline matches the reproduction manifest (0.819) exactly and falls within the paper's confidence interval.

## Reusable Resources

- UCR datasets: `/datasets/ucr/` — 121 pre-downloaded UCR archive datasets
- Mantis-8M checkpoint: `/autosota_cache/hf/hub/models--paris-noah--Mantis-8M/`
- Mantis source: `/repo/src/mantis/` — MantisV1, MantisV2, MantisTrainer, MantisPlus
- HF cache: `/autosota_cache/hf/` — Transformers, datasets, hub cache with HF mirror

## Safe Optimization Targets

All modifications MUST keep the encoder frozen and use the same UCR train/test splits.

### Low-risk, fast-to-test (CPU-only ML changes):
- RF hyperparameters: GridSearchCV over n_estimators, max_depth, min_samples_split
- Classifier family: LogisticRegression, RidgeClassifier with CV-tuned regularization
- Feature scaling: StandardScaler before classifier
- Statistical features: Per-series min/max/mean/std/skew/kurtosis

### Medium-risk, moderate-time (modified feature extraction):
- Multi-scale ensembling: Extract at multiple interpolation lengths
- Multi-layer fusion: Concatenate embeddings from layers 2, 3, 4
- Difference features: Separate encoding of first-order difference
- Input length search: Compare 256, 512, 1024
- Forward-reverse encoding: Bidirectional embedding averaging

### Higher-risk (recovery of skipped data):
- Recover skipped datasets: NaN interpolation + length normalization for 13 variable-length datasets

### Config sweep:
- Grid search over known levers: layer, token_aggregation, classifier, input_length

## Constraints

- Frozen Mantis encoder (no fine-tuning)
- Same UCR train/test splits
- Same 108 datasets (or 121 if skipped-dataset recovery succeeds)
- 60-minute evaluation timeout
- Must record results via `/tools/record_score.sh`
