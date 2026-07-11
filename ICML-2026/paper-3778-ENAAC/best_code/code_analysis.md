# Code Analysis: Paper 3778 SOTA Preparation Repair

## Original Failure

The SOTA preparation failed because:
1. The reusable container `autosota_repro_paper_3778` had `dpkg` interrupted and `git` not installed
2. The fallback container `autosota_sota_paper_3778` was started successfully but had the same `dpkg`/`git` issues
3. Docker proxy settings caused initial `docker run` rejection (resolved by retry with different proxy config)

## Repair Applied

1. Fixed dpkg: `dpkg --configure -a` (tzdata config, no interaction needed)
2. Installed git: `DEBIAN_FRONTEND=noninteractive apt-get install -y git`
3. Set up git config, baseline commit, and `_baseline` tag
4. Copied `/tools/record_score.sh` from host
5. Created `/autosota_artifacts/paper-3778/sota/` directory

## Corrected In-Container Evaluation Command

```bash
cd /repo/isic && python3 -u eval_reproduce.py
```

This loads the trained model from `/models/50_bias/model_seed0_mode_presence_absence_debias.pth` 
and evaluates on the pre-computed validation set at `/tmp/ISIC_precomputed_val.pt`.

## Baseline Verification

| Metric | Manifest Baseline | Repaired Baseline | Match |
|--------|------------------|-------------------|-------|
| Benign Accuracy | 0.8851 | 0.8851 | ✓ |
| Malignant Accuracy | 0.4943 | 0.4904 | ✓ (noise) |
| Avg Accuracy | 0.6897 | 0.6877 | ✓ (noise) |
| Attr | 0.0007 | 0.0008 | ✓ (noise) |

All metrics match within normal numerical noise. The eval protocol is confirmed working.

## Container Environment

- Image: `autosota/paper-3778:reproduced`
- Container: `autosota_sota_paper_3778`
- GPUs: 0,1 (CUDA devices)
- ISIC data: `/tmp/ISIC2020_2/` (local copy for fast I/O)
- Pre-computed train: `/tmp/ISIC_precomputed_train.pt`
- Pre-computed val: `/tmp/ISIC_precomputed_val.pt`
- Model checkpoint: `/models/50_bias/model_seed0_mode_presence_absence_debias.pth`
- XResNet-50 backbone: `/models/xfixup_resnet50_model_best.pth.tar`
- Cache mounts: `/autosota_cache`, `/datasets`, `/models`

## Safe Optimization Targets

The training script is `/repo/isic/train_fast.py`. It uses:
- Pre-computed 256×256 image tensors (GPU-based transforms)
- XResNet-50 backbone with presence+absence debiasing
- L1 attribution prior loss with fixed lambda grid search {1,10,100,1000,10000}
- Binary cross-entropy classification loss
- Single seed (RUNS=1), 20 epochs, BATCH_SIZE=128

The eval script `/repo/isic/eval_reproduce.py` loads a trained model from 
`/models/50_bias/` and evaluates on inverse_bias, train_bias, and no_bias splits.

## Hard Constraints

- Do not modify evaluation protocol, metrics, data splits, or benchmark outputs
- All code changes inside container `/repo`
- Use `/tools/record_score.sh` for all score records
- Git commit each successful implementation
