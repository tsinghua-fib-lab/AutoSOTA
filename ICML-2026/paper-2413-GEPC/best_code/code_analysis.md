# GEPC SOTA Preparation Repair — Code Analysis

## Original Preparation Failure

The orchestrator failed during SOTA preparation because:
1. `git` was not installed in the `autosota/paper-2413:reproduced` image.
2. The initial `apt-get install git` attempt failed because `dpkg` was in an interrupted state (`dpkg was interrupted, you must manually run "dpkg --configure -a"`).
3. A prior `apt-get` process had been interrupted during image build, leaving the lock.

## Repair Actions

1. **Fixed dpkg**: `DEBIAN_FRONTEND=noninteractive dpkg --configure -a`
2. **Installed git**: `apt-get install -y -qq git` (v2.25.1)
3. **Initialized git repo**: `git init`, configured user, committed baseline.
4. **Copied record_score.sh**: From host to `/tools/record_score.sh` in container.
5. **Created artifact directories**: `/autosota_artifacts/paper-2413/sota/`.

## Corrected In-Container Evaluation Command

```bash
cd /repo
python scripts/bench_gepc_images.py \
  --config configs/gepc_cifar10_vs_svhn.yaml \
  --device 0 \
  --seed 1337 \
  --strict_determinism \
  --verbose
```

This is the same command from the manifest, minus host-side shell wrapping. No translation needed — the manifest command was already valid inside the container.

## Baseline Verification

- **Expected**: AUROC = 0.9274
- **Observed**: AUROC = 0.9274 ✓
- **Runtime**: ~118 seconds (fit: ~78s, score ID: ~19s, score OOD: ~19s) on A100-80GB.
- **Config**: keep_k=2, features=[gepc_s], snr_levels=[0.99997,0.9999,0.9979,0.9969], mc_samples=1, amp=fp32

## Reusable Resources

- **Checkpoint**: `/repo/checkpoints/celeba_ema_0.9999_499999.pt` (210 MB) — pretrained CelebA-32 diffusion backbone
- **Datasets**: CIFAR-10 at `/datasets/cifar-10-batches-py/`, SVHN at `/datasets/train_32x32.mat` and `/datasets/test_32x32.mat`
- **Cache**: `/autosota_cache` with HuggingFace hub cache

## Safe Optimization Targets

The evaluation is read-only (no training). All optimization targets are:
1. **Config changes** in `configs/gepc_cifar10_vs_svhn.yaml` or `configs/gepc_cifar10.yaml`
2. **Method changes** in `gepc/methods/gepc.py` (inference-only code paths)
3. **Data limits** (id_train.limit, id_test.limit, ood.limit) — more samples for better KDE estimation

Red lines: no changes to evaluation protocol, metrics, dataset splits/labels, or model checkpoint.

## Key Hyperparameter Levers

| Parameter | Current | Range | File |
|-----------|---------|-------|------|
| features | [gepc_s] | gepc_s, gepc_s_cos, gepc_s_pair | config |
| keep_k | 2 | 1–6 | config |
| snr_levels | [0.99997, 0.9999, 0.9979, 0.9969] | 0.5–1.0 | config |
| vector_mode | none | none, mvn | config |
| bandwidth | 0.0 (auto) | 0.0–0.5 | config |
| topk_rho | 0.3 | 0.05–0.5 | config |
| shift_px | 1 | 1–4 | config |
| mc_samples | 1 | 1–5 | config |
| amp | fp32 | fp32, fp16 | config |
| fit_batches | 128 | 16–512 | config |
| id_train.limit | 2000 | 1000–50000 | config |
| spatial_pool | topk | topk, mean | config |
| group_shifts | true | true, false | config |
| weight_t | inv_cv | inv_cv, uniform | config |
| density_mode | kde | kde, gmm | config |

## Git State

- Baseline commit: `5cb1f5a1f2`
- Tag `_baseline`: points to baseline
- Tag `_best`: will track best result
