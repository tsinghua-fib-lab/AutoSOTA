# PGC Code Analysis (Paper 2350)

## Architecture Overview

PGC (Peak-Guided Calibration) is a two-stream AI-generated image detection model:
1. **RGB Stream**: DINOv2-large ViT backbone with LoRA adapters on attention/qkv/proj/MLP layers
2. **Residual Stream**: 3-layer CNN (3→64→128→256) with BatchNorm2d + ReLU
3. **PGCM (Peak-Guided Calibration Module)**: Computes per-patch/per-spatial scores, performs softmax peak aggregation with temperature τ, fuses via learnable λ_rgb
4. **Classifier**: Linear layer on concatenated [CLS token ∥ residual pooled] features (1024+256=1280 dims), L2-normalized before classification

## Key Files

| File | Purpose | Safe to modify? |
|------|---------|-----------------|
| `test.py` | Evaluation entry point | Yes (add flags, logging) |
| `train.py` | Training entry point | Yes (add warmup, eval changes) |
| `models/pgc.py` | Main model (PGCNetwork) | Yes (architecture changes) |
| `models/pgcm/peak_calibration.py` | PGCM module | Yes (learnable τ, multi-scale) |
| `models/encoder/residual_stream.py` | Residual CNN | Yes (BatchNorm→GroupNorm) |
| `models/encoder/rgb_stream.py` | DINOv2 backbone wrapper | Minimal (add hidden states) |
| `models/lora/lora.py` | LoRA adapter implementation | No (generic, correct) |
| `engine/trainer.py` | Training loop (PGCTrainer) | Yes (loss, scheduler, warmup) |
| `engine/evaluator.py` | Evaluation loop | Yes (add per-subset ranking) |
| `data/transforms.py` | Augmentation pipeline | Yes (add augmentations) |
| `data/train_dataset.py` | Training dataset (RealFakeDataset) | Yes (add generators) |
| `data/dataloader.py` | DataLoader factory | Yes (CutMix collator) |
| `data/eval_dataset.py` | Evaluation dataset | No (data loading only) |
| `utils/cli.py` | CLI argument parsing | Yes (add flags) |
| `utils/metrics.py` | Metric computation | No (metric definitions) |

## Evaluation Path

- Entry: `test.py` → `build_test_parser()` → `PGCNetwork(...)` → `evaluate_model()`
- Dataset: `UniversalFakeDetectDataset` scans `<root>/<subset>/{0_real,1_fake}/` 
- Metrics: `compute_all_metrics()` → `{acc, real_acc, fake_acc, ap, auc}`
- Output: `checkpoints/<name>/<name>_test.log`
- Key parsing line: `Overall test set mean - ACC: X.XXXX | ... | AP: Y.YYYY`

## Training Path

- Entry: `train.py` → `build_train_parser()` → `PGCTrainer` → `create_dataloader()`
- Dataset: `RealFakeDataset` scans real/fake dirs for images
- Loss: `BCEWithLogitsLoss` with label smoothing
- Optimizer: AdamW (lr=5e-5, wd=0.05, betas=(0.9,0.999))
- Scheduler: CosineAnnealingLR(T_max=total_steps, eta_min=lr*0.01)
- Gradient accumulation: 4 steps (effective batch 128 with batch_size=32)

## Available Resources

- **GPUs**: 2× NVIDIA A100-SXM4-80GB (as GPU 0,1 inside container)
- **Test Data**: 5 AIGI subsets (BLIP=9000, BlendFace=9000, GLIDE=9000, ProGAN=8000, WFIR=2000) = 37,000 images
- **Checkpoints**: PGC_train_progan_sdv1_4_ckpt.pth (ProGAN+SDv1.4, 600 steps), PGC_train_progan_ckpt.pth (ProGAN only)
- **Backbone**: DINOv2-large at /models/dinov2-large/
- **Training Data**: NOT AVAILABLE (need to download ProGAN from UniversalFakeDetect and SDv1.4 from GenImage)

## Red Lines (Do Not Modify)

- `utils/metrics.py` - metric computation definitions
- `data/eval_dataset.py` - dataset loading and subset splitting
- Test data at `/datasets/AIGI/test/` - labels and splits
- `/tools/record_score.sh` - score recording

## Current Baseline

- Iter 0: Acc=88.8%, AP=94.0% (5 subsets)
- GPUs: Use `--devices 0` (not 6,7 - those are host indices)
- No `CUDA_VISIBLE_DEVICES` needed inside container

## Optimization Constraints

- Cannot use test data for training (would violate evaluation protocol)
- Must maintain same evaluation command format
- Must report all metrics (Acc, AP, AUC) honestly
- Acc and AP are co-primary; AUC is secondary
