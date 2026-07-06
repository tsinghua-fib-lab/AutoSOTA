# SOTA Preparation Repair — Paper 512 (VIP)

## Original Failure

The evaluation command from the reproduction manifest crashed with:

```
FileNotFoundError: [Errno 2] No such file or directory: 
```

at `dinov3/hub/backbones.py:151` (`torch.load(weight_path, ...)`). The root cause was **missing model weights**: the DINOv3 backbone (`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`, ~1.2GB) and adapter (`dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth`, ~2.1GB) were not present at `/models/dinov3_vitl16_dinotxt/`.

## Repair Steps

### 1. Model Weights
- Source: Hugging Face repo `PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder`
- Downloaded both backbone and adapter `.pth` files to `/models/dinov3_vitl16_dinotxt/`
- Workaround: container `HF_ENDPOINT` pointed to `https://hf-mirror.com` which had SSL issues. Used `HF_ENDPOINT=https://huggingface.co` for download.
- Total model size: ~3.3GB

### 2. Dataset
- Download: Pascal VOC 2012 train+val from the Oxford server
- Extracted to: `/datasets/VOCdevkit/VOC2012/`
- Validation split: 1449 images in `ImageSets/Segmentation/val.txt`

### 3. Corrected In-Container Evaluation Command

```bash
cd /repo
CUDA_VISIBLE_DEVICES=0 python3 eval_seg.py --config ./configs/cfg_voc21.py --work-dir ./work_logs
```

This is the same command as the manifest; it now works because the weights and dataset are in place.

## Baseline Confirmation

| Metric   | Reproduction | This Run  | Match |
|----------|-------------|-----------|-------|
| mIoU     | 73.26%      | 73.26%    | ✓     |
| Memory   | ~1803 MB    | ~1803 MB  | ✓     |

Per-class IoU numbers all match the reproduction profile. The baseline is confirmed.

## Reusable Resources

- `/models/dinov3_vitl16_dinotxt/` — DINOv3 backbone + adapter weights
- `/datasets/VOCdevkit/VOC2012/` — Pascal VOC 2012 dataset
- `/autosota_cache/` — Hugging Face cache, pip cache, etc.

## Safe Optimization Targets

### Code-Level (dinosegmentor.py)
- `forward_slide()` (lines ~170-210): sliding window inference, crop blending
- `forward_feature()` (lines ~130-160): alias aggregation with tau/tem
- `__init__()` (lines ~60-80): alias loading, model initialization
- `predict()`: post-processing pipeline

### Config-Level (configs/cfg_voc21.py)
- `tau`: alias aggregation temperature (default 3.0)
- `tem`: alias softmax temperature (default 0.3)
- `logit_scale`: similarity logit scaling (default 40)
- `slide_stride`: sliding window stride (default 112)
- `slide_crop`: crop size (default 336)
- `prob_thd`: confidence threshold (default 0.28)
- `pamr_steps`: PAMR refinement steps (default 0)

### Alias-Level (configs/cls_voc21.txt)
- Class alias expansion/modification
- Does not require model reload — only alias text changes

### Red Lines (DO NOT CHANGE)
- Evaluation script (eval_seg.py) — benchmark protocol
- Dataset splits, labels, test data
- Metric computation (mmseg IoUMetric)
- Scoring script (/tools/record_score.sh)

## Optimization Objective

Primary: Improve mIoU (currently 73.26%)
Resource bounds: Time < 1000ms, Memory < 3000MB
Budget: 12 iterations max, target 6+ non-baseline attempts
