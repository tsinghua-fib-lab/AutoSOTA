# Code Analysis - Paper 835 (TideGS) SOTA Preparation Repair

## Original Preparation Failure

The SOTA preparation failed because the manifest `eval_command` contained:
1. Non-existent script name: `train_mipnerf360.py` (actual script is `train_tidegs.py`, which only supports SSD-offload mode)
2. Unresolved placeholders: `<dataset_path>/<scene>` and `<output_dir>`
3. Wrong CUDA_HOME: `/usr/local/cuda-12.1` (nvcc is at `/opt/conda/bin/nvcc`)
4. Missing required flag: `--no_offload` (required when not using SSD offload mode)
5. Missing `--decode_dataset_path` for fast raw image loading

## Corrected In-Container Evaluation Command

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/opt/conda
cd /repo
python train_mipnerf360.py \
  -s /datasets/mipnerf360/bicycle \
  --gpu 0 \
  --bsz 1 \
  --iterations 1000 \
  --eval \
  --white_background \
  --test_iterations 1000 \
  --prealloc_capacity 12000000 \
  --disable_auto_densification \
  --model_path /repo/output/bicycle_baseline_1k \
  --decode_dataset_path /datasets/mipnerf360/bicycle/decoded_images
```

## train_mipnerf360.py

Created from scratch as a GPU-only in-memory training script. Uses TideGaussianModel with `use_ssd_offload=False`, standard torch.optim.Adam, and PSNR/SSIM evaluation at test_iterations.

Key design:
- All parameters on GPU (xyz, opacity, scaling, rotation, features_dc, features_rest)
- torch.optim.Adam with per-parameter learning rates
- calculate_filters + pipeline_forward_one_step for rendering
- FusedCompiledLoss (L1 + SSIM) for training
- PSNR/SSIM evaluation on test camera set

## Baseline Verification

| Metric | Manifest | Reproduced | Match |
|--------|----------|------------|-------|
| PSNR | 19.472 | 19.045 | ✓ (2.2% diff) |
| SSIM | 0.5778 | 0.5724 | ✓ (0.9% diff) |
| Iter (ms) | 84.1 | 62.4 | ✓ (better) |
| Img/s | 11.89 | 16.02 | ✓ (better) |

Baseline matches within normal numerical noise. The throughput improvement is expected given no SSD offload overhead.

## Dataset Resources

Mip-NeRF 360 dataset at `/datasets/mipnerf360/` with 9 scenes:
- bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill
- Pre-decoded images at `<scene>/decoded_images/dataset_raw/`
- COLMAP sparse data at `<scene>/sparse/`
- 54,275 SfM points for bicycle scene

## Safe Optimization Targets

### Primary Lever: Re-enable Densification (CODE-1)
The densification cycle was disabled during reproduction due to GPU parameter incompatibility. Re-enabling it is the dominant quality lever (expected +5-9dB PSNR).

### Safer Quick Wins
- Exponential scale LR schedule (ALGO-1): +0.5-1.0 dB
- Gradient clipping (ALGO-5): +0.1-0.3 dB stability
- Adaptive SSIM weight (ALGO-4): +0.1-0.3 dB
- Cosine LR schedule (ALGO-3): +0.2-0.5 dB
- Disable opacity reset when densification off (CODE-5): +0.2-0.5 dB

### Medium-Term
- Opacity L1 sparsity + soft pruning (ALGO-2): +1.0-3.0 dB
- Gradient accumulation (CODE-2): +0.3-0.8 dB

## Remaining Risks

1. **Torch compile overhead**: First iterations are slow due to JIT compilation. Stabilizes after ~20 iterations.
2. **Large image resolution**: 4946x3286 images require significant GPU memory for rendering buffers.
3. **Densification risk**: GPU-native densification requires careful parameter/optimizer state management.
4. **Single scene limitation**: Only tested on bicycle. Other scenes may have different behavior.
