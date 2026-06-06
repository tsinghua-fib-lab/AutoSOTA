# Optimization Results: MixerCSeg — An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention

## Summary
- **Total iterations**: 11 (plus baseline)
- **Best mIoU**: **0.9198** (baseline: 0.9151, improvement: **+0.51%**)
- **Best ODS**: 0.9149 (baseline: 0.9095, +0.59%)
- **Best OIS**: 0.9292 (baseline: 0.9226, +0.72%)
- **Best commit**: `36c39335ae` (iter-3: TTA: Multi-Scale (1.0+1.25) + hflip)

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mIoU | 0.9151 | 0.9198 | +0.51% |
| ODS | 0.9095 | 0.9149 | +0.59% |
| OIS | 0.9226 | 0.9292 | +0.72% |
| F1 (thresh=0) | 0.5949 | 0.5861 | -1.5% |
| Precision (thresh=0) | 0.4236 | 0.4147 | -2.1% |
| Recall (thresh=0) | 0.9991 | 0.9995 | +0.04% |

Note: F1, Precision, Recall are computed at threshold=0.0 by the eval script (`cal_prf_metrics[0,3]`), which differs from the paper's F1 computation. mIoU, ODS, OIS match the paper's methodology.

## Key Changes Applied

| Change | File(s) | Effect |
|--------|---------|--------|
| TTA: Multi-scale + flip ensemble | `MixerCSeg.py`, `main.py` | +0.51% mIoU |
| Fix `.view()` → `.reshape()` for non-contiguous tensors | `hog_edge.py` | Enables multi-scale TTA |
| Adaptive HOG cell size for small feature maps | `hog_edge.py` | Enables multi-scale TTA |
| Morphological post-processing (optional) | `MixerCSeg.py`, `main.py` | No mIoU gain, kept as option |
| Add `--use_tta` and `--use_morph` CLI flags | `main.py` | Enables TTA/morph at inference |

## What Worked

1. **Test-Time Augmentation** — The single most effective optimization. Flip-only ensemble (+hflip, +vflip) gave +0.27% mIoU. Adding a multi-scale pass at 1.25x (640×640) with hflip at base scale gave an additional +0.24%, for a total of +0.51% over baseline.
2. **Multi-scale inference at 1.25x** — The higher resolution pass (640×640) captures finer crack details that are missed at 512×512. Combined with flip ensemble, this is the optimal configuration within the 900s evaluation timeout.
3. **`.reshape()` fix for non-contiguous tensors** — Required for flip augmentation to work correctly with the HOG edge gating module.

## What Didn't Work

1. **Training-based optimization** — Too slow. Each training epoch takes ~30-40 minutes. Even 1 epoch of fine-tuning exceeded practical time limits.
2. **3-scale TTA** — Timed out at 900s. The overhead of multiple scale passes is too high.
3. **Photometric TTA** (brightness/contrast augmentation) — Timed out, possibly due to GPU synchronization overhead from tensor clamping operations.
4. **Bilateral filter post-processing** — CPU-GPU transfer overhead pushed TTA beyond the timeout.
5. **Laplacian prediction sharpening** — Slightly regressive (-0.01% mIoU). No improvement.
6. **Morphological post-processing** — No change in mIoU or ODS. Only improved F1 at threshold=0 (less meaningful metric).
7. **Additional flip passes** — Adding vflip to the 3-pass config caused timeout (4 passes at ~878s est. vs 900s limit).

## Constraints That Limited Optimization

1. **900s evaluation timeout** — Severely limited TTA options. Each model forward pass takes ~136s at 512×512 and ~470s at 640×640.
2. **Pretrained weights constraint** — Training-based optimization was prohibited by the red line "never replace or fine-tune the paper's pretrained weights."
3. **Cannot modify eval script** — All optimizations had to be within the model's forward pass or post-processing.

## Top Remaining Ideas (for future runs)

1. **Focal Tversky Loss** — Research-backed loss function for extreme class imbalance. Would require retraining (IDEA-001).
2. **Boundary-aware loss with auxiliary edge head** — Direct edge supervision could improve crack boundary delineation (IDEA-002).
3. **Strip/Dilated convolutions in decoder** — Better capture thin, elongated crack structures (IDEA-005).
4. **Cosine LR with warmup** — Replace PolyLR for better convergence if retraining is enabled (IDEA-009).
5. **SWA/EMA weight averaging** — Requires multiple checkpoints, but could improve with longer training (IDEA-011).
6. **Gradient clipping** — Standard for SSM models, would improve training stability if retraining is done (IDEA-008).
7. **SE-Net channel attention in decoder** — Nearly free (128 params) channel attention for feature selection (IDEA-017).
