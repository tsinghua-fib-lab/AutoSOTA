# Optimization Results: IsoCLIP — Decomposing CLIP Projectors for Efficient Intra-modal Alignment

## Summary

- **Total iterations**: 11 (+ baseline + final)
- **Best mAP**: **27.39** (baseline: 27.03, improvement: **+0.36**, **+1.33%**)
- **Target**: 28.38 (not reached, 96.5% of target)
- **Best commit**: `76db6aec674e548b6ed261d54fe795f1a2223082`

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mAP | 27.03 | **27.39** | +0.36 (+1.33%) |
| mAP_at_R | 18.71 | 18.96 | +0.25 |
| precision_at_R | 29.74 | 30.03 | +0.29 |
| recall_at_1 | 60.96 | 60.95 | -0.01 |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Soft sigmoid thresholding (tau=5.0) | +0.09 mAP | Replaced hard binary band selection with sigmoid-weighted transition. Smooth boundaries prevent information loss. |
| Multi-band ensemble (4 bands) | +0.27 mAP (cumulative +0.36) | Averaged similarity matrices from 4 band configurations: (100,25), (150,50), (200,75), (250,100). Each band captures complementary spectral information. |

The best configuration uses both changes together: `--iso_tau 5.0 --iso_ensemble`.

## What Worked

1. **Multi-band ensemble** — The most impactful technique. Averaging similarities from multiple SVD band configurations captures complementary information. Different ktop/kbottom settings reveal different aspects of intra-modal alignment, and the ensemble benefits from all of them.
2. **Soft sigmoid thresholding** — Smooth transitions at band boundaries outperform hard binary selection. The sigmoid weighting prevents information loss at the cutoff points.
3. **Both techniques are robust** across a range of tau values (2.0-5.0 all give similar performance).

## What Didn't Work

1. **Gap-guided band selection** — The simple gap statistic on singular values (ktop=13) is far from optimal. The spectral boundary doesn't follow a simple gap heuristic — the paper's validation (ktop=150 on Caltech101) is much more reliable.
2. **ZCA whitening** — Catastrophic failure (mAP 12.74). CLIP features are already highly structured; whitening destroys the variance structure essential for discriminative retrieval.
3. **Residual component preservation** — Keeping 10% of removed components cancels the soft sigmoid benefit. The removed components are genuinely noise — even small amounts hurt performance.
4. **Temperature calibration** — Monotonic transforms (dividing by a constant) don't affect ranking metrics (mAP, recall@k). Only the ORDER of similarity scores matters.
5. **Feature concatenation** — Concatenating pre-projection features with ISO-projected features degrades performance (25.78). The pre-projection features are noisier and dilute the ISO benefit.
6. **Expanding from 4 to 6 bands** — Adding extreme bands (50,25) and (300,125) hurt performance. The current 4-band range already captures the useful spectral diversity.

## Top Remaining Ideas (for future runs)

1. **Per-dataset ktop/kbottom optimization** — Grid-search optimal (ktop, kbottom) specifically for CUB-2011. The paper's default (150, 50) was validated on Caltech101 only.
2. **ViT-B/16 or ViT-L/14 backbone** — The paper shows these backbones get larger absolute gains from ISO. Not tested here due to missing model weights.
3. **NNN post-processing** — Nearest Neighbor Normalization (Chowdhury et al., EMNLP 2024) is complementary to ISO and could add +1-5 mAP as a post-processing step.
4. **Weighted ensemble** — Instead of simple averaging, learn per-band weights optimized for the target dataset.
5. **OpenCLIP variants** — Different pretraining data distributions have different intra-modal alignment properties affecting optimal ISO parameters.
