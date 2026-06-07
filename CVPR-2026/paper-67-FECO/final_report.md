# Optimization Results: FECO — Foot Contact Estimation

## Summary
- **Total iterations**: 24
- **Best `cont_f1`**: 0.588 (baseline: 0.577, improvement: +0.011 / +1.9%)
- **Target**: 0.6058 (not reached, short by 0.018)
- **Best commit**: `4910985` — "iter-7: Per-sample Otsu adaptive thresholding"

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| cont_pre | 0.563 | 0.552 | -0.011 |
| cont_rec | 0.613 | 0.641 | +0.028 |
| cont_f1 | 0.577 | 0.588 | +0.011 |

## Key Changes Applied

The only effective change was implementing **per-sample Otsu adaptive thresholding** in the model's forward pass:

1. **`lib/models/model.py`** (+32 lines): Added Otsu's method for per-sample threshold selection. After the decoder produces 265-dim contact logits, the method:
   - Computes sigmoid probabilities
   - Builds a 32-bin histogram of the 265 vertex probabilities
   - Finds the threshold that maximizes inter-class variance (Otsu's criterion)
   - Adjusts the logits so that the evaluation threshold (0.48) produces the Otsu-optimal binary predictions

2. **`lib/utils/contact_utils.py`** (1 line): Changed vit-h-14 contact threshold from 0.50 to 0.48.

## Iteration Log Summary

| Iter | Idea | F1 | Delta | Status |
|------|------|----|-------|--------|
| 0 | Baseline | 0.577 | — | — |
| 1 | T=1.2 + thres=0.55 | 0.567 | -0.010 | Failed |
| 2 | thres=0.48 | **0.579** | +0.002 | **Success** |
| 3 | TTA flip | crash | — | Failed |
| 4 | 336×336 resolution | 0.576 | -0.001 | Failed |
| 5 | init_contact ×0.85 | 0.579 | tie | — |
| 6 | Soft temp ensemble | 0.579 | tie | — |
| 7 | Otsu + thres=0.48 | **0.588** | +0.009 | **BEST** |
| 8 | Otsu + thres=0.50 | 0.588 | tie | — |
| 9 | Style decoder ensemble | 0.583 | -0.005 | Failed |
| 10 | Otsu + precision bias | 0.587 | -0.001 | Failed |
| 11 | Otsu 64 bins | 0.587 | -0.001 | Failed |
| 12 | Per-region Otsu | 0.537 | -0.051 | Failed |
| 13 | Neighbor smoothing α=0.3 | 0.520 | -0.068 | Failed |
| 14 | Neighbor smoothing α=0.1 | 0.577 | -0.011 | Failed |
| 15 | Otsu + thres=0.46 | 0.588 | tie | — |
| 16 | init_contact ×0.5 | 0.573 | -0.015 | Failed |
| 17 | Simple TTA flip | 0.565 | -0.023 | Failed |
| 18 | Otsu bimodality check | 0.588 | tie | — |
| 19 | Top-foot constraint | 0.588 | tie | — |
| 20 | Spatial attn temp=2.0 | 0.588 | tie | — |
| 21 | Zero adv_gamma | 0.559 | -0.029 | Failed |
| 22 | Double adv_gamma | 0.292 | -0.296 | Failed |
| 23 | Half adv_gamma | 0.577 | -0.011 | Failed |
| 24 | Spatial coherence | 0.550 | -0.038 | Failed |

## What Worked

1. **Per-sample Otsu thresholding** (+0.011 F1): The single most effective technique. Adapting the contact threshold to each image's probability distribution significantly improved recall (+0.028) with only a small precision penalty (-0.011). This suggests the optimal threshold varies substantially across images (different poses, foot visibility, shoe types).

2. **Lower base threshold** (+0.002 F1): Reducing from 0.50 to 0.48 improved F1 by better balancing precision and recall. The model tends to under-predict contact, and a lower threshold compensates.

## What Didn't Work

1. **Test-time augmentation** (flip, multi-scale): Horizontal flip TTA consistently degraded performance. The model appears to be trained without flip augmentation, making predictions on flipped images unreliable.

2. **Higher input resolution** (336×336): Despite theoretically providing finer spatial detail, the 336 resolution degraded F1. The decoder was trained on 16×16 feature maps (224×224 ÷ 14 patch size) and doesn't generalize well to 24×24 maps.

3. **Architectural modifications** (style decoder ensemble, init_contact scaling, neighbor smoothing): All architectural changes degraded performance. The model's weights are tightly coupled to its architecture, and modifying even simple parameters like init_contact bias or the spatial attention temperature either had no effect or hurt.

4. **Adversarial gamma perturbation**: The adv_gamma values (0.02) are critical for test-time performance. Setting them to 0 dropped F1 from 0.588 to 0.559. Doubling them collapsed to 0.292. The original value is optimal.

5. **Spatial post-processing** (CRF, morphological ops, neighbor smoothing): All spatial coherence techniques degraded performance. The model's independent per-vertex predictions are already good, and adding spatial constraints dilutes their discriminative power.

## Key Insight

The FECO model produces well-calibrated per-vertex predictions, and the main room for improvement is in the **decision boundary** (threshold), not in the **feature representation**. The Otsu method succeeded because it adapts the decision boundary to each image's specific probability distribution. No amount of architectural modification could beat this simple post-hoc threshold optimization.

## Top Remaining Ideas (for future runs)

1. **Train with per-sample threshold loss**: If retraining were allowed, adding a loss that encourages bimodal probability distributions (making Otsu more effective) could improve results.

2. **Multi-threshold ensemble with learned weights**: Instead of simple voting, learn optimal weights for combining predictions at different thresholds.

3. **Per-region calibration**: Fit separate calibration curves for each foot region (heel, arch, toes) rather than using a single global calibration.

4. **Temporal smoothing** (for video datasets): When multiple frames are available, temporal smoothing of contact predictions could leverage the temporal consistency of foot contact.

5. **Test-time adaptation of decoder bias**: Instead of a fixed init_contact bias, adapt it per sample based on the prediction statistics.
