# Optimization Results: GeoMotion — Rethinking Motion Segmentation via Latent 4D Geometry

## Summary
- **Total iterations**: 8 (0-7)
- **Best J&F**: **0.86925** (Iter 2: TTA horizontal flip)
- **Baseline J&F**: 0.8590 (already above paper's reported 0.847)
- **Best J**: 0.8706 | **Best F**: 0.8679
- **Improvement over baseline**: +0.01025 J&F (+1.02%)
- **Improvement over paper**: +0.02225 J&F (+2.6%)
- **Dataset**: DAVIS 2016 validation (20 sequences)

## Key Changes Applied

### 1. Test-Time Horizontal Flip Augmentation — TTA (Iter 2) ✅ BEST
- **File**: `eval.py`
- **Change**: Added horizontal flip TTA in the motion mask prediction loop. For each chunk of frames, predict masks on both original and horizontally flipped frames, then flip masks back and average with original predictions.
- **Effect**: J&F 0.8590 → **0.86925** (+1.02%), J: 0.8706, F: 0.8679
- **Highlight**: The `breakdance` sequence alone improved by +0.20 J&F
- **Status**: ✅ SUCCESS — Best result, active in final optimized code

### 2. Remove Unused `split_components` Import (Repro Fix)
- **File**: `eval.py`
- **Change**: Removed `split_components` from the import statement (line 5). This import was unused and caused compatibility issues during reproduction.
- **Status**: ✅ Applied (pre-optimization repro fix)

### 3. Binary Evaluation Threshold 0.1 → 0.2 (Iter 4)
- **File**: `eval.py`
- **Change**: Increased the binary threshold for mask binarization from 0.1 to 0.2
- **Effect**: J&F 0.8590 → 0.86035 (+0.13%) — marginal improvement
- **Note**: Tested WITHOUT TTA. Since SAM2-refined masks are already near-binary, threshold tuning has limited impact. Not stacked with TTA in final code.
- **Status**: ✅ SUCCESS (minor, not included in final)

### 4. Flow Fusion max → mean (Iter 7)
- **File**: `motion_seg_inference.py`
- **Change**: Changed bidirectional flow fusion from 'max' to 'mean' in `fuse_flow_magnitudes()`
- **Effect**: J&F 0.8590 → 0.8598 (+0.08%) — marginal improvement
- **Note**: Tested WITHOUT TTA. Not stacked with TTA in final code.
- **Status**: ✅ SUCCESS (marginal, not included in final)

## Baseline vs. Best Metrics

| Metric | Paper Reported | Baseline (Iter 0) | Best (Iter 2, TTA) | vs Paper | vs Baseline |
|--------|---------------|-------------------|---------------------|----------|-------------|
| J&F | 0.847 | 0.8590 | **0.86925** | +2.6% | +1.02% |
| J (IoU) | 0.845 | 0.865 | **0.8706** | +3.0% | +0.65% |
| F (Boundary) | 0.850 | 0.854 | **0.8679** | +2.1% | +1.63% |

> Note: Baseline already exceeded paper's reported metrics (J&F 0.8590 vs 0.847), likely due to the SAM2 hiera_large model checkpoint and refined evaluation setup.

## What Worked

1. **TTA horizontal flip**: The single most impactful optimization. Flipping frames horizontally and averaging predictions reduces prediction variance and improves robustness to asymmetric motion patterns. The `breakdance` sequence with complex rotational motion benefited dramatically (+0.20 J&F), suggesting TTA helps most when motion direction is ambiguous.
2. **Binary threshold tuning (0.1→0.2)**: Marginal but positive impact. SAM2-refined masks are already near-binary, so the threshold has limited effect.
3. **Flow fusion mean**: Slightly smoother flow estimates, but the `max` operator already works well for most sequences.

## What Didn't Work

1. **Dense CRF post-processing (Iter 1)**: Severely degraded performance (e.g., `blackswan` J&F: 0.95 → 0.78). Root cause: **image edges ≠ motion boundaries**. CRF/guided filter blends static texture boundaries into motion masks, which is fundamentally wrong for motion segmentation where many static objects have sharp image edges but no motion.

2. **Temporal mask smoothing (Iter 3)**: No improvement (J&F unchanged at 0.86925). Simple Gaussian temporal smoothing blurs boundaries because objects move between frames. Proper optical-flow-based warping would be needed for effective temporal smoothing — but that requires accurate flow at motion boundaries, which is the hard part.

3. **SAM2 preprocess threshold tuning (Iters 5-6)**: Both 0.5 and 0.7 thresholds degraded performance. The default 0.8 is well-chosen — it provides SAM2 with high-confidence prompts while filtering noise. Lower thresholds introduce noisy prompts that confuse SAM2's mask decoder.

4. **Session termination**: The optimizer session died after iter-7 before writing the final scores and report. The `scores.jsonl` was not persisted to disk. The optimized code was reconstructed from the original repository clone + log-based patch reconstruction.

## Key Insight

GeoMotion's SAM2 refinement pipeline is already well-tuned — the SAM2 preprocess threshold (0.8), binary evaluation threshold (0.1), and max flow fusion are near-optimal defaults. The most impactful improvement came from **test-time augmentation** (TTA flip), which is a standard technique that wasn't part of the original evaluation script. This suggests the paper focused on model architecture rather than evaluation methodology.

The failed CRF experiment revealed a fundamental principle: **motion segmentation is NOT image segmentation**. Techniques that work for semantic/instance segmentation (CRF, guided filter, edge-aware refinement) fail here because motion boundaries are defined by 3D object movement, not 2D image edges. A static object with sharp texture edges should NOT have a motion boundary.

## Top Remaining Ideas (for future runs)

1. **Inference-time improvements** (no retraining needed):
   - Multi-scale TTA (0.75×, 1.0×, 1.25×) with flip — combine with existing TTA (+0.5–1.5 J&F)
   - Chunk overlap for temporal continuity (+0.2–0.5 J&F)
   - Morphological post-processing for small component cleanup (+0.2–0.5 J&F)

2. **SAM2 configuration exploration**:
   - SAM2 hiera_b+ (base+) model — faster, may have different bias characteristics
   - Multi-pass SAM2 with different initialization strategies and consistency-based fusion
   - SAM2 logit threshold tuning (separate from preprocess threshold)

3. **Flow quality improvements**:
   - Increase RAFT iteration count (12 → 32) for finer flow details
   - Camera motion compensation using Pi3's predicted camera poses
   - Flow gradient as boundary cue for mask refinement

4. **Retraining-required improvements**:
   - Replace RAFT with modern flow method (SEA-RAFT, GMFlow) — expected +0.5–1.5 J&F
   - Train lightweight boundary refinement head on motion boundary dataset
   - Depth consistency loss for static region regularization
