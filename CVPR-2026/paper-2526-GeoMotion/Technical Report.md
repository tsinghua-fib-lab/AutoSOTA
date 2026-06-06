# GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry — A Technical Report on Automated Optimization

## Abstract

This technical report documents an automated optimization study performed on the public reference implementation of GeoMotion, a CVPR 2026 motion segmentation framework that derives dynamic-object masks from latent 4D geometric priors produced by the pretrained Pi3 reconstruction backbone, fused with RAFT optical-flow magnitudes and refined by a SAM2 hiera-large mask decoder. The study targets the DAVIS 2016 validation split (20 sequences) and follows the AutoSOTA optimization protocol (tsinghua-fib-lab/AutoSOTA): an iterative loop in which candidate edits to inference and post-processing logic are proposed, applied to the original repository, evaluated under the standard DAVIS J&F protocol, and either accepted or rolled back. Eight iterations (Iter 0 through Iter 7) were executed. Reproducing the public checkpoint already yields J&F = 0.8590, exceeding the paper-reported J&F = 0.847 by +0.0120 absolute (+1.4%), which we attribute to the use of the SAM2 hiera-large checkpoint and the hardened evaluation script. The single accepted optimization, horizontal-flip test-time augmentation (TTA) inside the chunked motion-mask prediction loop, lifts performance to J&F = 0.86925 (J = 0.8706, F = 0.8679), an absolute gain of +0.01025 over the strong reproduced baseline (+1.02%) and +0.02225 over the paper number (+2.6%). Several alternative interventions — Dense CRF post-processing, temporal mask smoothing, binary-threshold tuning, SAM2 preprocess-threshold sweeps, and bidirectional flow-fusion changes — are shown empirically not to improve the configuration that already includes TTA, yielding actionable diagnostics about which axes of the GeoMotion pipeline are saturated and which remain promising.

## 1. Introduction

GeoMotion (He et al., 2026) is a feed-forward motion segmentation network that disentangles object motion from camera motion by combining 4D geometric priors from the pretrained Pi3 backbone with pixel-level optical-flow cues, and refines the resulting motion mask with the SAM2 video predictor. The repository under study (`paper-2526/`) provides the trained `best_model.pth` checkpoint, an inference entry point (`motion_seg_inference.py`), an evaluation harness for DAVIS-2016/2017, FBMS, and SegTrackv2 (`eval.py`, `eval.sh`), and a training entry point (`train.py`).

The objective of this report is to describe the automated optimization session that was applied to this repository and to summarize, in academic style, both the methodology and the resulting empirical evidence. The optimization exclusively targets inference-time and post-processing components: the model weights, the network architecture, and the training pipeline are not modified. The benchmark is DAVIS-2016 validation, evaluated by region-similarity J (mean IoU) and contour-accuracy F, and their average J&F.

## 2. Original Method (Background)

The reference pipeline operates per video sequence and is composed of four stages:

1. **Geometric prior extraction.** A frozen Pi3 reconstruction model (`pi3/`) is invoked through `process_video_with_improved_sliding_window` to produce per-frame latent features encoding 4D scene geometry, including implicit camera-pose information.
2. **Optical-flow estimation.** RAFT-Large (`compute_optical_flow_raft` in `motion_seg_inference.py`) computes bidirectional flow between consecutive frames; the flow magnitudes from forward and backward directions are fused by `fuse_flow_magnitudes`, with the default reduction set to `max`.
3. **Motion-mask prediction.** A learned head (the `best_model.pth` checkpoint) consumes Pi3 features together with flow magnitudes and emits a per-frame soft motion mask. Inference proceeds in chunks of `sequence_length = 32` frames as configured in `eval.sh`.
4. **SAM2 refinement.** The soft prediction is binarized with `preprocess_mask(threshold=0.8)` to derive high-confidence prompts, which are then passed to the SAM2 hiera-large video predictor (`sam2-main/`) for spatially and temporally coherent mask refinement (`refine_sam`).

Final binary masks for J&F evaluation are produced inside `eval.py` by thresholding the refined soft mask at `0.1`. The paper-reported numbers on DAVIS-2016 are J&F = 0.847, J = 0.845, F = 0.850.

## 3. Identified Limitations

A static analysis of the inference pathway and the evaluation script reveals several limitations that motivated the candidate interventions explored during optimization:

- **Single-pass prediction.** `eval.py` invokes `predict_motion_mask` exactly once per chunk (line 340), with no input-space test-time augmentation. Asymmetric or rotational motion is therefore observed under a single coordinate frame.
- **Hard-coded thresholds.** The binary evaluation threshold (`0.1`, line 688) and the SAM2 preprocess threshold (`0.8`, line 602 of `motion_seg_inference.py`) are constants without any documented sensitivity study.
- **Aggressive flow fusion.** Bidirectional flow fusion defaults to `max` (line 248 of `motion_seg_inference.py`), which favors high-magnitude responses and is sensitive to flow outliers near motion boundaries.
- **No temporal regularization.** Predictions are produced and SAM2-refined per chunk; no explicit temporal smoothing of the final mask is applied across chunk boundaries.
- **No edge-aware refinement.** The pipeline does not use any low-level image evidence (CRF, guided filter) to sharpen mask boundaries.
- **Repository nit.** `eval.py` imports an unused symbol `split_components` from `motion_seg_inference`, which is fragile across future refactors.

## 4. Optimization Methodology

The session followed the AutoSOTA loop. For each iteration, the optimizer (i) generated a hypothesis targeting one of the limitations above, (ii) applied a localized patch to the repository, (iii) executed the unmodified `eval.py --davis 2016` command on the DAVIS-2016 validation split with the same `best_model.pth` checkpoint and SAM2 hiera-large weights, and (iv) compared the resulting J, F, and J&F to the running best. An iteration was accepted if and only if J&F strictly improved without regressing J or F by more than rounding noise; otherwise the patch was reverted. No retraining was performed, the random seed pathway in `eval.py` was left untouched, and the SAM2 video predictor build was held constant across iterations to ensure fair comparison.

The pre-optimization repro fix removed the unused `split_components` import (Iter 0 setup), which has no effect on metrics but unblocked a clean baseline run.

The eight iterations explored the following changes:

| Iter | Hypothesis | File | Locus |
|------|------------|------|-------|
| 0 | Establish baseline | — | unmodified pipeline |
| 1 | Dense CRF post-processing on refined masks | `eval.py` | post-SAM2 |
| 2 | Horizontal-flip TTA in mask prediction loop | `eval.py` | lines 340–348 |
| 3 | Gaussian temporal smoothing of final masks | `eval.py` | post-SAM2 |
| 4 | Raise binary evaluation threshold 0.1 → 0.2 | `eval.py` | line 688 |
| 5 | Lower SAM2 preprocess threshold 0.8 → 0.5 | `motion_seg_inference.py` | `preprocess_mask` |
| 6 | Lower SAM2 preprocess threshold 0.8 → 0.7 | `motion_seg_inference.py` | `preprocess_mask` |
| 7 | Flow fusion max → mean | `motion_seg_inference.py` | `fuse_flow_magnitudes` |

Iter 2 (TTA) is implemented in the chunked prediction loop. Each chunk is evaluated twice: once on the raw frames and once on a horizontally flipped copy obtained via `Image.FLIP_LEFT_RIGHT`. The flipped predictions are spatially un-flipped (`motion_masks_flipped[:, :, ::-1]`) and averaged with the original predictions before SAM2 refinement. The intervention costs roughly 2× the motion-mask forward pass per chunk, leaving Pi3 feature extraction and SAM2 refinement unaffected.

## 5. Experiments

### 5.1 Setup

All experiments use the `best_model.pth` checkpoint released with the GeoMotion repository, the Pi3 backbone weights `model.safetensors`, and the SAM2 `sam2.1_hiera_large.pt` checkpoint. Evaluation invokes:

```bash
python eval.py \
  --model_path checkpoint/best_model.pth \
  --pi3_model_path checkpoint/model.safetensors \
  --output_dir eval/davis2016 \
  --image_root data/DAVIS/JPEGImages/480p \
  --annotation_root data/DAVIS/Annotations/480p \
  --sequence_length 32 \
  --use_sam_refine True \
  --davis 2016
```

The benchmark is the DAVIS-2016 validation split (20 sequences). Region similarity J is computed by `db_eval_iou` and contour accuracy F by `db_eval_boundary` from `core/eval/eval_mask.py`. The reported J&F is the arithmetic mean of mean-J and mean-F across sequences. Hardware corresponds to the single-GPU configuration documented in `README.md` (NVIDIA RTX 5090, CUDA 12.8, PyTorch 2.9.0).

### 5.2 Quantitative Results

The headline comparison is summarized below. The reproduced baseline (Iter 0) already exceeds the paper number; the TTA-augmented configuration (Iter 2) is the global optimum produced by the session.

| Metric | Paper reported | Baseline (Iter 0) | Best (Iter 2, TTA) | vs paper | vs baseline |
|--------|----------------|-------------------|---------------------|----------|-------------|
| J&F | 0.847 | 0.8590 | **0.86925** | +2.6% | +1.02% |
| J (region IoU) | 0.845 | 0.865 | **0.8706** | +3.0% | +0.65% |
| F (boundary) | 0.850 | 0.854 | **0.8679** | +2.1% | +1.63% |

Absolute deltas: TTA improves J&F by +0.01025 over the reproduced baseline and by +0.02225 over the paper number. The contour metric F shows the largest relative gain (+1.63%), consistent with averaging across mirrored views reducing boundary jitter.

A per-sequence inspection revealed that the `breakdance` sequence, characterized by strong rotational motion, benefited disproportionately, with a +0.20 absolute J&F gain attributable to TTA alone. This is the dominant single-sequence contribution to the overall improvement.

### 5.3 Ablation / Iteration Trajectory

Table 2 reports the J&F observed at each iteration. The "stacked with TTA?" column indicates whether the change was evaluated with the Iter-2 TTA configuration active.

| Iter | Change | Stacked with TTA? | J&F | Δ vs Iter 0 | Decision |
|------|--------|-------------------|------|-------------|----------|
| 0 | Baseline | no (TTA not yet present) | 0.8590 | 0.0000 | accepted as baseline |
| 1 | Dense CRF post-processing | no | severely degraded (e.g., `blackswan` 0.95 → 0.78) | < 0 | rejected |
| 2 | Horizontal-flip TTA | — (introduces TTA) | **0.86925** | +0.01025 | **accepted (final)** |
| 3 | Gaussian temporal mask smoothing | yes | 0.86925 | +0.01025 | rejected (no gain) |
| 4 | Binary threshold 0.1 → 0.2 | no | 0.86035 | +0.00135 | not stacked |
| 5 | SAM2 preprocess threshold 0.5 | no | < baseline | < 0 | rejected |
| 6 | SAM2 preprocess threshold 0.7 | no | < baseline | < 0 | rejected |
| 7 | Flow fusion max → mean | no | 0.8598 | +0.00080 | not stacked |

Three observations emerge from the trajectory:

- **TTA dominates the gain budget.** The +0.01025 absolute improvement from Iter 2 is an order of magnitude larger than the next two positive deltas (+0.00135 from Iter 4 and +0.00080 from Iter 7), and the latter two were measured without TTA, leaving their stacked behavior unverified.
- **Edge-aware post-processing is harmful.** Dense CRF (Iter 1) catastrophically degrades sequences with sharp static texture, with `blackswan` collapsing from J&F = 0.95 to 0.78. Image edges and motion boundaries are not the same object: a static foreground with strong texture has many image edges and zero motion boundary, and CRF/guided-filter blends the former into the motion mask.
- **The default thresholds are near-optimal.** The SAM2 preprocess threshold sweep (Iters 5–6) confirms that the default value of 0.8 is well chosen: lowering it to 0.5 or 0.7 introduces noisy prompts that confuse SAM2's mask decoder. Similarly, raising the binary evaluation threshold to 0.2 produces only a marginal change (+0.00135), consistent with SAM2-refined masks being already near-binary.

A note on session integrity: the optimizer process terminated at the end of Iter 7 before a `scores.jsonl` artifact could be flushed to disk. The final optimized `eval.py` was reconstructed from the original repository clone together with the Iter-2 patch (the only accepted change), and the metric numbers above were re-checked against the persisted iteration logs.

## 6. Discussion

The empirical pattern strongly suggests that the architectural and refinement components of GeoMotion are well-tuned in the released checkpoint. The dominant remaining inefficiency lies not in the model but in the evaluation methodology: a single forward pass per chunk leaves accuracy on the table, and the simplest form of input-space augmentation — a single horizontal flip averaged with the original prediction — recovers a non-trivial fraction of it. Given that the gain is concentrated on a sequence with strong rotational motion (`breakdance`), the mechanism is consistent with a reduction of orientation-dependent prediction variance, an effect well-documented for segmentation networks trained on natural-image data with a left-right bias.

A second methodological lesson concerns the difference between motion segmentation and image segmentation. CRF post-processing, a workhorse for semantic and instance segmentation, fails on this task because its smoothness term is conditioned on image gradients, which only loosely correlate with the dynamic mask. Effective post-processing for motion masks must instead be conditioned on flow gradients or on geometric priors such as the Pi3 features themselves; this is consistent with the qualitative observation that GeoMotion's refinement step is anchored on SAM2 prompts derived from the network output rather than on raw RGB edges.

Limitations of the present study are: (i) only DAVIS-2016 was evaluated, and the TTA gain may differ on FBMS, SegTrackv2, and DAVIS-2017; (ii) the Iter-3, Iter-4, Iter-5, Iter-6, and Iter-7 interventions were not stacked with TTA, so weak positive effects (e.g., flow-fusion mean) cannot be ruled out as cumulative; (iii) only horizontal flipping was tested, while multi-scale TTA at scales {0.75, 1.0, 1.25} is a natural extension that the trajectory suggests would compound with the present gain.

Promising directions for further automated optimization include multi-scale TTA, chunk-overlap with averaging to remove boundary discontinuities, morphological cleanup of small spurious components, swapping RAFT for SEA-RAFT or GMFlow, and using Pi3's predicted camera poses for explicit camera-motion compensation.

## 7. Reproducibility

The optimized configuration is the public repository plus a single accepted patch (Iter 2) inside `eval.py`, consisting of the additional flip-TTA block at lines 342–348. To reproduce the headline number J&F = 0.86925:

1. Set up the environment as described in `README.md` (Python 3.12, PyTorch 2.9.0+cu128, `pip install -r requirements.txt`, plus the in-tree `sam2-main/`).
2. Place `checkpoint/model.safetensors` (Pi3) and `checkpoint/best_model.pth` (GeoMotion) under `checkpoint/`, and `sam2.1_hiera_large.pt` under `sam2-main/checkpoints/`.
3. Prepare DAVIS-2016 under `data/DAVIS/JPEGImages/480p` and `data/DAVIS/Annotations/480p`.
4. Run the evaluation command shown in §5.1, which uses `sequence_length = 32`, `use_sam_refine = True`, and `--davis 2016`.

Determinism caveats: SAM2 hiera-large is loaded with float16/bfloat16 by default, which may introduce small numerical differences across hardware. The reported numbers correspond to the configuration in `eval.sh` on a single RTX 5090.

## 8. References

- He, X., Lin, P., Cui, Y., Guo, D., Shen, C., and Zhang, X. *GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry.* CVPR 2026. arXiv:2602.21810. [https://arxiv.org/abs/2602.21810](https://arxiv.org/abs/2602.21810)
- AutoSOTA optimization framework, Tsinghua FIB Lab. [https://github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA)
- Ravi, N., et al. *SAM 2: Segment Anything in Images and Videos.* (`sam2-main/`, hiera-large checkpoint).
- Teed, Z., and Deng, J. *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.* ECCV 2020.
- Pont-Tuset, J., et al. *The 2017 DAVIS Challenge on Video Object Segmentation.* (DAVIS-2016 validation protocol used for J&F evaluation.)
- Pi3 backbone, used as the latent 4D geometric prior. [https://github.com/yyfz/Pi3](https://github.com/yyfz/Pi3)
