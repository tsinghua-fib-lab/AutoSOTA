# MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention — A Technical Report on Automated Optimization

## Abstract

This technical report documents an automated optimization study conducted on the
official CVPR 2026 release of MixerCSeg, an efficient mixer architecture for
pixel-level crack segmentation that combines convolutional, Transformer-style
and Mamba-inspired pathways within a single encoder. The reference checkpoint
is held fixed throughout this study, in accordance with the constraint that the
paper's pretrained weights must not be re-trained or replaced. Optimization is
therefore restricted to the inference path, comprising the model's forward pass
and any deterministic post-processing applied before the evaluation script
consumes the predicted maps. Across eleven optimization iterations plus a
baseline, the best configuration improves the mean Intersection-over-Union
(mIoU) from 0.9151 to 0.9198 (+0.51%), Optimal Dataset Scale F1 (ODS) from
0.9095 to 0.9149 (+0.59%), and Optimal Image Scale F1 (OIS) from 0.9226 to
0.9292 (+0.72%). The winning configuration (commit `36c39335ae`, iteration 3)
applies a two-pass test-time augmentation (TTA) that ensembles the original
512×512 prediction, a horizontally flipped 512×512 prediction, and an
up-sampled 640×640 prediction obtained at scale factor 1.25. Two minor source
patches in the HOG-edge gating module were required to make multi-scale TTA
numerically and operationally stable. Several alternative ideas — three-scale
TTA, photometric TTA, bilateral and Laplacian post-filters, and morphological
blending — were attempted but either timed out under the 900-second evaluation
budget or were neutral-to-regressive on mIoU. The study is fully reproducible
with two new command-line flags (`--use_tta`, `--use_morph`).

## 1. Introduction

MixerCSeg [1] addresses pixel-level crack segmentation, where the target
structures are thin, elongated, and locally ambiguous against textured
backgrounds. The published model achieves state-of-the-art accuracy with only
2.05 GFLOPs and 2.54 M parameters, by combining a CNN-like spatial pathway, a
Transformer-style global pathway and a Mamba-inspired sequential pathway inside
a single TransMixer encoder, together with a Direction-guided Edge Gated
Convolution (DEGConv) and a Spatial Refinement Multi-Level Fusion (SRF)
decoder.

This report studies whether the released checkpoint can be improved further
without retraining, under conditions consistent with the AutoSOTA optimization
protocol [2]: (i) the pretrained weights are immutable, (ii) the evaluation
script (`eval/evaluate.py`) is immutable, and (iii) each candidate
configuration must finish within a 900-second wall-clock evaluation budget.
The remaining design space is therefore confined to inference-time modifications
of the model's forward pass and to deterministic post-processing applied before
metrics are computed.

## 2. Original Method (Background)

The official implementation in this repository is organised around the
`MixerCSeg` class defined in `models/segmentor/MixerCSeg.py`. The model is
instantiated by `build_MixerCSeg` with the embedding configuration
`embed_dim = [16, 32, 64, 128]`, `depths = [1, 1, 1, 1]` and
`state_dim = [8, 8, 16, 16]`. The encoder is a `VSSEncoder` built from VSS
blocks (`models/encoder/vss_block.py`) that contain MLP, gating and selective
state-space operators borrowed from VMamba. The decoder `SRFModule`
(`models/decoder/SRF.py`) is a Spatial Refinement Multi-Level Fusion module
operating at the input resolution `(load_width, load_height)`.

A key structural element is the `HoGEdgeGateConv` block in
`models/layers/hog_edge.py`. It computes a coarse histogram-of-oriented-
gradients descriptor over a 2×2 patch partition of each feature map and uses it
as a gate for an edge-sensitive `EdgeConv`. The HOG branch crops feature maps
to integer multiples of an 8×8 cell size, computes Sobel gradients, bins the
absolute gradient direction into `nbins` orientations, and modulates the
output of an axis-decoupled `EdgeConv`.

Training is driven by `main.py`, which combines `BCEWithLogitsLoss` (weight
0.87) with a soft `DiceLoss` (weight 0.13). The default scheduler is
`PolyLR`, optimization uses AdamW (`lr = 5e-4`, `weight_decay = 0.01`), and
input images are resized to 512×512. Evaluation is performed by
`eval/evaluate.py`, which computes mIoU, ODS and OIS by sweeping a 100-bin
threshold over [0, 1) on the predicted probability maps, and additionally
reports a degenerate F1/Precision/Recall triplet at `thresh = 0` through
`cal_prf_metrics[0,3]`.

## 3. Identified Limitations

Three limitations of the released inference pipeline motivate this study.
First, a single forward pass at 512×512 produces a single deterministic
prediction; the model is sensitive to orientation, since crack structures are
locally near-symmetric and many narrow cracks fall close to the decision
boundary. Second, the HOG branch in `hog_edge.py` is implemented under the
implicit assumption of a 512×512 input and an 8×8 cell, leaving the module
fragile when the spatial dimensions of intermediate features deviate from this
shape, which is precisely what happens under multi-scale inference. Third, the
2×2 patch reshape inside `HoGEdgeGateConv` (`image2patches` /
`patches2image`) is realised via `einops.rearrange` followed by tensor views
that assume contiguity. Flip-based augmentation breaks that assumption and
silently produces incorrect feature memory layouts.

Together, these issues prevent any naive use of multi-scale or flip-based TTA
on the released model and are therefore prerequisites for the rest of the
optimization.

## 4. Optimization Methodology

The optimization protocol follows AutoSOTA [2]: an iterative loop in which each
candidate change is committed, evaluated end-to-end on the target dataset, and
either retained or rolled back depending on its measured effect on mIoU under
the 900 s timeout. Twelve checkpoints in total were produced — the baseline
plus eleven optimization iterations. The accepted set of changes is summarised
in Table 1.

**Table 1. Accepted modifications and their files of effect.**

| Change | File(s) | Effect |
|--------|---------|--------|
| TTA: multi-scale + flip ensemble | `models/segmentor/MixerCSeg.py`, `main.py` | +0.51% mIoU |
| `.view()` → `.reshape()` for non-contiguous tensors | `models/layers/hog_edge.py` | Enables multi-scale TTA |
| Adaptive HOG cell size for small feature maps | `models/layers/hog_edge.py` | Enables multi-scale TTA |
| Morphological post-processing (optional) | `models/segmentor/MixerCSeg.py`, `main.py` | No mIoU gain, kept as option |
| Add `--use_tta` and `--use_morph` CLI flags | `main.py` | Enables TTA/morph at inference |

The TTA design is implemented in `MixerCSeg._forward_tta`. For each scale in
`tta_scales`, the input is bilinearly resampled to a multiple-of-64
resolution (so that the encoder's stride-32 path remains valid), passed through
`_forward_single`, resampled back to the original 512×512 grid in
probability space, and accumulated. Flip augmentation is restricted to the
base scale to fit inside the timeout. The averaged probability map is
re-projected to logits via a numerically clipped inverse sigmoid in order to
remain compatible with the loss reported during evaluation.

Two source patches in `hog_edge.py` are necessary preconditions. The first
substitutes the contiguity-sensitive `.view()` calls with `.reshape()` so that
horizontally and vertically flipped activations propagate through the
`image2patches` / `patches2image` reshape without raising a stride error.
The second adaptively shrinks the HOG cell size to `max(1, H)` and
`max(1, W)` whenever an intermediate feature map is smaller than the nominal
8×8 cell, which can occur at deeper stages when the input is up-sampled by
1.25× and then progressively halved. Without these two patches, multi-scale
TTA crashes before any metric can be reported.

A morphological post-processing branch (`_apply_morph`) is also added: it
threshold-binarises the sigmoid probability map at 0.5, applies a closing
followed by an opening realised with max-pool / inverse-max-pool primitives,
and blends the result back with the original probability map at α = 0.3. The
branch is exposed through `--use_morph` but, as documented below, does not
improve mIoU.

Two new command-line switches are added to `main.py`: `--use_tta` enables the
TTA path, and `--use_morph` enables the morphological branch.

## 5. Experiments

### 5.1 Setup

Experiments use the official MixerCSeg checkpoint and the official evaluator
in `eval/evaluate.py`. The four crack-segmentation benchmarks shipped with the
release are CamCrack79, DeepCrack, Crack500 and CrackMap; the optimization
results reported here are computed by the optimization driver on the dataset
fixed in the released `test.py`. Predictions are written by `test.py` /
`main.py` as 8-bit PNG probability maps and consumed by the evaluator without
any change to thresholds or sweep granularity.

Each candidate configuration is constrained by a wall-clock budget of 900 s
per evaluation run. Measured costs are approximately 136 s per forward pass at
512×512 and approximately 470 s per forward pass at 640×640. Six metrics are
tracked: mIoU, ODS and OIS (each computed by maximising over a 100-bin
threshold sweep), together with the degenerate F1, Precision and Recall at
`thresh = 0` reported by `cal_prf_metrics[0,3]`.

### 5.2 Quantitative Results

Table 2 reports the headline metrics for the released checkpoint and for the
best optimized configuration (commit `36c39335ae`, iteration 3).

**Table 2. Baseline vs. best metrics on the optimization dataset.**

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| mIoU | 0.9151 | 0.9198 | +0.51% |
| ODS | 0.9095 | 0.9149 | +0.59% |
| OIS | 0.9226 | 0.9292 | +0.72% |
| F1 (thresh = 0) | 0.5949 | 0.5861 | -1.5% |
| Precision (thresh = 0) | 0.4236 | 0.4147 | -2.1% |
| Recall (thresh = 0) | 0.9991 | 0.9995 | +0.04% |

The three threshold-swept metrics that match the paper's evaluation
methodology — mIoU, ODS and OIS — all improve monotonically. The
degenerate F1, Precision and Recall reported at `thresh = 0` regress
slightly. These three numbers are obtained from
`cal_prf_metrics[0, 3]`, i.e. the very first bin of the sweep, where
predictions are effectively un-thresholded; they are dominated by the
near-saturated recall and therefore do not reflect operating-point
performance. The improvement in mIoU, ODS and OIS is therefore the
faithful summary of the optimization effect.

### 5.3 Ablation / Iteration Trajectory

Table 3 summarises which categories of change were retained, neutral or
discarded. Numerical deltas are reported in mIoU points relative to the
baseline of 0.9151.

**Table 3. Iteration trajectory grouped by intervention category.**

| Intervention | Outcome | mIoU delta | Reason |
|--------------|---------|-----------|--------|
| Flip-only TTA (hflip + vflip) | Accepted | +0.27% | Cheap orientation ensemble at base scale |
| Multi-scale pass at 1.25× (640×640) with hflip at base | Accepted (final) | +0.51% | Captures finer crack details missed at 512×512 |
| 3-scale TTA | Rejected | — | Timed out under 900 s budget |
| Photometric TTA (brightness/contrast) | Rejected | — | Timed out; GPU sync from tensor clamping |
| Additional vflip in 3-pass config | Rejected | — | ~878 s estimated, exceeded 900 s |
| Bilateral filter post-processing | Rejected | — | CPU↔GPU transfer overhead exceeded timeout |
| Laplacian prediction sharpening | Rejected | −0.01% | Marginally regressive |
| Morphological post-processing | Kept as option | 0.00% | No mIoU/ODS gain; only improves F1 at `thresh = 0` |
| Training-based optimization (1 epoch fine-tune) | Rejected | — | ~30–40 min/epoch; weights are immutable |

The decisive observation is that flip-only TTA contributes the first +0.27%
mIoU and the additional 1.25× scale pass with horizontal flip at the base
scale contributes the remaining +0.24%, for a cumulative +0.51%. No tested
alternative recovered any further mIoU within the 900 s budget.

The two HOG-module patches do not themselves change mIoU on the baseline
single-pass configuration; their role is strictly to keep the TTA paths
numerically valid. They are therefore neutral when measured in isolation but
necessary for the +0.51% improvement to be obtainable.

## 6. Discussion

The improvement of +0.51% mIoU, +0.59% ODS and +0.72% OIS is small in
absolute terms but is obtained entirely without modifying the released
weights, without modifying the evaluator, and without exceeding the
900-second-per-run budget. This is consistent with the structural
characteristics of MixerCSeg: the released checkpoint is already
near-converged on these benchmarks (the baseline mIoU is 0.9151 on top of a
2.05 GFLOPs / 2.54 M-parameter model), so the only remaining inference-time
leverage is to reduce variance through ensembling and to expose finer detail
through limited multi-scale evaluation.

The pattern of negative results is also informative. Three-scale and
four-pass TTA configurations were rejected purely on timing grounds; their
effect on mIoU was not measured because they could not be evaluated within
the wall-clock budget. Post-processing strategies that crossed the CPU/GPU
boundary (bilateral filtering) or that introduced large element-wise
clamping kernels (photometric TTA) were also dominated by synchronization
overhead rather than by their numerical effect. Pure GPU-side post-processing
that stays inside the forward graph — such as the morphological blend and
Laplacian sharpening — fits within the timeout but does not move mIoU,
indicating that the residual error after TTA is no longer concentrated at the
crack boundary in a way that simple structural filters can recover.

Several research-grade ideas were left unexplored because they would require
retraining and therefore violate the immutable-weights constraint. These
include a Focal Tversky loss tailored to the extreme foreground/background
imbalance of crack data, an auxiliary boundary-supervised edge head with a
boundary-aware loss, strip and dilated convolutions in the SRF decoder to
better match the elongated topology of cracks, replacement of PolyLR with a
cosine schedule plus warmup, stochastic weight averaging or EMA over training
checkpoints, gradient clipping for the SSM components, and a lightweight
SE-style channel attention (≈128 parameters) inside the decoder. These are
documented as forward-looking suggestions for future runs that are permitted
to retrain.

## 7. Reproducibility

The repository is unchanged with respect to the release except for the
modifications described in Section 4, which are confined to
`models/segmentor/MixerCSeg.py`, `models/layers/hog_edge.py` and `main.py`.
The environment follows the installation recipe in `README.md` (Python 3.10,
PyTorch 2.1.0 + CUDA 11.8, `mmcv-full` via `openmim`, NumPy 1.23, and a local
build of the VMamba selective-scan kernel).

Inference under the winning configuration is reproduced by invoking the
existing test/evaluation entry points with the two new flags introduced by
this study:

```bash
python main.py --dataset_path [your_dataset_path] --phase test --use_tta
python eval/evaluate.py --result_path [your_results_path]
```

`--use_tta` activates the two-pass ensemble (512×512 original, 512×512
horizontal flip, 640×640 scale-up) implemented in
`MixerCSeg._forward_tta`. `--use_morph` is exposed for completeness but
should be omitted to reproduce the best mIoU. The default TTA scale list is
`[1.0, 1.25]` and is exposed as `tta_scales` on the `MixerCSeg` module.

The best checkpoint of the optimization study corresponds to commit
`36c39335ae` (iteration 3). All reported numbers in Tables 2 and 3 are
obtained with that configuration under the official evaluator
`eval/evaluate.py`, with no modification to the metric implementation or to
the released weights.

## 8. References

[1] Z. Zhao, Z. Ding, P. Niu, W. Sun, F. Guo. *MixerCSeg: An Efficient Mixer
Architecture for Crack Segmentation via Decoupled Mamba Attention*. CVPR
2026. arXiv:2603.01361.

[2] AutoSOTA. *Automated optimization protocol for reproducible state-of-the-
art search.* tsinghua-fib-lab/AutoSOTA, GitHub.
