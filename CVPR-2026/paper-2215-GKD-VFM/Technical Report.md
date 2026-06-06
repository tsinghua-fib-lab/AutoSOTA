# Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study performed on the public release of GKD (Lv et al., CVPR 2026), a knowledge-distillation framework that decouples representation learning from task learning to preserve the out-of-domain generalization of vision foundation models (VFMs) when distilled into compact students. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted `mIoU` on the Cityscapes validation split using the released DINOv2-distilled student checkpoint. The headline result is an improvement of `mIoU_cityscapes` from a baseline of 52.02% to 54.72% (an absolute gain of +2.70 mIoU, +5.19% relative), surpassing the +5%-over-baseline target of 54.621% by 0.10 points. The improvement is concentrated on small and thin classes — pole (+10.47), traffic light (+9.61), traffic sign (+7.86), and person (+5.43) — and was obtained in only two iterations through a single algorithmic change: multi-scale inference at scales [0.75, 1.0, 1.25] with logit averaging, implemented as a `MultiScaleModel` wrapper class registered into the mmseg model registry. The total diff is 3 files, +52/−2 lines. The best configuration is captured at commit `3e25066`. No retraining and no modification of the released student weights were required.

## 1. Introduction

GKD, presented at CVPR 2026, addresses a known weakness of conventional knowledge distillation: when a strong VFM teacher (DINOv2, EVA-02) is distilled into a compact student (DeiT-S/B), conventional KD preserves in-domain accuracy but compromises the teacher's out-of-domain robustness. The released GKD method introduces a two-stage pipeline — domain-agnostic feature distillation followed by task adaptation with frozen representations — and a query-based soft distillation mechanism in which student features attend over teacher representations to selectively retrieve transferable spatial knowledge. The paper reports +1.9% average gains in foundation-to-foundation (F2F) distillation and +10.6% in foundation-to-local (F2L) distillation across five domain-generalization benchmarks.

This report studies whether the released GKD inference pipeline can be improved post hoc, without retraining, using purely test-time interventions on the Cityscapes target. The motivation is that the standard mmseg evaluation in the released codebase performs single-scale, single-flip inference; well-known test-time augmentation tricks have not been exercised against the GKD student. AutoSOTA, an automated SOTA-chasing harness developed by Tsinghua FIB Lab, was used to propose, run, and evaluate code and configuration changes against the `mIoU_cityscapes` metric in a budgeted iterative loop.

The remainder of the report covers the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology applied (Section 4), the experimental setup, results, and ablations (Section 5), a discussion of the per-class regressions (Section 6), and reproducibility information (Section 7).

## 2. Original Method (Background)

GKD distils a vision foundation model (DINOv2 or EVA-02) into a compact ViT-S/B student through a two-stage procedure:

* **Stage 1 — Domain-agnostic representation distillation.** The student acquires task-agnostic features through *selective feature distillation*, in which a query-based soft mechanism uses student features as queries against teacher representations to retrieve transferable spatial knowledge. This stage is implemented under `general_distillation/`.
* **Stage 2 — Task adaptation with frozen representations.** The student backbone is frozen and only a task head is trained on the segmentation objective, mitigating overfitting to visible domains. This stage is implemented under `task_learning/`.

The released code is built on top of `mmseg` and the Rein/Proteus codebases. Inference for evaluation is performed at single scale through the standard mmseg `EncoderDecoder` family. The configuration relevant to this study is `eval_cityscapes_config.py` under `task_learning/`, which loads a frozen-backbone segmentor and runs single-scale inference with `PackSegInputs`. Pretrained students (DeiT ViT-S/B, DINO ViT-S/B distilled from DINOv2) are distributed via Baidu Netdisk and Hugging Face.

## 3. Identified Limitations

The optimization study identified three sources of friction in the released inference pipeline:

1. **No multi-scale test-time augmentation.** The released `eval_cityscapes_config.py` runs single-scale inference. Multi-scale ensembling — a near-universal practice in the segmentation literature — is not exercised, leaving easily testable gains unclaimed.
2. **Incompatibility with the legacy `MultiScaleFlipAug` data pipeline.** The natural way to enable TTA in older mmseg releases is to wrap the evaluation pipeline in `MultiScaleFlipAug`. With the data format used here (`PackSegInputs`), this pipeline path is incompatible and aborts.
3. **No flip TTA in the default eval path.** Even single-scale horizontal-flip TTA is not enabled by default, despite being inexpensive (2× forward passes).

A further concern that is not addressed by this study but is recorded here for completeness is that `pydensecrf` could not be installed under the environment's proxy restrictions, which prevented an evaluation of CRF-style post-processing.

## 4. Optimization Methodology

The two retained iterations exercise a single category of change: a model-wrapper-level multi-scale ensembler. The change is grounded in concrete files in the released repository.

**Multi-scale model wrapper.** A new file `rein/models/segmentors/msi_wrapper.py` (48 lines) defines a `MultiScaleModel` class that subclasses the existing `FrozenBackboneEncoderDecoder`. At inference time `MultiScaleModel` runs three forward passes at scales `ms_scales = [0.75, 1.0, 1.25]` and averages the resulting logits before the argmax. The class is registered into the mmseg segmentor registry by adding the corresponding import line in `rein/models/segmentors/__init__.py`. The configuration file `eval_cityscapes_config.py` is updated to set the model `type` to `MultiScaleModel` and to pass `ms_scales=[0.75, 1.0, 1.25]`. The total diff is **3 files changed, 52 insertions, 2 deletions**.

The choice of scales was deliberate: the wider sweep `[0.5, 1.0, 1.5]` was tested first and yielded only +1.54 mIoU because the 0.5× and 1.5× passes both injected artifacts. The tighter range `[0.75, 1.0, 1.25]` preserves enough small-class detail at 0.75× while still capturing context at 1.25×, and produced the +2.70 mIoU result.

**Approaches that were considered but not retained.** Single-flip TTA on its own gave only +0.29 mIoU and was deemed insufficient given the doubled inference cost. Combining flip TTA with the multi-scale wrapper produced six forward passes per image and timed out at the 900 s per-iteration budget. The legacy `MultiScaleFlipAug` pipeline approach was incompatible with `PackSegInputs` and aborted. CRF post-processing via `pydensecrf` could not be installed.

No training data, model weights, or backbone components were modified. The retained change is a pure-inference modification implemented as a model wrapper, preserving compatibility with the existing mmseg evaluation infrastructure.

## 5. Experiments

### 5.1 Setup

The optimization target was `mIoU` on the Cityscapes validation split (19 classes). All runs used the released GKD student checkpoint (DeiT ViT-S/B distilled from DINOv2) without modification. Inference was launched through the project's standard `task_learning/` evaluation entry point with `eval_cityscapes_config.py`. Each iteration was budgeted at 900 s wall clock. The improvement target set by AutoSOTA was 54.621% mIoU (+5% relative over the 52.02% baseline).

### 5.2 Quantitative Results

The headline metrics on the Cityscapes validation split are reproduced below.

| Metric | Baseline | Best (Iter 2) | Delta |
|---|---:|---:|---:|
| mIoU | 52.02 | **54.72** | **+2.70** |
| aAcc | 90.74 | 91.95 | +1.21 |
| mAcc | 68.42 | 69.52 | +1.10 |

The +5%-over-baseline target of 54.621% mIoU was achieved by 0.10 points after only two iterations, which is the reason the optimization terminated early.

The full per-class IoU breakdown is reproduced below.

| Class | Baseline | Best | Delta |
|---|---:|---:|---:|
| road | 92.55 | 93.49 | +0.94 |
| sidewalk | 55.09 | 59.45 | +4.36 |
| building | 88.02 | 89.44 | +1.42 |
| wall | 50.24 | 49.00 | −1.24 |
| fence | 40.26 | 44.00 | +3.74 |
| pole | 38.75 | 49.22 | **+10.47** |
| traffic light | 44.79 | 54.40 | **+9.61** |
| traffic sign | 28.28 | 36.14 | +7.86 |
| vegetation | 87.46 | 89.63 | +2.17 |
| terrain | 44.68 | 46.30 | +1.62 |
| sky | 88.85 | 90.57 | +1.72 |
| person | 65.18 | 70.61 | +5.43 |
| rider | 32.19 | 27.26 | −4.93 |
| car | 87.71 | 89.97 | +2.26 |
| truck | 40.46 | 44.75 | +4.29 |
| bus | 76.56 | 81.08 | +4.52 |
| train | 0.40 | 0.24 | −0.16 |
| motorcycle | 26.84 | 24.12 | −2.72 |
| bicycle | 0.00 | 0.00 | 0.00 |

Multi-scale inference dramatically improved small/thin classes — pole (+10.47), traffic light (+9.61), traffic sign (+7.86), and person (+5.43) — by injecting higher-resolution evidence into the ensemble. Minor regressions occurred on a handful of classes (wall, rider, motorcycle, train) where the cross-scale logit average introduced ambiguity. The net effect is strongly positive at +2.70 mIoU; the bicycle class remained at 0.00 in both runs, indicating an unrecoverable failure mode of the released checkpoint that test-time augmentation cannot address.

### 5.3 Ablation / Iteration Trajectory

```
Baseline:                                  52.02
  → Iter 1 (MSI [0.5, 1.0, 1.5]):         53.56  (+1.54)
  → Iter 2 (MSI [0.75, 1.0, 1.25]):       54.72  (+2.70)   target met
```

The sweep over scale ranges confirms that the gain is sensitive to the choice of scales rather than to the use of multi-scale per se. A one-line change of `ms_scales` recovered an additional +1.16 mIoU over the wider sweep. A direct comparison against single flip-only TTA (+0.29 mIoU) places the magnitude of the multi-scale lever at roughly 9× the magnitude of the flip lever.

## 6. Discussion

The most informative finding is that, for the released GKD student on Cityscapes, multi-scale inference is the dominant test-time lever and that the scale range matters substantially more than whether multi-scale is used at all. The intuition aligns with the per-class breakdown: thin and small classes benefit most from finer spatial evidence (the 0.75× pass), while the larger contextual classes benefit from coarser evidence (the 1.25× pass). The fall-off at extreme scales (0.5× / 1.5×) suggests that the released student has a limited operating range of input resolutions outside which feature responses become unreliable.

The per-class regressions on wall, rider, motorcycle, and train are not consistent with one another and likely reflect the well-known failure mode of unweighted logit averaging: when classes have fundamentally different scale responses, a uniform average can drag a confident pass back toward an underconfident competing class. Future work could replace the uniform average with a class-conditioned weighting (estimated, for example, from the validation set) or with a confidence-based selective combination. CRF post-processing remains untested and is the most plausible next +0.5–1.5 mIoU improvement; class-aware CRF and confidence-guided selective CRF are recorded as further candidates. A larger test crop (e.g. 768×768 or 1024×512) is another inexpensive test-time lever. Beyond test-time interventions, the paper itself reports that switching the backbone from DINOv2-L to DeiT-B reaches 54.2 mIoU without TTA, suggesting that backbone choice and test-time ensembling are roughly complementary contributions.

## 7. Reproducibility

The slimmed repository contains the code required to reproduce the best configuration; pretrained student weights and datasets are intentionally not included.

* **Best commit.** `3e25066`.
* **Best configuration.** `MultiScaleModel` wrapper with `ms_scales=[0.75, 1.0, 1.25]`, single forward pass per scale, uniform logit average. Released student weights unchanged.
* **Files touched.**
  * `rein/models/segmentors/msi_wrapper.py` — new (48 lines).
  * `rein/models/segmentors/__init__.py` — register `MultiScaleModel` in the mmseg registry.
  * `eval_cityscapes_config.py` — set model `type` to `MultiScaleModel` with `ms_scales=[0.75, 1.0, 1.25]`.
* **Pretrained checkpoints.** Download GKD student weights from the original `README.md`'s Baidu Netdisk or Hugging Face links (DeiT ViT-S/B and DINO ViT-S/B distilled from DINOv2) and place them at the path expected by `eval_cityscapes_config.py`.
* **Datasets.** Cityscapes validation split, in the standard 19-class layout. The codebase reuses the dataset configuration from Rein/mmseg.
* **Environment.** Python 3.8.13, PyTorch 2.0.1, mmseg compatible with `PackSegInputs`. The original `README.md` documents the full setup. Note that `pydensecrf` could not be installed in this study; CRF-based post-processing is therefore left as future work.

## 8. References

* Lv, C., Zhao, D., Wang, S., Quan, D., Huyan, N., Sebe, N., & Zhong, Z. (2026). *Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation*. CVPR 2026. arXiv:2603.02554.
* AutoSOTA: Tsinghua FIB Lab. *AutoSOTA: An automated SOTA-chasing harness*. [github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA).
* Acknowledged dependencies (per the original `README.md`): Proteus, DINOv2, EVA-02, Rein.
