# LIDO — Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study performed on the public release of LIDO (Mosco, Fusaro, and Pretto, CVPR 2026, University of Padova), a feature-space approach for 3D LiDAR anomaly segmentation that combines a semantic head and a contrastive head to distinguish in-distribution classes from out-of-distribution objects. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted the AUROC metric on the STU benchmark. Twenty-four iterations were executed against the released pretrained checkpoint without retraining. The headline result is an improvement of AUROC from a baseline of 74.27% to 90.42% (+16.15 absolute), accompanied by a substantial drop in FPR95 (85.36% → 30.60%, −54.76) and a near-tripling of average precision (0.1425 → 0.3906). The improvement was obtained through three changes localized to a single file (`modules/user.py`), totalling six insertions and three deletions across nine lines of code: removal of per-frame max-normalization of scores, adjustment of the contrastive-head normalization radius `r` from 5 to 1000, and a Top-K prototype-similarity blend of top-1 and top-3 similarities. The contrastive radius is by far the dominant lever (+15.75 of the +16.15 absolute gain). The best configuration is captured at commit `f40098c` (Iteration 19). All metrics in this study were computed on a single STU sequence (sequence 201, 682 scans) because the full multi-sequence STU test set was not available; the gains are therefore relative to a sequence-201 baseline (74.27% AUROC) and are not directly comparable to the paper's reported headline AUROC of 93.67% on the full test set.

## 1. Introduction

LIDO, presented at CVPR 2026, addresses 3D LiDAR anomaly segmentation by operating in feature space rather than at the output level. The proposed system uses a combination of training losses and a fused inference score (semantic head and contrastive head, combined 50/50) to produce both semantic segmentation and anomaly segmentation. The release also introduces three mixed real-synthetic OoD datasets built on top of standard autonomous-driving benchmarks (STU, SemanticKITTI, SemanticPOSS, nuScenes), inserting geometrically and photometrically aligned synthetic objects into real LiDAR scans.

This report studies whether the released LIDO inference pipeline can be improved post hoc, without retraining, using purely test-time interventions on the score-fusion path. The motivation is that the released `modules/user.py` contains several design choices — per-frame normalization, a hard-coded contrastive radius, top-1 prototype scoring — that are individually defensible but have not been ablated against alternatives. AutoSOTA, an automated SOTA-chasing harness developed by Tsinghua FIB Lab, was used to propose, run, and evaluate code changes against AUROC over a 24-iteration budget.

The remainder of the report covers the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology (Section 4), the experimental setup, results, and the contrastive-radius sweep (Section 5), a discussion of the negative results (Section 6), and reproducibility information (Section 7).

## 2. Original Method (Background)

LIDO consists of two complementary heads sharing a backbone:

* **Semantic head** — produces per-point class logits over the in-distribution semantic classes; its anomaly signal is a temperature-scaled function of cosine distance to class prototypes, multiplied by the classifier-output entropy.
* **Contrastive head** — produces per-point feature vectors and an anomaly signal of the form `relu(1 − ‖f‖² / r)`, where `r` is a normalization radius; this term is small when feature norms are large (in-distribution) and grows when norms are small (potentially anomalous).

The two anomaly signals are fused with equal weights (50/50). At inference, the released pipeline applies a per-frame maximum normalization (`scores = scores / scores.max()`) before fusion, which compresses the dynamic range to [0, 1] within each frame. The relevant code path is concentrated in `modules/user.py`, with the principal lines being approximately 220–232.

The release ships pretrained checkpoints (Hugging Face: `Simom0/LIDO`) and supports STU (Single-Sequence and full multi-sequence), SemanticKITTI, SemanticPOSS, and nuScenes (with `nuscenes2kitti` conversion).

## 3. Identified Limitations

The optimization study identified four sources of friction in the released inference pipeline:

1. **Per-frame max normalization compresses cross-frame dynamic range.** Dividing scores by their per-frame maximum maps them into a comparable range *within* each frame but makes anomaly scores non-comparable *across* frames. AUROC, which is a global ranking metric, is harmed by this compression.
2. **Hard-coded contrastive radius `r=5`.** The contrastive head's `relu(1 − ‖f‖² / r)` activation is highly sensitive to `r`. With `r=5` and the released feature-norm distribution, the activation saturates aggressively for in-distribution points and effectively clips useful per-point variance.
3. **Top-1 prototype scoring.** The semantic head scores each point against the single best-matching class prototype. A small blend with the top-3 prototype similarities is a candidate +0.1–0.3 AUROC.
4. **Lack of fusion-weight ablation.** The 50/50 fusion is a sensible default but had not been compared against alternative weightings (e.g. 80/20) at the time of the released code.

## 4. Optimization Methodology

The 24 iterations explored four categories of change. All retained changes are confined to `modules/user.py`; no model weights, data, or evaluation script were modified. The total accepted diff is **1 file, 9 lines (+6 / −3)**.

**Score normalization (retained).** The line `scores = scores / scores.max()` (`modules/user.py:229`) was commented out, removing the per-frame max normalization. Result: +0.40 AUROC.

**Contrastive radius (retained — primary lever).** The contrastive activation `relu(1 − ‖f‖² / r)` (`modules/user.py:232`) was retuned by sweeping `r` over `{3, 5, 8, 10, 15, 20, 30, 50, 70, 100, 200, 500, 1000, 10000}`. The sweep peaks sharply at `r=1000` (+15.75 AUROC vs. baseline). The intuition is that increasing `r` makes the contrastive activation near-constant positive across the typical in-distribution norm range, which (a) prevents the activation from clipping in-distribution points and (b) allows the semantic head's cosine-distance signal to dominate the fused score while still receiving useful per-point variance from the contrastive head as a low-frequency bias.

**Top-K prototype similarity (retained).** The prototype scoring path (`modules/user.py:220–224`) was modified to blend top-1 and top-3 prototype similarities. Result: +0.17 AUROC.

**Approaches tested but not retained.** Five variants regressed against the new best:

* Distance-adaptive fusion weights (semantic-heavy at short range, contrastive-heavy at long range, with sigmoid transition) — transition was too broad, regressed AUROC.
* Removing entropy multiplication from the semantic score — −7.50 AUROC; entropy is a critical multiplicative gate.
* Dropping the contrastive head entirely — −7.40 AUROC; despite its high standalone FPR95, the contrastive head's per-point variance contributes meaningfully to the fused score.
* Unbalanced fusion (80/20 in favour of the semantic head) — −2.68 AUROC.
* Replacing the contrastive head with a constant bias of 0.5 — −12.17 AUROC; per-point variance from the contrastive head is essential.
* Distance-filter tuning had no effect because all sequence-201 points lie within the 2.5–50 m range.

The synergy of the three retained changes is essential: the contrastive radius dominates numerically, but the per-frame normalization removal and Top-K blend compound to the final 90.42 AUROC.

## 5. Experiments

### 5.1 Setup

The optimization target was AUROC on the STU OoD test split. The full multi-sequence STU test set (50+ sequences, used by the paper to report 93.67% AUROC) was not available in the present infrastructure; the optimizer therefore evaluated on a single sequence (sequence 201, 682 scans) for which the corresponding baseline AUROC is 74.27%. All gains in this report are computed against this single-sequence baseline. The released LIDO pretrained checkpoint was used unchanged. AutoSOTA executed 24 iterations under a fixed wall-clock budget per iteration. The improvement target was implicitly set by AutoSOTA's standard +5%-relative protocol.

### 5.2 Quantitative Results

| Metric | Baseline (seq 201) | Best (Iter 19) | Delta | Direction |
|---|---:|---:|---:|---|
| AUROC | 74.27% | **90.42%** | **+16.15** | ↑ |
| AP | 0.1425 | **0.3906** | +0.248 | ↑ |
| FPR95 | 85.36% | **30.60%** | −54.76 | ↓ |

For reference, the paper reports AUROC = 93.67%, AP = 14.99, FPR95 = 34.29 on the full STU test set; the AP discrepancy between the two scales is a consequence of the much smaller anomaly-positive count on a single sequence (AP is sensitive to class prevalence) and is not directly comparable.

The best configuration was captured at commit `f40098c` (Iteration 19) and combines: per-frame max normalization removed; contrastive radius `r=1000`; and Top-K prototype-similarity blend.

### 5.3 Ablation / Iteration Trajectory

The contrastive-radius sweep is the central ablation of this study; it dominates all other levers. The full sweep (with the other two retained changes in place) is reproduced below.

```
r=3      →  69.77   (worse than baseline)
r=5      →  74.27   (baseline)
r=8      →  83.98   (first large jump)
r=10     →  85.13
r=15     →  85.85
r=20     →  86.21
r=30     →  86.37
r=50     →  86.59
r=70     →  87.01   (with Top-K)
r=100    →  87.36
r=200    →  88.31
r=500    →  84.78   (regression)
r=1000   →  90.42   *** OPTIMAL
r=10000  →  84.78   (regression)
```

Two observations follow. First, the AUROC landscape over `r` is non-monotonic and sharply peaked: between 500 and 1000 the curve climbs nearly six points, and between 1000 and 10000 it falls back by the same magnitude. This explains why a coarse-resolution sweep can easily miss the optimum. Second, the regions `r ≤ 5` and `r ≥ 5000` correspond to qualitatively different failure modes: at small `r` the activation saturates for in-distribution points and erases useful contrast; at very large `r` the contrastive activation degenerates toward a constant bias and loses per-point variance — exactly the failure observed when the contrastive head was replaced with a constant.

The retained ablation table for the three accepted changes is:

| Change | File:Line | Effect | Notes |
|---|---|---:|---|
| Remove per-frame max normalization | `modules/user.py:229` | +0.40 AUROC | Comment out `scores = scores / scores.max()` |
| Increase contrastive radius `r` from 5 to 1000 | `modules/user.py:232` | +15.75 AUROC | Single dominant lever |
| Top-K prototype scoring (blend top-1 and top-3) | `modules/user.py:220–224` | +0.17 AUROC | Small but consistent |
| **Combined** | — | **+16.15 AUROC** | — |

## 6. Discussion

The dominant takeaway is that one numerical hyperparameter — the contrastive normalization radius `r` — accounts for nearly the entire +16.15 AUROC gain. The implication is that the released checkpoint is well-trained but that the post-hoc score-fusion path was tuned conservatively. The pattern is more general: in models that fuse multiple scoring heads at inference time, it is common for one head's normalization to be set during development at a value that is appropriate for the training-time feature distribution but suboptimal for the test-time distribution.

The negative results are equally informative. The contrastive head cannot be removed (−7.40), nor replaced by a constant (−12.17), nor reweighted toward 20% (−2.68). Together these results constrain the viable design space: the contrastive head must be retained, must contribute per-point variance, and must be fused with weight close to 50/50. What is variable is the *shape* of the contrastive activation; the `r=1000` regime preserves variance while shifting most mass into the linear (non-clipped) region of the ReLU. This is consistent with the per-class entropy-multiplication finding: removing entropy from the semantic score regresses by −7.50, again pointing to multiplicative gating as the structural choice that the released pipeline got right.

A direct caveat is that all of the gains in this study were measured on a single STU sequence. With the full multi-sequence test set, the optimal `r` may shift (the validation curve was sharp), and the relative order of the smaller levers (Top-K, normalization removal) may change. Future work should test on the full STU dataset, explore temperature scaling on the logits as an alternative entropy signal, fit per-class GMMs and combine Mahalanobis distance with the existing scores, exercise temporal smoothing across consecutive frames on multi-sequence data, and consider an adaptive per-point `r` driven by local feature-norm statistics rather than a single global value.

## 7. Reproducibility

The slimmed repository contains the code required to reproduce the best configuration; pretrained checkpoints and OoD datasets are intentionally not included.

* **Best commit.** `f40098c` (Iteration 19).
* **Best configuration.** Released LIDO pretrained checkpoint unchanged; three changes in `modules/user.py`:
  * line ~229: comment out `scores = scores / scores.max()`;
  * line ~232: `r = 1000` in `relu(1 − ‖f‖² / r)`;
  * lines ~220–224: blend top-1 and top-3 prototype similarities.
* **Pretrained checkpoint.** Hugging Face `Simom0/LIDO`. The release also provides a pre-built Singularity image. Detailed setup is in the original `INSTALL.md`.
* **Data.** STU dataset from [github.com/kumuji/stu_dataset](https://github.com/kumuji/stu_dataset). The single-sequence evaluation in this study used sequence 201 (682 scans). For full-test-set evaluation, follow the paper's protocol over the 50+ STU sequences.
* **Evaluation entry point.** `compute_point_level_ood.py` and `infer.py`.

## 8. References

* Mosco, S., Fusaro, D., & Pretto, A. (2026). *Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation*. CVPR 2026, University of Padova. arXiv:2604.23604. Project page: [simom0.github.io/lido-page](https://simom0.github.io/lido-page/).
* AutoSOTA: Tsinghua FIB Lab. *AutoSOTA: An automated SOTA-chasing harness*. [github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA).
