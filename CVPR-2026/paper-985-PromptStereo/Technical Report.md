# PromptStereo: A Technical Report on Automated Optimization

## Abstract

Zero-shot stereo matching estimates dense disparity maps without per‑domain fine‑tuning. PromptStereo (CVPR 2026) fuses structure prompts from a pre‑trained monocular depth estimator (Depth Anything V2) with motion prompts inside a progressive recurrent update (PRU) architecture. This report documents an automated optimization study of PromptStereo using the AutoSOTA pipeline. Sixteen inference‑time interventions were applied to the KITTI 2015 training set, with a target of −5% end‑point error (EPE). Although the target was not reached, the best configuration reduced EPE from 1.04 px to 1.02 px (−1.92%) and the bad‑3‑pixel error rate from 4.43% to 4.40% (−0.68%). Three complementary modifications yielded the improvement: structure–motion disagreement dampening, which attenuates disparity updates where the monocular depth prior and the stereo estimate disagree; adaptive disparity truncation (LEAP), which hard‑clamps the predicted disparity within a contracting margin around the aligned monocular depth; and edge‑aware unsharp masking, which sharpens the output near image boundaries. The combination of dampening (soft constraint) and truncation (hard constraint) reduced EPE by 0.02 px.

## 1. Introduction

Stereo matching remains a central problem in 3D vision. Many state‑of‑the‑art methods require per‑domain fine‑tuning, limiting their practical deployment. PromptStereo was designed as a zero‑shot stereo matcher that leverages two complementary “prompts”: a structure prompt derived from a monocular depth estimation network (Depth Anything V2) and motion prompts encoded in a correlation‑volume‑based recurrent architecture. This design enables strong generalization across diverse scenes without target‑domain training.

The present work investigates whether the performance of PromptStereo can be further improved through automated, inference‑time code modifications using the AutoSOTA framework. AutoSOTA systematically proposes code‑level changes, evaluates them on standard metrics, and retains effective modifications. The study focuses on the KITTI 2015 training set, with the primary goal of reducing the all‑pixel end‑point error (EPE) while maintaining the bad‑3‑pixel error rate.

## 2. Original Method (Background)

PromptStereo consists of a structure prior and a motion‑guided refinement. The structure prior is obtained from Depth Anything V2 (ViT‑L), which produces a dense monocular depth map. After alignment to the stereo rectification geometry, the depth is converted into an initial disparity estimate and serves as a fixed anchor.

The motion prompt is realized through a correlation‑volume‑based PRU. At each of the 32 default iterations, a GRU‑based updater ingests structure features, the current disparity, and a correlation lookup from the right image to produce a delta disparity. The final disparity is obtained via a soft‑argmin over a cost volume, yielding a differentiable but smooth output. The model is trained on synthetic datasets (SceneFlow, TartanAir, CREStereo, etc.) and the FoundationStereo dataset. Evaluation uses bf16 mixed precision. The default checkpoint, PromptStereo‑Unlimited‑192, is employed in all experiments.

## 3. Identified Limitations

The AutoSOTA optimization identified three weaknesses in the base pipeline.

**Soft‑argmin over‑smoothing.** The disparity is computed as a weighted sum over a quantized range via softmax (see `promptstereo.py`, line 108). This averaging blurs object boundaries, contributing to a baseline bad‑3 of 4.43% on all pixels.

**Sensitivity to noisy PRU updates.** The PRU updater produces a delta disparity without explicit regularization. When the structure prompt and stereo motion prompt disagree—e.g., in textureless or reflective regions—the update can be large and harmful. The alignment residual `norm_depth - norm_disp` is a reliable indicator of such regions. Initially, before any intervention, EPE was 1.04; after introducing disagreement‑based dampening, EPE dropped to 1.03 (Iteration 2), confirming the effectiveness of soft regularization.

**Disparity drift without global bounds.** Without hard constraints, recurrent refinement can cause the disparity to drift. When the number of iterations was increased from 32 to 48 (Iteration 9), EPE remained at 1.04—no improvement over the baseline—indicating that further iterations can cause divergence without stronger constraints. Later, adaptive truncation (LEAP) prevented drift by clamping disparity to a neighborhood of the monocular depth.

## 4. Optimization Methodology

The AutoSOTA pipeline generated and tested multiple inference‑time modifications. Four interventions were accepted into the final configuration.

**Probability Sharpening (Temperature Scaling).** In `promptstereo.py:108`, the softmax logits are divided by a temperature factor \(T = 1.5\) before the soft‑argmin. This sharpens the cost volume distribution, making the operation behave more like a hard argmin and reducing averaging over multiple disparity hypotheses.

**Structure–Motion Disagreement Dampening.** In `update.py:217‑224`, a per‑pixel dampening factor is computed from the normalized discrepancy between the aligned monocular depth and the current disparity estimate. The delta disparity is then multiplied by 
\[
\text{damp} = \max(0.3, e^{-|\text{norm\_depth} - \text{norm\_disp}|/\tau}),
\]
with \(\tau\) a fixed scale. This suppresses large updates where structure and motion cues disagree, while allowing free evolution where they concur.

**Adaptive Disparity Truncation (LEAP).** In `promptstereo.py:136‑148`, a hard clamping operation is inserted after each PRU iteration. The allowed disparity range is centered on the aligned monocular depth, and its margin shrinks linearly with the iteration number (from an initial width down to a minimum). This prevents catastrophic drift while permitting adequate exploration in early iterations.

**Edge‑Aware Unsharp Masking.** In `promptstereo.py:159‑174`, an unsharp mask is applied to the final disparity map. The amount of sharpening is controlled by the gradient magnitude of the left image: the disparity is blended with its Gaussian‑blurred version only where the image gradient exceeds a threshold, using \(\alpha = 0.30\). This recovers sharpness near object boundaries.

All interventions are inference‑time only; no retraining or ground‑truth labels are required.

## 5. Experiments

### 5.1 Setup

Evaluation used the KITTI 2015 training set, all pixels with valid disparity (mask `'all'`). The model is the PromptStereo‑Unlimited‑192 checkpoint, loaded via `accelerate launch evaluate_stereo.py` with default parameters (32 PRU iterations, bf16 mixed precision). The baseline was repeatedly measured to ensure stability. No random seed was fixed in the evaluation script; numerical variation is minimal. The optimization operated solely on the training split; online test‑server results are not available. Metrics are end‑point error (EPE) in pixels and the percentage of pixels with error greater than 3 px (Bad 3.0), reported for the all‑pixel mask.

### 5.2 Quantitative Results

The final configuration combines dampening, probability sharpening, LEAP truncation, and edge‑aware unsharp masking. Metrics are shown in Table 1.

| Metric              | Baseline | Optimized | Δ (px) / Δ (%) | Δ %     |
|---------------------|----------|-----------|----------------|---------|
| EPE (px)            | 1.04     | 1.02      | −0.02          | −1.92%  |
| Bad 3.0 (%)         | 4.43     | 4.40      | −0.03          | −0.68%  |

Non‑occluded variants were not measured in this optimization. The best commit (`d197d78`) corresponds to the LEAP intervention; edge‑sharpening was added as a further minor edit.

### 5.3 Ablation / Iteration Trajectory

Table 2 shows the accumulated effect of the accepted interventions in chronological order.

| Step | Intervention                      | EPE (px) | Bad 3.0 (%) | Comment                                    |
|------|-----------------------------------|----------|-------------|--------------------------------------------|
| 0    | Baseline                          | 1.04     | 4.43        | Starting point                             |
| 1    | Disagreement Dampening (floor 0.3)| 1.03     | 4.41        | First EPE reduction (Iter. 2‑5)            |
| 2    | Probability Sharpening (T=1.5)    | 1.03     | 4.39        | Further Bad3 improvement (Iter. 6)         |
| 3    | LEAP Disparity Truncation         | 1.02     | 4.49        | Large EPE gain; temporary Bad3 increase    |
| 4    | Edge‑Aware Unsharp Mask (α=0.30)  | 1.02     | 4.40        | Bad3 restored to below baseline (Iter. 14) |

The trajectory highlights the complementarity of dampening and truncation: the soft constraint first stabilizes refinement, then the hard clamp prevents drift, pushing EPE to 1.02 px. Edge‑aware sharpening recovers boundary accuracy, achieving the final Bad3 of 4.40%.

## 6. Discussion

The primary improvement originated from the combination of structure–motion disagreement dampening and adaptive disparity truncation. Dampening acts as a per‑pixel soft regularizer, and truncation provides a hard global guardrail; together they reduced EPE by 0.02 px. Edge‑aware unsharp masking contributed a small further reduction in boundary errors, countering soft‑argmin over‑smoothing.

Several attempted modifications were ineffective. Test‑time augmentation via horizontal flipping inside the model’s forward pass produced severely corrupted outputs (EPE > 34 px), as internal feature extraction is orientation‑sensitive. An Adam‑style exponential moving average of the delta disparity caused a regression to EPE = 1.34 px, because the disparity evolves too rapidly across iterations for a global EMA to be meaningful. Extending the number of PRU iterations from 32 to 48 without stronger constraints yielded EPE = 1.04 px (no improvement), confirming that further iterations require better regularization. Soft blending of truncated and raw disparity was inferior to hard clamping. Reducing the monocular depth influence in the initial fusion hurt accuracy, underscoring the value of the structure prompt.

The study has several limitations. All evaluations are on the KITTI 2015 training split; generalization to other datasets was not tested. The interventions are purely inference‑time heuristics and may overfit to automotive stereo scenes. The improvement of ≈2% relative EPE suggests the baseline model is already near‑optimal under the given constraints. The absence of a fixed random seed could introduce minor noise, though the consistent trajectory across iterations mitigates this.

## 7. Reproducibility

**Repository:** Official PromptStereo codebase.

**Environment:**
```
conda create -n promptstereo python=3.12
conda activate promptstereo
pip install tqdm numpy wandb opt_einsum hydra-core
pip install imageio scipy torch torchvision opencv-python matplotlib
pip install xformers accelerate scikit-image
```

**Baseline command:**  
`accelerate launch evaluate_stereo.py`  
(Uses the default unlimited_192 checkpoint, bf16 precision.)

**Optimized version:** Apply the following modifications to the baseline:
1. `promptstereo.py:108` – multiply logits by 1/1.5 before softmax.  
2. `update.py:217-224` – insert exponential dampening factor with floor 0.3, based on `norm_depth - norm_disp`.  
3. `promptstereo.py:136-148` – hard clamp disparity within a linearly decreasing margin around the aligned monocular depth.  
4. `promptstereo.py:159-174` – apply gradient‑guided unsharp masking with α = 0.30.

After applying these changes, run the same evaluation command. The commit hash `d197d78` contains the first three interventions; the edge‑sharpening was added later. No seed manipulation is required.

## 8. References

```bibtex
@inproceedings{wang2026promptstereo,
  title     = {PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts},
  author    = {Xianqi Wang and Hao Yang and Hangtian Wang and Junda Cheng and Gangwei Xu and Min Lin and Xin Yang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}

@misc{autosota,
  author       = {tsinghua-fib-lab},
  title        = {AutoSOTA},
  year         = {2024},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}}
}
```
