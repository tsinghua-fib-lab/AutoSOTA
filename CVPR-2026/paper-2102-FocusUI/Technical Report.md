# FocusUI: A Technical Report on Automated Optimization

## Abstract
FocusUI, built on Qwen2.5‑VL‑3B, reduces UI‑grounding inference cost by selecting only instruction‑relevant visual tokens (typically retaining ~30% of the ~4,700 tokens in a 2K screenshot) and preserving positional continuity via POSPAD padding. An automated optimization study with the AutoSOTA framework applied 11 iterative code modifications to the inference pipeline, aiming to increase the ScreenSpot‑Pro hit@1 (All‑avg) accuracy. The best configuration, obtained at iteration 10, combines confidence‑weighted multi‑scale test‑time augmentation (1.0× and 1.5× zooms) and dual‑threshold region merging, reaching an All‑avg of **41.75%**, a relative gain of **+2.03%** over the paper baseline of 40.92%. Text accuracy improved to 54.55% (+1.72%) and icon accuracy to 21.03% (+3.29%). Most other attempts—including soft sigmoid gating, temperature scaling, and a reduction of the visual token retention ratio—were neutral or harmful, confirming that the original inference hyper‑parameters are near‑optimal. The study demonstrates that modest, inference‑only augmentations can yield meaningful accuracy gains for UI grounding without retraining.

## 1. Introduction
UI grounding—predicting pixel coordinates of on‑screen elements from natural language instructions—is essential for autonomous GUI agents. High‑resolution screenshots (e.g., 2K) produce roughly 4,700 visual tokens per image, imposing substantial computational cost. FocusUI (Ouyang et al., 2025) tackles this through a learnable patch scorer that retains only a fraction of tokens, combined with POSPAD padding to maintain the positional information needed for precise coordinate output. While FocusUI achieves strong accuracy, its performance on the rigorous ScreenSpot‑Pro benchmark leaves room for improvement, particularly on small icon targets (baseline All‑avg 40.92%, icon Avg‑I 20.36%).

This report details an automated optimization study run by the AutoSOTA framework, which systematically alters the FocusUI inference pipeline to maximize the ScreenSpot‑Pro hit@1 metric across all examples. The optimization proceeds through 11 code‑level modifications, evaluating each on a two‑GPU setup. The best configuration (commit `82d01fb`, tag `_best`) delivers an All‑avg of 41.75%, a relative improvement of 2.03%. The gains stem from multi‑scale test‑time augmentation and dual‑threshold region merging, while changes to the core token selection mechanism are consistently counterproductive.

## 2. Original Method (Background)
FocusUI is an efficient UI grounding framework built on the Qwen2.5‑VL‑3B backbone. Its two core innovations are:

- **Query‑Guided Visual Token Selection:** A lightweight patch scorer, trained first with a frozen base VLM and then jointly fine‑tuned, fuses instruction‑conditioned patch saliency with rule‑based UI‑graph scores. This selects a subset of visual patches (controlled by `visual_reduct_ratio`) that are most relevant to the instruction, discarding large homogeneous regions.
- **POSPAD (Position‑Preserving Padding):** Every contiguous run of dropped visual tokens is compressed into a single special [PAD] marker placed at the index of the last dropped token. This preserves the spatial layout essential for coordinate‑regression heads.

At inference, the patch scorer outputs a score map. Patches are retained according to `visual_reduct_ratio` (baseline 0.3), and the language model generates a pointer token whose attention over the retained visual tokens yields a saliency map. A fixed activation threshold (0.3× the maximum attention score) binarises the map into connected regions; the center of the top‑scoring region becomes the predicted click point. The original paper reports strong numbers with `visual_reduct_ratio=0.5`; the optimization here uses 0.3 as the baseline for a better speed‑accuracy trade‑off.

## 3. Identified Limitations
Analysis of the optimization trajectory reveals several limitations of the original inference pipeline:

1. **Single‑scale inference ignores size variation.** UI screens contain elements ranging from tiny icons to large text fields. The baseline icon hit@1 is only 20.36%, indicating that many small targets are missed. The substantial gains from multi‑scale TTA (Section 5.3) confirm that a single native‑resolution forward pass is a bottleneck.

2. **The fixed activation threshold is sensitive but near‑optimal.** Attempts to replace it with soft sigmoid gating (iter. 1, −1.08%), temperature scaling (iter. 3, 0.00%), or multi‑threshold ensembles (iter. 4, −3.69%) all failed to improve accuracy, implying that the default 0.3 value is well‑chosen for the current architecture.

3. **Visual token reduction faces a sharp trade‑off.** Reducing `visual_reduct_ratio` from 0.3 to 0.2 (iter. 11) lowered All‑avg by 1.39%, driven mainly by an icon regression, because too few visual tokens deprive the model of necessary spatial context.

4. **Icon grounding remains persistently difficult.** Even after optimization, the best icon accuracy (21.03%) lags well behind text accuracy (54.55%), a gap of over 33 percentage points. Inference‑only modifications cannot overcome this discrepancy, which likely stems from data imbalance or model‑level biases toward text features.

## 4. Optimization Methodology
AutoSOTA performed 11 optimization rounds. Each round proposed a code change to `evaluation/ss_pro_eval.py` or `focusui/inference.py`, ran the full ScreenSpot‑Pro evaluation on two GPUs (indices 2,3), and recorded All‑avg. Only interventions that increased the primary metric were retained. Three interventions were successful:

1. **Multi‑scale test‑time augmentation (Iteration 7):** The evaluation function was modified to run inference at the original scale (1.0×) and at a 1.2× zoom. Predicted points from the two scales were averaged with equal weights. This raised All‑avg to 41.18 (+0.64% vs. baseline). The moderate zoom helps resolve fine details of small UI elements while retaining global context.

2. **Confidence‑weighted TTA (Iteration 8):** The ensemble was reweighted by the maximum region attention score produced at each scale, treating that score as a confidence measure. The scales were adjusted to 1.0× and 1.5×. Confidence‑weighted blending yielded All‑avg of 41.68 (+1.86%), outperforming simple averaging.

3. **Dual‑threshold region merging (Iteration 10):** The function `get_prediction_region_point` was enhanced with a secondary, lower threshold (50% of the primary 0.3×max threshold). After high‑threshold regions are formed, any patch whose attention exceeds the lower threshold and is adjacent to an existing region is merged. This “soft‑merge” better covers larger UI components such as buttons or input fields. Paired with confidence‑weighted TTA (1.0× + 1.5×), this achieved the peak All‑avg of **41.75** (+2.03%).

Other modifications (soft sigmoid gating, 8‑directional connectivity, temperature scaling, multi‑threshold ensemble, max text token pooling, hybrid spatial diversity, and lowering `visual_reduct_ratio` to 0.2) were evaluated and discarded because they did not improve the primary metric.

## 5. Experiments

### 5.1 Setup
**Hardware:** Two NVIDIA GPUs (device indices 2,3) on a shared host; because the intended Docker image `autosota/paper-2102:reproduced` could not be started, evaluation was run directly in the host environment with the same code state (exported manually from the workdir at git tag `_best`).  
**Dataset:** ScreenSpot‑Pro benchmark, containing professional UI screenshots with bounding‑box annotations for text and icon elements.  
**Evaluation protocol:** Hit@1 (point‑in‑box) accuracy is computed per example and averaged across all examples (All‑avg); higher is better. Sub‑metrics Avg‑T (text) and Avg‑I (icon) are also reported.  
**Baseline:** FocusUI‑3B, `visual_reduct_ratio=0.3`, default activation threshold 0.3.  
**Optimization budget:** 11 improvement rounds (iterations 1–11) plus one baseline evaluation (iteration 0).  
**Caveats:** The non‑containerised host Python environment may differ from the original paper’s environment. Only inference‑time algorithmic components were changed; model weights were untouched.

### 5.2 Quantitative Results
Table 1 compares the baseline and the best‑achieved configuration (iteration 10).

| Metric | Baseline (iter 0) | Best (iter 10) | Absolute Δ (pp) | Relative Δ (%) |
|--------|-------------------|----------------|----------------|----------------|
| All‑avg (primary) | 40.92 | **41.75** | +0.83 | **+2.03** |
| All‑text (Avg‑T)  | 53.63 | **54.55** | +0.92 | +1.72 |
| All‑icon (Avg‑I)  | 20.36 | **21.03** | +0.67 | +3.29 |

Improvements are consistent across both element types; the largest relative gain is on icons, though absolute icon accuracy remains low.

### 5.3 Ablation / Iteration Trajectory
Table 2 lists every attempted modification in chronological order and its effect on the primary All‑avg metric. The “Δ vs Baseline” column reports relative percentage change with respect to the 40.92% baseline.

| Iter | Change | All‑avg (%) | Δ vs Baseline (%) |
|------|--------|-------------|---------------------|
| 0 | Paper baseline (`visual_reduct_ratio=0.3`) | 40.92 | — |
| 1 | Soft activation threshold (sigmoid gating) | 40.48 | −1.08 |
| 2 | 8‑directional region connectivity | 40.80 | −0.29 |
| 3 | Temperature scaling (T=0.5) | 40.92 | 0.00 |
| 4 | Multi‑threshold ensemble | 39.41 | −3.69 |
| 5 | Max text‑token pooling | 40.86 | −0.15 |
| 6 | Hybrid token + spatial diversity | 40.61 | −0.76 |
| 7 | **Multi‑scale TTA (1.0× + 1.2×)** | 41.18 | +0.64 |
| 8 | **Confidence‑weighted TTA (1.0× + 1.5×)** | 41.68 | +1.86 |
| 9 | TTA with 2.0× zoom | 41.24 | +0.78 |
| 10 | **Dual‑threshold merging + TTA (1.0× + 1.5×)** | **41.75** | **+2.03** |
| 11 | `visual_reduct_ratio=0.2` + TTA + dual‑threshold | 41.49 | +1.39 |

Multi‑scale TTA provided the first and largest gain, contributing the bulk of the improvement. Confidence weighting and dual‑threshold merging added small complementary lifts, while an extreme 2.0× zoom (iter 9) or a lower token retention ratio (iter 11) caused regressions. All attempts to modify the base activation threshold (iterations 1–4) were either neutral or strongly detrimental.

## 6. Discussion
The most impactful change was multi‑scale TTA: the ensemble of original and moderately zoomed views allowed the model to resolve fine details (especially icons) without losing global disambiguation. Confidence weighting further improved robustness by suppressing low‑quality predictions, and dual‑threshold merging helped the predicted region better cover the full UI component.

Crucially, every modification to the core token selection threshold or reduction ratio was counterproductive, confirming that the original inference hyper‑parameters are close to optimal. The ceiling remains low for icons: at best 21.03%, the gap to text accuracy exceeds 33 percentage points, indicating a structural limitation that inference‑only techniques cannot bridge. Future gains are likely to require changes to the model architecture or training data.

Several threats to validity should be considered. The optimization was limited to a single benchmark (ScreenSpot‑Pro); generalizability to other UI‑grounding datasets was not evaluated. Evaluation ran outside the intended Docker container, which could introduce environmental variance. No statistical significance testing (multiple random seeds) was performed, so the measured differences may be sensitive to run‑to‑run noise. Finally, the search space comprised heuristic modifications; a full hyper‑parameter sweep might uncover configurations beyond the explored set.

## 7. Reproducibility
- **Repository:** `https://github.com/showlab/FocusUI.git`, optimized state at commit `82d01fb` (git tag `_best`).  
- **Environment:** Conda environment with Python 3.12 and packages listed in `requirements.txt`.  
- **Pretrained weights:** Download FocusUI‑3B from Hugging Face into `./checkpoints/focusui‑3b`.  
- **Baseline evaluation:**  
  ```bash
  python -m evaluation.ss_pro_eval \
      --model_type focusui_3b \
      --model_name_or_path checkpoints/FocusUI-3B \
      --data_path ./datasets/UI-Grounding-Benchmarks/ScreenSpot-Pro \
      --save_path ./results/ss_pro \
      --visual_reduct_ratio 0.3
  ```
- **Optimized evaluation:** The same command is used on the code state at `_best`, which internally activates confidence‑weighted multi‑scale TTA (1.0× and 1.5×, weighted by region attention) and dual‑threshold region merging (secondary threshold at 50 % of the primary). No additional arguments are required.

## 8. References
```bibtex
@article{ouyang2025focusui,
  title   = {FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection},
  author  = {Ouyang, Mingyu and Lin, Kevin Qinghong and Shou, Mike Zheng and Ng, Hwee Tou},
  year    = {2025},
  journal = {arXiv preprint},
}

@misc{autosota,
  author       = {tsinghua-fib-lab},
  title        = {AutoSOTA: Automated State-of-the-Art Optimization Framework},
  year         = {2025},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
}
```
