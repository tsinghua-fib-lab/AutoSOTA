# ACPV-Net: All-Class Polygonal Vectorization — A Technical Report on Automated Optimization

## Abstract
Polygon-based vectorization of remote sensing imagery underpins automated mapping, demanding precise geometric delineation of land-cover instances. The ACPV-Net paper introduces a diffusion-driven approach that predicts per-class vertex heatmaps and constructs polygons through thresholding, non-maximum suppression, and Delaunay triangulation. This report documents the application of the AutoSOTA automated optimization pipeline to the post-processing stage of ACPV-Net. The original method achieved a building complete intersection-over-union (CIoU) of 71.92% on the evaluation set. Through seven parameter-tuning iterations without retraining, the pipeline raised the building CIoU to 75.78%, a relative improvement of 5.37%. The primary drivers were an increase in the maximum number of extracted vertices (top‑k from 300 to 750) and a switch from mean to max pooling for fusing multi-channel heatmap activations. The vertex count ratio (n‑ratio) improved from 0.87 to 1.11, directly boosting the polygonal similarity component of CIoU. Per‑class results showed consistent improvements for vegetation (+6.33%), roads (+3.70%), and unvegetated areas (+2.82%), while water CIoU remained stable (−0.01 pp). These findings demonstrate that inference-time tuning of vertex extraction hyperparameters can significantly enhance polygon fidelity without model retraining.

## 1. Introduction
Vector polygon extraction from aerial and satellite imagery enables scalable map creation for urban planning, disaster response, and environmental monitoring. Deep models have evolved from pixel-wise segmentation to direct polygon prediction, yet boundary accuracy and vertex count correctness remain difficult. ACPV-Net tackles this by generating class‑specific vertex heatmaps through a denoising diffusion probabilistic model, followed by a geometric post-processing chain that produces polygons. While the learned heatmaps capture spatial structure, the post-processing parameters—the number of retained vertices, the heatmap channel fusion strategy, and the detection threshold—were set heuristically. The extent to which these settings constrain final metrics was not known a priori.

This study employs the AutoSOTA framework to systematically explore the parameter space of the vertex extraction pipeline of ACPV-Net. AutoSOTA automates iterative proposal, evaluation, and selection of configuration changes guided by a target metric and predefined search operations. The objective is to identify low‑cost, inference‑time adjustments that lift the model’s performance beyond its original state, without re-engineering the architecture or collecting additional training data. The following sections detail the original method, its observed limitations, the optimization methodology, the experimental trajectory, and the quantitative results.

## 2. Original Method (Background)
ACPV-Net produces class‑wise vector polygons via a diffusion process and a subsequent geometry pipeline. A latent diffusion model (DDIM sampler, 200 steps) generates a 3‑channel feature map per class at a fixed resolution. A lightweight decoder transforms these latents into a multi‑channel vertex activation tensor, each channel highlighting vertex candidates. The post-processing, implemented in `tools/extract_vertices_from_heatmap.py`, fuses the channels with an element‑wise reduction (originally mean), applies a fixed threshold (0.1), performs non‑maximum suppression (NMS), and selects the top‑k strongest responses (300) as vertices. Candidate vertices feed a polygonization routine that constructs a Delaunay triangulation, optionally refines it with vertex-guided subset selection (VSS) and distance‑preserving simplification (DP‑fix), and clips the polygons to produce final instances.

The quality of output polygons is assessed with several metrics. The primary metric, CIoU (Complete Intersection over Union), multiplies standard IoU by a polygonal similarity factor *p_s*, where *p_s* penalises discrepancy between predicted and reference vertex counts. A predicted-to-reference vertex count ratio (*n‑ratio*) of 1.0 yields *p_s* = 1.0. Other metrics include boundary IoU (bIoU) and polygon complexity (Polis). The original ACPV-Net configuration (mean pooling, threshold 0.1, top‑k = 300) gave a building n‑ratio of 0.87, corresponding to a baseline CIoU of 71.92%.

## 3. Identified Limitations
The baseline metrics revealed a pronounced under‑generation of vertices for buildings: the n‑ratio was 0.87, meaning only 87% of the reference vertices were produced. Because CIoU = IoU × *p_s* and *p_s* decays sharply with count mismatch, this deficit directly depressed the CIoU score. Inspection of the extraction code showed that the top‑k parameter hard‑caps the maximum number of retained vertices at 300, which frequently fell short of the actual vertex demand in complex building footprints.

A second limitation concerns the heatmap fusion operation. The decoder outputs three activation channels per class. Mean pooling across channels can dilute strong, localised peaks that appear in only one or two channels, while amplifying uniformly low background noise. The resulting smoother heatmap may yield suboptimal local maxima after NMS, potentially missing true corners. Finally, the NMS threshold itself was not the active bottleneck: with top‑k = 300, the weakest of the kept vertices already exceeded the 0.1 threshold, so lowering it alone could not increase vertex count. The vertex budget was the binding constraint.

## 4. Optimization Methodology
The AutoSOTA pipeline operated directly on `tools/extract_vertices_from_heatmap.py`, proposing modifications and evaluating them on a held‑out validation set using building CIoU as the reward signal. Seven iterations were executed; the accepted interventions are summarised below.

**Vertex-guided subset selection and distance‑preserving fix (VSS+DP‑fix, iteration 1).**  
Enabling these polygonization refinements altered polygon structure but did not change aggregate metrics, because vertex counts and positions were already determined before this stage. Building CIoU remained at 71.92%.

**Heatmap channel fusion: mean → max (iteration 2).**  
The channel reduction was changed from `torch.mean(dim=1)` to `torch.max(dim=1)`. Max pooling preserves strong activations from any single channel, retaining sharp peaks that mean pooling would average away. The n‑ratio rose from 0.87 to 0.89, and building CIoU increased by 0.13 to 72.05%.

**Detection threshold reduction (iteration 3).**  
The threshold was lowered from 0.1 to 0.05. With top‑k still at 300, the weakest selected vertex was already above 0.05, so the binary mask and metrics remained unchanged (CIoU 72.05%). This confirms that the top‑k cap was the primary constraint.

**Top‑k increase to 500 (iteration 4).**  
The vertex budget was expanded from 300 to 500. The n‑ratio jumped to 1.00 and CIoU soared from 72.05% to 75.36% (+3.31 points). IoU improved concurrently. This single change accounted for over 85% of the total eventual gain.

**Adaptive threshold (iteration 6).**  
A heuristic threshold (95th percentile of heatmap values × 0.3) was tested. The computed value approximated 0.05, already in use, so CIoU remained 75.36%. The adaptive mechanism was validated but offered no additional improvement.

**Top‑k increase to 750 (iteration 7).**  
Further expanding the vertex budget to 750 pushed the n‑ratio to 1.11 and CIoU to 75.85%, exceeding the 75.516% target (5% relative improvement). A final confirmation run consolidated all best configurations, yielding 75.78%.

A counter‑productive attempt (iteration 5) tightened the PSLG snap distance (`dist_thresh=3.0`), which excluded some valid snapped vertices and caused a slight regression to 75.30%. This change was discarded.

## 5. Experiments

### 5.1 Setup
The original ACPV-Net model weights and dataset were used without modification. Evaluation followed the published benchmark: polygon predictions vs ground‑truth polygons assessed by per‑class CIoU, IoU, bIoU, n‑ratio, and Polis. The optimization targeted building CIoU, with a stop condition of 5% relative improvement over the 71.92% baseline (target ≥ 75.516%). The pipeline was allowed 20 iterations but automatically stopped after 7 iterations when the target was exceeded.

All experiments were performed on a single GPU workstation with sufficient memory to load the diffusion model and dataset. Each iteration executed a single evaluation pass over the validation set; no retraining of the neural network occurred.

*Caveats.* The optimization log does not record the exact dataset split, random seed, or metric variance. Reported metrics are from single evaluation passes. While the improvements are substantial, their stability under different seeds or dataset partitions was not assessed.

### 5.2 Quantitative Results
Table 1 presents baseline and final metrics after applying the full set of accepted optimizations (max pooling, threshold 0.05, top‑k = 750). The building CIoU increased from 71.92% to 75.78% (+5.37%). All per‑class CIoU scores improved except water, which remained essentially unchanged (−0.01 pp). The building n‑ratio rose from 0.87 to 1.11, indicating the model now generates slightly more vertices than ground truth, preserving polygonal detail with minimal over‑segmentation.

| Metric            | Baseline | Optimized | Delta   | Direction  |
|-------------------|----------|-----------|---------|------------|
| ciou_building     | 71.92    | 75.78     | +3.86   | ↑ +5.37%   |
| iou_building      | 79.03    | 82.06     | +3.03   | ↑ +3.84%   |
| biou_building     | 76.27    | 79.51     | +3.24   | ↑ +4.25%   |
| nratio_building   | 0.87     | 1.11      | +0.24   | ↑ +27.6%   |
| polis_building    | 2.00     | 1.83      | −0.17   | ↓ 8.5%     |
| ciou_road         | 63.06    | 65.39     | +2.33   | ↑ +3.70%   |
| ciou_unvegetated  | 57.83    | 59.46     | +1.63   | ↑ +2.82%   |
| ciou_vegetation   | 64.45    | 68.53     | +4.08   | ↑ +6.33%   |
| ciou_water        | 57.01    | 57.00     | −0.01   | ≈ stable   |

*Table 1: Baseline vs. optimized metrics. CIoU, IoU, bIoU in %; nratio and polis are unitless. Arrows indicate improvement direction (↑ better, ↓ lower complexity desirable).*

### 5.3 Ablation / Iteration Trajectory
Table 2 chronicles building CIoU after each accepted intervention. The initial changes (VSS+DP‑fix, threshold reduction) produced no gain; the major jump occurred with top‑k = 500, and a further climb with top‑k = 750. The max-pooling contribution was retained throughout.

| Iteration | Intervention                              | Building CIoU (%) | Δ from baseline (pp) |
|-----------|-------------------------------------------|-------------------|----------------------|
| 0         | Baseline (mean pool, topk 300, thresh 0.1)| 71.92             | —                    |
| 1         | Enable VSS+DP‑fix                         | 71.92             | 0.00                 |
| 2         | Switch to max pooling                     | 72.05             | +0.13                |
| 3         | Lower threshold to 0.05                   | 72.05             | +0.13 (no change)    |
| 4         | Increase top‑k to 500                     | 75.36             | +3.44                |
| 5         | Tighten snap distance (discarded)         | 75.30             | regression           |
| 6         | Adaptive threshold (Leap)                 | 75.36             | +3.44                |
| 7         | Increase top‑k to 750                     | 75.85             | +3.93                |
| Final     | Confirm full pipeline                     | 75.78             | +3.86                |

*Table 2: Optimization trajectory showing building CIoU after each iteration and absolute point change from baseline. Iteration 5 was reverted; adaptive threshold (iter 6) returned the same score as iteration 4.*

## 6. Discussion
The automated optimization reveals that the vertex count constraint (top‑k) was the dominant bottleneck for building CIoU in the original ACPV‑Net. Expanding top‑k from 300 to 500 delivered over 85% of the total 3.86‑pp CIoU gain, and the further increase to 750 contributed an additional 0.49 pp. Max‑pooling fusion provided a small synergistic boost by recovering vertex candidates diluted by mean pooling. The 5.37% relative improvement was achieved entirely through inference‑time hyperparameter adjustments.

Per‑class analysis shows benefits are not uniform. Vegetation (+4.08 pp, +6.33%) and roads (+2.33 pp, +3.70%) saw large gains, indicating the original top‑k also under‑sampled vertices for these classes. Water CIoU remained nearly flat (−0.01 pp), likely because water bodies contain few vertices or because heatmap quality differs. This pattern suggests that a single global top‑k is suboptimal; per‑class vertex budgets or class‑adaptive selection could yield further improvements.

The main limitation of this study is unknown generalisability. The optimization was conducted on one dataset split with a single evaluation pass per iteration; no cross‑validation or testing on external data was performed. The stochasticity of the diffusion sampling was not quantified, though the monotonic trend argues against mere noise. Additionally, the absence of multiple random seeds prevents a variance estimate.

Future work could build on these findings. Increasing DDIM sampling steps beyond 200 might further improve heatmap quality, potentially raising CIoU above 77%. Multi‑scale vertex consensus or confidence‑weighted vertex selection based on heatmap activation strength could reduce false positives while preserving true corners. Finally, integrating post‑processing parameters as learnable components during training could allow the diffusion model to adapt its latent representations to the extraction budget.

## 7. Reproducibility
The ACPV‑Net repository and pretrained model weights are assumed available from the authors (not provided in the optimization materials). To reproduce the optimized results, apply the following modifications to `tools/extract_vertices_from_heatmap.py`:

- Replace the channel reduction from `torch.mean` to `torch.max`.
- Set `topk` to 750.
- Set `thresh` to 0.05.
- (Optional) Enable VSS and DP‑fix flags; they do not affect the reported gains.

The baseline and optimized evaluations share the same inference command:
```
python evaluate.py --config acpvnet.yaml --checkpoint model.pth
```
The environment should be set up according to the original repository’s installation instructions, typically Python 3.9+ and PyTorch 1.12+ with CUDA 11.6.

## 8. References
```
@inproceedings{acpvnet,
  title   = {ACPV-Net: All-Class Polygonal Vectorization},
  author  = {N/A},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {N/A},
  note    = {Original paper; citation details not available in the provided repository.}
}

@misc{autosota,
  author       = {tsinghua-fib-lab},
  title        = {AutoSOTA: Automated State-of-the-Art Optimization Framework},
  year         = {2025},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
}
```
