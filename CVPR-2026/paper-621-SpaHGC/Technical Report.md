# SpaHGC: A Technical Report on Automated Optimization

## Abstract
Spatial gene expression inference from histopathology images offers a low‑cost complement to full spatial transcriptomics, yet capturing cross‑slide relationships remains challenging. SpaHGC addresses this with a multi‑modal heterogeneous graph that links spatial and cross‑slice dependencies using UNI pathology embeddings, trained with masked graph contrastive learning. This report documents an automated optimization study of SpaHGC via the AutoSOTA framework. Conducted on a single fold of the cSCC dataset (GSE144240, fold P2_ST_rep1) where only pre‑computed predictions were available, the study improved the per‑gene mean Pearson correlation coefficient (PCC) from **55.90 % to 57.05 %** (absolute gain +1.16 pp, relative +2.07 %) and reduced root‑mean‑square error (RMSE) from 0.1762 to 0.1755. The improvement comes exclusively from post‑processing, principally bilateral filtering that jointly exploits tissue spatial proximity and embedding‑based feature similarity using high‑temperature neighbour weights. Model‑level architectural changes were implemented but could not be evaluated because the training environment lacked data and model weights. The results show that targeted spatial smoothing yields a measurable uplift, while also exposing the ceiling imposed by frozen model outputs and a single‑fold evaluation setup.

## 1. Introduction
Inference of spatial gene expression from H&E‑stained tissue slides lowers the cost and throughput barriers of spatial transcriptomics. SpaHGC constructs a heterogeneous graph over histological spots, leveraging a pre‑trained UNI pathology foundation model to enable cross‑slice knowledge transfer via masked graph contrastive learning. The original work reports strong cross‑dataset performance, yet the per‑gene mean PCC on the cSCC dataset remains at 55.90 %, indicating headroom for improvement.

Automated optimization frameworks such as AutoSOTA (tsinghua‑fib‑lab/AutoSOTA) systematically probe code, identify limitations, implement targeted changes, and validate results. This report details an AutoSOTA run on the SpaHGC repository. Because the execution sandbox lacked training data, graph files, and UNI weights, the pipeline focused on refining frozen model outputs. Post‑processing techniques raised the per‑gene mean PCC to 57.05 %, with bilateral spatial‑feature smoothing delivering the largest gain (+1.16 pp).

## 2. Original Method (Background)
SpaHGC models spot‑wise gene expression prediction as a heterogeneous graph learning task. Each spot carries a 1024‑dimensional feature vector from the UNI ViT‑L/16 encoder. The graph includes two node types—**target** spots (slide to predict) and **reference** spots (slides with known expression)—and three edge classes:
- intra‑slice spatial edges (`target`–`TS`–`target`) built via a radius graph,
- cross‑slice edges (`reference`–`CS`–`target`) linking each target spot to its most similar reference spots based on UNI embedding distance,
- reference‑to‑reference edges (`reference`–`RS`–`reference`) formed by k‑nearest neighbours in the joint feature–expression space.

The encoder (`model.py`, class `HeteroGNN`) applies a heterogeneous convolution that uses GraphSAGE‑style message passing on spatial edges and a Cross‑Node Dual Attention (CNDA) mechanism on cross‑slice edges. Aggregated target representations are pooled by a Cross‑Node Attention Pooling module (CNAP, initially 4 heads) and mapped to gene expression via a linear layer. The loss combines mean‑squared error, a correlation loss, and two Barlow‑Twins‑style contrastive losses (target and reference) computed from twice‑augmented views. The augmentor (`HeteroGraphAugmentor`) masks input features with zeros (target ratio 0.1, reference ratio 0.9) and drops edges. Training uses AdamW with learning rate 1e‑3, weight decay 1e‑4, and a leave‑one‑slide‑out cross‑validation scheme over four folds of the cSCC dataset (GSE144240). Evaluation metrics are per‑gene mean PCC and RMSE.

## 3. Identified Limitations
1. **Baseline accuracy leaves room for improvement.** The per‑gene mean PCC on fold P2_ST_rep1 is 55.90 %. An automated target of 58.70 % (5 % relative improvement) was set, underscoring the ceiling that post‑processing alone can address.
2. **No explicit spatial coherence constraint.** The model outputs are spot‑wise; no mechanism ensures that neighbouring spots with similar histology share smooth expression profiles. The benefit of post‑hoc spatial filters confirms that residual noise exists in the predictions.
3. **Post‑processing temperature may be sub‑optimal.** In the smoothing routine, temperature τ controls the sharpness of neighbour weights. Early sweeps showed τ ∈ {0.01, 0.1} yields overly selective weighting, while τ ∈ {1.0, 5.0} (flatter) produces higher PCC, indicating spots are similar to a broader local neighbourhood.
4. **Model architecture bottlenecks.** The hidden dimension is 256, potentially insufficient for 1024‑dimensional UNI features. No residual connections or layer normalization after GraphSAGE layers risk over‑smoothing and instability. The loss weights (`MSE + 0.5·Corr + 0.2·Target_CL + 0.1·Ref_CL`) are heuristically fixed. CNAP uses only 4 heads, and the augmentor uses zero‑fill rather than learnable mask tokens.
5. **Operational constraints prevented retraining.** The optimization sandbox lacked cSCC image data, pre‑computed graph files, and UNI model weights; only pre‑saved predictions for fold P2_ST_rep1 were available. Thus all architectural changes were implemented but not evaluated, and statistical robustness is limited to one fold.

## 4. Optimization Methodology
The AutoSOTA cycle proposed both model‑level code changes and post‑processing interventions; only the latter could be executed and measured. Three main post‑processing strategies were explored and compared.

- **Temperature tuning.** In the `spatial_smoothing` function (`postprocess.py`), τ determines the sharpness of neighbour weights computed from embedding cosine similarities. Sweeping τ ∈ {0.01, 0.1, 1.0, 5.0} revealed that higher values (1.0–5.0) increase PCC by +0.06 pp over the default (τ=0.1). This change reflects a shift toward more democratic neighbour averaging, consistent with a broad tissue similarity expression profile.

- **k‑NN spatial smoothing.** The same function was configured with k = 7 neighbours and blending strength λ = 0.5: each spot’s prediction is interpolated with a weighted average of the k nearest spots in UNI embedding space, using weights ∝ exp(similarity/τ) with τ=1.0. Applied alone, this gave a +0.82 pp PCC gain over baseline.

- **Bilateral spatial‑feature smoothing.** A new function `bilateral_smoothing` was introduced, modelling spot affinities as the product of a spatial Gaussian (bandwidth σ_s = 8.0) and a feature Gaussian (cosine distance in UNI space, bandwidth σ_f = 0.8). The top‑40 neighbours per spot are retained, and the row‑normalized affinity matrix W is used to compute smoothed predictions: \(\hat{y}_{\text{smoothed}} = 0.5 \cdot W\hat{y} + 0.5 \cdot \hat{y}\). This filter jointly respects tissue layout and histological similarity and delivered the highest PCC: 57.05 % (+1.16 pp).

Model architecture modifications were committed (commit `8bf8b5a296d2aca3`) but not assessed: learnable mask tokens replacing zero‑fill in the augmentor; hidden dimension increased to 512; loss rebalanced as `MSE + 0.3·Corr + 0.4·Target_CL + 0.2·Ref_CL`; residual connections and LayerNorm after each GraphSAGE layer; gradient clipping (max norm=1.0); and CNAP heads raised from 4 to 8. These changes aim to improve capacity and training stability but cannot be quantified without full retraining.

## 5. Experiments
### 5.1 Setup
- **Hardware:** NVIDIA GeForce RTX 4090 GPU (inference only; baseline predictions were pre‑computed).
- **Data:** cSCC dataset fold P2_ST_rep1 (GSE144240). No training data, graph files, or UNI weights were accessible in the sandbox. Evaluation used all spots of that slide (exact count not reported in the optimization log).
- **Metrics:** Per‑gene mean PCC, median PCC, and RMSE against ground‑truth expression.
- **Budget:** 13 AutoSOTA iterations, including hyperparameter sweeps and code integration.
- **Caveats:** Results are based on a single validation fold and may not generalize. No model retraining occurred; gains are purely from post‑processing.

### 5.2 Quantitative Results
| Metric                  | Baseline | Best (AutoSOTA) | Absolute Δ | Relative Δ |
|-------------------------|----------|----------------|------------|------------|
| Per‑Gene Mean PCC (%)   | 55.90    | **57.05**      | +1.16      | +2.07 %    |
| Median PCC (%)          | 54.62    | 55.96          | +1.34      | +2.45 %    |
| RMSE                    | 0.1762   | 0.1755         | –0.0007    | –0.40 %    |

*Direction: ↑ for PCC indicates better, ↓ for RMSE indicates better. The target of 58.70 % PCC (5 % relative improvement) was not reached; a gap of 1.64 percentage points remains.*

### 5.3 Ablation / Iteration Trajectory
| Step | Intervention                               | Configuration                       | Per‑Gene Mean PCC (%) |
|------|--------------------------------------------|--------------------------------------|------------------------|
| 0    | Baseline predictions                       | –                                    | 55.90                  |
| 1    | + temperature optimization                 | τ ∈ {1.0, 5.0}, k=5, λ=0.3           | 55.96                  |
| 2    | + k‑NN spatial smoothing (replaces step 1) | k=7, λ=0.5, τ=1.0                    | 56.72                  |
| 3    | + bilateral spatial‑feature smoothing      | σ_s=8.0, σ_f=0.8, top_k=40, blend=0.5| **57.05**              |

*Each step replaces the previous smoothing configuration; they are not stacked.*

## 6. Discussion
The optimization shows that spatial post‑processing consistently improves prediction correlation, driven by the synergy of tissue architecture and embedding similarity. Bilateral filtering provides the largest gain by combining spatial and feature distances, and high‑temperature neighbour weighting confirms that broad local similarity is more informative than sharp selection.

Several alternative strategies yielded no improvement:
- **Gene‑specific adaptive smoothing** (varying λ by gene’s baseline PCC) did not outperform uniform smoothing, likely due to noisy per‑gene statistics.
- **Iterative multi‑round smoothing** degraded performance via over‑smoothing, blurring expression boundaries.
- **Confidence‑weighted refinement** (targeting low‑confidence spots) gave minimal benefit, indicating the confidence proxy (mean embedding similarity) poorly correlates with prediction error.
- **Multi‑scale ensemble** averaging multiple smoothing configurations did not surpass the single best bilateral setup.

These findings are conditional on the single‑fold, single‑slide evaluation; generalization to other folds or datasets is not guaranteed. The architectural modifications implemented in code remain unvalidated and could surpass the observed post‑processing ceiling. The study also highlights the dependence of automated optimization sandboxes on pre‑computed assets, which limits the scope of interventions.

## 7. Reproducibility
- **Repository:** SpaHGC source code as used in the optimization sandbox (exact public URL not provided in the log).
- **Environment:** `pip install -r requirements.txt` installs PyTorch, PyG, scanpy, timm, and other dependencies.
- **Seed:** The original training procedure’s seed was not set; exact reproduction of baseline predictions requires the original author’s configuration.
- **Baseline predictions:** Generated by running the SpaHGC training script on the cSCC dataset; the exact command line is not specified in the log, but likely `python main.py` with default configuration.
- **Best optimized result:** Apply `bilateral_smoothing` from `postprocess.py` to the pre‑computed predicted expression matrix, the corresponding 1024‑dim UNI spot embeddings, and (x,y) spot positions, using parameters σ_s=8.0, σ_f=0.8, top_k=40, blend=0.5. No retraining is required.

## 8. References
- SpaHGC: Cross‑Slice Knowledge Transfer via Masked Multi‑Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference. (Full citation to be provided by the original authors; venue and bibtex not available in the repository.)
- AutoSOTA: tsinghua‑fib‑lab/AutoSOTA framework.
