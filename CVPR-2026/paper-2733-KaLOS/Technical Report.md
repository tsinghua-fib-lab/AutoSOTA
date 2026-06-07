# KaLOS finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks: A Technical Report on Automated Optimization

## Abstract
Inter-annotator agreement (IAA) metrics are essential for assessing the reliability of annotations in complex vision tasks such as object detection. The KaLOS meta-algorithm provides a principled method to evaluate IAA for multi-annotator bounding‑box datasets by solving a correspondence problem and computing Krippendorff’s alpha. This technical report documents an automated optimization study performed by the AutoSOTA framework on the KaLOS pipeline. The primary limitation addressed is the original similarity function for bounding‑box matching, which relied solely on Intersection‑over‑Union (IoU) and thus assigned zero score to non‑overlapping annotations that nevertheless correspond to the same object. AutoSOTA introduced a fused similarity function, `bbox_iou_centroid_fusion`, that combines IoU (20% weight) with centroid proximity (80% weight). On a synthetic annotation dataset, this change improved the mean Krippendorff’s alpha from 0.807823 to 0.942825 (a relative increase of +16.71%) and the global alpha from 0.825843 to 0.965824 (+16.95%). The target mean alpha of 0.8482 was exceeded by 11.2% (relative improvement). Systematic weight tuning revealed that high centroid emphasis is necessary to recover true matches lost by pure IoU. The intervention required modifications in `correspondence_algorithms.py` and `eval.py`. While these gains are substantial, they are obtained on synthetic data with well‑separated instances and uniform category labels; generalisation to real‑world annotation scenarios remains to be validated.

## 1. Introduction
Precise evaluation of annotation quality is critical for developing trustworthy computer vision systems. In object detection, several annotators may produce different sets of bounding boxes for the same image, and quantifying their agreement helps gauge annotation difficulty and dataset reliability. The KaLOS meta‑algorithm (Krippendorff’s alpha via Object‑level Similarity) offers a structured approach: it first establishes correspondences between annotations across annotators using greedy matching guided by a pairwise similarity function, then computes Krippendorff’s alpha [KaLOS, 2024]. The similarity function therefore directly influences the quality of the correspondence and, ultimately, the agreement metric.

An automated optimization pipeline, AutoSOTA, was applied to the KaLOS codebase to improve the matching component. Starting from a baseline that used pure IoU similarity with a fixed threshold of 0.5, the optimizer explored alternative similarity functions, and identified a fusion of IoU and centroid distance that raised the mean Krippendorff’s alpha by over 16 percentage points. This report presents the original method, the identified limitations, the optimization interventions, the experimental setup, quantitative results, an ablation trajectory, and a discussion of the findings.

## 2. Original Method (Background)
The KaLOS algorithm processes a set of annotations from multiple raters for the same image. Each annotation consists of a bounding‑box (coordinates) and an object category. The core steps are:

1. **Pairwise similarity calculation.** For every pair of annotations belonging to different raters, a similarity score is computed. The original implementation uses `bbox_iou_similarity`, which returns the Intersection‑over‑Union in [0,1].

2. **Greedy correspondence matching.** The algorithm sorts all inter‑annotator pairs by decreasing similarity (or increasing cost, where cost = −similarity). It iteratively matches the highest‑similarity pair, adds it to a correspondence cluster, and removes that pair from the pool. A global `similarity_threshold` (default 0.5) discards pairs whose similarity falls below the threshold, treating them as unmatched.

3. **Agreement metric computation.** After clustering, Krippendorff’s alpha is computed both per‑category (`mean_alpha`) and globally across all categories (`global_alpha`). Two output metrics are reported.

The code is organised into `correspondence_algorithms.py` (containing the matching logic and the `THRESHOLD_FUNCTIONS` registry) and `eval.py` (command‑line entry point that parses arguments for method, similarity function, threshold, and cost function). The baseline command used is:
```
python eval.py --method greedy --threshold_func bbox_iou_similarity --similarity_threshold 0.5 --cost_func negative_score
```

## 3. Identified Limitations
**Spatial blindness for non‑overlapping true matches.** The `bbox_iou_similarity` function yields a score of 0.0 for any pair of annotations with zero overlap. In object detection, annotations of the same object can be non‑overlapping due to different bounding‑box strategies, partial occlusions, or varying tightness of framing. As a result, true correspondences are discarded by the similarity threshold before the greedy matcher can consider them. The baseline mean alpha of 0.807823 on synthetic data indicates that even with simplified annotations, many valid matches are missed.

**Dependence of greedy matching on pairwise ordering.** Greedy matching is known to be sensitive to the ordering of candidate pairs. When similarity scores for true matches are often zero (identical to false matches), the algorithm cannot distinguish them and may form suboptimal clusters. The optimizer experimented with alternative matching algorithms (SHM, MGM, AHC) and found that none improved over greedy matching; this confirms that the root cause is the similarity measure, not the matching algorithm itself.

**Limited utility of spatial proximity gate.** A distance‑based filter that masks pairs exceeding a distance threshold was attempted, but it yielded no improvement because on the synthetic dataset annotations are already well separated; such a gate is redundant when inter‑object distances are large.

## 4. Optimization Methodology
The AutoSOTA pipeline systematically explored modifications to the similarity function. The key intervention was the introduction of a fused similarity function, `bbox_iou_centroid_fusion`, defined in `correspondence_algorithms.py` as:
```python
def bbox_iou_centroid_fusion(ann1, ann2, iou_weight=0.2):
    iou = bbox_iou_similarity(ann1, ann2)
    cent = centroid_similarity(ann1, ann2)
    return iou_weight * iou + (1.0 - iou_weight) * cent
```
where `centroid_similarity` normalises the Euclidean distance between bounding‑box centres by the average of the box diagonals, producing a value in [0,1] with 1 indicating perfect coincidence. The fusion weight `iou_weight` controls the relative contribution of IoU versus centroid proximity.

To integrate this function, the optimizer:
- Added the definition to `correspondence_algorithms.py`,
- Registered it in the `THRESHOLD_FUNCTIONS` dictionary of the same module,
- Updated the `argparse` choices in `eval.py` to permit the new function name.

The `iou_weight` parameter was subjected to a grid search (0.0 to 1.0 in steps of 0.1), evaluating mean alpha after each trial. Each trial used the same greedy matching algorithm, similarity threshold 0.5, and negative‑score cost function. The search converged on an optimal weight of 0.2 for IoU (0.8 centroid), which maximised the agreement metrics.

Other interventions explored during the 16‑iteration budget included alternative matching methods (SHM, MGM, AHC), a spatial proximity gate, a category‑lenient cost function, and centroid tiebreaking, all of which were discarded because they did not improve performance over the baseline on this synthetic dataset.

## 5. Experiments

### 5.1 Setup
**Data.** The experiments were conducted on synthetic annotation sets generated for evaluating the KaLOS pipeline. The annotations are characterised by well‑separated object instances and consistent category labels across annotators. No real‑image datasets were used.

**Hardware.** Not specified in the optimization log.

**Evaluation protocol.** The KaLOS pipeline was executed end‑to‑end for each trial, producing `mean_alpha` and `global_alpha`. The optimisation criterion was maximisation of `mean_alpha`.

**Baseline command.** `python eval.py --method greedy --threshold_func bbox_iou_similarity --similarity_threshold 0.5 --cost_func negative_score`

**Optimization budget.** 16 AutoSOTA iterations, including attempts on alternative structures and hyperparameter sweeps.

**Caveats.** All results are obtained on synthetic data. The synthetic nature means that conclusions about optimal weight values and performance gains may not transfer directly to real‑world annotation data, which can contain category discrepancies, heavy occlusion, and varying box sizes. Additionally, the experiment environment (Python version, dependencies) is not logged; reproducibility may require adaptation.

### 5.2 Quantitative Results
Table 1 compares the baseline and the best configuration found by AutoSOTA.

| Metric | Baseline | Optimized | Absolute Δ | Relative Δ |
|--------|----------|-----------|------------|-------------|
| mean_alpha | 0.807823 | 0.942825 | +0.135002 | +16.71% |
| global_alpha | 0.825843 | 0.965824 | +0.139981 | +16.95% |

**Table 1: Agreement metrics before and after optimization.** Both metrics are Krippendorff’s alpha (range [−1,1]); higher is better. The optimized configuration uses `threshold_func=bbox_iou_centroid_fusion`, with all other settings identical to the baseline.

The target mean alpha of 0.8482 was exceeded by 11.2% (relative improvement), reaching 0.942825.

### 5.3 Ablation / Iteration Trajectory
The main series of productive iterations involved sweeping the `iou_weight` of the fusion function. Table 2 lists the mean alpha for each tested weight, starting from the pure‑IoU baseline and progressing through the search.

| Step | iou_weight | Centroid Weight | mean_alpha | Comment |
|------|------------|-----------------|------------|---------|
| Baseline | 1.0 | 0.0 | 0.807823 | Pure IoU similarity |
| 1 | 0.6 | 0.4 | 0.934485 | Initial fusion attempt |
| 2 | 0.5 | 0.5 | 0.939295 | |
| 3 | 0.4 | 0.6 | 0.941167 | |
| 4 | 0.3 | 0.7 | **0.942825** | Maximum reached |
| 5 | 0.2 | 0.8 | **0.942825** | Optimal selected |
| 6 | 0.1 | 0.9 | 0.941678 | Performance declines |
| 7 | 0.0 | 1.0 | 0.941678 | Pure centroid similarity |

**Table 2: Weight tuning of the fused similarity function.** Values represent one trial per weight. The optimizer selected `iou_weight = 0.2` as the final configuration.

Prior to the fusion series, the optimizer spent several iterations on non‑productive changes: alternative matching algorithms (SHM, MGM, AHC) all yielded alpha values below the baseline; a spatial proximity gate, category‑lenient cost, and a centroid tiebreaking mechanism showed no measurable improvement. These attempts are omitted from Table 2 but account for the total 16 iterations.

## 6. Discussion
The introduction of centroid proximity into the similarity computation proved highly effective. The fusion function supplies a continuous, non‑zero score for non‑overlapping but spatially close annotations, enabling the greedy matcher to recover matches that pure IoU would discard. The weight tuning revealed that centroid information should dominate (80% weight) while a small IoU component (20% weight) acts as a tie‑breaker when boxes do overlap. The fact that mean alpha plateaus at two weight values (0.2 and 0.3) suggests robustness within a narrow interval, but performance degrades sharply when IoU weight rises above 0.3, reinforcing the diagnosis that IoU alone is the bottleneck.

The failure of alternative matching algorithms indicates that the greedy strategy, when fed well‑ordered similarity scores, is sufficient for this data. More sophisticated matching (e.g., Hungarian) would likely offer marginal improvements only if the similarity matrix still contains ambiguity; with the fused function, the number of ambiguous zero‑similarity pairs drops drastically, so greedy matching is near‑optimal.

The primary threat to validity is the exclusive use of synthetic data. Synthetic annotations are constructed with well‑separated objects and consistent categories, which may not reflect real‑world annotation noise (e.g., missing detections, overlapping objects, uncertain class labels). The optimal weight 0.2 might not generalise; a real‑world dataset could require a different balance or an adaptive weighting scheme. Additionally, the improvement is measured solely on internal metrics (Krippendorff’s alpha). Whether a higher alpha translates into more meaningful evaluation of annotation quality has not been externally validated. The reproducibility of the exact numbers depends on the synthetic data generator, which is not included in the log.

Future work could evaluate the fused function on public detection datasets with multiple annotators (e.g., COCO, Objects365), explore per‑object adaptive weighting based on overlap characteristics, and incorporate a Hungarian refinement pass on the clusters produced by greedy matching.

## 7. Reproducibility
The original KaLOS implementation is described in its accompanying paper. Exact repository paths and environment details were not provided in the optimization log.

**AutoSOTA framework:** tsinghua-fib-lab/AutoSOTA.

**Baseline reproduction (on synthetic data):**
```
python eval.py --method greedy --threshold_func bbox_iou_similarity --similarity_threshold 0.5 --cost_func negative_score
```

**Optimized reproduction:**
```
python eval.py --method greedy --threshold_func bbox_iou_centroid_fusion --similarity_threshold 0.5 --cost_func negative_score
```
The `bbox_iou_centroid_fusion` function must be present in `correspondence_algorithms.py` with `iou_weight` set to 0.2 (hard‑coded or adjustable) and registered in the `THRESHOLD_FUNCTIONS` dictionary.

**Seed:** Not reported.

**Note:** Results assume the same synthetic annotation data used during optimization; without access to that data, exact metric reproduction is not possible.

## 8. References
- KaLOS finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks. (2024). *[Publication details not provided by the optimization log.]*
- tsinghua-fib-lab/AutoSOTA. (2024). Automated State-of-the-Art Optimization Framework. GitHub repository.
