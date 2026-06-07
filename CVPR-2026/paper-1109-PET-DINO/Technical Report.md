# PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training: A Technical Report on Automated Optimization

## Abstract

This report presents an inference‑only hyperparameter optimization study of the PET‑DINO detector—a CVPR 2026 Highlight that unifies text and visual prompting via Alignment‑Friendly Visual Prompt Generation (AFVPG), Intra‑Batch Parallel Prompting (IBP), and Dynamic Memory‑Driven Prompting (DMD). Using the AutoSOTA pipeline (tsinghua‑fib‑lab/AutoSOTA), four iterations were executed on the COCO 2017 validation set to maximize detection average precision (AP) of a frozen Swin‑T model. The optimal configuration lowered the score threshold from 0.05 to 0.0, yielding AP = 0.640—a negligible gain of +0.001 (+0.16 %) over the baseline 0.639. Switching to text‑only prompt mode caused a catastrophic drop to AP = 0.495 (−0.145), and increasing the maximum detections per image to 1000 triggered a runtime error. The pre‑set target of 0.6689 (5 % relative improvement) was not reached. The study reveals that PET‑DINO’s default inference configuration is already near‑optimal; the DETR bipartite‑matching architecture removes non‑maximum suppression, a common tuning lever, and infrastructure constraints prevented multi‑scale test‑time augmentation. For frozen‑weight detectors of this family, inference‑only parameter sweeps offer sub‑percent gains; material improvements require training‑level modifications.

## 1. Introduction

Object detection has been transformed by open‑vocabulary models that accept free‑form text prompts. PET‑DINO (Fu et al., 2026) extends this paradigm by introducing visual prompts—bounding boxes or extracted embeddings—alongside text, yielding a universal detector. At the core are AFVPG, which aligns visual prompt features with text representations, and two training strategies, IBP and DMD, that jointly handle multiple prompt modalities. The method is evaluated on COCO, LVIS, and the ODinW35 suite, demonstrating strong zero‑shot and fine‑tuned performance with both text (Text) and visual (Visual‑I and Visual‑G) modes. As a practical matter, the code release includes a default inference configuration thought to be well‑tuned. In this work, we subject the publicly available PET‑DINO repository to an automated optimization pipeline—AutoSOTA—that sweeps the most accessible inference hyperparameters while the checkpoint remains frozen. The goal is to quantify the attainable improvement and to identify the structural factors that bound it. The investigation proceeds by examining four targeted parameter changes, recording the impact on COCO AP, and analyzing why a 5 % relative gain was infeasible.

## 2. Original Method

PET‑DINO builds on MM‑Grounding‑DINO, a DETR‑style framework with a Swin‑Transformer backbone, a deformable‑DETR encoder‑decoder, and bipartite matching for direct set prediction. Its distinctive modules are:

- **AFVPG**: an alignment‑friendly pipeline that encodes visual exemplars (bounding‑box crops or external embeddings) into prompts that remain distributionally aligned with text features.
- **IBP**: a training technique that processes text and visual prompts within the same mini‑batch, enabling the decoder to learn cross‑modal correspondences.
- **DMD**: a memory‑driven strategy that expands the prompt space with dynamically retrieved visual examples, improving open‑set recognition.

At inference, the detector supports three prompt types: `Text` (class name queries), `Visual` (visual cues with instance‑level evaluation), and a gallery‑style Visual‑G mode that leverages pre‑extracted visual embeddings. For COCO evaluation, the released script (`tools/dist_test.sh`) defaults to `prompt_type='Visual'`. The default inference parameters originate from MMDetection’s base configuration and include a score threshold of 0.05 (`model.test_cfg.rcnn.score_thr`) and a maximum of 100 detections per image (`model.test_cfg.rcnn.max_per_img`). Because the DETR head trains a fixed set of object queries and uses bipartite matching, no non‑maximum suppression (NMS) is employed—a design that removes a conventional tuning dimension.

## 3. Identified Limitations

Analysis of the optimization log and the repository structure exposes four constraints that limited the outcome.

1. **Inference‑only scope.** All model weights are frozen; only post‑processing parameters are adjustable. The optimization log explicitly states that this restriction is insufficient for a 5 % improvement.
2. **Absence of NMS.** The bipartite matching pipeline eschews NMS, eliminating thresholds such as `nms_iou_threshold` and `pre_nms_topk` that frequently yield substantial gains in anchor‑based detectors.
3. **Already optimal defaults.** The baseline AP of 0.639 reflects a near‑optimal configuration; the log notes that “the default PET‑DINO inference config is already well‑tuned.” The only directional change (score threshold to zero) produced a sub‑noise‑floor improvement.
4. **Infrastructure barriers.** The Docker environment provided a 20 GB read‑only overlay filesystem with merely 400 MB free, making multi‑scale test‑time augmentation (TTA) or ensemble storage impossible. The NFS mount was read‑only, restricting config modifications to command‑line overrides, and the Docker proxy suppressed debug output, complicating failure analysis.

## 4. Optimization Methodology

The AutoSOTA pipeline attempted four interventions, each altering one parameter accessible via the `--cfg-options` flag of `tools/dist_test.sh`. The baseline evaluation used `model.test_cfg.prompt_type='Visual'`, the default score threshold 0.05, and `max_per_img=100`.

**Iteration 1: Score threshold set to zero.**
- **Parameter:** `model.test_cfg.rcnn.score_thr` from 0.05 → 0.0.
- **Rationale:** Removing the confidence floor recovers detections that might be suppressed, potentially improving recall at the cost of precision.
- **Outcome:** AP = 0.640 (+0.001); a marginal change indicating no false‑positive suppression by the default 0.05.

**Iteration 3: Text‑only prompt mode.**
- **Parameter:** `model.test_cfg.prompt_type` from `'Visual'` → `'Text'`.
- **Rationale:** A text‑only pathway might reduce visual‑prompt overhead and deliver cleaner class prototypes for standard COCO categories.
- **Outcome:** AP collapsed to 0.495 (−0.145), confirming that the visual prompt pathway is indispensable, as the AFVPG/I BP/DMD training priors are absent in text‑only mode.

**Iteration 4: Max detections per image raised to 1000.**
- **Parameter:** `model.test_cfg.rcnn.max_per_img` from 100 → 1000.
- **Rationale:** A larger cap may recover valid objects in crowded scenes.
- **Outcome:** Runtime error; the log documents that `max_per_img` exceeding 300 causes a failure, likely a hard‑coded buffer or memory limit.

No other accessible parameters (e.g., TTA) could be explored due to disk space constraints.

## 5. Experiments

### 5.1 Setup

- **Hardware:** 2× NVIDIA A100 GPUs; each distributed evaluation ran 8 processes and required approximately 8 minutes.
- **Data:** COCO 2017 validation set (5000 images, 80 foreground classes).
- **Evaluation protocol:** Official COCO API via MMDetection, reporting AP, AP₅₀, AP₇₅, AP_S, AP_M, AP_L (all scales use the standard IoU thresholds). Direction of improvement is positive for all metrics.
- **Model:** Swin‑T PET‑DINO checkpoint (baseline commit 7830a46), loaded from the official HuggingFace release.
- **Baseline command:** `CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py && bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 --cfg-options model.test_cfg.prompt_type='Visual'`. The default score threshold is 0.05 and max_per_img is 100.
- **Budget:** Four iterations, each a single parameter change followed by a full evaluation.

### 5.2 Quantitative Results

The best configuration (Iteration 1) lowered the score threshold to 0.0. All values are single‑run results on the full val2017 set.

| Metric | Baseline (score_thr = 0.05) | Best (score_thr = 0.0) | Δ        |
|--------|-----------------------------|------------------------|----------|
| AP     | 0.639                       | 0.640                  | +0.001   |
| AP₅₀    | 0.817                       | 0.819                  | +0.002   |
| AP₇₅    | 0.714                       | 0.715                  | +0.001   |
| AP_S   | 0.490                       | 0.492                  | +0.002   |
| AP_M   | 0.680                       | 0.682                  | +0.002   |
| AP_L   | 0.811                       | 0.814                  | +0.003   |

The improvement is within the typical COCO evaluation noise (~0.1–0.2 AP) and provides no practical benefit.

### 5.3 Ablation / Iteration Trajectory

| Iteration | Change                | AP     | Notes                        |
|-----------|-----------------------|--------|------------------------------|
| baseline  | default config        | 0.639  | Visual prompt, score_thr 0.05|
| 1         | score_thr = 0.0       | 0.640  | Minimal improvement          |
| 3         | prompt_type = 'Text'  | 0.495  | Catastrophic degradation     |
| 4         | max_per_img = 1000    | FAIL   | Runtime error                |

Iteration 2 is absent from the optimization log; it is presumed to have been a duplicate or aborted run. The trajectory confirms that only the first change was accepted; all others degraded or broke the evaluation.

## 6. Discussion

The negligible gain from score threshold relaxation suggests that the model’s confidence scores are already well calibrated, with virtually no true positives falling below 0.05. This is consistent with the DETR bipartite‑matching loss, which encourages sharp score distributions. The catastrophic result of switching to text‑only prompts underscores that the AFVPG and the IBP/DMD training strategies are central to PET‑DINO’s performance; without visual context, the detector reverts to a conventional text‑prompted Grounding DINO, whose COCO AP is markedly lower. The runtime failure at `max_per_img > 300` indicates a hard‑coded limit in the post‑processing pipeline or a memory allocation ceiling in the given environment, ruling out further exploration of that parameter.

The most consequential factor in the failure to approach the 5 % target was the infrastructure restriction on disk space. Multi‑scale inference and flip/jitter augmentations are known to contribute 1–2 AP points for DETR‑based models, yet the read‑only overlay filesystem precluded storing the intermediate forward passes. Thus, the meagre outcome is partly an artefact of the container environment rather than an intrinsic ceiling of the model. Nevertheless, for end‑to‑end transformer detectors with learned queries, the inference‑side parameter space is so narrow that even without storage constraints, gains of 0.5 AP or more are rare; material improvements demand training‑time changes (e.g., longer schedule, stronger backbone, additional training data).

## 7. Reproducibility

- **Code:** PET‑DINO repository, commit 7830a46 (available at the project’s GitHub and HuggingFace).
- **Environment:** Python 3.8+, MMDetection installed per the official guide, plus `pip install -r requirements/multimodal.txt && pip install emoji ddd-dataset && pip install git+https://github.com/lvis-dataset/lvis-api.git`. Numpy ≤ 1.23.
- **Model checkpoint:** Swin‑T PET‑DINO weight from HuggingFace.
- **Baseline evaluation (Visual‑I):**
  ```bash
  CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
  bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
      --cfg-options model.test_cfg.prompt_type='Visual'
  ```
- **Optimized run (score_thr = 0.0):**
  ```bash
  CONFIG=configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py
  bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 \
      --cfg-options model.test_cfg.prompt_type='Visual' model.test_cfg.rcnn.score_thr=0.0
  ```
  Substitute `$CHECKPOINT` with the local path to the pretrained Swin‑T checkpoint.

## 8. References

```bibtex
@article{fu2026pet,
  title={PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training},
  author={Fu, Weifu and Li, Jinyang and Gao, Bin-Bin and Li, Jialin and Lin, Yuhuan and Deng, Hanqiu and Tao, Wenbing and Liu, Yong and Wang, Chengjie},
  journal={arXiv preprint arXiv:2604.00503},
  year={2026}
}

@software{autosota,
  author = {tsinghua-fib-lab},
  title = {AutoSOTA: Automated State-of-the-Art Optimization Pipeline},
  year = {2025},
  url = {https://github.com/tsinghua-fib-lab/AutoSOTA}
}
```
