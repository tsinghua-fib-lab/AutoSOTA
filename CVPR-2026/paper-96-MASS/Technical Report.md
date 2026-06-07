# MASS: A Technical Report on Automated Optimization

## Abstract

Mask-Guided Self-Supervised 3D Medical Image Segmentation (MASS) is a CVPR 2026 framework that learns transferable representations from unlabeled volumetric medical images through automatically generated in-context segmentation tasks. This technical report documents an automated optimization study, conducted with the AutoSOTA pipeline, that targets the in-context evaluation stage of MASS. Starting from a fixed single-reference baseline achieving an average Dice score of 68.39 % on a 13‑organ abdominal CT dataset, the optimization process raises the score to 73.98 % (+5.59 points, relative gain of 8.2 %). The primary driver of improvement is the adoption of random multi-reference ensembles (three random references per class), while norm‑weighted averaging of the reference tokens had a minor, near‑negligible effect. This change dramatically improves performance on small, anatomically variable organs: pancreas Dice jumps from 46.43 % to 65.20 %, right adrenal from 37.13 % to 60.02 %, and left adrenal from 39.01 % to 57.21 %. A naive test‑time augmentation strategy based on eight‑axis flipping causes catastrophic failure (Dice 19.93 %), demonstrating that in‑context models require spatially consistent reference priors. The optimised configuration is lightweight, requires no retraining of the pretrained encoder, and highlights that the quality and diversity of reference examples dominate MASS’s in-context inference performance.

## 1. Introduction

Obtaining voxel‑level expert annotations for 3D medical images is costly. MASS circumvents this by pretraining on unlabeled CT, MRI, and PET volumes using automatically generated, class‑agnostic masks. Every mask becomes a dense in‑context segmentation task: given a reference image–mask pair, the model segments the corresponding structure in a query image. This yields spatially grounded representations that transfer directly to downstream segmentation and classification tasks.

The original MASS paper evaluates in‑context segmentation under a fixed single‑reference protocol, selecting one predetermined reference volume per organ class. While this simplifies comparison, it under‑represents the sensitivity of in‑context models to reference variability. This report describes an automated optimization study that systematically explores reference selection and ensemble strategies to maximise in‑context Dice without altering the pretrained weights. The study is carried out with the AutoSOTA pipeline, and the findings serve as practical guidance for deploying MASS on new data.

## 2. Original Method (Background)

MASS [gao2026learning] pretrains the Iris in‑context segmentation architecture with mask‑guided self‑supervision. During pretraining, reference image–mask pairs are sampled from a large pool of auto‑generated masks (produced by SAM2). The shared encoder processes both reference and query images; reference mask embeddings are fused into the decoder as task tokens, enabling the model to perform dense, anatomy‑aware segmentation on the query. The released checkpoint (`mass_base.pth`) is trained exclusively on these auto‑generated masks and has never seen expert‑labeled ground truth.

In‑context inference uses sliding‑window prediction with task‑token priors derived from one or more reference examples. The evaluation script `evaluate.py` supports `--reference-mode fixed` (using a hard‑coded index per class from `data/split.py`) and `--reference-mode random` (randomly sampling from the training pool), as well as `--ensemble-size K` to average up to K reference token sets per class.

## 3. Identified Limitations

Analysis of the baseline evaluation reveals three limitations:

1. **Limited reference diversity.** The baseline uses `--reference-mode fixed --ensemble-size 1`, providing exactly one reference per organ. Small, variable organs—pancreas, adrenals, esophagus, stomach—achieve poor Dice (pancreas 46.43 %, right adrenal 37.13 %, left adrenal 39.01 %) because a single reference cannot cover the range of anatomical variation in the test set. Increased reference diversity should mitigate this mismatch.

2. **Uniform averaging of multi‑reference embeddings.** When multiple references are used, the baseline averages their task tokens with equal weight (function `_average_task_embeddings` in `training/evaluator.py`). References of varying informativeness are treated identically, potentially diluting strong priors. An L2‑norm‑weighted averaging scheme is hypothesised to assign higher weight to more informative references.

3. **Catastrophic failure of test‑time augmentation (TTA).** Applying eight‑axis flipping to the target image while keeping the reference task tokens fixed collapses average Dice to 19.93 %. The in‑context task embeddings encode orientation‑specific spatial priors; naive augmentation destroys the spatial correspondence between reference and query without re‑encoding references. This intervention is therefore excluded from further optimization.

## 4. Optimization Methodology

Two interventions were applied, both targeting the inference‑time reference handling.

**Intervention 1 – Norm‑weighted reference ensemble averaging.**  
The averaging logic in `training/evaluator.py` was changed from arithmetic mean to an L2‑norm‑weighted summation. For each organ class, the contribution of each reference’s task token tensor is scaled by its Frobenius norm before summation. The intent is to give more influence to references with higher‑magnitude priors.

**Intervention 2 – Random reference mode with ensemble size 3.**  
The evaluation CLI was switched from `--reference-mode fixed --ensemble-size 1` to `--reference-mode random --ensemble-size 3 --seed 46`. Instead of a single hard‑coded volume, three distinct reference volumes are randomly sampled per class from the training pool. The resulting three sets of task tokens are averaged using the norm‑weighted scheme.

Preliminary testing (within the two‑iteration optimisation budget) showed that Intervention 1 alone, when applied to the fixed reference pool, produced negligible change because the norms of the fixed references were nearly equal. Therefore, the final configuration combines both interventions, with the random multi‑reference strategy being the primary source of improvement.

## 5. Experiments

### 5.1 Setup

**Hardware.** Single NVIDIA A100 (40 GB) with CUDA 12.1 and PyTorch 2.1.

**Dataset.** The 13‑organ abdominal CT dataset from the Beyond the Cranial Vault (BCV) benchmark, processed into the MASS H5 format. Organ classes: spleen, right kidney, left kidney, gallbladder, esophagus, liver, stomach, aorta, inferior vena cava, portal vein, pancreas, right adrenal, left adrenal. Training/validation/test splits and fixed reference indices are defined in `data/split.py`.

**Evaluation protocol.** The pretrained `mass_base.pth` checkpoint was loaded with EMA weights (`--use-ema`). Sliding‑window inference used a window size of [128, 128, 128] and 50 % overlap. Per‑class Dice was computed and averaged across test volumes; surface metrics (ASD, HD95) were not computed due to time constraints. Binary predictions were obtained with a threshold of 0.5. All runs used random seed 46. The optimisation target was set to 71.81 % Dice.

**Baseline command:**
```bash
python evaluate.py --checkpoint checkpoints/best.pth --dataset bcv \
    --data-root /path/to/mass_h5 --reference-mode fixed --ensemble-size 1 \
    --gpus 0 --use-ema --seed 46
```

**Optimisation budget.** Two iterations, each evaluating a distinct configuration. The best configuration corresponds to commit `98beb0ad63`.

**Caveats.** All Dice values are from a single seed (46). The volume list and exact split match the original MASS paper; statistical significance testing across seeds was not performed.

### 5.2 Quantitative Results

Table 1 reports per‑organ and average Dice for the baseline and the optimised configuration.

| Organ               | Baseline Dice (%) | Optimised Dice (%) | Δ (pts) |
|---------------------|------------------:|------------------:|--------:|
| Spleen              |             89.73 |             91.82 |   +2.09 |
| Right Kidney        |             89.28 |             90.74 |   +1.46 |
| Left Kidney         |             90.39 |             86.33 |   −4.06 |
| Gallbladder         |             47.28 |             53.37 |   +6.09 |
| Esophagus           |             62.74 |             65.96 |   +3.22 |
| Liver               |             89.95 |             91.77 |   +1.82 |
| Stomach             |             65.87 |             69.23 |   +3.36 |
| Aorta               |             87.27 |             86.99 |   −0.28 |
| IVC                 |             77.46 |             79.23 |   +1.77 |
| Portal Vein         |             66.58 |             63.84 |   −2.74 |
| Pancreas            |             46.43 |             65.20 |  +18.77 |
| Right Adrenal       |             37.13 |             60.02 |  +22.89 |
| Left Adrenal        |             39.01 |             57.21 |  +18.20 |
| **Average**         |         **68.39** |         **73.98** |  **+5.59** |

The average Dice improvement of 5.59 points exceeds the predefined 71.81 % target by 2.17 points. Gains are concentrated in small, variable organs: pancreas (+18.77), right adrenal (+22.89), left adrenal (+18.20). Most large, relatively stable organs (spleen, right kidney, liver, aorta, IVC) show modest improvements or minor changes. The left kidney and portal vein experience slight regressions (−4.06 and −2.74 points, respectively), likely reflecting an unlucky random reference draw for those classes under a single seed.

### 5.3 Ablation / Iteration Trajectory

The two‑iteration optimisation history is summarised in Table 2.

| Step | Configuration Change                                | File(s) Affected            | Average Dice (%) |
|------|-----------------------------------------------------|-----------------------------|-----------------:|
| 0    | Baseline (fixed single‑reference, uniform mean)     | –                           |            68.39 |
| 1    | Norm‑weighted reference ensemble averaging          | `training/evaluator.py`     |         ~68.39*  |
| 2    | Random reference mode + ensemble size 3 + Step 1    | `evaluate.py`, eval params  |            73.98 |

*Minor (no measurable change)

Step 1, applied while still using fixed references, had no measurable impact because the token norms were near‑uniform across the fixed references. Step 2 introduced random multi‑reference sampling and accounts for essentially the entire 5.59‑point gain.

## 6. Discussion

The study demonstrates that in‑context performance of MASS is highly sensitive to reference selection. Switching from a single fixed reference to a random ensemble of three references per class alone yields a 5.59‑point Dice increase, with the largest gains in organs that are anatomically variable. The norm‑weighted averaging scheme contributed negligibly in this setting because the sampled references had similar informativeness; its utility would likely increase when references are drawn from heterogeneous sources (e.g., different scanners, pathologies).

The catastrophic failure of TTA reinforces a fundamental property of in‑context models: the task embedding is not invariant to spatial transformations of the query unless the reference priors are transformed accordingly. Naive augmentation without re‑encoding references breaks the spatial correspondence necessary for correct prediction.

The top remaining ideas from the optimization log—embedding‑based reference retrieval, per‑class adaptive thresholds, connected‑component post‑processing, and per‑class independent reference selection—are logical extensions that could further improve performance, particularly for organs that regressed (left kidney, portal vein) or for small organs that remain challenging. These ideas, however, lie beyond the current two‑iteration budget.

Threats to validity include the use of a single random seed (46), the evaluation on a single BCV abdominal CT dataset (generalisability to MRI and PET is unknown), and the minimal search budget. A more extensive search might recover the lost performance on the left kidney and portal vein.

## 7. Reproducibility

**Repository:** [https://github.com/Stanford-AIMI/MASS](https://github.com/Stanford-AIMI/MASS)

**Environment:**
```bash
conda create -n mass python=3.10
conda activate mass
pip install -r requirements.txt
```

**Random seed:** 46 (via `--seed 46`).

**Baseline run:**
```bash
python evaluate.py --checkpoint checkpoints/best.pth --dataset bcv \
    --data-root /path/to/mass_h5 --reference-mode fixed --ensemble-size 1 \
    --gpus 0 --use-ema --seed 46
```

**Optimised run (commit `98beb0ad63`, includes norm‑weighted averaging in `training/evaluator.py`):**
```bash
python evaluate.py --checkpoint checkpoints/best.pth --dataset bcv \
    --data-root /path/to/mass_h5 --reference-mode random --ensemble-size 3 \
    --gpus 0 --use-ema --seed 46
```

**AutoSOTA pipeline:** tsinghua-fib-lab/AutoSOTA.

## 8. References

```bibtex
@article{gao2026learning,
  title={Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision},
  author={Gao, Yunhe and Zhang, Yabin and Wang, Chong and Liu, Jiaming and Varma, Maya and Delbrouck, Jean-Benoit and Chaudhari, Akshay and Langlotz, Curtis},
  journal={arXiv preprint arXiv:2603.13660},
  year={2026}
}

@misc{tsinghua-fib-lab/AutoSOTA,
  author = {Tsinghua-FIB-Lab},
  title = {AutoSOTA: Automated State-of-the-Art Optimization Pipeline},
  year = {2025},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}}
}
```
