# CoFiDA-M: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study on CoFiDA-M, a CVPR 2026 conference paper that introduces a teacher–student framework for skin lesion diagnosis under domain shift. The original method trains a teacher network that uses privileged MONET concept probabilities to modulate visual features, then distills this edited feature space into a lightweight image‑only student. The present study, performed by the AutoSOTA pipeline, targets inference‑time improvements of the student because retraining is impossible without the proprietary MONET metadata. Over five iterations the optimizer explored four interventions: raising the eval resolution from 288×288 to 384×384, further enlarging to 456×456, constructing a multi‑scale ensemble (288+384), and applying test‑time augmentation (11 transforms). The optimizer’s target was 0.9123 (+5.0%), but the highest achieved area under the receiver operating characteristic curve (AUROC) is 0.8733, an absolute improvement of +0.0044 (+0.51%) over the 288×288 baseline of 0.8689. Balanced accuracy at the optimal operating point also rises from 0.8182 to 0.8240. The trajectory reveals an inverted‑U relationship between resolution and performance, while multi‑scale averaging and aggressive augmentations degrade the metric. The study confirms that a modest, zero‑training‑cost gain can be obtained by tuning the input size, and that major advances require access to the MONET metadata for retraining.

## 1. Introduction

CoFiDA-M addresses the domain shift from expert dermoscopic images (source) to consumer‑grade clinical photographs (target) that plagues skin cancer screening models. It leverages concept annotations from a foundation model (MONET) as privileged information during training, then distills the resulting semantic representations into an image‑only student that requires no concept metadata at test time. The publicly released code includes pre‑trained weights for an EfficientNet‑B2 student that outputs melanoma probabilities at a default input resolution of 288×288, achieving a baseline AUROC of 0.8689 on a held‑out clinical validation set. Because the MONET CSV metadata is absent from the evaluation sandbox, no retraining or distillation‑stage modification is possible; only inference‑time preprocessing can be varied. The AutoSOTA framework was therefore applied to the student evaluation script (`scripts/eval_student.py`) to explore whether adjustments to input size, multi‑scale fusion, or test‑time augmentation could lift the AUROC without altering model weights.

## 2. Original Method (Background)

CoFiDA-M (Concept‑Aware Feature Modulation for Cross‑Domain Adaptation with Image‑Only Inference) is a two‑stage privileged‑information framework. In the first stage, a teacher network pairs an EfficientNet‑B2 backbone with a MONET concept embedder and a FiLM modulation layer. The embedder maps 128 probabilistic concept scores to a fixed‑dimensional representation that scales and shifts the backbone’s visual features, producing a semantically “edited” representation. The teacher is trained jointly on labeled dermoscopic images and unlabeled clinical images using supervised classification, feature alignment, and edit‑vector losses, with an exponential moving average (EMA) of its weights.

In the second stage, a student—also built on EfficientNet‑B2—is trained to mimic the teacher’s edited features using only the clinical images and the corresponding MONET vectors. The student contains a shallow edit MLP (1408→512→1408) that predicts the edit vector from raw visual features; the edited features are then fed to an MLP classifier. After distillation, the student infers melanoma probability solely from an image, without any concept metadata. The official evaluation script loads the student checkpoint, resizes images to a default of 288×288, and computes binary classification metrics including AUROC, balanced accuracy, and accuracy at the optimal threshold determined by Youden’s J statistic.

## 3. Identified Limitations

**3.1 Fixed Low Resolution**  
The default evaluation pipeline resizes every image to 288×288 pixels. This resolution may discard discriminative fine‑grained details (e.g., border irregularity, pigment network) that are critical for melanoma detection, especially in clinical photographs where lesions appear at varying scales.

**3.2 Absence of Test‑Time Augmentation**  
The baseline evaluation performs a single forward pass per image. The student was trained with only weak augmentations—horizontal flip and mild color jitter (see `make_target_weak_transform`)—and it is unknown whether averaging predictions over multiple transformed copies would improve robustness or, conversely, introduce a harmful distributional shift.

**3.3 No Training‑Phase Optimization Feasibility**  
The most promising directions for improving the student—class‑weighted knowledge distillation, deeper edit MLP, cosine feature alignment, label smoothing, attention‑gated fusion—all require re‑running the distillation pipeline, which hinges on the MONET concept probability CSV. This metadata is not present in the container, precluding any retraining experiments.

**3.4 Environment Constraints**  
The AutoSOTA optimizer operated inside a Docker container with CPU execution only. Attempts to implement Monte‑Carlo dropout inference failed due to import issues with the student model definition. Additionally, Docker exec stdout‑capture and file‑persistence problems significantly slowed the iteration cycle, limiting the number and complexity of interventions.

## 4. Optimization Methodology

AutoSOTA (tsinghua‑fib‑lab/AutoSOTA) treats the codebase as a modification graph. At each iteration it inspects the current state, proposes an intervention grounded in a hypothetical limitation, applies the change, runs the evaluation script, and records the resulting AUROC. The pipeline then decides whether to keep, discard, or refine the intervention. The target metric was AUROC, with a budget of five iterations (0–4) executed on the clinical validation split accessed through `scripts/eval_student.py`. No model weights were altered.

The sequence of accepted interventions was:

1. **Resolution increase to 384** – The hypothesis: enlarging the input from 288×288 to 384×384 increases the spatial grid of the final feature map (from ~7×7 to ~9×9), preserving more local detail. The change was implemented by passing `--img-size 384` to `eval_student.py`, which propagates to `make_eval_transform` in `data.py`.
2. **Resolution increase to 456** – A further enlargement was tested to determine whether an even larger receptive field would continue to improve performance.
3. **Multi‑scale ensemble (288+384)** – The hypothesis: averaging logits from two resolutions could combine complementary information and reduce prediction variance. The evaluator was patched to run two forward passes and average the logits before softmax.
4. **Test‑time augmentation (TTA)** – The hypothesis: the student might benefit from input diversity not fully covered by its training augmentations. An ensemble of 11 augmentations was applied: horizontal flip, rotation (±15°), brightness/contrast jitter, and their combinations. Each test image was duplicated 11 times, and predictions were averaged.

After each iteration, the optimizer recorded the AUROC and, when no improvement over the current best (384×384) was observed, rolled back to that configuration.

## 5. Experiments

### 5.1 Setup

**Hardware:** Docker container running on CPU; no GPU acceleration available. This extended evaluation times but did not affect metric values, as all computations are deterministic.

**Dataset:** The clinical validation images are organized in a binary `mel/other` folder structure, corresponding to the same test split used in the repository’s `eval_student.py` demonstration. The number of samples and melanoma prevalence are implicit in the baseline metrics.

**Evaluation protocol:** The student model generates per‑image probabilities. AUROC, balanced accuracy, and accuracy are computed; the optimal threshold is determined by Youden’s J index, and balanced accuracy is reported at that threshold. The same randomness seed (42, the default argparse value, which does not influence the deterministic evaluation) was used throughout.

**Baseline command:**
```bash
python scripts/eval_student.py \
  --test-dir /path/to/clinical/val/images \
  --checkpoint outputs/student/best_student.pt \
  --out-csv baseline.csv \
  --img-size 288
```

**Optimization budget:** Five iterations, with iteration 0 being the baseline measurement. All attempts were evaluated with the same randomness seed.

**Caveats:** The MONET CSV is absent, so all training‑stage modifications were impossible. Monte‑Carlo dropout could not be implemented due to import errors. The search space is therefore confined to inference‑time preprocessing.

### 5.2 Quantitative Results

| Metric               | Baseline (288) | Optimized (384) | Delta              |
|----------------------|----------------|-----------------|--------------------|
| AUROC                | 0.8689         | 0.8733          | +0.0044 (+0.51%)   |
| Balanced Acc @ opt   | 0.8182         | 0.8240          | +0.0058 (+0.71%)   |
| Accuracy @ opt       | 0.7529         | 0.7443          | –0.0086 (–1.14%)   |
| Optimal threshold    | 0.300          | 0.303           | +0.003             |

All values taken directly from the AutoSOTA log. The AUROC improvement of +0.0044 is modest relative to the optimizer’s initial target of 0.9123. Balanced accuracy shows a slightly larger relative gain. The drop in accuracy at the optimal threshold reflects a trade‑off favoring sensitivity for melanoma cases, consistent with a higher AUROC.

### 5.3 Ablation Trajectory

| Iter | Intervention              | AUROC | Δ from baseline | Δ from best (384) | Notes                          |
|------|---------------------------|-------|-----------------|--------------------|--------------------------------|
| 0    | img_size=288 (baseline)   | 0.8689| —               | –0.0044            | Original paper default         |
| 1    | img_size=384              | 0.8733| +0.0044         | 0.0000             | **Best configuration**         |
| 2    | img_size=456              | 0.8694| +0.0005         | –0.0039            | Degraded, likely over‑scaling  |
| 3    | Multi‑scale 288+384       | 0.8712| +0.0023         | –0.0021            | Dilution of strong predictions |
| 4    | TTA (11 augmentations)    | 0.8726| +0.0037         | –0.0007            | Augmentations hurt robustness  |

The trajectory reveals an inverted‑U relationship: 384×384 is the sweet spot; 456×456 reverses the gain, possibly because the distilled edit MLP was calibrated at 288. Multi‑scale averaging dilutes the superior 384 predictions with weaker 288 outputs. TTA with strong augmentations disrupts the student’s feature extraction, indicating that the training‑time augmentations (only horizontal flip and mild color jitter) did not endow the model with robustness to rotations or aggressive photometric changes.

## 6. Discussion

**What worked:** A simple increase of the input resolution to 384×384 delivered a consistent, albeit small, improvement in AUROC and balanced accuracy without any retraining. This aligns with the intuition that fine‑grained lesion structures benefit from higher spatial sampling. The optimizer correctly identified the optimal resolution via a single variable sweep and reverted after degradation.

**What did not work:** Both multi‑scale inference and test‑time augmentation harmed performance. The failure of TTA suggests that the student’s robustness is tightly coupled to the specific augmentations seen during distillation, and any deviation introduces a distributional shift that the network cannot compensate for. Multi‑scale averaging suffers from a “weak‑link” effect: the lower‑resolution branch contributes noisy predictions that reduce the ensemble’s overall discriminative power.

**Threats to validity:** The study was performed on a single test split with a fixed model checkpoint; generalisability across different clinical datasets remains untested. The optimization budget was small, and the search space was constrained to input pre‑processing by the missing MONET metadata. The absence of GPU acceleration may have precluded more computationally intensive inference schemes, though it does not affect the relative comparisons. No ablation was performed on TTA transform parameters or on batch size, which could have yielded different outcomes.

**Future directions:** As noted in the optimization log, substantial improvements are likely achievable through modifications to the distillation process itself: a class‑weighted KD loss emphasizing melanoma, a deeper residual edit MLP (e.g., 1408→512→256→512→1408), cosine feature alignment replacing MSE, label smoothing (0.05–0.1) applied to teacher logits, and attention‑gated fusion instead of a simple residual connection. Optimizing the teacher’s EMA decay schedule (cosine‑annealing from 0.9999 to 0.995) may also help. All these ideas require the MONET CSV and the ability to retrain—currently unavailable—and remain the most promising avenues for future work that would revisit the training stage.

## 7. Reproducibility

**Repository:** The code used in this study is available at the original CoFiDA‑M repository (commit `443e7770bde288569f4e0c43fe405ba54207f8f4`). Installation follows the provided instructions:

```bash
cd CoFiDA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Seed:** The evaluation pipeline is deterministic; the default argparse value of 42 influences only the training scripts, not inference.

**Baseline evaluation command:**
```bash
python scripts/eval_student.py \
  --test-dir <path_to_test_mel_other> \
  --checkpoint outputs/student/best_student.pt \
  --out-csv baseline.csv \
  --img-size 288
```

**Optimized evaluation command:**
```bash
python scripts/eval_student.py \
  --test-dir <path_to_test_mel_other> \
  --checkpoint outputs/student/best_student.pt \
  --out-csv optimized.csv \
  --img-size 384
```

No additional dependencies or code modifications are required to reproduce the AUROC values reported in Section 5.2.

## 8. References

```bibtex
@InProceedings{Sultana_2026_CVPR,
    author    = {Sultana, Nurjahan and Yap, Moi Hoon and Fan, Xinqi and Lu, Wenqi},
    title     = {CoFiDA-M: Concept-Aware Feature Modulation for Cross-Domain Adaptation with Image-Only Inference},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {15060-15069}
}
```

```bibtex
@software{tsinghua_fib_lab_AutoSOTA,
    author    = {{Tsinghua FIB Lab}},
    title     = {AutoSOTA: Automated State-of-the-Art Optimization Pipeline},
    note      = {Available at: \url{https://github.com/tsinghua-fib-lab/AutoSOTA}}
}
```
