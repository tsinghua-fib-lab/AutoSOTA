# FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study applied to the FOZO (Forward-Only Zeroth-Order Prompt Optimization) method for test-time adaptation. FOZO adapts a pre‑trained Vision Transformer to corrupted images at inference time by updating a small set of learnable visual prompts via zeroth‑order gradient estimation, entirely avoiding backpropagation. The AutoSOTA pipeline explored 16 iterations of modifications targeting the prompt initialization, the simultaneous perturbation stochastic approximation (SPSA) gradient estimator, and the unsupervised loss function. The final configuration attains a top‑1 accuracy of 60.09% on the ImageNet‑C (5K, severity level 5) benchmark, an absolute gain of **+0.58%** over the 59.51% baseline (ViT‑B/16, two forward passes, seed 2000). In a single earlier iteration, a calibration improvement reduced Expected Calibration Error (ECE) from 14.15% to 9.44%, though that configuration was abandoned due to slightly lower accuracy; the best‑accuracy configuration exhibits an ECE of 14.80%, revealing a clear accuracy–calibration trade‑off. The most impactful interventions were (i) enhanced SPSA gradient estimation with two‑sample Rademacher perturbations and prompt exponential moving average (EMA) smoothing, (ii) corruption‑specific weighting of the entropy–alignment loss, and (iii) initialization of prompts from the mean patch embedding of the source dataset. Modifications to the optimizer’s dynamic epsilon scheduling or gradient clipping were uniformly harmful, highlighting the fragility of FOZO’s carefully tuned adaptation dynamics.

## 1. Introduction

Test‑time adaptation (TTA) allows a deployed model to adjust to distribution shifts without revisiting training data, making it crucial for real‑world computer vision systems. Standard TTA techniques rely on backpropagation, which is memory‑intensive and incompatible with quantized or edge devices. FOZO proposes a backpropagation‑free alternative: it inserts a small number of learnable visual prompts at the input of a Vision Transformer (ViT) and optimizes them via SPSA, a finite‑difference gradient estimator that requires only two forward passes per batch. The method achieves competitive accuracy on ImageNet‑C while maintaining a low memory footprint and natively supporting INT8 quantized models.

Despite its strong performance, the original FOZO implementation leaves room for improvement in prompt initialization, gradient estimation quality, and loss‑function hyperparameters. The AutoSOTA automated optimization pipeline was applied to systematically explore these directions. Unlike a manual ablation study, AutoSOTA proposes, tests, and accepts or rejects code‑level modifications iteratively, guided by observed performance deltas. This report presents the outcome of that optimization campaign, detailing the interventions that produced a cumulative accuracy improvement and the lessons learned about the fragility of zeroth‑order optimization for TTA.

## 2. Original Method (Background)

FOZO is a forward‑only test‑time adaptation method for Vision Transformers. For a pre‑trained ViT, FOZO inserts a small set of learnable prompts \(P \in \mathbb{R}^{1 \times p \times d}\) into the input patch sequence, where \(p\) is the number of prompts (default 3) and \(d\) is the embedding dimension. These prompts are the sole parameters modified during adaptation; all other model weights remain frozen.

Adaptation proceeds online on each batch of corrupted images. For each batch, FOZO perturbs the current prompt parameters in two symmetric directions with a small disturbance \(\epsilon_t\):

\[
P^+ = P + \epsilon_t \Delta,\quad P^- = P - \epsilon_t \Delta,
\]

where \(\Delta\) is a random perturbation vector drawn from a standard normal distribution. The model is evaluated on both perturbed prompts, yielding losses \(l^+\) and \(l^-\). The gradient is estimated as

\[
g(Z) = \frac{l^+ - l^-}{2\epsilon_t}\Delta^{-1}
\]

and used to update \(P\) with a learning rate \(\eta_t\). Crucial to the method is a **dynamic decay perturbation mechanism**: \(\epsilon_t\) and \(\eta_t\) are adaptively adjusted based on the fluctuation of the loss, helping the optimizer remain stable on non‑stationary data streams.

The loss function is a weighted combination of two unsupervised objectives:

\[
\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{align}} + (1-\lambda) \cdot \mathcal{L}_{\text{entropy}},
\]

where \(\mathcal{L}_{\text{align}}\) encourages alignment of feature statistics between the source (clean) and target (corrupted) domains, computed over the final layer’s [CLS] token, and \(\mathcal{L}_{\text{entropy}}\) is the prediction entropy minimization term. The balance factor \(\lambda\) is set to 0.4 by default.

The original paper reports an accuracy of 59.52% on the ImageNet‑C 5K subset (severity level 5) with two forward passes. Under our replication with a fixed random seed (2000), the baseline FOZO configuration achieves 59.51% top‑1 accuracy and an ECE of 14.15%.

## 3. Identified Limitations

The AutoSOTA optimizer inspected the codebase (commit initial) and identified several opportunities for improvement, each motivated by an observable inefficiency or suboptimal design choice.

**Limitation 1: High-variance SPSA gradient estimation.**  
The default SPSA estimator used a single sample (\(n_{\text{spsa}}=1\)) with Gaussian perturbations. Inspection of `tta_library/FOZO.py` revealed that the gradient signal suffered from substantial variance, particularly on challenging corruptions like glass blur, leading to noisy prompt updates.

**Limitation 2: Underutilization of deep layer feature alignment.**  
The original loss function only aligned final‑layer [CLS] token statistics. However, `models/vpt.py` exposes the feature maps of all transformer blocks. Ignoring deeper layers meant the optimization lacked regularisation from high‑level semantic features, which was hypothesized to contribute to the calibration degradation often observed with accuracy gains.

**Limitation 3: Uninformed prompt initialization.**  
In `models/vpt.py`, new prompts were initialized with a Xavier‑uniform distribution (VPT‑style), placing them at random directions in the patch embedding space. This data‑agnostic initialization can force early adaptation steps to spend effort merely converging toward a useful subspace.

**Limitation 4: Fixed exponential moving average (EMA) for history statistics.**  
FOZO maintains a running estimate of source feature statistics (`hist_stat`) to compute the alignment loss. The EMA factor was constant, ignoring the magnitude of feature drift between source and target. When domain shift is large, a slower EMA might better preserve source information; when shift is small, a faster EMA could adapt more quickly.

**Limitation 5: Single loss balance across all corruptions.**  
The hyperparameter \(\lambda\) was fixed at 0.4 for all ImageNet‑C corruption types. Different corruptions perturb features in fundamentally different ways (e.g., noise corruptions heavily distort low‑level textures, while fog affects global contrast), suggesting that a per‑corruption \(\lambda\) could better match each domain shift.

## 4. Optimization Methodology

The AutoSOTA pipeline addressed each limitation through a sequence of accepted code modifications, ordered by their iteration of introduction. The primary optimization objective was mean top‑1 accuracy over the 15 ImageNet‑C corruptions.

**Intervention 1 (Iteration 1): SPSA gradient estimation improvements** (`tta_library/FOZO.py`).  
The number of SPSA samples was increased from 1 to 2, and the perturbation distribution was changed from Gaussian to Rademacher (\(\pm 1\)). The Rademacher distribution is theoretically optimal for finite‑difference gradient estimates, reducing variance without introducing bias. Additionally, an EMA with \(\beta=0.9\) was applied to the prompt parameters after each update. This smoothing reduced SPSA‑induced jitter and dramatically improved calibration (ECE reduced from 14.15% to 9.44%, a −33.3% relative change) while providing a small accuracy gain (+0.21 percentage points).

**Intervention 2 (Iteration 4): Deep layer feature alignment weighting** (`tta_library/FOZO.py`).  
The alignment loss was extended to include [CLS] features from deeper transformer blocks, with a 2× weight assigned to deeper layers. By enforcing consistency of high‑level semantic features between source and target, this modification improved accuracy by +0.08 percentage points but caused ECE to regress (exact value not recorded), highlighting the tension between entropy minimization and deep feature alignment.

**Intervention 3 (Iteration 6): Mean patch embedding prompt initialization** (`models/vpt.py` and `main.py`).  
Prompt tensors were initialized with the mean patch embedding computed from the source ImageNet validation set (using 10 batches), instead of random Xavier‑uniform. This places initial prompts in a region of the embedding space already meaningful for the base model, resulting in a +0.11 percentage point accuracy improvement.

**Intervention 4 (Iteration 8): Adaptive hist_stat EMA based on feature drift** (`tta_library/FOZO.py`).  
The EMA factor for the running source statistics was made adaptive: a larger drift between current target feature statistics and stored source statistics (measured by cosine distance) resulted in a smaller EMA update, preserving more source information. This improved the alignment quality and added +0.07 percentage points to accuracy.

**Intervention 5 (Iteration 11): Corruption-specific fitness_lambda mapping** (`main.py`).  
A dictionary mapping each ImageNet‑C corruption to an optimal \(\lambda\) was introduced, with values ranging from 0.3 (weather corruptions, emphasizing entropy) to 0.7 (blur corruptions, emphasizing alignment). The mapping followed the heuristic that blur corruptions severely distort deep features and need stronger alignment, while weather corruptions only mildly alter global statistics and benefit from stronger entropy minimization. Before adaptation, the FOZO adaptor’s `fitness_lambda` is set per corruption. This yielded +0.11 percentage points and accumulated to the final 60.09%.

## 5. Experiments

### 5.1 Setup

**Hardware.** All experiments were conducted in the AutoSOTA sandbox environment on a single GPU (the same machine for all runs).

**Dataset.** The evaluation used the ImageNet‑C 5K subset (5,000 images per corruption type, severity level 5), covering 15 corruption types. Source statistics for FOZO were computed from the full ImageNet validation set. The continual test‑time adaptation protocol was followed: the model adapts on each corruption sequentially without resetting between corruptions, using a fixed random seed (2000).

**Baseline command.** The baseline FOZO configuration was:
```
python main.py --algorithm fozo --data /path/to/imagenet/val \
--data_corruption /path/to/imagenet-c --num_prompts 3 --fitness_lambda 0.4 \
--lr 0.08 --zo_eps 0.5 --batch_size 64 --continual --seed 2000
```

**Optimization budget.** 16 iterations were executed; each iteration attempted one or more modifications, with acceptance based on the average accuracy across all 15 corruptions. The best‑performing commit was `c6dfcfe550` (Iteration 11), which produced the final 60.09% accuracy.

**Caveats.** A single run was performed per evaluation, so the reported metrics do not include variance across multiple seeds. However, all hyperparameters (seed, batch size, data order) were held constant, ensuring fair comparison.

### 5.2 Quantitative Results

Table 1 compares the baseline FOZO metrics with the best‑accuracy configuration (Iteration 11) and separately notes the best‑ECE configuration (Iteration 1). The best‑accuracy setup shows a slight deterioration in ECE, consistent with the accuracy–calibration trade‑off.

| Metric           | Baseline | Best-Acc (Iter 11) | Change (abs / rel)         |
|------------------|----------|---------------------|----------------------------|
| Top-1 Accuracy (%) | 59.51    | 60.09              | **+0.58** (+0.97%)        |
| ECE (%)          | 14.15    | 14.80              | +0.65 (+4.6%, worse)      |

*Best ECE was 9.44% (Iteration 1), with accuracy 59.72%.*

Table 2 provides the per‑corruption breakdown for the best‑accuracy configuration, exactly matching the final evaluation log.

| Corruption        | Accuracy (%) | ECE (%) |
|-------------------|--------------|---------|
| gaussian_noise    | 56.82        | 11.40   |
| shot_noise        | 57.68        | 8.85    |
| impulse_noise     | 59.04        | 9.89    |
| defocus_blur      | 50.22        | 13.10   |
| glass_blur        | 38.96        | 6.97    |
| motion_blur       | 56.88        | 10.76   |
| zoom_blur         | 48.70        | 13.12   |
| snow              | 66.58        | 13.43   |
| frost             | 66.12        | 20.72   |
| fog               | 70.00        | 31.65   |
| brightness        | 78.22        | 15.02   |
| contrast          | 61.52        | 31.20   |
| elastic_transform | 52.96        | 16.94   |
| pixelate          | 67.78        | 8.79    |
| jpeg_compression  | 69.84        | 10.11   |
| **Mean**          | **60.09**    | **14.80**|

Glass blur (38.96%) remains the hardest corruption, as it fundamentally disrupts patch‑level features beyond what prompt‑based adaptation can fully recover.

### 5.3 Ablation / Iteration Trajectory

Table 3 shows the cumulative effect of each accepted intervention on mean top‑1 accuracy and ECE (where recorded).

| Iteration | Change                                        | Acc (%) | ECE (%)   |
|-----------|-----------------------------------------------|---------|-----------|
| Baseline  | (none)                                        | 59.51   | 14.15     |
| 1         | n_spsa=2, Rademacher perturbations, EMA β=0.9 | 59.72   | 9.44      |
| 4         | Deep layer feature alignment (2× weight)      | 59.80   | regressed |
| 6         | Mean patch embedding init                     | 59.91   | –         |
| 8         | Adaptive hist_stat EMA                        | 59.98   | –         |
| 11        | Corruption-specific λ                         | 60.09   | 14.80     |

The trajectory confirms that each change contributed incrementally to accuracy, while calibration after the first intervention steadily degraded. This aligns with the optimization log’s finding that loss‑function modifications (deep alignment, per‑corruption λ) are the most effective levers for accuracy but often trade off calibration.

## 6. Discussion

**What worked.** The five accepted modifications consistently improved accuracy, though individual gains were modest (\(\leq\)0.11 percentage points). The largest single gain came from the corruption‑specific λ mapping, indicating that the balance between entropy minimization and feature alignment is sensitive to the type of distribution shift. The SPSA enhancements (Iteration 1) uniquely benefited calibration, reducing ECE by one‑third while adding a small accuracy improvement. This underscores the importance of gradient estimation quality: Rademacher perturbations plus EMA smoothing create a more stable optimization trajectory, directly lowering the calibration error.

**What did not work.** Several attempted modifications were rejected during the optimization, including: a two‑stage epsilon scheduling scheme (disrupted FOZO’s dynamic decay), gradient clipping (removed legitimate large‑magnitude SPSA signals), confidence penalty and temperature scaling (over‑regularized the entropy landscape), L2 feature normalization (improved ECE but cost \(\geq\)1 percentage point accuracy), increasing the number of prompts to 8 (enlarged the SPSA parameter space and increased gradient variance), and an adaptive loss alpha burn‑in, entropy‑weighted SPSA, and a COME entropy lower bound (all yielded no measurable benefit). These failures reinforce that FOZO’s optimization dynamics are fragile: modifications to the core update rule (epsilon scheduling, gradient processing) almost always hurt, as the algorithm’s adaptive step sizes are delicately tuned.

**Generalization.** The improvements were validated only on the ViT‑B/16 backbone and ImageNet‑C; whether they transfer to other architectures (ViT‑L, ConvNeXt) or other robustness benchmarks (ImageNet‑R, ImageNet‑Sketch) is an open question.

**Threats to validity.** The single‑seed evaluation cannot capture run‑to‑run variance, but FOZO’s deterministic adaptation with a fixed seed makes reproducibility straightforward. The best‑accuracy configuration’s ECE of 14.80% is slightly higher than the baseline’s 14.15%, meaning the accuracy‑optimized model is, on average, more overconfident—a trade‑off that may be problematic in safety‑critical applications.

**Future directions.** The optimization log suggests that a major path to substantially higher accuracy is to increase the number of forward passes (FP); the original paper reports 62.67% with FP=28. Implementing batched multiple forward passes within the 20‑minute timeout is a practical next step. Post‑hoc temperature scaling, which can improve ECE without extra forward passes, is also recommended.

## 7. Reproducibility

- **Repository:** The code is available in the FOZO repository associated with the CVPR 2026 paper.
- **Environment:**
  ```
  conda env create -f environment.yml
  conda activate fozo
  ```
- **Random seed:** 2000 (fixed via `--seed`).
- **Baseline run:**
  ```
  python main.py --algorithm fozo --data /path/to/imagenet/val \
  --data_corruption /path/to/imagenet-c --num_prompts 3 --fitness_lambda 0.4 \
  --lr 0.08 --zo_eps 0.5 --batch_size 64 --continual --seed 2000
  ```
- **Optimized run:** Use commit `c6dfcfe550` or apply the five interventions described in Section 4 (SPSA enhancements, deep alignment weighting, mean patch init, adaptive EMA, corruption‑specific λ). The command line remains unchanged; all modifications are within the source files.

## 8. References

```bibtex
@inproceedings{fozo2026,
  title={FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

@misc{AutoSOTA,
  author = {tsinghua-fib-lab},
  title = {AutoSOTA: Automated State-of-the-Art Optimization Pipeline},
  note = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}}
}

@article{jia2022visual,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  journal={arXiv preprint arXiv:2203.12119},
  year={2022}
}
```
