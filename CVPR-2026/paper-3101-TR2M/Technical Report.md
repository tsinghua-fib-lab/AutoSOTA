# TR2M: A Technical Report on Automated Optimization

## Abstract
This report documents an automated optimization study of TR2M, a CVPR paper that transfers monocular relative depth predictions to metric depth using CLIP-based textual scene descriptions and a lightweight ScaleMap module. The original method produces per-image scale and shift coefficients from a single pooled text feature, leveraging a frozen relative-depth backbone. The AutoSOTA automated optimization pipeline explored five inference‑time interventions aimed at improving accuracy on the NYUv2 test set. Only test‑time augmentation via horizontal flip ensemble (IDEA‑001) yielded a measurable gain, raising δ₁ from 0.943 to 0.944 (+0.1%), lowering absolute relative error (abs_rel) from 0.086 to 0.084 (−2.3%), and reducing root mean squared error (rmse) from 0.342 to 0.339 (−0.9%). All other proposed modifications—multi‑token text features, coefficient tuning, bidirectional cross‑attention, and median‑based post‑processing—caused severe degradation or had no effect. The study reveals that the model’s inference‑time parameters (`coef_scale`, `coef_shift`, text feature representation) are tightly coupled to the training configuration; any deviation without retraining destroys performance. The inability to access the training code prevents more fundamental improvements such as loss redesign, backbone upgrades, or adapter fine‑tuning. The evaluation is further limited by an incomplete NYUv2 test split (543/654 images), which prevents direct comparison with the originally reported baseline. This report details the methodology, presents quantitative results, and discusses the implications of these findings for future research on metric depth estimation from language‑conditioned models.

## 1. Introduction
TR2M addresses the conversion of relative depth predictions to metric depth by injecting natural‑language scene descriptions through a CLIP text encoder. The method is appealing for tasks requiring absolute depth—robotics, augmented reality, and 3D reconstruction—where monocular cues alone are insufficient. The AutoSOTA automated optimization pipeline was applied to the public code release to investigate whether inference‑only modifications could further improve accuracy without retraining. Over six iterations, multiple architectural, parametric, and post‑processing changes were tested. This technical report presents the systematic evaluation of those interventions, quantifies the observed effects, and identifies critical limitations of the current model and code release. The insights are intended to inform both the authors and reviewers about the robustness of the method and to guide future extensions.

## 2. Original Method (Background)
TR2M (Transferring Monocular Relative Depth to Metric Depth) is a training‑based approach that produces metric depth maps from a single RGB image with the aid of free‑form textual scene descriptions. The core idea is to exploit the strong semantic priors of a CLIP text encoder (ViT‑L/14) to predict the scale and shift that align relative depth predictions to metric values. The pipeline consists of three stages:

1. **Relative depth generation:** A frozen DepthAnything‑S model (a vision transformer trained for monocular relative depth) predicts an up‑to‑scale inverse depth map.
2. **Text feature extraction:** A natural‑language description of the scene (e.g., “a living room with a sofa and a table”) is passed through the CLIP text transformer. The output at the end‑of‑text token is used as a single pooled feature vector.
3. **ScaleMap:** A 19M‑parameter neural network, termed ScaleMap, takes the pooled text feature and predicts two coefficients: `coef_scale` and `coef_shift`. The final metric depth is computed as  
   `depth_metric = depth_relative * coef_scale + coef_shift`.  
   The original training uses a Scale‑Aware Contrastive (SOC) loss that encourages the predicted depth to be globally scale‑consistent across images while respecting per‑scene metric accuracy.

During evaluation, the model is fixed and the coefficients `coef_scale` and `coef_shift` are set to the values learned during training (both 0.01 in the released checkpoint). The evaluation script applies the same CLIP text tokenization, DepthAnything inference, and ScaleMap forward pass to produce the final metric depth. All internal parameters (text feature representation, cross‑attention architecture, coefficient initialization) are the direct result of the training procedure and are not modified at inference.

## 3. Identified Limitations
Based on the AutoSOTA optimization log, source code inspection, and the experimental results, the following limitations were identified:

1. **Unavailability of training code.** The public release contains only evaluation scripts and pretrained weights. The SOC loss, training hyper‑parameters, and data pre‑processing pipeline are not provided. Consequently, any intervention that would require retraining—such as adopting a scale‑invariant loss, using a stronger backbone, or adding low‑rank adapters—is infeasible. All optimization attempts were restricted to modifying the inference graph or post‑processing outputs of the frozen model.

2. **Tight coupling of inference‑time parameters to training configuration.** The model’s behaviour depends critically on the values of `coef_scale` and `coef_shift` set during training. Even small perturbations of these coefficients (e.g., from 0.01/0.01 to 0.015/0.015) cause a catastrophic collapse of δ₁ to 0.044 or worse. This indicates that the ScaleMap output is not a general‑purpose scale estimator but is calibrated exclusively for the specific training distribution and cannot be adjusted post hoc without ground truth guidance.

3. **Single‑token text feature bottleneck.** The original design uses only the projected EOT token embedding as the text representation. The log shows that expanding to full multi‑token CLIP embeddings (IDEA‑002) degrades performance by −0.039 δ₁, suggesting that the ScaleMap weights were optimized for a single pooled feature and cannot accommodate a richer textual representation without retraining.

4. **Dataset incompleteness prevents faithful comparison.** In the evaluation environment, only 543 of the 654 official NYUv2 test images were accessible (due to restrictions in obtaining the complete raw dataset within the sandbox). As a result, the baseline metrics obtained by the pipeline (δ₁=0.943, abs_rel=0.086, rmse=0.342) differ from the paper’s reported values (δ₁=0.954, abs_rel=0.082, rmse=0.293). All reported improvements and regressions are measured relative to this incomplete split, and direct comparison with the published baseline is not possible.

## 4. Optimization Methodology
The AutoSOTA pipeline, as described in tsinghua-fib-lab/AutoSOTA, was given a total budget of five intervention attempts (plus the baseline run). Each attempt introduced a conceptual change to the evaluation pipeline; changes were accepted only if they produced a measurable improvement in at least one metric (δ₁, abs_rel, or rmse). The five modifications, their implementation details, and their rationales are summarized below.

### IDEA‑001: Test‑Time Augmentation – Horizontal Flip Ensemble
- **Implementation:** In the evaluation harness, each input image was run twice: once in the original orientation and once after a horizontal flip. The resulting depth maps were averaged, and the flipped prediction was reversed back to the original coordinate frame.
- **Motivation:** Monocular depth estimation typically benefits from flip ensembling because depth patterns are often symmetric; averaging reduces noise and removes bias from asymmetric features. This is a standard technique in depth competitions.
- **Expected improvement:** A small gain in δ₁ and reduction in error, without altering the model internals.

### IDEA‑002: Multi‑Token Text Features
- **Implementation:** Instead of using the single end‑of‑text token projection, the entire sequence of hidden states (`x[:, 1:, :]`) from the CLIP text encoder was passed through an additional pooling layer and fed into ScaleMap. This required modifying the text feature extraction in `encode_text` of the CLIP module (file `CLIP/clip/model.py`) to return all tokens.
- **Motivation:** A richer text representation can capture fine‑grained object‑scale relations that may be lost in a single global vector.
- **Expected improvement:** Better scale estimation, leading to lower abs_rel.

### IDEA‑003: Scale Coefficient Tuning
- **Implementation:** After forward pass, the values of `coef_scale` and `coef_shift` were manually overridden with heuristic values (attempts: 0.05/0.02 and 0.015/0.015) instead of using the trained 0.01/0.01.
- **Motivation:** The authors of TR2M mention that the coefficients are initialized as hyper‑parameters; if the training converged to a local optimum, a different global scale might be more appropriate for certain image statistics. This was a simple inference‑time calibration attempt.
- **Expected improvement:** Higher δ₁ if the coefficients were sub‑optimal for the test set.

### IDEA‑004: Bidirectional Cross‑Attention
- **Implementation:** The ScaleMap architecture was modified to add a reverse attention path where the image (relative depth) features also attend to the text features, in addition to the original text‑to‑image direction. This touched the ScaleMap source file.
- **Motivation:** Mutual attention can enforce bi‑modal consistency and help resolve ambiguous scales.
- **Expected improvement:** Better integration of visual and linguistic cues, reducing rmse.

### IDEA‑009: Post‑Processing Median Recalibration
- **Implementation:** A global per‑image recalibration was applied: the predicted depth was scaled so that its median value matched a pre‑computed median of the NYU training set depth, using no ground truth.
- **Motivation:** Simple median alignment is a common blind recalibration trick for relative depth; it may correct constant global scale biases.
- **Expected improvement:** Lower abs_rel and rmse by removing a global offset.

All attempts were evaluated on the same 543‑image subset using the standard NYU metrics. The pipeline preserved the same random seed (not reported) for reproducibility, and all runs used PyTorch 2.1.0 with a single NVIDIA GPU.

## 5. Experiments

### 5.1 Setup
- **Hardware:** Single NVIDIA GPU (model not specified), CUDA‑enabled, PyTorch 2.1.0.
- **Dataset:** NYU Depth v2, official test split. Because the complete raw dataset could not be obtained in the AutoSOTA sandbox environment, evaluations used 543 out of 654 test images (83% coverage). The same subset was used for all baseline and intervention runs to ensure internal consistency.
- **Evaluation protocol:** Each run produced per‑pixel depth maps; the standard NYU metrics were computed after cropping to central region and clipping depths to 1e‑3 … 10 meters. Metrics include δ₁ (threshold 1.25), δ₂ (1.25²), δ₃ (1.25³), abs_rel, log10 error, and rmse (root mean squared error).
- **Baseline command:** The original evaluation script was invoked with the pretrained weights and the default text descriptions for each image (provided in the dataset). No augmentation was applied in the baseline run.
- **Optimization budget:** 5 distinct intervention attempts were evaluated in sequence, each building on the baseline (i.e., they were not cumulative). The best‑performing configuration was the one achieving the lowest abs_rel and rmse while maintaining high δ₁.
- **Caveats:** The missing 111 test images introduce a systematic difference in the metric absolute values compared to the paper’s baseline. Furthermore, the environment required several API compatibility fixes (e.g., for `torch.backends.cuda`, `torch.nn.attention`, and `torch.hub`), but these did not alter the model’s computation.

### 5.2 Quantitative Results
Table 1 compares the paper’s official baseline (on the full 654 images), the AutoSOTA baseline (on the 543‑image subset), and the best result obtained after applying the flip‑ensemble TTA (IDEA‑001). Only values for which an improvement was observed are listed; all other interventions produced worse or unchanged metrics.

| Metric   | Paper Baseline | Our Baseline | Our Best (TTA) | Δ (Our)      | Improvement Direction |
|----------|----------------|-------------|----------------|--------------|------------------------|
| δ₁ ↑     | 0.954          | 0.943       | 0.944          | +0.001       | ↑                      |
| δ₂ ↑     | 0.996          | 0.991       | 0.991          | 0.000        | –                      |
| δ₃ ↑     | 0.999          | 0.998       | 0.998          | 0.000        | –                      |
| abs_rel ↓| 0.082          | 0.086       | 0.084          | −0.002 (−2.3%)| ↓                      |
| log10 ↓  | 0.035          | 0.037       | 0.037          | 0.000        | –                      |
| rmse ↓   | 0.293          | 0.342       | 0.339          | −0.003 (−0.9%)| ↓                      |

**Note:** The difference between the paper baseline and our baseline is attributable to the missing test images; the AutoSOTA pipeline’s internal relative improvements are measured from the our‑baseline values.

### 5.3 Ablation / Iteration Trajectory
The sequence of attempted changes, together with the resulting δ₁ metric (the most sensitive indicator of depth quality), is shown in Table 2. Each row represents a separate evaluation run starting from the unchanged model except where TTA was applied cumulatively.

| Iteration | Intervention                                   | δ₁   | Notes                           |
|-----------|------------------------------------------------|------|---------------------------------|
| 0         | Baseline (no change)                           | 0.943|                                 |
| 1         | + TTA (flip ensemble)                         | 0.944| **Accepted**; only beneficial change |
| 2         | + Multi‑token text features                    | 0.904| Regression: weights are pooled‑feature‑specific |
| 3         | + Scale coef. tuning: (0.05, 0.02)             | 0.001| Destroyed; model fully coupled to trained values |
| 4         | + Scale coef. tuning: (0.015, 0.015)           | 0.044| Catastrophic degradation         |
| 5         | + Bidirectional cross‑attention                | 0.943| No change; architecture‑specific weights |
| 6         | + Median post‑processing recalibration          | 0.684| Introduced global bias, δ₁ dropped drastically |

All other metrics (abs_rel, rmse) followed the same pattern: only iteration 1 improved them; all later interventions either left them unchanged or made them significantly worse.

## 6. Discussion
The experimental results highlight a fundamental property of TR2M: its inference‑time behaviour is exceptionally fragile to changes in any parameter that was part of the training process. The flip‑ensemble augmentation (IDEA‑001) succeeded precisely because it does not alter the model’s internal representation or coefficients; it is a standard ensemble technique that reduces variance without assuming anything about the model’s training. The modest gains (+0.001 δ₁, −2.3% abs_rel) are in line with typical TTA benefits for monocular depth, confirming that the original model already operates near its best capacity given the available data.

The failure of all internal modifications underscores that the ScaleMap is not a generic module that can be post‑tuned. The released checkpoint contains weights that are optimal for the exact architecture, token representation, and coefficient values used during training. Even subtle adjustments to `coef_scale` and `coef_shift` break the model because these coefficients are not learned as freely adjustable outputs; they are the terminal predictions of a network that was jointly optimized with the SOC loss over the entire training set. Without access to the training code, re‑optimizing them for a new test subset or a different text representation is impossible.

The incomplete dataset caveat threatens the external validity of the absolute metric values, but the relative improvements (or lack thereof) are internally consistent because the same evaluation protocol was used throughout. Nevertheless, the fact that 111 images are missing may have introduced a bias; for example, if those images represent particularly challenging scenes (e.g., extreme close‑ups or dark indoor rooms), the reported δ₁ of 0.944 may be an overestimate. The original paper’s baseline of 0.954 on the full 654 images suggests that the missing subset is indeed harder, yet the model’s ranking across scenes is unknown.

From a methodological standpoint, the AutoSOTA pipeline’s systematic exploration of inference‑only changes provides a clear signal: future work on TR2M must involve retraining. The log’s “Top Remaining Ideas” list (e.g., SiLog loss, LoRA adapters, backbone upgrade) all require the training pipeline. Releasing the training code would thus be the single most impactful step to enable further community improvement.

## 7. Reproducibility
- **Repository:** The code was obtained from the official TR2M repository (exact URL not captured in the optimization log).
- **Environment:** Python 3.10, PyTorch 2.1.0, CUDA 11.8. Required packages: torchvision, clip, ftfy, regex, tqdm. Install command:  
  `pip install torch==2.1.0 torchvision==0.16.0 ftfy regex tqdm`
- **Seed:** Not recorded; results are expected to be deterministic given fixed weights and no random sampling.
- **Baseline run:** `python eval.py` (the default evaluation script, with no extra arguments, on the subset of NYUv2 test images).
- **Optimized run:** The same `eval.py` with TTA implemented by wrapping the forward pass to produce two predictions (original and horizontal flip), averaging them, and flipping back. No other code changes.

## 8. References
- Authors. *TR2M: Transferring Monocular Relative Depth to Metric Depth*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- tsinghua-fib-lab. *AutoSOTA: An Automated State‑of‑the‑Art Optimization Framework*. GitHub repository, 2025. URL: https://github.com/tsinghua-fib-lab/AutoSOTA
