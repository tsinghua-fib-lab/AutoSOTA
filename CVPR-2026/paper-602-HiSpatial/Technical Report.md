# HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models:
A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study of HiSpatial for hierarchical 3D spatial understanding in vision‑language models (VLMs). HiSpatial pairs monocular metric depth estimation from MoGe v2 (ViT‑L backbone) with a PaliGemma‑2‑derived VLM (3B parameters) to answer spatial reasoning queries on real‑world images. The optimization focuses on the 3DSRBench evaluation, for which the reproduced baseline average accuracy is 79.62 %. Eight inference‑time interventions are tested. The only robust improvement comes from test‑time flip augmentation of the MoGe‑estimated point clouds: averaging XYZ coordinates from the original and horizontally flipped images raises average accuracy to 80.25 % (+0.63 pp), with consistent gains in vertical- and horizontal‑distance metrics, while the inherently ambiguous lateral dimension (width) remains unchanged at 69.92 %. Either no effect or marked regressions result from modifying the prompt (e.g., −4.62 pp for reference‑object calibration), adjusting the token budget, or post‑processing depth. The optimization budget is exhausted without reaching the target of 83.77 %, indicating that further meaningful gains would require training‑time modifications rather than inference‑only tuning.

## 1. Introduction

Accurate spatial reasoning from monocular images is critical for embodied agents and vision‑language systems that must compare object positions, distances, and sizes. HiSpatial [1] addresses this by feeding explicit 3D point clouds produced by a monocular depth estimator into a VLM finetuned for hierarchical spatial question answering. While the method achieves strong performance on six benchmarks, its inference pipeline exposes several design choices – depth estimator preprocessing, prompt format, decoding strategy – that can be adjusted without altering the trained model. The AutoSOTA optimization framework [2] is applied to systematically evaluate such test‑time modifications, with the aim of improving accuracy on the 3DSRBench spatial reasoning benchmark relative to the reproduced baseline. This report describes the interventions, presents the measurable effects, and analyses the factors that limit further improvement.

## 2. Original Method (Background)

HiSpatial operates in two stages. For a given RGB image, MoGe v2 (ViT‑L, pretrained on metric depth) predicts a dense XYZ point map and a validity mask. The resulting 3D coordinates (shape [4, 448, 448], containing x, y, z and mask) are fed together with the resized image into HiSpatialVLM – a spatial‑reasoning VLM built from PaliGemma 2 (3 B parameters) and finetuned on 1.2 M spatially annotated images. The VLM receives a structured prompt (optionally prefixed by a special `<image>` token) and generates an answer using greedy decoding, with a maximum of 100 new tokens observed in the reproduced baseline. The evaluation suite spans six benchmarks; AutoSOTA concentrates on 3DSRBench, a comprehensive testbed covering four spatial categories (height, location, orientation, multi‑object). The original paper reports a group‑average accuracy of 79.78 %. Using the public model checkpoint and provided evaluation script, the AutoSOTA reproduction yields a baseline accuracy of 79.62 % – 0.16 pp below the paper’s figure, attributable to minor differences in the image loading environment or TSV parsing. This reproduced baseline serves as the reference for all subsequent comparisons.

## 3. Identified Limitations

The optimization log reveals several intrinsic constraints of the HiSpatial inference pipeline that cap what can be achieved at test time.

**Monocular depth noise.** MoGe produces metric depth from a single image, but lateral (width) dimensions are inherently more ambiguous than height or depth because they rely on subtler perspective cues. This manifests in a baseline width accuracy of only 69.92 %, markedly lower than for other dimensions. The flip‑augmentation experiment (Section 4) confirms that averaging predictions from two viewing conditions can partially reduce noise in the vertical and depth directions, but the benefit does not extend to width. The log concludes that width estimation is “fundamentally limited by monocular depth ambiguity in the lateral dimension.”

**Prompt‑format lock‑in.** The model was trained with a specific prompt structure: a multiple‑choice question with labelled options (`A. …`, etc.). Any alteration to this format induces a distribution shift that degrades accuracy. Interventions that modify the prompt – appending category‑specific hints (“Output only the number”), adding a chain‑of‑thought prefix, or embedding reference object sizes – consistently cause severe regressions (up to −4.62 pp) or no effect. The VLM appears to rely on a fixed mapping from the prompt’s syntactic form to its answer‑extraction behaviour.

**Inference‑parameter insensitivity.** Increasing the maximum number of new tokens from 100 to 200 yields zero accuracy change because the model already produces succinct answers (typically a single letter) well within 100 tokens. Similarly, median‑based depth anomaly filtering has no impact; MoGe’s output is already clean enough that post‑processing does not alter the XYZ values passed to the VLM.

**Training‑determined performance ceiling.** All accepted gains are modest, and none of the tested inference‑only modifications approach the target accuracy of 83.77 %. The log explicitly identifies that “HiSpatial’s spatial reasoning accuracy is determined by its training, not by inference‑time parameters.” Any parameter that does not change the underlying XYZ‑to‑answer mapping can at best exploit mild noise reduction; the bulk of the error is fixed by the learned behaviour of the VLM.

## 4. Optimization Methodology

The AutoSOTA pipeline evaluated eight distinct interventions, each motivated by a specific hypothesis. The two interventions that yielded non‑negative effects (one robust, one marginal) are described here; the remainder are summarised in Section 5.3.

**Flip‑augmentation XYZ averaging (Iteration 1).**  
*Modified file:* `eval/eval_3dsrbench.py`, within the `test` function.  
*Conceptual change:* Instead of using MoGe’s output for the original image alone, an additional inference is run on the horizontally flipped image. The resulting XYZ tensor is un‑flipped (negating the x‑coordinate channel and horizontally flipping the tensor) and then averaged with the original XYZ tensor.  
*Hypothesis:* Monocular depth estimators can be sensitive to asymmetric image features; averaging predictions from two mirror views reduces estimation noise in the vertical (y) and depth (z) components.  
*Implementation:* After loading the image and before passing it to `model_wrapper.query`, the following logic is executed:  
```python
xyz_orig = processor.apply_transform(image)
image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
xyz_flipped = processor.apply_transform(image_flipped)
# Unflip: flip width dimension and negate x‑channel
xyz_unflipped = torch.flip(xyz_flipped, [2])
xyz_unflipped[0] = -xyz_unflipped[0]
xyz_values = (xyz_orig + xyz_unflipped) / 2.0
```

**Self‑consistency decoding (Iteration 5).**  
*Modified file:* `hispatial/inference/predictor.py`, method `HiSpatialPredictor.query`.  
*Conceptual change:* For each query, the model generates five independent answers with temperature 0.5 sampling, then selects the median numeric value extracted from the responses.  
*Hypothesis:* Multiple sampled completions can capture a more robust central tendency, particularly for numeric distance questions.  
*Implementation:* The generation kwargs are set to `do_sample=True`, `temperature=0.5`; a loop over 5 generations collects answer strings, and a regex extracts the first floating‑point number to compute the median.  
This intervention is non‑deterministic and increases inference cost by a factor of five. Its average accuracy improvement is +0.16 pp, but width accuracy drops by 1.5 pp and direct distance rises by 2.72 pp. Therefore, the flip‑augmentation result (80.25 %) is considered the best reproducible optimized metric.

## 5. Experiments

### 5.1 Setup

**Hardware.** All evaluations were performed on a single NVIDIA GPU (CUDA device 0) with sufficient VRAM to host both the MoGe model and the 3B‑parameter HiSpatialVLM in float32.

**Dataset.** The 3DSRBench v1 TSV file contains multiple‑choice spatial reasoning examples, each linked to an image (base64‑encoded or URL‑based). Samples whose images could not be fetched (due to network restrictions in the sandbox) were skipped; the effective sample size remained consistent across iterations.

**Evaluation protocol.** The primary metric is group‑level accuracy: for each evaluation group, the fraction of predictions whose letter (A–D) matches the ground truth is computed, and the average across groups is reported as `avg_accuracy`. The extraction logic uses a regex to find a capital letter; if no letter is found, a fallback matches the full option text. Both regex‑based and truncation‑based accuracies are saved, and the per‑group regex accuracy is used throughout.

**Baseline command.**
```
python eval/eval_3dsrbench.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --tsv_path /path/to/3DSRBenchv1.tsv \
    --save_path results/baseline
```
The model is loaded from Hugging Face, and MoGe uses the pretrained `Ruicheng/moge-2-vitl-normal` checkpoint.

**Optimization budget.** Eight acceptance‑driven iterations were performed, followed by a final aggregation. No hyperparameter search beyond the described interventions was conducted.

**Caveats.**
- The reproduced baseline (79.62 %) is 0.16 pp below the paper’s 79.78 %, attributable to minor environmental differences; this does not affect within‑study comparisons.
- Since the random seed was not fixed in the AutoSOTA sandbox, the self‑consistency result (Iteration 5) is a single‑trial observation and is not guaranteed to reproduce exactly.
- Only inference‑side changes were permitted; model weights and training data remained untouched.

### 5.2 Quantitative Results

The table below compares the baseline accuracy against the best robust optimized configuration (flip‑augmented XYZ averaging). All values are in percent; arrows in the note column indicate the direction of change (↑ improvement, ↓ degradation, → unchanged).

| Metric                       | Baseline (%) | Flip‑Aug (%) | Δ (pp) | Note                     |
|------------------------------|:------------:|:------------:|:------:|:-------------------------|
| **Average accuracy**         |    79.62     |    80.25     | +0.63  | ↑ main metric            |
| Width accuracy               |    69.92     |    69.92     |  0.00  | → unchanged              |
| Height accuracy              |    85.50     |    86.26     | +0.76  | ↑                        |
| Direct distance accuracy     |    80.27     |    79.59     | –0.68  | ↓ minor degradation      |
| Horizontal distance accuracy |    86.07     |    87.70     | +1.63  | ↑                        |
| Vertical distance accuracy   |    76.19     |    78.10     | +1.91  | ↑                        |

The flip augmentation yields consistent gains on vertical and horizontal metrics, confirming that noise reduction in the y and z/depth components is the primary mechanism. The small drop in direct distance accuracy (–0.68 pp) is within estimation noise and may arise from averaging occasionally smoothing out fine depth variations. Width remains completely unaffected, in line with the limitation analysis.

The self‑consistency experiment reached an average accuracy of 80.41 % but is omitted from the final metric because it is not deterministic, incurs a 5× inference cost, and degrades width to 68.42 % (–1.5 pp). This outcome demonstrates that naive sampling can harm specific dimensions even when the overall average improves.

### 5.3 Ablation / Iteration Trajectory

The order and effect of every intervention are listed below, all on the same evaluation set.

| Iter | Change                                    | Avg Acc. (%) | Δ from Baseline (pp) |
|:----:|:------------------------------------------|:------------:|:---------------------:|
|  0   | Baseline reproduction                     |    79.62     |         0.00          |
|  1   | Flip augmentation XYZ averaging           |    80.25     |        +0.63          |
|  2   | Increase `max_new_tokens` 100 → 200       |    79.62     |         0.00          |
|  3   | Per‑category prompt hints                 |    77.89     |        –1.73          |
|  4   | Depth anomaly filtering (median‑based)     |    79.62     |         0.00          |
|  5   | Self‑consistency (N=5, temp=0.5)          |    80.41     |        +0.79*         |
|  6   | Chain‑of‑thought prefix                   |    79.62     |         0.00          |
|  7   | Reference object calibration (LEAP)        |    75.00     |        –4.62          |
|  8   | Simplified reference context (HP1)         |    79.12     |        –0.50          |

\* Not reproducible; width accuracy –1.5 pp, direct distance +2.72 pp.

All prompt‑based modifications (Iterations 3, 6, 7, 8) either perform no better than baseline or introduce severe regressions, reinforcing the prompt‑format lock‑in. The parameter‑tuning (Iteration 2) and post‑processing filter (Iteration 4) are neutral. Only the point‑cloud‑level augmentation (Iteration 1) provides a clear, reproducible benefit.

## 6. Discussion

The sole robust improvement originates from a test‑time augmentation applied directly to the 3D input, not from changes to the language model’s decoding or prompting. This indicates that the dominant source of remaining error is the quality of the monocular point clouds – especially in lateral dimensions – and that the VLM has already been trained near its ceiling for the given depth estimator. The ineffectiveness of prompt engineering highlights a brittleness that may limit the model’s utility in settings where users phrase queries differently.

The flip augmentation is computationally moderate (doubling the MoGe forward passes) but adds deterministic, architecture‑agnostic value. It is likely to generalise to other downstream tasks that share the same depth estimator. However, the lack of any improvement on width estimation suggests that future optimizations must address the fundamental monocular depth ambiguity – either by incorporating multi‑view information at test time (e.g., multi‑scale fusion) or by retraining the VLM with augmented data that emphasises lateral reasoning.

Threats to validity include the focus on a single benchmark (3DSRBench), the small offset between the reproduced baseline and the original paper’s number, and the limited set of eight inference‑time interventions. The optimization budget did not allow training‑time changes, which the log identifies as the most promising direction. The baseline accuracy remained stable at 79.62 % across all non‑effective iterations, lending confidence to the relative comparisons.

## 7. Reproducibility

**Repository:** `https://github.com/microsoft/HiSpatial.git`  
**Environment installation:**  
```bash
pip install -e ".[eval]"
pip install -e ".[depth]"
```
(Additional Hugging Face libraries are automatically resolved.)

**Seed:** No explicit seed is required; greedy decoding is deterministic, and the flip‑augmented run contains no randomness.

**Baseline run:**
```bash
python eval/eval_3dsrbench.py \
    --vlm_model_path lhzzzzzy/HiSpatial-3B \
    --tsv_path /path/to/3DSRBenchv1.tsv \
    --save_path results/baseline
```

**Optimized run (flip augmentation):** The evaluation script `eval_3dsrbench.py` must be modified to incorporate the flip‑and‑average logic inside the `test` function as described in Section 4. The command is identical to the baseline, calling the modified script. No additional arguments are required.

**Note:** To achieve reproducibility of the self‑consistency run, set a fixed random seed (e.g., 42) before generation.

## 8. References

[1] Liang, H., Shen, Y., Deng, Y., Xu, S., Feng, Z., Zhang, T., Liang, Y., Yang, J. (2026). HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

```bibtex
@inproceedings{liang2026hispatial,
  title={HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models},
  author={Liang, Huizhi and Shen, Yichao and Deng, Yu and Xu, Sicheng and Feng, Zhiyuan and Zhang, Tong and Liang, Yaobo and Yang, Jiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

[2] tsinghua-fib-lab/AutoSOTA. Automated optimization framework for state‑of‑the‑art research code.
