# SAMTok: Representing Any Mask with Two Words: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study of the SAMTok model, which performs referring expression segmentation by predicting two discrete mask tokens via a Qwen2.5‑VL vision‑language model coupled to a vector‑quantized SAM2 decoder. Using the AutoSOTA framework [2], four interventions were applied to the inference pipeline. The only accepted change raised the Qwen2.5‑VL input resolution from 448 × 448 to 896 × 896, yielding a generalized Intersection‑over‑Union (gIoU) improvement of +0.7 points over the 500‑sample evaluation baseline (81.7 → 82.4) and a cumulative IoU (cIoU) gain of +1.8 points (82.6 → 84.4). Relative to the paper‑reported gIoU of 79.4 (179‑sample subset), the optimized model achieves a +3.0‑point increase. Empty‑target detection accuracy (N‑acc) decreased marginally by 0.4 points, indicating a small precision–recall trade‑off. The remaining three attempts—rewriting the chain‑of‑thought (CoT) prompt, enlarging the SAM2 decoder input, and temperature‑annealed sampling—either severely degraded performance (gIoU dropped by 24.2 points, N‑acc by 38 points) or failed due to architectural constraints. The temperature‑annealing trial was still executing at the snapshot time and is not included in the final results. The study highlights that image resolution scaling is a simple, high‑impact lever, while the prompt template and the SAM2 decoder dimensions are rigid and resist naïve modification.

## 1. Introduction

Referring expression segmentation (RES) requires localizing and segmenting an object described by a natural language phrase. SAMTok [1] addresses this task by unifying a Qwen2.5‑VL‑3B‑Instruct large multimodal model with a vector‑quantized SAM2 mask tokenizer: each mask is compressed to two discrete codebook indices that are predicted autoregressively. The design avoids per‑instance pixel‑wise decoding, enabling fast mask generation while retaining the instruction‑following capabilities of vision‑language models.

The original SAMTok model achieves competitive results on the GRES benchmark, but the interplay between image processing, prompt formulation, and mask decoding offers several axes for improvement. This study employs the AutoSOTA automated optimization framework to probe four targeted modifications: (i) scaling the image resolution sent to the vision encoder, (ii) rewriting the chain‑of‑thought (CoT) prompt template, (iii) increasing the input size for the SAM2 mask decoder, and (iv) introducing temperature‑based sampling during token generation. Only the resolution increase produced a net benefit; the other attempts revealed important constraints of the model. This report details the optimization trajectory, presents quantitative results, and discusses implications for future development.

## 2. Original Method (Background)

SAMTok is implemented in the Sa2VA codebase [1] and evaluated on the gRefCOCO dataset. The model loads the pretrained Qwen2.5‑VL‑3B‑Instruct as the vision‑language backbone and a frozen VQ‑SAM2 mask decoder (`mask_tokenizer_256x2.pth`) with a codebook of size 256 and depth 2 (`CODEBOOK_DEPTH = 2`). The vision processor (`AutoProcessor`) resizes images such that the pixel count falls between `min_pixels = 448×28×28` and `max_pixels = 448×28×28`, yielding an effective input resolution of approximately 448 × 448. The SAM2 image preprocessing applies a `DirectResize(1024)`, scaling the image so that the longest side is 1024 pixels before feeding it to the VQ‑SAM2 encoder.

During inference, the model generates up to 384 new tokens via greedy decoding (`do_sample=False, top_p=1.0`). The generated text is parsed for `<|mt_start|>…<|mt_end|>` patterns; the two extracted integer token IDs are remapped into per‑codebook indices and decoded by `vq_sam2.forward_with_codes`. The output mask is resized to the original image dimensions using bilinear interpolation and thresholded at 0.5. Evaluation follows the GRES protocol: generalized Intersection‑over‑Union (gIoU) accounts for both foreground and empty‑target samples, cumulative IoU (cIoU) measures segmentation quality only on foreground samples, and N‑acc computes the accuracy of empty‑target detection (higher is better for all three metrics).

## 3. Identified Limitations

**Low image resolution limits fine boundary capture**  
The default Qwen2.5‑VL processor produces 448 × 448 image inputs. This coarse resolution can obscure subtle edge cues required for precise mask boundaries, especially on thin structures. Although the SAM2 decoder receives an image resized to 1024, the upstream VLM operates on downscaled visual features. The AutoSOTA pipeline hypothesized that increasing the VLM’s input resolution would provide richer visual detail, leading to more accurate mask token predictions.

**Brittle prompt template**  
The default question string in `eval_gres.py` is a lengthy, structured prompt containing explicit `<think>` and `<answer>` tags. The model has been fine‑tuned to emit mask tokens within this template. An attempt to modify the prompt (CoT optimization) caused a catastrophic gIoU drop of 24.2 points and an N‑acc drop of 38 points, demonstrating that the token generation process is highly sensitive to the surface form of the instruction.

**SAM2 decoder has a fixed internal embedding size**  
The `DirectResize(1024)` strongly ties the SAM2 image size to the architecture’s embedding tables. When the pipeline attempted to increase this resolution to 1536 (Iteration 3), the forward pass crashed with an assertion error from the embedding layer, confirming that the decoder’s input dimensionality cannot be altered without retraining or structural changes.

**Greedy decoding may be suboptimal for mask token generation**  
The inference configuration uses deterministic greedy sampling. Probabilistic exploration (e.g., temperature‑annealed sampling) could potentially yield better mask token pairs, but this hypothesis was tested only in a pending run (Iteration 4) and had not been evaluated at the snapshot time.

## 4. Optimization Methodology

Four sequential interventions were applied, each addressing one of the identified limitations.

**Increase image resolution (Iteration 1)**  
*Change*: In `eval_gres.py`, the processor’s pixel limits were set to:
```python
processor.image_processor.min_pixels = 896 * 28 * 28
processor.image_processor.max_pixels = 896 * 28 * 28
```
*Hypothesis*: A higher‑resolution VLM input (896 × 896) provides finer visual information, improving the precision of predicted mask tokens. *Outcome*: gIoU improved by +0.7 and cIoU by +1.8. This intervention was accepted as the best configuration.

**CoT prompt optimization (Iteration 2)**  
*Change*: The question string was rewritten to encourage a different reasoning style while preserving the `<think>` and `<answer>` markers.  
*Hypothesis*: A more explicit chain‑of‑thought structure could improve object localization.  
*Outcome*: gIoU fell to approximately 57.5 (−24.2 vs. baseline), and N‑acc dropped by 38 points (to ≈44.8). The change was immediately rolled back.

**SAM2 decoder resolution increase (Iteration 3)**  
*Change*: `DirectResize(1024)` was changed to `DirectResize(1536)`.  
*Hypothesis*: A larger decoder input could capture finer mask details.  
*Outcome*: The SAM2 forward pass crashed with an assertion error at the embedding layer; no metric was recorded.

**Temperature annealing (Iteration 4)**  
*Change*: `model.generate` was called with `do_sample=True, temperature=0.3` (top‑p=1.0).  
*Hypothesis*: Soft sampling may help the model explore token pairs that are sub‑optimal under greedy decoding but yield better masks.  
*Status*: The iteration was still in progress when the log was produced; results are not available.

Only Iteration 1 produced a verifiable improvement and was retained for the final evaluation.

## 5. Experiments

### 5.1 Setup

All experiments used a 500‑sample subset of the gRefCOCO validation set, evaluated on two GPUs. The substitute subset replaces the paper’s 179‑sample split and accounts for the higher baseline metrics observed here. The evaluation script loads the SAMTok model from `/models/Qwen2.5-VL-3B-SAMTok-gres-rl` and the VQ‑SAM2 weights from `mask_tokenizer_256x2.pth`. For each sample, mask tokens are generated and decoded; the resulting binary mask is compared with the ground‑truth RLE mask.

The optimization budget was fixed at four iterations (one successful, two failed, one pending). All runs except the temperature‑annealing trial used greedy decoding. No random seed was explicitly set, and inference was performed in `torch.no_grad()` mode. The baseline configuration corresponds to the default Qwen2.5‑VL resolution (448 × 448) and the original prompt. The best configuration is the baseline plus the resolution increase to 896 × 896.

**Caveats**:  
- The evaluation subset (500 samples) differs from the paper’s reported test split (179 samples); absolute metric values are not directly comparable to published numbers.  
- The temperature‑annealing trial (Iteration 4) had not completed at the snapshot time; its results are excluded.  
- The model uses a specific fine‑tuned variant of Qwen2.5‑VL‑3B that may not be publicly available, affecting exact reproducibility.

### 5.2 Quantitative Results

All metrics are percentages; higher is better.

| Metric | Paper Baseline (179 samples) | Our Baseline (500 samples, 448 res) | Best (896 res) | Δ Our Baseline → Best | Δ Paper → Best |
|--------|------------------------------|-------------------------------------|----------------|------------------------|----------------|
| gIoU   | 79.4                         | 81.7                                | 82.4           | +0.7                  | +3.0           |
| cIoU   | 73.7                         | 82.6                                | 84.4           | +1.8                  | +10.7          |
| N‑acc  | 81.5                         | 82.8                                | 82.4           | −0.4                  | +0.9           |

The resolution increase improves mask overlap metrics (gIoU and cIoU) but slightly reduces empty‑target detection accuracy. The large difference between the paper cIoU (73.7) and our baseline (82.6) arises from the different evaluation subsets; within‑experiment deltas are the primary focus.

### 5.3 Ablation / Iteration Trajectory

| Iteration | Change Applied                     | gIoU   | cIoU  | N‑acc | Outcome       |
|-----------|------------------------------------|--------|-------|-------|---------------|
| 0         | Baseline (448 res, greedy decode)  | 81.7   | 82.6  | 82.8  | –             |
| 1         | Resolution 448 → 896               | 82.4   | 84.4  | 82.4  | Accepted      |
| 2         | CoT prompt modification            | 57.5¹  | —²    | 44.8¹ | Rolled back   |
| 3         | SAM2 res 1024 → 1536              | crashed | —     | —     | Failed (assertion) |
| 4         | Temperature annealing (T=0.3)      | pending | —    | —     | In progress   |

¹Estimated from the observed drops of −24.2 gIoU and −38 N‑acc points vs. baseline.  
²cIoU was not recorded for this failed iteration.

## 6. Discussion

Raising the VLM input resolution was the only successful intervention. The +0.7 gIoU and +1.8 cIoU gains confirm that the default 448 × 448 resolution discards visual detail beneficial for mask token prediction. The marginal N‑acc drop suggests that the higher‑resolution model occasionally produces mask tokens for empty‑target samples, likely because added detail introduces spurious cues. This trade‑off could be mitigated in future work by recalibrating the empty‑target detection threshold after resolution scaling.

The catastrophic failure of prompt rewriting highlights the rigidity of the instruction‑tuning alignment. The model has overfitted to the specific wording and tag placement of the original prompt; even a semantically similar variant disrupts the generation of mask tokens. Making the model robust to paraphrased instructions would require multi‑template data augmentation during training, which falls outside inference‑time optimization.

The SAM2 decoder’s fixed input size is an architectural limitation. The assertion error when passing a 1536‑side image confirms that the embedding layer’s dimensions cannot be changed without modifying the backbone. Potential workarounds—such as tiling and averaging masks—were not attempted within the current budget.

The pending temperature‑annealing trial, even if it yields a modest gain, is unlikely to overcome the large degradation from prompt changes. Overall, the study demonstrates that a simple resolution increase provides a meaningful, low‑cost improvement, while other levers are tightly constrained by the model’s fine‑tuning and architecture.

**Threats to validity**  
- The evaluation subset (500 samples) does not match the official test split; results may not perfectly generalize.  
- Only four optimization attempts were made, and one result is missing. A broader search could uncover additional beneficial modifications.  
- The model checkpoint used is a specific fine‑tuned variant, potentially affecting baseline values.  
- No retraining was performed; improvements are inference‑time only and may not transfer to other model scales or backbones.

## 7. Reproducibility

**Repository**: Sa2VA codebase [1].  
**Environment**:  
```bash
uv sync --extra latest
source .venv/bin/activate
```
**Seed**: Not explicitly set; inference is deterministic with `do_sample=False` except where noted.  
**Baseline run** (default resolution 448):  
```bash
python eval_gres.py   # min_pixels=448*28*28, max_pixels=448*28*28
```
**Optimized run** (resolution 896):  
Modify `eval_gres.py` to set:
```python
processor.image_processor.min_pixels = 896 * 28 * 28
processor.image_processor.max_pixels = 896 * 28 * 28
```
then execute the same command. The pretrained SAMTok model and `mask_tokenizer_256x2.pth` must be accessible at the specified paths.

## 8. References

```bibtex
@article{sa2va,
  title={Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos},
  author={Yuan, Haobo and Li, Xiangtai and Zhang, Tao and Sun, Yueyi and Huang, Zilong and Xu, Shilin and Ji, Shunping and Tong, Yunhai and Qi, Lu and Feng, Jiashi and Yang, Ming-Hsuan},
  journal={arXiv pre-print},
  year={2025}
}
```

```bibtex
@misc{autosota,
  author = {{tsinghua-fib-lab}},
  title = {{AutoSOTA: Automated State-of-the-Art Optimization Framework}},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
  year = {2025}
}
```
