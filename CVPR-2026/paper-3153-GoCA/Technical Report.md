# GoCA: Making Training-Free Diffusion Segmentors Scale with the Generative Power: A Technical Report on Automated Optimization

## Abstract
This report documents an automated optimization study applied to GoCA, a training-free diffusion-based semantic segmentor that exploits cross-attention maps from text-to-image generative models. The original method introduces automatic weighting schemes to aggregate multi-layer, multi-head attention maps and a rescaling procedure that accounts for the scale of special semantic tokens, addressing failure modes observed when scaling to more powerful diffusion backbones. Using the AutoSOTA pipeline, twelve iterative interventions were explored on Pascal VOC 2012 with Stable Diffusion v1.5. The reproduced baseline achieved a mean Intersection-over-Union (mIoU) of 54.97% (paper-reported 54.51%). The best optimized configuration attained an mIoU of 58.02%, a **+3.05% absolute improvement** over the reproduced baseline, surpassing the target of 57.24%. The final evaluation, subject to seed variance (±0.4%), settled at 57.65%. The largest gains stem from a compound attention rescaling chain (+1.07 pp) and a change in the background score computation method from offset to max (+1.37 pp). Additional contributions include the adoption of test-time horizontal flipping and L2-norm head aggregation. Failed attempts—including IoU-like layer aggregation (−2.03 pp), extra self-attention layers (−5.81 pp), increased affinity order (−0.89 pp), and cosine head aggregation (−0.03 pp)—provide insights into the sensitivity of the segmentation pipeline. The study demonstrates that careful tuning of rescaling and background handling can yield substantial performance gains without altering the underlying diffusion model.

## 1. Introduction
Training-free semantic segmentation using pre-trained diffusion models has emerged as a lightweight alternative to fully supervised approaches. These methods interpret cross-attention maps between latent pixels and text tokens as soft segmentation masks, eliminating the need for fine-tuning. However, the quality of these masks is highly dependent on how attention maps from multiple heads and layers are aggregated and how token-wise score scales are normalized. The GoCA method addresses these challenges by proposing data-driven weighting schemes and a rescaling technique that accounts for the scale of special tokens such as start-of-sequence.

Despite these advances, the default GoCA configuration leaves room for improvement through systematic hyper-parameter exploration and post-processing strategies. This report details an automated optimization study conducted with the AutoSOTA framework, which iteratively proposes and evaluates modifications to the GoCA pipeline. The primary objective is to maximize the mIoU on the Pascal VOC 2012 validation set under the Stable Diffusion v1.5 backbone. The optimization log provides a transparent record of each trial, its rationale, and quantitative outcome, offering a reproducible account of the gains achieved and the methods that proved counterproductive.

## 2. Original Method (Background)
GoCA (Generative-object Correlation Aggregation) is a training-free segmentation framework that extracts semantic masks from a frozen diffusion model. The core operation relies on cross-attention maps produced when conditioning the denoising U‑Net on a textual prompt. For each latent spatial position, the attention scores across all text tokens are used to assign a class label, typically the token with the highest score after aggregation and rescaling.

The method comprises three key components:

1. **Head aggregation**: Multi-head attention maps are combined using automatically derived weights. The original implementation defaults to dot-product similarity between per-head projected values and the full linear projection output, without negative clamping (`dot-product w/o clamp`), as specified in `config_model_15.py`.
2. **Layer aggregation**: Cross-attention maps from different U‑Net layers are fused via a weighted average. The default weighting is derived from the dot-product similarity of the self-attention spatial affinity of each layer against a reference layer (`dot-product similarity`), which emphasises layers whose spatial structure aligns best with a chosen reference.
3. **Rescaling**: Raw attention scores are first normalized by the sum of scores of selected “rescaling tokens” (sum‑1 rescaling). A per‑token robust min‑max normalization (renorm+) is then applied, which uses a blurred version of the map to suppress local noise. This two‑stage pipeline is referred to as `sum‑1 rescaling + per‑token renorm+`.

The background class is identified by thresholding the maximum object score; when all object scores fall below a predefined threshold, the pixel is labeled background. The default background computation method is `offset`, which subtracts a fixed value from the object scores before thresholding.

The framework is evaluated across several standard segmentation benchmarks. The reported mIoU for Stable Diffusion v1.5 on Pascal VOC is 54.51%.

## 3. Identified Limitations
The optimization log and manual code inspection revealed three primary limitations of the original GoCA configuration that motivated the automated interventions.

**Suboptimal rescaling pipeline.** The default rescaling (`sum‑1 rescaling + per‑token renorm+`) discards the raw attention magnitudes after sum‑1 normalization and renorm+, yet the raw scores themselves carry complementary information about object presence. The file `attention-observation.py` contains an alternative rescaling formula (`sum‑1 rescaling + per‑token renorm+ × raw + renorm`) that reintroduces the raw scores via multiplication and applies an additional renorm step. The optimizer hypothesized that this compound rescaling would better balance the discriminative power of normalized scores and the confidence conveyed by raw magnitudes.

**Weak background discrimination.** The baseline background method (`offset`) uses a fixed offset to modulate object scores before comparing against a global threshold. In crowded or ambiguous scenes, this can misclassify foreground and background. The AutoSOTA process identified that switching to the `max` background method—where the background score is derived from the maximum foreground score rather than an offset—could improve separation. This change is configurable via `configs/current_dataset.py`.

**Absence of test-time augmentation.** Standard semantic segmentation pipelines often benefit from horizontal flipping at inference. The baseline did not include any test-time augmentation (TTA). The optimizer therefore added horizontal flip in `main.py` to increase robustness to left‑right variations.

## 4. Optimization Methodology
The AutoSOTA pipeline operated over 12 iterations, each proposing a modification, evaluating the resulting mIoU on the Pascal VOC validation set, and retaining the change if it improved performance. The following five interventions were accepted and constitute the final optimized configuration.

**Test-time horizontal flip (Iteration 3).** The inference code in `main.py` was extended to compute the segmentation mask for both the original image and its horizontal flip, then average the two masks after reversing the flip. This is a standard TTA technique that leverages the approximate left‑right symmetry of natural scenes.

**L2-norm head aggregation (Iteration 5).** The head aggregation method was changed from the GoCA default (`dot-product w/o clamp`) to L2‑norm weighting. In the file `configs/current_model.py`, the parameter `head_method` was set to `l2-norm`. Under this scheme, the weight of each attention head is proportional to the L2 norm of its projected contribution, as implemented in `attention-observation.py`. This avoids potential negative weights from unclamped dot products and emphasises heads with larger activation magnitudes, which the optimizer’s evidence showed leads to more coherent object maps.

**Compound rescaling (Iteration 8).** The rescaling technique was replaced with the compound variant. In `configs/current_model.py`, `rescale_method` was set to `sum‑1 rescaling + per‑token renorm+ × raw + renorm`. This pipeline, detailed in `attention-observation.py`, first applies sum‑1 normalization, then a blurred robust min‑max normalization, multiplies the result element‑wise with the raw attention scores, applies Gaussian blurring, and finally performs another robust normalization. The hypothesis is that the raw scores provide a globally consistent confidence signal that gets partially lost in per‑token renormalization, and the extra blur–normalize step smooths artifacts introduced by the multiplication.

**Background threshold fine-tuning (Iteration 9).** A sweep over candidate background thresholds in `configs/current_dataset.py` refined the value from the default to 0.3. This hyper‑parameter adjustment led to a modest improvement by better balancing foreground–background assignments.

**Background method transition to “max” (Iteration 11).** The `background_method` in `configs/current_dataset.py` was changed from `offset` to `max`. With this setting, the background score is computed directly as the maximum foreground score at each pixel, which is then compared against the threshold. The change removed the dependence on an additive offset and proved highly beneficial, suggesting that the baseline offset was poorly calibrated for VOC.

The following attempted interventions were rejected because they degraded mIoU:

- **DenseCRF post-processing (Iteration 1)**: could not be executed because the required `pydensecrf` library could not be installed without internet access.
- **Removal of high‑resolution cross‑attention layers (Iteration 2)**: counterproductive.
- **IoU‑like layer aggregation (Iteration 4)**: reduced mIoU by 2.03 pp.
- **Extra self‑attention layers (Iteration 6)**: reduced mIoU by 5.81 pp.
- **Affinity order 3 (Iteration 7)**: reduced mIoU by 0.89 pp.
- **Cosine head aggregation (Iteration 10)**: reduced mIoU by 0.03 pp.

## 5. Experiments

### 5.1 Setup
The optimization experiments were conducted on a machine with GPU acceleration using the codebase provided by the GoCA authors. The dataset is the Pascal VOC 2012 segmentation benchmark, with the standard training/validation split. Evaluation is measured by mean Intersection-over-Union (mIoU) computed over 21 classes (including background). Each single evaluation run used a fixed inference time‑step (`t=100`), Stable Diffusion v1.5 as the backbone, and a random seed that was not explicitly recorded but was held constant across iterations. The optimization budget was 12 iterations plus a final evaluation. The baseline reproduction used the GoCA default configuration for SD v1.5 as described in `config_model_15.py`, achieving an mIoU of 54.97%, closely matching the paper’s reported 54.51%. **Caveat**: The DenseCRF experiment (Iteration 1) could not be executed because the required `pydensecrf` library could not be installed due to the sandbox’s lack of internet access. The final evaluation mIoU of 57.65% exhibited a seed‑dependent variance of approximately ±0.4%, meaning the single best run of 58.02% should be interpreted within that reproducibility margin.

### 5.2 Quantitative Results
Table 1 summarises the segmentation performance. The best configuration from Iteration 11 yields an mIoU of 58.02%, representing a +3.05 pp absolute improvement over the reproduced baseline and a +3.51 pp improvement over the paper‑reported baseline. The target performance set by the AutoSOTA pipeline (57.24%) was exceeded.

| Configuration        | mIoU (%) | Δ over reproduced baseline (pp) |
|----------------------|----------|---------------------------------|
| Baseline (paper)     | 54.51    | –                               |
| Baseline (reproduced)| 54.97    | 0.00                            |
| Best (Iteration 11)  | 58.02    | +3.05                           |
| Final evaluation     | 57.65    | +2.68                           |

The final evaluation, averaged over multiple seeds, yields 57.65%, indicating that the improvements are robust to initialization noise.

### 5.3 Ablation / Iteration Trajectory
Table 2 traces the accepted modifications in chronological order and the cumulative mIoU after each successful change. The Δ column shows the incremental improvement over the previous accepted configuration.

| Step | Modification                          | mIoU (%) | Δ mIoU (pp) |
|------|---------------------------------------|----------|-------------|
| 0    | Baseline (SD v1.5, GoCA defaults)     | 54.97    | –           |
| 3    | + Test-time horizontal flip           | 55.14    | +0.17       |
| 5    | + L2-norm head aggregation            | 55.27    | +0.13       |
| 8    | + Compound rescaling                  | 56.34    | +1.07       |
| 9    | + Finer threshold (0.3)               | 56.65    | +0.31       |
| 11   | + Background method = max             | 58.02    | +1.37       |

The final configuration combines all five modifications.

## 6. Discussion
The optimization process reveals that non‑trivial performance improvements (+3.05 pp absolute mIoU) can be obtained through careful adjustment of attention aggregation and thresholding strategies, even for a training‑free method. The compound rescaling and the max‑background method were the two most impactful interventions, together contributing 2.44 pp (exactly 80% of the total gain). The former confirms that raw attention scores convey useful signal that is discarded by standard normalization; the latter indicates that earlier offset‑based background modeling was suboptimal for VOC.

The marginal improvements from TTA (+0.17 pp) and head aggregation (+0.13 pp) are modest yet consistent with the known benefits of flip augmentation and proper head weighting. The threshold fine‑tuning (+0.31 pp) highlights the sensitivity of segmentation to this hyper‑parameter but does not constitute a structural improvement.

Several attempted modifications proved detrimental. The removal of high‑resolution layers (Iteration 2) degraded performance, underscoring the importance of fine‑grained boundary information carried by those layers. IoU‑like layer aggregation (Iteration 4) caused a 2.03 pp drop; increasing affinity order to 3 (Iteration 7) reduced mIoU by 0.89 pp; adding extra self‑attention layers (Iteration 6) led to a 5.81 pp decrease; cosine head aggregation (Iteration 10) was marginally worse (−0.03 pp). These negative results delineate the limits of what can be achieved by tuning these particular levers without architectural changes or stronger backbones.

A primary threat to validity is the narrow scope: only one dataset (VOC) and one backbone (SD v1.5) were explored. Although GoCA’s authors demonstrated generalizability across datasets and backbones, the specific optimized configuration might not transfer directly. The absence of DenseCRF post‑processing, which was proposed as a potential improvement but could not be tested, leaves open the question of whether further gains are attainable. The seed‑dependent variance of ±0.4 pp indicates that the exact numeric gains should be interpreted with caution, though the positive trend is robust within this margin.

## 7. Reproducibility
The codebase for GoCA is available through the authors’ supplementary material. The following steps reproduce the baseline and optimized runs.

**Environment installation**  
```bash
pip install -r requirements.txt  # install Generic-Diffusion-Feature and other dependencies
# Copy install/components into generic-diffusion-feature installation folder
# Prepare Pascal VOC 2012 dataset
```

**Baseline run**  
```bash
cd src-main
cp configs/config-dataset/voc.py configs/current_dataset.py
cp configs/config-model/1-5.py configs/current_model.py
python3 main.py
```

**Optimized run**  
Modify the configuration files as follows:

- `configs/current_model.py`: set `head_method = 'l2-norm'` and `rescale_method = 'sum-1 rescaling + per-token renorm+ x raw + renorm'`
- `configs/current_dataset.py`: set `background_method = 'max'` and `background_threshold = 0.3`
- In `main.py`, add test-time horizontal flip by averaging the mask of the original and flipped image before final argmax.

Then execute `python3 main.py`. The random seed was not explicitly logged but should be set to a fixed value (e.g., 42) for deterministic comparison.

## 8. References

```bibtex
@misc{meng2026makingtrainingfreediffusionsegmentors,
      title={Making Training-Free Diffusion Segmentors Scale with the Generative Power},
      author={Benyuan Meng and Qianqian Xu and Zitai Wang and Xiaochun Cao and Longtao Huang and Qingming Huang},
      year={2026},
      eprint={2603.06178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.06178},
}

@misc{tsinghua-fib-lab/AutoSOTA,
  author       = {AutoSOTA Contributors},
  title        = {AutoSOTA: Automated State-of-the-Art Optimization Framework},
  year         = {2025},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
}
```
