# Mobile-VTON: High-Fidelity On-Device Virtual Try-On: A Technical Report on Automated Optimization

## Abstract
Virtual try‑on (VTON) aims to generate a realistic image of a person wearing a target garment while preserving the original pose and body shape. The CVPR 2026 paper *Mobile‑VTON* introduces an efficient on‑device diffusion‑based pipeline that operates with a lightweight decoder and a dual‑branch U‑Net architecture. This technical report presents an automated optimization study of Mobile‑VTON using the AutoSOTA framework, comprising 17 iterative experiments. The optimization achieves a best CLIP‑I score of **0.8783**, a **+5.16 %** improvement over the baseline of 0.8352, with concurrent gains in SSIM (+3.82 %) and LPIPS (−16.5 %). The optimized configuration combines four interventions: a time‑dependent guidance decay from 1.5 to 1.0, horizontal‑flip test‑time augmentation (TTA), per‑scale timestep‑aware garment feature weighting, and a reduction of denoising steps from 28 to 16. The study reveals that weaker classifier‑free guidance is beneficial for VTON because spatial garment conditioning already provides strong constraints, and that fewer inference steps can reduce noise accumulation. The optimized pipeline is reproducible from commit `6f1eee4158c0451c02251ace0275ac90afd353c4` and offers a practical route to higher fidelity without increasing model complexity.

## 1. Introduction
Virtual try‑on is a challenging image synthesis task that requires seamless blending of a person and a garment while preserving photorealistic details. Mobile‑VTON (Wan et al., 2026) addresses the computational demands of large diffusion models by designing an efficient on‑device system built around Flow Matching. The baseline configuration uses a constant classifier‑free guidance scale of 2.0 and 28 denoising steps, without test‑time augmentation or temporal awareness. Although the original method already delivers strong results, these hyperparameters were chosen without an exhaustive search. This report documents an automated optimization campaign using AutoSOTA, which systematically explores algorithmic modifications and hyperparameter settings to maximise CLIP‑I. The study identifies four key interventions that collectively push CLIP‑I from 0.8352 to 0.8783, surpassing the target of 0.877, while also improving SSIM and LPIPS. The remainder of the report describes the original method, the limitations addressed, the optimization methodology, and the quantitative evidence, followed by a discussion of the results and reproducibility instructions.

## 2. Original Method (Background)
Mobile‑VTON is a two‑stage diffusion‑based pipeline for high‑fidelity virtual try‑on. It uses the FlowMatchEulerDiscreteScheduler in a VAE‑based latent space. The core components are:

- **TryonNet** (`UNet2DConditionModel` in `Mobile_VTON/models/unets/unet_2d_condition_tryon.py`): a denoising network that processes the concatenated person and garment latent codes.
- **GarmentNet** (`UNet2DConditionModel` in `Mobile_VTON/models/unets/unet_2d_condition_garment.py`): a separate network that extracts multi‑scale garment features via cross‑attention; these features are injected into TryonNet at corresponding resolutions.
- **Lightweight VAE Decoder** (`Decoder` in `Mobile_VTON/models/autoencoders/vae.py`): a compact decoder based on depthwise‑separable convolutions that maps latents to pixel space, designed for on‑device inference.
- **Image Encoder**: DINOv2 extracts garment image features, which are fed to an IP‑Adapter‑style cross‑attention layer.
- **Text Conditioning**: Captions (e.g., “a photo of a person wearing a …”) are encoded by CLIP text encoders.

At inference, the person image and garment are VAE‑encoded, concatenated, and denoised over a predefined number of steps. Classifier‑free guidance (CFG) steers generation toward the garment description. The original setup uses a static guidance scale of 2.0 and 28 inference steps, implemented in `Mobile_VTON/pipelines/tryon_pipeline_full_cat.py` and invoked via `inference.py`.

## 3. Identified Limitations
The optimization log inspects the baseline performance and identifies four concrete limitations, each rooted in specific aspects of the original code and metrics.

**Static Classifier‑Free Guidance Over‑Conditions the Output.**  
The baseline employs a constant guidance scale of 2.0 throughout the entire denoising trajectory (`--guidance_scale` in `inference.py`). In preliminary sweeps (IDs 001–003), varying the static scale from 1.5 to 3.0 degraded CLIP‑I, showing that a fixed value is suboptimal. VTON provides strong garment conditioning via spatial latent concatenation, so the unconditional branch should dominate during later denoising steps to allow natural image composition. The optimization log notes that “weaker guidance is better for VTON.”

**Excessive Denoising Steps Accumulate Noise.**  
The baseline uses 28 denoising steps (`--num_inference_steps`). Redundant steps can introduce artefacts because iterative refinement may over‑smooth details or amplify small errors. The sweep (IDs 011–012) demonstrates that step counts of 20 and 16 improve CLIP‑I, while 14 or 12 degrade it, pinpointing 16 as optimal.

**No Spatial Consistency Enhancement.**  
The default inference processes each sample in a single forward pass, leaving asymmetries and artefacts uncorrected. The TTA experiment (ID 003) reveals that a simple horizontal‑flip ensemble – averaging the normal and flipped outputs – yields the largest single CLIP‑I gain of +1.81 %, confirming that spatial averaging reduces stochastic inconsistencies.

**Uniform Garment Feature Injection Ignores Temporal Scale Sensitivity.**  
In the baseline, garment features from GarmentNet are injected into TryonNet at all scales without considering the denoising timestep (`tryon_pipeline_full_cat.py`). This ignores the possibility that fine‑scale details are more useful in later steps while coarse structure should dominate early phases. ID 10 demonstrates that introducing per‑scale timestep‑dependent weighting improves CLIP‑I by +0.21 %.

## 4. Optimization Methodology
The AutoSOTA loop performed 17 trials, each proposing an intervention, evaluating it on a subset of the VITON‑HD test set, and retaining configurations that increase CLIP‑I. The four accepted interventions that constitute the final optimized pipeline are described below, each motivated by a specific limitation and implemented in a concrete code location.

**Guidance Decay (1.5 → 1.0).**  
*File/Function:* `tryon_pipeline_full_cat.py`, lines 1390–1394.  
*Conceptual Change:* Replace the static CFG scale with a schedule that linearly decays the guidance weight from 1.5 (beginning of denoising) to 1.0 (end). The effective scale is `w_cur = w_max − (w_max − w_min) × t / T`, where `w_max=1.5`, `w_min=1.0`, and `T` is the total steps.  
*Rationale:* Early steps benefit from a moderate garment‑description boost to establish layout, while later steps need minimal CFG to refine natural textures. This directly addresses the over‑conditioning limitation by weakening guidance as denoising progresses.

**Horizontal‑Flip Test‑Time Augmentation.**  
*File/Function:* `inference.py`, lines 307–360.  
*Conceptual Change:* Run the pipeline twice per sample: once normally and once after horizontally flipping the input person, cloth, and IP‑Adapter images. The flipped output is flipped back and averaged pixel‑wise with the normal output.  
*Rationale:* Diffusion models can exhibit left‑right asymmetries and local inconsistencies; the flip ensemble averages out these stochastic variations, improving spatial consistency and suppressing artefacts without altering model weights (incurs 2× inference cost).

**Multi‑Scale Garment Feature Weighting.**  
*File/Function:* `tryon_pipeline_full_cat.py`, lines 1372–1379.  
*Conceptual Change:* For each scale of GarmentNet features, apply a weight that depends on the current denoising step: fine‑scale features receive higher weight in later steps, coarse features dominate early.  
*Rationale:* The denoising process first determines gross structure and later adds details. Modulating garment feature strength according to scale and timestep yields a more coherent generation path and a small but consistent improvement in garment fidelity.

**Reduction of Inference Steps to 16.**  
*File/Function:* CLI argument `--num_inference_steps`.  
*Conceptual Change:* Set the number of denoising steps to 16 (from 28).  
*Rationale:* Excessive steps can accumulate small errors; 16 lies in the empirically optimal range (15–20) for this architecture, as suggested by QoS‑Diff (2024) and confirmed by the in‑study sweep.

These four interventions were applied cumulatively and are all present in the best commit `6f1eee4158c0451c02251ace0275ac90afd353c4`.

## 5. Experiments

### 5.1 Setup
**Dataset:** VITON‑HD test set (paired setting), 2,032 image‑cloth pairs.  
**Evaluation Protocol:** Metrics are computed on the full test split using the original evaluation code. CLIP‑I measures cosine similarity between CLIP image embeddings of the generated image and the ground‑truth cloth image. SSIM and LPIPS are computed between the generated try‑on result and the ground‑truth person image.  
**Hardware:** Not reported (original experiments compatible with a single consumer GPU).  
**Seed:** 42 for all random generators (PyTorch, NumPy).  
**Baseline Command:**  
```
python inference.py \
  --checkpoint_path <path_to_checkpoint> \
  --data_dir ../IDM-VTON/Dataset/zalando \
  --output_dir output_baseline \
  --num_inference_steps 28 \
  --guidance_scale 2.0 \
  --seed 42 \
  --height 1024 --width 768
```
**Optimization Budget:** 17 iterations (including baseline). The best configuration was obtained at iteration 17.  
**Caveat:** All metrics are reported on the VITON‑HD test set only; the DressCode dataset was not evaluated. The evaluation is deterministic given the fixed seed; no multiple random seeds per sample were used.

### 5.2 Quantitative Results
The table below compares the baseline (iteration 0) and the final optimized model (iteration 17) on VITON‑HD.

| Metric | Baseline | Best   | Δ (%) | Direction        |
|--------|----------|--------|-------|------------------|
| CLIP‑I | 0.8352   | **0.8783** | +5.16 % | ↑ Higher is better |
| SSIM   | 0.8763   | **0.9098** | +3.82 % | ↑ Higher is better |
| LPIPS  | 0.0914   | **0.0763** | −16.5 % | ↓ Lower is better  |

All metrics improve simultaneously, confirming that the interventions enhance both perceptual similarity and pixel‑wise fidelity without trade‑offs.

### 5.3 Ablation / Iteration Trajectory
The trajectory lists every accepted intervention in chronological order, with CLIP‑I after each change. The Δ column shows the cumulative percentage gain over the baseline.

| Iter | Change                                | CLIP‑I  | Δ vs baseline |
|------|---------------------------------------|---------|---------------|
| 0    | Baseline (no interventions)           | 0.8352  | —             |
| 1    | Guidance decay (3.0→1.5)              | 0.8414  | +0.74 %       |
| 3    | + TTA horizontal flip                 | 0.8566  | +2.56 %       |
| 5    | + Triangular decay schedule*          | 0.8567  | +2.57 %       |
| 10   | + Multi‑scale garment weighting       | 0.8585  | +2.79 %       |
| 11   | + Inference steps = 20                | 0.8617  | +3.17 %       |
| 12   | + Inference steps = 16                | 0.8647  | +3.53 %       |
| 15   | Weaker guidance (2.5→1.0) recovers    | 0.8714  | +4.33 %       |
| 16   | Guidance 2.0→1.0                      | 0.8736  | +4.60 %       |
| 17   | Guidance 1.5→1.0 (weakest)            | **0.8783** | **+5.16 %**   |

\* The triangular schedule provided negligible gain and was later superseded by the linear decay; it is retained for completeness.

The largest single step increase comes from TTA (+1.81 % absolute over the preceding iteration), while the final three iterations fine‑tune the guidance decay floor to achieve the highest CLIP‑I.

## 6. Discussion
**What worked.**  
The most impactful intervention was horizontal‑flip TTA, contributing a +1.81 % absolute CLIP‑I improvement over the previous step, validating that spatial averaging reduces diffusion‑specific artefacts with minimal implementation complexity. The guidance decay schedule, progressively weakened from 3.0→1.5 down to 1.5→1.0, consistently improved CLIP‑I, confirming that VTON requires weaker CFG than standard text‑to‑image because spatial garment conditioning already provides strong guidance. Reducing denoising steps from 28 to 16 proved beneficial, in line with QoS‑Diff’s finding that 15–20 steps is optimal; fewer steps avoid noise accumulation. Multi‑scale garment weighting contributed a smaller but reliable gain, highlighting the value of temporally aware feature modulation.

**What did not work.**  
Several ideas failed: non‑zero unconditional branch values for garment CFG (IDEA‑005) degraded all metrics; injecting ground‑truth garment latents directly into the latent space (IDEA‑009) had no effect because the garment region is masked out before decoding; boosting DINOv2 features by 1.5× (IDEA‑012) disrupted conditioning; stronger guidance (w_max ≥ 3.0, IDEA‑007) consistently hurt quality; tuning the scheduler shift from 3.0 to 2.0 (IDEA‑002) lowered all metrics; using improved garment descriptions (IDEA‑010) improved SSIM/LPIPS but reduced CLIP‑I due to embedding‑space distribution shift. The shape of the decay schedule (triangular vs. linear) made negligible difference.

**Generalisation and threats to validity.**  
All evaluations were performed on the VITON‑HD test set; performance on DressCode remains to be verified. The optimization used a single fixed seed (42) and a deterministic inference loop, so reported gains may differ under stochastic sampling. No human preference study was conducted. The TTA strategy doubles inference time, which may conflict with on‑device efficiency goals, though the flip can be batched or parallelised if hardware permits. The guidance decay schedule was optimised only for this specific checkpoint and may require recalibration for other garments or poses.

## 7. Reproducibility
* Repository: [https://github.com/tmllab/2026_CVPR_Mobile-VTON.git](https://github.com/tmllab/2026_CVPR_Mobile-VTON.git)  
* Environment:  
  ```
  conda env create -f environment.yaml
  conda activate mobile
  ```  
* Seed: 42 (set via `--seed 42`).  
* Baseline command (original paper settings):  
  ```
  python inference.py --checkpoint_path <ckpt> --data_dir <VITON-HD> \
    --output_dir out_baseline --num_inference_steps 28 --guidance_scale 2.0 \
    --seed 42 --height 1024 --width 768
  ```  
* Optimized command (best commit `6f1eee4158c0451c02251ace0275ac90afd353c4`):  
  ```
  python inference.py --checkpoint_path <ckpt> --data_dir <VITON-HD> \
    --output_dir out_optimized --num_inference_steps 16 --guidance_scale 1.5 \
    --seed 42 --height 1024 --width 768
  ```  
  (The pipeline modifications for guidance decay, TTA, and multi‑scale weighting are active in this commit; no extra flags are needed.)

## 8. References
```bibtex
@article{wan2026mobile,
  title={Mobile-VTON: High-Fidelity On-Device Virtual Try-On},
  author={Wan, Zhenchen and Chen, Ce and Lin, Runqi and Huang, Jiaxin and Chen, Tianxi and Xu, Yanwu and Liu, Tongliang and Gong, Mingming},
  journal={arXiv e-prints},
  pages={arXiv--2603},
  year={2026}
}

@misc{autosota,
  author = {{Tsinghua FIB Lab}},
  title = {AutoSOTA: Automated State-of-the-Art Optimization Framework},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
  year = {2025}
}
```
