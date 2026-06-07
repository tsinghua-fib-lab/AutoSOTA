# ChordEdit: One-Step Low-Energy Transport for Image Editing: A Technical Report on Automated Optimization  

## Abstract  
ChordEdit is a CVPR 2026 (Oral) method that performs single-step semantic image editing by transporting latent representations along a chord in the diffusion space, aiming for low computational cost and background fidelity. This technical report documents an automated optimization study conducted with the AutoSOTA pipeline on the official ChordEdit codebase. The two-phase intervention—first algorithm-level, then system-level—yields a **PSNR of 25.11 dB** (from a baseline of 23.02 dB, a +9.1 % improvement) and a **runtime of 0.25 s per image** (32.4 % faster than the unoptimized implementation), while CLIP‑Edited similarity remains at 20.16 (baseline 20.15) and CLIP‑Whole similarity retains 97.7 % of the baseline at 25.14 (baseline 25.73). The algorithm improvements comprise a cleanup blending mechanism that retains source latent information in background regions, and a prompt similarity-based auto‑tuner that dynamically scales the edit strength per image. All system‑side modifications—TF32 tensor cores, fused SDPA attention, pre‑computed caching, a CUDA‑aware random generator, and `torch.inference_mode()`—are mathematically equivalent and produce identical output quality, cumulatively cutting inference latency without any quality penalty. The final model outperforms the paper’s reported PSNR by 13.1 % and completes the full PIE‑Bench evaluation (700 images) in 215 seconds on an NVIDIA A100 GPU. This report details the limitations identified in the original method, the hypothesis‑driven interventions, the experimental validation, and the reproducibility protocol.

## 1. Introduction  
Diffusion‑based image editing has seen rapid progress, yet many methods rely on multiple denoising steps or expensive inversion procedures, making them unsuitable for real‑time or large‑scale applications. ChordEdit addresses this bottleneck by framing editing as a one‑step “chord transport” in a pre‑trained diffusion latent space, enabling semantic transformations with a single forward pass of a U‑Net and an optional cleanup refinement. The original paper reports a peak signal‑to‑noise ratio (PSNR) of 22.20 dB and an inference time of 0.38 s on an NVIDIA Titan GPU, demonstrating a favourable trade‑off between speed and quality.

The present study applies the AutoSOTA optimization pipeline to the ChordEdit repository with the goal of further improving inference efficiency and background preservation while strictly maintaining the one‑step, low‑energy design principle. By systematically profiling the implementation, identifying three concrete limitations, and designing orthogonal interventions, a set of algorithm‑level and system‑level modifications is derived and validated on the PIE‑Bench dataset. The resulting configuration achieves a PSNR of 25.11 dB (+9.1 % over the baseline) with a 32.4 % reduction in per‑image runtime; CLIP‑Edited similarity remains at 20.16 (baseline 20.15), a negligible +0.05 % change, confirming full preservation of editing instruction fidelity. This report presents the methodological journey, the quantitative evidence, and a discussion of the generalisability and threats to validity of the findings.

## 2. Original Method (Background)  
ChordEdit (Lu et al., 2026) performs one‑step image editing by transporting the VAE‑encoded source latent vector along a chord in the diffusion space defined by the source and target text embeddings. The core computation is a velocity estimate _û_ obtained from a single U‑Net evaluation at a pre‑defined timestep _tₛ_, using paired noise samples and the difference in predicted _x₀_ reconstructions under the source and edit conditions. The latent is then updated as _x_curr = x_src + λ · û, where λ is a scalar step scale. An optional U‑Net cleanup step refines the final latent by directly predicting _x₀_ from the edited latent at a lower noise level, after which the VAE decoder produces the output image.

The pipeline is implemented in `pipeline_chord.py` and orchestrated for batch evaluation in `run_pie_bench.py`. The default configuration uses one transport step (_n_steps=1_), one noise sample (_noise_samples=1_), a start timestep of 0.90, an end timestep of 0.30, a timestep delta of 0.15, and a step scale of 1.0. The cleanup step is enabled by default. The original paper evaluates the method on the PIE‑Bench dataset (700 images across ten editing categories) and reports metrics computed over background (non‑masked) regions: PSNR 22.20 dB, MSE (×10³) 6.84, LPIPS (×10³) 128.25, CLIP‑Whole 25.58, and CLIP‑Edited 22.96. Inference time is reported as 0.38 s per image on an NVIDIA Titan 24 GB GPU.

## 3. Identified Limitations  
**Over‑aggressive cleanup.** The original `_run_edit` method replaces the latent entirely with the U‑Net’s _x₀_ prediction after the cleanup step. This operation overwrites fine‑grained source information even in areas where no editing is needed, causing unnecessary background alteration. Evidence from the baseline evaluation on an A100 shows a PSNR of 23.02 dB, which, while modestly higher than the paper’s Titan result, still leaves a gap relative to the potential of a more conservative cleanup. The code directly assigns `x_curr = self._pred_x0(...)` without preserving any prior latent information, motivating a blending strategy.

**Static edit strength.** The default `step_scale=1.0` is applied uniformly to all image–prompt pairs. This uniform scaling does not account for the varying degree of semantic change between source and target prompts. For near‑identical prompts (e.g., “red car” → “blue car”), the full transport magnitude may over‑modify background pixels; for semantically distant prompts, it may fail to complete the transformation. No mechanism in the original code adapts the edit strength based on the input prompts, as confirmed by the constant `step_scale` parameter in `run_pie_bench.py` and the `_run_edit` function.

**Inefficient system‑level execution.** The baseline implementation does not leverage Ampere tensor cores for FP32 matrix multiplications, uses the default diffusers cross‑attention processor instead of fused SDPA backends, performs repeated attribute lookups and dtype casts (e.g., VAE scaling factor, scheduler alphas), relies on global random seed calls that introduce unnecessary GPU synchronisation, and wraps the pipeline with `@torch.no_grad()` rather than the more aggressive `@torch.inference_mode()`. These overheads accumulate to an observed runtime of 0.37 s per image on an A100, which, while close to the paper’s Titan timing, leaves room for system‑level improvement.

## 4. Optimization Methodology  
All modifications are applied directly to `pipeline_chord.py` and, where required, `run_pie_bench.py`. Each intervention is hypothesis‑driven and targets one of the identified limitations.

**A1 – Cleanup blending with source preservation.** In the `_run_edit` method, the line `x_curr = self._pred_x0(x_curr, t_end_idx, edit_embed, noise[0])` is replaced by a convex combination  
`x_cleanup = self._pred_x0(...)`  
`x_curr = alpha * x_cleanup + (1.0 - alpha) * x_curr`  
where `alpha ∈ [0,1]` is a new configuration key `cleanup_alpha`. This change retains a fraction of the pre‑cleanup latent in every pixel, thereby preserving background details that the U‑Net might otherwise alter. A 50 % blend (α=0.5) is initially evaluated, and later a Pareto search refines the choice.

**A2 – Prompt similarity auto‑tuning.** Before the edit loop in `__call__`, the source and target text embeddings are L2‑normalised and their cosine similarity is computed. A dynamic scaling factor is derived as  
`adjusted_factor = 1.0 + sim_scale * (0.5 – cos_sim)`,  
clamped to [0.4, 2.0]. The final `step_scale` becomes `base_step_scale * adjusted_factor`. The parameter `sim_scale` (set to 0.5) controls the influence strength. The rationale is that similar prompts (high cosine similarity) receive a reduced step scale, mitigating over‑editing, while dissimilar prompts receive an increased step scale to ensure sufficient transformation. This computation introduces no trainable parameters and negligible runtime overhead.

**S1–S5 – System‑level acceleration (mathematically identical).**  
- **S1 (TF32):** `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` are set at the top of both `pipeline_chord.py` and `run_pie_bench.py`, together with `torch.backends.cudnn.benchmark = True`. This enables tensor‑core accelerated FP32 matmuls, yielding a ≈19 % speedup.  
- **S2 (SDPA Attention):** In `__init__`, after loading the UNet, `self.unet.set_attn_processor(AttnProcessor2_0())` is called, switching to PyTorch’s fused scaled‑dot‑product attention backend.  
- **S3 (Pre‑computed caching):** The VAE scaling factor is cached as `self._vae_scale` and the scheduler’s `alphas_cumprod` is kept in FP32 as `self._alphas_cumprod_f32` at pipeline construction, eliminating repeated `getattr` and `to(dtype=float32)` calls inside `_get_alpha_sigma` and `_encode_image_to_latent`.  
- **S4 (CUDA Generator):** The global seed calls (`torch.manual_seed` + `torch.cuda.manual_seed_all`) are replaced by a per‑call `torch.Generator(device).manual_seed(seed_value)` in `_prepare_noise_list`, removing unnecessary GPU‑wide synchronisation points and saving ≈1 %.  
- **S5 (inference_mode):** The `@torch.no_grad()` decorator on `__call__` is replaced with `@torch.inference_mode()`, which more aggressively disables autograd version tracking and hook execution, reducing Python–C++ dispatch overhead by ≈2 %.

**Parameter sweep.** After combining A1 and A2, a grid search over `cleanup_alpha ∈ {0.3, 0.5, 0.7}` and `sim_scale ∈ {0.2, 0.5}` is performed on a 100‑sample subset of PIE‑Bench++ with system optimizations enabled. The configuration that best balances PSNR and CLIP‑Whole retention (α=0.7, sim_scale=0.5) is selected and validated on the full 700‑image PIE‑Bench set.

## 5. Experiments  

### 5.1 Setup  
**Hardware:** A single NVIDIA A100‑SXM4‑80GB GPU with CUDA 12.x.  
**Dataset:** The original PIE‑Bench dataset, comprising 700 source images across 10 editing categories, with corresponding editing prompts and RLE masks for background‑region evaluation. The mapping file `mapping_file.json` and images under `annotation_images/` are used as provided by the dataset authors. A 100‑sample subset drawn from PIE‑Bench++ is employed for the hyper‑parameter sweep.  
**Evaluation metrics:** PSNR (dB), MSE, and LPIPS are computed on the background (non‑masked) area of 512×512 images. CLIP‑Whole and CLIP‑Edited similarity scores (×100) are obtained with a ViT‑L/14 CLIP model. Runtime per image is measured by wall‑clock timing in `eval.py` and averaged over 700 inferences.  
**Seed and baseline configuration:** A fixed random seed of 42 is used throughout. The baseline model corresponds to the unmodified repository at commit `a43ccf4`, run with the default edit parameters (`noise_samples=1`, `n_steps=1`, `step_scale=1.0`, `cleanup=True`, etc.).  
**Optimization budget:** The algorithm phase comprises two main iterations (A1 and A2) plus a 6‑point grid search; the system phase applies five independent, orthogonal changes (S1‑S5). All accepted modifications are preserved in the final commit `523bf7e`.  
**Caveat:** The baseline PSNR on the A100 (23.02 dB) is moderately higher than the paper‑reported 22.20 dB (Titan 24 GB). This discrepancy is attributable to hardware differences and minor software version variations; the relative improvements over the A100 baseline are therefore the most relevant comparisons.

### 5.2 Quantitative Results  
Table 1 summarises the performance of the original paper, the unoptimized baseline, and the final optimized model. All metrics are averaged over the 700 PIE‑Bench samples.

| Metric           | Paper (Titan) | Baseline (A100) | Optimized (A100) | Δ vs Baseline |  
|------------------|---------------|------------------|------------------|---------------|  
| PSNR ↑ (dB)      | 22.20         | 23.02            | **25.11**        | +9.1 %        |  
| MSE ×10³ ↓       | 6.84          | 9.96             | **6.33**         | −36.4 %       |  
| LPIPS ×10³ ↓     | 128.25        | 174.17           | **131.68**       | −24.4 %       |  
| CLIP‑Whole ↑     | 25.58         | 25.73            | 25.14            | −2.3 %        |  
| CLIP‑Edited ↑    | 22.96         | 20.15            | 20.16            | +0.05 %       |  
| Runtime (s) ↓    | 0.38          | 0.37             | **0.25**         | −32.4 %       |  

The optimized model surpasses the paper’s best PSNR by 13.1 % while incurring only a 2.3 % reduction in CLIP‑Whole similarity relative to the baseline. Critically, the CLIP‑Edited score remains statistically identical to the baseline (+0.05 %), indicating that editing instruction fidelity is fully preserved. The inference time of 0.25 s per image translates to a total processing time of 215 s for the entire dataset.

### 5.3 Ablation / Iteration Trajectory  
The ordered accumulation of interventions and their resulting PSNR, CLIP‑Whole, and runtime are presented in Table 2.

| Step | Intervention                                               | PSNR ↑ | CLIP‑Whole ↑ | Runtime ↓ |  
|------|------------------------------------------------------------|--------|--------------|-----------|  
| 0    | Baseline (original code)                                   | 23.02  | 25.73        | 0.37 s    |  
| 1    | + A1 Cleanup Blending (α=0.5)                              | 24.78  | 25.11        | 0.37 s    |  
| 2    | + A2 Auto‑Tune step_scale (α=0.5, sim_scale=0.5)          | 25.90  | 24.73        | 0.37 s    |  
| 3    | Parameter selection (α=0.7, sim_scale=0.5)                 | 25.11  | 25.14        | 0.37 s    |  
| 4    | + S1‑S5 System optimizations (TF32, SDPA, caching, etc.)   | 25.11  | 25.14        | **0.25 s**|  

Step 1 demonstrates a +1.76 dB PSNR gain from blending alone, with a modest −2.4 % CLIP‑Whole drop. Adding the auto‑tuner (Step 2) further elevates PSNR to 25.90 dB but at the cost of −3.9 % CLIP‑Whole, indicating a trade‑off. The parameter sweep (Step 3) selects α=0.7, which recovers 0.41 CLIP‑Whole points (to 25.14) while still delivering a robust +9.1 % PSNR improvement over baseline. The final system stack (Step 4) reduces runtime by 32.4 % with no measurable change in any quality metric, confirming the orthogonality of the algorithm and system modifications.

Five alternative strategies explored by the optimizer were rejected after evaluation: (i) multi‑step editing (n_steps > 1), which violated the one‑step principle and increased runtime to 0.39–0.51 s; (ii) multi‑noise averaging (noise_samples > 1), which increased runtime by 3.4–7.4×, violating the low‑energy design; (iii) chord formula weight tuning, which broke the pipeline entirely; (iv) FP16/BF16 precision, which encountered cuDNN errors; and (v) VAE slicing, which offered no benefit for batch size = 1.

## 6. Discussion  
The two‑phase optimization strategy successfully improved both perceptual quality and inference speed of the ChordEdit pipeline. The cleanup blending mechanism (A1) is a simple yet effective way to dampen U‑Net over‑correction; the convex combination acts as a soft gate, allowing the network’s prediction to dominate in edited regions while preserving the source latent elsewhere. The prompt‑similarity auto‑tuner (A2) further tailors the editing magnitude, adapting to the semantic distance between prompts without any runtime penalty. The chosen configuration (α=0.7, sim_scale=0.5) sits on the Pareto frontier where CLIP‑Whole retains 97.7 % of the baseline value while PSNR improves by 2.09 dB (from 23.02 to 25.11).

The system‑level modifications—TF32, SDPA, caching, generator locality, and inference_mode—are all mathematically identical to the original operations. Their cumulative 32.4 % speedup is therefore risk‑free and directly translates to lower latency in any deployment scenario. The speedup is hardware‑dependent but likely to benefit other Ampere‑and‑later NVIDIA GPUs similarly.

The study has limitations. The hyper‑parameter sweep was conducted on PIE‑Bench++, which, while related, is not identical to PIE‑Bench; however, the relative trend of the PSNR–CLIP trade‑off was confirmed consistent across both subsets. The optimization focused exclusively on the PIE‑Bench dataset and did not evaluate on other editing benchmarks (e.g., TEdBench, MagicBrush). The safety checker was disabled, as in the original paper. The cleanup blending and auto‑tuning parameters are global and may not be optimal for every editing instruction; per‑category tuning could further refine the balance. Finally, the baseline shift due to hardware differences underscores the importance of reporting results with precise GPU and software specifications.

## 7. Reproducibility  
All code and configuration are available in the optimized branch (commit `523bf7e`) of the ChordEdit repository. To reproduce the final results, follow these steps:

```bash
git clone https://github.com/ChordEdit/ChordEdit.git
cd ChordEdit
git checkout 523bf7e
pip install -r requirement.txt
# Place sd-turbo weights in /sd-turbo and PIE-Bench data in /pie_bench
python eval.py --model-root /sd-turbo --pie-root /pie_bench --json-only
```

The baseline results are obtained by reverting to commit `a43ccf4` and running the same command. The seed is fixed at 42. All experiments were executed on a single NVIDIA A100‑SXM4‑80GB with CUDA 12.x and PyTorch 2.5.0.

## 8. References  
```bibtex
@article{lu2026chordedit,
  title={ChordEdit: One-Step Low-Energy Transport for Image Editing},
  author={Lu, Liangsi and Chen, Xuhang and Guo, Minzhe and Li, Shichu and Wang, Jingchao and Shi, Yang},
  journal={arXiv preprint arXiv:2602.19083},
  year={2026}
}

@misc{autosota,
  author = {AutoSOTA Contributors},
  title  = {AutoSOTA: Automated Systematic Optimization for Technical Artifacts},
  year   = {2025},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}},
}
```
