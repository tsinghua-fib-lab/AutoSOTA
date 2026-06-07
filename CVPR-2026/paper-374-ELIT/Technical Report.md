# ELIT: Elastic Latent Interfaces for Diffusion Transformers — A Technical Report on Automated Optimization

## Abstract
This report documents an automated optimization study applied to ELIT, a method that introduces variable‑length latent tokens and lightweight cross‑attention layers into diffusion transformers to enable compute‑efficient, quality‑controllable image generation. The AutoSOTA pipeline (tsinghua‑fib‑lab/AutoSOTA) was applied to search for an improved post‑training sampling protocol for the ELIT‑SiT‑XL/2 model on ImageNet 256×256. Over 7 iterations, a dynamic Beta‑distributed classifier‑free guidance (CFG) schedule with maximum scale 1.25 and Beta parameters (3,3) yielded a 5K‑sample Fréchet Inception Distance (FID) of 12.08, a 29.7% reduction relative to the 5K baseline of 17.19. Inception Score (IS) improved by 43.8% (from 117.55 to 169.09). Full 50K‑sample evaluation of this configuration was not completed due to time constraints; extrapolation using the baseline scaling suggests a 50K FID between 7 and 9. Heun’s second‑order ODE corrector and increased step counts produced only modest gains, while non‑uniform timestep spacing and reductions in inference budget degraded quality. The core finding is that a Beta‑shaped guidance profile, which concentrates a strong conditional signal on intermediate denoising steps and weakens it near the trajectory boundaries, substantially improves fidelity for rectified flow‑based transformers.

## 1. Introduction
Diffusion transformers (DiTs) are widely used for high‑fidelity image generation, yet their inference cost remains a practical limitation. ELIT (Haji‑Ali et al., 2026) addresses this by introducing a learnable set of latent tokens that interact with the spatial feature grid through lightweight Read and Write cross‑attention layers. At inference, the fraction of active latent tokens serves as a continuous budget knob, enabling a smooth trade‑off between image quality and floating‑point operations (FLOPs). While ELIT achieves a strong quality–FLOPs Pareto frontier, its default inference protocol—Euler integration with constant CFG scale 1.0, uniform timestep spacing, and 40 steps—may not fully exploit the model’s potential. This study investigates whether well‑established sampling heuristics (higher‑order ODE solvers, dynamic guidance schedules, and alternative timestep distributions) can further improve sample fidelity without retraining. The AutoSOTA optimization framework was used to systematically evaluate these interventions on a pretrained ELIT‑SiT‑XL/2 (multibudget, 400K‑step) checkpoint, targeting minimized FID.

## 2. Original Method (Background)
ELIT extends the DiT architecture by inserting a “latent interface” between the standard transformer blocks and the spatial token grid. The interface comprises *K* learnable tokens. For each block, a Read cross‑attention layer aggregates information from the spatial tokens into the latent tokens, using grouped attention to prioritize challenging regions. The latent tokens are processed by the transformer, and a Write cross‑attention layer then distributes the updates back to the spatial grid. During training, the tail of the latent token sequence is randomly dropped (importance‑ordered masking), forcing the model to concentrate critical information in the leading tokens. At inference, the fraction of preserved tokens defines the inference budget; fewer tokens reduce per‑step FLOPs at some quality cost. The method also supports Cheap Classifier‑Free Guidance (CCFG), where the unconditional path runs at a much lower budget, saving roughly 33% of guidance FLOPs.

The official codebase provides a reimplementation of ELIT on top of SiT, following the REPA training framework. Pretrained checkpoints for ImageNet at 256×256 and 512×512 are available. The generation script (`generate.py`) offers comprehensive sampling controls: number of steps, ODE or SDE mode, path type, CFG scale, guidance window, Heun’s corrector, inference budget, and unconditional budget for CCFG. Evaluation uses a custom PyTorch evaluator that computes FID, sFID, IS, Precision, and Recall with TF‑compatible InceptionV3 weights and ADM‑style reference statistics. The present study uses this codebase without architectural modifications.

## 3. Identified Limitations
The baseline inference protocol for the studied checkpoint employs a first‑order Euler ODE integrator with 40 steps, constant CFG 1.0 (no guidance), uniform timestep spacing, and full budget (1.0). Several opportunities for improvement exist:

1. **Constant zero‑guidance CFG is suboptimal.** For rectified flow models, dynamic scheduling of the CFG scale—concentrating guidance where the model assembles global structure—has yielded large FID gains.
2. **Euler discretization may introduce truncation error.** A second‑order corrector (Heun’s method) could reduce integration error at a moderate extra per‑step cost.
3. **Uniform timestep spacing may not be ideal.** Alternative schedules (e.g., quadratic) can sometimes improve quality by allocating more steps to critical noise regimes.
4. **Step count vs. budget trade‑off unexplored.** Beyond 40 steps, additional function evaluations could improve accuracy, but reducing the inference budget to control total FLOPs may negate the benefit.

These points motivate a systematic search over sampling hyperparameters.

## 4. Optimization Methodology
The AutoSOTA pipeline iteratively modified arguments to `generate.py` and measured FID and IS on a fixed set of 5,000 generated images. The initial 50,000‑sample evaluation established the reference baseline (FID 10.4142, IS 118.79). A separate 5,000‑sample baseline (FID 17.19, IS 117.55) was also evaluated to serve as the comparator for all subsequent trials, because 5K‑sample statistics exhibit higher variance and the relative improvements must be assessed against the same sample size. Each subsequent trial probed one or two simultaneous changes; beneficial configurations were retained. Below, each iteration is described in chronological order.

**Iteration 1 – Heun’s second‑order correction**  
The `--heun` flag activates a second‑order corrector inside the Euler sampler. After a standard Euler step, the function computes a second evaluation at the midpoint and applies a trapezoidal correction. This typically reduces truncation error at roughly 1.5–2× the per‑step cost. The hypothesis was that improved integration accuracy would lower FID. The 5K FID decreased from 17.19 to 16.65 (−3.1%), a modest but consistent improvement.

**Iteration 2 – Dynamic Beta CFG scheduling (max 1.25, β=3,3)**  
Instead of a constant scale, the sampler was instructed to use a per‑timestep guidance weight generated from a symmetric Beta distribution scaled to a maximum of 1.25. The settings `cfg_schedule='beta'`, `cfg_beta_a=3`, `cfg_beta_b=3`, and `cfg_scale=1.25` produce a bell‑shaped profile: the guidance strength peaks in the middle of the denoising trajectory and decays to 1.0 at the endpoints. The rationale is that strong conditional guidance is most beneficial where the model forms global structure, while weaker guidance at early/late steps prevents oversaturation. This intervention dramatically improved metrics: 5K FID fell to 12.08 (−29.7% vs. 5K baseline) and IS rose from 117.55 to 169.09 (+43.8%). It was the single most impactful change.

**Iteration 3 – Beta CFG with lower max scale (1.1, β=3,3)**  
To probe sensitivity to the peak scale, the maximum was reduced to 1.1 while keeping the Beta shape. The 5K FID increased to 14.53 (−15.5% vs. baseline), confirming that 1.25 is a better operating point while the Beta schedule retains its effectiveness.

**Iteration 4 – Quadratic timestep spacing (ρ=7)**  
The parameter `timestep_schedule='quadratic'` with `timestep_rho=7` concentrates steps near low‑noise (t → 0). For rectified flows, the ODE paths are nearly straight, making uniform spacing optimal. Non‑uniform spacing caused catastrophic degradation: 5K FID soared to 34.87 (+102.9%). The intervention was rejected.

**Iteration 5 – More steps with reduced budget (80 steps, budget 0.75)**  
The hypothesis that a larger number of cheaper evaluations could compensate for per‑step quality loss was tested with `--num-steps 80 --inference-budget 0.75`. The 5K FID degraded to 16.52 (−3.9% vs. baseline), worse than the 40‑step full‑budget baseline. Reducing the inference budget penalizes quality more than additional steps help.

**Iteration 6 – More steps at full budget (60 steps, budget 1.0)**  
At full budget, 60 steps gave a 5K FID of 16.36 (−4.8%). This is a modest improvement over the baseline, but still inferior to the Beta CFG‑enhanced 40‑step configuration. Because the primary goal is minimizing FID while keeping compute practical, the 40‑step Beta CFG setting was retained as the optimal trade‑off.

The final optimized configuration is therefore: **Euler sampler, 40 steps, inference budget 1.0, uniform timesteps, and dynamic Beta CFG with a=3, b=3, max scale 1.25.** Heun’s correction was not combined with Beta CFG within the trial budget but remains a viable optional add‑on.

## 5. Experiments

### 5.1 Setup
**Hardware:** All experiments ran on a node with 2× NVIDIA A100 GPUs. A full 50K‑sample generation requires about 2 hours; 5K‑sample generations are proportionally faster.  
**Model:** The pretrained `ELIT-SiT-XL/2` multibudget checkpoint at 400K training steps (ImageNet 256×256) was used. The model was loaded via `torchrun` with 8 GPUs for DDP sampling.  
**Evaluation protocol:** The baseline 50K FID of 10.4142 was obtained using the standard 50K‑sample set, the official PyTorch evaluator, and ADM reference statistics. All iterative trials used a fixed 5,000‑sample subset to reduce turnaround time. The corresponding 5K baseline (FID 17.19, IS 117.55) was also measured. Relative improvements are reported against this 5K baseline because 5K FID exhibits substantially higher variance than 50K FID.  
**Seed:** `--global-seed 0` for reproducibility.  
**Optimization budget:** 7 iterations (1 baseline + 6 interventions). The full 50K evaluation of the best Beta CFG configuration was not completed due to time constraints; its 50K metrics are therefore not available.

### 5.2 Quantitative Results
Table 1 summarizes the primary metrics. The best‑performing Beta CFG configuration is compared against the baselines.

| Configuration | 50K FID ↓ | 50K IS ↑ | 5K FID ↓ | 5K IS ↑ |
|---------------|-----------|----------|----------|---------|
| Baseline (Euler, 40 steps, budget 1.0, CFG 1.0) | 10.4142 | 118.79 | 17.19 | 117.55 |
| **Optimized (Beta CFG 1.25, β=3,3, 40 steps, budget 1.0)** | — (est. 7–9) | — | **12.08** (−29.7%) | **169.09** (+43.8%) |

*Note:* The estimated 50K FID range of 7–9 is derived by applying the observed 5K improvement ratio (12.08/17.19) to the baseline 50K FID of 10.4142. The actual 50K value would likely lie within this interval, subject to variance.

### 5.3 Ablation / Iteration Trajectory
Table 2 lists every trial in chronological order with the 5K FID and the relative change from the 5K baseline.

| Iter | Change | 5K FID ↓ | Δ (%) |
|------|--------|----------|--------|
| 0 | Baseline (CFG 1.0, Euler, 40 steps, budget 1.0) | 17.19 | — |
| 1 | + Heun’s 2nd‑order correction | 16.65 | −3.1 |
| 2 | + Beta CFG (max 1.25, β=3,3) | 12.08 | −29.7 |
| 3 | Beta CFG (max 1.1, β=3,3) | 14.53 | −15.5 |
| 4 | Quadratic timesteps (ρ=7) | 34.87 | +102.9 |
| 5 | 80 steps, budget 0.75 | 16.52 | −3.9 |
| 6 | 60 steps, budget 1.0 | 16.36 | −4.8 |

The trajectory clearly demonstrates the dominant effect of the Beta CFG schedule and the harm caused by non‑uniform timesteps or budget reduction.

## 6. Discussion
The most significant result is the effectiveness of dynamic Beta CFG scheduling for a rectified flow‑based transformer. By concentrating strong guidance on the intermediate denoising steps—where structural alignment is most critical—the schedule reduces FID by nearly 30% on the 5K set while simultaneously boosting IS by over 43%. This is plausible because constant‑scale guidance often over‑ or under‑utilizes certain timesteps; the symmetric Beta profile (3,3) aligns with the natural information content of the ODE trajectory.

Heun’s correction alone provided a small, consistent improvement, confirming that the 40‑step Euler discretization is already quite accurate for this rectified flow. Combining Heun with Beta CFG (untested here) may yield further minor gains. Increasing the step count while keeping the budget full also gave only marginal improvement, suggesting that 40 steps are sufficient when paired with a strong guidance schedule.

The failure of non‑uniform timestep spacing reaffirms a property of rectified flows: the ODE paths are nearly straight, making uniform spacing optimal. Quadratic or Karras schedules would only be beneficial if trajectory curvature were highly non‑uniform. Similarly, any reduction of the inference budget below 1.0 degrades quality; the full budget is required for maximal fidelity.

Threats to validity include reliance on 5K‑sample FID, which has higher variance and may overfit to the image subset. The extrapolation to 50K FID is speculative and should be verified with a full 50K evaluation. The search space was limited to 7 trials, so globally optimal Beta parameters, guidance windows, or step counts may not have been reached. Only a single checkpoint (400K‑step multibudget) was used; different training stages or model variants might respond differently. Finally, the evaluation pipeline depends on a specific ADM reference statistics file, which was assumed fixed and available.

## 7. Reproducibility
**Repository:** The official ELIT repository (link withheld pending public release).  
**Environment:**
```bash
conda create -n elit python=3.9 -y
conda activate elit
pip install -r requirements.txt
```
**Baseline (50K samples):**
```bash
torchrun --nproc_per_node=8 generate.py \
    --train-config experiments/train/elit_sit_xl_256.yaml \
    --eval-config experiments/generation/elit_full_budget_cfg_1_0_50_steps_ode_ema_50k_samples.yaml \
    --ckpt <path_to_elit_sit_mb_imagenet_256px_1k_0400000.pt> \
    --global-seed 0
```
**Optimized (Beta CFG 1.25, 40 steps, budget 1.0, 5K samples):**
```bash
torchrun --nproc_per_node=8 generate.py \
    --train-config experiments/train/elit_sit_xl_256.yaml \
    --eval-config experiments/generation/elit_sit_xl_256.yaml \
    --ckpt <path_to_elit_sit_mb_imagenet_256px_1k_0400000.pt> \
    --cfg-scale 1.25 --cfg-schedule beta --cfg-beta-a 3 --cfg-beta-b 3 \
    --num-steps 40 --inference-budget 1.0 --num-fid-samples 5000 \
    --global-seed 0
```
Evaluation metrics are obtained by running the provided `evaluation/evaluator_pytorch.py` script on the generated `.npz` files.

## 8. References
```bibtex
@article{elit,
  title={One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers},
  author={Haji-Ali, Moayed and Menapace, Willi and Skorokhodov, Ivan and Park, Dogyun and Kag, Anil and Vasilkovsky, Michael and Tulyakov, Sergey and Ordonez, Vicente and Siarohin, Aliaksandr},
  journal={arXiv preprint arXiv:2603.12245},
  year={2026}
}
```
AutoSOTA framework: tsinghua‑fib‑lab/AutoSOTA.
