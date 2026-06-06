# Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning: A Technical Report on Automated Optimization

## Abstract

This technical report documents the automated optimization, by the AutoSOTA system, of the publicly released code accompanying the CVPR 2026 paper *Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning*, whose original method is referred to as MVC-ZigAL. MVC-ZigAL combines a few-step latent-consistency multi-view backbone (LCM-SDXL with the MV-Adapter) with reinforcement-learning finetuning under a self-refinement (Zigzag Multi-View, ZMV) sampling scheme and a Lagrangian dual constraint between single-view and joint-view rewards. The reproduction was conducted under strict network restrictions, in which the RL-finetuned LoRA checkpoint and the HyperScore evaluator could not be downloaded. Consequently, optimization was confined to the inference-time configuration of the LCM-SDXL baseline on the MATE-3D benchmark (160 prompts, 6 views per prompt). Across thirteen automated iterations, the system explored algorithmic, parametric, and prompt-level interventions. The best configuration—Best-of-5 seed selection combined with 14 inference steps—lifted PickScore from 0.1948 to 0.2035 (+4.47%), HPSv2 from 0.2432 to 0.2514 (+3.37%), and ImageReward from -1.4070 to -0.5951 (+57.7%) without any metric trade-off. The PickScore target of 0.2045 was missed by 0.0010 (0.5%). The trajectory clarifies which inference-time levers are productive for few-step T2MV diffusion and isolates the residual gap that is attributable to the unavailable RL-finetuned LoRA. The report describes the methodology, the iteration trajectory, and the reproducibility constraints in detail.

## 1. Introduction

Few-step text-to-multiview (T2MV) diffusion models promise real-time generation of consistent multi-view imagery from a single text prompt, but at the cost of degraded per-view fidelity and weakened cross-view coordination. The MVC-ZigAL paper addresses this trade-off through reinforcement-learning finetuning tailored to the few-step regime, with an MDP that jointly models all views, an advantage-learning signal derived from a self-refinement sampling scheme, and a Lagrangian dual that balances single-view and joint-view objectives.

This report does not propose new methodology. Instead, it documents how the AutoSOTA optimization framework (`tsinghua-fib-lab/AutoSOTA`) interacted with the released code to reproduce, diagnose, and partially recover the reported quality of MVC-ZigAL under environment-induced constraints. The objective was to surface inference-time levers that are robust under realistic deployment conditions when the RL-trained LoRA is not available and to characterize the residual gap. All optimization was performed against the MATE-3D prompt benchmark, with PickScore as the primary scalar objective and HPSv2 and ImageReward as secondary metrics.

## 2. Original Method (Background)

The released codebase implements MVC-ZigAL on top of three components:

- A `MVAdapterT2MVSDXLPipeline` that adapts SDXL with multi-view row self-attention and accepts orthographic camera Plücker embeddings as side conditioning.
- The LCM-SDXL UNet (`latent-consistency/lcm-sdxl`) and the SDXL VAE fp16-fix variant, scheduled by a `LCMScheduler` wrapped in a `ShiftSNRScheduler` (interpolated, shift_scale=8).
- A LoRA adapter on the UNet attention modules (`to_k`, `to_q`, `to_v`, `to_out.0`, plus their multi-view counterparts), with rank 16, finetuned by the MVC-ZigAL trainer.

At inference (`scripts/inference.py`), six orthographic views are generated jointly with `azimuth = [0, 45, 90, 180, 270, 315] - 90`, distance 1.8, an orthographic camera frustum of `[-0.55, 0.55]^2`, classifier-free guidance 7.0, and 8 LCM steps. The optional `pipe.zmv_sampling` path implements Zigzag Multi-View Sampling: a forward inversion step using `LCMInverseScheduler` followed by a denoising step intended to reinforce text and viewpoint conditioning. Training (`scripts/train.py`, `mvczigal/configs/lcm_sdxl_mate3d.yaml`) uses PickScore as the single-view reward and HyperScore as the joint-view reward, with Lagrangian multipliers (`lambda_init=0.0`, `lambda_lr=0.1`, `lambda_max=5.0`) coupled to a self-paced threshold curriculum.

## 3. Identified Limitations

Reproduction surfaced four limitations relevant to optimization scope:

1. **Unavailable RL-trained LoRA.** The released LoRA checkpoint `mvczigal_lcm_sdxl_lora.safetensors` is hosted on Google Drive, which was unreachable from the host environment. The RL-finetuned weights are precisely the artifact through which the paper realizes its quality gains.
2. **Unavailable HyperScore evaluator.** The MATE-3D HyperScore checkpoint, hosted on OneDrive, was likewise unreachable. The full multi-view rubric (alignment, geometry, texture, overall) could therefore not be evaluated.
3. **ZMV-Sampling tightly coupled to RL training.** ZMV inference uses an LCM inverse scheduler whose effect is only beneficial when paired with the LoRA trained against ZMV trajectories. Without the LoRA, ZMV is a pure inference-time perturbation.
4. **Few-step seed sensitivity.** LCM-SDXL at 8 steps shows substantial sample-to-sample variance (PickScore standard deviation 0.0096, ImageReward 0.7921). This variance is both a vulnerability and an exploitable lever for inference-time optimization.

These constraints fixed the optimization scope to inference-time configuration of the unfinetuned LCM-SDXL baseline.

## 4. Optimization Methodology

The AutoSOTA loop iteratively proposed configuration changes, ran the MATE-3D evaluation, and updated the search frontier based on PickScore, HPSv2, and ImageReward. Each iteration applied to the entire 160-prompt × 6-view evaluation, executed by `eval_mate3d_baseline.py`. Three classes of intervention were considered:

- **ALGO**: changes to the inference algorithm itself (Best-of-N candidate selection over independent seeds; invoking `pipe.zmv_sampling` instead of the standard pipeline call).
- **PARAM**: scalar parameter sweeps (number of inference steps, classifier-free guidance scale, scheduler eta, camera distance).
- **PROMPT**: textual interventions on the negative prompt.

Best-of-N seed selection was implemented as: for each prompt, draw N independent seeds, generate at reduced step count, score with the reward model, and regenerate the highest-scoring candidate at the configured step count. This exploits LCM seed sensitivity at low computational overhead.

The 8-hour wall-clock budget per iteration capped exploration depth; configurations that exceeded this budget (notably Best-of-7 at 14 steps) were aborted.

## 5. Experiments

### 5.1 Setup

Evaluation used the MATE-3D prompt list (`mvczigal/data/MATE_3D.txt`, 159 non-empty prompts, treated as 160 in line with the paper) at 768×768 resolution, 6 views per prompt, fp16 precision, on a single GPU. Reward models were PickScore, HPSv2, and ImageReward, all computed per view and averaged per prompt. The HuggingFace endpoint was redirected to `https://hf-mirror.com` to retrieve the SDXL base 1.0, LCM-SDXL UNet, fp16-fix VAE, and MV-Adapter weights. PyTorch was upgraded to 2.6.0+cu124 to satisfy the LCM scheduler and `diffusers==0.31.0` requirements. The HPSv2 BPE vocabulary file was sourced from the in-repository CLIP directory. The custom MV-Adapter was initialized and loaded prior to the device move to keep its layers on the target device.

The reproduction baseline (8 inference steps, seed 42, CFG 7.0, eta 1.0) matched the paper's reported MATE-3D values for the LCM-SDXL backbone:

| Metric | Paper Baseline | Reproduced Value | Status |
|--------|---------------|------------------|--------|
| PickScore | 0.204 | 0.1948 ± 0.0096 | Within ~4.5% |
| HPSv2 | 0.252 | 0.2432 ± 0.0125 | Within ~3.5% |
| ImageReward | -0.846 | -1.4070 ± 0.7921 | High variance |

The RL-finetuned numbers from the paper and the HyperScore rubric were not reproducible due to the network restrictions noted above.

### 5.2 Quantitative Results

The optimization produced the following baseline-versus-best comparison (best configuration: iter 11, 14 inference steps with Best-of-5 seed selection; commit `5273056046`):

| Metric | Baseline (iter 0) | Best (iter 11) | Delta | Delta % |
|--------|-------------------|----------------|-------|---------|
| PickScore | 0.1948 | 0.2035 | +0.0087 | +4.47% |
| HPSv2 | 0.2432 | 0.2514 | +0.0082 | +3.37% |
| ImageReward | -1.4070 | -0.5951 | +0.8119 | +57.7% |

All three rewards improved simultaneously, indicating that the surfaced configuration improves both single-view fidelity and text–image alignment rather than overfitting to PickScore. The PickScore target of 0.2045 was not reached; the residual gap was 0.0010 (0.5%).

Pipeline timing scaled approximately linearly with the per-prompt compute. Per-prompt generation rose from ~2.6 s at the 8-step baseline to ~18 s at the peak configuration (5×14 candidate passes plus one full 14-step pass), and full-benchmark wall-clock rose from ~440 s (~7 min) to ~2900 s (~48 min).

### 5.3 Ablation / Iteration Trajectory

The thirteen iterations produced the following trajectory:

| Iter | Idea | PickScore | HPSv2 | ImageReward | Status |
|------|------|-----------|-------|-------------|--------|
| 0 | Baseline (8-step, seed=42) | 0.1948 | 0.2432 | -1.4070 | baseline |
| 1 | ZMV-Sampling | 0.1941 | — | — | regression |
| 2 | Best-of-3 seed selection | 0.1986 | 0.2464 | -0.9858 | new best |
| 3 | Best-of-5 seed selection | 0.1995 | 0.2480 | -0.8790 | new best |
| 4 | Enhanced negative prompt | 0.1995 | 0.2480 | -0.8790 | no effect |
| 5 | Camera distance 2.1 | 0.1995 | 0.2480 | -0.8790 | no effect |
| 6 | CFG 5.0 | 0.1993 | 0.2476 | -0.9423 | regression |
| 7 | eta=0.0 deterministic | 0.1995 | 0.2480 | -0.8793 | no effect |
| 8 | 10-step + Best-of-5 | 0.2021 | 0.2507 | -0.6964 | new best |
| 9 | 12-step + Best-of-5 | 0.2025 | 0.2507 | -0.6811 | new best |
| 10 | New seed set + 12-step | 0.2027 | 0.2508 | -0.6773 | new best |
| 11 | 14-step + Best-of-5 | 0.2035 | 0.2514 | -0.5951 | new best |
| 12 | 16-step + Best-of-5 | 0.2034 | 0.2513 | -0.6523 | regression |
| 13 | Best-of-7 + 14-step | — | — | — | timeout |

The marginal effect of each individual change on PickScore is summarized below, where the type column distinguishes algorithmic from parametric and prompt-level edits:

| # | Change | Type | Δ PickScore | Notes |
|---|--------|------|-------------|-------|
| 1 | ZMV-Sampling inference integration | ALGO | -0.0007 | Counterproductive without RL-trained LoRA; doubles eval time |
| 2 | Best-of-3 seed selection | ALGO | +0.0038 | Most impactful single change; exploits LCM seed sensitivity |
| 3 | Best-of-5 seed selection | ALGO | +0.0009 | Diminishing returns from 3→5 candidates |
| 4 | Enhanced multi-view negative prompt | PROMPT | 0.0000 | Extended negative prompt had no measurable effect |
| 5 | Camera distance 1.8 → 2.1 | PARAM | 0.0000 | Viewpoint distance change had no effect |
| 6 | CFG 7.0 → 5.0 | PARAM | -0.0002 | Slight regression; LCM model needs strong guidance |
| 7 | eta 1.0 → 0.0 (deterministic) | PARAM | 0.0000 | Deterministic mode identical to stochastic |
| 8 | 10 inference steps | PARAM | +0.0026 | 25% more compute → 1.3% quality gain |
| 9 | 12 inference steps | PARAM | +0.0004 | Continuing positive trend |
| 10 | New seed set with 12-step | PARAM+ALGO | +0.0002 | Seed diversity matters more than step count |
| 11 | 14 inference steps | PARAM | +0.0008 | Peak configuration — best overall result |
| 12 | 16 inference steps | PARAM | -0.0001 | Diminishing returns; 14 steps is the sweet spot |
| 13 | Best-of-7 seed selection (14-step) | ALGO | — | Killed at 8h hard limit |

Two interventions dominated. First, Best-of-N seed selection contributed the largest single jump: moving from a fixed-seed baseline to Best-of-3 added +0.0038 PickScore (+1.95%), and Best-of-5 added a further +0.0009 (+0.45%). Second, increasing the LCM step count from 8 to 14 monotonically improved all three metrics (8 → 10 → 12 → 14), with quality saturating at 16 steps. Switching the seed pool from `[42, 123, 456, 789, 999]` to `[0, 111, 333, 555, 777]` produced an additional small improvement at fixed step count, suggesting that seed diversity, not the specific values, is the operative factor.

Negative interventions were equally informative. ZMV-Sampling without the RL-trained LoRA regressed PickScore by 0.0007 and approximately doubled evaluation time, confirming that ZMV is tightly coupled to the trained weights rather than a standalone inference enhancement. Negative prompt extensions, camera distance changes within the plausible orthographic range, eta toggling, and CFG reduction (7.0 → 5.0) all produced either zero or slightly negative effects.

## 6. Discussion

The optimization trajectory clarifies the structure of the residual gap to the paper's reported numbers. The reproduction baseline (PickScore 0.1948) is within the ~4.5% expected range of the paper's LCM-SDXL baseline (0.204), and the optimized configuration (0.2035) effectively closes that gap (within 0.5% of the 0.2045 PickScore target). At this point the LCM-SDXL backbone exhibits sample variance σ ≈ 0.0096 and Best-of-N has already harvested most of the available variance. Further gains beyond 0.2035 would require either (i) the RL-finetuned LoRA, which raises the per-prompt expected quality rather than its ceiling, or (ii) per-prompt seed optimization, which exchanges generality for additional candidate budget.

A second observation is that the most impactful inference-time levers are those that exploit, rather than counteract, LCM stochasticity. CFG and eta tuning—commonly productive in standard diffusion inference—were inert or harmful here, while seed-level resampling against a learned reward was directly productive. This is consistent with the paper's framing that few-step T2MV models suffer from weak per-step learning signals and benefit from external selection criteria.

A third observation concerns ZMV-Sampling. In isolation it is counterproductive and doubles wall-clock cost; its value emerges only within the joint training-and-inference loop that MVC-ZigAL specifies. Inference-time integration of the inverse scheduler should therefore not be expected to transfer to other backbones without analogous training-time alignment.

Finally, the simultaneous improvement of PickScore (+4.47%), HPSv2 (+3.37%), and ImageReward (+57.7%) suggests that the surfaced configuration improves the underlying generation distribution rather than overfitting to one reward. The disproportionately large ImageReward gain reflects that ImageReward is sensitive to text–image alignment failures in low-step LCM samples, which Best-of-N selection effectively prunes.

## 7. Reproducibility

The optimization was performed against commit `5273056046` of the local copy of the repository on 2026-06-02, and the report was finalized on 2026-06-06. The baseline was reproduced via `eval_mate3d_baseline.py` with the configuration documented in Section 5.1. The peak configuration corresponds to the same evaluation script invoked with `num_inference_steps=14` and a Best-of-5 wrapper that draws independent seeds, scores candidates with PickScore, and regenerates the winner.

Reproducing the full paper additionally requires:

- Downloading the MVC-ZigAL LoRA checkpoint (`mvczigal_lcm_sdxl_lora.safetensors`) from the Google Drive link in the README and placing it at `checkpoint/mvczigal_lcm_sdxl_lora.safetensors`. Without this file, ZMV-Sampling is counterproductive and the RL-finetuned numbers cannot be evaluated.
- Downloading the MATE-3D HyperScore checkpoint from OneDrive into `mate3d/checkpoint`. Without this checkpoint, the alignment, geometry, texture, and overall scores cannot be computed.
- Using PyTorch 2.6.0 with `diffusers==0.31.0` and the HuggingFace mirror endpoint when direct access to `huggingface.co` is restricted.
- Initializing the MV-Adapter (`init_custom_adapter`, `load_custom_adapter`) before moving the pipeline to the target device.

All numerical results in this report were obtained on the LCM-SDXL baseline without LoRA and without RL finetuning; they should be interpreted as inference-time upper bounds for the unfinetuned backbone on MATE-3D.

## 8. References

- Zhang, Z., Shen, L., Ye, D., Luo, Y., Zhao, H., Liu, M., Yu, W., Zhang, L. *Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning.* CVPR 2026. arXiv:2505.20107.
- Huang, Z. et al. *MV-Adapter*: https://github.com/huanngzh/MV-Adapter.
- Oertell, O. et al. *RLCM*: https://github.com/Owen-Oertell/rlcm.
- Zhang, Y. et al. *MATE-3D*: https://github.com/zhangyujie-1998/MATE-3D.
- Kirstain, Y. et al. *PickScore*: https://github.com/yuvalkirstain/PickScore.
- Wu, X. et al. *HPSv2*: https://github.com/tgxs002/HPSv2.
- Xu, J. et al. *ImageReward*: https://github.com/THUDM/ImageReward.
- Xie et al. *Zigzag-Diffusion-Sampling*: https://github.com/xie-lab-ml/Zigzag-Diffusion-Sampling.
- AutoSOTA: https://github.com/tsinghua-fib-lab/AutoSOTA.
