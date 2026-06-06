# Rethinking SNN Online Training and Deployment — Gradient-Coherent Learning via Hybrid-Driven LIF Model: A Technical Report on Automated Optimization

## Abstract

This report documents a partially complete automated optimization study performed on the public release of the HD-LIF (Hybrid-Driven Leaky Integrate-and-Fire) spiking neural network framework (Hao et al., CVPR 2026), which introduces gradient-coherent learning via single-timestep backpropagation through randomly selected steps. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted `cifar100_top1_accuracy` on CIFAR-100. The headline result is an improvement from a reproduced baseline of 78.79% Top-1 accuracy to 78.93% (+0.14 absolute), obtained in a single successful iteration by enabling the `--use_eca` flag, which activates an existing but disabled ECA (Efficient Channel Attention) module in `resnet.py`. The optimization terminated when the 8-hour wall-clock budget was exhausted partway through Iteration 2 (a combined ECA + label smoothing + LR-warmup run), which had reached only epoch 27/300 (51.11%) when killed. The 5%-relative improvement target of 82.50% was therefore not reached. The headline takeaway is that, on the released HD-LIF baseline, the single per-iteration training cost (~3.5 hours per 300-epoch CIFAR-100 run) is the dominant constraint, not the supply of optimization ideas — 21 candidate levers were enumerated and red-line audited, but only one could be evaluated within budget. The best configuration is captured at commit `adf4b332698b7120b3904932a721ce32368e95a4`.

## 1. Introduction

HD-LIF, presented at CVPR 2026, addresses two long-standing pain points of spiking neural network (SNN) training: the prohibitive memory cost of backpropagating through time across all simulated timesteps, and the gradient inconsistency between timesteps that hinders convergence. The proposed solution is twofold: a hybrid-driven LIF neuron (the `OnlineNeuron`) that decouples temporal dynamics from gradient flow, and a gradient-coherent training procedure (`opt_backprop`) that backpropagates through one randomly selected timestep per minibatch. Together with 4-bit spike quantization and ternary weights (`compression` mode with `--use_ter`), the framework achieves competitive CIFAR-100 accuracy with substantially reduced memory and energy footprints relative to standard surrogate-gradient SNNs.

This report studies whether the released HD-LIF training pipeline can be improved through automated test-time and recipe-level interventions. The motivation is that recent advances in attention modules, label smoothing, learning-rate warmup, and weight averaging are well known to improve standard CNN training but have not been systematically exercised against a quantized spiking ResNet-18 trained with single-timestep backpropagation. AutoSOTA was used to enumerate, audit, and execute candidate changes against the `cifar100_top1_accuracy` metric in an 8-hour budget.

The remainder of the report covers the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology and the broader idea library (Section 4), the experimental setup and results (Section 5), a discussion focused on the cost ceiling that bounded the study (Section 6), and reproducibility information (Section 7).

## 2. Original Method (Background)

The HD-LIF model used in this study is a ResNet-18 backbone (`resnet.py`) in which the standard ReLU activations are replaced by a custom `OnlineNeuron` LIF unit. Three components define the released training recipe:

* **`OnlineNeuron` (Hybrid-Driven LIF).** A LIF neuron implementation that maintains the temporal membrane dynamics in the forward pass while structuring the backward pass to be stable under the gradient-coherent rule.
* **`opt_backprop` (Gradient-Coherent Learning).** Instead of backpropagating through every simulated timestep `t = 1..T`, a single timestep is sampled uniformly at random per minibatch and backprop is performed only through that step. This trades full BPTT fidelity for a substantial memory reduction and an empirical improvement in gradient stability.
* **Compression mode.** With `--use_ter`, weights are ternarized and the spike outputs are quantized to 4 bits (`--mode compression`).

The canonical training/evaluation command in the released codebase is:

```
python main.py --dataset CIFAR100 --mode compression --use_ter --mixup --opt_backprop --use_parallel --amp
```

A configuration parameter `--beta` is referenced in `config.yaml` but does not exist in `main.py`; the working configuration ignores this knob. The result extraction protocol is `grep "Best Acc" <log_file> | tail -1` from the per-experiment log subdirectory. Pretrained weights are not used; the model is trained from random initialization for 300 epochs (~3.5 hours on the available 2-GPU configuration).

## 3. Identified Limitations

The optimization study identified four sources of friction in the released training and evaluation pipeline:

1. **Latent ECA module.** `resnet.py` already implements an `ECAAttention` block but routes around it by default; the `--use_eca 1` flag is the only switch needed to enable it. This is a free lever in the sense that no new code is required, only configuration.
2. **Default training recipe lacks modern stabilizers.** Standard CNN-training tricks — label smoothing in `CrossEntropyLoss`, linear LR warmup, EMA of weights, AdamW in place of SGD — are not used in the released training script. Each is a candidate +0.1–0.5% gain on CIFAR-100 in standard backbones.
3. **Aggressive single-step backprop.** `opt_backprop` reduces the gradient signal to a single timestep per batch, which is by construction a high-variance estimator of the full BPTT gradient. Backpropagating through `k > 1` randomly selected timesteps would lower variance at proportional memory cost.
4. **Iteration cost dominates.** A 300-epoch run takes ~3.5 hours under the available infrastructure, allowing roughly two full evaluations per 8-hour pipeline window. The cost ceiling is the dominant practical constraint and bounds the breadth of any optimization study far more than the supply of ideas.

In addition, several environment-level frictions had to be repaired before any iteration could complete: `docker exec` stdout was unreliable on the available container image, the image had an empty `/repo/`, the container's `TMPDIR` did not exist, and the initial `git clone` failed with TLS handshake errors that required retrying from `/tmp`.

## 4. Optimization Methodology

Twenty-one candidate ideas were enumerated, organized into a tiered library, and audited against six red lines (R1–R6: no metric redefinition, no metric computation modification, no hard-coding outputs, no sacrificing other metrics, no train/test contamination, no dataset modification). Of these, **one** was evaluated to completion (Iteration 1) and **one** was started but killed by the 8-hour wall-clock cap (Iteration 2).

**Iteration 1 — IDEA-001: Enable ECA Channel Attention.** The `--use_eca 1` flag activates the already-implemented `ECAAttention` module in `resnet.py`, adding ~5 parameters per layer and negligible compute. No new code is introduced; the change is a single CLI flag.

**Iteration 2 — IDEA-014: Combined ECA + Label Smoothing + LR Warmup.** Three changes were combined in one run: `--use_eca 1` retained from Iteration 1; `label_smoothing=0.1` added to `CrossEntropyLoss`; and `CosineAnnealingLR` replaced with `SequentialLR(LinearLR warmup 5 epochs + CosineAnnealingLR)`. The run was terminated at epoch 27/300 (~51.11% intermediate accuracy) when the 8-hour pipeline limit fired, so no comparable end-of-training metric was recorded.

**Idea library (not evaluated).** Nineteen further ideas remained in queue when the budget expired. They are catalogued in the takeaway log and grouped by tier:

* *Tier 1 — high-impact:* EMA of weights (IDEA-004); backprop through multiple random timesteps (IDEA-019, addressing the single-step variance issue directly); AdamW in place of SGD (IDEA-006); enable membrane BatchNorm `--use_mem_bn` (IDEA-008).
* *Tier 2 — code-level:* RandAugment (IDEA-009); standalone label smoothing (IDEA-002); standalone LR warmup (IDEA-003); Mixup hyperparameter tuning (IDEA-010); gradient accumulation (IDEA-011); cosine warm restarts (IDEA-012); SE blocks (IDEA-016); parameter-group-specific weight decay (IDEA-017); stochastic depth (IDEA-020).
* *Tier 3 — parameter tuning:* train for 400 epochs (IDEA-005); increase SNN timesteps `T=6` (IDEA-007); batch size 128 (IDEA-018); `T=6` with proportional epoch reduction (IDEA-021).
* *Higher-risk architecture changes:* surrogate-gradient `LIFNeuron` (IDEA-013); VGG-13 backbone (IDEA-015).

In addition, ten transferable patterns from the AutoSOTA IdeaPool were mapped to HD-LIF — Learned Attention (mapped to ECA), Knowledge Distillation (ANN→SNN teacher), Sharpness-Aware Minimization, Label Smoothing, Modern Training Recipes (ConvNeXt), Data-Augmentation Search, Weight Averaging (Model Soup), Coarse-to-Fine (mapped to multi-timestep backprop), Stochastic Depth, and Multi-Scale Temporal (mapped to larger `T`). Only the first pattern was evaluated.

No data, evaluation script, or metric definition was modified. All twenty-one ideas pass the R1–R6 red-line checks.

## 5. Experiments

### 5.1 Setup

The optimization target was `cifar100_top1_accuracy` on CIFAR-100, computed by the released training script's standard validation pipeline. The training command is the canonical compressed-mode command listed in Section 2. Each iteration trained for 300 epochs from random initialization on two GPUs (CUDA IDs 0 and 1) inside a Docker container provisioned by AutoSOTA. A single iteration took approximately 3.5 hours of wall clock; the pipeline was capped at 8 hours, allowing roughly two full evaluations per session. The CIFAR-100 dataset was auto-downloaded via `torchvision`; the repository was cloned at runtime from `https://github.com/hzc1208/HD_LIF` because the Docker image shipped with an empty `/repo/`.

The improvement target set by AutoSOTA was 82.50% Top-1 (+5% relative over the 78.57% reproduction-log baseline).

### 5.2 Quantitative Results

| Metric | Paper | Reproduced (log) | Optimizer Baseline (Iter 0) | Best (Iter 1) | Delta |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 Top-1 | 78.45% | 78.57% | 78.79% | **78.93%** | **+0.14** |

The reproduced baseline (78.79%) modestly exceeds both the paper-reported figure (78.45%) and the original reproduction-log value (78.57%); this is consistent with run-to-run variance on a stochastic SNN training pipeline rather than with any deliberate change. The +0.14% improvement at Iteration 1 was obtained at near-zero compute overhead.

The 5%-over-baseline target of 82.50% was not reached and remains 3.57 percentage points away. At the observed +0.14% per ~3.5 h experiment, reaching the target by accumulating ECA-class wins alone would require on the order of 25 successful iterations and substantially more than 8 hours of wall clock.

### 5.3 Ablation / Iteration Trajectory

| Iter | Idea | Type | Before | After | Delta | Status | Note |
|---|---|---|---:|---:|---:|---|---|
| 0 | Baseline reproduction | — | — | 78.79% | — | success | Above paper (+0.34) and repro log (+0.22) |
| 1 | IDEA-001: enable ECA channel attention (`--use_eca 1`) | ALGO | 78.79% | **78.93%** | **+0.14%** | success | New best |
| 2 | IDEA-014: ECA + label smoothing 0.1 + LinearLR(5)+CosineAnnealing warmup | ALGO | 78.93% | — | — | killed | Stopped at epoch 27/300 (~51.11%) at 8 h limit |

## 6. Discussion

The principal lesson of this study is structural rather than algorithmic: when each iteration costs 3.5 hours and the optimization budget is 8 hours, the breadth of accessible levers is bounded by experiment cost rather than by idea supply. Twenty-one ideas were enumerated and red-line audited; only one was completed. The +0.14% gain from enabling a latent module that was already in the codebase is a useful but small data point and should not be over-interpreted as evidence about the relative merit of attention-style augmentation versus the alternatives in the queue (multi-step backprop, EMA, AdamW).

A second observation is that several of the most promising un-evaluated ideas target the gradient-coherent rule itself rather than the surrounding training recipe. IDEA-019 (backprop through multiple random timesteps) directly addresses the high variance of the single-step gradient estimator and is the candidate most likely to produce an architectural improvement rather than a recipe-level improvement. Knowledge distillation from a strong ANN teacher (IdeaPool Pattern 2) is similarly aligned with the released framework's underlying constraint that SNN gradients are noisy.

A third observation concerns infrastructure. Significant pipeline time was spent repairing the Docker environment — broken `docker exec` stdout, empty `/repo/`, missing `TMPDIR`, TLS-failing initial `git clone` — before productive iteration could begin. For continuation runs, a longer (24 h+) wall-clock budget and a pre-baked container image with the repository, dataset cache, and a working `record_score.sh` already in place would be the highest-impact infrastructure improvements.

## 7. Reproducibility

The slimmed repository contains the code required to reproduce the best configuration; the CIFAR-100 dataset and training logs are intentionally not included.

* **Best commit.** `adf4b332698b7120b3904932a721ce32368e95a4`.
* **Best configuration.** Enable ECA channel attention via `--use_eca 1` on top of the canonical compressed-mode HD-LIF command. All other hyperparameters at their released values.
* **Canonical evaluation command.**
  ```
  python main.py --dataset CIFAR100 --mode compression --use_ter --mixup --opt_backprop --use_parallel --amp --use_eca 1
  ```
* **Result extraction.** `grep "Best Acc" <log_file> | tail -1`, where `<log_file>` is the per-experiment log file in the run directory.
* **Known config issue.** The `--beta` parameter referenced in `config.yaml` does not exist in `main.py` and should be ignored or removed; do not use it as an evaluation lever.
* **Hardware used in this study.** Two GPUs (IDs 0 and 1) inside a Docker container `paper_opt_paper-2274`, ~3.5 h per 300-epoch CIFAR-100 run.
* **Original repository.** [github.com/hzc1208/HD_LIF](https://github.com/hzc1208/HD_LIF).

## 8. References

* Hao et al. (2026). *Rethinking SNN Online Training and Deployment — Gradient-Coherent Learning via Hybrid-Driven LIF Model*. CVPR 2026.
* AutoSOTA: Tsinghua FIB Lab. *AutoSOTA: An automated SOTA-chasing harness*. [github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA).
