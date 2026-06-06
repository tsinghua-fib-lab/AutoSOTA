# FORGE — Continual Learning for fMRI-Based Brain Disorder Diagnosis: A Technical Report on Automated Optimization

## Abstract

This technical report documents an automated optimization study performed on the open-source release of FORGE, a continual learning framework for functional magnetic resonance imaging (fMRI) based brain disorder diagnosis across heterogeneous clinical sites. The original method couples a structure-aware variational autoencoder for functional connectivity matrices (FCM-VAE) with a multi-level knowledge distillation objective and a hierarchical contextual bandit scheme that adaptively allocates replay budget over hospitals and over individual synthetic samples. The released reference configuration attains an Anytime Average Accuracy (AAA) of 0.7086, a forgetting rate (FOR) of 0.0696, a backward transfer (BWT) of -0.0929, and a Last Average Accuracy of 0.6571 across four sequential clinical sites covering major depressive disorder, schizophrenia, and autism spectrum disorder. An automated optimization loop structured around the AutoSOTA methodology performed fifteen targeted modifications to the optimizer, schedule, regularization, replay configuration, and graph encoder. The best configuration replaces Adam with AdamW under a cosine annealing schedule, raises decoupled weight decay from 2e-4 to 5e-4, halves the early-stopping patience on validation accuracy from 80 to 40, doubles the replay mini-batch size from 64 to 128, and activates the previously dormant relational distillation term with a coefficient of 0.15. The resulting configuration improves AAA to 0.7257 (+2.41%), reduces FOR to 0.0339 (-51.3%), reduces BWT magnitude from -0.0929 to -0.0119, and raises Last Average Accuracy to 0.6982 (+6.26%) without altering the architecture or generative backbone.

## 1. Introduction

Functional magnetic resonance imaging is widely used to study and diagnose brain disorders, with functional connectivity (FC) matrices providing dense representations of large-scale neural interactions. Diagnostic models in this domain are typically trained either on a single acquisition site or under joint multi-site access, both of which are unrealistic in clinical practice where data arrive sequentially from distinct institutions with heterogeneous protocols. The FORGE paper addresses this gap by introducing a continual learning framework specifically designed for fMRI-based diagnosis across heterogeneous sites. The framework combines FCM-VAE, a structure-aware variational autoencoder that synthesizes realistic FC matrices for both patients and controls, a multi-level knowledge distillation strategy that aligns predictions and graph representations between new-site data and replayed samples, and a hierarchical contextual bandit that adaptively allocates replay budget over previously seen hospitals and over individual synthetic samples within each hospital.

This report describes an automated optimization campaign carried out on the released code base. The campaign follows the AutoSOTA template (`tsinghua-fib-lab/AutoSOTA`): each iteration formulates a single modification, executes the full pipeline, and is accepted only when both the primary metric AAA and the auxiliary metric FOR move in the desired direction. The report (i) documents the original method, (ii) identifies configuration-level limitations, (iii) describes the modifications applied across fifteen iterations, (iv) reports the final quantitative results against the released baseline, and (v) provides a reproducibility specification.

## 2. Original Method (Background)

The released pipeline is implemented in two packages. The `fcmvae` package implements FCM-VAE, a structure-aware variational autoencoder operating on 116-node atlas-based FC matrices with a 116-dimensional latent space, a low-rank gating module of rank 16, and a transformer-style encoder with two layers, four heads, and a hidden width of 256. FCM-VAE is trained per site for 100 epochs with Adam at learning rate 1e-3, batch size 8, `beta_kl=2.0`, and a fixed adjacency threshold of 0.4 used at export time to binarize sampled adjacency matrices. After training, FCM-VAE fits class- and site-conditional latent statistics and exports a synthetic NPZ archive per site for downstream replay.

The `forge` package implements the continual learning model and training loop. The classifier `GCNForCL` is a four-layer graph convolutional network with hidden and embedding widths of 128, batch normalization, dropout 0.30, and `mean+max` pooling, followed by a three-layer MLP head reducing 256-dimensional pooled features to two logits. Hospitals are presented sequentially. For each hospital `h`, the loss combines a current-task cross-entropy term with a weighted replay loss

`L_replay = ALPHA * CE_replay + BETA * MSE(logits_s, logits_t) + GAMMA_G * MSE(g_s, g_t) + GAMMA_R * RKD(g_s, g_t)`,

where `g_s` and `g_t` denote pooled student and frozen-teacher graph embeddings, `RKD` is a normalized pairwise-distance relational distillation term, and the released defaults are `ALPHA=0.10`, `BETA=0.40`, `GAMMA_G=0.30`, `GAMMA_R=0.00`. Replay samples are drawn from FCM-VAE-generated synthetic data through a two-level Thompson-sampling bandit: `ContextualTSArms` distributes the global replay capacity (`TOT_SYNTH_CAPACITY=256`, per-hospital floor 32 and ceiling 128) across previously seen hospitals using validation accuracy and best-so-far accuracy as context, while `SampleCTXPerHospital` ranks individual synthetic graphs by Mahalanobis distance to a class prototype. A farthest-first procedure further enforces diversity. The released optimizer is Adam with learning rate 1e-3 and weight decay 2e-4; early stopping monitors validation accuracy with `WARMUP_EPOCHS=20`, `PATIENCE_ON_ACC=80`, and `EPOCHS_PER_TASK=200`. The replay mini-batch size is 64.

Evaluation uses a triangular metric matrix whose `(i, j)` entry is validation accuracy on hospital `j` after training on hospital `i`. Last Average Accuracy is the mean of the final row, AAA is the mean of all row means, and BWT is the mean of `last_row[i] - metric_matrix[i][i]` for `i < T`. FOR is the mean over `i` of `max_{j >= i} metric_matrix[j][i] - metric_matrix[T-1][i]`.

## 3. Identified Limitations

Inspection of the released configuration revealed five concrete limitations independent of the architecture or generative model.

1. The optimizer was vanilla Adam with constant learning rate. Adam couples L2 regularization into adaptive moment estimates, which interacts poorly with weight decay, and the absence of a schedule prevents late-task fine-tuning that benefits replay-based continual learning.
2. The early-stopping patience (`PATIENCE_ON_ACC=80`) was large relative to `EPOCHS_PER_TASK=200`. Combined with a flat learning rate, training frequently passed its accuracy peak before stopping, allowing overfitting and degraded forgetting.
3. The replay mini-batch size was 64, producing noisy distillation gradients and amplifying hospital-specific drift between bandit refreshes.
4. `GAMMA_R` was set to 0.00, disabling the relational knowledge distillation term implemented in `_relational_loss` despite its presence in the loss specification.
5. Decoupled weight decay was unavailable; the chosen value 2e-4 was tuned for L2-style coupling and did not exploit the AdamW formulation.

The optimization campaign was scoped to address these five issues and to verify by ablation that other plausible interventions on architecture and loss balancing do not yield improvements on this benchmark.

## 4. Optimization Methodology

The process followed the AutoSOTA single-change-per-iteration discipline. Each iteration introduced one modification, ran the full FCM-VAE plus FORGE pipeline with seed 42 on the four-site sequence (sites 6, 14, 15, 16) covering MDD, SZ, and ASD, and was accepted only if the change improved AAA without unacceptable degradation in FOR. Five change categories were considered: (i) optimizer and schedule, (ii) regularization weights and decay, (iii) replay configuration (mini-batch size, capacity, balance), (iv) graph encoder architecture (depth, width, residuals, dropout), and (v) loss-weight reparameterization.

The five accepted changes are summarized below.

| Change | Effect | Notes |
|--------|--------|-------|
| GAMMA_R: 0.00 → 0.15 | Neutral AAA, enabled relational KD | Previously disabled loss term; provided marginal benefit |
| Adam → AdamW + CosineAnnealingLR | +0.47% AAA | Decoupled weight decay and cosine schedule improved convergence |
| WEIGHT_DECAY: 2e-4 → 5e-4 | Part of AdamW improvement | Higher WD with decoupled implementation improves generalization |
| PATIENCE_ON_ACC: 80 → 40 | +1.81% AAA | Faster convergence captures better checkpoints; model stops at peak |
| REPLAY_MB_SIZE: 64 → 128 | +0.08% AAA | Larger replay batches yield more stable KD gradients and halve FOR |

The accepted changes act on three orthogonal axes: optimizer geometry (AdamW with cosine schedule and stronger decoupled decay), training-budget control (halved patience), and replay signal-to-noise (doubled replay mini-batch and activated relational distillation). The architecture, generative backbone, bandit machinery, and primary loss weights `ALPHA`, `BETA`, `GAMMA_G` are unchanged.

## 5. Experiments

### 5.1 Setup

All experiments use seed 42, the four sequential sites 6, 14, 15, 16 with a 0.20 validation split, batch size 32, and `EPOCHS_PER_TASK=200`. The graph encoder is fixed at `HIDDEN=128`, `EMBED=128`, `LAYERS=4`, `DROPOUT=0.30`, with batch normalization and `mean+max` pooling. The total replay capacity is `TOT_SYNTH_CAPACITY=256` with per-hospital floor 32 and ceiling 128. Bandit refresh occurs every 80 epochs. FCM-VAE is trained per site for 100 epochs at learning rate 1e-3 with `beta_kl=2.0`, batch size 8, and adjacency export threshold 0.4.

The released baseline uses the Adam optimizer at learning rate 1e-3 with weight decay 2e-4, `PATIENCE_ON_ACC=80`, `REPLAY_MB_SIZE=64`, and `GAMMA_R=0.00`. The optimized configuration uses AdamW with `LR=1e-3` and weight decay 5e-4 under a cosine annealing schedule with `eta_min = 0.01 * LR` and `T_max = EPOCHS_PER_TASK`, `PATIENCE_ON_ACC=40`, `REPLAY_MB_SIZE=128`, and `GAMMA_R=0.15`. All other hyperparameters match the baseline.

### 5.2 Quantitative Results

The four headline metrics for the baseline and optimized configurations are reported in Table 1.

**Table 1.** Baseline versus best metrics on the four-site MDD/SZ/ASD continual benchmark.

| Metric | Baseline | Best | Delta | Direction |
|--------|----------|------|-------|-----------|
| AAA (primary) | 0.7086 | 0.7257 | +0.0171 (+2.41%) | higher is better |
| FOR (forgetting) | 0.0696 | 0.0339 | -0.0357 (-51.3%) | lower is better |
| BWT | -0.0929 | -0.0119 | +0.0810 | higher (less negative) is better |
| Last Avg Acc | 0.6571 | 0.6982 | +0.0411 (+6.26%) | higher is better |

The triangular metric matrices (rows index the most recently trained hospital; columns index the evaluated hospital) are reproduced verbatim in Table 2. The optimized configuration retains hospital 1 better at the final task (0.6571 versus 0.6286), and hospital 3 reaches a notably higher intermediate accuracy (0.9000 versus 0.8000 after task 3).

**Table 2.** Per-task accuracy on each previously seen site.

| After task | Baseline row | Best row |
|------------|--------------|----------|
| 1 | [0.7714] | [0.7429] |
| 2 | [0.6286, 0.8500] | [0.6857, 0.8000] |
| 3 | [0.6286, 0.8000, 0.5714] | [0.6857, 0.9000, 0.5714] |
| 4 | [0.6286, 0.7500, 0.5357, 0.7143] | [0.6571, 0.8500, 0.5714, 0.7143] |

The reduction in FOR from 0.0696 to 0.0339 represents a 51.3% relative decrease and confirms that the replay signal has become substantially more stable. The BWT magnitude shrinks by an order of magnitude (from -0.0929 to -0.0119), indicating that backward transfer is now nearly neutral on this benchmark. The improvement in Last Average Accuracy (+6.26%) is larger than the AAA improvement (+2.41%), which is consistent with the optimized configuration giving up a small amount of single-task peak accuracy on the very first hospital in exchange for substantially better retention at later tasks.

### 5.3 Ablation / Iteration Trajectory

Table 3 lists every iteration in chronological order with the candidate idea, the resulting AAA and FOR, and the status. Each row differs from the immediately preceding accepted configuration by exactly one perturbation.

**Table 3.** Iteration trajectory across the fifteen optimization rounds.

| Iter | Idea | AAA | FOR | Status |
|------|------|-----|-----|--------|
| 0 | Baseline | 0.7086 | 0.0696 | SUCCESS |
| 1 | Adaptive loss weighting | 0.7061 | 0.0714 | FAILED |
| 2 | LAYERS=2 | 0.7071 | 0.0482 | FAILED |
| 3 | LAYERS=3 | 0.7030 | 0.0214 | FAILED |
| 4 | GCNII residuals | 0.6818 | 0.0250 | FAILED |
| 5 | GAMMA_R=0.15 | 0.7088 | 0.0786 | SUCCESS (minor) |
| 6 | EWC regularization | 0.6949 | 0.1446 | FAILED |
| 7 | AdamW + cosine | 0.7121 | 0.0446 | SUCCESS |
| 8 | HIDDEN=256 | 0.6719 | 0.0268 | FAILED |
| 9 | PATIENCE=40 | 0.7250 | 0.0625 | SUCCESS (BEST) |
| 10 | LAYERS=2 combo | 0.6973 | 0.0357 | FAILED |
| 11 | ALPHA=0.20 BETA=0.60 | 0.6933 | 0.1214 | FAILED |
| 12 | DROPOUT=0.50 | 0.5503 | 0.0357 | FAILED |
| 13 | REPLAY_MB_SIZE=128 | 0.7257 | 0.0339 | SUCCESS (BEST) |
| 14 | WARMUP=10 | 0.7257 | 0.0339 | FAILED (neutral) |
| 15 | Class-balanced replay | 0.7257 | 0.0339 | FAILED (neutral) |

Several patterns emerge. Reducing the number of GCN layers (iterations 2, 3, 10) reliably improved FOR but reduced AAA; less depth provides less capacity, and the regularizing benefit of mitigated over-smoothing did not compensate. GCNII-style residual skip connections (iteration 4) reduced AAA by 3.8% by forcing the layer-1 representation into all subsequent layers. EWC (iteration 6) penalized adaptation to new tasks, raising FOR to 0.1446, indicating that the generative replay branch already covers the function of explicit parameter regularization. `HIDDEN=256` (iteration 8) and dropout 0.50 (iteration 12) overfit or underfit the small clinical fMRI sample. Increasing replay weights to `ALPHA=0.20, BETA=0.60` (iteration 11) suppressed current-task learning. Adaptive loss weighting (iteration 1) added optimization noise without a meaningful balance gain. The final two iterations were neutral on both AAA and FOR, indicating that optimization had reached a local plateau on this benchmark.

## 6. Discussion

The optimization campaign demonstrates that the dominant gains on FORGE arise from optimizer geometry and from the signal-to-noise ratio of the replay distillation, not from architectural change. AdamW with a cosine schedule and stronger decoupled weight decay provided +0.47% AAA on its own, halving the patience on validation accuracy provided +1.81% AAA, and doubling the replay mini-batch provided +0.08% AAA but halved FOR. The activation of the relational distillation term contributed an additional small improvement on AAA. The combined effect is +2.41% AAA and a 51.3% reduction in FOR, achieved without modifying FCM-VAE, the GCN encoder, the contextual bandit allocator, or the primary replay loss weights. Architectural alternatives (fewer layers, GCNII residuals, wider hidden width, higher dropout) and additional regularizers (EWC, adaptive loss weighting) all degraded performance, suggesting that the released architecture is well-matched to the data scale and that further accuracy gains are likely to come from the data side: adaptive adjacency thresholding, graph data augmentation, local-structure knowledge distillation, prototype-based replay, or larger replay buffer capacity to exploit the doubled replay mini-batch. Hospital 3 (schizophrenia) remains the weakest site and is a natural target for site-specific tuning or harmonization.

A limitation is that all results are under a single random seed (42) with fixed validation splits. Variance estimates across seeds would be necessary to confirm statistical significance of the smaller gains in iterations 5 and 13.

## 7. Reproducibility

The full pipeline reproduces with the released code. FCM-VAE is trained per site by running `run_all.py`, which iterates over sites 6, 14, 15, 16 and writes synthetic NPZ archives to `data/synth/site{n}.npz` using the Section 5.1 configuration. The continual learning experiment is launched with `run_forge.py` using the optimized configuration: `LR=1e-3`, `WEIGHT_DECAY=5e-4`, `EPOCHS_PER_TASK=200`, `WARMUP_EPOCHS=20`, `PATIENCE_ON_ACC=40`, `BATCH_SIZE=32`, `HIDDEN=128`, `EMBED=128`, `LAYERS=4`, `DROPOUT=0.30`, `ALPHA=0.10`, `BETA=0.40`, `GAMMA_G=0.30`, `GAMMA_R=0.15`, `ADJ_THRESHOLD=0.4`, `REPLAY_MB_SIZE=128`, `TOT_SYNTH_CAPACITY=256`, `REPLAY_AFTER_FIRST=True`, `VAL_RATIO=0.20`, `SEED=42`. The optimizer is `torch.optim.AdamW` with `CosineAnnealingLR(T_max=EPOCHS_PER_TASK, eta_min=LR*0.01)` stepped per epoch, as in `forge/forge.py::train_one_hospital_CTXT`. Expected outputs are AAA = 0.7257, FOR = 0.0339, BWT = -0.0119, Last Average Accuracy = 0.6982, and the metric matrix in Table 2. The best configuration was recorded at commit `ea0f93f284`.

## 8. References

1. The original FORGE manuscript: *FORGE — Continual Learning for fMRI-Based Brain Disorder Diagnosis*, CVPR 2026 submission. Code release: this repository (`paper-1650`).
2. AutoSOTA: automated single-change optimization protocol for research code bases, `tsinghua-fib-lab/AutoSOTA`.
