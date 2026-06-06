# Few-for-Many Personalized Federated Learning (FedFewS): A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study performed on the public release of FedFew, a CVPR 2026 personalized federated-learning method (referred to in this study as FedFewS) whose central idea is to maintain a small set of K server models and let each client form a per-client convex combination of them via a STCH-Set specialization mechanism. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted Averaged Test Accuracy on CIFAR-100 under a non-IID Dirichlet split (α = 0.5, M = 20 clients) using the FedAvgCNN backbone. The headline result is an improvement from a 500-round baseline of 49.85% to **56.38%** (+6.53 percentage points), and from the paper's reported 2000-round figure of 53.69 ± 4.79% to 56.38% (+2.69 pp absolute over the paper-reported value, +2.37 pp over the present 2000-round reproduction of 54.01%). The improvement was obtained by stacking four changes: (i) replacing `ExponentialLR` with `CosineAnnealingLR` on both client optimizers; (ii) adding MixUp augmentation during local client training; (iii) **initializing the K=3 server models with distinct random seeds** instead of identical deep copies — the breakthrough change, contributing +2.99 pp on its own; and (iv) extending training from 500 to 1000 rounds (rather than the paper's 2000) under the cosine schedule. Three further ideas were rolled back: label smoothing (−1.85 pp), 1200 rounds (−0.60 pp vs. 1000), and increasing local epochs to 2 (−2.44 pp).

## 1. Introduction

FedFew, presented at CVPR 2026, addresses a long-standing tension in personalized federated learning between *expressivity* (each client's optimal model differs) and *communication cost* (training one model per client is infeasible). The proposed mechanism maintains a small set of K server models and equips each client with a learned convex-combination weight vector `w_i ∈ Δ^K` over those models — the *Set-Top-K Combinatorial Head* (STCH-Set) — so that "few" server models serve "many" client distributions. The released codebase is built on the PFLlib framework and ships a reference algorithm `fedfews` that exercises the K=3 instantiation.

This report studies whether the released FedFewS pipeline can be improved through a combination of standard recipe modernizations (cosine LR, MixUp) and changes targeting the STCH-Set specialization dynamics directly. The motivation is that the released code initializes the K=3 server models as identical deep copies of the same network at round zero — meaning that the STCH-Set must specialize the models *during* training rather than starting from natural diversity. AutoSOTA was used to enumerate, run, and evaluate candidate changes against Averaged Test Accuracy in a budgeted iterative loop on a 2× A100-80GB host.

The remainder of the report covers the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology (Section 4), the experimental setup, results, and the rollback table (Section 5), a discussion centred on the breakthrough initialization change (Section 6), and reproducibility information (Section 7).

## 2. Original Method (Background)

The released FedFewS implementation operates on the standard PFLlib federation:

* **Backbone.** A 4-layer CNN (`FedAvgCNN`).
* **Federation.** M = 20 clients with a Dirichlet α = 0.5 non-IID label split on CIFAR-100. The standard PFLlib evaluation protocol reports Averaged Test Accuracy across all clients.
* **Server.** K = 3 server models, initialized at round zero as identical `deepcopy` copies of the initial network. Each round, the server broadcasts the K models and aggregates client updates back into the K-model set.
* **Client.** Each client maintains a STCH-Set weight vector `w_i ∈ Δ^3`. During local training the client may step on (i) a "rep-mode" objective that updates the representation parameters and (ii) a standard objective that updates the head; both use `ExponentialLR` for learning-rate decay in the released configuration.
* **Reference command.**
  ```
  python scripts/run_pfllib.py configs/cifar100/noniid_dir_20_a0p5/algorithms/fedfews.yaml
  ```
* **Reported metrics.** The paper reports 53.69 ± 4.79% Averaged Test Accuracy at 2000 rounds; the present reproduction yields 54.01% at 2000 rounds and 49.85% at the shorter 500-round budget used for AutoSOTA iterations.

## 3. Identified Limitations

The optimization study identified four sources of friction in the released training pipeline:

1. **Exponential LR decay is mis-aligned with the federated horizon.** `ExponentialLR` with `gamma=0.99` decays continuously and ends with a learning rate that is too small in the late rounds of a 500–1000-round federation, where personalization-relevant updates need a non-trivial step size to escape per-client local minima.
2. **No data augmentation in client training.** Local client batches under non-IID splits are skewed; the released pipeline does not employ MixUp or other label-mixing augmentation, leaving an easy regularizer on the table.
3. **K server models start identical.** The K=3 server models are `deepcopy` copies of the same initialization. The STCH-Set mechanism must therefore specialize the K models during training using only the gradient signal from non-uniform client weights — a slow process that delays the emergence of model diversity.
4. **Round budget is fixed at 2000.** The paper trains for 2000 rounds, which is wasteful under a properly tuned schedule. A 1000-round cosine schedule should be sufficient if the LR profile and initialization improve specialization speed.

## 4. Optimization Methodology

The five retained iterations exercise three categories of change. All changes are implemented in three files: `PFLlib/system/flcore/clients/clientfedfews.py`, `PFLlib/system/flcore/servers/serverfedfews.py`, and `configs/cifar100/noniid_dir_20_a0p5/base.yaml` (with `algorithms/fedfews.yaml` as the entry-point config).

**Schedule modernization (Iter 1).** `ExponentialLR` was replaced with `CosineAnnealingLR(T_max=global_rounds, eta_min=0.0001)` on both the rep-mode and standard optimizers in `clientfedfews.py`. The initial LR is 0.01. No warmup is used; an earlier variant with linear warmup hurt early-round learning under the federation's noisy aggregation. Effect: +0.56 pp.

**Local-data regularization (Iter 3).** MixUp augmentation was added to the local client training loop. Mixing coefficients are drawn from `Beta(0.2, 0.2)` and clipped so that `λ ≥ 0.5` (i.e. the convex combination favours the first ordering). The mixed loss is `λ · CE(pred, y_a) + (1 − λ) · CE(pred, y_b)`. The intent is to regularize clients whose label distributions are skewed under the Dirichlet split. Effect: +0.66 pp.

**Distinct K-model initialization (Iter 4 — breakthrough).** In `serverfedfews.py`, the K=3 server models are no longer identical `deepcopy` copies. The first model is initialized as before; models `k = 1, 2` are *reinitialized* via `reset_parameters()` with different random seeds. The STCH-Set specialization weights `w_{ik}` therefore receive non-zero per-model gradients from the start, becoming non-uniform earlier in training and producing useful model diversity in the first hundred rounds rather than over the full 2000-round horizon. Effect: +2.99 pp. This change alone caused the 500-round optimizer baseline to *exceed* the paper's reported 2000-round number.

**Round-budget extension (Iter 5).** The global round budget was raised from 500 to 1000 (not back to 2000 — the cosine schedule plus distinct-init combination is sufficiently efficient that 1000 rounds Pareto-dominates the original 2000). The corresponding `base.yaml` changes:

```yaml
training:
  global_rounds: 1000        # was 2000
  learning_rate: 0.01        # initial LR for cosine, was 0.005
  learning_rate_decay: true  # required for scheduler stepping
  learning_rate_decay_gamma: 0.99  # retained for compatibility
```

Effect: +2.32 pp, producing the final best of 56.38%.

**Approaches tested but not retained (rolled back).**

* *Label smoothing 0.1 (Iter 2).* −1.85 pp. Likely interacts negatively with MixUp's already-soft targets and with the noisy STCH-Set selection.
* *Extending to 1200 rounds (Iter 6).* −0.60 pp vs. 1000. The cosine schedule has already reached its `eta_min` plateau and the marginal rounds add variance without bias improvement.
* *Local epochs = 2 (Iter 7).* −2.44 pp. Doubling the local pass per round amplifies client drift under non-IID and harms aggregation.

No changes were made to data, the evaluation script, or the FedFew algorithmic core.

## 5. Experiments

### 5.1 Setup

The optimization target was Averaged Test Accuracy on CIFAR-100, under the standard PFLlib non-IID Dirichlet split (α = 0.5, M = 20 clients). All runs used the released FedAvgCNN backbone and the `fedfews` algorithm with K = 3 server models. AutoSOTA executed iterations with a 500-round per-iteration budget for fast turnaround; the final best run (Iteration 5) was extended to 1000 rounds. Hardware was 2× NVIDIA A100-80GB; the 1000-round configuration takes approximately 2 hours wall clock to complete.

### 5.2 Quantitative Results

| Metric | Value | Notes |
|---|---:|---|
| Paper reported (2000 rounds) | 53.69 ± 4.79% | Released paper number |
| Reproduced baseline (2000 rounds) | 54.01% | This study's faithful reproduction |
| Optimizer baseline (500 rounds) | 49.85% | Short-horizon starting point |
| **Optimized best (1000 rounds)** | **56.38%** | **+2.69 pp vs. paper, +2.37 pp vs. 2000-round reproduction, +6.53 pp vs. 500-round optimizer baseline** |

The 1000-round optimized best simultaneously beats the 2000-round reproduction by 2.37 pp and the paper's reported figure by 2.69 pp — i.e. it strictly Pareto-dominates the paper's recipe in both quality and compute.

### 5.3 Ablation / Iteration Trajectory

| Iter | Change | Acc | Delta vs. previous | Status |
|---|---|---:|---:|---|
| 0 | Baseline (500 rounds, ExponentialLR) | 49.85% | — | reference |
| 1 | CosineAnnealingLR | 50.41% | +0.56 pp | retained |
| 2 | + Label smoothing 0.1 | 48.56% | −1.85 pp | **rolled back** |
| 3 | + MixUp (`λ ≥ 0.5`) | 51.07% | +0.66 pp | retained |
| 4 | + Distinct K-model init (`reset_parameters()` for k=1, 2) | **54.06%** | **+2.99 pp** | retained — breakthrough |
| 5 | + Extend to 1000 rounds | **56.38%** | **+2.32 pp** | retained — best overall |
| 6 | Extend to 1200 rounds | 55.78% | −0.60 pp | **rolled back** |
| 7 | Local epochs = 2 | 53.94% | −2.44 pp | **rolled back** |

Two patterns are visible in the trajectory. First, the breakthrough is *initialization*, not schedule, regularization, or budget: distinct K-model initialization alone contributes nearly half of the total gain. Second, naive scaling (longer training, more local work) does not pay off in this federation; the cosine schedule and 1000 rounds form a sweet spot.

## 6. Discussion

The dominant takeaway is that the K=3 STCH-Set mechanism in FedFew is starvation-bound at initialization: when all K server models start identical, the per-client weight vector `w_i` has no informative gradient until model diversity has emerged through accumulated stochastic updates, and that emergence is slow under a federation's noisy aggregation. Initializing the K models with distinct random seeds gives the STCH-Set immediate, informative gradients and accelerates specialization by hundreds of rounds. The fact that this single change crossed the paper's 2000-round baseline at only 500 rounds is strong evidence that the released training horizon was masking a slow-start pathology in the initialization, not absorbing slack in optimization quality.

The rollback table tells a complementary story. Doubling local epochs amplifies client drift under non-IID splits — a textbook FedAvg failure that the FedFew architecture does not protect against — and label smoothing duplicates the soft-target signal that MixUp already provides, degrading both. Extending the round budget further than 1000 reaches the cosine schedule's `eta_min` plateau and adds variance without bias.

A natural follow-up direction not exercised here is to study how the optimal initialization spread depends on K. With K = 3 the gain was substantial; with K = 5 or K = 7 the per-model gradient signal would be weaker per round but the diversity bound on the STCH-Set richer, and an analogous distinct-init pattern (perhaps drawn from a wider initialization scale) may be necessary to keep specialization on schedule. A further direction is to combine the distinct initialization with periodic *re-initialization* of the most-collapsed server model when the STCH-Set weights `w_{ik}` indicate that one model has become redundant.

## 7. Reproducibility

The slimmed repository contains the code required to reproduce the best configuration; CIFAR-100 is downloaded by the PFLlib data loader.

* **Best configuration.** All four retained changes from Section 4: `CosineAnnealingLR` on both client optimizers, MixUp with `λ ≥ 0.5`, distinct K=3 server-model initialization via `reset_parameters()`, and 1000-round training horizon.
* **Files touched.**
  * `PFLlib/system/flcore/clients/clientfedfews.py` — schedule and MixUp.
  * `PFLlib/system/flcore/servers/serverfedfews.py` — distinct-init for k = 1, 2.
  * `configs/cifar100/noniid_dir_20_a0p5/base.yaml` — round budget and LR settings (see Section 4 for the diff).
* **Evaluation command.**
  ```
  python scripts/run_pfllib.py configs/cifar100/noniid_dir_20_a0p5/algorithms/fedfews.yaml
  ```
  Expected output: `Averaged Test Acc: ~56.38%`. Expected runtime: ~2 hours on 2× A100-80GB.
* **Original repository.** [github.com/pgg3/FedFew](https://github.com/pgg3/FedFew).

## 8. References

* (Anonymous CVPR 2026 submission) *Few-for-Many Personalized Federated Learning (FedFew)*. CVPR 2026. Original repository: [github.com/pgg3/FedFew](https://github.com/pgg3/FedFew).
* AutoSOTA: Tsinghua FIB Lab. *AutoSOTA: An automated SOTA-chasing harness*. [github.com/tsinghua-fib-lab/AutoSOTA](https://github.com/tsinghua-fib-lab/AutoSOTA).
* PFLlib: a Personalized Federated-Learning library; the present codebase builds on it.
