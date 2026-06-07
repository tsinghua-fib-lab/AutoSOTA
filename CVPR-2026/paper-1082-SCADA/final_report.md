# Optimization Results: SCADA — Source Models Leak What They Shouldn't

## Summary
- **Total iterations**: 20 (0 baseline + 19 experiments + 1 final)
- **Best `score`**: **0.777** (baseline: 0.775, improvement: +0.26%)
- **Best `adt_r`**: 77.67 (baseline: 77.52)
- **Best `adt_f`**: 0.0 (maintained perfect unlearning throughout)
- **Best commit**: `b41180db8a` (iter 14)
- **Target**: 0.819 (not reached — gap of 5.4%)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| adt_r | 77.52 | 77.67 | +0.15 |
| adt_f | 0.0 | 0.0 | 0.0 |
| score | 0.775 | 0.777 | +0.002 (+0.26%) |

## Key Changes Applied

| Change | Type | Effect | Notes |
|--------|------|--------|-------|
| **Gradient clipping** (max_norm=1.0) | ALGO | +0.001 score | Added training stability; marginal but positive |
| **Increased epochs** (5→10) | CONFIG | +0.002 score | Most impactful single change; 10 epochs is sweet spot |

## What Worked

1. **Gradient Clipping** (+0.06%): Small but consistent improvement. Added `torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)` in the minimax training loop. This is standard for adversarial training.

2. **Increased Training Epochs** (5→10, +0.10%): The most impactful change. 5 epochs was insufficient for ViT-B/16 convergence. 10 epochs improved retain accuracy by ~0.2% on average. However, 15 epochs showed overfitting (score decreased to 0.774).

## What Didn't Work

1. **Cosine Annealing LR** — Catastrophic collapse. Warm restarts reset LR every epoch, destabilizing adversarial minimax training.
2. **Model EMA** — Forget accuracy spiked to 86-100%. EMA lags behind unlearning gradients, retaining forget-class knowledge.
3. **Feature Normalization** — Destroyed retain accuracy (-3 to -6 per task). Head weights incompatible with normalized inputs.
4. **Temperature Scaling for Adv Labels** — Forget accuracy collapsed (63-80%). Softer labels weaken unlearning gradient signal.
5. **Adversarial Mixup** — Net zero effect. Redistributed accuracy across tasks without improvement.
6. **SFDA² IFA Loss Boost** (alpha_1 1e-4→1e-2) — Regressed retain accuracy. IFA loss competes with unlearning.
7. **Adaptive m_alpha Annealing** — Net regression. Early unlearning damage to retain classes was irreversible.
8. **SNC Alpha Decay Tuning** — Net zero. Redistributed accuracy.
9. **Higher Base LR** (1e-2→2e-2) — Destabilized (w→a -1.8).
10. **Reduced Adv Sample Reinit** — Forget accuracy collapsed (a→d 31.6%). Frequent reinit essential.
11. **Entropy-Weighted Unlearning** — Hurt w→a (-2.2).
12. **Weight Decay Reduction** (1e-3→5e-4) — Net neutral (helped d→a, hurt a→w).
13. **Increased Iter/Epoch** (100→150) — Worse (0.770). Overfitting within epochs.
14. **Uniform Labels** — Forget collapse (61-80%). Rescaled labels essential.

## Key Insights

### The Minimax Training Is Extremely Fragile
Almost every change that touched the loss function, optimizer, or training dynamics caused either:
- Regression in w→a task (most sensitive pair)
- Collapse of forget accuracy
- Redistribution without net gain

### Perfect Forgetting Has Headroom
With adt_f=0.0, there's headroom to allow small forget accuracy increases (1-2%) in exchange for retain accuracy gains. However, attempts to exploit this (reduced m_alpha, annealing) didn't materialize because the minimax balance is delicate.

### The SCADA Framework Is Well-Tuned
The paper's default parameters (m_alpha=10, rescaled labels, 5 epochs) are remarkably well-calibrated. The two improvements found (gradient clipping, 10 epochs) each contributed ~0.1% — well within noise level.

### Significant Variance Across Tasks
Per-task results showed ±1-3% variation across runs with identical parameters. The 6-task Office31 evaluation has inherent noise, making small improvements difficult to validate.

## Top Remaining Ideas (for future runs)

1. **Orthogonal Gradient Projection** (ZS-PAG): Constrain forget gradients to be orthogonal to retain-class subspace. Unimplemented due to complexity.
2. **Stochastic Weight Averaging**: Apply SWA only to the final classification head (not full model). May avoid the EMA forget-collapse issue.
3. **Per-Domain m_alpha Tuning**: Different domain pairs have different difficulty — customized alpha per pair could help.
4. **Source Feature Distillation**: Add a distillation loss anchoring adapted model features to source features for retain classes.
5. **Larger Datasets**: The paper reports stronger results on OfficeHome (65 classes) and DomainNet (126 classes). Office31's small size (31 classes, ~4000 images) may be inherently noisy.
