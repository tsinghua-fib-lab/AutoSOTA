# PLATE SOTA Optimization — Final Report

## Paper
- **ID**: 2663
- **Title**: PLATE: Plasticity-Tunable Efficient Adapters for Geometry-Aware Continual Learning
- **Repo**: https://github.com/SalesforceAIResearch/PLATE
- **Container**: autosota_repro_paper_2663

## Best Result

| Metric | Baseline | Best (Iter 6) | Paper Reported | Δ vs Baseline | Δ vs Paper |
|--------|----------|---------------|----------------|---------------|------------|
| Task 2 Accuracy | 98.14% | **98.25%** | 98.28% | +0.11% | -0.03% |
| Task 1 Retention | 96.46% | **98.96%** | 97.45% | **+2.50%** | +1.51% |
| Forgetting | 2.77% | **0.27%** | 1.85% | **-2.50%** | -1.58% |

**Best commit**:  — KD λ=0.1 + 15 epochs

## Evaluation Command

```bash
python3 run_mnist_plate.py --r 350 --tau 0.8 --alpha 0.5 --n_runs 10 --gpu 0 --epochs 15 --lr 1e-3 --kd_lambda 0.1
```

## Key Finding

**Feature-level knowledge distillation (MSE on backbone features) virtually eliminates catastrophic forgetting without any cost to new-task learning.**

Adding a simple MSE loss between the PLATE-adapted backbone's output features and the frozen pre-adaptation backbone's output features (on task-2 training data) reduces forgetting from 2.77% to 0.27% — a 10× reduction. At the same time, Task 2 accuracy improves from 98.14% to 98.25%, matching the paper's reported 98.28%.

The KD loss acts as a soft regularization: the adapter is free to learn task-2 features, but must not deviate the backbone's representations too far from their pre-trained state. This is especially effective for PLATE because PLATE's adapter output is additive (y = base(x) + adapter(x)), so the KD loss directly penalizes large adapter magnitudes.

## All Score Records

| Iter | Idea | T2 Acc | T1 Ret | Forget | Status |
|------|------|--------|--------|--------|--------|
| 0 | Baseline (reproduced) | 98.14% | 96.46% | 2.77% | success |
| 1 | I-02: Cosine annealing LR | 97.85% | 96.20% | 3.03% | success (regression) |
| 2 | I-03: AMP + 15 epochs | 98.20% | 96.40% | 2.84% | success (marginal) |
| 3 | I-01: Per-layer tau | 98.14% | 96.45% | 2.78% | success (neutral) |
| 4 | I-05: KD λ=0.1, 10ep | 98.17% | 98.87% | 0.37% | success **(breakthrough)** |
| 5 | I-05: KD λ=0.05, 10ep | 98.22% | 98.66% | 0.57% | success |
| 6 | **I-05: KD λ=0.1, 15ep** | **98.25%** | **98.96%** | **0.27%** | success **(best)** |
| 7 | I-05: KD λ=0.1, 15ep, τ=0.75 | 98.25% | 98.96% | 0.27% | success (tied) |

## Ideas Attempted

### Successful
- **I-05 (Feature Distillation KD)** — **Breakthrough**. MSE distillation from frozen teacher on Stage 2 data eliminates forgetting while maintaining/improving T2 accuracy. Three variants tested:
  - λ=0.1, 10 epochs: T2=98.17%, T1=98.87%
  - λ=0.05, 10 epochs: T2=98.22%, T1=98.66%
  - **λ=0.1, 15 epochs: T2=98.25%, T1=98.96% (BEST)**
- **I-03 (AMP)** — Marginal T2 improvement (+0.06%), T1 unchanged. Useful infrastructure (--epochs/--lr propagation).

### Neutral / No Change
- **I-01 (Per-layer tau)** — No effect on MNIST (±0.05 tau changes too small for 3-layer MLP). Infrastructure (col_tau_pattern) added to PLATEConfig.
- **I-07 (tau=0.75 with KD)** — Identical to τ=0.8 when KD is active.

### Regressions
- **I-02 (Cosine annealing LR)** — Regression on both T2 (-0.29%) and T1 (-0.26%). Cosine decay too aggressive for tiny 34K-param adapter.

## Red-Line Confirmations

All iterations verified:
- ✓ Evaluation command unchanged (same test data, splits, labels, metric definitions)
- ✓ No hard-coded outputs or dataset-specific shortcuts
- ✓ All metrics reported honestly (including regressions)
- ✓ No evaluation protocol changes
- ✓ Optimization objective respected (multi-metric trade-off, both T2 and T1 tracked)

## Changes Made to Codebase

1. **run_mnist_plate.py**: 
   - Propagated --epochs and --lr CLI args to single_run() (previously hardcoded)
   - Added torch.cuda.amp autocast + GradScaler support
   - Added --kd_lambda CLI arg with MSE feature distillation from frozen teacher
   - Added per-layer tau pattern (col_tau_pattern) to PLATEConfig call

2. **plate/config.py**: Added  field to PLATEConfig for per-layer energy threshold configuration.

3. **plate/model.py**: Added per-layer col_tau resolution in _create_and_replace(), matching existing rank_pattern/alpha_pattern logic.

## Conclusions

1. **Feature distillation KD is the single most impactful lever** for PLATE on continual learning tasks. It nearly eliminates catastrophic forgetting while preserving (or slightly improving) new-task learning.

2. **Extended training epochs help when combined with KD**. The 15-epoch + KD variant outperformed both 10-epoch KD variants, suggesting the KD constraint prevents overfitting even with longer training.

3. **LR scheduling (cosine annealing) is counterproductive** for tiny adapters. The constant LR works better because the adapter has few parameters and needs sustained plasticity.

4. **Per-layer hyperparameter tuning (tau, trainable fraction) has minimal effect on small models**. These levers are likely more impactful for larger models with diverse layer dimensions.

5. **AMP provides negligible benefit** for MNIST-scale training where GPU utilization is already low.
