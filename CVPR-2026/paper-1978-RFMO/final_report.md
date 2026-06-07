# Optimization Results: Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach

## Summary
- **Total iterations**: 9
- **Best `instance_f1`**: 0.5655 (baseline: 0.5372, improvement: **+5.3%**)
- **Target**: 0.5641 — **EXCEEDED** (+0.14% above target)
- **Best commit**: `2ccf0b94e4d357c371dff3fc7111730b6109a6f2`

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | % Change |
|--------|----------|------|-------|----------|
| instance_f1 | 0.5372 | **0.5655** | +0.0283 | **+5.3%** |
| hamming_accuracy | 0.9602 | 0.9701 | +0.0099 | +1.0% |
| subset_accuracy | 0.2853 | 0.3116 | +0.0263 | +9.2% |
| micro_f1 | 0.4685 | 0.5482 | +0.0797 | +17.0% |
| macro_f1 | 0.2628 | 0.2754 | +0.0126 | +4.8% |

**All metrics improved simultaneously** — no metric-dimension trade-off.

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| `num_samples_to_infer`: 200 → 500 | +~2% instance_f1 | Reduces MC variance in P-matrix estimation |
| `temperature`: 1.0 → 0.5 | +~2% instance_f1 | Better exploration-exploitation balance at inference |
| `num_ensemble_runs`: 1 → 4 (multi-run P-matrix averaging) | +~1% instance_f1 | Averages P-matrices across 4 independent sampling runs (125 samples each) |

## What Worked

1. **Inference-time parameter tuning** — The winning combination was purely inference-time changes (no retraining needed). Increasing MC samples from 200→500 and lowering temperature from 1.0→0.5, combined with multi-run P-matrix averaging, produced a 5.3% improvement.
2. **Multi-run ensemble** — Running 4 independent sampling passes and averaging the resulting P-matrices reduced sampling noise and improved the fidelity of the F1 inference.
3. **Temperature tuning** — The default temperature of 1.0 was too exploratory. T=0.5 provided the right balance of exploration vs. exploitation for this dataset.

## What Didn't Work

1. **AdamW + CosineAnnealingWarmRestarts** — Regressed instance_f1 by 2.6%
2. **Early stopping on validation instance_f1** — Regressed by 6.4%; val F1 is too noisy for reliable early stopping
3. **GELU activation** — Catastrophic regression (-15.8%); ReLU sparsity is beneficial for this architecture
4. **Temperature annealing during training** — No effect; BCE loss uses true labels, not sampled labels, so annealing doesn't change training dynamics
5. **Label-sensitive pos_weight loss** — Distorted training even with conservative caps; uniform BCE works well with the autoregressive model

## Top Remaining Ideas (for future runs)

1. **Per-label inference temperature** — Learn or tune different temperatures for different labels (rare labels may benefit from higher T for better exploration)
2. **F1-optimal weight kernel tuning** — Introduce a beta parameter in the 1/(beta*l + k) kernel and optimize on validation set
3. **Test-time adaptively optimized thresholds** — Run a quick per-instance grid search over label cardinality during inference
4. **Stochastic Weight Averaging (SWA)** — Average model weights across last N checkpoints for improved generalization
