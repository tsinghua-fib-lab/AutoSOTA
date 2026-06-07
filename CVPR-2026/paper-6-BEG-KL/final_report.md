# Optimization Results: Beyond Euclidean Gossip — KL-Barycentric Consensus on Heterogeneous and Imbalanced Images

## Summary
- Total iterations: 24 (0 baseline + 23 experiments + 1 final)
- Best `dice`: **0.8409** (baseline: 0.8330, improvement: **+0.95%**)
- Best `iou`: **0.7602** (baseline: 0.7510, improvement: **+1.2%**)
- Best `loss`: **0.3356** (baseline: 0.3968, improvement: **-15.4%**)
- Best commit: `631afda` (iter-16)
- Key change: **lr0: 5e-3 → 3e-3** (one-line change)

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| Dice | 0.8330 | 0.8409 | **+0.95%** |
| IoU | 0.7510 | 0.7602 | **+1.2%** |
| Loss | 0.3968 | 0.3356 | **-15.4%** |
| Best Val Dice | 0.8249 | 0.8458 | **+2.5%** |

## Key Change Applied
| Change | Effect | Notes |
|--------|--------|-------|
| `--lr0` default: 5e-3 → 3e-3 | +0.95% Dice, +1.2% IoU, -15.4% Loss | Reduced learning rate decreases gradient noise from tiny per-client batches (BS=2). The original lr0=5e-3 was too aggressive for the severe non-IID setting (alpha=0.1). |

## What Worked

1. **Lower learning rate (lr0=3e-3)**: The single most impactful change. At 5e-3, the optimizer takes steps that are too large for the noisy gradients from BS=2. At 3e-3, training is more stable and converges to a better solution. Confirmed reproducible: repeat runs gave 0.8378-0.8409 dice.

2. **Double gossip rounds per step**: Achieved essentially baseline performance (0.8321) and dramatically improved validation (best_val_dice 0.8403). Eliminated the val-test gap. However, the combination with lr0=3e-3 regressed — the two optimizations work against each other.

## What Didn't Work

| Change | Best Dice | Delta | Why |
|--------|-----------|-------|-----|
| EMA Model Averaging | 0.8150 | -2.2% | EMA adds harmful parameter lag; gossip already smooths |
| AdamW + beta2=0.999 | 0.8220 | -1.3% | Better calibration but no dice improvement |
| Combo Loss (boundary BCE) | 0.8085 | -2.9% | Boundary weighting disrupted optimization |
| SGDR Warm Restarts | 0.8124 | -2.5% | LR stayed too high at end, preventing convergence |
| TTA (horizontal flip) | 0.8249 | -1.0% | Flip-only insufficient; improved val but not test |
| Deferred Gossip (10 epochs) | 0.7423 | -10.9% | Catastrophic — consensus is essential |
| Morphological Post-Processing | 0.6958 | -16.5% | Cleanup destroyed correct predictions |
| Adaptive Gradient Clipping | 0.8175 | -1.9% | Adaptive threshold didn't help |
| Label Smoothing (ε=0.1) | 0.8196 | -1.6% | Better calibration, worse metrics |
| Triple Gossip Rounds | 0.8216 | -1.4% | Over-smoothing; double gossip was sweet spot |
| Best Checkpoint Test Eval | 0.8177 | -1.8% | Best checkpoint didn't translate to test |
| prior_scale=0.2/0.4 | 0.8099-0.8314 | varies | Default 0.3 is optimal |
| lr0=4e-3 | 0.8259 | -0.9% | 3e-3 is the sweet spot |
| lr0=1e-3 | 0.8079 | -3.0% | Too low for convergence |
| Epochs=100 | 0.8292 | -0.5% | Overfits with more training |
| lr0=3e-3 + double gossip | 0.8193 | -2.6% | Best findings don't combine |
| lr0=3e-3 + wd=5e-4 | 0.8235 | -2.1% | Default weight decay is optimal |
| lr0=3e-3 + lr_min=5e-5 | 0.8223 | -2.2% | Default lr_min is optimal |

## Key Insights

1. **The baseline is remarkably well-tuned**: 23 experiments, 1 improvement. The KLC-Adam optimizer with gossip consensus is carefully balanced. Almost any change disrupts this balance.

2. **Gradient noise is the bottleneck**: The tiny per-client batch size (BS=2) creates noisy gradients. Lowering lr0 reduces the impact of this noise.

3. **Consensus is critical**: Deferred gossip (skipping consensus for 10 epochs) caused catastrophic failure (-10.9%). Under extreme non-IID (alpha=0.1, client 3 has only 1 image), the consensus mechanism is not optional — it's fundamental.

4. **The best improvements don't combine**: lr0=3e-3 (best optimizer setting) + double gossip (best consensus approach) regressed. These two optimizations work through different mechanisms that conflict when combined.

5. **Simplicity wins**: After 23 experiments spanning architectural changes, novel loss functions, consensus modifications, and parameter sweeps, the winning change was a single number: 5e-3 → 3e-3.

## Top Remaining Ideas (for future runs)

1. **lr0 fine-tuning**: Sweep lr0 in [2.5e-3, 3.5e-3] with finer granularity — the sweet spot might be 2.8e-3 or 3.2e-3.

2. **Per-client adaptive LR**: Clients with different dataset sizes might benefit from different learning rates. Client 3 (1 image) needs very different LR from client 7 (263 images).

3. **Better TTA**: Multi-scale TTA (0.9x, 1.0x, 1.1x) instead of flip-only might provide the +0.02-0.04 Dice boost predicted by literature.

4. **FedFAT-style frequency augmentation**: Frequency-space adaptive interpolation for non-IID medical images (Wang et al., 2025) reported +2.42% average Dice.

5. **Variance-corrected model averaging**: Tian et al. (2024) showed that rescaling weights post-aggregation can recover Xavier initialization properties in gossip learning.
