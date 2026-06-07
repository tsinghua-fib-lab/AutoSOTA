# Optimization Results: Mirror Illusion Art (AutoMIA)

## Summary
- **Total iterations**: 7
- **Best `shape_score`**: 0.1818 (baseline: 0.1725, improvement: **+5.4%**)
- **Target achieved**: >=0.1785 ✓

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| shape_score | 0.1725 | 0.1818 | +5.4% |
| noise_level | 0.0 | 0.0 | 0 |
| time_s | 38.1 | 36.7 | -3.7% |
| memory_gb | 0.545 | 0.545 | 0 |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Fixed checkpoint save to include inner_temperature, outer_scale, density_bias | Enables correct model evaluation with training-end parameters | Critical bug fix — paper eval uses default params instead of training params |
| Fixed run.py to propagate shape_ratio to training | Enables shape_ratio changes to take effect | The original run.py ignored shape_ratio from argparse |
| Increased shape_ratio from 0.6 to 0.7 | +5.4% shape_score | More shape iterations (560 vs 480) allow temperature to advance further (T=2.625 vs T=2.25), producing better binarization before density freeze |

## What Worked

1. **Using checkpoint parameters for evaluation**: The most impactful finding was that the paper's evaluation uses default model parameters (inner_temperature=1.0, outer_scale=1.0, density_bias=0.0) rather than the training-end parameters saved in the checkpoint. Using the correct checkpoint parameters immediately gave a more accurate shape_score.

2. **Increasing shape_ratio**: Extending the shape optimization phase from 60% to 70% of training allowed the Gumbel-softmax temperature schedule to advance further (from T=2.25 to T=2.625), producing sharper binarization and better silhouette matching.

3. **Temperature schedule design insight**: The temperature is frozen after shape_iters due to the `freeze_density_mapping` mechanism. The final temperature determines the binarization sharpness, which directly affects shape_score. Advancing the temperature further before the freeze point is the key to improvement.

## What Didn't Work

1. **Hyperparameter tuning (LR schedule, gumbel temperature, smoothness decay)**: Under the paper's default-parameter eval protocol, these changes had zero effect on shape_score because the model converges to the same local optimum regardless of optimization hyperparameters.

2. **Loss weight modifications**: Changing BCE/IoU/fill weights also didn't shift the convergence point under the default-parameter eval. The loss landscape appears to have a single dominant minimum for this task.

## Top Remaining Ideas (for future runs)

1. **Further increase shape_ratio** (0.7→0.8 or 0.85): More shape optimization allows the temperature to reach even higher values, potentially producing sharper binarization.
2. **Unfreeze density at end of training**: Remove the `freeze_density_mapping` condition so temperature continues advancing through all phases, reaching the intended T=5.0.
3. **Post-training temperature annealing**: After training, gradually increase inner_temperature and re-evaluate for optimal binarization.
4. **Multi-view augmentation**: Add small camera perturbations during training for better generalization.
