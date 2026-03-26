# Optimization Report: Q-Mamba

**Paper ID**: 81
**Repository folder**: `paper-81-QMamba`
**Source**: AutoSota optimizer run artifact (`final_report.md`).
**Synced to AutoSota_list**: 2026-03-26

---

# Final Report: Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)

**Run**: run_20260321_045918
**Date**: 2026-03-21
**Baseline**: 0.987897
**Best Score**: 0.991972 (+0.41% over baseline)
**Target**: 1.007700 (+2.0%)
**Target Achieved**: No (gap: -1.56%)

---

## Summary

The primary optimization lever for Q-Mamba is **trajectory length** — more optimization steps consistently improve mean reward across all 8 BBOB problems. We exhausted the 1200-second time budget by extending from 500→1180 base steps, plus an adaptive stagnation-aware extension block that adds up to 150 more steps for stuck runs.

**Key finding**: Stochastic action selection (temperature softmax, even at T=0.01) catastrophically disrupts Q-Mamba's sequential greedy action decisions. Q-values are normalized 0-1 per position making them extremely temperature-sensitive. Greedy argmax is mandatory.

---

## Optimization History

| Iter | Idea | Before | After | Delta | Status |
|------|------|--------|-------|-------|--------|
| 0 | Baseline | — | 0.9879 | — | — |
| 1 | Trajectory 500→750 steps | 0.9879 | 0.9903 | +0.25% | SUCCESS |
| 2 | Trajectory 750→1000 steps | 0.9903 | 0.9913 | +0.10% | SUCCESS |
| 3 | Temperature softmax T=0.1 | 0.9913 | 0.9794 | -1.2% | FAILED |
| 4 | Temperature softmax T=0.01 | 0.9913 | 0.9794 | -1.2% | FAILED |
| 5 | Trajectory 1000→1100 steps | 0.9913 | 0.9915 | +0.02% | SUCCESS |
| 6 | Trajectory 1100→1150 steps | 0.9915 | 0.9918 | +0.02% | SUCCESS |
| 7 | Trajectory 1150→1180 steps | 0.9918 | 0.9919 | +0.01% | SUCCESS |
| 8 | Adaptive stagnation ext (thresh=0.2, 100 steps) | 0.9919 | 0.9919 | +0.001% | SUCCESS |
| 9 | Adaptive stagnation ext (thresh=0.15, 150 steps) | 0.9919 | 0.9920 | +0.007% | SUCCESS |
| 10 | model.eval() on load | 0.9920 | 0.9920 | 0% | FAILED |
| 11 | Stagnation thresh=0.10 | 0.9920 | timeout | — | FAILED |
| 12 | Stagnation thresh=0.12 | 0.9920 | timeout | — | FAILED |

---

## Final State (Best: Iter-9)

**Score**: 0.991972
**Git tag**: `_best` → commit `4f1881666a`

### Per-problem improvement (baseline → best)

| Problem | Baseline | Best | Delta |
|---------|----------|------|-------|
| q0 Ellipsoidal | 0.9997 | 0.9999 | +0.02% |
| q1 Rastrigin | 0.9462 | 0.9655 | +2.04% |
| q2 Linear_Slope | 0.9848 | 0.9884 | +0.37% |
| q3 Attractive_Sector | 0.9993 | 0.9999 | +0.06% |
| q4 Step_Ellipsoidal | 0.9754 | 0.9832 | +0.80% |
| q5 Rosenbrock_original | 0.9993 | 0.9994 | +0.01% |
| q6 Rosenbrock_rotated | 0.9995 | 0.9997 | +0.02% |
| q7 Ellipsoidal_high_cond | 0.9990 | 0.9998 | +0.08% |
| **Overall** | **0.9879** | **0.9920** | **+0.41%** |

---

## Code Changes (git diff _baseline → _best)

The only modified file is `/repo/q_mamba.py`:

```python
# In rollout_trajectory():
maxGens = max(maxGens, 1180)  # ITER7: extended trajectory to 1180
_base_gens = maxGens

# ... main loop (1180 steps) ...

# After main loop: adaptive extension for stagnating runs
if not need_trajectory and state[0, 0, 4].item() > 0.15:
    for _ext_gen in range(150):
        # ... same greedy rollout ...
```

**fea_5** = `stag_count / MaxGen` (already encoded in the 9-dim state). When `fea_5 > 0.15` after 1180 steps, the optimizer is still stagnating, so 150 additional greedy steps are run.

---

## What Didn't Work

1. **Temperature softmax** (any T > 0): Q-values are normalized to [0,1] per position and sequential action decisions are interdependent. Any stochasticity destroys the policy.
2. **model.eval()**: Mamba SSM has no dropout or batch normalization, so train/eval mode is identical.
3. **Lower stagnation thresholds (≤0.12)**: Too many runs get the extension, pushing eval time beyond 1200s.

## Why Target Was Not Reached

The +2% target (0.9879 → 1.0077) was unreachable with test-time-only modifications:

1. **Diminishing returns**: Each trajectory length doubling gives ~half the improvement of the previous. The trend (0.25% → 0.10% → 0.02% → 0.02% → 0.01%) asymptotes.
2. **Hard 1200s time budget**: More steps would directly improve Rastrigin (the bottleneck at 0.9655), but eval time scales linearly with steps.
3. **No retraining allowed**: The target would realistically require fine-tuning the model (longer training trajectories, Rastrigin-focused data augmentation, beam search lookahead). All require model weight modification.
4. **Rastrigin ceiling**: Starting at 0.9462, Rastrigin improved to 0.9655 (+2.04%) — the most improved per-problem. But even with infinite steps, the greedy policy can only do so much without retraining.

## Recommendations for Future Work

If model retraining were allowed:
1. **Fine-tune on longer trajectories**: Retrain Q-Mamba with 1500+ step trajectories, especially on Rastrigin.
2. **Rastrigin-augmented training**: Over-sample Rastrigin during training.
3. **Beam search rollout**: Maintain K candidate trajectories at each step; computationally feasible with small K.
4. **Lookahead Q-value**: Multi-step Q-values rather than greedy 1-step argmax could dramatically improve performance.
