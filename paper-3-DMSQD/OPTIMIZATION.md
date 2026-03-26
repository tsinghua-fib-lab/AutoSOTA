# Paper 3 — DMSQD

**Full title:** *Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces*

**Registered metric movement (internal ledger, ASCII only):** +7.32%(6500.85->6976.81)

# Optimization Results: Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces

## Summary
- Total iterations: 1
- Best `average_qd_score`: 6976.81 (baseline: 6500.85, improvement: **+7.3%**)
- Target: 6630.867 — **ACHIEVED on iteration 1**
- Best commit: 291e5ac7fa

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| average_qd_score | 6500.85 | 6976.81 | +7.3% |
| sphere_qd_score | 6425.37 | 6790.28 | +5.7% |
| rastrigin_qd_score | 5128.54 | 5493.60 | +7.1% |
| flat_qd_score | 7948.65 | 8646.55 | +8.8% |

## Key Changes Applied
| Change | File | Effect | Notes |
|--------|------|--------|-------|
| emitters 15→20 for multi_100d domains | `config/algo/e1_num.yaml` | +7.3% avg QD score | Modified sphere_multi_100d, rastrigin_multi_100d, flat_multi_100d |
| Added eval.sh | `eval.sh` | Enables standardized evaluation | Runs 3 domains × 20 seeds, prints metrics |

## What Worked
- **Increasing number of emitters from 15 to 20** was the single most impactful change, yielding +7.3% improvement across all 3 domains.
 - The logic: more CMA-ES emitters = more parallel search threads, each with different covariance matrices and positions in solution space, covering more measure space cells per iteration
 - Each emitter explores ~1 region independently, so 20 emitters (vs 15) gives 33% more coverage channels while the computation is parallelized across all seeds with joblib
 - The effect was consistent across all 3 domains: +5.7% sphere, +7.1% rastrigin, +8.8% flat

## What Didn't Work / Wasn't Tried
- Only 1 iteration was needed to reach the target
- Additional ideas pending (sigma0 tuning, MLP size increase, empty_points tuning) were not needed

## Architecture Insight
The DMS algorithm uses CMA-ES emitters that each independently maintain a Gaussian distribution in solution space. Each emitter:
- Maintains its own covariance matrix (CMA-ES adaptation)
- Samples `batch_size=36` solutions per iteration from its distribution
- Gets guided by the discount model's improvement values
- Restarts every 100 iterations if stuck (`restart_rule=100`)

With 15 emitters → 15×36=540 solutions per iteration
With 20 emitters → 20×36=720 solutions per iteration (+33% coverage)

This directly translates to more archive cells being filled per iteration, yielding higher QD scores at the same number of training iterations.

## Deep-research memo (excerpt from `research_report.md`)

**3. Parameter Optimization Insights**
Based on DMS’s ablation and analogous QD works, the following hyperparameter ranges are effective:

- **Discount-archive learning rate (α):** Tested values [0.0, 10^-3, 10^-2, 0.1, 1.0], with **0.1** often best in prior work (www.emergentmind.com). It controls how quickly the archive’s discount predictions update. In practice, one might vary α in [0.01, 0.2]. A smaller α (<0.05) makes the model very conservative (slow to learn improvements), risking slow coverage. A larger α (>0.2) can chase noisy improvements. 

- **Discount update rule parameters:** Aside from α, the *discount factor* itself (in `discount.yaml`) may include thresholds or decay rates. Common settings keep the factor in [0.5,1.0] to ensure only gradual “forgetting” of old values.

- **Number of Emitters:** Typical QD runs use on the order of 5–20 emitters. The DMS config uses ~10. More emitters yields more parallel search and faster coverage (higher QD-score) but also more memory/compute. If GPU/CPU allows, increasing to ~20 can give modest gains. Risks: diminishing returns and crowding (too many simultaneous searches covering overlapping areas).

- **Emitter step-size (σ₀):** CMA-ES’s initial step σ₀ is critical. Common practice (from CMA-ES literature) is to set σ₀ to ~0.2–0.5 of the typical search range or distance in the solution space. A too-small σ₀ (<0.05) means emitters make tiny moves (low exploration, low score), whereas too large (>1.0) may overshoot. In DMS settings, the default is often 0.3–0.5; try ±50% around that. Gains are substantial if well-chosen (maybe tens of percent in QD-score).

- **Population Size:** If CMA-ES emitters use a population (e.g. default λ=(4+⌊3 log(n)⌋)), one could experiment with larger λ to increase diversity per generation, at the cost of more evaluations.

- **CVT Archive Size:** The CVT archive divides the measure space into a fixed number of cells (centroids). Increasing the number of centroids (e.g. from 1000 to 5000) yields finer coverage and can boost QD-score, but risks many cells staying empty (especially in high dims). A rule of thumb is a few dozen points per cell; beyond that, diminishing returns. The current `cvt_multi_100d.yaml` likely uses O(10k) centroids for 100D; you could try ±50% of that count.

- **Iterations (`itrs`):** More iterations directly increase score (security by law, but runtime goes up). Doubling itrs (e.g. 10k→20k) might raise QD Scores ~10–30%, depending on plateauing. This is a linear gain but with high cost (risk: very long runs).

- **Parallelism (`n_jobs`):** On multi-GPU servers, raising `n_jobs` from 10 to e.g. 20+ can double throughput. Inference risk is only running out of memory if set too high.

- **Restart Strategy:** The DMS script mentions `restart_basic/restart_100/n_empty`. These might restart emitters every 100 iters, or only if <N new solutions found. Tuning this (e.g. restart every 50 iters or change to `restart_full`) can help if you see stagnation. The trade-off is losing partial progress.

**4. Concrete Optimization Ideas**
Below are ten actionable ideas to improve the QD metrics, ordered from simple to complex. *Gains* are very approximate; **Risk** is a qualitative assessment (Low: easy to try, unlikely to hurt; High: could degrade performance or be hard to get right).

1. **Tune Discount-LR (α):** Test values around the default 0.1 (e.g. 0.05, 0.2). A slightly higher α can adapt faster to improvements (gain +2–5% coverage), while lowering α yields more stable learning (risk: slow gains). **Risk:** Low (just experiments). 
2. **Increase `n_jobs`:** If hardware available, raise job count (e.g. 10→20). This directly increases samples/time. On a GPU cluster this can yield ~10–30% score boost for the same wall-clock time. **Risk:** Low (just resource use). 
3. **More Emitters:** Double the number of CMA-ES emitters (e.g. 10→20). This multiplies exploration channels. Expect moderate gains (maybe +5–10% score) at cost of double evals. **Risk:** Medium (can saturate overlapping coverage; running too many small populations can interfere). 
4. **CVT Resolution Adjustment:** If many CVT cells are empty or only sparsely populated, reduce the number of centroids; if almost full, increase them. A better-sized archive can boost QD-score by 5–15%. **Risk:** Medium (if mis-set, coverage can drop or evaluation wasted on too-sparse cells). 
5. **Adaptive Restart:** Switch to a more aggressive restart mode (e.g. restart all emitters every 50 iters). This helps avoid local stagnation. Potential gain: +5% (or more) by exploring new regions. **Risk:** Medium–High (resets lose accumulated covariance info, might oscillate without net gain if done unwisely). 
6. **Augment Measure Data:** For each reference image in the measure dataset, include a few random augmentations (e.g. affine transforms). This effectively creates new measure points for the model to learn, improving its granularity. Could yield +5–10% in coverage by guiding exploration into neighborhoods. **Risk:** Medium (augmented images might duplicate measures undesirably or bias the archive). 
7. **Discount Model Regularization:** Increase the L2 penalty (λ in [62]) or add dropout in the discount NN so it generalizes better. This may improve long-term exploration (gain ~+3% by avoiding overfit) at the risk of underfitting. **Risk:** Medium (if over-regularized, model fails to learn and the algorithm reverts to “no discount” mode). 
8. **Parallel Archive Ensemble:** Run two copies of DMS with different seeds for N/2 iterations each, then merge their archives (take the best cell entries). This ensemble can easily squeeze an extra +5% coverage from diverse search paths. **Risk:** Low (just bookkeeping, no new algorithm training). 
9. **Local Fine-Tuning (no model retrain):** Take the top solutions in each filled cell and do a few steps of local optimization (e.g. a couple of CMA-ES iterations or even gradient steps if a differentiable surrogate is available). Even a small refinement might add +1–2% to the score. **Risk:** Medium (could collapse multiple near-identical solutions to the same high-quality point, slightly reducing diversity). 
10. **Custom Emitter Heuristics:** Modify or add an emitter that specifically targets underfilled regions. For example, after some iterations, identify measure space quadrants with few elites and initialize a dedicated CMA-ES in those regions. This targeted search could boost QD-score by focusing on blind spots (+5–10% in worst-covered tasks). **Risk:** High (hand-crafted focus might neglect objective quality or destabilize balance; requires careful criterion design).

**5. Common Failure Modes**
When tuning or extending DMS-like QD systems, practitioners often stumble in predictable ways:

- **Mode Collapse in High-D Measures:** If the discount model or CVT is misconfigured, many solutions “pile up” in a few measure cells. This eliminates diversity. To avoid this, always check coverage; if collapse occurs, reduce model confidence (lower α) or increase CVT resolution.

- **Overfitting the Discount Model:** With too large a model or too aggressive updates, the neural discount can memorize noise (predict exaggerated rewards on some measures). This misguides search. Lacking a proper validation set, watch out for “all entries get the same high discount” – a bad sign.

- **Insufficient Exploration:** If σ₀ or population sizes are too small, CMA-ES emitters will take tiny steps and never discover new cells. Conversely, too large σ₀ wastes samples. Gradually tuning σ₀ (e.g. decreasing it over time) is often safer.

- **Hyperparameter Extremes:** Sweeping hyperparameters indiscriminately can backfire. For example, α→1 or learning_rate in CMA→0.5 might cause erratic archive updates. It’s safer to make one change at a time and monitor metrics.

_(Research digest truncated.)_

## Idea library snapshot (`idea_library.md`)

### IDEA-001: Increase n_jobs parallelism (10→20)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change `hydra.launcher.n_jobs` in eval.sh from 10 to 20. This doesn't change the quality of any single run, but allows more seeds to run in parallel — with 2 GPUs and CPU-based domains (sphere/rastrigin/flat are pure Python), this should be fine.
- **Hypothesis**: No direct effect on quality; primarily reduces wall-clock time. But since eval.sh already runs all 20 seeds, this won't change the final scores — SKIP this.
- **Status**: SKIPPED — no effect on final QD score

### IDEA-002: Increase number of emitters (15→20)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change `domain_to_e1_num.sphere_multi_100d`, `rastrigin_multi_100d`, and `flat_multi_100d` in `/repo/config/algo/e1_num.yaml` from 15 to 20. More emitters = more parallel CMA-ES searches, each exploring different regions of the space.
- **Hypothesis**: +3-8% QD score improvement. More emitters cover more cells and provide more diverse search trajectories. Risk: may slow each iteration (more evaluations: 20×36=720 vs 15×36=540).
- **Status**: SUCCESS — avg 6500.85→6976.81 (+7.3%), sphere 6425.37→6790.28, rastrigin 5128.54→5493.60, flat 7948.65→8646.55

### IDEA-003: Increase sigma0 for exploration (0.5→0.7)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW-MEDIUM
- **Description**: Change sigma0 for sphere/rastrigin/flat multi 100d from 0.5 to 0.7 in `/repo/config/algo/emitters/sigma0.yaml`. Larger initial step size encourages more exploration.
- **Hypothesis**: Wider initial exploration could lead emitters to discover new cells. Risk: too large might overshoot. Expected +2-5%.
- **Status**: PENDING

### IDEA-004: Increase empty_points (100→200)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Change `empty_points` in `/repo/config/algo/archive/discount.yaml` from 100 to 200. More empty points regularize the discount model better, keeping discount values low in unexplored areas.
- **Hypothesis**: Better regularization = more exploration encouraged. Expected +1-4%.
- **Status**: PENDING

### IDEA-005: Increase MLP hidden size (128→256)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW-MEDIUM
- **Description**: Change MLP layer_specs from `[[100,128],[128,128],[128,1]]` to `[[100,256],[256,256],[256,1]]`. A larger model can better represent the discount function's smooth manifold over 10D measure space.
- **Hypothesis**: More expressive model = more accurate discount predictions = better emitter guidance. Expected +2-5%. Risk: slower training per iteration.
- **Status**: PENDING

### IDEA-006: Increase train_epochs (5→10)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `train_epochs` in `/repo/config/algo/discount_model/mlp.yaml` from 5 to 10. More epochs per training step allows the discount model to better fit the current data.
- **Hypothesis**: Better model accuracy → better improvement values → better solution acceptance. Expected +1-3%.
- **Status**: PENDING

### IDEA-007: Reduce train_cutoff_loss
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: The default `train_cutoff_loss = 0.05 * (obj_high - obj_low) = 0.05`. Reduce this to 0.01 to force more thorough training even if current loss seems acceptable.
- **Hypothesis**: More accurate model = better discount estimates. Expected +1-3%.
- **Status**: PENDING

### IDEA-008: Increase batch_size for emitters (36→48)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `batch_size` in `/repo/config/algo/emitters/es_imp.yaml` from 36 to 48. Larger batch gives each CMA-ES step more information.
- **Hypothesis**: Better CMA-ES updates with more samples. Expected +1-3%.
- **Status**: PENDING

### IDEA-009: Change restart_rule to 'basic' (from 100)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Change `restart_rule` for multi_100d domains from '100' to 'basic'. Basic restart triggers on CMA-ES convergence criteria rather than fixed 100-iteration stagnation.
- **Hypothesis**: May help or hurt — depends on whether 100-iteration restarts are optimal for these domains. Risk: basic might restart too early or too late.
- **Status**: PENDING

### IDEA-010: Increase init_train_points (1000→2000)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: LOW
- **Description**: Change `init_train_points` from 1000 to 2000. More points for initial discount model training → better initialization.
- **Hypothesis**: Better starting point for the model. Expected +0.5-2%.
- **Status**: PENDING

### IDEA-011: Increase iterations (10000→12000)
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Increase `itrs` in eval.sh from 10000 to 12000. More iterations = more evaluations = more coverage.
- **Hypothesis**: Direct improvement but proportional runtime increase. If runtime budget allows, this is a safe bet for +3-8%.
- **Status**: PENDING

### IDEA-012: Add deeper MLP with skip connections (LEAP)
- **Type**: LEAP
- **Priority**: LOW
- **Risk**: HIGH
- **Description**: Replace the basic 3-layer MLP with a residual MLP that has skip connections. This addresses potential gradient vanishing in deeper networks when learning smooth manifold structure.
- **Hypothesis**: Better representational capacity. Risk: more complex, might not help with small networks.
- **Status**: PENDING

### IDEA-013: Combine emitter count + sigma0 tuning
- **Type**: PARAM
- **Priority**: HIGH
- **Risk**: LOW
- **Description**: Combined increase: emitters 15→20 AND sigma0 0.5→0.6. Both changes together could be synergistic — more emitters each making bolder moves could cover more territory.
- **Hypothesis**: Synergistic effect, expected +5-10%.
- **Status**: PENDING

### IDEA-014: Adaptive learning_rate schedule for discount model
- **Type**: CODE
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Start with a higher discount learning_rate (e.g. 0.2) and gradually reduce it over iterations. Modify `train_discount_model` to decay the rate. Faster initial adaptation, then stable fine-tuning.
- **Hypothesis**: Better early exploration, then more stable discounting. Expected +2-4%.
- **Status**: PENDING

### IDEA-015: MLP with larger hidden size and more layers (100→256→256→256→1)
- **Type**: PARAM
- **Priority**: MEDIUM
- **Risk**: MEDIUM
- **Description**: Add a 4th layer to the MLP architecture. More depth + wider layers for better discount function approximation in 10D space.
- **Hypothesis**: Richer feature representation of measure space topology. Expected +2-5% at cost of slower training.
- **Status**: PENDING
