# Paper 73 — RLD

**Full title:** *Regularized Langevin Dynamics for Combinatorial Optimization*

**Registered metric movement:** +0.235% (rlsa_size: 19.969 → 20.016)

---

# Optimization Results: Regularized Langevin Dynamics for Combinatorial Optimization

## Summary
- Total iterations: 12
- Best `rlsa_size`: **20.015625** (baseline: 19.968750, improvement: **+0.235%**)
- Target was 20.3694. We achieved 20.016, which is +0.235% above baseline but did not reach the 2.0% target.

## Baseline vs. Best Metrics
| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| rlsa_size | 19.968750 | **20.015625** | +0.046875 (+0.235%) |
| rlsa_time (per instance) | 0.1092 s | 0.0978 s | -0.0114 s (faster!) |

## Key Changes Applied
| Change | File | Effect | Notes |
|--------|------|--------|-------|
| Increased tau0 default | main.py | Minimal (+0%) | 0.01→0.05 |
| Population-based refresh (LEAP) | solvers.py | +0.08% | Core algorithmic improvement |
| Tuned refresh interval to 30 | solvers.py | +0.08% | Optimal frequency |
| Global best_sol source + adaptive perturb | solvers.py | +0.08% | Final tuning |

---

## Key Algorithmic Change (Population Refresh)
The core improvement is a **population-based chain refresh** mechanism in `RLSA.evaluate_single()`:

Every 25 Langevin steps:
1. Identify the 15% worst-performing chains (by global best energy)
2. Copy the 15% best-performing chains' global best solutions
3. Apply adaptive perturbation (15% early → 5% late flip rate)
4. Replace worst chains with perturbed best

---

## What Worked
1. **Population-based chain refresh** (LEAP in iter 4): the single biggest improvement
2. **Refresh timing tuning** (iter 5): interval=30 beats 50 and 20
3. **Using global best_sol as refresh source**: slightly better than using current x

---

## Conclusion
The population-based refresh LEAP successfully improved rlsa_size from 19.97 to 20.02, but did not reach the 2.0% improvement target. The RLSA algorithm with fixed CLI parameters is fairly constrained in what algorithmic changes help. The main lever was introducing diversity between chains via population refresh.
