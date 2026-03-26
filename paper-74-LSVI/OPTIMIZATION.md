# Paper 74 — LSVI (Pima logistic regression)

**Full title:** *Least squares variational inference*

**Registered metric movement (`results.md`):** ELBO / KL-style objective **367.7902 → 367.7304** (**↓0.0598**, ~**0.016%**; lower is better). **Rubric “−2% vs baseline” target not met** — see “Why the bar is unreachable” below.

**Latest pipeline run:** `run_20260323_230402` (sota-5 `auto-pipeline`).

---

## What changed (high level)

The shipped **eval harness fixes K≈10** for the Pima logistic-regression posterior. Under that budget, **LSVI’s inner least-squares step is badly underdetermined** (many more sufficient statistics than samples), so **LSVI drifts away from the Laplace init** and returns **worse** KL than Laplace alone.

**Winning approach:** replace the unstable LSVI loop with **black-box VI (BBVI)** for a **Gaussian** variational family: **Adam + reparameterization gradients**, **8 random seeds**, **best-iterate tracking**, **phase1 8000 @ lr=0.01** + **phase2 1500 @ lr=0.001** (final deployed config = **iter 11** in the run log).

That configuration reaches **KL ≈ 367.7304**, extremely close to an estimated **Gaussian-family floor (~367.727)** from importance-sampling estimates of **−log Z** — i.e. there is almost no room left **without changing the variational family or the evaluation contract**.

---

## Why the −2% rubric target was not met

The rubric target **KL ≤ 360.3754** sits **~7.35 below** the **Gaussian** approximation’s effective minimum on this posterior. Hitting it would require **richer posteriors** (e.g. mixtures, flows) or **different problem/eval settings**, not more tuning inside a single Gaussian BBVI run.

---

## Files to read first

| Area | Path |
|------|------|
| Deployed Gaussian BBVI experiment | `LSVI/experiments/logisticRegression/pima_logistic_regression/gaussian.py` |
| Original LSVI machinery (reference) | `LSVI/variational/*lsvi*.py`, `numpy_gaussian_lsvi.py` |

---

## Reproduce / sanity-check

Run the Pima logistic regression experiment through the repo entrypoint used in the pipeline; confirm **best-tracked KL** aligns with **~367.7304** under the **8-seed, 8000+1500** BBVI schedule.
