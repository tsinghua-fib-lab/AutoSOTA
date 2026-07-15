# Theory Mechanism Check: Misranking → Update Dispersion → Progress Loss (Quadratic)

Goal: provide a small, fully controlled mechanistic experiment that validates the core
mechanism behind the analysis:

- misranking increases **update dispersion** (conditional variance of the rank-based update),
- under curvature (strong convexity), this induces a **one-step progress loss**,
- PEM (conditional expectation update) removes that variance-driven loss.

We run a synthetic strongly convex objective:

> `f(x) = 0.5 * ||x||^2`  (1-strongly convex)

and measure (for many randomly sampled candidate sets) the two-draw misranking metric
$M_{RD}$, the update dispersion $||\Delta m^{(1)}-\Delta m^{(2)}||^2$,
and a Jensen/strong-convexity inequality check:

> `E[f(m+Δm)] - f(m+E[Δm]) >= 0.5 * Var(Δm)`

## What’s inside

- `update_dispersion_quadratic.csv`: per-candidate-set measurements.
- `update_dispersion_quadratic.png`: (1) dispersion vs misranking; (2) Jensen-gap vs `0.5*Var`.

## Reproduce

```bash
python3 tools/diagnose_update_dispersion_quadratic.py \
  --out-dir evidence/theory_update_dispersion_quadratic \
  --dim 40 --lam 16 --mu-frac 0.5 \
  --sigma-x 0.5 --noise-sigma 1.0 \
  --num-sets 200 --mc-draws 256 --seed 123
```
