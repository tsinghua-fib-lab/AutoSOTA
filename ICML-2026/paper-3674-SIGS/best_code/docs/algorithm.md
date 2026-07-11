# SIGS Algorithm: Two-Stage Symbolic PDE Discovery

## Overview

Given a PDE system `L[u] = F` on domain Ω × [0,T] with initial and boundary
conditions, SIGS returns a closed-form symbolic expression for each unknown
field that satisfies the PDE to within a specified RMSE tolerance.

---

## Stage I: Symbolic Structure Discovery

### 1. Grammar and Latent Space

The grammar `G` (defined in `src/sigs/grammar.py`) is a context-free grammar
over the terminal alphabet `{x, y, t, +, -, *, /, sin, cos, exp, log, sqrt,
tanh, 0–9}`. Every expression decoded by the Grammar-VAE is guaranteed to be
syntactically valid under `G`.

The Grammar-VAE (`src/sigs/model.py`) consists of:
- **Encoder**: three Conv1d layers (sizes 64→128→256, kernels 2,3,4) with LayerNorm → linear head → (μ, log σ²) in ℝ^32
- **Decoder**: linear projection → GRU (1 layer, 512 hidden, positional mode) → softmax over grammar rules
- **Training**: cross-entropy reconstruction loss + β-KL divergence with linear warmup,
  optional geometric regularization (convex hull, topological persistence) to improve
  cluster structure

### 2. MathClass Clustering

Latent means `μ_i` of the training corpus are classified by the variable
dependence of their decoded expression:

| MathClass | Variables | Typical form |
|-----------|-----------|--------------|
| SPATIOTEMPORAL_3D | x, y, t | `exp(-r²/σt) · cos(ωr - φt)` |
| SPATIOTEMPORAL_2D | (x or y), t | `sin(kx) · exp(-αt)` |
| SPATIAL_2D | x, y | `sin(πx) · sin(πy)` |
| SPATIAL_1D | x or y | `tanh(ax + b)` |
| TEMPORAL_1D | t | `exp(-βt)` |
| CONSTANT | none | `3.14`, `-0.5` |

Each class forms a separate cluster database. Within each class, k-means
creates subclusters of related structural forms; k is caller-specified
(typical values: 2–6 for spatiotemporal classes, up to 100 for spatial-only
classes).

### 3. Subcluster Sampling

For each candidate expression the sampler:
1. Selects one subcluster per required `MathClass` role
2. Draws a random latent vector `z` from the cluster member pool
3. Decodes `z` via the GRU decoder using masked greedy decoding (argmax at each step, invalid rule logits masked to −∞)
4. Assembles multi-term expressions using the coherent-sum template (e.g., `f1 * f2 + f3`)

The `FlexibleVectorSampler` in `src/sigs/sampler.py` implements both
simple subcluster sampling (`sample_from_subclusters`) and the coherent-sum
method (`sample_coherent_sum_expressions`).

### 4. PDE Residual Scoring

Each candidate expression is evaluated on a structured mesh. The loss is:

```
L = w_pde · RMSE(PDE residual)
  + w_ic  · RMSE(initial condition error)
  + w_bc  · RMSE(boundary condition error)
```

Default weights are equal (`w_pde=w_ic=w_bc=1.0`). For problems where the PDE
residual should dominate (e.g., the shallow-water system loss), the caller
passes larger weights such as `w_pde=10000`.

Symbolic differentiation is performed with `symengine` for speed;
`sympy.lambdify` converts the result to a NumPy-vectorized function for
mesh evaluation. The top-k expressions by combined RMSE are retained for
Stage II.

---

## Stage II: Parameter Optimization

Given the symbolic structure `u*(x,y,t; θ)` from Stage I, Stage II treats
every numeric constant in the expression as a learnable parameter `θ_i` and
minimizes the PDE + IC + BC residuals via gradient descent.

Residuals are computed via JAX automatic differentiation through the
parametric solution functions (`src/sigs/loss.py`). This gives exact gradients
at zero approximation error. The optimizer is Adam with global gradient clipping
(`optax.chain(clip_by_global_norm(1.0), adam(lr))`), default lr = 5e-3,
for up to 2 000 iterations.

```python
# Simplified Stage II loop (see scripts/optimize.py for full version)
params = initial_params   # from Stage I best match
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
for step in range(2000):
    loss, grads = jax.value_and_grad(compute_loss)(params, sample_points)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

---

## Shallow Water: Specific Configuration

The 2-D shallow water system has three unknown fields:
`rho(x,y,t)`, `Sx(x,y) = ux/|u|`, `Sy(x,y) = uy/|u|`.

Recommended sampling configuration for Stage I:

```python
# rho: wave × Gaussian-decay × temporal-decay structure
rho_categories = {
    MathClass.SPATIOTEMPORAL_3D: 20,   # Gaussian-envelope forms
    MathClass.TEMPORAL_1D:        6,   # Temporal decay factors
}

# velocity: radial form x/sqrt(r²), y/sqrt(r²)
vel_categories = {
    MathClass.SPATIAL_2D: 100,   # 2-factor multiply gives radial forms
}
```

Increasing `SPATIOTEMPORAL_3D` subclusters from 2 to 20+ significantly
improves the probability of sampling the Gaussian-envelope structure
`exp(-r²/(σ(1+t)))` that characterizes the manufactured solution.

---

## References

- Kusner et al. (2017). *Grammar Variational Autoencoder.* ICML 2017.
  arXiv:1703.01925
- Cao et al. (2024). *An Interpretable Approach to High-Dimensional PDEs.*
  AAAI 2024. doi:10.1609/aaai.v38i18.30050
- Wei et al. (2024). *Closed-form Solutions: A New Perspective on Solving
  Differential Equations.* ICML 2025. arXiv:2405.14620
