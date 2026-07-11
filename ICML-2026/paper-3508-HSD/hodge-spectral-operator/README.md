# Hodge Spectral Operator (HSO)

Neural operator learning on manifolds with Hodge decomposition inductive bias.

## Features

- **Any geometric input**: mesh, point cloud, or graph — one unified interface
- **Any differential form task**: 0→0, 0→1, 1→0, 1→1, 0→2, etc.
- **Hodge inductive bias**: de Rham cross-form operators (grad, div, curl) as physics priors
- **Pseudo-spectral bilinear layers**: spectral convolution theorem for PDE nonlinearities
- **Whitney-KDE spatial branch**: mesh-aware Whitney form interpolation + KDE smoothing for high-frequency residuals
- **Adaptive dual-branch**: neural gating auto-allocates between spectral and spatial
- **Commutator correction**: inter-branch disagreement learning
- **Fully configurable**: spectral modes, layer depth/width, dropout, learning rate, etc.

## Install

```bash
pip install hodge-spectral-operator
```

## Quick Start

```python
from hodge_spectral import HodgeOperator

# From mesh: 0-form → 1-form (scalar source → vector velocity)
model = HodgeOperator.from_mesh(points, faces, task="0to1")
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
metrics = model.evaluate(X_test, Y_test)
# {'relative_l2': 0.025, 'riemannian_ip_fidelity': 0.997, 'mse': ...}
```

### Input Modes

```python
# From triangulated mesh (direct)
model = HodgeOperator.from_mesh(points, faces, task="0to1")

# From point cloud (auto-triangulates via KNN + Alpha complex)
model = HodgeOperator.from_pointcloud(points, task="0to0")

# From graph (finds triangles or falls back to Delaunay)
model = HodgeOperator.from_graph(edge_index, n_nodes, positions, task="0to1")
```

### Task Types

| Task | Input | Output | Example |
|------|-------|--------|---------|
| `"0to0"` | Scalar on nodes | Scalar on nodes | Heat diffusion, advection |
| `"0to1"` | Scalar on nodes | Vector on nodes | Darcy flow, Navier-Stokes |
| `"1to0"` | Vector on nodes | Scalar on nodes | Pressure from velocity |
| `"0to2"` | Scalar on nodes | Density on faces | Flux prediction |

## Hyperparameters

All architecture and training parameters are configurable:

```python
model = HodgeOperator.from_mesh(points, faces,
    task="0to1",

    # --- Spectral decomposition ---
    k=64,                       # eigenmodes per form (higher = more expressive)

    # --- Spectral branch ---
    hidden_dims=(256, 192),     # MLP layers: depth = len(tuple), width = values
    dropout=0.05,               # spectral branch dropout

    # --- Whitney-KDE spatial branch ---
    res_hidden=128,             # Whitney-KDE encoder/decoder hidden size
    res_dropout=0.1,            # spatial branch dropout

    # --- Training defaults (overridable in .fit()) ---
    default_lr=3e-3,            # learning rate
    default_epochs=100,         # max training epochs
    default_patience=25,        # early stopping patience
)

# Training can override any default:
model.fit(X_train, Y_train,
    epochs=200,                 # override
    lr=1e-3,                    # override
    batch_size=128,
    weight_decay=1e-4,
    patience=40,
)
```

### Configuration Presets

```python
# Lightweight (fast prototyping)
model = HodgeOperator.from_mesh(pts, fcs, task="0to1",
    k=32, hidden_dims=(64,), res_hidden=32, default_epochs=50)

# Standard (good default)
model = HodgeOperator.from_mesh(pts, fcs, task="0to1")

# High capacity (large datasets)
model = HodgeOperator.from_mesh(pts, fcs, task="0to1",
    k=128, hidden_dims=(512, 256, 128), res_hidden=256, dropout=0.1)
```

## Architecture

```
Input f (0-form) → Spectral Lift: c₀ = f·Φ₀, c₁ = d(f)·Φ₁
  │
  ├─ De Rham cross-terms: div(c₁) = δ₀(c₁), grad(c₀) = d₀(c₀)
  │
  ▼
Pseudo-Spectral Bilinear Layer × L
  │  linear path:   GELU(W · x)
  │  bilinear path: W_q · [c₀ ⊙ δ(c₁), c₁ ⊙ d(c₀)]   ← spectral convolution theorem
  │  output:        LayerNorm(linear + bilinear + skip)
  │
  ├──────────── latent ────────────────────┐
  │                                        │
  ▼                                        ▼
Spectral Branch                   Whitney-KDE Spatial Branch
  │ head → Φ₀ coefficients          │ Whitney interpolation on
  │ → decode to physical space       │ simplicial complex + KDE
  │                                  │ smoothing → decode residual
  ▼                                  ▼
  base (low-frequency)          residual (high-frequency)
  │                                  │
  ├──────── Neural Gate ─────────────┤
  │  α(x) = σ(g(latent)) ∈ (0,1)   │
  │  auto-allocates spectral/spatial │
  │                                  │
  └──────── Commutator ─────────────┘
     correction = MLP(base, res, latent)
     pred = gated + correction
```

**Dual-branch design:**
- **Spectral branch** predicts in the Hodge eigenbasis (Φ₀). Captures global low-frequency structure via de Rham operators.
- **Whitney-KDE spatial branch** uses [Whitney form](https://en.wikipedia.org/wiki/Whitney_forms) interpolation on the simplicial complex as a mesh-aware encoder, smoothed by a Kernel Density Estimator. Captures local high-frequency residuals that the spectral truncation misses.
- **Neural gate** learns per-node, per-sample mixing weights — automatically allocating capacity between branches based on the frequency content of each input.
- **Commutator** corrects inter-branch disagreement by taking both predictions + spectral context as input.

## Benchmarks

Relative L2 error (lower is better):

| Task | Topology | HSO (ours) | FNO | DeepONet |
|------|----------|------------|-----|---------|
| Ellipsoid Aero | genus-0, 0→1 | **0.025** | 0.259 | 0.113 |
| Torus Helmholtz | genus-1, 0→1 | **0.049** | 0.418 | 0.277 |

Cross-input consistency (same model, different input representations):

| Input Mode | Ellipsoid | Torus |
|------------|-----------|-------|
| Mesh | 0.025 | 0.049 |
| Point Cloud | 0.026 | 0.050 |
| Graph | 0.025 | 0.049 |
| **Spread** | **0.001** | **0.001** |

## Built-in Examples

```bash
# Ellipsoid external aerodynamics (genus-0, 0→1)
python examples/example_ellipsoid_aero.py

# Torus Helmholtz vortex flow (genus-1, 0→1)
python examples/example_torus_helmholtz.py

# Minimal quickstart
python examples/quickstart.py
```

## Save / Load

```python
model.save("my_model.pt")
model.load("my_model.pt")
```

## License

MIT
