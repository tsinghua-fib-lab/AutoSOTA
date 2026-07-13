[NbConvertApp] Converting notebook EX1_SDE_repro.ipynb to python
#!/usr/bin/env python
# coding: utf-8

# # SDE LieDynNet / FP Symmetry Discovery — Quick Cell Guide (Short)

# ## What this notebook does (3 stages)
# **A) Learn SDE surrogate** (f̂, σ̂) from simulated paths  
# **B) Learn FP density surrogate** û(x,t) using (f̂, σ̂)  
# **C) Learn symmetry generators** X_i = τ_i(t)∂_t + ξ_i(t,x)∂_x (and β_i for FP), via losses + training loop
# 
# ---
# 
# ## Stage A — Neural SDE surrogate
# 1) **Install + imports + float64**
# 2) **Data generation** `make_bm_data` → `t, x` (edit `cfg.x0_mode`, `sigma0`, `T`, `dt`, `n_traj`)
# 3) **Increment dataset** `build_increment_dataset` → `X_raw=[t_n,x_n]`, `dX=Δx`
# 4) **Normalize inputs** `in_norm = fit_normalizer(X_raw)` → `X = in_norm(X_raw)`
# 5) **MLP (t,x)→(f̂,σ̂)** + **increment NLL training** → exports:
#    - `params_sde`, `in_norm_sde`, `cfg`, `loss_history_sde`
# 6) (Optional) drift/diffusion plots + animations
# 
# ---
# 
# ## Stage B — Neural FP surrogate û(x,t)
# 7) **Define Ω domain** from path percentiles + pad; build **KDE u0** from x[:,0]
# 8) **FP density net**: `u_raw=softplus(nn)+eps`, then **normalize on Ω** via MC integral
# 9) **Train FP PINN** with:
#    - FP residual: u_t + ∂x(fu) − ½∂xx(σ²u)
#    - IC match: u(t0,x) ≈ u0_kde(x)
#    → exports: `params_fp`, `x_min_fp`, `x_max_fp`, `hist_fp`
# 10) (Optional) **Compare vs analytic FP** (BM case)
# 
# ---
# 
# ## Stage C — Generator nets + symmetry training
# 11) **Surrogate eval helper** `surrogate_f_sigma(...)`
# 12) **Generate TX_gen**: simulate surrogate paths, flatten to cloud `TX_gen` (training samples for sym losses)
# 13) **Init generator nets** (set in `GenConfig`):
#    - τ_i(t) net, ξ_i(t,x) net, β_i(t,x) net (β used for general FP sym)
#    → exports: `params_gen`, `eval_generators(_jit)`, normalizers
# 
# ---
# 
# ## Loss blocks
# 14) **Algebraic losses**: S1–S5 (closure, Jacobi, skew, bilinear, independence)
# 15) Choose physics constraints:
#    - **SDE sym**: S6 (Ito determining eq), S7 (finite-ε pushforward on μ,σ)
#    - **FP sym**: S8 (FP determining eq with β), S9 (after-flow/pushforward on u)
# 
# ---
# 
# ## IMPORTANT: the “Knob” (switch FP symmetry mode)
# In the **Knob** cell choose ONE:
# 
# ### (A) Learn **all FP symmetries** (β learned by net)
# ```python
# master_loss = master_loss_fp
# master_loss_jit = master_loss_fp_jit
# ```
# 
# ### (B) Learn FP symmetries **under normalization constraint**
# Hard constraint: β(t,x) = −∂x ξ(t,x)
# ```python
# master_loss = master_loss_fp_norm
# master_loss_jit = master_loss_fp_norm_jit
# ```
# 
# > After switching the knob, rerun **Generator Training** (and usually re-init `params_gen`).
# 
# ---
# 
# ## Generator training + eval
# 16) **Generator training loop**: optimizes `params_gen` using minibatches from `TX_gen`
# 17) **Evaluations**:
#    - τ_i(t) curves, ξ_i(t,x) heatmaps
#    - (If m=3) span/principal-angle check vs known BM sym basis
# 
# ---
# 
# ## Common reruns
# - **Change only generator settings/weights** → (re-init gen optional) → knob → train → eval  
# - **Change SDE surrogate** → rerun Stage A → Stage B → regenerate TX_gen → gen init/train  
# - **Change FP surrogate** → rerun Stage B → knob → gen train  
# 

# In[ ]:


# Install (if needed)
# pip install --quiet jax jaxlib optax dm-haiku

import math
import functools
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

jax.config.update("jax_enable_x64", True)

import itertools
from functools import partial

import matplotlib.pyplot as plt

Array = jnp.ndarray


# In[ ]:


print(jnp.array([0.]).dtype)   # should print float64


# In[ ]:


# @title Neural SDE surrogate for dx_t = σ₀ dW_t (1D, constant noise example) + plots
# Runs as-is in Google Colab (JAX + optax)

import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

try:
    import optax
    OPTAX_AVAILABLE = True
except Exception:
    OPTAX_AVAILABLE = False
    raise RuntimeError("This cell expects optax to be available in Colab.")

# Use float64
jax.config.update("jax_enable_x64", True)

# -------------------------------------------------------------------
# 0. Config and helpers
# -------------------------------------------------------------------

@dataclass
class CFG:
    # Ground-truth SDE: dx_t = σ0 dW_t
    sigma0: float = 1.0

    # Time grid
    T: float = 5.0          # final time
    dt: float = 0.01        # time step
    n_traj: int = 2048       # number of trajectories

    # Initial condition distribution for x0 across trajectories
    x0_mode: str = "point"   # "point", "normal", "uniform", "mixture_points", "grid_points"
    x0_value: float = 0.0    # used if x0_mode == "point"

    # normal: x0 ~ N(x0_mean, x0_std^2)
    x0_mean: float = 0.0
    x0_std: float = 1.0

    # uniform: x0 ~ U(x0_low, x0_high)
    x0_low: float = -1.0
    x0_high: float = 1.0

    # mixture/grid of fixed points
    x0_points: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0)


    # NN + training
    hidden: int = 64
    steps: int = 10000
    batch_size: int = 4096
    lr: float = 3e-3
    weight_decay: float = 1e-6
    sigma_min: float = 1e-3  # lower bound on σ̂ for stability

cfg = CFG(x0_mode="uniform", x0_low=-3.0, x0_high=3.0) # change here for different IC distributions

key_main = jax.random.PRNGKey(0)

# -------------------------------------------------------------------
# 1. Data generation: sample Brownian paths for dx = σ0 dW
# -------------------------------------------------------------------

def make_bm_data(key, cfg: CFG):
    """
    Simulate Brownian motion with variance σ0^2 using Euler increments:
        x_{n+1} = x_n + σ0 * sqrt(dt) * ξ_n,   ξ_n ~ N(0,1).
    Returns:
      t : (N+1,)
      x : (n_traj, N+1)
    """
    sigma0 = cfg.sigma0
    dt = cfg.dt
    N = int(cfg.T / dt)

    t = jnp.linspace(0.0, cfg.T, N + 1)  # (N+1,)

    k_init, k_noise = jax.random.split(key, 2)

    # Start at x0 = 0 for all trajectories (or random if desired)
    #x0 = jnp.zeros((cfg.n_traj, 1), dtype=jnp.float64)  # (n_traj, 1)

    # --- UPDATED: sample x0 across trajectories according to cfg.x0_mode ---
    if cfg.x0_mode == "point":
        x0 = jnp.full((cfg.n_traj, 1), cfg.x0_value, dtype=jnp.float64)

    elif cfg.x0_mode == "normal":
        x0 = (cfg.x0_mean
          + cfg.x0_std * jax.random.normal(k_init, (cfg.n_traj, 1), dtype=jnp.float64))

    elif cfg.x0_mode == "uniform":
        x0 = jax.random.uniform(
            k_init, (cfg.n_traj, 1),
            minval=cfg.x0_low, maxval=cfg.x0_high,
            dtype=jnp.float64
        )

    elif cfg.x0_mode == "mixture_points":
        # Sample each trajectory's x0 from a discrete set of points (with replacement)
        pts = jnp.array(cfg.x0_points, dtype=jnp.float64)  # (K,)
        idx = jax.random.randint(k_init, (cfg.n_traj,), 0, pts.shape[0])
        x0 = pts[idx].reshape(cfg.n_traj, 1)

    elif cfg.x0_mode == "grid_points":
        # Deterministic: repeat a set of points to fill n_traj (needs divisibility or truncation)
        pts = jnp.array(cfg.x0_points, dtype=jnp.float64)  # (K,)
        K = pts.shape[0]
        reps = int(math.ceil(cfg.n_traj / K))
        x0 = jnp.tile(pts, reps)[:cfg.n_traj].reshape(cfg.n_traj, 1)

    else:
        raise ValueError(f"Unknown cfg.x0_mode: {cfg.x0_mode}")


    # Gaussian increments: (n_traj, N)
    dW = jax.random.normal(k_noise, (cfg.n_traj, N), dtype=jnp.float64) * math.sqrt(dt)
    dx = sigma0 * dW  # (n_traj, N)

    # Build full paths by cumulative sum
    x_increments = jnp.concatenate([x0, dx], axis=1)  # (n_traj, N+1)
    x = jnp.cumsum(x_increments, axis=1)

    return t, x

t, x = make_bm_data(key_main, cfg)
print("Data shapes: t =", t.shape, ", x =", x.shape)

# --- ADDED: plot a subset of raw trajectories x(t) ---
t_np = np.asarray(jax.device_get(t))
x_np = np.asarray(jax.device_get(x))

n_plot = min(30, cfg.n_traj)   # number of trajectories to plot
stride = max(1, len(t_np) // 500)  # downsample for speed/clarity

plt.figure(figsize=(7, 4))
for i in range(n_plot):
    plt.plot(t_np[::stride], x_np[i, ::stride], lw=1, alpha=0.6)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title(f"Sample Brownian trajectories (showing {n_plot} of {cfg.n_traj})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- ADDED: inspect raw training source paths (small slice) ---
t_np = np.asarray(jax.device_get(t))
x_np = np.asarray(jax.device_get(x))

print("\n[RAW PATH DATA]")
print("t[:10] =", t_np[:10])
print("x first traj, first 10 =", x_np[0, :10])
print("x traj stats: mean =", x_np.mean(), "std =", x_np.std(), "min =", x_np.min(), "max =", x_np.max())
print("x0 unique values (should be all 0):", np.unique(x_np[:, 0])[:10], "(showing up to 10)")


# -------------------------------------------------------------------
# 2. Build training dataset: (t_n, x_n) -> Δx_n
# -------------------------------------------------------------------

def build_increment_dataset(t, x, dt):
    """
    Given paths x[:, n], builds a dataset of:
      inputs: (t_n, x_n)
      targets: Δx_n = x_{n+1} - x_n
    """
    # x: (n_traj, N+1)
    x_n = x[:, :-1]      # (n_traj, N)
    x_np1 = x[:, 1:]     # (n_traj, N)
    dx = x_np1 - x_n     # (n_traj, N)

    # Broadcast t: (N+1,) -> (n_traj, N)
    t_n = jnp.broadcast_to(t[:-1], x_n.shape)

    # Flatten over (traj, time)
    t_flat = t_n.reshape(-1, 1)    # (B,1)
    x_flat = x_n.reshape(-1, 1)    # (B,1)
    dx_flat = dx.reshape(-1, 1)    # (B,1)

    # Stack inputs as (t,x)
    inp = jnp.concatenate([t_flat, x_flat], axis=1)  # (B,2)
    return inp, dx_flat

X_raw, dX = build_increment_dataset(t, x, cfg.dt)
print("Increment dataset shapes: X_raw =", X_raw.shape, ", dX =", dX.shape)

# --- ADDED: histogram of increments Δx ---
dX_np = np.asarray(jax.device_get(dX)).ravel()

# optional: subsample for speed if huge
max_hist = 200_000
if dX_np.size > max_hist:
    rng = np.random.default_rng(0)
    dX_plot = rng.choice(dX_np, size=max_hist, replace=False)
else:
    dX_plot = dX_np

plt.figure(figsize=(6, 4))
plt.hist(dX_plot, bins=80, density=True, alpha=0.8)
plt.xlabel(r"$\Delta x$")
plt.ylabel("density")
plt.title(r"Training increments distribution $\Delta x = x_{n+1}-x_n$")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- ADDED: scatter plot Δx vs x_n to visualize dependence ---
Xraw_np = np.asarray(jax.device_get(X_raw))  # columns: [t_n, x_n]
x_n_np = Xraw_np[:, 1]
dX_np = np.asarray(jax.device_get(dX)).ravel()

max_scatter = 50_000
if dX_np.size > max_scatter:
    rng = np.random.default_rng(1)
    idx = rng.choice(dX_np.size, size=max_scatter, replace=False)
    x_sc = x_n_np[idx]
    dx_sc = dX_np[idx]
else:
    x_sc = x_n_np
    dx_sc = dX_np

plt.figure(figsize=(6, 4))
plt.scatter(x_sc, dx_sc, s=2, alpha=0.25)
plt.xlabel(r"$x_n$")
plt.ylabel(r"$\Delta x_n$")
plt.title(r"Training data: scatter of increments vs state")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- ADDED: binned variance of Δx over time ---
Xraw_np = np.asarray(jax.device_get(X_raw))  # [t_n, x_n]
t_n_np = Xraw_np[:, 0]
dX_np = np.asarray(jax.device_get(dX)).ravel()

# bin by time into, say, 25 bins
n_bins = 25
bins = np.linspace(t_n_np.min(), t_n_np.max(), n_bins + 1)
bin_id = np.digitize(t_n_np, bins) - 1
bin_centers = 0.5 * (bins[:-1] + bins[1:])

var_by_bin = np.full(n_bins, np.nan)
mean_by_bin = np.full(n_bins, np.nan)
count_by_bin = np.zeros(n_bins, dtype=int)

for b in range(n_bins):
    m = (bin_id == b)
    count_by_bin[b] = int(m.sum())
    if count_by_bin[b] > 0:
        mean_by_bin[b] = dX_np[m].mean()
        var_by_bin[b] = dX_np[m].var()

plt.figure(figsize=(7, 4))
plt.plot(bin_centers, mean_by_bin, lw=2, label="mean(Δx) per time bin")
plt.plot(bin_centers, var_by_bin, lw=2, label="var(Δx) per time bin")
plt.axhline((cfg.sigma0**2) * cfg.dt, ls="--", lw=2, label="theory var = σ0² dt")
plt.xlabel("t (bin center)")
plt.title("Training increments summary vs time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --- ADDED: inspect increment training dataset (unnormalized inputs) ---
Xraw_np = np.asarray(jax.device_get(X_raw))
dX_np   = np.asarray(jax.device_get(dX))

print("\n[INCREMENT DATASET (UNNORMALIZED)]")
print("First 5 rows of X_raw = [t_n, x_n]:\n", Xraw_np[:5])
print("First 5 rows of dX = Δx_n:\n", dX_np[:5].ravel())

# empirical mean ~ 0, var ~ sigma0^2 * dt
print("Empirical mean(Δx) =", dX_np.mean(), " ; Empirical var(Δx) =", dX_np.var())
print("Theoretical var(Δx) = sigma0^2 * dt =", (cfg.sigma0 ** 2) * cfg.dt)


# -------------------------------------------------------------------
# 3. Normalization (z-score) for inputs (t,x)
# -------------------------------------------------------------------

class Normalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

def fit_normalizer(X):
    return Normalizer(jnp.mean(X, axis=0), jnp.std(X, axis=0))

in_norm = fit_normalizer(X_raw)
X = in_norm(X_raw)  # normalized inputs
print("Input mean ~", X.mean(0), "std ~", X.std(0))

# --- ADDED: inspect normalized network inputs ---
X_np = np.asarray(jax.device_get(X))

print("\n[NORMALIZED INPUTS USED BY THE NET]")
print("First 5 rows of X = normalized [t_n, x_n]:\n", X_np[:5])
print("Per-feature mean(X) ~", X_np.mean(axis=0), " ; std(X) ~", X_np.std(axis=0))


# -------------------------------------------------------------------
# 4. MLP model: (t,x) -> (f̂, g) with σ̂ = softplus(g) + σ_min
# -------------------------------------------------------------------

def glorot(key, fan_in, fan_out):
    lim = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-lim, maxval=lim)

def init_mlp_params(key, sizes):
    """
    sizes: list of layer widths [in_dim, h1, h2, ..., out_dim]
    returns: list of {'W', 'b'} dicts
    """
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        params.append({
            "W": glorot(k, m, n),
            "b": jnp.zeros((n,), dtype=jnp.float64),
        })
    return params

def mlp_forward(params, x, activation="tanh"):
    h = x
    for i, layer in enumerate(params):
        W, b = layer["W"], layer["b"]
        h = h @ W + b
        if i < len(params) - 1:
            if activation == "tanh":
                h = jnp.tanh(h)
            elif activation == "relu":
                h = jax.nn.relu(h)
            else:
                raise ValueError(f"Unknown activation {activation}")
    return h  # (..., out_dim)

def f_sigma_hat(params, tx_norm, activation="tanh", sigma_min=1e-3):
    """
    tx_norm: (..., 2) normalized [t,x].
    Returns:
      f_hat: drift estimate
      sigma_hat: diffusion estimate (positive)
    """
    out = mlp_forward(params, tx_norm, activation=activation)  # (..., 2)
    f_hat = out[..., 0:1]           # (...,1)
    g = out[..., 1:2]               # (...,1)
    sigma_hat = jax.nn.softplus(g) + sigma_min
    return f_hat, sigma_hat

# Init model params
key_main, k_model = jax.random.split(key_main, 2)
params_sde = init_mlp_params(k_model, sizes=[2, cfg.hidden, cfg.hidden, 2])

# -------------------------------------------------------------------
# 5. Increment-based NLL loss
# -------------------------------------------------------------------

N = X.shape[0]
print("Total samples:", N)

def make_increment_loss(dt, sigma_min, weight_decay):

    def loss_fn(params, xb, dxb):
        """
        xb: (B,2) normalized inputs [t,x]
        dxb: (B,1) increments Δx
        """
        f_hat, sigma_hat = f_sigma_hat(params, xb, activation="tanh", sigma_min=sigma_min)
        # mean and variance for increments
        mu = f_hat * dt                      # (B,1)
        var = (sigma_hat ** 2) * dt          # (B,1)

        # Negative log-likelihood (up to constant 0.5 log(2π))
        residual = dxb - mu
        ell = (residual ** 2) / (2.0 * var) + 0.5 * jnp.log(var + 1e-12)

        # Average over minibatch
        nll = jnp.mean(ell)

        # Weight decay
        def l2_tree(p):
            return sum([jnp.sum(v**2) for v in jax.tree_util.tree_leaves(p)])

        reg = weight_decay * l2_tree(params)
        return nll + reg

    return loss_fn

loss_fn = make_increment_loss(cfg.dt, cfg.sigma_min, cfg.weight_decay)
loss_fn_jit = jax.jit(loss_fn)

# -------------------------------------------------------------------
# 6. Training loop (Adam + minibatches) + record loss for plotting
# -------------------------------------------------------------------

if not OPTAX_AVAILABLE:
    raise RuntimeError("optax is required for this training loop.")

optimizer = optax.adam(cfg.lr)
opt_state = optimizer.init(params_sde)

rng_np = np.random.default_rng(0)

@jax.jit
def train_step(params, opt_state, xb, dxb):
    def _loss(p):
        return loss_fn(p, xb, dxb)
    val, grads = jax.value_and_grad(_loss)(params)
    updates, opt_state_new = optimizer.update(grads, opt_state, params)
    params_new = optax.apply_updates(params, updates)
    return params_new, opt_state_new, val

def sample_minibatch(X, dX, batch_size):
    N = X.shape[0]
    if batch_size >= N:
        idx = np.arange(N)
    else:
        idx = rng_np.choice(N, size=batch_size, replace=False)
    xb = X[idx]
    dxb = dX[idx]
    return xb, dxb

print_every = 100
loss_history = []

for step in range(1, cfg.steps + 1):
    xb, dxb = sample_minibatch(X, dX, cfg.batch_size)
    xb = jnp.asarray(xb)
    dxb = jnp.asarray(dxb)

    params_sde, opt_state, loss_val = train_step(params_sde, opt_state, xb, dxb)
    loss_float = float(loss_val)
    loss_history.append(loss_float)

    if step % print_every == 0 or step == 1 or step == cfg.steps:
        print(f"step {step:5d}/{cfg.steps} | NLL+reg = {loss_float:.6e}")

print("\nTraining finished.")

# -------------------------------------------------------------------
# 7. Plot training loss curve
# -------------------------------------------------------------------

steps_arr = np.arange(1, cfg.steps + 1)

plt.figure(figsize=(6, 4))
plt.plot(steps_arr, loss_history, lw=2)
plt.xlabel("Training step")
plt.ylabel("NLL + reg")
plt.title("Training loss for neural SDE surrogate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 8. Quick diagnostics: learned f̂ and σ̂ on a small grid
# -------------------------------------------------------------------

def eval_model(params, t_vals, x_vals):
    TT, XX = jnp.meshgrid(t_vals, x_vals, indexing="ij")
    TX = jnp.stack([TT.ravel(), XX.ravel()], axis=-1)   # (B,2)
    TX_norm = in_norm(TX)
    f_hat, sigma_hat = f_sigma_hat(params, TX_norm, activation="tanh", sigma_min=cfg.sigma_min)
    return TT, XX, f_hat.reshape(TT.shape), sigma_hat.reshape(TT.shape)

t_eval = jnp.linspace(0.0, cfg.T, 21)
x_eval = jnp.linspace(-2.0, 2.0, 21)

TT, XX, F_hat_grid, Sigma_hat_grid = eval_model(params_sde, t_eval, x_eval)

print("\nGround-truth σ0 =", cfg.sigma0)
print("Mean σ̂ over eval grid:", float(Sigma_hat_grid.mean()))
print("Std  σ̂ over eval grid:", float(Sigma_hat_grid.std()))
print("Mean |f̂| over eval grid:", float(jnp.abs(F_hat_grid).mean()))

# Heatmaps of f̂ and σ̂
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# σ̂(t,x)
im0 = axes[0].imshow(
    np.asarray(Sigma_hat_grid),
    origin="lower",
    extent=(float(x_eval.min()), float(x_eval.max()),
            float(t_eval.min()), float(t_eval.max())),
    aspect="auto"
)
axes[0].set_title(r"$\hat{\sigma}(t,x)$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(im0, ax=axes[0], shrink=0.9)

# f̂(t,x)
im1 = axes[1].imshow(
    np.asarray(F_hat_grid),
    origin="lower",
    extent=(float(x_eval.min()), float(x_eval.max()),
            float(t_eval.min()), float(t_eval.max())),
    aspect="auto"
)
axes[1].set_title(r"$\hat{f}(t,x)$")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(im1, ax=axes[1], shrink=0.9)

plt.show()

# -------------------------------------------------------------------
# 9. Export trained objects to globals for later stages (optional)
# -------------------------------------------------------------------

globals().update({
    "cfg": cfg,
    "params_sde": params_sde,
    "in_norm_sde": in_norm,
    "loss_history_sde": list(zip(steps_arr, loss_history)),
})

print("\nExported: params_sde, in_norm_sde, cfg, loss_history_sde.")


# In[ ]:


# Helper: evaluate surrogate drift and diffusion at (t, x)
# Requires: params_sde, in_norm_sde, f_sigma_hat, cfg, jax, jnp

def surrogate_f_sigma(params_sde, in_norm_sde, t, x, activation="tanh"):
    """
    Evaluate the learned SDE surrogate (f̂, σ̂) at (t, x).

    Inputs:
      t : scalar, 1D array, or broadcastable with x
      x : scalar, 1D array, or broadcastable with t
    Returns:
      f_hat      : array with same broadcasted shape as t and x
      sigma_hat  : array with same broadcasted shape as t and x
    """
    t_arr = jnp.asarray(t, dtype=jnp.float64)
    x_arr = jnp.asarray(x, dtype=jnp.float64)

    # Broadcast t and x to a common shape
    t_b, x_b = jnp.broadcast_arrays(t_arr, x_arr)

    # Stack into (B, 2) with B = number of (t,x) points
    tx = jnp.stack([t_b.ravel(), x_b.ravel()], axis=-1)   # (B, 2)

    # Normalize using the training-time normalizer
    tx_norm = in_norm_sde(tx)

    # Forward through neural surrogate
    f_hat_flat, sigma_hat_flat = f_sigma_hat(
        params_sde,
        tx_norm,
        activation=activation,
        sigma_min=cfg.sigma_min,
    )

    # Reshape back to broadcasted shape
    f_hat = f_hat_flat.reshape(t_b.shape)
    sigma_hat = sigma_hat_flat.reshape(t_b.shape)

    return f_hat, sigma_hat


# In[ ]:


# Generate data from the learned neural SDE surrogate for Stage-2 training
# Requires: surrogate_f_sigma, params_sde, in_norm_sde, cfg, jax, jnp, np, matplotlib

from dataclasses import dataclass
import math

@dataclass
class CFGGen:
    T: float = cfg.T           # reuse same horizon
    dt: float = cfg.dt         # reuse same time step
    n_traj: int = 256          # fewer trajectories than Stage 1 (enough for (t,x) sampling)

cfg_gen = CFGGen()

def simulate_surrogate_paths(key, params_sde, in_norm_sde, cfg_gen: CFGGen):
    """
    Simulate paths from the neural SDE surrogate using Euler–Maruyama:
        x_{n+1} = x_n + f̂(t_n, x_n) * dt + σ̂(t_n, x_n) * sqrt(dt) * ξ_n.
    Returns:
      t_sim : (N+1,) time grid
      x_sim : (n_traj, N+1) simulated paths
    """
    dt = cfg_gen.dt
    T = cfg_gen.T
    n_traj = cfg_gen.n_traj

    N = int(T / dt)
    t_sim = jnp.linspace(0.0, T, N + 1, dtype=jnp.float64)

    # Initial condition: x(0) = 0 for all trajectories
    x0 = jnp.zeros((n_traj,), dtype=jnp.float64)

    # Pre-generate Gaussian noise
    key_noise, _ = jax.random.split(key)
    dW = jax.random.normal(
        key_noise,
        shape=(N, n_traj),
        dtype=jnp.float64,
    ) * math.sqrt(dt)

    def step(x_t, inputs):
        k_idx, t_curr = inputs  # k_idx is just a dummy index here
        # Evaluate surrogate drift/diffusion at current time and state
        t_vec = jnp.full_like(x_t, t_curr)
        f_hat, sigma_hat = surrogate_f_sigma(params_sde, in_norm_sde, t_vec, x_t)
        # Use pre-generated dW[k_idx]
        dW_t = dW[k_idx]
        x_next = x_t + f_hat * dt + sigma_hat * dW_t
        return x_next, x_next

    # Use lax.scan over time steps
    idxs = jnp.arange(N, dtype=jnp.int32)
    _, x_hist = jax.lax.scan(
        step,
        x0,
        (idxs, t_sim[:-1]),
    )  # x_hist: (N, n_traj)

    # Stack initial condition
    x_sim = jnp.concatenate([x0[None, :], x_hist], axis=0)  # (N+1, n_traj)
    x_sim = x_sim.T  # (n_traj, N+1) for consistency with earlier code

    return t_sim, x_sim

# Simulate surrogate paths
key_main, key_surr = jax.random.split(key_main)
t_surr, x_surr = simulate_surrogate_paths(key_surr, params_sde, in_norm_sde, cfg_gen)

print("Surrogate sim shapes: t_surr =", t_surr.shape, ", x_surr =", x_surr.shape)

# Build flattened (t, x) dataset for generator training
def build_tx_dataset(t, x):
    """
    Flatten paths into a cloud of (t_n, x_n) points:
      t : (N+1,)
      x : (n_traj, N+1)
    Returns:
      TX_gen : (B, 2) where B = n_traj * (N+1)
    """
    n_traj, Np1 = x.shape
    # Broadcast t to match (n_traj, N+1)
    t_b = jnp.broadcast_to(t[None, :], x.shape)  # (n_traj, N+1)

    t_flat = t_b.reshape(-1, 1)
    x_flat = x.reshape(-1, 1)
    TX_gen = jnp.concatenate([t_flat, x_flat], axis=1)
    return TX_gen

TX_gen = build_tx_dataset(t_surr, x_surr)
print("TX_gen shape (flattened (t,x) pairs):", TX_gen.shape)

# Export to globals for later use
globals().update({
    "t_surr": t_surr,
    "x_surr": x_surr,
    "TX_gen": TX_gen,
    "cfg_gen": cfg_gen,
})

# Quick plot: sample a few surrogate paths to visualize data
import matplotlib.pyplot as plt
import numpy as np

n_plot = min(20, x_surr.shape[0])  # plot up to 20 paths
idx_plot = np.linspace(0, x_surr.shape[0] - 1, n_plot, dtype=int)

plt.figure(figsize=(6, 4))
for idx in idx_plot:
    plt.plot(np.asarray(t_surr), np.asarray(x_surr[idx]), alpha=0.6)

plt.xlabel("t")
plt.ylabel("x_t (surrogate paths)")
plt.title("Sample trajectories from neural SDE surrogate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[ ]:


# Generator neural net setup: τ_i(t) and ξ_i(t,x) for SDE symmetries
# Requires:
#   - jax, jax.numpy as jnp
#   - init_mlp_params, mlp_forward
#   - Normalizer, fit_normalizer
#   - TX_gen (flattened (t,x) pairs from surrogate simulation)
#   - key_main (PRNG key)
#   - cfg (for dtype / consistency)

from dataclasses import dataclass
import jax
import jax.numpy as jnp

# ----------------- Config for generator networks ------------------------------

@dataclass
class GenConfig:
    n_generators: int = 6
    hidden_tau: int = 32
    hidden_xi: int = 64
    hidden_beta: int = 64     # NEW: width for β(t,x) MLP
    activation: str = "tanh"


gen_cfg = GenConfig()

# ----------------- Normalizers for generator inputs ---------------------------

# TX_gen has shape (B, 2) with columns [t, x]
t_flat_gen = TX_gen[:, 0:1]   # (B,1)
# We could also normalize x separately if desired, but a joint (t,x) norm is handy.
tx_norm_gen = fit_normalizer(TX_gen)       # for ξ(t,x)
t_norm_gen = fit_normalizer(t_flat_gen)    # for τ(t)

print("Gen t_norm mean/std:", t_norm_gen.mean, t_norm_gen.std)
print("Gen tx_norm mean/std:", tx_norm_gen.mean, tx_norm_gen.std)

# ----------------- Per-generator network builders -----------------------------

def tau_forward(params, t_norm, activation="tanh"):
    """
    Forward pass for τ(t).
    t_norm: (..., 1) normalized time.
    """
    return mlp_forward(params, t_norm, activation=activation)[..., 0:1]

def xi_forward(params, tx_norm, activation="tanh"):
    """
    Forward pass for ξ(t,x).
    tx_norm: (..., 2) normalized (t,x).
    """
    return mlp_forward(params, tx_norm, activation=activation)[..., 0:1]

def beta_forward(params, tx_norm, activation="tanh"):
    """
    Forward pass for β(t,x).
    tx_norm: (..., 2) normalized (t,x).
    Returns (..., 1).
    """
    return mlp_forward(params, tx_norm, activation=activation)[..., 0:1]

def init_generator_params(key, gen_cfg: GenConfig):
    """
    Initialize parameters for all generators.
    Returns:
      params_gen = {
        "tau":  [params_tau_i  for i in range(m)],
        "xi":   [params_xi_i   for i in range(m)],
        "beta": [params_beta_i for i in range(m)],   # NEW
      }
    """
    m = gen_cfg.n_generators
    keys = jax.random.split(key, 3 * m)

    params_tau_list = []
    params_xi_list = []
    params_beta_list = []

    for i in range(m):
        k_tau  = keys[3 * i]
        k_xi   = keys[3 * i + 1]
        k_beta = keys[3 * i + 2]

        # τ_i(t): input dim 1 -> hidden_tau -> hidden_tau -> output dim 1
        params_tau = init_mlp_params(
            k_tau,
            sizes=[1, gen_cfg.hidden_tau, gen_cfg.hidden_tau, 1],
        )

        # ξ_i(t,x): input dim 2 -> hidden_xi -> hidden_xi -> output dim 1
        params_xi = init_mlp_params(
            k_xi,
            sizes=[2, gen_cfg.hidden_xi, gen_cfg.hidden_xi, 1],
        )

        # β_i(t,x): input dim 2 -> hidden_beta -> hidden_beta -> output dim 1  (NEW)
        params_beta = init_mlp_params(
            k_beta,
            sizes=[2, gen_cfg.hidden_beta, gen_cfg.hidden_beta, 1],
        )

        params_tau_list.append(params_tau)
        params_xi_list.append(params_xi)
        params_beta_list.append(params_beta)

    return {
        "tau": params_tau_list,
        "xi": params_xi_list,
        "beta": params_beta_list,   # NEW
    }


# Initialize generator parameters
key_main, key_gen = jax.random.split(key_main)
params_gen = init_generator_params(key_gen, gen_cfg)

# ----------------- Convenience evaluator for all generators -------------------

def eval_generators(params_gen, t, x, u=None, activation=None, return_phi=False):
    """
    Evaluate all generators at (t,x).

    If return_phi=True, pass u (same shape as broadcast(t,x)),
    and we return phi = beta(t,x) * u.

    Returns:
      tau_vals : (m, ...) τ_i(t)
      xi_vals  : (m, ...) ξ_i(t,x)
      beta_vals: (m, ...) β_i(t,x)
      phi_vals : (m, ...) φ_i(t,x,u) = β_i(t,x) * u   (optional)
    """
    if activation is None:
        activation = gen_cfg.activation

    t_arr = jnp.asarray(t, dtype=jnp.float64)
    x_arr = jnp.asarray(x, dtype=jnp.float64)

    # Broadcast t and x to common shape
    t_b, x_b = jnp.broadcast_arrays(t_arr, x_arr)
    B_shape = t_b.shape

    # Flatten
    t_flat = t_b.reshape(-1, 1)
    x_flat = x_b.reshape(-1, 1)
    tx_flat = jnp.concatenate([t_flat, x_flat], axis=1)

    # Normalize
    t_norm  = t_norm_gen(t_flat)     # (B,1)
    tx_norm = tx_norm_gen(tx_flat)   # (B,2)

    tau_list, xi_list, beta_list = [], [], []

    for params_tau, params_xi, params_beta in zip(
        params_gen["tau"], params_gen["xi"], params_gen["beta"]
    ):
        tau_flat  = tau_forward(params_tau, t_norm, activation=activation)        # (B,1)
        xi_flat   = xi_forward(params_xi,  tx_norm, activation=activation)        # (B,1)
        beta_flat = beta_forward(params_beta, tx_norm, activation=activation)     # (B,1)

        tau_list.append(tau_flat.reshape(B_shape))
        xi_list.append(xi_flat.reshape(B_shape))
        beta_list.append(beta_flat.reshape(B_shape))

    tau_vals  = jnp.stack(tau_list, axis=0)    # (m, ...)
    xi_vals   = jnp.stack(xi_list, axis=0)     # (m, ...)
    beta_vals = jnp.stack(beta_list, axis=0)   # (m, ...)

    if return_phi:
        if u is None:
            raise ValueError("return_phi=True requires u to be provided.")
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        u_b = jnp.broadcast_to(u_arr, B_shape)  # (...,)
        phi_vals = beta_vals * u_b[None, ...]   # (m, ...)
        return tau_vals, xi_vals, beta_vals, phi_vals

    return tau_vals, xi_vals, beta_vals


# JIT-ed version for speed if desired
eval_generators_jit = jax.jit(
    eval_generators,
    static_argnames=("activation", "return_phi")
)

print(f"Initialized generator nets with m = {gen_cfg.n_generators} generators.")

# Export to globals for later stages (invariance loss, Lie algebra loss, etc.)
globals().update({
    "gen_cfg": gen_cfg,
    "params_gen": params_gen,
    "t_norm_gen": t_norm_gen,
    "tx_norm_gen": tx_norm_gen,
    "tau_forward": tau_forward,
    "xi_forward": xi_forward,
    "beta_forward": beta_forward,   # NEW
    "eval_generators": eval_generators,
    "eval_generators_jit": eval_generators_jit,
})



# In[ ]:


import jax
import jax.numpy as jnp

def eval_generators_tau_xi(params_gen, t, x, *, activation="tanh", normalize_tx=None):
    """
    SDE generator evaluator: returns only (tau, xi).

    Inputs:
      t, x: (B,) arrays
    Returns:
      tau_all: (m, B)
      xi_all:  (m, B)
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("eval_generators_tau_xi expects t and x as 1D arrays of shape (B,)")

    # Optional preprocessing in raw (t,x) space
    if normalize_tx is None:
        t_raw, x_raw = t, x
    else:
        t_raw, x_raw = normalize_tx(t, x)
        t_raw = jnp.asarray(t_raw, dtype=jnp.float64)
        x_raw = jnp.asarray(x_raw, dtype=jnp.float64)

    # Build (B,1) and (B,2) inputs
    t_col  = t_raw.reshape(-1, 1)                         # (B,1)
    x_col  = x_raw.reshape(-1, 1)                         # (B,1)
    tx_col = jnp.concatenate([t_col, x_col], axis=1)      # (B,2)

    # Apply the SAME normalizers used in generator training
    t_norm  = t_norm_gen(t_col)       # (B,1)
    tx_norm = tx_norm_gen(tx_col)     # (B,2)

    taus = []
    xis  = []

    for params_tau, params_xi in zip(params_gen["tau"], params_gen["xi"]):
        tau_flat = tau_forward(params_tau, t_norm, activation=activation).reshape(-1)  # (B,)
        xi_flat  = xi_forward(params_xi,  tx_norm, activation=activation).reshape(-1) # (B,)
        taus.append(tau_flat)
        xis.append(xi_flat)

    tau_all = jnp.stack(taus, axis=0)  # (m,B)
    xi_all  = jnp.stack(xis,  axis=0)  # (m,B)
    return tau_all, xi_all

# JIT wrapper (recommended)
eval_generators_tau_xi_jit = jax.jit(eval_generators_tau_xi, static_argnames=("activation", "normalize_tx"))


# In[ ]:


# ============================ S1 — Lie bracket closure + constancy (fixed version) ============================
# Compatible with:
#   - params_gen: {"tau": [params_tau_i], "xi": [params_xi_i]}
#   - gen_cfg.n_generators
#   - t_norm_gen, tx_norm_gen
#   - tau_forward, xi_forward
#
# Vector fields: X_i = τ_i(t) ∂_t + ξ_i(t,x) ∂_x.
# We enforce:
#   - [X_i, X_j] stays in span{X_k} via projection
#   - structure coefficients c_ij^k approximately constant over (t,x).

import jax
import jax.numpy as jnp

def _ordered_pair_indices(n: int):
    """
    Return (i,j) pairs with i != j, in a deterministic order.
    Used to enumerate all brackets [X_i, X_j].
    """
    idx = jnp.arange(n, dtype=jnp.int32)
    ii  = jnp.repeat(idx, repeats=n-1)
    base = jnp.arange(n - 1, dtype=jnp.int32)
    i_col = idx[:, None]
    jj_mat = base + (base >= i_col).astype(jnp.int32)
    jj = jj_mat.reshape(-1)
    return ii, jj

def make_s1_lie_loss(n_generators: int, rcond: float = 1e-6):
    """
    Constructs the S1 Lie algebra loss:
      - Closure: project [X_i, X_j] onto span{X_k} at each (t,x)
                 and penalize the projection error.
      - Constancy: penalize variation of the (pointwise) structure
                   coefficients c_ij^k over (t,x).

    Args:
      n_generators: number of learned generators m.
      rcond: small regularization parameter for Gram matrix inversion.

    Returns:
      loss_fn(params_gen, tx_batch) -> (scalar_loss, aux_dict)
    """
    idx_i, idx_j = _ordered_pair_indices(n_generators)
    K = int(idx_i.shape[0])
    reg = jnp.asarray(rcond, dtype=jnp.float64) ** 2

    # ---------- helper: scalar forwards for τ_i and ξ_i ----------------------

    def _tau_val_and_dt(params_tau_i, t_scalar):
        """τ_i(t) and ∂_t τ_i(t) for a single generator."""
        def tau_scalar(tt):
            t_arr = jnp.asarray([[tt]], dtype=jnp.float64)    # (1,1)
            t_norm = t_norm_gen(t_arr)                        # (1,1)
            out = tau_forward(params_tau_i, t_norm,
                              activation=gen_cfg.activation)
            return out[0, 0]  # scalar
        tau_val = tau_scalar(t_scalar)
        tau_t = jax.grad(tau_scalar)(t_scalar)
        return tau_val, tau_t

    def _xi_val_and_derivs(params_xi_i, t_scalar, x_scalar):
        """ξ_i(t,x), ξ_t, ξ_x for a single generator at (t,x)."""
        def xi_t_fun(tt):
            tx_arr = jnp.asarray([[tt, x_scalar]], dtype=jnp.float64)
            tx_norm = tx_norm_gen(tx_arr)
            out = xi_forward(params_xi_i, tx_norm,
                             activation=gen_cfg.activation)
            return out[0, 0]

        def xi_x_fun(xx):
            tx_arr = jnp.asarray([[t_scalar, xx]], dtype=jnp.float64)
            tx_norm = tx_norm_gen(tx_arr)
            out = xi_forward(params_xi_i, tx_norm,
                             activation=gen_cfg.activation)
            return out[0, 0]

        xi_val = xi_t_fun(t_scalar)
        xi_t = jax.grad(xi_t_fun)(t_scalar)
        xi_x = jax.grad(xi_x_fun)(x_scalar)
        return xi_val, xi_t, xi_x

    # ---------- helper: fields and derivatives at a single (t,x) -------------

    def _fields_and_derivs_at_point(params_gen, t_scalar, x_scalar):
        """
        Compute τ_i, ξ_i and their needed derivatives at (t,x) for all generators.
        Returns:
          tau   : (m,)
          xi    : (m,)
          tau_t : (m,)
          xi_t  : (m,)
          xi_x  : (m,)
        """
        tau_params = params_gen["tau"]
        xi_params  = params_gen["xi"]

        tau_list   = []
        xi_list    = []
        tau_t_list = []
        xi_t_list  = []
        xi_x_list  = []

        for p_tau, p_xi in zip(tau_params, xi_params):
            tau_i, tau_t_i = _tau_val_and_dt(p_tau, t_scalar)
            xi_i, xi_t_i, xi_x_i = _xi_val_and_derivs(p_xi, t_scalar, x_scalar)

            tau_list.append(tau_i)
            xi_list.append(xi_i)
            tau_t_list.append(tau_t_i)
            xi_t_list.append(xi_t_i)
            xi_x_list.append(xi_x_i)

        tau   = jnp.stack(tau_list, axis=0)    # (m,)
        xi    = jnp.stack(xi_list, axis=0)     # (m,)
        tau_t = jnp.stack(tau_t_list, axis=0)  # (m,)
        xi_t  = jnp.stack(xi_t_list, axis=0)   # (m,)
        xi_x  = jnp.stack(xi_x_list, axis=0)   # (m,)

        return tau, xi, tau_t, xi_t, xi_x

    # ---------- helper: bracket closure + coefficients at a point ------------

    def _point_err_and_C(tau, xi, tau_t, xi_t, xi_x):
        """
        Single-point closure + structure-coefficient computation.

        X_i = τ_i ∂_t + ξ_i ∂_x

        Lie bracket components:
          [X_i, X_j]^t = τ_i τ_{j,t} - τ_j τ_{i,t}
          [X_i, X_j]^x = τ_i ξ_{j,t} + ξ_i ξ_{j,x}
                         - τ_j ξ_{i,t} - ξ_j ξ_{i,x}
        """
        # V: (2,m) with rows [τ; ξ]
        V = jnp.stack([tau, xi], axis=0)  # (2, m)

        # Slice i,j components
        tau_i, tau_j = tau[idx_i], tau[idx_j]       # (K,)
        xi_i,  xi_j  = xi[idx_i],  xi[idx_j]        # (K,)
        tau_t_i, tau_t_j = tau_t[idx_i], tau_t[idx_j]
        xi_t_i,  xi_t_j  = xi_t[idx_i],  xi_t[idx_j]
        xi_x_i,  xi_x_j  = xi_x[idx_i],  xi_x[idx_j]

        # Bracket components for all ordered pairs (i,j)
        a = tau_i * tau_t_j - tau_j * tau_t_i
        b = (
            tau_i * xi_t_j + xi_i * xi_x_j
            - tau_j * xi_t_i - xi_j * xi_x_i
        )

        B = jnp.stack([a, b], axis=0)  # (2, K)

        # Project B onto span(V)
        G = V @ V.T                               # (2,2)
        G_reg = G + reg * jnp.eye(2, dtype=G.dtype)
        X = jnp.linalg.solve(G_reg, B)           # (2,K)
        C = V.T @ X                              # (m,K)
        P_B = V @ C                              # (2,K)
        E = B - P_B                              # (2,K)

        err = jnp.sum(jnp.abs(E))
        return err, C

    # ---------- main loss over batch -----------------------------------------

    def _loss_impl(params_gen, tx_batch: jnp.ndarray):
        """
        tx_batch: (B,2) with columns [t, x].
        """
        def eval_at_z(z):
            t_z, x_z = z[0], z[1]
            return _fields_and_derivs_at_point(params_gen, t_z, x_z)

        taus, xis, tau_ts, xi_ts, xi_xs = jax.vmap(eval_at_z)(tx_batch)
        # shapes: each (B, m)

        errs, Cs = jax.vmap(_point_err_and_C)(taus, xis, tau_ts, xi_ts, xi_xs)
        # errs: (B,)
        # Cs:   (B, m, K)

        error_sum = jnp.sum(errs)

        # Constancy across batch: variance of C over (t,x)
        C_var = jnp.var(Cs, axis=0)  # (m, K)
        var_sum = jnp.sum(C_var)

        total = error_sum + var_sum
        aux = {
            "error_sum": error_sum,
            "var_sum": var_sum,
        }
        return total, aux

    return jax.jit(_loss_impl)


# In[ ]:


# ============================ S2 — Jacobi identity (nested brackets, fixed) ============================
# Uses per-generator Xi_i(t,x) with its own (2x2) Jacobian and (2x2x2) Hessian,
# so that all matrix–vector products are dimensionally consistent.

import jax
import jax.numpy as jnp

def make_s2_jacobi_loss_nested(n_generators: int):
    # All distinct index triples i < j < k
    triples = [
        (i, j, k)
        for i in range(n_generators)
        for j in range(i + 1, n_generators)
        for k in range(j + 1, n_generators)
    ]
    if not triples:
        def _zero(params_gen, tx_batch):
            return jnp.array(0.0, dtype=jnp.float64), {
                "per_point": jnp.zeros((tx_batch.shape[0],), dtype=jnp.float64),
                "num_triples": 0,
            }
        return jax.jit(_zero)

    tri_i = jnp.array([t[0] for t in triples], dtype=jnp.int32)
    tri_j = jnp.array([t[1] for t in triples], dtype=jnp.int32)
    tri_k = jnp.array([t[2] for t in triples], dtype=jnp.int32)

    # All 6 permutations for symmetrized Jacobi expression
    perms6 = jnp.array(
        [[0, 1, 2],
         [0, 2, 1],
         [1, 0, 2],
         [1, 2, 0],
         [2, 0, 1],
         [2, 1, 0]],
        dtype=jnp.int32,
    )

    # -------- helper: F, J, H at a single (t,x) via per-generator Xi_i -------

    def _fields_jac_hess(params_gen, z):
        """
        Compute:
          F: (m,2)      vector field values at z = (t,x)
          J: (m,2,2)    Jacobians D X_i(z)
          H: (m,2,2,2)  Hessians D^2 X_i(z)
        for all generators X_i.
        """
        t_z, x_z = z[0], z[1]

        F_list = []
        J_list = []
        H_list = []

        # Iterate over generators; loop size m is static so jit-friendly.
        for params_tau_i, params_xi_i in zip(params_gen["tau"], params_gen["xi"]):

            def Xi(zz):
                tt, xx = zz[0], zz[1]

                t_arr = jnp.asarray([[tt]], dtype=jnp.float64)          # (1,1)
                tx_arr = jnp.asarray([[tt, xx]], dtype=jnp.float64)     # (1,2)

                t_norm = t_norm_gen(t_arr)      # (1,1)
                tx_norm = tx_norm_gen(tx_arr)   # (1,2)

                tau_val = tau_forward(
                    params_tau_i,
                    t_norm,
                    activation=gen_cfg.activation,
                )[0, 0]  # scalar

                xi_val = xi_forward(
                    params_xi_i,
                    tx_norm,
                    activation=gen_cfg.activation,
                )[0, 0]   # scalar

                return jnp.array([tau_val, xi_val], dtype=jnp.float64)  # (2,)

            # Value, Jacobian, Hessian for generator i
            Fi = Xi(z)                                                # (2,)
            Ji = jax.jacobian(Xi)(z)                                  # (2,2)
            Hi = jax.jacobian(lambda zz: jax.jacobian(Xi)(zz))(z)     # (2,2,2)

            F_list.append(Fi)
            J_list.append(Ji)
            H_list.append(Hi)

        F = jnp.stack(F_list, axis=0)   # (m,2)
        J = jnp.stack(J_list, axis=0)   # (m,2,2)
        H = jnp.stack(H_list, axis=0)   # (m,2,2,2)

        return F, J, H

    # ------------- helper: basic bracket and nested bracket algebra ----------

    def _bracket_val(F, J, p, q):
        """
        [X_p, X_q](z) = J_q(z) @ F_p(z) - J_p(z) @ F_q(z)
        where F_i ∈ R^2, J_i ∈ R^{2x2}.
        Returns a 2-vector (in (∂_t, ∂_x) basis).
        """
        Jp, Jq = J[p], J[q]   # (2,2)
        fp, fq = F[p], F[q]   # (2,)
        return (Jq @ fp) - (Jp @ fq)   # (2,)

    def _dir_along(F, J, H, r, p, q):
        """
        Directional action of [X_p, X_q] on X_r, using first and second derivatives.
        Matches original nested-bracket algebra, specialized to (t,x).
        """
        Jr, Jp, Jq = J[r], J[p], J[q]     # (2,2)
        Hp, Hq = H[p], H[q]               # (2,2,2)
        fr, fp, fq = F[r], F[p], F[q]     # (2,)

        # (2,) results
        t1 = Jq @ (Jp @ fr)
        t2 = ((Hq * fr[None, None, :]).sum(axis=2)) @ fp
        t3 = Jp @ (Jq @ fr)
        t4 = ((Hp * fr[None, None, :]).sum(axis=2)) @ fq

        return t1 + t2 - t3 - t4

    def _double_bracket(F, J, H, r, p, q):
        """
        Nested bracket [[X_r, X_p], X_q] at a point.
        """
        inner = _bracket_val(F, J, p, q)
        return _dir_along(F, J, H, r, p, q) - (J[r] @ inner)

    def _jacobi_one_order(F, J, H, u, v, w):
        """
        Jacobi combination for one ordering (u, v, w):
          [[X_u, X_v], X_w] + [[X_w, X_u], X_v] + [[X_v, X_w], X_u]
        """
        return (
            _double_bracket(F, J, H, u, v, w)
            + _double_bracket(F, J, H, w, u, v)
            + _double_bracket(F, J, H, v, w, u)
        )

    def _triple_sum_over_6(F, J, H, i, j, k):
        """
        Symmetrize over all 6 permutations of (i,j,k), summing absolute values.
        """
        inds = jnp.array([i, j, k], dtype=jnp.int32)

        def _one_perm(p):
            u, v, w = inds[p[0]], inds[p[1]], inds[p[2]]
            r = _jacobi_one_order(F, J, H, u, v, w)  # (2,)
            return jnp.sum(jnp.abs(r))

        vals = jax.vmap(_one_perm)(perms6)  # (6,)
        return jnp.sum(vals)

    def _point_loss(params_gen, z):
        """
        Jacobi loss at a single point z = (t,x), summed over all triples (i,j,k).
        """
        F, J, H = _fields_jac_hess(params_gen, z)
        per_tr = jax.vmap(
            lambda a, b, c: _triple_sum_over_6(F, J, H, a, b, c)
        )(tri_i, tri_j, tri_k)  # (num_triples,)
        return jnp.sum(per_tr)

    _point_loss_jit = jax.jit(_point_loss)

    def _loss_impl(params_gen, tx_batch: jnp.ndarray):
        """
        tx_batch: (B,2) array with columns [t, x].
        Returns:
          total_loss, {
              "per_point": (B,),
              "num_triples": int,
          }
        """
        per_point = jax.vmap(lambda z: _point_loss_jit(params_gen, z))(tx_batch)
        total = jnp.sum(per_point)
        aux = {
            "per_point": per_point,
            "num_triples": int(tri_i.shape[0]),
        }
        return total, aux

    return jax.jit(_loss_impl)


# In[ ]:


# ============================ S3 — Skew-symmetry (fixed) ============================
# Uses per-generator Xi_i(t,x) so that F and J have shapes:
#   F: (m,2),  J: (m,2,2)
# and [X_i, X_j](z) = J_j @ F_i - J_i @ F_j is always well-typed.

import jax
import jax.numpy as jnp

def make_s3_skewsym_loss(n_generators: int):
    # All distinct pairs i < j
    pairs = [
        (i, j)
        for i in range(n_generators)
        for j in range(i + 1, n_generators)
    ]
    if not pairs:
        def _zero(params_gen, tx_batch):
            return jnp.array(0.0, dtype=jnp.float64), {
                "per_point": jnp.zeros((tx_batch.shape[0],), dtype=jnp.float64),
                "num_pairs": 0,
            }
        return jax.jit(_zero)

    pi = jnp.array([p[0] for p in pairs], dtype=jnp.int32)
    pj = jnp.array([p[1] for p in pairs], dtype=jnp.int32)

    # ---------- helper: F and J at a single (t,x) ----------------------------

    def _fields_and_jac(params_gen, z):
        """
        Compute:
          F: (m,2)  vector field values at z = (t,x)
          J: (m,2,2) Jacobians w.r.t (t,x)
        for all generators X_i.
        """
        t_z, x_z = z[0], z[1]

        F_list = []
        J_list = []

        for params_tau_i, params_xi_i in zip(params_gen["tau"], params_gen["xi"]):

            def Xi(zz):
                tt, xx = zz[0], zz[1]

                t_arr = jnp.asarray([[tt]], dtype=jnp.float64)          # (1,1)
                tx_arr = jnp.asarray([[tt, xx]], dtype=jnp.float64)     # (1,2)

                t_norm = t_norm_gen(t_arr)      # (1,1)
                tx_norm = tx_norm_gen(tx_arr)   # (1,2)

                tau_val = tau_forward(
                    params_tau_i,
                    t_norm,
                    activation=gen_cfg.activation,
                )[0, 0]
                xi_val = xi_forward(
                    params_xi_i,
                    tx_norm,
                    activation=gen_cfg.activation,
                )[0, 0]

                return jnp.array([tau_val, xi_val], dtype=jnp.float64)  # (2,)

            Fi = Xi(z)                         # (2,)
            Ji = jax.jacobian(Xi)(z)          # (2,2)

            F_list.append(Fi)
            J_list.append(Ji)

        F = jnp.stack(F_list, axis=0)  # (m,2)
        J = jnp.stack(J_list, axis=0)  # (m,2,2)

        return F, J

    # ---------- bracket at a point ------------------------------------------

    def _bracket_val(F, J, p, q):
        """
        [X_p, X_q](z) = J_q(z) @ F_p(z) - J_p(z) @ F_q(z)
        """
        Jp, Jq = J[p], J[q]   # (2,2)
        fp, fq = F[p], F[q]   # (2,)
        return (Jq @ fp) - (Jp @ fq)

    def _point_loss(params_gen, z):
        """
        Skew-symmetry loss at a single point z = (t,x):
          sum_{i<j} || [X_i, X_j] + [X_j, X_i] ||_1
        """
        F, J = _fields_and_jac(params_gen, z)

        def one(i, j):
            r = _bracket_val(F, J, i, j) + _bracket_val(F, J, j, i)
            return jnp.sum(jnp.abs(r))

        vals = jax.vmap(one)(pi, pj)  # (num_pairs,)
        return jnp.sum(vals)

    _pl = jax.jit(_point_loss)

    def _loss_impl(params_gen, tx_batch: jnp.ndarray):
        """
        tx_batch: (B,2) with columns [t, x].
        Returns:
          total_loss, {
              "per_point": (B,),
              "num_pairs": int,
          }
        """
        per_point = jax.vmap(lambda z: _pl(params_gen, z))(tx_batch)
        total = jnp.sum(per_point)
        aux = {
            "per_point": per_point,
            "num_pairs": int(pi.shape[0]),
        }
        return total, aux

    return jax.jit(_loss_impl)


# In[ ]:


# ============================ S4 — Bilinearity (fixed) ============================
# Uses per-generator Xi_i(t,x) so F and J are well-typed:
#   F: (m,2), J: (m,2,2)
# Checks:
#   [c u + c' v, w] = c [u, w] + c' [v, w]
#   [u, c v + c' w] = c [u, v] + c' [u, w]

import jax
import jax.numpy as jnp

def make_s4_bilinearity_loss(
    n_generators: int,
    num_cc: int = 4,
    cc_list=None,
    normalize: bool = True,
):
    # All distinct triples i < j < k
    triples = [
        (i, j, k)
        for i in range(n_generators)
        for j in range(i + 1, n_generators)
        for k in range(j + 1, n_generators)
    ]
    if not triples:
        def _zero(params_gen, tx_batch, key=None):
            return jnp.array(0.0, dtype=jnp.float64), {
                "per_point": jnp.zeros((tx_batch.shape[0],), dtype=jnp.float64),
            }
        return jax.jit(_zero)

    tri_i = jnp.array([t[0] for t in triples], dtype=jnp.int32)
    tri_j = jnp.array([t[1] for t in triples], dtype=jnp.int32)
    tri_k = jnp.array([t[2] for t in triples], dtype=jnp.int32)

    perms6 = jnp.array(
        [[0, 1, 2],
         [0, 2, 1],
         [1, 0, 2],
         [1, 2, 0],
         [2, 0, 1],
         [2, 1, 0]],
        dtype=jnp.int32,
    )

    # ----------- helper: F and J at a single (t,x) ---------------------------

    def _fields_and_jac(params_gen, z):
        """
        Compute:
          F: (m,2)  vector field values at z = (t,x)
          J: (m,2,2) Jacobians w.r.t (t,x)
        for all generators X_i.
        """
        t_z, x_z = z[0], z[1]

        F_list = []
        J_list = []

        for params_tau_i, params_xi_i in zip(params_gen["tau"], params_gen["xi"]):

            def Xi(zz):
                tt, xx = zz[0], zz[1]

                t_arr = jnp.asarray([[tt]], dtype=jnp.float64)          # (1,1)
                tx_arr = jnp.asarray([[tt, xx]], dtype=jnp.float64)     # (1,2)

                t_norm = t_norm_gen(t_arr)      # (1,1)
                tx_norm = tx_norm_gen(tx_arr)   # (1,2)

                tau_val = tau_forward(
                    params_tau_i,
                    t_norm,
                    activation=gen_cfg.activation,
                )[0, 0]
                xi_val = xi_forward(
                    params_xi_i,
                    tx_norm,
                    activation=gen_cfg.activation,
                )[0, 0]

                return jnp.array([tau_val, xi_val], dtype=jnp.float64)  # (2,)

            Fi = Xi(z)                         # (2,)
            Ji = jax.jacobian(Xi)(z)          # (2,2)

            F_list.append(Fi)
            J_list.append(Ji)

        F = jnp.stack(F_list, axis=0)  # (m,2)
        J = jnp.stack(J_list, axis=0)  # (m,2,2)

        return F, J

    # ----------- bracket and bilinearity terms at a point --------------------

    def _bracket(F, J, p, q):
        """
        [X_p, X_q](z) = J_q(z) @ F_p(z) - J_p(z) @ F_q(z)
        """
        Jp, Jq = J[p], J[q]   # (2,2)
        fp, fq = F[p], F[q]   # (2,)
        return (Jq @ fp) - (Jp @ fq)

    def _triple_terms(F, J, i, j, k, cc):
        """
        Bilinearity residuals for a single triple (i,j,k) at a fixed point,
        averaged over coefficient pairs in cc.
        """
        inds = jnp.array([i, j, k], dtype=jnp.int32)

        def one_perm(p):
            u, v, w = inds[p[0]], inds[p[1]], inds[p[2]]
            fu, fv, fw = F[u], F[v], F[w]        # (2,)
            Ju, Jv, Jw = J[u], J[v], J[w]        # (2,2)

            def one_cc(cpair):
                c, cp = cpair[0], cpair[1]

                # Linear combinations in the first slot
                f_uv = c * fu + cp * fv
                J_uv = c * Ju + cp * Jv

                # Linear combinations in the second slot
                f_vw = c * fv + cp * fw
                J_vw = c * Jv + cp * Jw

                # [c u + c' v, w]
                term1 = (Jw @ f_uv) - (J_uv @ fw)
                rhs1  = c * _bracket(F, J, u, w) + cp * _bracket(F, J, v, w)
                r1 = term1 - rhs1

                # [u, c v + c' w]
                term2 = (J_vw @ fu) - (Ju @ f_vw)
                rhs2  = c * _bracket(F, J, u, v) + cp * _bracket(F, J, u, w)
                r2 = term2 - rhs2

                if normalize:
                    denom = jnp.abs(c) + jnp.abs(cp) + 1e-12
                    r1 = r1 / denom
                    r2 = r2 / denom

                return jnp.sum(jnp.abs(r1)) + jnp.sum(jnp.abs(r2))

            vals_cc = jax.vmap(one_cc)(cc)  # (num_cc,)
            return jnp.mean(vals_cc)

        vals = jax.vmap(one_perm)(perms6)  # (6,)
        return jnp.sum(vals)

    def _point_loss(params_gen, z, cc):
        """
        Bilinearity loss at a single point z = (t,x), summed over all triples.
        """
        F, J = _fields_and_jac(params_gen, z)
        per_tr = jax.vmap(
            lambda a, b, c_: _triple_terms(F, J, a, b, c_, cc)
        )(tri_i, tri_j, tri_k)  # (num_triples,)
        return jnp.sum(per_tr)

    _pl = jax.jit(_point_loss)

    def _loss_impl(params_gen, tx_batch: jnp.ndarray, key=None):
        """
        tx_batch: (B,2) with columns [t, x].
        key: optional PRNGKey for sampling coefficient pairs if cc_list is None.

        Returns:
          total_loss, {
              "per_point": (B,),
              "num_triples": int,
              "num_cc": int,
          }
        """
        if cc_list is not None:
            cc = jnp.asarray(cc_list, dtype=jnp.float64)  # (num_cc, 2)
        else:
            key = jax.random.PRNGKey(0) if key is None else key
            cc = jax.random.uniform(
                key,
                (num_cc, 2),
                minval=-1.0,
                maxval=1.0,
                dtype=jnp.float64,
            )

        per_point = jax.vmap(lambda z: _pl(params_gen, z, cc))(tx_batch)
        total = jnp.sum(per_point)
        aux = {
            "per_point": per_point,
            "num_triples": int(tri_i.shape[0]),
            "num_cc": int(cc.shape[0]),
        }
        return total, aux

    return jax.jit(_loss_impl)


# In[ ]:


# ============================ PATCH: S5 loss to support eval_generators_jit returning (tau, xi, beta) ============================
import jax
import jax.numpy as jnp

def make_s5_column_independence_loss(
    n_generators: int,
    *,
    mode: str = "sigma",
    tau: float = 0.0,
    eps: float = 1e-12,
):
    # Normalize mode to a small integer code for jit-friendliness
    mode = "sigma" if mode == "sigma" else "corr_l2"
    mode_code = 0 if mode == "sigma" else 1

    def _A_from_batch(params_gen, tx_batch: jnp.ndarray):
        """
        Build A ∈ R^{2N x m} from a batch of points:
          - rows: all (τ_i(t_n), ξ_i(t_n, x_n)) stacked over n
          - columns: generators i = 0..m-1

        tx_batch: (N,2) with columns [t, x].
        """
        t_batch = tx_batch[:, 0]  # (N,)
        x_batch = tx_batch[:, 1]  # (N,)

        # eval_generators_jit now returns (tau, xi, beta) or (tau, xi, beta, phi) if return_phi=True
        outs = eval_generators_jit(params_gen, t_batch, x_batch)
        tau_vals = outs[0]  # (m, N)
        xi_vals  = outs[1]  # (m, N)

        comp = jnp.stack([tau_vals, xi_vals], axis=2)   # (m, N, 2)
        comp_B2m = jnp.transpose(comp, (1, 2, 0))       # (N, 2, m)
        A = comp_B2m.reshape(-1, n_generators)          # (2N, m)
        return A

    def _loss_impl(params_gen, tx_batch: jnp.ndarray):
        A = _A_from_batch(params_gen, tx_batch)  # (2N, m)

        # Normalize columns
        col_norms = jnp.linalg.norm(A, axis=0) + eps   # (m,)
        Ahat = A / col_norms                           # (2N, m)

        # Gram matrix of normalized columns
        G = Ahat.T @ Ahat                              # (m, m)

        if mode_code == 0:
            lam = jnp.linalg.eigvalsh(G)
            lam_min = jnp.clip(jnp.min(lam), 0.0, None)
            sigma_min = jnp.sqrt(lam_min)
            loss = jnp.maximum(
                0.0,
                jnp.asarray(tau, dtype=G.dtype) - sigma_min,
            )
            aux = {"sigma_min": sigma_min}
        else:
            I = jnp.eye(G.shape[0], dtype=G.dtype)
            off = G - I
            off = off - jnp.diag(jnp.diag(off))
            loss = jnp.sum(off * off)
            aux = {"gram_diag_mean": jnp.mean(jnp.diag(G))}

        return loss, aux

    return jax.jit(_loss_impl)

# Export patched symbol (overwrites old one)
#globals().update({"make_s5_column_independence_loss": make_s5_column_independence_loss})
#print("Patched: make_s5_column_independence_loss now supports eval_generators_jit returning (tau, xi, beta).")


# In[ ]:


# ============================ S6 — SDE determining-equation loss (Gaeta–Quintero) ============================
# Implements the 1D projectable SDE symmetry determining equations:
#
#   r1 = ξ_t + f ξ_x - ξ f_x - f_t τ - f τ_t + 0.5 σ^2 ξ_xx = 0
#   r2 = σ ξ_x - ξ σ_x - τ σ_t - 0.5 σ τ_t = 0
#
# for each generator X_i = τ_i(t) ∂_t + ξ_i(t,x) ∂_x, using drift f = mu_fn(t,x),
# diffusion σ = sig_fn(t,x), and autodiff for all derivatives.

import jax
import jax.numpy as jnp

def make_s6_commutator_loss_ito(*, mu_fn, sig_fn, use_abs: bool = False):
    """
    Construct L6 loss that enforces Gaeta–Quintero SDE determining equations.

    Args:
      mu_fn(t, x):   drift function f(t,x); should be JAX-differentiable.
      sig_fn(t, x):  diffusion function σ(t,x); JAX-differentiable.
      use_abs:       if False, use squared residuals (L2); if True, use |residual| (L1).

    Returns:
      loss_fn(params_gen, tx_batch) -> (loss_scalar, aux_dict)
        - params_gen: {"tau": [...], "xi": [...]}
        - tx_batch:   (B,2) array with columns [t, x]
        - aux_dict:   {"per_point": (B,)} total residual per (t,x).
    """

    # ---- local scalar wrappers for τ(t) and ξ(t,x) with gradients ------------

    def tau_val_and_dt(params_tau, t_scalar):
        """Return τ(t), τ_t(t)."""
        def tau_scalar(tt):
            t_arr = jnp.asarray([[tt]], dtype=jnp.float64)     # (1,1)
            t_norm = t_norm_gen(t_arr)
            out = tau_forward(params_tau, t_norm,
                              activation=gen_cfg.activation)
            return out[0, 0]  # scalar
        tau_val = tau_scalar(t_scalar)
        tau_t = jax.grad(tau_scalar)(t_scalar)
        return tau_val, tau_t

    def xi_val_and_derivs(params_xi, t_scalar, x_scalar):
        """Return ξ(t,x), ξ_t, ξ_x, ξ_xx at a single (t,x)."""
        # ξ as function of t (x fixed)
        def xi_t_fun(tt):
            tx_arr = jnp.asarray([[tt, x_scalar]], dtype=jnp.float64)  # (1,2)
            tx_norm = tx_norm_gen(tx_arr)
            out = xi_forward(params_xi, tx_norm,
                             activation=gen_cfg.activation)
            return out[0, 0]

        # ξ as function of x (t fixed)
        def xi_x_fun(xx):
            tx_arr = jnp.asarray([[t_scalar, xx]], dtype=jnp.float64)  # (1,2)
            tx_norm = tx_norm_gen(tx_arr)
            out = xi_forward(params_xi, tx_norm,
                             activation=gen_cfg.activation)
            return out[0, 0]

        xi_val = xi_t_fun(t_scalar)
        xi_t = jax.grad(xi_t_fun)(t_scalar)
        xi_x = jax.grad(xi_x_fun)(x_scalar)
        xi_xx = jax.grad(lambda xx: jax.grad(xi_x_fun)(xx))(x_scalar)

        return xi_val, xi_t, xi_x, xi_xx

    # ---- drift/diffusion + their derivatives at a point ----------------------

    def f_sigma_and_derivs(t_scalar, x_scalar):
        """Compute f, f_t, f_x, σ, σ_t, σ_x at (t,x)."""
        # Drift
        def f_t_fun(tt):
            return mu_fn(tt, x_scalar)

        def f_x_fun(xx):
            return mu_fn(t_scalar, xx)

        f_val = mu_fn(t_scalar, x_scalar)
        f_t = jax.grad(f_t_fun)(t_scalar)
        f_x = jax.grad(f_x_fun)(x_scalar)

        # Diffusion
        def s_t_fun(tt):
            return sig_fn(tt, x_scalar)

        def s_x_fun(xx):
            return sig_fn(t_scalar, xx)

        sigma_val = sig_fn(t_scalar, x_scalar)
        sigma_t = jax.grad(s_t_fun)(t_scalar)
        sigma_x = jax.grad(s_x_fun)(x_scalar)

        return f_val, f_t, f_x, sigma_val, sigma_t, sigma_x

    # ---- per-point residual over all generators ------------------------------

    def _point_residual(params_gen, z):
        """
        Compute total determining-equation residual at a single point (t,x),
        summed over all generators i.
        """
        t, x = z[0], z[1]

        # SDE coefficients and their derivatives
        f_val, f_t, f_x, sigma_val, sigma_t, sigma_x = f_sigma_and_derivs(t, x)
        sigma2 = sigma_val * sigma_val

        total_res = 0.0

        for params_tau_i, params_xi_i in zip(params_gen["tau"], params_gen["xi"]):
            # Generator derivatives
            tau_i, tau_t_i = tau_val_and_dt(params_tau_i, t)
            xi_i, xi_t_i, xi_x_i, xi_xx_i = xi_val_and_derivs(params_xi_i, t, x)

            # r1 (drift determining eq)
            r1 = (
                xi_t_i
                + f_val * xi_x_i
                - xi_i * f_x
                - f_t * tau_i
                - f_val * tau_t_i
                + 0.5 * sigma2 * xi_xx_i
            )

            # r2 (diffusion determining eq)
            r2 = (
                sigma_val * xi_x_i
                - xi_i * sigma_x
                - tau_i * sigma_t
                - 0.5 * sigma_val * tau_t_i
            )

            if use_abs:
                total_res = total_res + jnp.abs(r1) + jnp.abs(r2)
            else:
                total_res = total_res + r1 * r1 + r2 * r2

        return total_res

    _point_residual_jit = jax.jit(_point_residual)

    # ---- batched loss over tx_batch -----------------------------------------

    def _loss_impl(params_gen, tx_batch: jnp.ndarray):
        """
        tx_batch: (B,2) array with columns [t, x].
        """
        per_point = jax.vmap(lambda z: _point_residual_jit(params_gen, z))(tx_batch)
        loss = jnp.mean(per_point)  # or jnp.sum(per_point); here we take mean
        aux = {"per_point": per_point}
        return loss, aux

    return jax.jit(_loss_impl)


# In[ ]:


# ============================ S7 — Prolonged pushforward residual (SDE generators only: tau/xi) ============================
import jax
import jax.numpy as jnp

def make_s7_pushforward_coeff_loss_sde_only(
    *,
    mu_fn,
    sig_fn,
    eps: float = 1e-2,
    num_steps: int = 1,
    sigma_floor: float = 1e-8,
    dt_floor: float = 1e-10,
    dt_neg_penalty: float = 100.0,
    activation: str = "tanh",
    normalize_tx=None,
    jit: bool = True,
    # NEW safety
    tau_clip: float = 5.0,
    xi_clip: float = 5.0,
    t_clip_lo: float = -1e6,
    t_clip_hi: float =  1e6,
    x_clip_abs: float = 50.0,
):

    """
    S7 (trajectory-level) validity for SDE symmetries (tau, xi only), mirroring paper-style "after-flow residual".

    For each learned generator X_i:
      1) Integrate the prolonged epsilon-flow on (t,x,mu,sigma):
            dt/dε = τ(t)
            dx/dε = ξ(t,x)
            dσ/dε = σ(∂xξ - 1/2 ∂tτ)
            dμ/dε = ∂tξ + μ∂xξ + 1/2 σ^2 ∂xxξ - μ∂tτ
      2) Compare predicted pushed coefficients (μ_pred, σ_pred) to the surrogate evaluated at pushed points:
            μ_eval = μ(t_push, x_push),  σ_eval = σ(t_push, x_push)
      3) Penalize MSE(μ_pred-μ_eval) + MSE(σ_pred-σ_eval) + dt_neg_penalty * mean(max(0, -Δt_push)).

    Requires a callable `eval_generators_tau_xi_jit(params_gen, t, x, activation=..., normalize_tx=...)`
    that returns (tau_all, xi_all) with shapes (m,B).
    """

    if "eval_generators_tau_xi_jit" not in globals() or (not callable(globals()["eval_generators_tau_xi_jit"])):
        raise NameError(
            "S7(SDE-only) requires a callable eval_generators_tau_xi_jit(params_gen, t, x, ...)"
        )

    eval_gen_tau_xi = globals()["eval_generators_tau_xi_jit"]

    eps = jnp.asarray(eps, dtype=jnp.float64)
    num_steps = int(num_steps)

    # finite-difference steps for derivatives (kept internal to avoid changing signature)
    fd_t = jnp.asarray(1e-3, dtype=jnp.float64)
    fd_x = jnp.asarray(1e-3, dtype=jnp.float64)

    def _rhs_diag_with_derivs(params_gen, t_stack, x_stack):
        """
        Returns diag-evaluations for each generator i at its own points plus FD derivatives:
          tau, xi, tau_t, xi_t, xi_x, xi_xx   all shape (m,B)
        """
        m, B = t_stack.shape
        t_flat = t_stack.reshape(-1)  # (mB,)
        x_flat = x_stack.reshape(-1)  # (mB,)

        def diag_from_flat(t_in, x_in):
            tau_all, xi_all = eval_gen_tau_xi(
                params_gen, t_in, x_in, activation=activation, normalize_tx=normalize_tx
            )  # (m, mB)

            tau_blk = tau_all.reshape(m, m, B)
            xi_blk  = xi_all.reshape(m, m, B)

            idx = jnp.arange(m, dtype=jnp.int32)
            tau_diag = tau_blk[idx, idx, :]  # (m,B)
            xi_diag  = xi_blk[idx, idx, :]   # (m,B)

            tau_diag = jnp.nan_to_num(tau_diag, nan=0.0, posinf=0.0, neginf=0.0)
            xi_diag  = jnp.nan_to_num(xi_diag,  nan=0.0, posinf=0.0, neginf=0.0)

            # differentiable soft-clip
            tau_diag = tau_clip * jnp.tanh(tau_diag / tau_clip)
            xi_diag  = xi_clip  * jnp.tanh(xi_diag  / xi_clip)
            return tau_diag, xi_diag

        # base
        tau0, xi0 = diag_from_flat(t_flat, x_flat)

        # time perturbations
        tau_p, xi_p = diag_from_flat(t_flat + fd_t, x_flat)
        tau_m, xi_m = diag_from_flat(t_flat - fd_t, x_flat)

        tau_t = (tau_p - tau_m) / (2.0 * fd_t)
        xi_t  = (xi_p  - xi_m)  / (2.0 * fd_t)

        # space perturbations
        _, xi_xp = diag_from_flat(t_flat, x_flat + fd_x)
        _, xi_xm = diag_from_flat(t_flat, x_flat - fd_x)

        xi_x  = (xi_xp - xi_xm) / (2.0 * fd_x)
        xi_xx = (xi_xp - 2.0 * xi0 + xi_xm) / (fd_x * fd_x)

        # clean any non-finites
        tau_t = jnp.nan_to_num(tau_t, nan=0.0, posinf=0.0, neginf=0.0)
        xi_t  = jnp.nan_to_num(xi_t,  nan=0.0, posinf=0.0, neginf=0.0)
        xi_x  = jnp.nan_to_num(xi_x,  nan=0.0, posinf=0.0, neginf=0.0)
        xi_xx = jnp.nan_to_num(xi_xx, nan=0.0, posinf=0.0, neginf=0.0)

        return tau0, xi0, tau_t, xi_t, xi_x, xi_xx

    def _flow_heun_allgens(params_gen, t0, x0):
        """
        Push point cloud under each generator, AND integrate prolonged (mu,sig) along epsilon.

        Inputs:
          t0, x0: (B,)
        Returns:
          t_push, x_push, mu_pred, sg_pred: each (m,B)
        """
        tau0, _ = eval_gen_tau_xi(params_gen, t0, x0, activation=activation, normalize_tx=normalize_tx)
        m = tau0.shape[0]
        B = t0.shape[0]

        t_stack = jnp.broadcast_to(t0[None, :], (m, B))
        x_stack = jnp.broadcast_to(x0[None, :], (m, B))

        # initial coefficients on the base cloud
        t0c = jnp.clip(jnp.nan_to_num(t0, nan=0.0, posinf=0.0, neginf=0.0), t_clip_lo, t_clip_hi)
        x0c = jnp.clip(jnp.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0), -x_clip_abs, x_clip_abs)

        mu0 = jnp.nan_to_num(mu_fn(t0c, x0c), nan=0.0, posinf=0.0, neginf=0.0)
        sg0 = jnp.nan_to_num(sig_fn(t0c, x0c), nan=0.0, posinf=0.0, neginf=0.0)
        sg0 = jnp.maximum(jnp.abs(sg0), sigma_floor)

        mu_stack = jnp.broadcast_to(mu0[None, :], (m, B))
        sg_stack = jnp.broadcast_to(sg0[None, :], (m, B))

        def pos_floor(u):
            return sigma_floor + jax.nn.softplus(u - sigma_floor)

        def one_step(_, state):
            tS, xS, muS, sgS = state

            # k1
            tau, xi, tau_t, xi_t, xi_x, xi_xx = _rhs_diag_with_derivs(params_gen, tS, xS)
            k1_t = tau
            k1_x = xi
            k1_mu = xi_t + muS * xi_x + 0.5 * (sgS * sgS) * xi_xx - muS * tau_t
            k1_sg = sgS * xi_x - 0.5 * sgS * tau_t

            t_pred  = tS  + eps * k1_t
            x_pred  = xS  + eps * k1_x
            mu_pred = muS + eps * k1_mu
            sg_pred = pos_floor(sgS + eps * k1_sg)

            # k2
            tau2, xi2, tau_t2, xi_t2, xi_x2, xi_xx2 = _rhs_diag_with_derivs(params_gen, t_pred, x_pred)
            k2_t = tau2
            k2_x = xi2
            k2_mu = xi_t2 + mu_pred * xi_x2 + 0.5 * (sg_pred * sg_pred) * xi_xx2 - mu_pred * tau_t2
            k2_sg = sg_pred * xi_x2 - 0.5 * sg_pred * tau_t2

            t_new  = tS  + 0.5 * eps * (k1_t  + k2_t)
            x_new  = xS  + 0.5 * eps * (k1_x  + k2_x)
            mu_new = muS + 0.5 * eps * (k1_mu + k2_mu)
            sg_new = pos_floor(sgS + 0.5 * eps * (k1_sg + k2_sg))

            return (t_new, x_new, mu_new, sg_new)

        t_stack, x_stack, mu_stack, sg_stack = jax.lax.fori_loop(
            0, num_steps, one_step, (t_stack, x_stack, mu_stack, sg_stack)
        )
        return t_stack, x_stack, mu_stack, sg_stack

    def _loss_impl(params_gen, t_grid, x_paths):
        """
        t_grid:  (N+1,)
        x_paths: (n_traj, N+1)
        """
        t_grid  = jnp.asarray(t_grid,  dtype=jnp.float64)
        x_paths = jnp.asarray(x_paths, dtype=jnp.float64)

        n_traj, Np1 = x_paths.shape
        N = Np1 - 1

        # Base grid for dt-neg penalty computed on pushed trajectory times
        t_mat  = jnp.broadcast_to(t_grid[None, :], (n_traj, Np1))

        # ---- Use LEFT endpoints only for coefficient residual (N points per traj) ----
        t_left = t_mat[:, :-1].reshape(-1)     # (B_left,)
        x_left = x_paths[:, :-1].reshape(-1)   # (B_left,)
        B_left = t_left.shape[0]

        # Prolonged push from (t_left,x_left,mu(t_left,x_left),sig(t_left,x_left))
        t_push_L, x_push_L, mu_pred, sg_pred = _flow_heun_allgens(params_gen, t_left, x_left)  # each (m,B_left)

        # Evaluate surrogate coefficients at pushed points
        tpc = jnp.clip(jnp.nan_to_num(t_push_L, nan=0.0, posinf=0.0, neginf=0.0), t_clip_lo, t_clip_hi)
        xpc = jnp.clip(jnp.nan_to_num(x_push_L, nan=0.0, posinf=0.0, neginf=0.0), -x_clip_abs, x_clip_abs)

        mu_eval = jnp.nan_to_num(mu_fn(tpc.reshape(-1), xpc.reshape(-1)), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        sg_eval = jnp.nan_to_num(sig_fn(tpc.reshape(-1), xpc.reshape(-1)), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)

        m = t_push_L.shape[0]
        mu_eval = mu_eval.reshape(m, B_left)
        sg_eval = jnp.maximum(jnp.abs(sg_eval.reshape(m, B_left)), sigma_floor)

        # Per-generator coefficient residuals
        mu_mse = jnp.mean((mu_pred - mu_eval) ** 2, axis=1)  # (m,)
        sg_mse = jnp.mean((sg_pred - sg_eval) ** 2, axis=1)  # (m,)

        # ---- dt-negative penalty computed on pushed full trajectories (optional but kept) ----
        # build full point cloud (N+1) just for dt-neg; reuse same prolonged flow (cheap enough)
        t_full = t_mat.reshape(-1)        # (n_traj*(N+1),)
        x_full = x_paths.reshape(-1)      # (n_traj*(N+1),)
        t_push_full, _, _, _ = _flow_heun_allgens(params_gen, t_full, x_full)  # (m, B_full)

        # reshape to (m, n_traj, N+1)
        B_full = t_full.shape[0]
        t_push_full = t_push_full.reshape(m, n_traj, Np1)
        dtp = t_push_full[:, :, 1:] - t_push_full[:, :, :-1]  # (m,n_traj,N)

        dt_neg = jax.nn.softplus(-dtp)                         # ~ max(0, -dt)
        dt_neg_mean = jnp.mean(dt_neg, axis=(1, 2))            # (m,)

        per_gen_loss = mu_mse + sg_mse + dt_neg_penalty * dt_neg_mean
        loss = jnp.mean(per_gen_loss)

        aux = {
            "per_gen_loss": per_gen_loss,
            # keep the same key name; now (m,3): [mu_mse, sg_mse, dt_neg_mean]
            "per_gen_stats": jnp.stack([mu_mse, sg_mse, dt_neg_mean], axis=1),
            "eps": eps,
            "num_steps": jnp.asarray(num_steps, dtype=jnp.int32),
        }
        return loss, aux

    return jax.jit(_loss_impl) if jit else _loss_impl


# In[ ]:


# ============================ Master loss: weighted sum of L1–L7 ============================
# Assumes in scope:
#   - params_sde, in_norm_sde, surrogate_f_sigma
#   - gen_cfg, params_gen
#   - TX_gen  (for sampling tx_batch in the training loop)
#   - make_s1_lie_loss, make_s2_jacobi_loss_nested, make_s3_skewsym_loss,
#     make_s4_bilinearity_loss, make_s5_column_independence_loss,
#     make_s6_commutator_loss_ito, make_s7_pushforward_coeff_loss
#   - t_norm_gen, tx_norm_gen, tau_forward, xi_forward, eval_generators_jit
#
# This cell defines:
#   - LossWeights dataclass (hyperparams)
#   - mu_fn, sig_fn wrapping the neural SDE surrogate
#   - instantiated loss functions s1..s7
#   - master_loss(params_gen, tx_batch, key=None)
#   - master_loss_jit

from dataclasses import dataclass
import jax
import jax.numpy as jnp

# ----------------------- Loss weights / hyperparameters ----------------------

@dataclass
class LossWeights:
    # Algebraic structure terms
    w_s1_closure: float = 1.0   # L1: closure + constancy
    w_s2_jacobi:  float = 0.1   # L2: nested Jacobi identity
    w_s3_skew:    float = 0.1   # L3: skew-symmetry
    w_s4_bilin:   float = 0.1   # L4: bilinearity

    # Column independence
    w_s5_indep:   float = 0.1   # L5: functional independence

    # SDE determining equations
    w_s6_det:     float = 1.0   # L6: Gaeta–Quintero determining equations

    # Finite-ε flow-validity
    w_s7_push:    float = 0.1   # L7: pushforward on (μ, σ)

    # Generator weight decay
    weight_decay: float = 1e-6

    # S5 options
    s5_mode: str = "sigma"      # "sigma" or "corr_l2"
    s5_tau:  float = 0.8

    # S7 options
    s7_eps:   float = 1e-2
    s7_steps: int   = 1

loss_cfg = LossWeights()

# ----------------------- Drift / diffusion wrappers (μ, σ) -------------------

MU_CLIP  = 50.0
SIG_CLIP = 50.0

def mu_fn(t, x):
    f_hat, _ = surrogate_f_sigma(params_sde, in_norm_sde, t, x)
    f_hat = jnp.nan_to_num(f_hat, nan=0.0, posinf=0.0, neginf=0.0)
    f_hat = jnp.clip(f_hat, -MU_CLIP, MU_CLIP)
    return f_hat

def sig_fn(t, x):
    _, sigma_hat = surrogate_f_sigma(params_sde, in_norm_sde, t, x)
    sigma_hat = jnp.nan_to_num(sigma_hat, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_hat = jnp.clip(sigma_hat, -SIG_CLIP, SIG_CLIP)
    return sigma_hat


# ----------------------- Instantiate per-term loss functions -----------------

# L1: Lie bracket closure + constancy
s1_lie_loss = make_s1_lie_loss(n_generators=gen_cfg.n_generators)

# L2: Jacobi identity (nested brackets)
s2_jacobi_loss = make_s2_jacobi_loss_nested(n_generators=gen_cfg.n_generators)

# L3: Skew-symmetry
s3_skew_loss = make_s3_skewsym_loss(n_generators=gen_cfg.n_generators)

# L4: Bilinearity (random coefficient pairs)
s4_bilin_loss = make_s4_bilinearity_loss(
    n_generators=gen_cfg.n_generators,
    num_cc=4,
    cc_list=None,
    normalize=True,
)

# L5: Column independence (functional independence of generators)
s5_indep_loss = make_s5_column_independence_loss(
    n_generators=gen_cfg.n_generators,
    mode=loss_cfg.s5_mode,
    tau=loss_cfg.s5_tau,
    eps=1e-12,
)

# L6: SDE determining equations (Gaeta–Quintero)
s6_det_loss = make_s6_commutator_loss_ito(
    mu_fn=mu_fn,
    sig_fn=sig_fn,
    use_abs=False,   # L2-style penalty
)

# L7: Finite-ε pushforward validity on (μ, σ)
s7_push_loss_raw = make_s7_pushforward_coeff_loss_sde_only(
    mu_fn=mu_fn,
    sig_fn=sig_fn,
    eps=1e-2,
    num_steps=1,
    sigma_floor=1e-6,
    dt_floor=1e-10,
    dt_neg_penalty=10.0,
    activation=gen_cfg.activation,
    normalize_tx=None,
    jit=True,
    tau_clip=5.0,
    xi_clip=5.0,
    x_clip_abs=25.0,
)

# Wrap it so Master Loss can call it as (params_gen, tx_batch)
# ----------------------- S7 data (auto-define once, no extra cell) -----------------------
if ("t_s7" not in globals()) or ("x_paths_s7" not in globals()):
    # pick a deterministic small slice for S7
    n_traj_use = int(min(64, x_surr.shape[0]))
    Np1_use    = int(min(401, x_surr.shape[1]))  # 400 increments

    t_s7 = jnp.asarray(t_surr[:Np1_use], dtype=jnp.float64)                 # (T,)
    x_paths_s7 = jnp.asarray(x_surr[:n_traj_use, :Np1_use], dtype=jnp.float64)  # (n_traj,T)
    globals().update({"t_s7": t_s7, "x_paths_s7": x_paths_s7})

def s7_push_loss(params_gen, tx_batch):
    return s7_push_loss_raw(params_gen, t_s7, x_paths_s7)



# ----------------------- Helper: L2 norm over a pytree -----------------------

def l2_tree(params):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

# ----------------------- Master loss ----------------------------------------

def master_loss(params_gen, tx_batch: jnp.ndarray, key=None):
    """
    Compute weighted sum of L1–L7 plus weight decay.

    Args:
      params_gen: generator parameters {"tau": [...], "xi": [...]}
      tx_batch:   (B,2) array of (t,x) points used for all losses.
      key:        optional PRNGKey for L4 (bilinearity) coefficients; if None,
                  L4 uses its own default key each call.

    Returns:
      total_loss, aux where aux is a dict with per-term components.
    """
    total = 0.0
    aux = {}

    # L1: closure + constancy
    loss_s1, aux_s1 = s1_lie_loss(params_gen, tx_batch)
    total = total + loss_cfg.w_s1_closure * loss_s1
    aux["L1"] = {"loss": loss_s1, **aux_s1}

    # L2: Jacobi
    loss_s2, aux_s2 = s2_jacobi_loss(params_gen, tx_batch)
    total = total + loss_cfg.w_s2_jacobi * loss_s2
    aux["L2"] = {"loss": loss_s2, **aux_s2}

    # L3: skew-symmetry
    loss_s3, aux_s3 = s3_skew_loss(params_gen, tx_batch)
    total = total + loss_cfg.w_s3_skew * loss_s3
    aux["L3"] = {"loss": loss_s3, **aux_s3}

    # L4: bilinearity (needs key)
    loss_s4, aux_s4 = s4_bilin_loss(params_gen, tx_batch, key=key)
    total = total + loss_cfg.w_s4_bilin * loss_s4
    aux["L4"] = {"loss": loss_s4, **aux_s4}

    # L5: column independence
    loss_s5, aux_s5 = s5_indep_loss(params_gen, tx_batch)
    total = total + loss_cfg.w_s5_indep * loss_s5
    aux["L5"] = {"loss": loss_s5, **aux_s5}

    # L6: SDE determining equations
    loss_s6, aux_s6 = s6_det_loss(params_gen, tx_batch)
    total = total + loss_cfg.w_s6_det * loss_s6
    aux["L6"] = {"loss": loss_s6, **aux_s6}

    # L7: pushforward (flow-validity)
    loss_s7, aux_s7 = s7_push_loss(params_gen, tx_batch)
    total = total + loss_cfg.w_s7_push * loss_s7
    aux["L7"] = {"loss": loss_s7, **aux_s7}

    # Weight decay on generator parameters
    wd = loss_cfg.weight_decay * l2_tree(params_gen)
    total = total + wd
    aux["weight_decay"] = wd

    aux["total"] = total
    return total, aux

master_loss_jit = jax.jit(master_loss)

# Export to globals so the training cell can just call master_loss_jit
globals().update({
    "loss_cfg": loss_cfg,
    "master_loss": master_loss,
    "master_loss_jit": master_loss_jit,
})


# In[ ]:


# ============================ Generator training loop (Stage 2) ============================
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

# ---------- training config ----------
@dataclass
class GenTrainConfig:
    steps: int = 3000
    batch_size: int = 2048
    lr: float = 1e-3
    print_every: int = 100

gen_train_cfg = GenTrainConfig()

# ---------- re-init params_gen if any int32 leaves leaked in (grad can't handle ints) ----------
def _reinit_params_gen(key, gen_cfg):
    assert "init_mlp_params" in globals(), "Missing init_mlp_params."
    m = int(getattr(gen_cfg, "n_generators"))
    hidden_tau  = int(getattr(gen_cfg, "hidden_tau", 32))
    hidden_xi   = int(getattr(gen_cfg, "hidden_xi", 64))
    hidden_beta = int(getattr(gen_cfg, "hidden_beta", 64))

    keys = jax.random.split(key, 3 * m)
    params_tau_list, params_xi_list, params_beta_list = [], [], []

    for i in range(m):
        k_tau  = keys[3 * i]
        k_xi   = keys[3 * i + 1]
        k_beta = keys[3 * i + 2]

        params_tau  = init_mlp_params(k_tau,  sizes=[1, hidden_tau,  hidden_tau,  1])
        params_xi   = init_mlp_params(k_xi,   sizes=[2, hidden_xi,   hidden_xi,   1])
        params_beta = init_mlp_params(k_beta, sizes=[2, hidden_beta, hidden_beta, 1])

        params_tau_list.append(params_tau)
        params_xi_list.append(params_xi)
        params_beta_list.append(params_beta)

    return {"tau": params_tau_list, "xi": params_xi_list, "beta": params_beta_list}

_leaves = jax.tree_util.tree_leaves(params_gen)
_has_int = any(isinstance(a, jnp.ndarray) and jnp.issubdtype(a.dtype, jnp.integer) for a in _leaves)
if _has_int:
    key_main, key_gen = jax.random.split(key_main)
    params_gen = _reinit_params_gen(key_gen, gen_cfg)

# ---------- optimizer ----------
assert "optax" in globals(), "optax must be imported."
gen_train_cfg.lr = 1e-4  # <-- LR

optimizer_gen = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(gen_train_cfg.lr),
)

opt_state_gen = optimizer_gen.init(params_gen)

# ---------- minibatch sampler ----------
TX_gen_np = np.asarray(TX_gen, dtype=np.float64)
mask = np.isfinite(TX_gen_np).all(axis=1)
TX_gen_np = TX_gen_np[mask]
N_tx = TX_gen_np.shape[0]
assert N_tx > 0, "All TX_gen points are non-finite after filtering."

rng_np_gen = np.random.default_rng(42)

def sample_tx_batch(batch_size: int):
    if batch_size >= N_tx:
        idx = np.arange(N_tx)
    else:
        idx = rng_np_gen.choice(N_tx, size=batch_size, replace=False)
    return jnp.asarray(TX_gen_np[idx], dtype=jnp.float64)

# ---------- one JIT-ed training step ----------
@jax.jit
def train_step(params_gen, opt_state_gen, tx_batch, key):
    def loss_for_grad(p):
        loss_val, aux = master_loss(p, tx_batch, key)
        return loss_val, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_for_grad, has_aux=True)(params_gen)
    updates, opt_state_new = optimizer_gen.update(grads, opt_state_gen, params_gen)
    params_new = optax.apply_updates(params_gen, updates)
    return params_new, opt_state_new, loss_val, aux

# ---------- helpers for logging ----------
LOSS_KEYS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

def _safe_float(x):
    # works for scalars and 0-d arrays
    return float(jnp.asarray(x))

def format_log(step, total_loss, aux):
    # Pull per-term losses if present
    parts = [f"step {step:5d}/{gen_train_cfg.steps}", f"total={_safe_float(total_loss):.6e}"]
    for k in LOSS_KEYS:
        if k in aux and "loss" in aux[k]:
            parts.append(f"{k}={_safe_float(aux[k]['loss']):.3e}")
    if "weight_decay" in aux:
        parts.append(f"wd={_safe_float(aux['weight_decay']):.3e}")
    return " | ".join(parts)

# ---------- training loop ----------
loss_history_gen = []
loss_hist = {k: [] for k in LOSS_KEYS}
wd_history = []

key_main, key_train = jax.random.split(key_main, 2)

print(f"Starting generator training for {gen_train_cfg.steps} steps "
      f"with batch_size={gen_train_cfg.batch_size}, lr={gen_train_cfg.lr}")

for step in range(1, gen_train_cfg.steps + 1):
    tx_batch = sample_tx_batch(gen_train_cfg.batch_size)
    key_train, key_step = jax.random.split(key_train)

    params_gen, opt_state_gen, loss_val, aux = train_step(params_gen, opt_state_gen, tx_batch, key_step)

    loss_history_gen.append(_safe_float(loss_val))

    for k in LOSS_KEYS:
        if k in aux and "loss" in aux[k]:
            loss_hist[k].append(_safe_float(aux[k]["loss"]))
        else:
            loss_hist[k].append(np.nan)

    wd_history.append(_safe_float(aux.get("weight_decay", 0.0)))

    if (step % gen_train_cfg.print_every == 0) or (step == 1) or (step == gen_train_cfg.steps):
        print(format_log(step, loss_val, aux))

print("\nGenerator training finished.")

globals().update({
    "params_gen": params_gen,
    "opt_state_gen": opt_state_gen,
    "loss_history_gen": loss_history_gen,
    "loss_hist": loss_hist,          # dict: keys L1..L7 -> list
    "wd_history": wd_history,
    "gen_train_cfg": gen_train_cfg,
})


# In[ ]:


# === Evaluation 2b: stronger span check (block-balanced + best-mixing residual) ===
# Compatible with upstream: uses eval_generators_jit(params_gen, t, x) and ignores extra returns.

import jax
import jax.numpy as jnp
import numpy as np

# -------- helper: robustly extract (tau, xi) from eval_generators_jit ----------
def _eval_tau_xi(params_gen, t, x):
    out = eval_generators_jit(params_gen, t, x)
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError("eval_generators_jit must return (tau, xi, ...) with at least 2 outputs.")
    return out[0], out[1]

# -------- helper: per-generator block balancing (tau/xi comparable energy) ----
def _stack_balanced(tau, xi, eps=1e-12):
    """
    tau, xi: (m, B)
    Returns V: (2B, m) where each column i has tau and xi separately standardized to unit RMS.
    """
    # RMS per generator per block
    tau_rms = jnp.sqrt(jnp.mean(tau**2, axis=1, keepdims=True) + eps)
    xi_rms  = jnp.sqrt(jnp.mean(xi**2,  axis=1, keepdims=True) + eps)
    tau_s = tau / tau_rms
    xi_s  = xi  / xi_rms
    V = jnp.concatenate([tau_s, xi_s], axis=1).T  # (2B, m)
    return V

# -------- helper: principal angles between column spans -----------------------
def principal_angles(V, W):
    # V,W: (D,k)
    Q1, _ = jnp.linalg.qr(V, mode="reduced")
    Q2, _ = jnp.linalg.qr(W, mode="reduced")
    s = jnp.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = jnp.clip(s, -1.0, 1.0)
    return jnp.sort(jnp.arccos(s))

# -------- helper: best-mixing residual ---------------------------------------
def best_mixing_residual(V, W, ridge=1e-10):
    """
    Solve A* = argmin_A ||V - W A||_F (with tiny ridge), report relative residual.
    V,W: (D,k), k is small (e.g. 3)
    """
    # A = (W^T W + ridge I)^{-1} W^T V
    k = W.shape[1]
    WT_W = W.T @ W
    A = jnp.linalg.solve(WT_W + ridge * jnp.eye(k, dtype=W.dtype), W.T @ V)
    Vhat = W @ A
    rel = jnp.linalg.norm(V - Vhat) / (jnp.linalg.norm(V) + 1e-12)
    return rel, A

# -------- choose evaluation points: prefer empirical TX_gen if available -------
USE_TX_GEN = True
B_max = 5000  # cap for speed

if USE_TX_GEN and ("TX_gen" in globals()):
    TX = jnp.asarray(TX_gen, dtype=jnp.float64)
    TX = TX[jnp.isfinite(TX).all(axis=1)]
    if TX.shape[0] == 0:
        raise ValueError("TX_gen is empty after filtering non-finites.")
    B = int(min(B_max, TX.shape[0]))
    t_flat = TX[:B, 0]
    x_flat = TX[:B, 1]
    print(f"[span-check] Using empirical TX_gen points: B={B}")
    print("t range:", float(t_flat.min()), float(t_flat.max()))
    print("x range:", float(x_flat.min()), float(x_flat.max()))

else:
    # fallback to a uniform grid using t_surr/x_surr if present
    t_min_eval = float(t_surr.min()) if "t_surr" in globals() else 0.0
    t_max_eval = float(t_surr.max()) if "t_surr" in globals() else 5.0
    x_min_eval = float(x_surr.min()) if "x_surr" in globals() else -4.0
    x_max_eval = float(x_surr.max()) if "x_surr" in globals() else 6.0
    Nt_eval, Nx_eval = 40, 40
    t_eval = jnp.linspace(t_min_eval, t_max_eval, Nt_eval)
    x_eval = jnp.linspace(x_min_eval, x_max_eval, Nx_eval)
    TT, XX = jnp.meshgrid(t_eval, x_eval, indexing="ij")
    t_flat = TT.reshape(-1)
    x_flat = XX.reshape(-1)
    print("t range:", float(t_flat.min()), float(t_flat.max()))
    print("x range:", float(x_flat.min()), float(x_flat.max()))

    B = int(t_flat.shape[0])
    print(f"[span-check] Using uniform grid: B={B} on t∈[{t_min_eval},{t_max_eval}], x∈[{x_min_eval},{x_max_eval}]")

# -------- learned generators --------------------------------------------------
m = int(gen_cfg.n_generators)
tau_learn, xi_learn = _eval_tau_xi(params_gen, t_flat, x_flat)  # (m,B)

# -------- ground-truth subspace for m=3 check --------------------------------
# If trained m != 3, compare against a 3D ground-truth span by taking k=3.
# Here we keep the same ground-truth basis used: {∂t, ∂x, 2t∂t + x∂x}.
w1_tau = jnp.ones_like(t_flat)
w1_xi  = jnp.zeros_like(x_flat)

w2_tau = jnp.zeros_like(t_flat)
w2_xi  = jnp.ones_like(x_flat)

w3_tau = 2.0 * t_flat
w3_xi  = x_flat

W_tau = jnp.stack([w1_tau, w2_tau, w3_tau], axis=0)  # (3,B)
W_xi  = jnp.stack([w1_xi,  w2_xi,  w3_xi ], axis=0)  # (3,B)

# -------- build balanced stacked matrices V and W -----------------------------
# For learned: take first k=3 generators if m>3
k = 3
if m < k:
    raise ValueError(f"Need at least {k} learned generators for this check; got m={m}.")

V_bal = _stack_balanced(tau_learn[:k], xi_learn[:k])   # (2B, k)
W_bal = _stack_balanced(W_tau, W_xi)                   # (2B, k)

# Also compute the "raw" (unbalanced) version for comparison
V_raw = jnp.concatenate([tau_learn[:k], xi_learn[:k]], axis=1).T  # (2B,k)
W_raw = jnp.concatenate([W_tau, W_xi], axis=1).T                  # (2B,k)

# -------- principal angles (raw vs balanced) ---------------------------------
angles_raw = principal_angles(V_raw, W_raw)
angles_bal = principal_angles(V_bal, W_bal)

def _deg(x): return x * (180.0 / jnp.pi)

print("\nPrincipal angles (RAW stacking):")
for i, a in enumerate(angles_raw, 1):
    print(f"  angle {i}: {float(a):.6f} rad = {float(_deg(a)):.4f} deg")

print("\nPrincipal angles (BALANCED tau/xi per generator):")
for i, a in enumerate(angles_bal, 1):
    print(f"  angle {i}: {float(a):.6f} rad = {float(_deg(a)):.4f} deg")

# -------- best-mixing residual (raw vs balanced) ------------------------------
rel_raw, A_raw = best_mixing_residual(V_raw, W_raw)
rel_bal, A_bal = best_mixing_residual(V_bal, W_bal)

print("\nBest-mixing relative residuals (lower is better):")
print(f"  RAW:      ||V - W A*|| / ||V|| = {float(rel_raw):.6e}")
print(f"  BALANCED: ||V - W A*|| / ||V|| = {float(rel_bal):.6e}")

print("\nBest-mixing matrix A* (BALANCED), columns correspond to learned generators in V:")
print(np.asarray(A_bal))

# -------- optional: pairwise cosines after balancing --------------------------
def cos_sim(a, b):
    return float((a @ b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12))

print("\nPairwise cosine similarities using BALANCED stacked columns:")
for i in range(k):
    for j in range(k):
        cij = cos_sim(V_bal[:, i], W_bal[:, j])
        print(f"  <X_{i+1}, v_{j+1}> = {cij:.4f}")

