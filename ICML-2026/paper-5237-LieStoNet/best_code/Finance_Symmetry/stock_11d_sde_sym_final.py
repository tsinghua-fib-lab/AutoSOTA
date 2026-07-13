"""
11D Stock SDE Symmetry Discovery via LieStoNet
===============================================
Discovers Lie-point symmetries of an 11-dimensional SDE system
learned from real stock price data (11 tickers).

Pipeline:
  Stage A: Download 11 stock log-prices via yfinance, fit neural SDE surrogate
  Stage B: Train m generators X_i = tau_i(t)d_t + xi_i(t,x)d_x (xi: R^12->R^11)
  Stage C: m-sweep, after-push evaluation, visualization

Run on Colab:
  !pip install yfinance
  import os;
  exec(open('stock_11d_sde_sym.py').read())
"""

import math, time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass

jax.config.update("jax_enable_x64", True)
print(f"JAX devices: {jax.devices()}")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

N_SPATIAL = 11          # spatial dimension (number of stocks)
N_INPUT   = N_SPATIAL + 1  # = 12 (time + stocks)
SIGMA_MIN = 1e-3
SIGMA_FLOOR = 1e-8
fd = 1e-3               # finite-difference step

TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ",
           "CVX", "PG", "UNH", "HD", "NEE", "GLD"]

DATA_START = "2020-01-01"
DATA_END   = "2025-12-31"

key_main = jax.random.PRNGKey(42)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class Normalizer:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

def fit_normalizer(X):
    return Normalizer(jnp.mean(X, axis=0), jnp.std(X, axis=0))

def glorot(key, fi, fo):
    lim = math.sqrt(6.0 / (fi + fo))
    return jax.random.uniform(key, (fi, fo), minval=-lim, maxval=lim, dtype=jnp.float64)

def init_mlp_params(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    return [{"W": glorot(k, m, n), "b": jnp.zeros((n,), dtype=jnp.float64)}
            for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:]))]

def mlp_forward(params, x, activation="tanh"):
    h = x
    for i, layer in enumerate(params):
        h = h @ layer["W"] + layer["b"]
        if i < len(params) - 1:
            if activation == "tanh":
                h = jnp.tanh(h)
            elif activation == "swish":
                h = h * jax.nn.sigmoid(h)
            else:
                h = jax.nn.relu(h)
    return h

def l2_tree(params):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA DOWNLOAD + PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(" Section 1: Downloading and preprocessing stock data")
print("="*60)

try:
    import yfinance as yf
    print(f"  Downloading {len(TICKERS)} tickers: {TICKERS}")
    prices_df = yf.download(TICKERS, start=DATA_START, end=DATA_END, auto_adjust=True)["Close"]
    prices_df = prices_df[TICKERS]  # enforce order
    prices_df = prices_df.ffill().bfill().dropna()  # forward-fill gaps, then drop any remaining
    if len(prices_df) < 100:
        raise ValueError(f"Only {len(prices_df)} rows after cleaning — download likely failed")
    prices_np = prices_df.values.astype(np.float64)  # (n_days, 11)
    n_days = prices_np.shape[0]
    print(f"  Downloaded: {n_days} trading days, {prices_np.shape[1]} stocks")
    USE_REAL_DATA = True
except Exception as e:
    print(f"  yfinance failed ({e}), generating synthetic correlated GBM data")
    USE_REAL_DATA = False
    n_days = 1260
    rng_syn = np.random.default_rng(123)
    # Correlated GBM: dS = mu*S*dt + sigma*S*dW, with cross-correlation
    mu_syn = 0.05 + 0.1 * rng_syn.standard_normal(N_SPATIAL)
    sigma_syn = 0.15 + 0.05 * rng_syn.standard_normal(N_SPATIAL)
    # Random correlation matrix
    A = rng_syn.standard_normal((N_SPATIAL, N_SPATIAL))
    corr = A @ A.T
    corr = corr / np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    L = np.linalg.cholesky(corr)
    dt_syn = 1.0 / 252.0
    prices_np = np.zeros((n_days, N_SPATIAL))
    prices_np[0] = 50.0 + 50.0 * rng_syn.uniform(size=N_SPATIAL)  # initial prices
    for i in range(1, n_days):
        dW = L @ rng_syn.standard_normal(N_SPATIAL)
        prices_np[i] = prices_np[i-1] * np.exp(
            (mu_syn - 0.5 * sigma_syn**2) * dt_syn + sigma_syn * np.sqrt(dt_syn) * dW)

# Log prices
log_prices = np.log(prices_np)  # (n_days, 11)
t_full = np.linspace(0.0, (n_days - 1) / 252.0, n_days)  # normalized time in years
dt_data = t_full[1] - t_full[0]

# Increments
dx_full = np.diff(log_prices, axis=0)  # (n_days-1, 11)

# Outlier removal: per-dimension 4-sigma filter
keep = np.ones(dx_full.shape[0], dtype=bool)
for j in range(N_SPATIAL):
    keep &= np.abs(dx_full[:, j]) <= 4.0 * np.std(dx_full[:, j])
t_inc = t_full[:-1][keep]
x_inc = log_prices[:-1][keep]  # (n_keep, 11)
dx_inc = dx_full[keep]  # (n_keep, 11)
n_keep = dx_inc.shape[0]

# Augment: mirror increments
X_raw_aug = np.concatenate([
    np.concatenate([t_inc[:, None], x_inc], axis=1),
    np.concatenate([t_inc[:, None], x_inc], axis=1)
], axis=0)  # (2*n_keep, 12)
dX_aug = np.concatenate([dx_inc, -dx_inc], axis=0)  # (2*n_keep, 11)

# Domain bounds
T_MIN, T_MAX = 0.0, float(t_full.max())
X_MIN = log_prices.min(axis=0) - 0.01  # (11,)
X_MAX = log_prices.max(axis=0) + 0.01  # (11,)

print(f"  Time range: [{T_MIN:.2f}, {T_MAX:.2f}] years, dt={dt_data:.6f}")
print(f"  Increments: {n_keep} usable (after filter), augmented to {X_raw_aug.shape[0]}")
print(f"  Log-price ranges: min={log_prices.min():.2f}, max={log_prices.max():.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: SDE SURROGATE TRAINING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(" Section 2: Training 11D SDE surrogate")
print("="*60)

def f_sigma_hat(params, tx_norm):
    """Returns f: (..., 11), sigma: (..., 11)."""
    out = mlp_forward(params, tx_norm, activation="tanh")
    f = out[..., :N_SPATIAL]
    sigma = jax.nn.softplus(out[..., N_SPATIAL:]) + SIGMA_MIN
    return f, sigma

X_raw_jnp = jnp.array(X_raw_aug, dtype=jnp.float64)
dX_jnp = jnp.array(dX_aug, dtype=jnp.float64)
in_norm_sde = fit_normalizer(X_raw_jnp)
X_normed = in_norm_sde(X_raw_jnp)

key_main, k_model = jax.random.split(key_main)
params_sde = init_mlp_params(k_model, sizes=[N_INPUT, 256, 256, 256, 2 * N_SPATIAL])

SDE_STEPS, SDE_BATCH, SDE_WD = 30000, 1024, 1e-5

def sde_loss_fn(params, xb, dxb):
    f_hat, sigma_hat = f_sigma_hat(params, xb)  # (B, 11), (B, 11)
    mu = f_hat * dt_data
    var = (sigma_hat ** 2) * dt_data
    nll = jnp.mean(jnp.sum((dxb - mu)**2 / (2.0 * var) + 0.5 * jnp.log(var + 1e-12), axis=1))
    return nll + SDE_WD * l2_tree(params)

sde_loss_fn = jax.jit(sde_loss_fn)
sde_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-6, peak_value=1e-3, warmup_steps=500,
    decay_steps=SDE_STEPS, end_value=1e-5)
optimizer_sde = optax.adam(sde_schedule)
opt_state_sde = optimizer_sde.init(params_sde)

@jax.jit
def train_step_sde(params, opt_state, xb, dxb):
    val, grads = jax.value_and_grad(sde_loss_fn)(params, xb, dxb)
    updates, new_state = optimizer_sde.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_state, val

rng_np = np.random.default_rng(0)
print(f"  Training: {SDE_STEPS} steps, batch={SDE_BATCH}, arch=[{N_INPUT},256,256,256,{2*N_SPATIAL}]")
t0_train = time.time()
for step in range(1, SDE_STEPS + 1):
    idx = rng_np.choice(X_normed.shape[0], size=min(SDE_BATCH, X_normed.shape[0]), replace=False)
    params_sde, opt_state_sde, lv = train_step_sde(
        params_sde, opt_state_sde, X_normed[idx], dX_jnp[idx])
    if step % 5000 == 0 or step == 1:
        print(f"    step {step:5d}/{SDE_STEPS} | NLL = {float(lv):.6e}")
print(f"  Done in {time.time()-t0_train:.1f}s")

# Surrogate evaluation wrappers
def surrogate_f_sigma(t, x):
    """t: (B,), x: (B, 11). Returns f: (B, 11), sigma: (B, 11)."""
    t_b = jnp.asarray(t, jnp.float64).reshape(-1)
    x_b = jnp.asarray(x, jnp.float64).reshape(-1, N_SPATIAL)
    tx = jnp.concatenate([t_b[:, None], x_b], axis=1)  # (B, 12)
    f, s = f_sigma_hat(params_sde, in_norm_sde(tx))
    return f, s

def mu_fn(t, x):
    f, _ = surrogate_f_sigma(t, x)
    return jnp.clip(jnp.nan_to_num(f, nan=0.0), -50.0, 50.0)

def sig_fn(t, x):
    _, s = surrogate_f_sigma(t, x)
    return jnp.clip(jnp.nan_to_num(s, nan=0.0), -50.0, 50.0)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: SURROGATE PATH SIMULATION + TX_gen
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(" Section 3: Simulating surrogate paths (11D)")
print("="*60)

def simulate_surrogate_paths(key, x0, T, dt_sim):
    """x0: (n_traj, 11). Returns t: (N+1,), x: (n_traj, N+1, 11)."""
    N_sim = int(T / dt_sim)
    t_sim = jnp.linspace(0.0, T, N_sim + 1, dtype=jnp.float64)
    n_tr = x0.shape[0]
    dW = jax.random.normal(key, shape=(N_sim, n_tr, N_SPATIAL), dtype=jnp.float64) * math.sqrt(dt_sim)
    def step(x_t, inputs):
        dw, t_n = inputs
        t_batch = jnp.full(n_tr, t_n)
        f, s = surrogate_f_sigma(t_batch, x_t)
        x_new = x_t + f * dt_sim + s * dw
        return x_new, x_new
    x0_jnp = jnp.asarray(x0, jnp.float64)
    _, x_hist = jax.lax.scan(step, x0_jnp, (dW, t_sim[:-1]))  # (N, n_traj, 11)
    x_all = jnp.concatenate([x0_jnp[None, :, :], x_hist], axis=0)  # (N+1, n_traj, 11)
    return t_sim, x_all.transpose(1, 0, 2)  # (n_traj, N+1, 11)

n_gen = 512
x0_mean = log_prices.mean(axis=0)  # (11,)
x0_std = np.maximum(log_prices.std(axis=0), 1e-4)  # (11,)
rng_ic = np.random.default_rng(0)
x0_gen = x0_mean[None, :] + 0.5 * x0_std[None, :] * rng_ic.standard_normal((n_gen, N_SPATIAL))

key_main, key_sim = jax.random.split(key_main)
sim_T = float(T_MAX)
t_surr, x_surr = simulate_surrogate_paths(key_sim, x0_gen, sim_T, dt_data)
print(f"  Simulated {n_gen} paths, T={sim_T:.2f}, shape: t={t_surr.shape}, x={x_surr.shape}")

# Build TX_gen: (B, 12) point cloud
t_mat = jnp.broadcast_to(t_surr[None, :, None], (n_gen, t_surr.shape[0], 1))
tx_all = jnp.concatenate([t_mat, x_surr], axis=2)  # (n_gen, N+1, 12)
TX_gen = tx_all.reshape(-1, N_INPUT)
mask = jnp.isfinite(TX_gen).all(axis=1)
TX_gen = TX_gen[mask]
print(f"  TX_gen: {TX_gen.shape}")

tx_norm_gen = fit_normalizer(TX_gen)
t_norm_gen = fit_normalizer(TX_gen[:, 0:1])

# L7 trajectory data: subsample
n_traj_s7 = min(48, n_gen)
N_s7 = 32
stride_s7 = max(1, (t_surr.shape[0] - 1) // N_s7)
idx_s7 = np.arange(0, t_surr.shape[0], stride_s7)[:N_s7 + 1]
t_s7 = jnp.asarray(t_surr[idx_s7], dtype=jnp.float64)
x_s7 = jnp.asarray(x_surr[:n_traj_s7][:, idx_s7, :], dtype=jnp.float64)  # (n_traj_s7, N_s7+1, 11)
print(f"  L7 data: t_s7={t_s7.shape}, x_s7={x_s7.shape}")

TX_gen_np = np.asarray(TX_gen, dtype=np.float64)
TX_gen_np = TX_gen_np[np.isfinite(TX_gen_np).all(axis=1)]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: GENERATOR NETWORKS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print(" Section 4: Generator network setup (11D)")
print("="*60)

def init_generator_params(key, m):
    keys = jax.random.split(key, 2 * m)
    return {
        "tau": [init_mlp_params(keys[2*i], [1, 64, 64, 1]) for i in range(m)],
        "xi":  [init_mlp_params(keys[2*i+1], [N_INPUT, 128, 128, N_SPATIAL]) for i in range(m)]
    }

def make_eval_generators():
    def eval_gen(params_gen, t, x):
        """t: (B,), x: (B, 11). Returns tau: (m, B), xi: (m, B, 11)."""
        t_in = t_norm_gen(jnp.asarray(t, jnp.float64).reshape(-1, 1))  # (B, 1)
        x_arr = jnp.asarray(x, jnp.float64).reshape(-1, N_SPATIAL)  # (B, 11)
        tx_in = tx_norm_gen(jnp.concatenate([t_in * (t_norm_gen.std + 1e-8) + t_norm_gen.mean,
                                              x_arr], axis=1).reshape(-1, N_INPUT))
        # Re-normalize the full (t,x) vector
        tx_raw = jnp.concatenate([jnp.asarray(t, jnp.float64).reshape(-1, 1), x_arr], axis=1)
        tx_in = tx_norm_gen(tx_raw)
        tau_l = [mlp_forward(pt, t_in, activation="tanh")[:, 0] for pt in params_gen["tau"]]
        xi_l = [mlp_forward(px, tx_in, activation="tanh") for px in params_gen["xi"]]
        return jnp.stack(tau_l, 0), jnp.stack(xi_l, 0)  # (m, B), (m, B, 11)
    return jax.jit(eval_gen)

eval_gen_jit = make_eval_generators()
print(f"  tau: [1, 64, 64, 1] per generator")
print(f"  xi:  [{N_INPUT}, 128, 128, {N_SPATIAL}] per generator")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: FINITE-DIFFERENCE DERIVATIVES (11D)
# ═══════════════════════════════════════════════════════════════════════════

def _gen_derivs(pg, t, x):
    """Compute all FD derivatives for the 11D generator.
    t: (B,), x: (B, 11)
    Returns:
        tau:    (m, B)
        xi:     (m, B, 11)
        tau_t:  (m, B)
        xi_t:   (m, B, 11)
        xi_x:   (m, B, 11, 11)  — Jacobian dxi^a/dx_j
        xi_xx:  (m, B, 11, 11)  — diagonal Hessian d^2 xi^a/dx_j^2
    """
    tau, xi = eval_gen_jit(pg, t, x)
    tau_tp, xi_tp = eval_gen_jit(pg, t + fd, x)
    tau_tm, xi_tm = eval_gen_jit(pg, t - fd, x)
    tau_t = (tau_tp - tau_tm) / (2 * fd)
    xi_t = (xi_tp - xi_tm) / (2 * fd)

    xi_xp_list, xi_xm_list = [], []
    for j in range(N_SPATIAL):
        e_j = jnp.zeros((1, N_SPATIAL), dtype=jnp.float64).at[0, j].set(fd)
        _, xp = eval_gen_jit(pg, t, x + e_j)
        _, xm = eval_gen_jit(pg, t, x - e_j)
        xi_xp_list.append(xp)
        xi_xm_list.append(xm)

    xi_x = jnp.stack([(xp - xm) / (2 * fd) for xp, xm in zip(xi_xp_list, xi_xm_list)], axis=-1)
    xi_xx = jnp.stack([(xp - 2 * xi + xm) / (fd**2) for xp, xm in zip(xi_xp_list, xi_xm_list)], axis=-1)
    return tau, xi, tau_t, xi_t, xi_x, xi_xx


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: LIE BRACKET (12-component)
# ═══════════════════════════════════════════════════════════════════════════

def _bracket_from_derivs(tau, xi, tau_t, xi_t, xi_x, i, j):
    """Compute [X_i, X_j] from pre-computed derivatives.
    Returns bt: (B,), bx: (B, 11).
    """
    bt = tau[i] * tau_t[j] - tau[j] * tau_t[i]  # (B,)
    # bx^a = tau_i * dxi^a_j/dt + sum_k xi^k_i * dxi^a_j/dx_k  - (i<->j)
    conv_ij = jnp.einsum('bk,bak->ba', xi[i], xi_x[j])  # (B, 11)
    conv_ji = jnp.einsum('bk,bak->ba', xi[j], xi_x[i])  # (B, 11)
    bx = (tau[i][:, None] * xi_t[j] + conv_ij
         - tau[j][:, None] * xi_t[i] - conv_ji)
    return bt, bx


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: LOSS FUNCTIONS L1-L7 (all 11D)
# ═══════════════════════════════════════════════════════════════════════════

# ── L1: Closure ──────────────────────────────────────────────────────────

def s1_loss(pg, tx):
    t, x = tx[:, 0], tx[:, 1:]  # t: (B,), x: (B, 11)
    tau, xi, tau_t, xi_t, xi_x, _ = _gen_derivs(pg, t, x)
    m = tau.shape[0]; B = t.shape[0]
    if m < 2:
        return jnp.float64(0.0), {}
    total = jnp.float64(0.0); count = 0
    # A matrix: (12*B, m) — each column is the full vector field of generator k
    A_tau = tau.T  # (B, m)
    A_xi = xi.transpose(1, 2, 0).reshape(B * N_SPATIAL, m)  # (11*B, m)
    A = jnp.concatenate([A_tau, A_xi], axis=0)  # (12*B, m)
    for i in range(m):
        for j in range(i + 1, m):
            bt, bx = _bracket_from_derivs(tau, xi, tau_t, xi_t, xi_x, i, j)
            b = jnp.concatenate([bt, bx.reshape(-1)], axis=0)  # (12*B,)
            c, _, _, _ = jnp.linalg.lstsq(A, b)
            total = total + jnp.mean((A @ c - b)**2)
            count += 1
    return total / max(count, 1), {}


# ── L2: Jacobi identity ─────────────────────────────────────────────────

def s2_loss(pg, tx):
    t, x = tx[:, 0], tx[:, 1:]
    tau, xi, tau_t, xi_t, xi_x, _ = _gen_derivs(pg, t, x)
    m = tau.shape[0]; B = t.shape[0]
    if m < 3:
        return jnp.float64(0.0), {}
    # Fit structure constants c_ij^l for all pairs
    A_tau = tau.T; A_xi = xi.transpose(1, 2, 0).reshape(B * N_SPATIAL, m)
    A = jnp.concatenate([A_tau, A_xi], axis=0)
    c_ij = {}
    for i in range(m):
        for j in range(m):
            if i == j: continue
            bt, bx = _bracket_from_derivs(tau, xi, tau_t, xi_t, xi_x, i, j)
            b = jnp.concatenate([bt, bx.reshape(-1)], axis=0)
            c, _, _, _ = jnp.linalg.lstsq(A, b)
            c_ij[(i, j)] = c
    total = jnp.float64(0.0); count = 0
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                for s in range(m):
                    val = jnp.float64(0.0)
                    for l in range(m):
                        if (i, j) in c_ij and (l, k) in c_ij:
                            val = val + c_ij[(i, j)][l] * c_ij[(l, k)][s]
                        if (k, i) in c_ij and (l, j) in c_ij:
                            val = val + c_ij[(k, i)][l] * c_ij[(l, j)][s]
                        if (j, k) in c_ij and (l, i) in c_ij:
                            val = val + c_ij[(j, k)][l] * c_ij[(l, i)][s]
                    total = total + val**2; count += 1
    return total / max(count, 1), {}


# ── L3: Skew-symmetry ────────────────────────────────────────────────────

def s3_loss(pg, tx):
    t, x = tx[:, 0], tx[:, 1:]
    tau, xi, tau_t, xi_t, xi_x, _ = _gen_derivs(pg, t, x)
    m = tau.shape[0]
    if m < 2:
        return jnp.float64(0.0), {}
    total = jnp.float64(0.0); count = 0
    for i in range(m):
        for j in range(i + 1, m):
            bt_ij, bx_ij = _bracket_from_derivs(tau, xi, tau_t, xi_t, xi_x, i, j)
            bt_ji, bx_ji = _bracket_from_derivs(tau, xi, tau_t, xi_t, xi_x, j, i)
            total = total + jnp.mean((bt_ij + bt_ji)**2) + jnp.mean((bx_ij + bx_ji)**2)
            count += 1
    return total / max(count, 1), {}


# ── L4: Bilinearity ──────────────────────────────────────────────────────

def s4_loss(pg, tx, key=None):
    t, x = tx[:, 0], tx[:, 1:]
    tau, xi, tau_t, xi_t, xi_x, _ = _gen_derivs(pg, t, x)
    m = tau.shape[0]
    if m < 3:
        return jnp.float64(0.0), {}
    key = jax.random.PRNGKey(0) if key is None else key
    cc = jax.random.uniform(key, (4, 2), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    total = jnp.float64(0.0); count = 0
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                bt_ik, bx_ik = _bracket_from_derivs(tau, xi, tau_t, xi_t, xi_x, i, k)
                bt_jk, bx_jk = _bracket_from_derivs(tau, xi, tau_t, xi_t, xi_x, j, k)
                for ci in range(cc.shape[0]):
                    c1, c2 = cc[ci, 0], cc[ci, 1]
                    # LHS: [c1*X_i + c2*X_j, X_k] via linearity of derivatives
                    tau_combo_t = c1 * tau_t[i] + c2 * tau_t[j]
                    xi_combo_t = c1 * xi_t[i] + c2 * xi_t[j]
                    xi_combo = c1 * xi[i] + c2 * xi[j]
                    tau_combo = c1 * tau[i] + c2 * tau[j]
                    lhs_bt = tau_combo * tau_t[k] - tau[k] * tau_combo_t
                    conv_lhs = jnp.einsum('bk,bak->ba', xi_combo, xi_x[k])
                    conv_k = jnp.einsum('bk,bak->ba', xi[k],
                                        c1 * xi_x[i] + c2 * xi_x[j])
                    lhs_bx = tau_combo[:, None] * xi_t[k] + conv_lhs - tau[k][:, None] * xi_combo_t - conv_k
                    rhs_bt = c1 * bt_ik + c2 * bt_jk
                    rhs_bx = c1 * bx_ik + c2 * bx_jk
                    denom = jnp.abs(c1) + jnp.abs(c2) + 1e-12
                    total = total + jnp.mean(((lhs_bt - rhs_bt) / denom)**2)
                    total = total + jnp.mean(((lhs_bx - rhs_bx) / denom)**2)
                    count += 1
    return total / max(count, 1), {}


# ── L5: Independence ──────────────────────────────────────────────────────

def s5_loss(pg, tx):
    t, x = tx[:, 0], tx[:, 1:]
    tau, xi = eval_gen_jit(pg, t, x)  # (m, B), (m, B, 11)
    m = tau.shape[0]
    vecs = jnp.concatenate([tau[:, :, None], xi], axis=2)  # (m, B, 12)
    vecs_flat = vecs.reshape(m, -1)  # (m, 12*B)
    G = vecs_flat @ vecs_flat.T / vecs_flat.shape[1]
    return -jnp.log(jnp.linalg.det(G + 1e-6 * jnp.eye(m)) + 1e-12), {}


# ── L6: SDE determining equations (n-dim Gaeta-Quintero) ────────────────

def s6_loss(pg, tx):
    t, x = tx[:, 0], tx[:, 1:]  # t: (B,), x: (B, 11)
    tau, xi, tau_t, xi_t, xi_x, xi_xx = _gen_derivs(pg, t, x)
    m = tau.shape[0]; B = t.shape[0]

    f0, s0 = mu_fn(t, x), sig_fn(t, x)  # each (B, 11)
    # Surrogate time derivatives
    f_t = (mu_fn(t + fd, x) - mu_fn(t - fd, x)) / (2 * fd)
    s_t = (sig_fn(t + fd, x) - sig_fn(t - fd, x)) / (2 * fd)
    # Surrogate spatial derivatives: df^a/dx_j, ds_a/dx_j
    f_x_list, s_x_list = [], []
    for j in range(N_SPATIAL):
        e_j = jnp.zeros((1, N_SPATIAL), dtype=jnp.float64).at[0, j].set(fd)
        f_x_list.append((mu_fn(t, x + e_j) - mu_fn(t, x - e_j)) / (2 * fd))
        s_x_list.append((sig_fn(t, x + e_j) - sig_fn(t, x - e_j)) / (2 * fd))
    f_x = jnp.stack(f_x_list, axis=-1)  # (B, 11, 11) — f_x[b, a, j] = df^a/dx_j
    s_x = jnp.stack(s_x_list, axis=-1)  # (B, 11, 11) — s_x[b, a, j] = ds_a/dx_j

    total = jnp.float64(0.0)
    for i in range(m):
        # Drift residual R^a_i  (B, 11):
        # dxi^a/dt + sum_j f^j * dxi^a/dx_j - sum_j xi^j * df^a/dx_j
        # - (df^a/dt * tau + f^a * dtau/dt) + 0.5 * sum_j sigma_j^2 * d^2xi^a/dx_j^2
        term_advect = jnp.einsum('bj,baj->ba', f0, xi_x[i])      # sum_j f^j * dxi^a/dx_j
        term_drag = jnp.einsum('bj,baj->ba', xi[i], f_x)          # sum_j xi^j * df^a/dx_j
        term_forcing = f_t * tau[i][:, None] + f0 * tau_t[i][:, None]
        term_hessian = 0.5 * jnp.einsum('bj,baj->ba', s0**2, xi_xx[i])
        r_drift = xi_t[i] + term_advect - term_drag - term_forcing + term_hessian  # (B, 11)

        # Diffusion residual S^a_i (diagonal case) (B, 11):
        # sigma_a * dxi^a/dx_a - sum_j xi^j * ds_a/dx_j - tau * ds_a/dt - 0.5*sigma_a*dtau/dt
        diag_xi_x = jnp.diagonal(xi_x[i], axis1=1, axis2=2)  # (B, 11)
        term_diff_jac = jnp.einsum('bj,baj->ba', xi[i], s_x)
        r_diff = (s0 * diag_xi_x - term_diff_jac
                  - tau[i][:, None] * s_t - 0.5 * s0 * tau_t[i][:, None])  # (B, 11)

        total = total + jnp.mean(r_drift**2) + jnp.mean(r_diff**2)
    return total / m, {}


# ── L7: Pushforward (11D prolonged flow) ─────────────────────────────────

def make_s7_loss(eps=1e-2, num_steps=1, domain_penalty_w=10.0):
    def _flow_derivs(pg, ts, xs):
        """Compute per-generator derivatives during flow.
        ts: (m, B), xs: (m, B, 11).
        Returns all derivatives for each generator at its own pushed points.
        """
        m, B = ts.shape
        t_flat = ts.reshape(-1)  # (m*B,)
        x_flat = xs.reshape(-1, N_SPATIAL)  # (m*B, 11)
        offsets = jnp.arange(m)[:, None] * B + jnp.arange(B)[None, :]  # (m, B)

        tau_all, xi_all = eval_gen_jit(pg, t_flat, x_flat)
        tau = tau_all[jnp.arange(m)[:, None], offsets]
        xi = xi_all[jnp.arange(m)[:, None], offsets, :]

        tau_tp_all, xi_tp_all = eval_gen_jit(pg, t_flat + fd, x_flat)
        tau_tm_all, xi_tm_all = eval_gen_jit(pg, t_flat - fd, x_flat)
        tau_t = (tau_tp_all[jnp.arange(m)[:, None], offsets] - tau_tm_all[jnp.arange(m)[:, None], offsets]) / (2*fd)
        xi_t = (xi_tp_all[jnp.arange(m)[:, None], offsets, :] - xi_tm_all[jnp.arange(m)[:, None], offsets, :]) / (2*fd)

        xi_xp_l, xi_xm_l = [], []
        for j in range(N_SPATIAL):
            e_j = jnp.zeros((1, N_SPATIAL), dtype=jnp.float64).at[0, j].set(fd)
            _, xp = eval_gen_jit(pg, t_flat, x_flat + e_j)
            _, xm = eval_gen_jit(pg, t_flat, x_flat - e_j)
            xi_xp_l.append(xp[jnp.arange(m)[:, None], offsets, :])
            xi_xm_l.append(xm[jnp.arange(m)[:, None], offsets, :])
        xi_x = jnp.stack([(xp - xm) / (2*fd) for xp, xm in zip(xi_xp_l, xi_xm_l)], axis=-1)
        xi_xx = jnp.stack([(xp - 2*xi + xm) / (fd**2) for xp, xm in zip(xi_xp_l, xi_xm_l)], axis=-1)
        return tau, xi, tau_t, xi_t, xi_x, xi_xx

    def _loss(pg, t_grid, x_paths):
        """t_grid: (T,), x_paths: (n_traj, T, 11)."""
        nt, Tp1, _ = x_paths.shape
        t_mat = jnp.broadcast_to(t_grid[None, :, None], (nt, Tp1, 1))
        tl = jnp.broadcast_to(t_grid[None, :-1], (nt, Tp1 - 1)).ravel()  # (B_left,)
        xl = x_paths[:, :-1, :].reshape(-1, N_SPATIAL)  # (B_left, 11)
        B_left = tl.shape[0]

        # Initialize flow state
        tau0, _ = eval_gen_jit(pg, tl, xl)
        m_gen = tau0.shape[0]
        ts = jnp.broadcast_to(tl[None, :], (m_gen, B_left))
        xs = jnp.broadcast_to(xl[None, :, :], (m_gen, B_left, N_SPATIAL))
        f0 = mu_fn(tl, xl)   # (B_left, 11)
        s0 = sig_fn(tl, xl)  # (B_left, 11)
        mus = jnp.broadcast_to(f0[None, :, :], (m_gen, B_left, N_SPATIAL))
        sgs = jnp.broadcast_to(jnp.maximum(jnp.abs(s0), SIGMA_FLOOR)[None, :, :], (m_gen, B_left, N_SPATIAL))

        h = jnp.float64(eps) / num_steps

        def one_step(_, state):
            tS, xS, muS, sgS = state
            tau, xi, tau_t, xi_t, xi_x, xi_xx = _flow_derivs(pg, tS, xS)
            # Prolonged mu: dmu^a/deps = xi_t^a + sum_j mu^j * dxi^a/dx_j + 0.5*sum_j sg_j^2 * xi_xx^a_j - mu^a*tau_t
            k1_mu = (xi_t + jnp.einsum('mba,mbaj->mba', muS, xi_x)
                     + 0.5 * jnp.einsum('mba,mbaj->mba', sgS**2, xi_xx)
                     - muS * tau_t[:, :, None])
            # Prolonged sigma (diagonal): dsigma_a/deps = sigma_a * (dxi^a/dx_a - 0.5*tau_t)
            diag_xi_x = jnp.diagonal(xi_x, axis1=2, axis2=3)  # (m, B, 11)
            k1_sg = sgS * (diag_xi_x - 0.5 * tau_t[:, :, None])

            tP = tS + h * tau; xP = xS + h * xi
            muP = muS + h * k1_mu; sgP = jnp.maximum(sgS + h * k1_sg, SIGMA_FLOOR)

            # k2 at predicted point
            tau2, xi2, tau_t2, xi_t2, xi_x2, xi_xx2 = _flow_derivs(pg, tP, xP)
            k2_mu = (xi_t2 + jnp.einsum('mba,mbaj->mba', muP, xi_x2)
                     + 0.5 * jnp.einsum('mba,mbaj->mba', sgP**2, xi_xx2)
                     - muP * tau_t2[:, :, None])
            diag2 = jnp.diagonal(xi_x2, axis1=2, axis2=3)
            k2_sg = sgP * (diag2 - 0.5 * tau_t2[:, :, None])

            return (tS + 0.5 * h * (tau + tau2),
                    xS + 0.5 * h * (xi + xi2),
                    muS + 0.5 * h * (k1_mu + k2_mu),
                    jnp.maximum(sgS + 0.5 * h * (k1_sg + k2_sg), SIGMA_FLOOR))

        tF, xF, muF, sgF = jax.lax.fori_loop(0, num_steps, one_step, (ts, xs, mus, sgs))

        # Compare to surrogate at pushed points
        mu_eval = mu_fn(tF.reshape(-1), xF.reshape(-1, N_SPATIAL)).reshape(m_gen, B_left, N_SPATIAL)
        sg_eval = jnp.maximum(jnp.abs(sig_fn(tF.reshape(-1), xF.reshape(-1, N_SPATIAL)).reshape(
            m_gen, B_left, N_SPATIAL)), SIGMA_FLOOR)

        drift_mse = jnp.mean((muF - mu_eval)**2, axis=(1, 2))  # (m,)
        diff_mse = jnp.mean((sgF - sg_eval)**2, axis=(1, 2))  # (m,)
        coeff_loss = jnp.mean(drift_mse + diff_mse)

        # Domain penalty
        X_MIN_j = jnp.asarray(X_MIN, jnp.float64)
        X_MAX_j = jnp.asarray(X_MAX, jnp.float64)
        dom_pen = (jnp.mean(jax.nn.softplus(tF - T_MAX) + jax.nn.softplus(T_MIN - tF))
                   + jnp.mean(jnp.sum(jax.nn.softplus(xF - X_MAX_j) + jax.nn.softplus(X_MIN_j - xF), axis=-1)))
        return coeff_loss + domain_penalty_w * dom_pen, {"drift_mse": drift_mse, "diff_mse": diff_mse, "domain_pen": dom_pen}
    return jax.jit(_loss)


# ── Magnitude regularization ─────────────────────────────────────────────

def mag_reg(pg, tx):
    t, x = tx[:, 0], tx[:, 1:]
    tau, xi = eval_gen_jit(pg, t, x)
    return jnp.mean(tau**2) + jnp.mean(xi**2), {}


# JIT compile
s1_loss = jax.jit(s1_loss); s2_loss = jax.jit(s2_loss)
s3_loss = jax.jit(s3_loss); s4_loss = jax.jit(s4_loss)
s5_loss = jax.jit(s5_loss); s6_loss = jax.jit(s6_loss)
mag_reg = jax.jit(mag_reg)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: m-SWEEP TRAINING
# ═══════════════════════════════════════════════════════════════════════════

GEN_STEPS = 15000
BATCH_SIZE = 1024

W = {"s1": 0.5, "s2": 0.1, "s3": 0.1, "s4": 0.1,
     "s5": 0.5, "s6": 1.0, "s7": 0.05,
     "mag": 0.01, "wd": 1e-6}

m_values = [1, 2, 3]
sweep_results = {}

for m_try in m_values:
    print(f"\n{'='*60}")
    print(f"Training m={m_try} generators ({GEN_STEPS} steps, 11D, all 7 losses)")
    print(f"{'='*60}")

    key_main, key_gen = jax.random.split(key_main)
    pg = init_generator_params(key_gen, m_try)
    eval_gen_jit = make_eval_generators()
    s7_train = make_s7_loss(eps=1e-2, num_steps=1, domain_penalty_w=10.0)

    def master_loss(pg, tx_batch, key=None):
        total = jnp.float64(0.0); aux = {}
        l1, _ = s1_loss(pg, tx_batch); total = total + W["s1"] * l1; aux["L1"] = l1
        l2, _ = s2_loss(pg, tx_batch); total = total + W["s2"] * l2; aux["L2"] = l2
        l3, _ = s3_loss(pg, tx_batch); total = total + W["s3"] * l3; aux["L3"] = l3
        l4, _ = s4_loss(pg, tx_batch, key=key); total = total + W["s4"] * l4; aux["L4"] = l4
        l5, _ = s5_loss(pg, tx_batch); total = total + W["s5"] * l5; aux["L5"] = l5
        l6, _ = s6_loss(pg, tx_batch); total = total + W["s6"] * l6; aux["L6"] = l6
        l7, a7 = s7_train(pg, t_s7, x_s7); total = total + W["s7"] * l7; aux["L7"] = l7
        if "domain_pen" in a7: aux["dom"] = a7["domain_pen"]
        mr, _ = mag_reg(pg, tx_batch); total = total + W["mag"] * mr; aux["mag"] = mr
        wd = W["wd"] * l2_tree(pg)
        return total + wd, aux

    gen_sched = optax.warmup_cosine_decay_schedule(
        init_value=1e-6, peak_value=3e-4, warmup_steps=300,
        decay_steps=GEN_STEPS, end_value=1e-5)
    opt_gen = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(gen_sched))
    opt_state = opt_gen.init(pg)

    @jax.jit
    def train_step(pg, opt_state, tx_batch, key):
        (lv, aux), grads = jax.value_and_grad(master_loss, has_aux=True)(pg, tx_batch, key)
        updates, new_state = opt_gen.update(grads, opt_state, pg)
        return optax.apply_updates(pg, updates), new_state, lv, aux

    rng_gen = np.random.default_rng(42 + m_try)
    key_main, key_train = jax.random.split(key_main)
    l1_hist = []
    t0_m = time.time()

    for step in range(1, GEN_STEPS + 1):
        idx = rng_gen.choice(TX_gen_np.shape[0], size=min(BATCH_SIZE, TX_gen_np.shape[0]), replace=False)
        tx_batch = jnp.asarray(TX_gen_np[idx], dtype=jnp.float64)
        key_train, key_step = jax.random.split(key_train)
        pg, opt_state, lv, aux = train_step(pg, opt_state, tx_batch, key_step)
        l1_hist.append(float(aux["L1"]))
        if step % 3000 == 0 or step == 1:
            parts = [f"step {step:5d}", f"tot={float(lv):.3e}", f"L1={float(aux['L1']):.2e}",
                     f"L6={float(aux['L6']):.2e}", f"mag={float(aux['mag']):.2e}"]
            if "dom" in aux: parts.append(f"dom={float(aux['dom']):.2e}")
            print("  " + " | ".join(parts))

    elapsed = time.time() - t0_m
    final_L1 = np.mean(l1_hist[-500:]) if len(l1_hist) >= 500 else np.mean(l1_hist)
    print(f"  m={m_try}: final L1 = {final_L1:.4e}  ({elapsed:.0f}s)")
    sweep_results[m_try] = {"params_gen": pg, "final_L1": final_L1, "l1_hist": l1_hist}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: m-SWEEP SUMMARY + EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("m-SWEEP SUMMARY (11D, all 7 losses)")
print(f"{'='*60}")
l1_vals = []
for mt in m_values:
    fl = sweep_results[mt]["final_L1"]
    l1_vals.append(fl)
    print(f"  m={mt}: L1 = {fl:.4e}")

l1_nontrivial = {m: l for m, l in zip(m_values, l1_vals) if m >= 2}
m_star = min(l1_nontrivial, key=l1_nontrivial.get)
print(f"\n  >>> m* = {m_star} <<<")

# m-sweep plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(m_values, l1_vals, 'bo-', markersize=10, lw=2)
axes[0].axvline(m_star, color='red', ls='--', alpha=0.7, label=f'm* = {m_star}')
axes[0].set_xlabel('m'); axes[0].set_ylabel('L1'); axes[0].set_title('Lie Algebra Dimension Selection (11D)')
axes[0].set_xticks(m_values); axes[0].legend(); axes[0].grid(True, alpha=0.3)
for mt in m_values:
    axes[1].plot(sweep_results[mt]['l1_hist'], label=f'm={mt}', alpha=0.8)
axes[1].set_xlabel('Step'); axes[1].set_ylabel('L1'); axes[1].set_title('Closure Loss Curves')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('stock_11d_m_sweep.png', dpi=150); plt.close()
print("Saved: stock_11d_m_sweep.png")

# After-push evaluation
params_best = sweep_results[m_star]["params_gen"]
eval_gen_jit = make_eval_generators()

eps_values = [0.5, 1.0, 2.0]
results = {}

print(f"\n{'='*60}")
print(f"AFTER-PUSH RESIDUALS (m*={m_star}, 11D)")
print(f"{'='*60}")
for eps_val in eps_values:
    s7_ev = make_s7_loss(eps=eps_val, num_steps=10, domain_penalty_w=0.0)
    _, aux_e = s7_ev(params_best, t_s7, x_s7)
    dm, sm = np.array(aux_e["drift_mse"]), np.array(aux_e["diff_mse"])
    results[eps_val] = {"drift": dm, "diff": sm}
    print(f"\n  eps = {eps_val}:")
    for i in range(m_star):
        print(f"    Gen {i+1}: Drift MSE = {dm[i]:.4e}  |  Diff MSE = {sm[i]:.4e}")

# After-push bar plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
xp = np.arange(m_star); w = 0.25
for ie, ev in enumerate(eps_values):
    d = results[ev]
    axes[0].bar(xp + ie * w, d["drift"], w, label=f"eps={ev}")
    axes[1].bar(xp + ie * w, d["diff"], w, label=f"eps={ev}")
for ax, title in zip(axes, ["Drift", "Diffusion"]):
    ax.set_xlabel("Generator"); ax.set_ylabel("MSE"); ax.set_title(f"After-Push {title} (11D)")
    ax.set_xticks(xp + w); ax.set_xticklabels([f"Gen {i+1}" for i in range(m_star)])
    ax.legend(); ax.set_yscale("log")
plt.tight_layout(); plt.savefig('stock_11d_after_push.png', dpi=150); plt.close()
print("\nSaved: stock_11d_after_push.png")

# tau curves
t_line = jnp.linspace(T_MIN, T_MAX, 200)
x_mid = jnp.broadcast_to(jnp.asarray(log_prices.mean(axis=0), jnp.float64)[None, :], (200, N_SPATIAL))
tau_vals, _ = eval_gen_jit(params_best, t_line, x_mid)
fig, axes = plt.subplots(1, m_star, figsize=(6 * m_star, 4.5))
if m_star == 1: axes = [axes]
for i in range(m_star):
    axes[i].plot(np.array(t_line), np.array(tau_vals[i]), lw=2, color='steelblue')
    axes[i].set_xlabel('t (years)'); axes[i].set_ylabel(f'tau_{i+1}(t)')
    axes[i].set_title(f'Generator {i+1}: tau(t)'); axes[i].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('stock_11d_tau_curves.png', dpi=150); plt.close()
print("Saved: stock_11d_tau_curves.png")

# Save results
save_dict = {
    "m_star": np.array(m_star), "m_values": np.array(m_values),
    "l1_by_m": np.array(l1_vals), "tickers": np.array(TICKERS),
    "n_spatial": np.array(N_SPATIAL),
}
for ev in eps_values:
    save_dict[f"drift_mse_eps{ev}"] = results[ev]["drift"]
    save_dict[f"diff_mse_eps{ev}"] = results[ev]["diff"]
np.savez("stock_11d_results.npz", **save_dict)
print("\nSaved: stock_11d_results.npz")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Dimension: {N_SPATIAL}D (stocks: {', '.join(TICKERS)})")
print(f"  Losses: L1-L7 + domain penalty + magnitude reg")
print(f"  m* = {m_star}")
print(f"  Final L1 = {sweep_results[m_star]['final_L1']:.4e}")
print(f"  After-push (eps=1.0):")
for i in range(m_star):
    print(f"    Gen {i+1}: Drift={results[1.0]['drift'][i]:.4e}, Diff={results[1.0]['diff'][i]:.4e}")
print(f"\nAll done!")
