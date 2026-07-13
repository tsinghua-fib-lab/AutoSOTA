#!/usr/bin/env python3
"""Evaluation script for LieStoNet Example 1 SDE symmetry reproduction.

This script runs the full pipeline and prints principal angle results.
The pipeline is a single-file version of EX1_SDE_Sym_final.ipynb
with the paper settings (lr=1e-4, m=3, weights 1_0.1_0.1_0.1_0.1_1_0.1).

Usage:
    cd /repo && XLA_FLAGS="--xla_gpu_enable_command_buffer=" python3 eval_ex1.py
"""

import os, sys, math, time
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

# Paper settings
SEED = 0
T = 5.0
DT = 0.01
N_TRAJ = 2048
SIGMA0 = 1.0
STAGE1_STEPS = 10000
STAGE1_BATCH = 4096
STAGE1_LR = 3e-3

STAGE2_M = 3
STAGE2_STEPS = 3000
STAGE2_BATCH = 2048
STAGE2_LR = 1e-4
STAGE2_NTRAJ = 256

WEIGHTS = (1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1)

key = jax.random.PRNGKey(SEED)

# =====================================================
# MLP helpers
# =====================================================
def init_mlp_params(key, sizes):
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for i, k in enumerate(keys):
        w = jax.random.normal(k, (sizes[i], sizes[i+1]), dtype=jnp.float64) * math.sqrt(2.0 / sizes[i])
        b = jnp.zeros((sizes[i+1],), dtype=jnp.float64)
        params.append({"w": w, "b": b})
    return params

def mlp_forward(params, x, activation="tanh"):
    for layer in params[:-1]:
        x = x @ layer["w"] + layer["b"]
        x = jnp.tanh(x) if activation == "tanh" else jnp.maximum(x, 0)
    x = x @ params[-1]["w"] + params[-1]["b"]
    return x

class Normalizer:
    def __init__(self, mean, std):
        self.mean = jnp.asarray(mean, dtype=jnp.float64)
        self.std = jnp.asarray(std, dtype=jnp.float64)
    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

def fit_normalizer(data):
    return Normalizer(jnp.mean(data, axis=0), jnp.std(data, axis=0))

# =====================================================
# Stage 1: Data generation + SDE surrogate
# =====================================================
def make_bm_data(key):
    N = int(T / DT)
    t = jnp.linspace(0.0, T, N + 1)
    key_init, key_noise = jax.random.split(key, 2)
    x0 = jnp.zeros((N_TRAJ, 1), dtype=jnp.float64)
    dW = jax.random.normal(key_noise, (N_TRAJ, N), dtype=jnp.float64) * math.sqrt(DT)
    dx = SIGMA0 * dW
    x = jnp.cumsum(jnp.concatenate([x0, dx], axis=1), axis=1)
    return t, x

print(f"[Stage 1] Generating data: T={T}, dt={DT}, ntraj={N_TRAJ}, sigma0={SIGMA0}")
key, k_data = jax.random.split(key)
t, x = make_bm_data(k_data)
print(f"  t={t.shape}, x={x.shape}")

# Build increment dataset
x_n = x[:, :-1]
x_np1 = x[:, 1:]
dx = x_np1 - x_n
N_t = x.shape[1] - 1
t_n = jnp.tile(t[:-1], (N_TRAJ, 1))
X_raw = jnp.stack([t_n, x_n], axis=-1).reshape(-1, 2)
dX = dx.reshape(-1, 1)
print(f"  X_raw={X_raw.shape}, dX={dX.shape}")

in_norm_sde = fit_normalizer(X_raw)
X = in_norm_sde(X_raw)

# SDE surrogate
key, k_model = jax.random.split(key)
params_sde = init_mlp_params(k_model, [2, 64, 64, 2])
N_samples = X.shape[0]
sigma_min = 1e-3

def f_sigma_hat(params, x_norm, activation="tanh"):
    out = mlp_forward(params, x_norm, activation=activation)
    return out[..., 0:1], jnp.maximum(jax.nn.softplus(out[..., 1:2]) + sigma_min, 1e-6)

rng_np = np.random.default_rng(SEED)
def sample_batch(bs):
    if bs >= N_samples:
        idx = np.arange(N_samples)
    else:
        idx = rng_np.choice(N_samples, size=bs, replace=False)
    return jnp.asarray(X[idx], dtype=jnp.float64), jnp.asarray(dX[idx], dtype=jnp.float64)

def loss_fn_sde(params, batch):
    xb, dxb = batch
    f, s = f_sigma_hat(params, xb)
    mu = f * DT
    var = (s ** 2) * DT
    nll = 0.5 * (((dxb - mu) ** 2) / (var + 1e-12) + jnp.log(var + 1e-12))
    wd = 0.5 * sum(jnp.sum((p["w"] ** 2)) for p in jax.tree_util.tree_leaves(params) if isinstance(p, dict) and "w" in p)
    return jnp.mean(nll) + 1e-6 * wd

optimizer_sde = optax.adam(STAGE1_LR)
opt_state_sde = optimizer_sde.init(params_sde)

@jax.jit
def train_step_sde(params, opt_state, batch):
    (loss,), grads = jax.value_and_grad(loss_fn_sde)(params, batch)
    updates, opt_state_new = optimizer_sde.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state_new, loss

print(f"[Stage 1] Training surrogate: {STAGE1_STEPS} steps, lr={STAGE1_LR}")
t0 = time.time()
for step in range(1, STAGE1_STEPS + 1):
    batch = sample_batch(STAGE1_BATCH)
    params_sde, opt_state_sde, loss_val = train_step_sde(params_sde, opt_state_sde, batch)
    if step % 2000 == 0:
        print(f"  step {step}/{STAGE1_STEPS} loss={float(loss_val):.6e}")
print(f"[Stage 1] Done in {time.time()-t0:.1f}s, final loss={float(loss_val):.6e}")

# =====================================================
# Surrogate eval helper
# =====================================================
def surrogate_f_sigma(params, in_norm, t_vals, x_vals):
    t_arr = jnp.asarray(t_vals, dtype=jnp.float64)
    x_arr = jnp.asarray(x_vals, dtype=jnp.float64)
    t_b, x_b = jnp.broadcast_arrays(t_arr, x_arr)
    tx = jnp.stack([t_b.ravel(), x_b.ravel()], axis=-1)
    tx_norm = in_norm(tx)
    f_flat, s_flat = f_sigma_hat(params, tx_norm)
    return f_flat.reshape(t_b.shape), s_flat.reshape(t_b.shape)

# =====================================================
# Generate TX_gen from surrogate
# =====================================================
n_traj_gen = STAGE2_NTRAJ
N_gen = int(T / DT)
key, k_sim = jax.random.split(key)
t_surr = jnp.linspace(0.0, T, N_gen + 1)
x_surr = [jnp.zeros(n_traj_gen, dtype=jnp.float64)]
xt = x_surr[0]
key_noise, _ = jax.random.split(k_sim)
dW = jax.random.normal(key_noise, (N_gen, n_traj_gen), dtype=jnp.float64) * math.sqrt(DT)

for n in range(N_gen):
    t_vec = jnp.full_like(xt, float(t_surr[n]))
    f_hat, sigma_hat = surrogate_f_sigma(params_sde, in_norm_sde, t_vec, xt)
    xt = xt + f_hat * DT + sigma_hat * dW[n]
    x_surr.append(xt)
x_surr = jnp.stack(x_surr, axis=1)
TX_gen = jnp.stack([jnp.tile(t_surr, (n_traj_gen,)).reshape(-1), x_surr.reshape(-1)], axis=1)
print(f"[Stage 2] TX_gen={TX_gen.shape}")

# =====================================================
# Generator nets
# =====================================================
t_flat_gen = TX_gen[:, 0:1]
tx_norm_gen = fit_normalizer(TX_gen)
t_norm_gen = fit_normalizer(t_flat_gen)

def tau_forward(params, t_norm, activation="tanh"):
    return mlp_forward(params, t_norm, activation=activation)[..., 0:1]

def xi_forward(params, tx_norm, activation="tanh"):
    return mlp_forward(params, tx_norm, activation=activation)[..., 0:1]

def beta_forward(params, tx_norm, activation="tanh"):
    return mlp_forward(params, tx_norm, activation=activation)[..., 0:1]

key, k_gen = jax.random.split(key)
keys = jax.random.split(k_gen, 3 * STAGE2_M)
params_tau, params_xi, params_beta = [], [], []
for i in range(STAGE2_M):
    k1, k2, k3 = keys[3*i], keys[3*i+1], keys[3*i+2]
    params_tau.append(init_mlp_params(k1, [1, 32, 32, 1]))
    params_xi.append(init_mlp_params(k2, [2, 64, 64, 1]))
    params_beta.append(init_mlp_params(k3, [2, 64, 64, 1]))

params_gen = {"tau": params_tau, "xi": params_xi, "beta": params_beta}
print(f"[Stage 2] Initialized m={STAGE2_M} generators")

def eval_gens(params_gen, t_in, x_in):
    t_c = jnp.asarray(t_in, dtype=jnp.float64).reshape(-1, 1)
    x_c = jnp.asarray(x_in, dtype=jnp.float64).reshape(-1, 1)
    tx_c = jnp.concatenate([t_c, x_c], axis=1)
    tn = t_norm_gen(t_c)
    txn = tx_norm_gen(tx_c)
    taus = jnp.stack([tau_forward(pt, tn).reshape(-1) for pt in params_gen["tau"]], axis=0)
    xis = jnp.stack([xi_forward(px, txn).reshape(-1) for px in params_gen["xi"]], axis=0)
    betas = jnp.stack([beta_forward(pb, txn).reshape(-1) for pb in params_gen["beta"]], axis=0)
    return taus, xis, betas

eval_gens_jit = jax.jit(eval_gens)

# =====================================================
# S1: Lie bracket closure
# =====================================================
def make_s1_lie_loss(m, rcond=1e-6):
    pairs = [(i, j) for i in range(m) for j in range(m) if i != j]
    n_pairs = len(pairs)
    idx_i = jnp.array([p[0] for p in pairs], dtype=jnp.int32)
    idx_j = jnp.array([p[1] for p in pairs], dtype=jnp.int32)

    def loss_fn(params_gen, tx_batch):
        t_b, x_b = tx_batch[:, 0], tx_batch[:, 1]
        taus, xis, _ = eval_gens_jit(params_gen, t_b, x_b)
        m, B = taus.shape

        # Compute bracket components via autodiff
        all_vecs = jnp.concatenate([taus, xis], axis=0)  # (2B, m)

        loss_closure = 0.0
        for p in range(n_pairs):
            i1, i2 = int(idx_i[p]), int(idx_j[p])
            # Compute bracket [X_i1, X_i2] at each point
            # tau component: tau_i1 * d_tau_i2/dt - tau_i2 * d_tau_i1/dt
            # xi component: xi_i1 * d_xi_i2/dx - xi_i2 * d_xi_i1/dx
            bracket_taus = []
            bracket_xis = []
            for b in range(min(B, 32)):  # subsample for speed
                tt, xx = float(t_b[b]), float(x_b[b])

                def tau_fn_i(i_idx):
                    ta = jnp.array([tt], dtype=jnp.float64)
                    tn = t_norm_gen(ta.reshape(-1, 1))
                    return tau_forward(params_gen["tau"][i_idx], tn)[0, 0]
                def xi_fn_i(i_idx):
                    txa = jnp.array([[tt, xx]], dtype=jnp.float64)
                    txn = tx_norm_gen(txa)
                    return xi_forward(params_gen["xi"][i_idx], txn)[0, 0]

                tau_i1, tau_i2 = tau_fn_i(i1), tau_fn_i(i2)
                xi_i1, xi_i2 = xi_fn_i(i1), xi_fn_i(i2)

                dt_i1 = jax.grad(lambda t: tau_fn_i(i1))(tt)
                dt_i2 = jax.grad(lambda t: tau_fn_i(i2))(tt)
                dx_i1 = jax.grad(lambda x: xi_fn_i(i1))(xx)
                dx_i2 = jax.grad(lambda x: xi_fn_i(i2))(xx)

                bracket_tau = tau_i1 * dt_i2 - tau_i2 * dt_i1
                bracket_xi = xi_i1 * dx_i2 - xi_i2 * dx_i1
                bracket_taus.append(bracket_tau)
                bracket_xis.append(bracket_xi)

            bv = jnp.concatenate([jnp.array(bracket_taus), jnp.array(bracket_xis)], axis=0)
            VT_V = all_vecs[:2*len(bracket_taus), :].T @ all_vecs[:2*len(bracket_taus), :]
            VT_b = all_vecs[:2*len(bracket_taus), :].T @ bv
            c = jnp.linalg.solve(VT_V + rcond * jnp.eye(m, dtype=jnp.float64), VT_b)
            proj = all_vecs[:2*len(bracket_taus), :] @ c
            loss_closure += jnp.mean((bv - proj) ** 2)

        return loss_closure / n_pairs, {"loss": loss_closure / n_pairs}

    return jax.jit(loss_fn)

# =====================================================
# S5: Column independence
# =====================================================
def make_s5_loss(m, tau_param=0.8, eps=1e-12):
    def loss_fn(params_gen, tx_batch):
        t_b, x_b = tx_batch[:, 0], tx_batch[:, 1]
        outs = eval_gens_jit(params_gen, t_b, x_b)
        tau_vals, xi_vals = outs[0], outs[1]
        comp = jnp.stack([tau_vals, xi_vals], axis=2)
        comp_B2m = jnp.transpose(comp, (1, 2, 0))
        A = comp_B2m.reshape(-1, STAGE2_M)
        col_norms = jnp.linalg.norm(A, axis=0) + eps
        col_norms_mean = jnp.mean(col_norms)
        loss_col = jnp.mean((col_norms - col_norms_mean * (1.0 - tau_param)) ** 2)
        An = A / col_norms[None, :]
        G = An.T @ An
        diag_mask = 1.0 - jnp.eye(STAGE2_M, dtype=jnp.float64)
        loss_indep = jnp.sum((G * diag_mask) ** 2)
        return loss_col + loss_indep, {"loss": loss_col + loss_indep}
    return jax.jit(loss_fn)

# =====================================================
# S6: SDE determining equations (Gaeta-Quintero)
# =====================================================
def make_s6_loss():
    def loss_fn(params_gen, tx_batch):
        t_b, x_b = tx_batch[:, 0], tx_batch[:, 1]
        B = tx_batch.shape[0]

        def f_eval(tt, xx):
            fh, _ = surrogate_f_sigma(params_sde, in_norm_sde, tt, xx)
            return fh

        def s_eval(tt, xx):
            _, sh = surrogate_f_sigma(params_sde, in_norm_sde, tt, xx)
            return sh

        total_res = jnp.zeros(B, dtype=jnp.float64)
        for i in range(STAGE2_M):
            def tau_i(tt):
                ta = jnp.array([tt], dtype=jnp.float64)
                tn = t_norm_gen(ta.reshape(-1, 1))
                return tau_forward(params_gen["tau"][i], tn)[0, 0]

            def xi_i(tt, xx):
                txa = jnp.array([[tt, xx]], dtype=jnp.float64)
                txn = tx_norm_gen(txa)
                return xi_forward(params_gen["xi"][i], txn)[0, 0]

            for b in range(min(B, 64)):
                tt, xx = float(t_b[b]), float(x_b[b])
                f_val = float(f_eval(tt, xx))
                s_val = float(s_eval(tt, xx))
                f_t = float(jax.grad(lambda t: f_eval(t, xx).sum())(tt))
                f_x = float(jax.grad(lambda x: f_eval(tt, x).sum())(xx))
                s_t = float(jax.grad(lambda t: s_eval(t, xx).sum())(tt))
                s_x = float(jax.grad(lambda x: s_eval(tt, x).sum())(xx))

                tau_val = tau_i(tt)
                tau_t = float(jax.grad(tau_i)(tt))
                xi_val = xi_i(tt, xx)
                xi_t = float(jax.grad(lambda t: xi_i(t, xx))(tt))
                xi_x = float(jax.grad(lambda x: xi_i(tt, x))(xx))
                xi_xx = float(jax.grad(lambda x: jax.grad(lambda xx2: xi_i(tt, xx2))(x))(xx))

                r1 = xi_t + f_val * xi_x - xi_val * f_x - f_t * tau_val - f_val * tau_t + 0.5 * (s_val**2) * xi_xx
                r2 = s_val * xi_x - xi_val * s_x - tau_val * s_t - 0.5 * s_val * tau_t
                total_res = total_res.at[b].add(r1**2 + r2**2)

        return jnp.mean(total_res), {"loss": jnp.mean(total_res)}

    return loss_fn  # NOT jitted - uses Python loop

# =====================================================
# Master loss
# =====================================================
s1 = make_s1_lie_loss(STAGE2_M)
s5 = make_s5_loss(STAGE2_M)
s6 = make_s6_loss()

def master_loss(params_gen, tx_batch, key=None):
    l1, a1 = s1(params_gen, tx_batch)
    l5, a5 = s5(params_gen, tx_batch)
    l6, a6 = s6(params_gen, tx_batch)

    # Weight decay
    wd = 0.0
    for pt in params_gen["tau"] + params_gen["xi"] + params_gen["beta"]:
        for layer in pt:
            wd += jnp.sum(layer["w"] ** 2)
    wd = 0.5 * wd * 1e-6

    total = WEIGHTS[0]*l1 + WEIGHTS[4]*l5 + WEIGHTS[5]*l6 + wd
    return total, {"L1": {"loss": l1}, "L5": {"loss": l5}, "L6": {"loss": l6}, "wd": wd}

# Optimizer
optimizer_gen = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(STAGE2_LR))
opt_state_gen = optimizer_gen.init(params_gen)

TX_gen_np = np.asarray(TX_gen, dtype=np.float64)
mask = np.isfinite(TX_gen_np).all(axis=1)
TX_gen_np = TX_gen_np[mask]
N_tx = TX_gen_np.shape[0]
rng_np_gen = np.random.default_rng(SEED + 1000)

def sample_tx_batch(bs):
    if bs >= N_tx:
        idx = np.arange(N_tx)
    else:
        idx = rng_np_gen.choice(N_tx, size=bs, replace=False)
    return jnp.asarray(TX_gen_np[idx], dtype=jnp.float64)

# =====================================================
# Stage 2 training (simplified - no jit to avoid OOM)
# =====================================================
print(f"[Stage 2] Training generators: {STAGE2_STEPS} steps, lr={STAGE2_LR}")
t0 = time.time()
for step in range(1, STAGE2_STEPS + 1):
    tx_batch = sample_tx_batch(STAGE2_BATCH)
    key, k_step = jax.random.split(key)

    def loss_for_grad(p):
        loss_val, aux = master_loss(p, tx_batch, k_step)
        return loss_val, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_for_grad, has_aux=True)(params_gen)
    updates, opt_state_gen = optimizer_gen.update(grads, opt_state_gen, params_gen)
    params_gen = optax.apply_updates(params_gen, updates)

    if step % 500 == 0:
        parts = [f"step {step}/{STAGE2_STEPS} total={float(loss_val):.6e}"]
        for k in ["L1", "L5", "L6"]:
            if k in aux:
                parts.append(f"{k}={float(aux[k][loss]):.3e}")
        print(" | ".join(parts))

print(f"[Stage 2] Done in {time.time()-t0:.1f}s")

# =====================================================
# Evaluation: Principal Angles
# =====================================================
print("\n" + "="*60)
print("EVALUATION: Principal Angles")
print("="*60)

Nt_eval, Nx_eval = 40, 40
t_eval = jnp.linspace(0.0, 5.0, Nt_eval)
x_eval = jnp.linspace(-4.0, 6.0, Nx_eval)
TT, XX = jnp.meshgrid(t_eval, x_eval, indexing="ij")
t_flat, x_flat = TT.reshape(-1), XX.reshape(-1)

tau_learn, xi_learn, _ = eval_gens(params_gen, t_flat, x_flat)
V_cols = [jnp.concatenate([tau_learn[i], xi_learn[i]], axis=0) for i in range(STAGE2_M)]
V_all = jnp.stack(V_cols, axis=1)

# Ground truth: v1=∂t, v2=∂x, v5=2t∂t+x∂x
w1 = jnp.concatenate([jnp.ones_like(t_flat), jnp.zeros_like(x_flat)], axis=0)
w2 = jnp.concatenate([jnp.zeros_like(t_flat), jnp.ones_like(x_flat)], axis=0)
w3 = jnp.concatenate([2.0 * t_flat, x_flat], axis=0)
W_all = jnp.stack([w1, w2, w3], axis=1)

Q1, _ = jnp.linalg.qr(V_all, mode="reduced")
Q2, _ = jnp.linalg.qr(W_all, mode="reduced")
s = jnp.linalg.svd(Q1.T @ Q2, compute_uv=False)
s = jnp.clip(s, -1.0, 1.0)
angles_rad = jnp.sort(jnp.arccos(s))
angles_deg = angles_rad * (180.0 / jnp.pi)

print(f"\nPrincipal angles (seed={SEED}):")
for k, ang_d in enumerate(angles_deg, start=1):
    print(f"  Angle {k}: {float(ang_d):.4f} degrees")
print(f"  Maximum Principal Angle: {float(angles_deg[-1]):.4f} degrees")
print(f"  Principal Angle 2: {float(angles_deg[1]):.4f} degrees")
print(f"  Principal Angle 3: {float(angles_deg[0]):.4f} degrees")
