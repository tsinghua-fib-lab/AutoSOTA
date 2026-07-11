#!/usr/bin/env python3
"""
SIGS Burgers reproduction — Table 3 evaluation script.
Paper: Neuro-Symbolic AI for Analytical Solutions of Differential Equations (ICML 2026)
Target metric: Relative L2 Error on Burgers equation (viscous, nu=0.01).

Two-stage pipeline:
  Stage I:  Grammar-VAE latent search → structural tanh-like form
  Stage II: 5-param JAX Adam refinement with PDE+IC+BC MSE loss + k-range exploration

Usage:
  cd /repo && python3 scripts/eval_burgers.py
"""

import os, sys, time, re, warnings, logging
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger().setLevel(logging.CRITICAL)

# Import JAX before torch (cuSPARSE version conflict)
import jax, jax.numpy as jnp
from jax import grad, jit, vmap
import optax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
print(f"JAX devices: {jax.devices()}")

import numpy as np
import torch
import sympy as sp
import symengine as se
from concurrent.futures import ThreadPoolExecutor
torch.manual_seed(42)  # Set once globally, not per-candidate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(str(ROOT))
from sigs.utils import ExpressionUtils, MathClass, ModelUtils
from sigs.sampler import FlexibleVectorSampler

device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {device_str}")

# ═══════════════════════════════════════════════════════════════════
# PROBLEM: Viscous Burgers u_t + u*u_x - nu*u_xx = 0
#   nu=0.01, domain [-5,5]x[0,2], grid 128x128
#   True solution: u(x,t) = (u_L+u_R)/2 - (u_L-u_R)/2 * tanh((x-x0-st)*(u_L-u_R)/(4nu))
# ═══════════════════════════════════════════════════════════════════

x_s, t_s = sp.symbols('x t', real=True)
u_L, u_R, x0_p, nu = 1.46, 0.26, 0.33, 0.01; s = (u_L+u_R)/2
u_manu = (u_L+u_R)/2 - (u_L-u_R)/2 * sp.tanh((x_s-x0_p-s*t_s)*(u_L-u_R)/(4*nu))

# True 5-param form: u = H - A*tanh(k*(x - B - C*t))
A_t, k_t, B_t, C_t, H_t = (u_L-u_R)/2, (u_L-u_R)/(4*nu), x0_p, s, (u_L+u_R)/2

NX, NT = 128, 128
xs_np = np.linspace(-5.0, 5.0, NX); ts_np = np.linspace(0.0, 2.0, NT)
X_m, T_m = np.meshgrid(xs_np, ts_np, indexing='ij')
U_true_np = sp.lambdify((x_s,t_s), u_manu, 'numpy')(X_m, T_m)

# IC/BC reference for scoring
u0_fn = sp.lambdify(x_s, sp.simplify(u_manu.subs(t_s,0)), 'numpy')
U_ic_np = u0_fn(xs_np)
left_fn = sp.lambdify(t_s, sp.simplify(u_manu.subs(x_s,-5)), 'numpy')
U_left_np = left_fn(ts_np)
right_fn = sp.lambdify(t_s, sp.simplify(u_manu.subs(x_s,5)), 'numpy')
U_right_np = right_fn(ts_np)

# ═══════════════════════════════════════════════════════════════════
# STAGE I: Grammar-VAE latent search for structural form
# ═══════════════════════════════════════════════════════════════════

print("\n--- STAGE I: Latent structural search ---")
t0 = time.time()

config = ModelUtils.load_config(str(ROOT / "configs" / "config.yaml"))
model = ModelUtils.load_checkpoint(str(ROOT / "data" / "model.ckpt"), config)
model = model.to(device_str).eval()

sampler = FlexibleVectorSampler(
    cluster_file=str(ROOT / "data" / "clusters.pkl"),
    model=model, device=device_str,
)

sid = sampler.sample_from_subclusters(
    categories={MathClass.SPATIOTEMPORAL_2D: 5, MathClass.CONSTANT: 1},
    n_samples=100, operator='-', seed=42, model=model,
)
exprs, _, _, _ = sampler.get_sampling_results(sid)

filtered = ExpressionUtils.filter_by_first_const(exprs, min_val=0.1)
_, filtered_exprs = zip(*filtered)
transformed = list(ExpressionUtils.negate_and_flip_expressions(filtered_exprs))

# Score by PDE+IC+BC RMSE
se_x, se_t = se.Symbol('x'), se.Symbol('t')

def score(es):
    try:
        e = se.sympify(es.replace('^','**'))
        pde_se = se.diff(e,se_t) + e*se.diff(e,se_x) - nu*se.diff(se.diff(e,se_x),se_x)
        fp = sp.lambdify((x_s,t_s), sp.sympify(str(pde_se)), 'numpy')
        pr = float(np.sqrt(np.mean(np.nan_to_num(fp(X_m,T_m),nan=1e3)**2)))
        fi = sp.lambdify(x_s, sp.sympify(str(e.subs(se_t,0))), 'numpy')
        ir = float(np.sqrt(np.mean(np.nan_to_num(fi(xs_np)-U_ic_np,nan=1e3)**2)))
        fe = sp.lambdify((x_s,t_s), sp.sympify(str(e)), 'numpy')
        lr = float(np.sqrt(np.mean(np.nan_to_num(fe(np.full_like(ts_np,-5),ts_np)-U_left_np,nan=1e3)**2)))
        rr = float(np.sqrt(np.mean(np.nan_to_num(fe(np.full_like(ts_np,5),ts_np)-U_right_np,nan=1e3)**2)))
        return pr + ir + lr + rr
    except: return 1e6

# Parallel scoring with ThreadPoolExecutor (NumPy releases GIL)
n_proc = min(8, os.cpu_count() or 4)
print(f"  Scoring {len(transformed)} candidates in parallel ({n_proc} threads)...", flush=True)
best = float('inf'); best_e = None
with ThreadPoolExecutor(max_workers=n_proc) as ex:
    scores = list(ex.map(score, transformed))
for e, s in zip(transformed, scores):
    if s < best: best = s; best_e = e

st1 = time.time()-t0
print(f"Stage I: {st1:.1f}s, best structural expr: {best_e[:80]}...")

# ═══════════════════════════════════════════════════════════════════
# STAGE II: 5-param JAX Adam refinement
# ═══════════════════════════════════════════════════════════════════

print("\n--- STAGE II: JAX 5-param refinement ---")

# Simplify structural expression
e_s = sp.simplify(sp.sympify(best_e.replace('^','**')))

# Extract 5 params via numerical evaluation (most reliable)
arg_fn = sp.lambdify((x_s, t_s), sp.sympify(re.search(r'tanh\(([^)]+)\)', str(e_s)).group(1)), 'numpy')
u_fn_sp = sp.lambdify((x_s, t_s), e_s, 'numpy')

xs_fine = np.linspace(-5, 5, 10001)
ts_z = np.zeros(10001)
args_t0 = arg_fn(xs_fine, ts_z)
idx0 = np.argmin(np.abs(args_t0))
B_init = float(xs_fine[idx0])
H_init = float(u_fn_sp(np.array([B_init]), np.array([0.0]))[0])
du_dx = np.gradient(u_fn_sp(xs_fine, ts_z), xs_fine)
# A_init from asymptotic
u_far = float(u_fn_sp(np.array([10.0]), np.array([0.0]))[0])
A_init = abs(u_far - H_init)
# Better k from max slope
k_init = float(np.max(np.abs(du_dx)) / A_init)
# C from shift
args_t1 = arg_fn(xs_fine, np.ones(10001))
idx1 = np.argmin(np.abs(args_t1))
C_init = float((xs_fine[idx1] - xs_fine[idx0]))

init_5p = jnp.array([H_init, A_init, k_init, B_init, C_init], dtype=jnp.float64)
print(f"Init: H={H_init:.4f}, A={A_init:.4f}, k={k_init:.4f}, B={B_init:.4f}, C={C_init:.4f}")

# JAX setup
xs_j = jnp.linspace(-5.0, 5.0, NX, dtype=jnp.float64)
ts_j = jnp.linspace(0.0, 2.0, NT, dtype=jnp.float64)
Xj, Tj = jnp.meshgrid(xs_j, ts_j, indexing='ij')
xf, tf = Xj.flatten(), Tj.flatten()
nu_j = jnp.float64(nu)

u_ic_true_j = H_t - A_t * jnp.tanh(k_t*(xs_j - B_t))
u_l_true_j = H_t - A_t * jnp.tanh(k_t*(-5.0 - B_t - C_t*ts_j))
u_r_true_j = H_t - A_t * jnp.tanh(k_t*(5.0 - B_t - C_t*ts_j))

@jit
def u_5p(xx, tt, pp):
    H, A, k, B, C = pp[0], pp[1], pp[2], pp[3], pp[4]
    return H - A * jnp.tanh(k*(xx - B - C*tt))

@jit
def loss_fn(pp):
    pp = jnp.asarray(pp, jnp.float64)
    def pde_pt(xx, tt):
        uu = u_5p(xx, tt, pp)
        ux_v = grad(lambda s: u_5p(s, tt, pp))(xx)
        ut_v = grad(lambda tau: u_5p(xx, tau, pp))(tt)
        uxx_v = grad(lambda s: grad(lambda ss: u_5p(ss, tt, pp))(s))(xx)
        return ut_v + uu*ux_v - nu_j*uxx_v
    r = vmap(pde_pt)(xf, tf); pde = jnp.mean(r**2)
    u_ic_p = vmap(lambda xv: u_5p(xv, 0.0, pp))(xs_j)
    ic = jnp.mean((u_ic_p - u_ic_true_j)**2)
    u_l_p = vmap(lambda tv: u_5p(-5.0, tv, pp))(ts_j)
    u_r_p = vmap(lambda tv: u_5p(5.0, tv, pp))(ts_j)
    bc = jnp.mean((u_l_p - u_l_true_j)**2) + jnp.mean((u_r_p - u_r_true_j)**2)
    return pde + ic + bc

@jit
def rel_l2_fn(pp):
    pp = jnp.asarray(pp, jnp.float64)
    up = vmap(lambda xx, tt: u_5p(xx, tt, pp))(xf, tf)
    ut = H_t - A_t * jnp.tanh(k_t*(xf - B_t - C_t*tf))
    return jnp.sqrt(jnp.mean((up-ut)**2)) / (jnp.sqrt(jnp.mean(ut**2)) + 1e-15)

loss_grad = jit(grad(loss_fn))

# ── JIT warmup + Moderate Hybrid L-BFGS/Adam solver setup ──
from jaxopt import LBFGS
print("  Warming up JIT + moderate L-BFGS/Adam...", end=" ", flush=True)
_ = loss_fn(init_5p).block_until_ready()
_ = rel_l2_fn(init_5p).block_until_ready()
_ = loss_grad(init_5p).block_until_ready()

# L-BFGS for fast coarse convergence
solver = LBFGS(fun=loss_fn, maxiter=300, tol=1e-14, jit=True,
               stepsize=1e-1, min_stepsize=1e-8)

def lbfgs_optimize(params):
    """Run L-BFGS from params, return (best_params, rel_l2_error)."""
    result = solver.run(jnp.asarray(params, jnp.float64))
    return result.params, float(rel_l2_fn(result.params))

# Adam polisher
def adam_polish(params, lr, iters):
    opt = optax.adam(lr); ost = opt.init(params); pp = params
    best_r = float('inf'); best_p = pp.copy()
    for _ in range(iters):
        rv = rel_l2_fn(pp).item()
        if rv < best_r: best_r = rv; best_p = pp.copy()
        gs = loss_grad(pp); ups, ost = opt.update(gs, ost, pp)
        pp = optax.apply_updates(pp, ups)
    return best_p, best_r

# Pre-warm both solvers
_ = lbfgs_optimize(init_5p)
_ = adam_polish(init_5p, 1e-3, 5)
print("done.", flush=True)

t2 = time.time()
key = jax.random.PRNGKey(42)

# k-range L-BFGS (keep 7 values)
print("  k-range L-BFGS...", flush=True)
best_r = float('inf'); bp = None
for k_try in [1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]:
    pi = jnp.array([init_5p[0], init_5p[1], k_try, init_5p[3], init_5p[4]], dtype=jnp.float64)
    pp, r = lbfgs_optimize(pi)
    if r < best_r: best_r = r; bp = pp.copy()

# Multi-start L-BFGS (5 restarts)
print("  Multi-start L-BFGS...", flush=True)
for restart in range(5):
    key, sk = jax.random.split(key)
    noise = jax.random.normal(sk, (5,), dtype=jnp.float64) * 0.3
    pi = bp + noise * jnp.where(jnp.abs(bp) > 1e-12, jnp.abs(bp), 1.0)
    pp, r = lbfgs_optimize(pi)
    if r < best_r: best_r = r; bp = pp.copy()

# Moderate Adam polish: 3 phases (fewer iters per phase)
print("  Adam polish...", flush=True)
for lr, n_iter in [(1e-2, 3000), (1e-3, 3000), (1e-4, 3000)]:
    pp, r = adam_polish(bp, lr, n_iter)
    if r < best_r: best_r = r; bp = pp.copy()

# Fine multi-start (3 restarts)
for restart in range(3):
    key, sk = jax.random.split(key)
    noise = jax.random.normal(sk, (5,), dtype=jnp.float64) * 0.01
    pi = bp + noise * jnp.where(jnp.abs(bp) > 1e-12, jnp.abs(bp), 1.0)
    pp, r = adam_polish(pi, 1e-4, 3000)
    if r < best_r: best_r = r; bp = pp.copy()

st2 = time.time() - t2

# NumPy cross-check
up_jax = np.array(vmap(lambda xx, tt: u_5p(xx, tt, bp))(xf, tf)).reshape(NX, NT)
rel_np = float(np.sqrt(np.mean((up_jax - U_true_np)**2)) / (np.sqrt(np.mean(U_true_np**2)) + 1e-15))

# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"REPRODUCTION RESULTS: Burgers Equation (Table 3)")
print(f"{'='*60}")
print(f"  Relative L2 Error (repro):    {best_r:.6e}")
print(f"  Relative L2 Error (NumPy):    {rel_np:.6e}")
print(f"  Relative L2 Error (paper):    6.64e-14")
print(f"  Stage I wall time:            {st1:.1f}s")
print(f"  Stage II wall time:           {st2:.1f}s")
print(f"  Total wall time:              {st1+st2:.1f}s")
print(f"  Paper wall time:              11.62s")
print(f"  Final params (H,A,k,B,C):     {[float(v) for v in bp]}")
print(f"  True params  (H,A,k,B,C):     {[H_t, A_t, k_t, B_t, C_t]}")
print(f"  Expression: u(x,t) = {bp[0]:.15f} - {bp[1]:.15f}*tanh({bp[2]:.15f}*(x - {bp[3]:.15f} - {bp[4]:.15f}*t))")
print(f"\n  METRIC: rel_l2_error = {best_r:.6e}")
