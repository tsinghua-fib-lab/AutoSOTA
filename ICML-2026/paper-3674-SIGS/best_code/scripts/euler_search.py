# ──────────────────────────────────────────────────────────────────────────────
# Portable setup: paths, imports, device, multiprocessing (cross-platform)
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import os, sys, multiprocessing as mp

os.environ.setdefault("LODE_CONFIG", "configs/config.yaml")
os.environ.setdefault("LODE_H5", "data/expressions.h5")

# Put the repo root on sys.path so sigs.* packages can be imported
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Multiprocessing mode that's safe in notebooks and cross-platform
if mp.get_start_method(allow_none=True) != "spawn":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Dependencies (standard + ML)
import re, pickle, logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, List, Optional, Set, FrozenSet
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import h5py
import yaml
import sympy as sp
import symengine as se
from cma import CMAEvolutionStrategy as cmaES
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns  # optional

# Your modules (in this folder)
from sigs.model import GrammarVAE
from sigs.training import GrammarVAEModel
from sigs.grammar import GCFG, S, get_mask
from sigs.stack import Stack
from nltk import Nonterminal

from sigs.utils import *
from sigs.sampler import FlexibleVectorSampler
from sigs.evaluator import DerivativeEvaluator, ExpressionEvaluator

from problems.compressible_euler import create_compressible_euler_problem, build_mesh
# ──────────────────────────────────────────────────────────────────────────────
# Paths (relative to ROOT) with environment-variable overrides
# Users can override with:
#   export LODE_CONFIG=...
#   export LODE_CKPT=...
#   export LODE_H5=...
#   export LODE_CSV=...
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(os.getenv("LODE_CONFIG", "configs/config.yaml"))

# If the user doesn't set LODE_CKPT, try to pick the newest *.ckpt anywhere under ROOT
_ckpt_env = os.getenv("LODE_CKPT")
if _ckpt_env:
    CKPT_PATH = Path(_ckpt_env)
else:
    ckpt_candidates = sorted(ROOT.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpt_candidates:
        raise FileNotFoundError(
            "No .ckpt found under project root. Set LODE_CKPT env var or add a checkpoint."
        )
    CKPT_PATH = ckpt_candidates[-1]

H5_INPUT  = Path(os.getenv("LODE_H5",  "data/expressions.h5"))
CSV_INPUT = Path(os.getenv("LODE_CSV", ROOT / "tsoulos_dataset_1.csv"))

print("Using config: ", CONFIG_PATH)
print("Using ckpt:   ", CKPT_PATH)
print("Using H5:     ", H5_INPUT)
print("Using CSV:    ", CSV_INPUT)



# ──────────────────────────────────────────────────────────────────────────────
# Load model & data
# ──────────────────────────────────────────────────────────────────────────────
config = ModelUtils.load_config(CONFIG_PATH)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ModelUtils.load_checkpoint(CKPT_PATH, config).to(device).eval()
print(f"Device: {device} — Model: {CKPT_PATH.name}")

with h5py.File(H5_INPUT, 'r') as hf:
    data = torch.from_numpy(hf['data'][:]).float().squeeze(1).permute(0, 2, 1)
print(f"Data shape: {tuple(data.shape)}")
print("Setup complete!")



# ──────────────────────────────────────────────────────────────────────────────
# Build/load decoded/flags/clusters pickles (portable)
# ──────────────────────────────────────────────────────────────────────────────
clu_pkl = Path("data/clusters.pkl")

if not clu_pkl.exists():
    dec_pkl = ROOT / 'decoded_helm_latest_final-geo-loss-final-t.pkl'
    flg_pkl = ROOT / 'expression_flags_helm_latest_final-geo-loss-final-t.pkl'
    dec_pkl = ensure_decoded(model, data, dec_pkl)
    flg_pkl = ensure_flags(dec_pkl, flg_pkl)
    clu_pkl = ensure_clusters(flg_pkl, clu_pkl)
    print(f"Built clusters: {clu_pkl}")
else:
    print(f"Found clusters: {clu_pkl}")

with open(clu_pkl, 'rb') as f:
    clusters_data = pickle.load(f)

print("Loaded clusters for classes:", list(clusters_data.keys()))



import sympy as sp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def euler_system_loss(
    rho_expr_str: str,
    u_expr_str: str,
    v_expr_str: str,
    p_expr_str: str,
    problem: dict,
    meshes,
    weight_eqs: dict = None,
    lambda_bc: float = 100.0,   # <--- NEW: weight for boundary-condition loss
) -> float:
    """
    Compute total loss for the compressible Euler system given 4 ansätze:
        rho_hat(x,y), u_hat(x,y), v_hat(x,y), p_hat(x,y).

    - PDE loss: RMSE of residuals for rho, u, v, E
    - BC loss:  RMSE of mismatch vs manufactured solution on boundary
    """
    if weight_eqs is None:
        weight_eqs = {"rho": 1.0, "u": 1.0, "v": 1.0, "E": 1.0}

    # ------------------------------------------------------------------
    # 0. Unpack problem + mesh
    # ------------------------------------------------------------------
    F_sym   = problem["F_symbolic"]          # sympy expressions
    gamma   = problem["parameters"]["gamma"]
    manu    = problem["manufactured_solution"]

    rho_star = manu["rho"]
    u_star   = manu["u"]
    v_star   = manu["v"]
    p_star   = manu["p"]

    X, Y = meshes  # meshes was [X, Y]
    x, y = sp.symbols('x y')

    # ------------------------------------------------------------------
    # 1. Parse ansätze as sympy expressions
    # ------------------------------------------------------------------
    try:
        rho_hat = sp.sympify(rho_expr_str.replace("^", "**"))
        u_hat   = sp.sympify(u_expr_str.replace("^", "**"))
        v_hat   = sp.sympify(v_expr_str.replace("^", "**"))
        p_hat   = sp.sympify(p_expr_str.replace("^", "**"))
    except Exception:
        # parsing failed → huge penalty
        return 1e9

    # Energy from ansätze
    E_hat = p_hat / (gamma - 1) + sp.Rational(1, 2) * rho_hat * (u_hat**2 + v_hat**2)

    # ------------------------------------------------------------------
    # 2. Build residuals in conservative form
    # ------------------------------------------------------------------
    f_rho = F_sym["rho"]
    f_u   = F_sym["u"]
    f_v   = F_sym["v"]
    f_E   = F_sym["E"]

    R_rho = sp.diff(rho_hat * u_hat, x) + sp.diff(rho_hat * v_hat, y) - f_rho
    R_u   = sp.diff(rho_hat * u_hat**2 + p_hat, x) + sp.diff(rho_hat * u_hat * v_hat, y) - f_u
    R_v   = sp.diff(rho_hat * u_hat * v_hat, x) + sp.diff(rho_hat * v_hat**2 + p_hat, y) - f_v
    R_E   = sp.diff((E_hat + p_hat) * u_hat, x) + sp.diff((E_hat + p_hat) * v_hat, y) - f_E

    # ------------------------------------------------------------------
    # 3. Force evaluation of any remaining Derivative terms
    # ------------------------------------------------------------------
    residuals = [R_rho, R_u, R_v, R_E]
    cleaned = []

    for R in residuals:
        R = R.doit()          # try to evaluate derivatives
        R = sp.sympify(R)

        # As a last-resort safety: kill any remaining Derivative pieces
        if R.has(sp.Derivative):
            R = R.replace(sp.Derivative, lambda *args, **kwargs: sp.Integer(0))

        cleaned.append(R)

    R_rho, R_u, R_v, R_E = cleaned

    # ------------------------------------------------------------------
    # 4. Lambdify PDE residuals & evaluate on the mesh
    # NOTE: Skip simplify() - takes too long on complex expressions
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_R_rho = sp.lambdify((x, y), R_rho, "numpy")
            f_R_u   = sp.lambdify((x, y), R_u,   "numpy")
            f_R_v   = sp.lambdify((x, y), R_v,   "numpy")
            f_R_E   = sp.lambdify((x, y), R_E,   "numpy")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        # e.g. ComplexInfinity / zoo inside the expression -> lambdify fails
        # Just penalize this system and move on.
        # print(f"  ⚠️  Skipping PDE residuals - lambdify error: {type(e).__name__}")
        return 1e9

    R_rho_vals = f_R_rho(X, Y)
    R_u_vals   = f_R_u(X, Y)
    R_v_vals   = f_R_v(X, Y)
    R_E_vals   = f_R_E(X, Y)

    def rmse(a):
        a = np.nan_to_num(a, nan=1e10, posinf=1e10, neginf=1e10)
        return float(np.sqrt(np.mean(a**2)))

    rmse_rho = rmse(R_rho_vals)
    rmse_u   = rmse(R_u_vals)
    rmse_v   = rmse(R_v_vals)
    rmse_E   = rmse(R_E_vals)

    # CRITICAL FIX: Normalize by typical flux scales to balance contributions
    # Mass flux ~ O(1), Momentum flux ~ O(2), Energy flux ~ O(7-10)
    # Without normalization, energy dominates 55% of loss!
    # With normalization: each equation contributes ~25%
    scale_rho = 1.3  # typical |ρu| or |ρv|
    scale_u = 1.8    # typical |ρu²| or |p|
    scale_v = 1.2    # typical |ρuv|
    scale_E = 7.5    # typical |(E+p)u| or |(E+p)v|

    L_PDE = (
        weight_eqs["rho"] * (rmse_rho / scale_rho)
      + weight_eqs["u"]   * (rmse_u / scale_u)
      + weight_eqs["v"]   * (rmse_v / scale_v)
      + weight_eqs["E"]   * (rmse_E / scale_E)
    )

    # # ------------------------------------------------------------------
    # # 5. Boundary-condition loss vs manufactured solution
    # # ------------------------------------------------------------------
    # # Lambdify manufactured solution and ansatz fields (with error handling)
    # try:
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             rho_star_fn = sp.lambdify((x, y), rho_star, "numpy")
    #             u_star_fn   = sp.lambdify((x, y), u_star,   "numpy")
    #             v_star_fn   = sp.lambdify((x, y), v_star,   "numpy")
    #             p_star_fn   = sp.lambdify((x, y), p_star,   "numpy")

    #             rho_hat_fn  = sp.lambdify((x, y), rho_hat,  "numpy")
    #             u_hat_fn    = sp.lambdify((x, y), u_hat,    "numpy")
    #             v_hat_fn    = sp.lambdify((x, y), v_hat,    "numpy")
    #             p_hat_fn    = sp.lambdify((x, y), p_hat,    "numpy")
    # except (KeyError, ValueError, TypeError, AttributeError) as e:
    #         # Skip expressions that can't be lambdified (e.g., contain ComplexInfinity, undefined symbols)
    #         print(f"  ⚠️  Skipping expression combination - lambdify error: {type(e).__name__}")
    #         return 1000.0  # Return a large penalty to skip this combination

    # # Build boundary mask on the same mesh
    # nx, ny = X.shape
    # i = np.arange(nx)[:, None]
    # j = np.arange(ny)[None, :]

    # boundary_mask = (i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1)

    # Xb = X[boundary_mask]
    # Yb = Y[boundary_mask]

    # # Evaluate on boundary
    # rho_hat_b  = rho_hat_fn(Xb, Yb)
    # u_hat_b    = u_hat_fn(Xb, Yb)
    # v_hat_b    = v_hat_fn(Xb, Yb)
    # p_hat_b    = p_hat_fn(Xb, Yb)

    # rho_star_b = rho_star_fn(Xb, Yb)
    # u_star_b   = u_star_fn(Xb, Yb)
    # v_star_b   = v_star_fn(Xb, Yb)
    # p_star_b   = p_star_fn(Xb, Yb)

    # bc_rho_rmse = rmse(rho_hat_b - rho_star_b)
    # bc_u_rmse   = rmse(u_hat_b   - u_star_b)
    # bc_v_rmse   = rmse(v_hat_b   - v_star_b)
    # bc_p_rmse   = rmse(p_hat_b   - p_star_b)
        # ------------------------------------------------------------------
    # 5. Periodic boundary-condition loss
    # ------------------------------------------------------------------
    # We enforce:
    #   rho(0,y) ≈ rho(1,y),    rho(x,0) ≈ rho(x,1)
    #   same for u, v, p

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho_hat_fn  = sp.lambdify((x, y), rho_hat,  "numpy")
            u_hat_fn    = sp.lambdify((x, y), u_hat,    "numpy")
            v_hat_fn    = sp.lambdify((x, y), v_hat,    "numpy")
            p_hat_fn    = sp.lambdify((x, y), p_hat,    "numpy")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        # If ansatz cannot be evaluated safely, penalize heavily
        # print(f"  ⚠️  Skipping expression combination - lambdify error: {type(e).__name__}")
        return 1e9

    nx, ny = X.shape

    # Left / right boundaries: i = 0, i = nx-1, vary j
    xL, yL = X[0, :],     Y[0, :]
    xR, yR = X[-1, :],    Y[-1, :]

    # Bottom / top boundaries: j = 0, j = ny-1, vary i
    xB, yB = X[:, 0],     Y[:, 0]
    xT, yT = X[:, -1],    Y[:, -1]

    # Evaluate ansatz on these boundaries
    rho_L = rho_hat_fn(xL, yL);   rho_R = rho_hat_fn(xR, yR)
    rho_B = rho_hat_fn(xB, yB);   rho_T = rho_hat_fn(xT, yT)

    u_L   = u_hat_fn(xL, yL);     u_R   = u_hat_fn(xR, yR)
    u_B   = u_hat_fn(xB, yB);     u_T   = u_hat_fn(xT, yT)

    v_L   = v_hat_fn(xL, yL);     v_R   = v_hat_fn(xR, yR)
    v_B   = v_hat_fn(xB, yB);     v_T   = v_hat_fn(xT, yT)

    p_L   = p_hat_fn(xL, yL);     p_R   = p_hat_fn(xR, yR)
    p_B   = p_hat_fn(xB, yB);     p_T   = p_hat_fn(xT, yT)

    # Periodic mismatch: left–right + bottom–top
    bc_rho_rmse = rmse(rho_L - rho_R) + rmse(rho_B - rho_T)
    bc_u_rmse   = rmse(u_L   - u_R)   + rmse(u_B   - u_T)
    bc_v_rmse   = rmse(v_L   - v_R)   + rmse(v_B   - v_T)
    bc_p_rmse   = rmse(p_L   - p_R)   + rmse(p_B   - p_T)
    L_BC = bc_rho_rmse + bc_u_rmse + bc_v_rmse + bc_p_rmse

    # ------------------------------------------------------------------
    # 6. Total loss = PDE interior + boundary mismatch
    # ------------------------------------------------------------------
    total_loss = L_PDE + lambda_bc * L_BC

    # Optional: debug print
    # print("RMSE PDE:", rmse_rho, rmse_u, rmse_v, rmse_E,
    #       " | BC:", bc_rho_rmse, bc_u_rmse, bc_v_rmse, bc_p_rmse,
    #       " | L_PDE:", L_PDE, " L_BC:", L_BC, " total:", total_loss)

    return total_loss
import numpy as np
import random


# Global variables for parallel evaluation (needed for multiprocessing)
_global_ce_problem = None
_global_meshes = None
_global_rho_candidates = None
_global_u_candidates = None
_global_v_candidates = None
_global_p_candidates = None
_global_n_per_field = None
_global_seed = None
_global_use_same_series = False

def _evaluate_combination_worker(it):
    """Worker function for parallel evaluation. Must be at module level for pickling."""
    # Each worker needs its own random state
    local_random = random.Random(_global_seed + it)
    # If using the same series for all fields, pick one index and reuse it
    if _global_use_same_series:
        idx = local_random.randrange(_global_n_per_field)
        i_rho = i_u = i_v = i_p = idx
    else:
        i_rho = local_random.randrange(_global_n_per_field)
        i_u   = local_random.randrange(_global_n_per_field)
        i_v   = local_random.randrange(_global_n_per_field)
        i_p   = local_random.randrange(_global_n_per_field)

    rho_expr = _global_rho_candidates[i_rho]
    u_expr   = _global_u_candidates[i_u]
    v_expr   = _global_v_candidates[i_v]
    p_expr   = _global_p_candidates[i_p]

    loss = euler_system_loss(
        rho_expr_str=rho_expr,
        u_expr_str=u_expr,
        v_expr_str=v_expr,
        p_expr_str=p_expr,
        problem=_global_ce_problem,
        meshes=_global_meshes,
    )

    return (it, loss, (i_rho, i_u, i_v, i_p), (rho_expr, u_expr, v_expr, p_expr))


def _evaluate_combination_with_index(it, idx):
    """Worker variant that uses a provided index for all four fields.
    This guarantees the same sampled series is used for rho,u,v,p for this
    combination. Useful when `use_same_series=True` to avoid any race or
    seeding issues inside the worker.
    """
    # Respect global seed for reproducibility in any other random choices
    local_random = random.Random(_global_seed + it)
    i_rho = i_u = i_v = i_p = int(idx)

    rho_expr = _global_rho_candidates[i_rho]
    u_expr   = _global_u_candidates[i_u]
    v_expr   = _global_v_candidates[i_v]
    p_expr   = _global_p_candidates[i_p]

    loss = euler_system_loss(
        rho_expr_str=rho_expr,
        u_expr_str=u_expr,
        v_expr_str=v_expr,
        p_expr_str=p_expr,
        problem=_global_ce_problem,
        meshes=_global_meshes,
    )

    return (it, loss, (i_rho, i_u, i_v, i_p), (rho_expr, u_expr, v_expr, p_expr))

def search_best_ce_system(
    sampler,
    model,
    n_series_samples: int = 2000,
    n_per_field: int = 20,
    n_combinations: int = 200,
    seed: int = 123,
    sin_sin_only: bool = False,
    use_same_series: bool = False,
):
    """
    1) Use sampler.sample_coherent_sum_expressions to get spatial 2D series S(x,y)
    2) Build ansätze:
         rho(x,y) = exp(S_rho(x,y))
         u(x,y)   = S_u(x,y)
         v(x,y)   = S_v(x,y)
         p(x,y)   = exp(S_p(x,y))
    3) Random search over 4-tuples (rho,u,v,p) → minimize full CE system loss.
    """

    np.random.seed(seed)
    random.seed(seed)

    # ----------------------------------------------------------
    # 1. Compressible Euler problem and mesh
    # ----------------------------------------------------------
    ce_problem = create_compressible_euler_problem()
    X, Y = build_mesh(ce_problem)
    meshes = [X, Y]

    # ----------------------------------------------------------
    # 2. Sample 2D spatial series S(x,y) with your sampler
    # ----------------------------------------------------------
    # Use your SPATIOTEMPORAL_2D category (x,y) and coherent sum.
    series_sample_id = sampler.sample_coherent_sum_expressions(
        expression_template="A*B",
        role_categories={"A": "SPATIAL_2D", "B": "CONSTANT"},
        role_subclusters={"SPATIAL_2D": 1000, "CONSTANT": 1},  # 5 subclusters: KMeans puts pure trig (sin*cos) in subcluster 1!
        n_sum_terms=5,          # 4 terms to match K=2 Fourier modes in manufactured solution
        sum_operator="+",
        n_samples=n_series_samples,
        seed=seed,
        model=model,
    )

    S_exprs, S_vecs, S_sub_idxs, S_expr_idxs = sampler.get_sampling_results(series_sample_id)
    print(f"Sampled {len(S_exprs)} spatial series S(x,y).")

    # Filter for pure trig expressions (sin/cos with pi*)
    pure_trig_exprs = []
    for expr in S_exprs:
        has_trig = ('sin' in expr or 'cos' in expr) and 'pi*' in expr
        has_bad = any(bad in expr for bad in ['exp(-('])
        if has_trig and not has_bad:
            pure_trig_exprs.append(expr)

    print(f"Filtered to {len(pure_trig_exprs)} pure trigonometric expressions ({100*len(pure_trig_exprs)/len(S_exprs):.1f}%)")

    # Use pure trig if we have enough, otherwise use all
    if len(pure_trig_exprs) >= n_per_field:
        S_exprs = pure_trig_exprs
        print(f"Using {len(S_exprs)} pure trig expressions")
    else:
        print(f"Only {len(pure_trig_exprs)} pure trig, using all {len(S_exprs)}")

    if len(S_exprs) < n_per_field:
        raise ValueError(f"Not enough series sampled ({len(S_exprs)}) for n_per_field={n_per_field}")

    # Optional stronger filter: only allow sums composed of terms A*sin(pi*i*x)*sin(pi*j*y)
    if sin_sin_only:
        def is_sin_sin_expr(expr: str) -> bool:
            """Return True if expr is a sum/difference of terms each matching
            A*sin(pi*<int>*x)*sin(pi*<int>*y) where A is a numeric literal (optional sign).
            This is a conservative string-based check (no full parsing) intended to
            bias the search toward pure sin*sin Fourier sums.
            """
            # quick rejects
            if 'cos' in expr or 'tan' in expr or 'sinh' in expr or 'cosh' in expr:
                return False
            # split by + and - while preserving signs
            # replace '-' with '+-' so split yields signed terms
            toks = expr.replace('-', '+-').split('+')
            toks = [t.strip() for t in toks if t.strip()]
            for t in toks:
                # each term must contain two 'sin(' occurrences and contain '*x' and '*y' and 'pi*'
                if t.count('sin(') != 2:
                    return False
                if 'pi*' not in t:
                    return False
                if '*x' not in t or '*y' not in t:
                    return False
                # reject other function names or variables
                if any(bad in t for bad in ['exp', 'log', 'sqrt', '/', 'cos', 'tan', 'sinh', 'cosh']):
                    return False
            return True

        sin_sin_filtered = [s for s in S_exprs if is_sin_sin_expr(s)]
        print(f"sin_sin_only=True → {len(sin_sin_filtered)}/{len(S_exprs)} expressions match sin(pi*i*x)*sin(pi*j*y) pattern")
        if len(sin_sin_filtered) >= n_per_field:
            S_exprs = sin_sin_filtered
        else:
            print(f"Not enough sin*sin expressions ({len(sin_sin_filtered)}) for n_per_field={n_per_field}, proceeding with available trig expressions")

    # ----------------------------------------------------------
    # 3. Choose candidate series for each field
    # ----------------------------------------------------------
    if use_same_series:
        # Use the same spatial series S(x,y) for all fields; this reduces
        # structural search and focuses on finding coefficient differences.
        if len(S_exprs) < n_per_field:
            raise ValueError(f"Not enough series sampled ({len(S_exprs)}) for n_per_field={n_per_field}")
        S_list = random.sample(S_exprs, n_per_field)
        # For all fields use the same S; wrap rho and p with exp
        rho_candidates = [f"exp({s})" for s in S_list]
        u_candidates   = [s for s in S_list]
        v_candidates   = [s for s in S_list]
        p_candidates   = [f"exp({s})" for s in S_list]
        # keep per-field lists for downstream indexing/unpacking used later
        S_rho = S_list
        S_u = S_list
        S_v = S_list
        S_p = S_list
    else:
        S_rho = random.sample(S_exprs, n_per_field)
        S_u   = random.sample(S_exprs, n_per_field)
        S_v   = random.sample(S_exprs, n_per_field)
        S_p   = random.sample(S_exprs, n_per_field)

        # Build ansätze: exp for rho, p; identity for u, v
        rho_candidates = [f"exp({s})" for s in S_rho]
        u_candidates   = [s for s in S_u]
        v_candidates   = [s for s in S_v]
        p_candidates   = [f"exp({s})" for s in S_p]

    # ----------------------------------------------------------
    # 4. PARALLEL random search over combinations
    # ----------------------------------------------------------
    best_loss = float("inf")
    best_tuple_indices = None  # (i_rho, i_u, i_v, i_p)

    # Set up global variables for worker processes
    global _global_ce_problem, _global_meshes
    global _global_rho_candidates, _global_u_candidates, _global_v_candidates, _global_p_candidates
    global _global_n_per_field, _global_seed

    _global_ce_problem = ce_problem
    _global_meshes = meshes
    _global_rho_candidates = rho_candidates
    _global_u_candidates = u_candidates
    _global_v_candidates = v_candidates
    _global_p_candidates = p_candidates
    _global_n_per_field = n_per_field
    _global_seed = seed
    _global_use_same_series = use_same_series

    # Parallel evaluation using threads (avoids multiprocessing pickling issues)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm.auto import tqdm
        use_tqdm = True
    except Exception:
        use_tqdm = False

    n_workers = min((os.cpu_count() or 1), 16)
    print(f"Starting threaded search with {n_workers} workers...")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        if use_same_series:
            # Precompute one index per combination to guarantee the same S is used
            # for rho,u,v,p in each tuple (deterministic per-seed + it).
            futures = {
                executor.submit(
                    _evaluate_combination_with_index,
                    it,
                    random.Random(_global_seed + it).randrange(_global_n_per_field),
                ): it
                for it in range(n_combinations)
            }
        else:
            futures = {executor.submit(_evaluate_combination_worker, it): it for it in range(n_combinations)}

        if use_tqdm:
            # iterate as futures complete with progress bar
            for fut in tqdm(as_completed(futures), total=n_combinations, desc="Evaluating"):
                it, loss, indices, exprs = fut.result()
                if loss < best_loss:
                    best_loss = loss
                    best_tuple_indices = indices
                    print(f"\n[iter {it}] New best loss: {best_loss:.3e}")
                    print(f"  rho: {exprs[0]}")
                    print(f"  u: {exprs[1]}")
                    print(f"  v: {exprs[2]}")
                    print(f"  p: {exprs[3]}")
        else:
            idx = 0
            for fut in as_completed(futures):
                it, loss, indices, exprs = fut.result()
                idx += 1
                if idx % 100 == 0:
                    print(f"Progress: {idx}/{n_combinations} ({100*idx/n_combinations:.1f}%)")
                if loss < best_loss:
                    best_loss = loss
                    best_tuple_indices = indices
                    print(f"[iter {it}] New best loss: {best_loss:.3e}")
                    print(f"  rho: {exprs[0]}")
                    print(f"  u: {exprs[1]}")
                    print(f"  v: {exprs[2]}")
                    print(f"  p: {exprs[3]}")

    i_rho, i_u, i_v, i_p = best_tuple_indices
    best_rho_S = S_rho[i_rho]
    best_u_S   = S_u[i_u]
    best_v_S   = S_v[i_v]
    best_p_S   = S_p[i_p]

    best_rho_expr = rho_candidates[i_rho]
    best_u_expr   = u_candidates[i_u]
    best_v_expr   = v_candidates[i_v]
    best_p_expr   = p_candidates[i_p]

    print("\n=== BEST SYSTEM FOUND ===")
    print(f"Total loss: {best_loss:.6e}")
    print("rho(x,y) = exp(S_rho(x,y)),  with S_rho = ", best_rho_S)
    print("u(x,y)   = S_u(x,y),         with S_u   = ", best_u_S)
    print("v(x,y)   = S_v(x,y),         with S_v   = ", best_v_S)
    print("p(x,y)   = exp(S_p(x,y)),    with S_p   = ", best_p_S)

    return {
        "loss": best_loss,
        "S_rho": best_rho_S,
        "S_u":   best_u_S,
        "S_v":   best_v_S,
        "S_p":   best_p_S,
        "rho_expr": best_rho_expr,
        "u_expr":   best_u_expr,
        "v_expr":   best_v_expr,
        "p_expr":   best_p_expr,
    }

if __name__ == "__main__":
    # 1) Build sampler
    sampler = FlexibleVectorSampler(
        cluster_file="data/clusters.pkl",
        model=model,
        device="cuda",
    )

    # 2) Run the system-level search with optimized parameters
    # Strategy: Sample MANY expressions to increase chance of finding K=2 frequencies (2π, 4π)
    # With 20k samples, ~4% will have 2π or 4π per term → ~800 good candidates per field
    # This gives us a fighting chance to find compatible frequency combinations!
    result = search_best_ce_system(
        sampler=sampler,
        model=model,
        n_series_samples=100,  # 4x more samples → better frequency coverage
        n_per_field=100,         # 200 candidates per field (top ones after filtering)
        n_combinations=100,    # 10k combinations with parallel evaluation = fast!
        seed=42,
        sin_sin_only=True,
        use_same_series=True,
    )

    print("Best loss:", result["loss"])
    print("rho ansatz:", result["rho_expr"])
    print("u ansatz:",   result["u_expr"])
    print("v ansatz:",   result["v_expr"])
    print("p ansatz:",   result["p_expr"])



#optimization
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import optax
import re
import time

# High precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


# ---------------------------------------------------------------------------
# 1. Utilities: parse parameters and compile expressions
# ---------------------------------------------------------------------------

def parse_params(expr):
    """
    Extract numeric literals from expression and replace with learnable parameters.

    Returns:
        new_expr: string with p0, p1, ...
        values: jnp.array of initial values
        names: list of parameter names ["p0", "p1", ...]
    """
    # Convert ^ to ** first
    result_expr = expr.replace('^', '**')

    # Float / int literals with boundaries
    pattern = r'(?<![a-zA-Z0-9_.])([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(?![a-zA-Z0-9_.])'
    matches = []

    for match in re.finditer(pattern, result_expr):
        val_str = match.group(1)
        val = float(val_str)
        start, end = match.span()

        # Skip approximate pi, e
        if abs(val - 3.141592653589793) < 1e-10 or abs(val - 2.718281828459045) < 1e-10:
            continue

        # Skip exponents like **2, **3
        if start >= 2 and result_expr[start-2:start] == '**':
            continue

        # Skip obvious exponent-like integers (2,3,4,5,6,0.5) when they look like powers
        if val in [2.0, 3.0, 4.0, 5.0, 6.0, 0.5] and start > 0:
            context_start = max(0, start-3)
            context_end = min(len(result_expr), end+3)
            context = result_expr[context_start:context_end]
            if '**' in context or '^' in context:
                continue

        matches.append((start, end, val_str, val))

    # Replace from right to left (avoid index shifting)
    matches.sort(key=lambda x: x[0], reverse=True)

    param_values = []
    param_names = []

    for i, (start, end, val_str, val) in enumerate(matches):
        param_idx = len(matches) - 1 - i
        pname = f"p{param_idx}"

        if val_str.startswith('-'):
            replacement = f"-{pname}"
            param_values.insert(0, abs(val))
        else:
            replacement = pname
            param_values.insert(0, val)

        param_names.insert(0, pname)
        result_expr = result_expr[:start] + replacement + result_expr[end:]

    return result_expr, jnp.array(param_values, dtype=jnp.float64), param_names


def make_fn(expr, names):
    """
    Compile string expression into JAX function f(x, y, params).

    Here names are string parameter names: ["p0", "p1", ...].
    We expect params as a jnp.array with same length as names.
    """
    expr_fixed = expr.replace('^', '**')

    def fn(x, y, params):
        x = jnp.asarray(x, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64)

        ctx = {'x': x, 'y': y}

        if params is not None and names:
            params = jnp.asarray(params, dtype=jnp.float64)
            for n, p in zip(names, params):
                ctx[n] = p

        # math functions
        for f_name in ['sin', 'cos', 'exp', 'log', 'sqrt', 'tanh', 'sinh', 'cosh', 'abs']:
            ctx[f_name] = getattr(jnp, f_name)
        ctx['pi'] = jnp.pi
        ctx['e'] = jnp.e
        ctx['E'] = jnp.e

        return jnp.asarray(eval(expr_fixed, {"__builtins__": {}}, ctx), dtype=jnp.float64)

    return jit(fn)


# ---------------------------------------------------------------------------
# 2. Build Euler system: rho, u, v, p
# ---------------------------------------------------------------------------

def build_euler_system(rho_expr_str, u_expr_str, v_expr_str, p_expr_str):
    """
    From 4 expression strings, build:
    - parsed exprs
    - parameter arrays and names
    - JAX evaluators rho_fn, u_fn, v_fn, p_fn
    Returns:
        system: dict with everything
    """

    rho_expr, rho_vals, rho_names = parse_params(rho_expr_str)
    u_expr,   u_vals,   u_names   = parse_params(u_expr_str)
    v_expr,   v_vals,   v_names   = parse_params(v_expr_str)
    p_expr,   p_vals,   p_names   = parse_params(p_expr_str)

    rho_fn = make_fn(rho_expr, rho_names)
    u_fn   = make_fn(u_expr,   u_names)
    v_fn   = make_fn(v_expr,   v_names)
    p_fn   = make_fn(p_expr,   p_names)

    params_init = {
        "rho": rho_vals,
        "u":   u_vals,
        "v":   v_vals,
        "p":   p_vals,
    }

    system = {
        "rho_expr": rho_expr,
        "u_expr":   u_expr,
        "v_expr":   v_expr,
        "p_expr":   p_expr,
        "rho_names": rho_names,
        "u_names":   u_names,
        "v_names":   v_names,
        "p_names":   p_names,
        "rho_fn": rho_fn,
        "u_fn":   u_fn,
        "v_fn":   v_fn,
        "p_fn":   p_fn,
        "params_init": params_init,
    }

    return system


def detect_frequency_param_mask(expr_with_p, param_names):
    """Return a boolean mask (len==len(param_names)) marking params that appear
    to be frequency-like inside trig arguments, e.g. patterns like p0*pi*x or p3*pi*y
    """
    mask = [False] * len(param_names)
    try:
        # find all sin(...) and cos(...) arguments
        import re
        trig_args = re.findall(r"(?:sin|cos)\s*\(([^)]*)\)", expr_with_p)
        for arg in trig_args:
            for i, pname in enumerate(param_names):
                # common patterns where parameter multiplies pi and x or y
                # e.g. p0*pi*x, pi*p0*x, p0*pi*y
                if re.search(rf"\b{pname}\b\s*\*\s*pi\b", arg) or re.search(rf"pi\s*\*\s*\b{pname}\b", arg) or re.search(rf"\b{pname}\b\s*\*\s*x\b", arg) or re.search(rf"\b{pname}\b\s*\*\s*y\b", arg):
                    mask[i] = True
                # also if pname directly adjacent to 'pi' like p0*pi*x (no spaces)
                if f"{pname}*pi" in arg or f"pi*{pname}" in arg:
                    mask[i] = True
    except Exception:
        pass
    return jnp.array(mask, dtype=jnp.bool_)


# ---------------------------------------------------------------------------
# 3. Euler residuals on a grid + Dirichlet-to-manufactured BC
# ---------------------------------------------------------------------------

def make_euler_loss(system,
                    gamma=1.4,
                    forcing=None,
                    manufactured=None,
                    domain=(0.0, 1.0, 0.0, 1.0),
                    nx=32, ny=32,
                    lambda_bc=0.0):
    """
    Build a JAX loss function for the steady 2D compressible Euler system.

    Equations (conservative form, steady, 2D):
      ∂(ρu)/∂x + ∂(ρv)/∂y = f_ρ
      ∂(ρu²+p)/∂x + ∂(ρuv)/∂y = f_u
      ∂(ρuv)/∂x + ∂(ρv²+p)/∂y = f_v
      ∂((E+p)u)/∂x + ∂((E+p)v)/∂y = f_E

    where E = p/(γ - 1) + 0.5*ρ(u²+v²) (ideal gas).

    forcing: optional dict of callables f_rho(x,y), f_u(x,y), f_v(x,y), f_E(x,y)
             (defaults to 0 if not provided).

    manufactured: optional dict of callables giving manufactured solution:
        {
          "rho": rho_star(x,y),
          "u":   u_star(x,y),
          "v":   v_star(x,y),
          "p":   p_star(x,y),
        }
      If provided, we add a BC loss enforcing:
          u_hat ≈ u_star, etc., on ∂Ω (boundary of the domain).
    """

    rho_fn = system["rho_fn"]
    u_fn   = system["u_fn"]
    v_fn   = system["v_fn"]
    p_fn   = system["p_fn"]

    # Domain & grid
    x0, x1, y0, y1 = domain
    xs = jnp.linspace(x0, x1, nx, dtype=jnp.float64)
    ys = jnp.linspace(y0, y1, ny, dtype=jnp.float64)
    X, Y = jnp.meshgrid(xs, ys, indexing='ij')
    coords = jnp.stack([X.ravel(), Y.ravel()], axis=-1)  # shape (N, 2)

    # Default forcing (zeros)
    if forcing is None:
        def f_rho(x, y): return jnp.array(0.0, dtype=jnp.float64)
        def f_u(x, y):   return jnp.array(0.0, dtype=jnp.float64)
        def f_v(x, y):   return jnp.array(0.0, dtype=jnp.float64)
        def f_E(x, y):   return jnp.array(0.0, dtype=jnp.float64)
    else:
        f_rho = forcing.get("rho", lambda x, y: jnp.array(0.0, dtype=jnp.float64))
        f_u   = forcing.get("u",   lambda x, y: jnp.array(0.0, dtype=jnp.float64))
        f_v   = forcing.get("v",   lambda x, y: jnp.array(0.0, dtype=jnp.float64))
        f_E   = forcing.get("E",   lambda x, y: jnp.array(0.0, dtype=jnp.float64))

    def residuals_at_point(coord, params):
        """
        Compute the 4 Euler residuals at a single (x,y) point.
        params is a dict with keys "rho","u","v","p".
        """
        x, y = coord[0], coord[1]

        # Helper: evaluate fields
        def rho_xy(xx, yy): return rho_fn(xx, yy, params["rho"])
        def u_xy(xx, yy):   return u_fn(xx, yy, params["u"])
        def v_xy(xx, yy):   return v_fn(xx, yy, params["v"])
        def p_xy(xx, yy):   return p_fn(xx, yy, params["p"])

        rho = rho_xy(x, y)
        u   = u_xy(x, y)
        v   = v_xy(x, y)
        p   = p_xy(x, y)

        # Ideal gas total energy
        E = p/(gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

        # --- Mass equation: ∂(ρu)/∂x + ∂(ρv)/∂y - f_ρ = 0

        def Fx_mass(xx):
            return rho_xy(xx, y) * u_xy(xx, y)

        def Fy_mass(yy):
            return rho_xy(x, yy) * v_xy(x, yy)

        dFx_mass_dx = grad(Fx_mass)(x)
        dFy_mass_dy = grad(Fy_mass)(y)
        R_mass = dFx_mass_dx + dFy_mass_dy - f_rho(x, y)

        # --- Momentum-x: ∂(ρu² + p)/∂x + ∂(ρuv)/∂y - f_u = 0

        def Fx_momx(xx):
            rho_ = rho_xy(xx, y)
            u_   = u_xy(xx, y)
            p_   = p_xy(xx, y)
            return rho_ * u_**2 + p_

        def Fy_momx(yy):
            rho_ = rho_xy(x, yy)
            u_   = u_xy(x, yy)
            v_   = v_xy(x, yy)
            return rho_ * u_ * v_

        dFx_momx_dx = grad(Fx_momx)(x)
        dFy_momx_dy = grad(Fy_momx)(y)
        R_momx = dFx_momx_dx + dFy_momx_dy - f_u(x, y)

        # --- Momentum-y: ∂(ρuv)/∂x + ∂(ρv² + p)/∂y - f_v = 0

        def Fx_momy(xx):
            rho_ = rho_xy(xx, y)
            u_   = u_xy(xx, y)
            v_   = v_xy(xx, y)
            return rho_ * u_ * v_

        def Fy_momy(yy):
            rho_ = rho_xy(x, yy)
            v_   = v_xy(x, yy)
            p_   = p_xy(x, yy)
            return rho_ * v_**2 + p_

        dFx_momy_dx = grad(Fx_momy)(x)
        dFy_momy_dy = grad(Fy_momy)(y)
        R_momy = dFx_momy_dx + dFy_momy_dy - f_v(x, y)

        # --- Energy: ∂((E+p)u)/∂x + ∂((E+p)v)/∂y - f_E = 0

        def Fx_energy(xx):
            rho_ = rho_xy(xx, y)
            u_   = u_xy(xx, y)
            v_   = v_xy(xx, y)
            p_   = p_xy(xx, y)
            E_   = p_/(gamma - 1.0) + 0.5 * rho_ * (u_**2 + v_**2)
            return (E_ + p_) * u_

        def Fy_energy(yy):
            rho_ = rho_xy(x, yy)
            u_   = u_xy(x, yy)
            v_   = v_xy(x, yy)
            p_   = p_xy(x, yy)
            E_   = p_/(gamma - 1.0) + 0.5 * rho_ * (u_**2 + v_**2)
            return (E_ + p_) * v_

        dFx_energy_dx = grad(Fx_energy)(x)
        dFy_energy_dy = grad(Fy_energy)(y)
        R_energy = dFx_energy_dx + dFy_energy_dy - f_E(x, y)

        return jnp.array([R_mass, R_momx, R_momy, R_energy], dtype=jnp.float64)

    # Vectorize over grid
    v_residuals = vmap(lambda coord, params: residuals_at_point(coord, params),
                       in_axes=(0, None))

    @jit
    def loss_fn(params):
        """
        Total loss = PDE interior residuals + lambda_bc * Dirichlet BC mismatch
        wrt manufactured solution (if provided).
        params is a dict with keys "rho","u","v","p".
        """
        # --- 1) PDE interior loss ---
        R = v_residuals(coords, params)  # shape (N, 4)
        L_pde = jnp.mean(R**2)

               # --- 2) Dirichlet BC loss vs manufactured solution ---
        if manufactured is None or lambda_bc == 0.0:
            L_bc = jnp.array(0.0, dtype=jnp.float64)
        else:
            # evaluate current ansatz on grid
            rho_vals = rho_fn(X, Y, params["rho"])  # shape (nx, ny)
            u_vals   = u_fn(  X, Y, params["u"])
            v_vals   = v_fn(  X, Y, params["v"])
            p_vals   = p_fn(  X, Y, params["p"])

            # evaluate manufactured solution on grid
            rho_star_vals = manufactured["rho"](X, Y)
            u_star_vals   = manufactured["u"](  X, Y)
            v_star_vals   = manufactured["v"](  X, Y)
            p_star_vals   = manufactured["p"](  X, Y)

            def edge_mse(field, field_star):
                # left/right edges
                lr = jnp.concatenate([
                    (field[0, :]  - field_star[0, :])**2,
                    (field[-1, :] - field_star[-1, :])**2,
                ])
                # bottom/top edges
                bt = jnp.concatenate([
                    (field[:, 0]  - field_star[:, 0])**2,
                    (field[:, -1] - field_star[:, -1])**2,
                ])
                return jnp.mean(jnp.concatenate([lr, bt]))

            bc_rho = edge_mse(rho_vals, rho_star_vals)
            bc_u   = edge_mse(u_vals,   u_star_vals)
            bc_v   = edge_mse(v_vals,   v_star_vals)
            bc_p   = edge_mse(p_vals,   p_star_vals)

            L_bc = bc_rho + bc_u + bc_v + bc_p

        # --- 3) Total loss ---
        return L_pde + lambda_bc * L_bc

    return loss_fn, coords


# ---------------------------------------------------------------------------
# 4. Optimization loop
# ---------------------------------------------------------------------------

def optimize_euler_system(rho_expr_str, u_expr_str, v_expr_str, p_expr_str,
                          gamma=1.4,
                          domain=(0.0, 1.0, 0.0, 1.0),
                          nx=32, ny=32,
                          lr=1e-3, iters=1000,
                          lambda_bc=0.0,
                          forcing=None,
                          manufactured=None,
                          lambda_l1: float = 1e-4,
                          freeze_freqs: bool = False,
                          verbose=True):
    """
    High-level driver:
      - Build system from expressions
      - Build Euler loss (PDE + optional Dirichlet BC vs manufactured)
      - Run Adam optimization
      - Return optimized parameters + reconstructed expressions
    """

    system = build_euler_system(rho_expr_str, u_expr_str, v_expr_str, p_expr_str)
    params = system["params_init"]

    loss_fn, coords = make_euler_loss(
        system,
        gamma=gamma,
        forcing=forcing,
        manufactured=manufactured,
        domain=domain,
        nx=nx,
        ny=ny,
        lambda_bc=lambda_bc,
    )

    # Optionally wrap loss_fn to include L1 regularization on amplitudes
    if lambda_l1 is not None and lambda_l1 > 0.0:
        def loss_with_l1(params):
            base = loss_fn(params)
            l1 = 0.0
            for k, v in params.items():
                l1 = l1 + jnp.sum(jnp.abs(v))
            return base + lambda_l1 * l1

        loss_and_grad = jit(value_and_grad(loss_with_l1))
    else:
        loss_and_grad = jit(value_and_grad(loss_fn))

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    if verbose:
        print("Initial parameter sizes:")
        for k, v in params.items():
            print(f"  {k}: {v.shape}")
        print("Initial loss:", float(loss_fn(params)))

    best_params = params
    best_loss = float(loss_fn(params))

    # If requested, detect frequency-like parameter indices for each field
    freq_masks = {}
    if freeze_freqs:
        for k in ['rho', 'u', 'v', 'p']:
            names = system.get(f"{k}_names", [])
            expr = system.get(f"{k}_expr", "")
            if names:
                try:
                    mask = detect_frequency_param_mask(expr, names)
                except Exception:
                    mask = jnp.zeros((len(names),), dtype=jnp.bool_)
            else:
                mask = jnp.zeros((0,), dtype=jnp.bool_)
            freq_masks[k] = mask
        if verbose:
            print("Freeze frequencies enabled. Frequency masks:")
            for k, m in freq_masks.items():
                print(f"  {k}: {m.tolist()}")

    t0 = time.time()
    for i in range(iters):
        loss_val, grads = loss_and_grad(params)
        updates, opt_state = opt.update(grads, opt_state)

        # If freezing frequencies, zero-out updates for those indices so they don't change
        if freeze_freqs:
            for k in ['rho', 'u', 'v', 'p']:
                mask = freq_masks.get(k, None)
                if mask is None or mask.size == 0:
                    continue
                # Only zero where mask=True (frequency-like params)
                # updates[k] is a jnp.array; set masked positions to 0
                u = updates[k]
                if u is not None and u.size > 0:
                    # convert mask to same dtype and shape
                    zero_mask = jnp.asarray(mask, dtype=jnp.bool_)
                    # create zeros array where masked
                    updates_k = jnp.where(zero_mask, jnp.zeros_like(u), u)
                    updates = dict(updates)
                    updates[k] = updates_k

        params = optax.apply_updates(params, updates)

        if loss_val < best_loss:
            best_loss = float(loss_val)
            best_params = params

        if verbose and (i % 50 == 0 or i == iters-1):
            rmse = float(jnp.sqrt(loss_val))
            print(f"[{i:4d}] loss = {float(loss_val):.3e}, rmse = {rmse:.3e}")

    if verbose:
        print("Optimization finished in %.2f s" % (time.time() - t0))
        print("Best loss:", best_loss, "RMSE:", float(jnp.sqrt(best_loss)))

    # Reconstruct expressions with best parameters
    rho_opt_expr = reconstruct_expr(system["rho_expr"], system["rho_names"], best_params["rho"])
    u_opt_expr   = reconstruct_expr(system["u_expr"],   system["u_names"],   best_params["u"])
    v_opt_expr   = reconstruct_expr(system["v_expr"],   system["v_names"],   best_params["v"])
    p_opt_expr   = reconstruct_expr(system["p_expr"],   system["p_names"],   best_params["p"])

    return {
        "best_loss": best_loss,
        "best_params": best_params,
        "rho_expr_opt": rho_opt_expr,
        "u_expr_opt":   u_opt_expr,
        "v_expr_opt":   v_opt_expr,
        "p_expr_opt":   p_opt_expr,
        "system": system,
    }


def reconstruct_expr(expr_with_p, param_names, param_values):
    """
    Replace p0, p1, ... in expr_with_p by their numeric values.
    """
    out = expr_with_p
    # Replace in reverse to avoid substring clashes
    for i, name in enumerate(reversed(param_names)):
        idx = len(param_names) - 1 - i
        out = re.sub(rf'\b{name}\b', f"{float(param_values[idx]):.16f}", out)
    return out.replace('**', '^')


# ---------------------------------------------------------------------------
# 5. Example usage with your SIGS-found form (no forcing, no BC)
# ---------------------------------------------------------------------------
def build_forcing_and_manufactured_from_sympy_problem(problem):
    """
    Take the SymPy 'problem' dict from create_compressible_euler_problem
    and turn it into:
      - forcing:     f_rho, f_u, f_v, f_E : (x,y) -> scalar (JAX-compatible)
      - manufactured: rho*, u*, v*, p*    : (X,Y) -> field (JAX-compatible)
    """

    x = problem["symbols"]["x"]
    y = problem["symbols"]["y"]

    manu = problem["manufactured_solution"]
    Fsym = problem["F_symbolic"]

    # Small module dict that tells lambdify to use jax.numpy instead of numpy
    jax_modules = {
        "sin":  jnp.sin,
        "cos":  jnp.cos,
        "exp":  jnp.exp,
        "log":  jnp.log,
        "sqrt": jnp.sqrt,
        "tanh": jnp.tanh,
        "sinh": jnp.sinh,
        "cosh": jnp.cosh,
        "Abs":  jnp.abs,
        "pi":   jnp.pi,
        "E":    jnp.e,
    }

    # ----- manufactured solution → JAX -----
    rho_star_jax = sp.lambdify((x, y), manu["rho"], modules=jax_modules)
    u_star_jax   = sp.lambdify((x, y), manu["u"],   modules=jax_modules)
    v_star_jax   = sp.lambdify((x, y), manu["v"],   modules=jax_modules)
    p_star_jax   = sp.lambdify((x, y), manu["p"],   modules=jax_modules)

    manufactured = {
        # These can take JAX arrays X, Y directly inside `loss_fn`
        "rho": lambda X, Y: rho_star_jax(X, Y),
        "u":   lambda X, Y: u_star_jax(  X, Y),
        "v":   lambda X, Y: v_star_jax(  X, Y),
        "p":   lambda X, Y: p_star_jax(  X, Y),
    }

    # ----- forcing terms → JAX -----
    f_rho_expr = Fsym["rho"].doit()
    f_u_expr   = Fsym["u"].doit()
    f_v_expr   = Fsym["v"].doit()
    f_E_expr   = Fsym["E"].doit()

    f_rho_jax = sp.lambdify((x, y), f_rho_expr, modules=jax_modules)
    f_u_jax   = sp.lambdify((x, y), f_u_expr,   modules=jax_modules)
    f_v_jax   = sp.lambdify((x, y), f_v_expr,   modules=jax_modules)
    f_E_jax   = sp.lambdify((x, y), f_E_expr,   modules=jax_modules)

    forcing = {
        # These take scalar x,y (tracers) inside residuals_at_point
        "rho": lambda x_, y_: f_rho_jax(x_, y_),
        "u":   lambda x_, y_: f_u_jax(  x_, y_),
        "v":   lambda x_, y_: f_v_jax(  x_, y_),
        "E":   lambda x_, y_: f_E_jax(  x_, y_),
    }

    return forcing, manufactured

if __name__ == "__main__":
    # 1) Build SymPy problem
    # from your other file or same notebook:
    # problem = create_compressible_euler_problem(seed=42)

    forcing, manufactured = build_forcing_and_manufactured_from_sympy_problem(problem)

    rho_expr_str = "exp(1*((-0.31*cos(2*pi*x)*sin(7*pi*y)) + (-1.32*sin(1*pi*x)*cos(6*pi*y)) + (0.38*sin(6*pi*x)*sin(6*pi*y)) + (-0.31*sin(1*pi*x)*sin(1*pi*y))))"
    u_expr_str   = "(0.15*cos(8*pi*x)*sin(3*pi*y)) + (0.14*cos(5*pi*x)*sin(4*pi*y)) + (-0.54*cos(1*pi*x)*sin(5*pi*y)) + (-0.11*cos(8*pi*x)*sin(7*pi*y))"
    v_expr_str   = "(sin(3*pi*x)*cos(3*pi*y)) + (sin(3*pi*x)*sin(1*pi*y)) + (sin(3*pi*x)*sin(2*pi*y)) + (sin(4*pi*x)*cos(5*pi*y))"
    p_expr_str   = "exp(1*((-0.26*cos(8*pi*x)*sin(3*pi*y)) + (-0.57*sin(2*pi*x)*sin(2*pi*y)) + (-0.46*sin(5*pi*x)*sin(5*pi*y)) + (-0.64*cos(1*pi*x)*sin(2*pi*y))))"

    mx = problem["mesh"]["x"]
    my = problem["mesh"]["y"]
    domain = (mx["start"], mx["end"], my["start"], my["end"])

    results = optimize_euler_system(
        rho_expr_str, u_expr_str, v_expr_str, p_expr_str,
        gamma=problem["parameters"]["gamma"],
        domain=domain,
        nx=mx["points"],
        ny=my["points"],
        lr=5e-2,
        iters=100,
        lambda_bc=5.0,
        forcing=forcing,
        manufactured=manufactured,
        verbose=True,
    )

    print("\nOptimized expressions:")
    print("rho(x,y) =", results["rho_expr_opt"])
    print("u(x,y)   =", results["u_expr_opt"])
    print("v(x,y)   =", results["v_expr_opt"])
    print("p(x,y)   =", results["p_expr_opt"])

import sympy as sp
import numpy as np
from netCDF4 import Dataset
import os

np.random.seed(0)

jobarray_size = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
jobarray_index = int(os.environ['SLURM_ARRAY_TASK_ID'])

K = 10
r = 0.5
num_samples = 40000 // jobarray_size

gamma = 1.4

x, y = sp.symbols('x y')
i, j = sp.symbols('i j', integer=True)

X = np.linspace(0, 1, 128)
Y = np.linspace(0, 1, 128)
X, Y = np.meshgrid(X, Y)

# forward the random number generator
for _ in range(jobarray_index * num_samples):
    A = np.random.uniform(-1, 1, (K, K))
    B = np.random.uniform(-1, 1, (K, K))
    C = np.random.uniform(-1, 1, (K, K))
    D = np.random.uniform(-1, 1, (K, K))

dataset = Dataset(f"/cluster/scratch/herdem/CE_manufactured_sincos_{jobarray_index}.nc", "w")
dataset.createDimension("num_samples", num_samples)
dataset.createDimension("channels", 8)
dataset.createDimension("x", 128)
dataset.createDimension("y", 128)
dataset.createVariable("solution", "f4", ("num_samples", "channels", "x", "y"), chunksizes=(1, 8, 128, 128))

for s in range(num_samples):
    rho, u, v, p = sp.Function('rho')(x, y), sp.Function('u')(x, y), sp.Function('v')(x, y), sp.Function('p')(x, y)

    A = np.random.uniform(-1, 1, (K, K))
    B = np.random.uniform(-1, 1, (K, K))
    C = np.random.uniform(-1, 1, (K, K))
    D = np.random.uniform(-1, 1, (K, K))

    rho_symb = sp.exp((1 / K**2) * sum(
        A[i-1, j-1] * (i**2 + j**2)**r *
        sp.cos(2 * sp.pi * i * x) * sp.sin(2 * sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    ))
    u_symb = (sp.pi / K**2) * sum(
        B[i-1, j-1] * (i**2 + j**2)**r *
        sp.cos(2 * sp.pi * i * x) * sp.sin(2 * sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    )
    v_symb = (sp.pi / K**2) * sum(
        C[i-1, j-1] * (i**2 + j**2)**r *
        sp.cos(2 * sp.pi * i * x) * sp.sin(2 * sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    )
    p_symb = sp.exp((1 / K**2) * sum(
        D[i-1, j-1] * (i**2 + j**2)**r *
        sp.cos(2 * sp.pi * i * x) * sp.sin(2 * sp.pi * j * y)
        for i in range(1, K + 1) for j in range(1, K + 1)
    ))

    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
    f_rho = sp.diff(rho * u, x) + sp.diff(rho * v, y)
    f_u = sp.diff(rho * u**2 + p, x) + sp.diff(rho * u * v, y)
    f_v = sp.diff(rho * u * v, x) + sp.diff(rho * v**2 + p, y)
    f_E = sp.diff((E + p) * u, x) + sp.diff((E + p) * v, y)

    E_subs = E.subs({rho: rho_symb, u: u_symb, v: v_symb, p: p_symb})
    f_rho_subs = f_rho.subs({rho: rho_symb, u: u_symb, v: v_symb, p: p_symb})
    f_u_subs = f_u.subs({rho: rho_symb, u: u_symb, v: v_symb, p: p_symb})
    f_v_subs = f_v.subs({rho: rho_symb, u: u_symb, v: v_symb, p: p_symb})
    f_E_subs = f_E.subs({rho: rho_symb, u: u_symb, v: v_symb, p: p_symb, E: E_subs})

    rho_numeric = sp.lambdify((x, y), rho_symb, modules='numpy')
    u_numeric = sp.lambdify((x, y), u_symb, modules='numpy')
    v_numeric = sp.lambdify((x, y), v_symb, modules='numpy')
    p_numeric = sp.lambdify((x, y), p_symb, modules='numpy')

    rho_vals = rho_numeric(X, Y)
    u_vals = u_numeric(X, Y)
    v_vals = v_numeric(X, Y)
    p_vals = p_numeric(X, Y)

    dataset.variables["solution"][s, 0] = rho_vals
    dataset.variables["solution"][s, 1] = u_vals
    dataset.variables["solution"][s, 2] = v_vals
    dataset.variables["solution"][s, 3] = p_vals

    f_rho_numeric = sp.lambdify((x, y), f_rho_subs.doit(), modules='numpy')
    f_u_numeric = sp.lambdify((x, y), f_u_subs.doit(), modules='numpy')
    f_v_numeric = sp.lambdify((x, y), f_v_subs.doit(), modules='numpy')
    f_E_numeric = sp.lambdify((x, y), f_E_subs.doit(), modules='numpy')

    f_rho_vals = f_rho_numeric(X, Y)
    f_u_vals = f_u_numeric(X, Y)
    f_v_vals = f_v_numeric(X, Y)
    f_E_vals = f_E_numeric(X, Y)

    dataset.variables["solution"][s, 4] = f_rho_vals
    dataset.variables["solution"][s, 5] = f_u_vals
    dataset.variables["solution"][s, 6] = f_v_vals
    dataset.variables["solution"][s, 7] = f_E_vals

dataset.close()
