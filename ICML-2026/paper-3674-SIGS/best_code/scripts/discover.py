import os
import json
import time

import numpy as np
import symengine as sm
import sympy as sp
import warnings
import itertools
import random
from tqdm import tqdm

from problems.shallow_water import create_shallow_water_problem, setup_shallow_symbols_meshes
from sigs.sampler import FlexibleVectorSampler
from sigs.utils import MathClass, ExpressionUtils


def combined_rmse_loss(residuals_list):
    """Compute a single RMSE over a list of residual arrays.

    Uses JAX (jax.numpy) when available for faster array ops; falls back to
    NumPy otherwise. Non-finite residuals are clamped to a large value so
    they contribute heavily to the RMSE (matching earlier inf-handling).
    """
    # Lazy import jax to keep dependency optional
    try:
        import jax.numpy as jnp  # type: ignore
        use_jax = True
    except Exception:
        import numpy as jnp  # type: ignore
        use_jax = False

    if not residuals_list:
        return 0.0

    # Ensure everything is a 1D array and concatenate
    try:
        parts = [jnp.ravel(r) for r in residuals_list]
        allr = jnp.concatenate(parts)
    except Exception:
        # Fallback: convert each to numpy then concatenate
        import numpy as _np

        parts = [_np.ravel(r) for r in residuals_list]
        allr = _np.concatenate(parts)

    # Clamp non-finite entries so they give a large RMSE rather than crash
    if use_jax:
        finite = jnp.isfinite(allr)
        allr = jnp.where(finite, allr, 1e6)
        mse = jnp.mean(allr ** 2)
        rmse = jnp.sqrt(mse)
        # Convert to native Python float
        return float(rmse)
    else:
        import numpy as _np

        allr = _np.asarray(allr, dtype=_np.float64)
        allr[~_np.isfinite(allr)] = 1e6
        return float(_np.sqrt(_np.mean(allr ** 2)))



def shallow_rho_loss(
    rho_sym,
    problem,
    symbols,
    meshes,
    w_pde: float = 1.0,   # Increased: PDE should dominate
    w_ic: float = 1.0,      # Reduced: IC is just one time point
    w_bc: float = .0,       # Reduced: BC helps but shouldn't dominate
):
    """Rho-focused loss using only the continuity equation + IC/BC for rho.

    This is used as a fast screen to rank rho candidates while keeping the
    velocity structure fixed to the manufactured ux,uy. This way we assess
    rho in the *full* continuity equation without exploring Sx,Sy yet.
    """

    x, y, t = symbols["x"], symbols["y"], symbols["t"]
    g = problem["parameters"]["g"]  # unused but kept for completeness
    F1, _, _ = problem["F"]
    X, Y, T = meshes

    # Manufactured exact solution (for IC/BC targets and fixed velocities)
    rho_exact = problem["solution"]["rho"]
    ux_exact = problem["solution"]["ux"]
    uy_exact = problem["solution"]["uy"]

    # ------------------------------------------------------------------
    # 1) PDE residual - DISABLED for rho-only search
    # ------------------------------------------------------------------
    # For rho-only search, we ONLY use IC/BC to rank candidates
    # Any fixed velocity creates bias toward certain spatial patterns
    # In phase 2, we'll find Sx,Sy that work with the best rho
    L_pde = 0.0  # Disabled for rho-only search

    # ------------------------------------------------------------------
    # 2) IC loss for rho at t = t_start
    # ------------------------------------------------------------------
    mesh_cfg = problem["mesh"]
    t_start = mesh_cfg["t"]["start"]

    rho_ic = rho_sym.subs(t, t_start)
    ic_rho_exact, _, _ = problem["initial_conditions"]

    rho_ic_sp = sp.sympify(str(rho_ic))
    ic_rho_sp = sp.sympify(str(ic_rho_exact))

    rho_ic_fn = sp.lambdify((x, y), rho_ic_sp, "numpy")
    ic_rho_fn = sp.lambdify((x, y), ic_rho_sp, "numpy")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rho_ic_vals = rho_ic_fn(X[:, :, 0], Y[:, :, 0])
        ic_rho_vals = ic_rho_fn(X[:, :, 0], Y[:, :, 0])

    res_ic = ic_rho_vals - rho_ic_vals
    rmse_ic = float(np.sqrt(np.mean(res_ic**2)))
    if not np.isfinite(rmse_ic):
        return float("inf")

    L_ic = rmse_ic

    # ------------------------------------------------------------------
    # 3) BC loss for rho on spatial boundaries, all times
    # ------------------------------------------------------------------
    rho_sp = sp.sympify(str(rho_sym))
    rho_exact_sp = sp.sympify(str(rho_exact))

    rho_fn = sp.lambdify((x, y, t), rho_sp, "numpy")
    rho_exact_fn = sp.lambdify((x, y, t), rho_exact_sp, "numpy")

    boundary_slices = [
        (slice(0, 1), slice(None), slice(None)),
        (slice(-1, None), slice(None), slice(None)),
        (slice(None), slice(0, 1), slice(None)),
        (slice(None), slice(-1, None), slice(None)),
    ]

    bc_losses = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rho_all = rho_fn(X, Y, T)
        rho_exact_all = rho_exact_fn(X, Y, T)

    for sl in boundary_slices:
        cand_b = rho_all[sl]
        target_b = rho_exact_all[sl]

        res_bc = target_b - cand_b
        rmse_bc = float(np.sqrt(np.mean(res_bc**2)))
        if not np.isfinite(rmse_bc):
            return float("inf")

        bc_losses.append(rmse_bc)

    L_bc = float(sum(bc_losses))

    total_loss = w_pde * L_pde + w_ic * L_ic + w_bc * L_bc
    return float(total_loss)


def shallow_system_loss(
    rho_sym,
    ux_sym,
    uy_sym,
    problem,
    symbols,
    meshes,
    w_pde: float = 10000.0,
    w_ic: float = 10.0,
    w_bc: float = 10.0,
    use_combined_loss: bool = False,
):
    """Compute RMSE of shallow-water system + IC/BC residuals vs manufactured solution.

    - PDE part: same equations as in ``create_shallow_water_problem`` but with
      candidate (rho, ux, uy) instead of the exact manufactured solution.
    - IC part: enforce h, h*u, h*v at t = t_start.
    - BC part: enforce h, h*u, h*v on the spatial boundaries x/y = const using
      the manufactured exact solution from ``problem['solution']``.
    """

    x, y, t = symbols["x"], symbols["y"], symbols["t"]
    g = problem["parameters"]["g"]
    F1, F2, F3 = problem["F"]
    X, Y, T = meshes

    # Manufactured exact solution (for IC/BC targets)
    rho_exact = problem["solution"]["rho"]
    ux_exact = problem["solution"]["ux"]
    uy_exact = problem["solution"]["uy"]

    rho = rho_sym
    ux = ux_sym
    uy = uy_sym

    rho_ux = rho * ux
    rho_uy = rho * uy
    g_rho_sq = 0.5 * g * rho**2

    # ------------------------------------------------------------------
    # 1) PDE residuals on full interior mesh
    # ------------------------------------------------------------------
    eqs = [
        sm.diff(rho, t) + sm.diff(rho_ux, x) + sm.diff(rho_uy, y),
        sm.diff(rho_ux, t)
        + sm.diff(rho_ux * ux + g_rho_sq, x)
        + sm.diff(rho_ux * uy, y),
        sm.diff(rho_uy, t)
        + sm.diff(rho_ux * uy, x)
        + sm.diff(rho_uy * uy + g_rho_sq, y),
    ]
    Fs = [F1, F2, F3]

    pde_losses = []
    pde_res_arrays = []
    for eq, F in zip(eqs, Fs):
        # Convert via sympy for lambdify robustness
        eq_sp = sp.sympify(str(eq))
        F_sp = sp.sympify(str(F))

        eq_fn = sp.lambdify((x, y, t), eq_sp, "numpy")
        F_fn = sp.lambdify((x, y, t), F_sp, "numpy")

        # Suppress numerical RuntimeWarnings (e.g., invalid value in log) and
        # treat any NaNs/Infs as a very bad candidate.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eq_val = eq_fn(X, Y, T)
            F_val = F_fn(X, Y, T)

        res = F_val - eq_val
        # Keep the raw residuals for optional combined loss calculation
        try:
            pde_res_arrays.append(np.ravel(res))
        except Exception:
            # If something odd happens, fall back to converting via numpy
            import numpy as _np

            pde_res_arrays.append(_np.ravel(_np.asarray(res)))

        rmse = float(np.sqrt(np.mean(res ** 2)))

        if not np.isfinite(rmse):
            return float("inf")

        pde_losses.append(rmse)

    L_pde = float(sum(pde_losses))

    # ------------------------------------------------------------------
    # 2) Initial condition loss at t = t_start
    # ------------------------------------------------------------------
    mesh_cfg = problem["mesh"]
    t_start = mesh_cfg["t"]["start"]

    # Candidate values at t = t_start
    rho_ic = rho.subs(t, t_start)
    rho_ux_ic = (rho_ux).subs(t, t_start)
    rho_uy_ic = (rho_uy).subs(t, t_start)

    # Exact IC from problem dict
    ic_rho_exact, ic_rho_ux_exact, ic_rho_uy_exact = problem["initial_conditions"]

    ic_candidates = [rho_ic, rho_ux_ic, rho_uy_ic]
    ic_targets = [ic_rho_exact, ic_rho_ux_exact, ic_rho_uy_exact]

    ic_losses = []
    ic_res_arrays = []
    for cand, target in zip(ic_candidates, ic_targets):
        cand_sp = sp.sympify(str(cand))
        target_sp = sp.sympify(str(target))

        cand_fn = sp.lambdify((x, y), cand_sp, "numpy")
        target_fn = sp.lambdify((x, y), target_sp, "numpy")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Slice t = t_start plane from meshes
            ic_vals = cand_fn(X[:, :, 0], Y[:, :, 0])
            ic_targets_vals = target_fn(X[:, :, 0], Y[:, :, 0])

        res_ic = ic_targets_vals - ic_vals

        try:
            ic_res_arrays.append(np.ravel(res_ic))
        except Exception:
            import numpy as _np

            ic_res_arrays.append(_np.ravel(_np.asarray(res_ic)))

        rmse_ic = float(np.sqrt(np.mean(res_ic ** 2)))

        if not np.isfinite(rmse_ic):
            return float("inf")

        ic_losses.append(rmse_ic)

    L_ic = float(sum(ic_losses))

    # ------------------------------------------------------------------
    # 3) Boundary condition loss on spatial boundaries, all times
    # ------------------------------------------------------------------
    # We enforce boundaries for rho, rho*ux, rho*uy using manufactured solution
    rho_exact_ux = rho_exact * ux_exact
    rho_exact_uy = rho_exact * uy_exact

    bc_candidates = [rho, rho_ux, rho_uy]
    bc_targets = [rho_exact, rho_exact_ux, rho_exact_uy]

    # Pre-compute boundary index slices: x=0, x=-1, y=0, y=-1 (all times)
    boundary_slices = [
        (slice(0, 1), slice(None), slice(None)),   # x=min
        (slice(-1, None), slice(None), slice(None)),  # x=max
        (slice(None), slice(0, 1), slice(None)),   # y=min
        (slice(None), slice(-1, None), slice(None)),  # y=max
    ]

    bc_losses = []
    bc_res_arrays = []
    for cand, target in zip(bc_candidates, bc_targets):
        cand_sp = sp.sympify(str(cand))
        target_sp = sp.sympify(str(target))

        cand_fn = sp.lambdify((x, y, t), cand_sp, "numpy")
        target_fn = sp.lambdify((x, y, t), target_sp, "numpy")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cand_all = cand_fn(X, Y, T)
            target_all = target_fn(X, Y, T)

        for sl in boundary_slices:
            cand_b = cand_all[sl]
            target_b = target_all[sl]

            res_bc = target_b - cand_b
            try:
                bc_res_arrays.append(np.ravel(res_bc))
            except Exception:
                import numpy as _np

                bc_res_arrays.append(_np.ravel(_np.asarray(res_bc)))

            rmse_bc = float(np.sqrt(np.mean(res_bc ** 2)))

            if not np.isfinite(rmse_bc):
                return float("inf")

            bc_losses.append(rmse_bc)

    L_bc = float(sum(bc_losses))

    # Optionally compute a single combined RMSE across PDE / IC / BC residuals
    if use_combined_loss:
        try:
            total_loss = combined_rmse_loss(pde_res_arrays + ic_res_arrays + bc_res_arrays)
        except Exception:
            # If combined loss computation fails for any reason, fallback to
            # the original weighted combination to avoid hard breaks.
            total_loss = w_pde * L_pde + w_ic * L_ic + w_bc * L_bc
    else:
        total_loss = w_pde * L_pde + w_ic * L_ic + w_bc * L_bc

    return float(total_loss)


def main(
    # Slightly increase rho samples; spatial samples unused when rho_only=True
    n_rho_samples: int = 1500,
    n_spatial_samples: int = 800,
    rho_start: int = 0,
    rho_end: int | None = None,
    seed: int = 123,
    max_spatial_tries: int = 20,
    target_rmse: float = 1.0,
    # If True, only perform rho sampling + rho-only loss and skip Sx,Sy search
    rho_only: bool = False,
    # Rho-focused weighting: emphasize IC/BC so rho matches manufactured bump
    w_pde: float = 1.0,
    w_ic: float = 1.0,
    w_bc: float = 1.0,
    use_combined_loss: bool = False,
    use_numeric_matching: bool = False,
    rho_time_pattern: str | None = None,
):
    # === REPRODUCIBILITY: Seed all RNG sources ===
    random.seed(seed)
    np.random.seed(seed)
    
    # Environment variables for SIGS setup (reuse your usual pattern)
    os.environ.setdefault("LODE_CONFIG", "configs/config.yaml")
    os.environ.setdefault("LODE_H5", "data/expressions.h5")

    # 1) Problem + symbols/meshes
    shallow_problem = create_shallow_water_problem()
    # Bump mesh resolution for better discrimination (was coarse 16^3 during debug)
    # Increase to 24 points per dimension (x,y,t) for this sampling run
    try:
        shallow_problem['mesh'] = {
            'x': {'start': shallow_problem['mesh']['x']['start'], 'end': shallow_problem['mesh']['x']['end'], 'points': 128},
            'y': {'start': shallow_problem['mesh']['y']['start'], 'end': shallow_problem['mesh']['y']['end'], 'points': 128},
            't': {'start': shallow_problem['mesh']['t']['start'], 'end': shallow_problem['mesh']['t']['end'], 'points': 128},
        }
    except Exception:
        # If problem dict uses a different layout, fall back to the original setup
        pass
    symbols, meshes = setup_shallow_symbols_meshes(shallow_problem)

    # Pull commonly used symbol and mesh variables into local scope
    x, y, t = symbols["x"], symbols["y"], symbols["t"]
    X, Y, T = meshes

    # 2) Sampler
    sampler = FlexibleVectorSampler(
        cluster_file="data/clusters.pkl",
    )

    # 3) Sample rho ansatz: use two SPATIOTEMPORAL clusters (distinct instances)
    # and six temporal clusters. This biases sampling toward combinations of
    # wave + Gaussian (one cluster) and complementary spatial-temporal patterns
    # in the other cluster with diverse temporal structures.
    categories_rho = {
        MathClass.SPATIOTEMPORAL_3D: 2,   # keep 2 big subclusters for spatiotemporal
        MathClass.TEMPORAL_1D: 6,         # 6 temporal subclusters for diverse sampling
    }
    category_instances_rho = {
        MathClass.SPATIOTEMPORAL_3D: 2,   # two spatial-temporal instances per rho
        MathClass.TEMPORAL_1D: 1,         # one temporal factor per expression
    }

    print("Sampling rho candidates ...")
    # Request distinct instances for the two spatiotemporal factors so they come
    # from different subclusters (avoids duplicating the same pattern twice).
    rho_sample_id = sampler.sample_from_subclusters(
        categories=categories_rho,
        category_instances=category_instances_rho,
        n_samples=n_rho_samples,
        operator="*",
        seed=seed,
        model=None,
        distinct_instances=True,
    )

    rho_exprs, rho_vecs, rho_sub_idxs, rho_expr_idxs = sampler.get_sampling_results(rho_sample_id)
    print("rho_sample_id:", rho_sample_id)
    print("# raw rho exprs:", len(rho_exprs))

    # Optional: filter by first constant like in damped.py
    filtered = ExpressionUtils.filter_by_first_const(rho_exprs, min_val=0.1)
    if not filtered:
        print("No rho expressions after filtering; aborting.")
        return
    _, rho_candidates = zip(*filtered)
    rho_candidates = list(rho_candidates)
    
    # Store subcluster IDs for each candidate (for adaptive resampling)
    # rho_sub_idxs is a list of tuples: [(temporal_subcluster, st1_subcluster, st2_subcluster), ...]
    rho_subcluster_map = {}
    for idx, (expr, sub_ids) in enumerate(zip(rho_exprs, rho_sub_idxs)):
        if expr in rho_candidates:
            rho_subcluster_map[str(expr)] = sub_ids
    
    print(f"# filtered rho exprs: {len(rho_candidates)}")

    # Optional: filter rho candidates by a regex pattern (e.g. require
    # an exponential-time factor like `exp(...t)`). This lets the user
    # softly steer the search toward particular temporal forms without
    # changing the cluster-only sampling mechanism.
    if rho_time_pattern is not None:
        import re
        try:
            pat = re.compile(rho_time_pattern)
            rho_candidates = [r for r in rho_candidates if pat.search(str(r))]
            print(f"After rho_time_pattern filter '{rho_time_pattern}': {len(rho_candidates)} candidates")
        except Exception as e:
            print("Invalid rho_time_pattern, ignoring it:", e)

    # Restrict to a slice of rho candidates for parallel runs
    total_rhos = len(rho_candidates)
    if rho_end is None or rho_end > total_rhos:
        rho_end = total_rhos
    if rho_start < 0:
        rho_start = 0
    if rho_start >= rho_end:
        print(f"Empty rho slice: rho_start={rho_start}, rho_end={rho_end}; nothing to do.")
        return

    rho_slice = rho_candidates[rho_start:rho_end]
    print(f"Using rho candidates slice [{rho_start}, {rho_end}) of {total_rhos} total")

    # 4) First: score rho candidates with rho-focused loss
    rho_scores = []
    print("\nScoring rho candidates with rho-focused loss ...")
    best_so_far = float("inf")
    for global_idx, rho_str in enumerate(tqdm(rho_slice, desc="rho-only", ncols=80), start=rho_start):
        try:
            rho_sym = sm.sympify(rho_str)
        except Exception as e:
            print("Skipping invalid rho expr for rho-only loss:", rho_str, "error:", e)
            continue

        try:
            loss_rho = shallow_rho_loss(
                rho_sym,
                problem=shallow_problem,
                symbols=symbols,
                meshes=meshes,
                # Rho-only loss: emphasize IC/BC for rho selection
                w_pde=w_pde,
                w_ic=w_ic,
                w_bc=w_bc,
            )
        except Exception as e:
            print("Error evaluating rho-only loss:", e)
            continue

        if np.isfinite(loss_rho):
            # Get subcluster IDs for this expression
            sub_ids = rho_subcluster_map.get(rho_str, None)
            rho_scores.append((loss_rho, rho_str, sub_ids))
            if loss_rho < best_so_far:
                best_so_far = loss_rho
                tqdm.write(f"[rho-only] new best loss {best_so_far:.4e} at candidate {global_idx}")
                if sub_ids:
                    tqdm.write(f"           subclusters: {sub_ids}")

    if not rho_scores:
        print("No valid rho candidates after rho-only scoring.")
        return

    # Sort rho by their rho-only loss (best first)
    rho_scores.sort(key=lambda x: x[0])
    print("\nTop 5 rho candidates by rho-only loss:")
    for rank, (score, r_str, sub_ids) in enumerate(rho_scores[:5], start=1):
        print(f"  #{rank}: loss={score:.4e}, subclusters={sub_ids}")
        print(f"         rho={r_str}")

    # Decide how many top rhos to refine with full system loss
    n_rho_refine = min(1, len(rho_scores))
    top_rho_list = rho_scores[:n_rho_refine]

    # If requested, stop after reporting top rho candidates
    if rho_only:
        print("\n[RHO-ONLY MODE] Skipping Sx,Sy search. Top rho candidates by rho-only loss:")
        for rank, (score, r_str, sub_ids) in enumerate(rho_scores[:10], start=1):
            print(f"  #{rank}: loss={score:.4e}, subclusters={sub_ids}")
            print(f"         rho={r_str}")

        # Optional: dump per-worker summary for later aggregation of global minimum
        worker_id = os.environ.get("WORKER_ID", "0")
        out_name = os.environ.get("RHO_WORKER_OUT", f"rho_worker_{worker_id}.json")

        summary = {
            "worker_id": worker_id,
            "rho_start": rho_start,
            "rho_end": rho_end,
            "total_rhos": int(total_rhos),
            "top_rhos": [
                {
                    "rank": int(rank), 
                    "loss": float(score), 
                    "rho": str(r_str),
                    "subclusters": sub_ids if sub_ids else None
                }
                for rank, (score, r_str, sub_ids) in enumerate(rho_scores[:10], start=1)
            ],
        }

        try:
            with open(out_name, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[RHO-ONLY MODE] Saved worker summary to {out_name}")
        except Exception as e:
            print("[RHO-ONLY MODE] Failed to write worker summary:", e)
        return

    # 5) For each *good* rho, sample many Sx,Sy and evaluate full system loss.
    categories_spatial = {MathClass.SPATIAL_2D: 100}

    best_loss = float("inf")
    best_triple = None  # (rho_str, Sx_str, Sy_str)

    start_time = time.time()

    for rank, (rho_loss, rho_str, sub_ids) in enumerate(top_rho_list, start=1):
        print(f"\n=== Refining rho rank #{rank}/{n_rho_refine} (rho-only loss={rho_loss:.4e}, subclusters={sub_ids}) ===")
        try:
            rho_sym = sm.sympify(rho_str)
        except Exception as e:
            print("Skipping invalid rho expr in refinement:", rho_str, "error:", e)
            continue

        # Sample spatial expressions once per rho
        spatial_id = sampler.sample_from_subclusters(
            categories=categories_spatial,
            n_samples=n_spatial_samples,
            operator="*",
            seed=seed + rank,  # vary seed per rho
            model=None,
        )
        spatial_exprs, *_ = sampler.get_sampling_results(spatial_id)

        # Filter spatial expressions by first constant to avoid extremely small
        # leading coefficients (e.g., 1e-3*exp(...)). The user requested we avoid
        # tiny prefactors, especially when multiplied by exponentials, so we
        # increase the minimum first-constant threshold slightly and then apply
        # an additional reject rule for small constants in front of `exp(..)`.
        filter_first_const_min = 0.5
        spatial_filtered = ExpressionUtils.filter_by_first_const(spatial_exprs, min_val=filter_first_const_min)
        if len(spatial_filtered) < 2:
            print("Not enough spatial expressions for Sx,Sy after coeff filter; continuing.")
            continue
        _, spatial_filtered_exprs = zip(*spatial_filtered)

        # Additional pruning: reject expressions of the form `c*exp(...)` when
        # the leading constant c is small (< filter_first_const_min). Also avoid
        # pure leading `exp(...)` if it has an explicit very small multiplier.
        import re
        cleaned_spatial = []
        leading_const_re = re.compile(r"^\s*([0-9.eE+-]+)\s*\*\s*(.*)$")
        for s in spatial_filtered_exprs:
            s_strip = s.strip()
            m = leading_const_re.match(s_strip)
            reject = False
            if m:
                try:
                    c = float(m.group(1))
                except Exception:
                    c = None
                rest = m.group(2)
                if c is not None and c < filter_first_const_min and "exp(" in rest:
                    reject = True
            else:
                # If no leading multiplicative constant and expression begins
                # with exp(...), keep it (user did not forbid pure exp), but
                # if it's likely to be numerically tiny we prefer to keep only
                # those with explicit constants — here we keep it but it's
                # already subject to later numeric prechecks.
                reject = False

            if not reject:
                cleaned_spatial.append(s)

        if len(cleaned_spatial) < 2:
            print("Not enough spatial expressions after exp-prefactor pruning; continuing.")
            continue
        spatial_filtered_exprs = cleaned_spatial

        # Keep variable-dominance and coefficient filtering, but drop the
        # explicit 'radial' string heuristic so we don't force 'sqrt(' or
        # gaussian-like `exp(-(...))` forms. This widens the candidate pool
        # while still preferring Sx to be x-dominant and Sy to be y-dominant.
        def var_dominance_ok(expr_str: str, primary: str, secondary: str) -> bool:
            """Determine if `primary` appears noticeably more than `secondary`.

            Use regex word-boundary matching to avoid counting letters that are
            part of other names (e.g., the 'x' in 'exp'). This is a minimal fix
            that keeps the original intent while avoiding false positives.
            """
            import re
            # match primary/secondary as standalone tokens (not part of identifiers)
            p_pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(primary) + r'(?![A-Za-z0-9_])')
            s_pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(secondary) + r'(?![A-Za-z0-9_])')
            cx = len(p_pat.findall(expr_str))
            cy = len(s_pat.findall(expr_str))
            return cx >= (cy + 1)

        # Accept any expression that contains the relevant variable, but keep
        # the var-dominance constraint to preserve directional bias.
        sx_candidates = [s for s in spatial_filtered_exprs if ("x" in s) and var_dominance_ok(s, 'x', 'y')]
        sy_candidates = [s for s in spatial_filtered_exprs if ("y" in s) and var_dominance_ok(s, 'y', 'x')]

        if len(sx_candidates) < 1 or len(sy_candidates) < 1:
            print("Not enough structurally valid spatial exprs for Sx/Sy; continuing.")
            continue

        # Prepare holders for local best pair (shared by numeric-matching and
        # random-sampling branches)
        local_best_loss = float("inf")
        local_best_Sx = None
        local_best_Sy = None

        # Optionally perform numeric matching of Sx/Sy by implied amplitude A(r).
        # This matching is a stronger (harder) bias toward radial-like pairs. To
        # avoid imposing that bias by default we make it optional via
        # `use_numeric_matching`. When disabled we sample Sx/Sy pairs uniformly
        # at random from the candidate pools.
        if use_numeric_matching:
            print("Attempting numeric matching of Sx/Sy by implied amplitude A(r) ...")
            # Prepare lambdified functions for radial candidates
            sx_fns = {}
            sy_fns = {}
            eps = 1e-6
            sample_pts_x = None
            sample_pts_y = None
            try:
                # create sample points: flatten interior plane t=0 (avoid boundaries)
                xs_all = X[1:-1, 1:-1, 0].ravel()
                ys_all = Y[1:-1, 1:-1, 0].ravel()
                # choose up to 200 random sample points
                idxs = np.arange(xs_all.size)
                np.random.seed(seed + rank)
                if xs_all.size > 200:
                    idxs = np.random.choice(idxs, 200, replace=False)
                sample_pts_x = xs_all[idxs]
                sample_pts_y = ys_all[idxs]
            except Exception:
                # fallback: small 3x3 block
                sample_pts_x = X[:3, :3, 0].ravel()
                sample_pts_y = Y[:3, :3, 0].ravel()

            for s in sx_candidates:
                try:
                    s_sp = sp.sympify(str(s))
                    sx_fns[s] = sp.lambdify((x, y), s_sp, "numpy")
                except Exception:
                    sx_fns[s] = None
            for s in sy_candidates:
                try:
                    s_sp = sp.sympify(str(s))
                    sy_fns[s] = sp.lambdify((x, y), s_sp, "numpy")
                except Exception:
                    sy_fns[s] = None

            matched_pairs = []  # tuples (rel_rmse, Sx_str, Sy_str)
            for sx in sx_candidates:
                sx_fn = sx_fns.get(sx)
                if sx_fn is None:
                    continue
                try:
                    svals_x = sx_fn(sample_pts_x, sample_pts_y)
                except Exception:
                    continue
                # mask where x near zero to avoid division blowups
                mask_x = np.abs(sample_pts_x) > 1e-3
                if not np.any(mask_x):
                    continue
                a1 = np.zeros_like(svals_x, dtype=float)
                a1[:] = np.nan
                a1[mask_x] = svals_x[mask_x] / sample_pts_x[mask_x]

                for sy in sy_candidates:
                    sy_fn = sy_fns.get(sy)
                    if sy_fn is None:
                        continue
                    try:
                        svals_y = sy_fn(sample_pts_x, sample_pts_y)
                    except Exception:
                        continue
                    mask_y = np.abs(sample_pts_y) > 1e-3
                    common_mask = mask_x & mask_y
                    if not np.any(common_mask):
                        continue
                    a2 = np.zeros_like(svals_y, dtype=float)
                    a2[:] = np.nan
                    a2[common_mask] = svals_y[common_mask] / sample_pts_y[common_mask]

                    # compute relative RMSE on common_mask (ignore NaNs)
                    valid = common_mask
                    if not np.any(valid):
                        continue
                    d = a1[valid] - a2[valid]
                    num = np.sqrt(np.nanmean(d**2))
                    den = 1.0 + np.sqrt(np.nanmean(a2[valid]**2))
                    rel_rmse = num / den
                    if np.isfinite(rel_rmse) and rel_rmse < 0.2:
                        matched_pairs.append((float(rel_rmse), sx, sy))
            matched_pairs.sort(key=lambda x: x[0])
            print(f"Found {len(matched_pairs)} numeric-matched Sx/Sy pairs (rel_rmse<0.2).")

            local_best_loss = float("inf")
            local_best_Sx = None
            local_best_Sy = None

            # Form up to max_spatial_tries pairs from the separate Sx / Sy pools
            # Build the list of pairs to test. To avoid a hard bias we sample a
            # mixture of numerically-matched pairs and random pairs. Fraction
            # p_bias of the budget is drawn from the matched set (softly) and the
            # rest is random. This keeps exploration while preferring likely
            # radial pairs.
            pairs_to_test = []
            used = set()
            p_bias = 0.3  # fraction of pairs drawn from matched set (soft bias)
            gamma = 10.0  # softness for weighting (higher -> stronger preference)

            if matched_pairs:
                # matched_pairs is list of (rel_rmse, sx, sy) sorted ascending
                rels = np.array([m[0] for m in matched_pairs], dtype=float)
                # convert rel -> weights via a softmax-like transform
                weights = np.exp(-gamma * rels)
                weights = weights / (weights.sum() + 1e-12)
                n_bias = int(max(1, round(p_bias * max_spatial_tries)))
                n_bias = min(n_bias, len(matched_pairs))
                # sample without replacement according to weights
                try:
                    chosen_idx = np.random.choice(len(matched_pairs), size=n_bias, replace=False, p=weights)
                except Exception:
                    # fallback deterministic: take top-n_bias
                    chosen_idx = np.arange(n_bias)
                for ci in chosen_idx:
                    _, sx, sy = matched_pairs[int(ci)]
                    pairs_to_test.append((sx, sy))
                    used.add((sx, sy))

            # fill remaining slots with random shuffled Cartesian pairs
            if len(pairs_to_test) < max_spatial_tries:
                all_pairs = list(itertools.product(sx_candidates, sy_candidates))
                all_pairs = [p for p in all_pairs if p not in used]
                random.shuffle(all_pairs)
                remaining = max_spatial_tries - len(pairs_to_test)
                pairs_to_test.extend(all_pairs[:remaining])
        else:
            # Numeric matching disabled: sample Sx/Sy pairs uniformly at random
            print("Numeric-matching disabled; sampling random Sx/Sy pairs from candidate pools.")
            all_pairs = list(itertools.product(sx_candidates, sy_candidates))
            random.shuffle(all_pairs)
            pairs_to_test = all_pairs[:max_spatial_tries]

        n_pairs = len(pairs_to_test)

        for j, (Sx_str, Sy_str) in enumerate(pairs_to_test):

            # Skip identical Sx/Sy (we want distinct directional factors)
            if Sx_str == Sy_str:
                continue

            try:
                Sx_sym = sm.sympify(Sx_str)
                Sy_sym = sm.sympify(Sy_str)
            except Exception as e:
                print("Skipping invalid Sx/Sy exprs:", e)
                continue

            # Quick numeric pre-check: evaluate Sx,Sy on a small slice of the mesh
            # to filter NaN/Inf or wildly large values before expensive loss eval.
            try:
                Sx_sp = sp.sympify(str(Sx_sym))
                Sy_sp = sp.sympify(str(Sy_sym))
                Sx_fn = sp.lambdify((x, y), Sx_sp, "numpy")
                Sy_fn = sp.lambdify((x, y), Sy_sp, "numpy")

                xs = X[:3, :3, 0]
                ys = Y[:3, :3, 0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    svals_x = Sx_fn(xs, ys)
                    svals_y = Sy_fn(xs, ys)

                # Flatten and check finiteness and magnitude
                if not (np.isfinite(svals_x).all() and np.isfinite(svals_y).all()):
                    # reject numerically unstable candidate
                    continue
                if np.nanmax(np.abs(svals_x)) > 1e3 or np.nanmax(np.abs(svals_y)) > 1e3:
                    continue
            except Exception:
                # any error -> skip this pair
                continue

            ux_sym = rho_sym * Sx_sym
            uy_sym = rho_sym * Sy_sym

            try:
                loss = shallow_system_loss(
                    rho_sym,
                    ux_sym,
                    uy_sym,
                    problem=shallow_problem,
                    symbols=symbols,
                    meshes=meshes,
                    w_pde=w_pde,
                    w_ic=w_ic,
                        w_bc=w_bc,
                        use_combined_loss=use_combined_loss,
                )
            except Exception as e:
                print("Error evaluating system loss for this triple:", e)
                continue

            print(f"Loss for rho-rank #{rank}, pair {j+1}/{n_pairs}: {loss:.4e}")

            if loss < local_best_loss:
                local_best_loss = loss
                local_best_Sx = Sx_str
                local_best_Sy = Sy_str

        if local_best_loss < best_loss and local_best_Sx is not None:
            best_loss = local_best_loss
            best_triple = (rho_str, local_best_Sx, local_best_Sy)
            print("New best loss!", best_loss)
            print("  rho(x,y,t):", rho_str)
            print("  Sx(x,y):   ", local_best_Sx)
            print("  Sy(x,y):   ", local_best_Sy)

        # NOTE: we no longer early-stop on target_rmse; we want to explore
        # multiple rho structures and many Sx,Sy for each.

    elapsed = time.time() - start_time
    print("\n==== SEARCH FINISHED ====")
    print(f"Elapsed time (refinement phase): {elapsed:.2f} s")
    if best_triple is None:
        print("No valid triple found.")
        return

    rho_str, Sx_str, Sy_str = best_triple
    print(f"Best loss: {best_loss:.6e}")
    print("Best rho(x,y,t):", rho_str)
    print("Best Sx(x,y):   ", Sx_str)
    print("Best Sy(x,y):   ", Sy_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rho_samples", type=int, default=1500)
    parser.add_argument("--rho_start", type=int, default=0)
    parser.add_argument("--rho_end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--use_combined_loss", action="store_true", help="Use a single combined RMSE across PDE/IC/BC computed with JAX if available")
    args = parser.parse_args()
    
    main(
        n_rho_samples=args.n_rho_samples,
        rho_start=args.rho_start,
        rho_end=args.rho_end,
        seed=args.seed,
        use_combined_loss=args.use_combined_loss,
    )
