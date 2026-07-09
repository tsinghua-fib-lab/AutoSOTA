"""QP solver configuration, selection, and KKT multiplier computation."""

import numpy as np
import qpsolvers
import scipy.sparse as spa
import scipy.sparse.linalg as spla

_DUAL_AVAILABLE_SOLVERS = frozenset([
    "clarabel", "cvxopt", "daqp", "ecos", "gurobi", "highs", "hpipm",
    "mosek", "osqp", "piqp", "proxqp", "qpalm", "qpoases", "qpswift", "quadprog", "scs"
])


# Solver tolerance configuration
def set_solver_tolerance(qp_solver_keywords, qp_solver, eps_abs, eps_rel):
    if eps_abs is None or eps_rel is None:
        print("If a tolerance is None, leaves empty and lets the solvers choose their default.")

    if qp_solver ==  "clarabel":
        if eps_abs is not None:
            qp_solver_keywords["tol_feas"] = eps_abs
            qp_solver_keywords["tol_gap_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["tol_gap_rel"] = eps_rel
    elif qp_solver == "cvxopt":
        if eps_abs is not None:
            qp_solver_keywords["feastol"] = eps_abs
    elif qp_solver == "daqp":
        if eps_abs is not None:
            qp_solver_keywords["dual_tol"] = eps_abs
            qp_solver_keywords["primal_tol"] = eps_abs
    elif qp_solver == "ecos":
        if eps_abs is not None:
            qp_solver_keywords["feastol"] = eps_abs
    elif qp_solver == "gurobi":
        if eps_abs is not None:
            qp_solver_keywords["FeasibilityTol"] = eps_abs
            qp_solver_keywords["OptimalityTol"] = eps_abs
    elif qp_solver == "highs":
        if eps_abs is not None:
            qp_solver_keywords["dual_feasibility_tolerance"] = eps_abs
    elif qp_solver == "hpipm":
        if eps_abs is not None:
            qp_solver_keywords["tol_comp"] = eps_abs
            qp_solver_keywords["tol_stat"] = eps_abs
            qp_solver_keywords["tol_eq"] = eps_abs
            qp_solver_keywords["tol_ineq"] = eps_abs
    elif qp_solver == "osqp":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "piqp":
        qp_solver_keywords["check_duality_gap"] = True
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
            qp_solver_keywords["eps_duality_gap_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_duality_gap_rel"] = eps_rel
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "proxqp":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
            qp_solver_keywords["eps_duality_gap_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_duality_gap_rel"] = eps_rel
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "qpalm":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_rel"] = eps_rel
    elif qp_solver == "qpswift":
        if eps_abs is not None:
            qp_solver_keywords["RELTOL"] = eps_abs * np.sqrt(3.0) # TODO : check? this is what qpbenchmarkhas in solver_settings.py Line 116
    elif qp_solver == "scs":
        if eps_abs is not None:
            qp_solver_keywords["eps_abs"] = eps_abs
        if eps_rel is not None:
            qp_solver_keywords["eps_rel"] = eps_rel

    return qp_solver_keywords


# Default QP solver selection
def _select_default_qp_solver(solve_type):
    """Select default QP solver based on solve_type.
    
    - dense: prefer cvxopt, else first available dense solver
    - sparse: prefer gurobi, else first available sparse solver
    """
    if solve_type == "dense":
        if "cvxopt" in qpsolvers.dense_solvers:
            return "cvxopt"
        if len(qpsolvers.dense_solvers) > 0:
            return qpsolvers.dense_solvers[0]
        # Fallback: try any available solver
        if qpsolvers.available_solvers:
            return qpsolvers.available_solvers[0]
        raise RuntimeError("No QP solver available! Install at least one (e.g. pip install cvxopt).")
    else:  # sparse
        if "gurobi" in qpsolvers.sparse_solvers:
            return "gurobi"
        if len(qpsolvers.sparse_solvers) > 0:
            return qpsolvers.sparse_solvers[0]
        if qpsolvers.available_solvers:
            return qpsolvers.available_solvers[0]
        raise RuntimeError("No QP solver available! Install at least one (e.g. pip install gurobipy).")


# KKT-based multiplier recovery
def _compute_multipliers_from_kkt(Q_np, q_np, G_np, h_np, A_np, x_star_np,
                                   has_eq, use_sparse, nIneq, nEq, eps_active=1e-5):
    """
    Compute Lagrange multipliers from KKT stationarity when the QP solver doesn't return them.
    
    KKT stationarity: Q x* + q + G^T nu* + A^T mu* = 0
    With nu_i = 0 for inactive constraints.
    
    Solves: [G_active^T, A^T] @ [nu_active; mu] = -(Q x* + q)
    in a least-squares sense, then enforces nu >= 0 (dual feasibility).
    """
    # Residual from stationarity: r = -(Q x* + q)
    r = -(Q_np @ x_star_np + q_np)

    # Identify active inequality constraints
    r_pri = h_np - G_np @ x_star_np
    active_mask = r_pri < eps_active
    nActive = int(np.sum(active_mask))

    nu_star_np = np.zeros(nIneq)
    mu_star_np = np.zeros(nEq) if has_eq else None

    if nActive == 0 and (not has_eq or nEq == 0):
        return nu_star_np, mu_star_np

    # Build constraint matrix and solve
    if use_sparse:
        blocks = []
        if nActive > 0:
            blocks.append(G_np[active_mask, :].T)
        if has_eq and nEq > 0:
            blocks.append(A_np.T)
        if len(blocks) == 0:
            return nu_star_np, mu_star_np
        C = spa.hstack(blocks, format='csc')
        y = spla.lsqr(C, r)[0]
    else:
        blocks = []
        if nActive > 0:
            blocks.append(G_np[active_mask, :].T)  # dim x nActive
        if has_eq and nEq > 0:
            blocks.append(A_np.T)  # dim x nEq
        if len(blocks) == 0:
            return nu_star_np, mu_star_np
        C = np.hstack(blocks)
        y, _, _, _ = np.linalg.lstsq(C, r, rcond=None)

    # Extract nu_active and mu
    offset = 0
    if nActive > 0:
        nu_active = y[offset:offset + nActive]
        nu_active = np.maximum(nu_active, 0.0)  # Dual feasibility: nu >= 0
        nu_star_np[active_mask] = nu_active
        offset += nActive
    if has_eq and nEq > 0:
        mu_star_np = y[offset:offset + nEq]

    return nu_star_np, mu_star_np
