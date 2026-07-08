import os
import copy
from concurrent.futures import ThreadPoolExecutor

from contextlib import contextmanager
from threadpoolctl import threadpool_limits

@contextmanager
def _limit_native_threads(n: int = 1):
    if threadpool_limits is None:
        yield
    else:
        with threadpool_limits(limits=n):
            yield

import cvxpy as cp
import numpy as np
import torch

try:
    from cvxtorch import TorchExpression
except Exception:
    TorchExpression = None

def _require_cvxtorch():
    if TorchExpression is None:
        raise ImportError(
            "cvxtorch is required for this feature. Install it with:\n"
            "  pip install git+https://github.com/cvxpy/cvxtorch.git"
        )

from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC

from .utils import to_numpy, to_torch, slice_params_for_batch


@torch.no_grad()
def _compare_grads(params_req, grads, ground_truth_grads):
    est_chunks, gt_chunks = [], []
    for p, ge, gg in zip(params_req, grads, ground_truth_grads):
        ge = torch.zeros_like(p) if ge is None else ge.detach()
        gg = torch.zeros_like(p) if gg is None else gg.detach()
        est_chunks.append(ge.reshape(-1))
        gt_chunks.append(gg.reshape(-1))
    est = torch.cat(est_chunks)
    gt = torch.cat(gt_chunks)
    eps = 1e-12
    denom = (est.norm() * gt.norm()).clamp_min(eps)
    cos_sim = torch.dot(est, gt) / denom
    l2_diff = (est - gt).norm()
    return cos_sim, l2_diff


def _cvx_sum_or_zero(terms):
    return cp.sum(terms) if len(terms) > 0 else cp.Constant(0.0)


def _has_pnorm_atom(expr) -> bool:
    try:
        nm_fn = getattr(expr, "name", None)
        if callable(nm_fn):
            nm = nm_fn()
            if nm in {"pnorm", "norm1", "norm_inf"}:
                return True
    except Exception:
        pass

    try:
        cls = expr.__class__.__name__.lower()
        if cls in {"pnorm", "norm1", "norminf", "norm_inf"}:
            return True
    except Exception:
        pass

    for a in getattr(expr, "args", []) or []:
        if _has_pnorm_atom(a):
            return True
    return False


def _infer_objective_expr(problem: cp.Problem):
    obj = problem.objective
    if isinstance(obj, cp.Minimize):
        return obj.expr
    if isinstance(obj, cp.Maximize):
        return -obj.expr
    expr = getattr(obj, "expr", None)
    if expr is None:
        raise ValueError("Unsupported objective type; expected Minimize/Maximize.")
    return expr


def _expcone_dual_dot(u_triplet, c: ExpCone):
    ux, uy, uz = u_triplet
    x, y, z = c.args
    return cp.sum(cp.multiply(ux, x)) + cp.sum(cp.multiply(uy, y)) + cp.sum(cp.multiply(uz, z))


def _split_expcone_dual_value(dv, shapes3):
    if isinstance(dv, (list, tuple)) and len(dv) == 3:
        out = [np.asarray(d, dtype=float) for d in dv]
        for k in range(3):
            if tuple(out[k].shape) != tuple(shapes3[k]):
                if out[k].size == int(np.prod(shapes3[k])):
                    out[k] = out[k].reshape(shapes3[k])
                else:
                    raise ValueError(f"ExpCone dual block {k} shape mismatch: got {out[k].shape}, expected {shapes3[k]}")
        return out

    dv_arr = np.asarray(dv, dtype=float)

    if dv_arr.ndim >= 1 and dv_arr.shape[-1] == 3:
        base = dv_arr.shape[:-1]
        if tuple(base) == tuple(shapes3[0]) and tuple(base) == tuple(shapes3[1]) and tuple(base) == tuple(shapes3[2]):
            return [dv_arr[..., k].reshape(shapes3[k]) for k in range(3)]

    block = int(np.prod(shapes3[0]))
    if int(dv_arr.size) == 3 * block and int(np.prod(shapes3[1])) == block and int(np.prod(shapes3[2])) == block:
        tmp = dv_arr.reshape((block, 3))
        return [tmp[:, k].reshape(shapes3[k]) for k in range(3)]

    raise ValueError(
        f"Cannot parse ExpCone dual_value with shape {dv_arr.shape} into 3 blocks of shapes {shapes3}."
    )


def _active_counts_one(b, ctx, i: int, tol: float):
    out = {}

    out["eq"] = sum(int(np.prod(f.shape)) for f in b["eq_functions"])

    out["ineq"] = sum(
        int(np.sum(np.asarray(ctx.scalar_ineq_slack[j][i]) <= tol))
        for j in range(len(b["scalar_ineq_functions"]))
    )

    soc_cnt = 0
    for c in b["soc_constraints"]:
        t_val = c.args[0].expr.value
        x_val = c.args[1].expr.value
        if t_val is None or x_val is None:
            continue

        t = np.asarray(t_val, dtype=float).reshape(-1)   # (k,) or (1,)
        x = np.asarray(x_val, dtype=float)

        if t.size == 1:
            soc_cnt += int((t.item() - np.linalg.norm(x.ravel())) <= tol)
        else:
            if x.ndim == 1:
                norms = np.full(t.size, np.linalg.norm(x.ravel()))
            elif x.shape[-1] == t.size:
                norms = np.linalg.norm(x.reshape(-1, t.size), axis=0)
            elif x.shape[0] == t.size:
                norms = np.linalg.norm(x.reshape(t.size, -1), axis=1)
            else:
                flat = x.ravel()
                if flat.size % t.size == 0:
                    norms = np.linalg.norm(flat.reshape(t.size, -1), axis=1)
                else:
                    norms = np.full(t.size, np.linalg.norm(flat))

            soc_cnt += int(np.sum((t - norms) <= tol))
    out["soc"] = soc_cnt

    exp_cnt = 0
    for c in b["exp_cones"]:
        x_val, y_val, z_val = (c.args[0].value, c.args[1].value, c.args[2].value)
        if x_val is None or y_val is None or z_val is None:
            continue

        xv = np.asarray(x_val, dtype=float).reshape(-1)  # (k,) or (1,)
        yv = np.asarray(y_val, dtype=float).reshape(-1)
        zv = np.asarray(z_val, dtype=float).reshape(-1)

        k = max(xv.size, yv.size, zv.size)

        # broadcast scalars to vector length k if needed
        if xv.size == 1 and k > 1: xv = np.full(k, xv.item())
        if yv.size == 1 and k > 1: yv = np.full(k, yv.item())
        if zv.size == 1 and k > 1: zv = np.full(k, zv.item())

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            active = np.zeros(k, dtype=bool)
            active[yv <= tol] = True
            mask = yv > tol
            resid = zv[mask] - yv[mask] * np.exp(xv[mask] / yv[mask])
            active[mask] = resid <= tol

        exp_cnt += int(np.sum(active))
    out["exp"] = exp_cnt

    psd_cnt = 0
    for c in b["psd_cones"]:
        X = np.asarray(c.expr.value, dtype=float)
        X = 0.5 * (X + X.T)
        psd_cnt += int(np.linalg.eigvalsh(X).min() <= tol)
    out["psd"] = psd_cnt

    out["cone_total"] = out["soc"] + out["exp"] + out["psd"]
    out["total"] = out["eq"] + out["ineq"] + out["cone_total"]
    return out

def active_counts_dict(ctx, tol = None, reduce: str = "sum"):
    tol = float(ctx.mt.slack_tol if tol is None else tol)
    per_batch = [_active_counts_one(ctx.bundles[i], ctx, i, tol) for i in range(ctx.batch_size)]
    if reduce is None:
        return per_batch
    keys = per_batch[0].keys()
    return {k: sum(d[k] for d in per_batch) for k in keys}


def _build_problem_bundle(
    problem: cp.Problem,
    parameters,
    variables,
    alpha: float,
    dual_cutoff: float,
    slack_tol: float,
    eps: float,
):
    """
    Build and return a dict containing EVERYTHING needed for one problem:
      - forward problem, perturbed problem
      - cvxpy Parameters for dvars/duals/masks
      - torch callables for phi and each dual term
      - pnorm tangent caches + TorchExpression for g (for gradient wrt variables)
    """
    objective_expr = _infer_objective_expr(problem)

    # ---- split constraints ----
    eq_funcs = []
    scalar_ineq_funcs = []
    soc_constraints = []
    exp_cones = []
    psd_cones = []

    for c in problem.constraints:
        if isinstance(c, cp.constraints.zero.Equality):
            eq_funcs.append(c.expr)
        elif isinstance(c, cp.constraints.nonpos.Inequality):
            scalar_ineq_funcs.append(c.expr)
        elif isinstance(c, SOC):
            soc_constraints.append(c)
        elif isinstance(c, ExpCone):
            exp_cones.append(c)
        elif isinstance(c, PSD):
            psd_cones.append(c)
        else:
            raise ValueError(f"Unsupported constraint type: {type(c)}")

    param_order = list(parameters)
    variables = list(variables)

    # ---- original problem (forward) ----
    eq_constraints = [f == 0 for f in eq_funcs]
    scalar_ineq_constraints = [g <= 0 for g in scalar_ineq_funcs]

    forward_problem = cp.Problem(
        cp.Minimize(objective_expr),
        eq_constraints + scalar_ineq_constraints + soc_constraints + exp_cones + psd_cones,
    )

    # ---- dvar params ----
    dvar_params = [cp.Parameter(shape=v.shape) for v in variables]

    # ---- dual params (old) for eq/scalar ineq ----
    eq_dual_params = [cp.Parameter(shape=f.shape) for f in eq_funcs]
    scalar_ineq_dual_params = [cp.Parameter(shape=g.shape, nonneg=True) for g in scalar_ineq_funcs]

    # ---- scalar active masks ----
    scalar_active_mask_params = [cp.Parameter(shape=g.shape, nonneg=True) for g in scalar_ineq_funcs]

    # ---- SOC dual placeholders (old) and linear constraints ----
    soc_dual_params_0 = [cp.Parameter(shape=c.dual_variables[0].shape, nonneg=True) for c in soc_constraints]
    soc_dual_params_1 = [cp.Parameter(shape=c.dual_variables[1].shape) for c in soc_constraints]

    soc_dual_product = _cvx_sum_or_zero([
        cp.multiply(cp.pnorm(c.args[1].expr, p=2) - c.args[0].expr, u)
        for u, c in zip(soc_dual_params_0, soc_constraints)
    ])
    soc_lin_constraints = [
        (soc_dual_params_1[j].T @ soc_constraints[j].args[1].expr
         + cp.multiply(soc_constraints[j].args[0].expr, soc_dual_params_0[j])) == 0
        for j in range(len(soc_constraints))
    ]

    # ---- ExpCone dual placeholders (old) ----
    exp_dual_params = [[cp.Parameter(shape=dv.shape) for dv in c.dual_variables] for c in exp_cones]
    exp_dual_product = _cvx_sum_or_zero([
        _expcone_dual_dot(u3, c) for u3, c in zip(exp_dual_params, exp_cones)
    ])

    # ---- PSD dual placeholders (old) ----
    psd_dual_params = [cp.Parameter(shape=c.dual_variables[0].shape) for c in psd_cones]
    psd_dual_product = _cvx_sum_or_zero([
        cp.sum(cp.multiply(u, c.expr)) for u, c in zip(psd_dual_params, psd_cones)
    ])

    # ---- pnorm tangent support for scalar inequalities (scalar-only) ----
    pnorm_ineq_ids = []
    non_pnorm_scalar_ids = []
    pnorm_xstar_params = []
    pnorm_grad_params = []
    pnorm_tangent_constraints = []
    pnorm_g_torch = []

    for j, g in enumerate(scalar_ineq_funcs):
        is_scalar = int(np.prod(g.shape)) == 1
        is_pnorm = is_scalar and _has_pnorm_atom(g)
        if not is_pnorm:
            non_pnorm_scalar_ids.append(j)
            continue

        local_id = len(pnorm_ineq_ids)
        pnorm_ineq_ids.append(j)

        xs = []
        gs = []
        for v in variables:
            xs.append(cp.Parameter(shape=v.shape))
            gs.append(cp.Parameter(shape=v.shape))
        pnorm_xstar_params.append(xs)
        pnorm_grad_params.append(gs)

        lin = cp.Constant(0.0)
        for v_id, v in enumerate(variables):
            dv = v - pnorm_xstar_params[local_id][v_id]
            lin += cp.sum(cp.multiply(pnorm_grad_params[local_id][v_id], dv))

        pnorm_tangent_constraints.append(cp.multiply(scalar_active_mask_params[j], lin) == 0)

        pnorm_g_torch.append(
            TorchExpression(
                g,
                provided_vars_list=[*variables, *param_order],
            ).torch_expression
        )

    # ---- perturbed problem ----
    vars_dvars_product = _cvx_sum_or_zero([cp.sum(cp.multiply(dv, v)) for dv, v in zip(dvar_params, variables)])
    scalar_ineq_dual_product = _cvx_sum_or_zero([
        cp.sum(cp.multiply(lm, g)) for lm, g in zip(scalar_ineq_dual_params, scalar_ineq_funcs)
    ])

    new_objective = (1.0 / float(alpha)) * vars_dvars_product + objective_expr
    new_objective += scalar_ineq_dual_product + soc_dual_product + exp_dual_product
    # Note: psd_dual_product is intentionally NOT added to the perturbed objective.
    # For PSD cones, keeping them as explicit constraints in the perturbed problem
    # lets the solver handle the conic geometry directly.

    active_eq_constraints = [
        cp.multiply(scalar_active_mask_params[j], scalar_ineq_funcs[j]) == 0
        for j in non_pnorm_scalar_ids
    ]

    perturbed_problem = cp.Problem(
        cp.Minimize(new_objective),
        eq_constraints + active_eq_constraints + soc_lin_constraints + pnorm_tangent_constraints + psd_cones,
    )

    # ---- TorchExpressions for loss pieces (phi and dual terms) ----
    phi_torch = TorchExpression(
        objective_expr,
        provided_vars_list=[*variables, *param_order],
    ).torch_expression

    eq_terms = [cp.sum(cp.multiply(du, f)) for du, f in zip(eq_dual_params, eq_funcs)]
    eq_dual_term_torch = TorchExpression(
        _cvx_sum_or_zero(eq_terms),
        provided_vars_list=[*variables, *param_order, *eq_dual_params],
    ).torch_expression

    ineq_terms = [cp.sum(cp.multiply(du, g)) for du, g in zip(scalar_ineq_dual_params, scalar_ineq_funcs)]
    ineq_dual_term_torch = TorchExpression(
        _cvx_sum_or_zero(ineq_terms),
        provided_vars_list=[*variables, *param_order, *scalar_ineq_dual_params],
    ).torch_expression

    if len(exp_cones) > 0:
        exp_terms = [_expcone_dual_dot(du3, c) for du3, c in zip(exp_dual_params, exp_cones)]
        exp_dual_term_torch = TorchExpression(
            _cvx_sum_or_zero(exp_terms),
            provided_vars_list=[*variables, *param_order, *[u for tri in exp_dual_params for u in tri]],
        ).torch_expression
    else:
        exp_dual_term_torch = None

    if len(psd_cones) > 0:
        psd_terms = [cp.sum(cp.multiply(du, c.expr)) for du, c in zip(psd_dual_params, psd_cones)]
        psd_dual_term_torch = TorchExpression(
            _cvx_sum_or_zero(psd_terms),
            provided_vars_list=[*variables, *param_order, *psd_dual_params],
        ).torch_expression
    else:
        psd_dual_term_torch = None

    non_pnorm_set = set(non_pnorm_scalar_ids)
    pnorm_set = set(pnorm_ineq_ids)
    pnorm_map = {j: lid for lid, j in enumerate(pnorm_ineq_ids)}

    scalar_is_scalar = [int(np.prod(g.shape)) == 1 for g in scalar_ineq_funcs]
    scalar_scalar_indices = [j for j, f in enumerate(scalar_is_scalar) if f]
    scalar_nonscalar_indices = [j for j, f in enumerate(scalar_is_scalar) if not f]

    return dict(
        alpha=float(alpha),
        dual_cutoff=float(dual_cutoff),
        slack_tol=float(slack_tol),
        eps=float(eps),

        param_order=param_order,
        variables=variables,

        objective=objective_expr,
        eq_functions=eq_funcs,
        scalar_ineq_functions=scalar_ineq_funcs,
        scalar_is_scalar=scalar_is_scalar,
        scalar_scalar_indices=scalar_scalar_indices,
        scalar_nonscalar_indices=scalar_nonscalar_indices,
        soc_constraints=soc_constraints,
        exp_cones=exp_cones,
        psd_cones=psd_cones,

        eq_constraints=eq_constraints,
        scalar_ineq_constraints=scalar_ineq_constraints,
        soc_lin_constraints=soc_lin_constraints,
        active_eq_constraints=active_eq_constraints,

        # problems
        problem=forward_problem,
        perturbed_problem=perturbed_problem,

        # cvx params
        dvar_params=dvar_params,
        eq_dual_params=eq_dual_params,
        scalar_ineq_dual_params=scalar_ineq_dual_params,
        scalar_active_mask_params=scalar_active_mask_params,
        soc_dual_params_0=soc_dual_params_0,
        soc_dual_params_1=soc_dual_params_1,
        exp_dual_params=exp_dual_params,
        psd_dual_params=psd_dual_params,

        # pnorm tangent
        pnorm_ineq_ids=pnorm_ineq_ids,
        non_pnorm_scalar_ids=non_pnorm_scalar_ids,
        pnorm_xstar_params=pnorm_xstar_params,
        pnorm_grad_params=pnorm_grad_params,
        pnorm_tangent_constraints=pnorm_tangent_constraints,
        pnorm_g_torch=pnorm_g_torch,

        # precomputed sets/maps
        non_pnorm_set=non_pnorm_set,
        pnorm_set=pnorm_set,
        pnorm_map=pnorm_map,

        # torch callables
        phi_torch=phi_torch,
        eq_dual_term_torch=eq_dual_term_torch,
        ineq_dual_term_torch=ineq_dual_term_torch,
        exp_dual_term_torch=exp_dual_term_torch,
        psd_dual_term_torch=psd_dual_term_torch,
    )


def FFOLayer(
    problem,
    parameters,
    variables,
    alpha: float = 100.0,
    dual_cutoff: float = 1e-3,
    slack_tol: float = 1e-8,
    eps: float = 1e-13,
    compute_cos_sim: bool = False,
    max_workers: int = 8,
    backward_eps: float = 1e-3,
    verbose: bool = False,
):
    _require_cvxtorch()
    print(f"FFOLayer forward eps = {eps}, backward eps = {backward_eps}")
    return _FFOLayer(
        problem=problem,
        parameters=parameters,
        variables=variables,
        alpha=alpha,
        dual_cutoff=dual_cutoff,
        slack_tol=slack_tol,
        eps=eps,
        backward_eps=backward_eps,
        compute_cos_sim=compute_cos_sim,
        max_workers=max_workers,
        verbose=verbose,
    )


class _FFOLayer(torch.nn.Module):
    def __init__(
        self,
        problem,
        parameters,
        variables,
        alpha,
        dual_cutoff,
        slack_tol,
        eps,
        backward_eps,
        compute_cos_sim,
        max_workers: int = 8,
        verbose: bool = False,
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.dual_cutoff = float(dual_cutoff)
        self.slack_tol = float(slack_tol)
        self.eps = float(eps)
        self.backward_eps = float(backward_eps)
        self._compute_cos_sim = bool(compute_cos_sim)
        self.verbose = bool(verbose)

        self._problem_proto = problem

        # If problem is a list, user may pass parameters/variables as list-of-list (one list per problem).
        self._params_list_proto = None
        self._vars_list_proto = None

        if isinstance(problem, (list, tuple)):
            problem_list = list(problem)
            if len(problem_list) == 0:
                raise ValueError("Empty problem_list.")

            # Case A: parameters/variables are list-of-list aligned with problem_list
            if (
                isinstance(parameters, (list, tuple)) and len(parameters) == len(problem_list)
                and len(parameters) > 0 and isinstance(parameters[0], (list, tuple))
            ):
                if not (isinstance(variables, (list, tuple)) and len(variables) == len(problem_list)
                        and len(variables) > 0 and isinstance(variables[0], (list, tuple))):
                    raise ValueError("When problem is a list and parameters is list-of-list, variables must be list-of-list too.")

                self._params_list_proto = [list(pi) for pi in parameters]
                self._vars_list_proto = [list(vi) for vi in variables]

                self._param_templates = list(self._params_list_proto[0])
                self._var_templates = list(self._vars_list_proto[0])

            # Case B: parameters/variables are flat templates; we'll map by name in _lazy_init_from_B
            else:
                self._param_templates = list(parameters)
                self._var_templates = list(variables)
        else:
            self._param_templates = list(parameters)
            self._var_templates = list(variables)
        # self._problem_proto = problem
        # self._param_templates = list(parameters)
        # self._var_templates = list(variables)
        self._max_workers_user = max_workers
        self._initialized = False

        self.num_problems = 0
        self.bundles = None
        self.problem_list = None
        self.perturbed_problem_list = None
        self._ref_param_order = None
        self._ref_vars = None
        self._ws_primal_fwd = None
        self._executor = None

        self.forward_solve_time = 0.0
        self.backward_solve_time = 0.0
        self.forward_setup_time = 0.0
        self.backward_setup_time = 0.0

        self._solver_args_fwd = None
        self._solver_args_bwd = None
    
    def _infer_B_from_params(self, params):
        ref_param_order = self._param_templates
        batch_sizes = []
        for i, (p, qtmpl) in enumerate(zip(params, ref_param_order)):
            if p.ndimension() == qtmpl.ndim:
                bs = 0
            elif p.ndimension() == qtmpl.ndim + 1:
                bs = int(p.size(0))
                if bs <= 0:
                    raise ValueError(f"Parameter {i} has empty batch dimension.")
            else:
                raise ValueError(
                    f"Invalid dim for parameter {i}: got {p.ndimension()}, expected {qtmpl.ndim} or {qtmpl.ndim+1}."
                )

            p_shape = p.shape if bs == 0 else p.shape[1:]
            if tuple(p_shape) != tuple(qtmpl.shape):
                raise ValueError(f"Parameter {i} shape mismatch: expected {qtmpl.shape}, got {p.shape}.")

            batch_sizes.append(bs)

        batch_sizes = np.array(batch_sizes, dtype=int)
        if np.any(batch_sizes > 0):
            nonzero = batch_sizes[batch_sizes > 0]
            B = int(nonzero[0])
            if np.any(nonzero != B):
                raise ValueError(f"Inconsistent batch sizes: {batch_sizes}.")
        else:
            B = 1
        return B


    def _lazy_init_from_B(self, B: int, solver_args: dict):
        if self._initialized:
            return

        if isinstance(self._problem_proto, (list, tuple)):
            problem_list = list(self._problem_proto)
            if len(problem_list) != B:
                raise ValueError(f"Got batch size B={B}, but problem_list has len={len(problem_list)}.")

            # parameters_list = list(self._param_templates)
            # variables_list = list(self._var_templates)
            # if not (len(parameters_list) == len(variables_list) == len(problem_list)):
            #     raise ValueError("When passing problem as list, parameters and variables must be list-of-list aligned.")
            if self._params_list_proto is not None:
                parameters_list = self._params_list_proto
                variables_list = self._vars_list_proto
                if not (len(parameters_list) == len(variables_list) == len(problem_list)):
                    raise ValueError("When passing problem as list, parameters and variables must be list-of-list aligned.")
                # sanity: each inner list length matches template length
                P = len(self._param_templates)
                V = len(self._var_templates)
                for i in range(B):
                    if len(parameters_list[i]) != P:
                        raise ValueError(f"parameters_list[{i}] length mismatch: expected {P}, got {len(parameters_list[i])}")
                    if len(variables_list[i]) != V:
                        raise ValueError(f"variables_list[{i}] length mismatch: expected {V}, got {len(variables_list[i])}")
            else:
                # Otherwise, map by name from each problem
                pnames = [p.name() for p in self._param_templates]
                vnames = [v.name() for v in self._var_templates]
                parameters_list, variables_list = [], []
                for prob_i in problem_list:
                    pmap = prob_i.param_dict
                    vmap = prob_i.var_dict
                    parameters_list.append([pmap[n] for n in pnames])
                    variables_list.append([vmap[n] for n in vnames])
        else:
            pnames = [p.name() for p in self._param_templates]
            vnames = [v.name() for v in self._var_templates]

            problem_list, parameters_list, variables_list = [], [], []
            # Clear solver cache before deepcopy – solver objects are not picklable
            saved_cache = getattr(self._problem_proto, '_solver_cache', None)
            if saved_cache is not None:
                self._problem_proto._solver_cache = {}
            for _ in range(int(B)):
                prob_i = copy.deepcopy(self._problem_proto)
                pmap = prob_i.param_dict
                vmap = prob_i.var_dict
                params_i = [pmap[n] for n in pnames]
                vars_i = [vmap[n] for n in vnames]
                problem_list.append(prob_i)
                parameters_list.append(params_i)
                variables_list.append(vars_i)
            if saved_cache is not None:
                self._problem_proto._solver_cache = saved_cache

        self.num_problems = len(problem_list)
        if self.num_problems == 0:
            raise ValueError("Empty problem_list.")

        self.max_workers = int(self._max_workers_user or min(os.cpu_count() or 1, self.num_problems))
        print(f"max_workers: {self.max_workers}")

        bundles = []
        for prob_i, params_i, vars_i in zip(problem_list, parameters_list, variables_list):
            bundles.append(_build_problem_bundle(
                prob_i,
                parameters=params_i,
                variables=vars_i,
                alpha=self.alpha,
                dual_cutoff=self.dual_cutoff,
                slack_tol=self.slack_tol,
                eps=self.eps,
            ))
        self.bundles = bundles

        self.problem_list = [b["problem"] for b in bundles]
        self.perturbed_problem_list = [b["perturbed_problem"] for b in bundles]

        self._ref_param_order = bundles[0]["param_order"]
        self._ref_vars = bundles[0]["variables"]

        self._ws_cache_fwd = {}  # key -> {scs_x, scs_y, scs_s}
        self._ws_cache_bwd = {}  # key -> {scs_x, scs_y, scs_s}
        self._scs_solvers = {}  # i -> SCS solver instance (for direct SCS path)
        self._scs_data_hash = None  # hash of (A, b) params to detect changes
        self._scs_mapping = None  # {primal_slice, eq_dual_slice, ineq_dual_slice, c_p_slice}

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self._FFOLayerFn = _make_ffo_fn(self, solver_args=solver_args)
        self._initialized = True


    def close(self):
        ex = getattr(self, "_executor", None)
        if ex is not None:
            ex.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    def forward(self, *params, solver_args=None):
        if solver_args is None:
            solver_args = {}
        solver = solver_args.get("solver", cp.SCS)
        if solver == cp.SCS:
            default_solver_args = dict(
                solver=cp.SCS,
                warm_start=False,
                ignore_dpp=True,
                max_iters=200,
                eps=self.eps,
                verbose=False,
            )
        else:
            default_solver_args = dict(ignore_dpp=False)

        solver_args = {**default_solver_args, **solver_args}
        self._warm_start = bool(solver_args.get("warm_start", False))
        self._ws_keys = solver_args.pop("ws_keys", None)

        if not self._initialized:
            B = self._infer_B_from_params(params)
            self._lazy_init_from_B(B, solver_args)

        self._solver_args_fwd = dict(solver_args)

        self._solver_args_bwd = dict(solver_args)
        self._solver_args_bwd["max_iters"] = 2500
        self._solver_args_bwd["warm_start"] = True
        if "eps" in self._solver_args_bwd:
            self._solver_args_bwd["eps"] = float(self.backward_eps)

        # Fn = _make_ffo_fn(self, solver_args)
        # return Fn.apply(*params)
        return self._FFOLayerFn.apply(*params)

def _make_ffo_fn(mt: "_FFOLayer", solver_args: dict):
    solver_args = dict(solver_args)
    class _FFOLayerFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            ctx.mt = mt
            ctx.bundles = mt.bundles
            ctx.solver_args = solver_args
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device if isinstance(params[0], torch.Tensor) else 'cpu'

            ref_param_order = mt._ref_param_order
            batch_sizes = []
            for i, (p, qtmpl) in enumerate(zip(params, ref_param_order)):
                if p.dtype != ctx.dtype or p.device != ctx.device:
                    raise ValueError(f"Parameter {i} dtype/device mismatch.")
                if p.ndimension() == qtmpl.ndim:
                    bs = 0
                elif p.ndimension() == qtmpl.ndim + 1:
                    bs = int(p.size(0))
                    if bs <= 0:
                        raise ValueError(f"Parameter {i} has empty batch dimension.")
                else:
                    raise ValueError(f"Invalid dim for parameter {i}: got {p.ndimension()}, expected {qtmpl.ndim} or {qtmpl.ndim+1}.")
                batch_sizes.append(bs)

                p_shape = p.shape if bs == 0 else p.shape[1:]
                if tuple(p_shape) != tuple(qtmpl.shape):
                    raise ValueError(f"Parameter {i} shape mismatch: expected {qtmpl.shape}, got {p.shape}.")

            ctx.batch_sizes = np.array(batch_sizes, dtype=int)
            ctx.batch = bool(np.any(ctx.batch_sizes > 0))
            if ctx.batch:
                nonzero = ctx.batch_sizes[ctx.batch_sizes > 0]
                B = int(nonzero[0])
                if np.any(nonzero != B):
                    raise ValueError(f"Inconsistent batch sizes: {ctx.batch_sizes}.")
            else:
                B = 1
            if ctx.batch and B != mt.num_problems:
                raise ValueError(f"Batch size ({B}) must equal number of problems ({mt.num_problems}).")
            ctx.batch_size = B

            params_np_all = [to_numpy(p) for p in params]

            def _slice_params_np(i: int):
                if ctx.batch:
                    return [arr[i] if bs > 0 else arr for arr, bs in zip(params_np_all, ctx.batch_sizes)]
                return params_np_all

            ref_bundle = ctx.bundles[0]
            variables = ref_bundle["variables"]
            eq_functions = ref_bundle["eq_functions"]
            scalar_ineq_functions = ref_bundle["scalar_ineq_functions"]
            soc_constraints = ref_bundle["soc_constraints"]
            exp_cones = ref_bundle["exp_cones"]
            psd_cones = ref_bundle["psd_cones"]

            sol_numpy = [np.empty((B,) + v.shape, dtype=float) for v in variables]
            eq_dual = [np.empty((B,) + f.shape, dtype=float) for f in eq_functions]
            scalar_ineq_dual = [np.empty((B,) + g.shape, dtype=float) for g in scalar_ineq_functions]
            scalar_ineq_slack = [np.empty((B,) + g.shape, dtype=float) for g in scalar_ineq_functions]

            soc_dual_0 = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in soc_constraints]
            soc_dual_1 = [np.empty((B,) + c.dual_variables[1].shape, dtype=float) for c in soc_constraints]

            exp_dual = [
                [np.empty((B,) + dv.shape, dtype=float) for dv in c.dual_variables]
                for c in exp_cones
            ]
            psd_dual = [np.empty((B,) + c.dual_variables[0].shape, dtype=float) for c in psd_cones]

            pnorm_xstar = []
            pnorm_grad = []
            for _local_id in range(len(ref_bundle["pnorm_ineq_ids"])):
                pnorm_xstar.append([np.empty((B,) + v.shape, dtype=float) for v in variables])
                pnorm_grad.append([np.empty((B,) + v.shape, dtype=float) for v in variables])

            def _slice_params_torch(i: int):
                if ctx.batch:
                    return [p[i] if bs > 0 else p for p, bs in zip(params, ctx.batch_sizes)]
                return list(params)

            fwd_solver_iters = [0] * B
            _fwd_solve_times = [0.0] * B

            # ---- Direct SCS path: setup solvers for this forward pass ----
            if mt._warm_start:
                import scs as _scs
                import time as _time_mod

                # Detect if A/b changed by checking param tensor versions
                _param_versions = tuple(
                    p.data_ptr() if hasattr(p, 'data_ptr') else id(p)
                    for p in params
                )
                if mt._scs_data_hash != _param_versions:
                    # Use first problem to get the SCS data template via CVXPY
                    b0 = ctx.bundles[0]
                    prob0 = mt.problem_list[0]
                    params_0_np = _slice_params_np(0)
                    for pval, pparam in zip(params_0_np, b0["param_order"]):
                        pparam.value = pval
                    data0, _, _ = prob0.get_problem_data(
                        solver=cp.SCS, ignore_dpp=True
                    )
                    cone = {
                        'z': data0['dims'].zero,
                        'l': data0['dims'].nonneg,
                    }
                    scs_args = dict(
                        max_iters=int(mt._solver_args_fwd.get('max_iters', 2500)),
                        eps_abs=float(mt._solver_args_fwd.get('eps', 1e-6)),
                        eps_rel=float(mt._solver_args_fwd.get('eps', 1e-6)),
                        verbose=False,
                    )
                    # Discover the mapping once
                    if mt._scs_mapping is None:
                        n_vars = sum(int(np.prod(v.shape)) for v in b0["variables"])
                        n_eq = sum(int(np.prod(f.shape)) for f in b0["eq_functions"])
                        n_ineq = sum(int(np.prod(f.shape)) for f in b0["scalar_ineq_functions"])
                        # SCS x layout: [aux(n_vars), y_var(n_vars)]
                        # SCS y layout: [aux_dual(n_vars), eq_dual(n_eq), ineq_dual(n_ineq)]
                        mt._scs_mapping = {
                            'primal_slice': slice(n_vars, 2 * n_vars),
                            'eq_dual_slice': slice(n_vars, n_vars + n_eq),
                            'ineq_dual_slice': slice(n_vars + n_eq, n_vars + n_eq + n_ineq),
                            'c_p_slice': slice(n_vars, 2 * n_vars),
                            'b_eq_slice': slice(n_vars, n_vars + n_eq),
                        }

                    # Store dimensions for later use
                    mt._scs_c_dim = len(data0['c'])

                    # Build SCS solvers in parallel (SCS releases GIL)
                    scs_template = {
                        'P': data0['P'],
                        'A': data0['A'],
                        'b': data0['b'].copy(),
                        'c': data0['c'].copy(),
                    }
                    def _build_scs(j):
                        sd = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                              for k, v in scs_template.items()}
                        return _scs.SCS(sd, cone, **scs_args)

                    with _limit_native_threads(1):
                        futs = [mt._executor.submit(_build_scs, j) for j in range(B)]
                        mt._scs_solvers = {j: f.result() for j, f in enumerate(futs)}
                    mt._scs_data_hash = _param_versions

                    # Mark that backward template needs to be built
                    mt._bwd_scs_template = None

                _m = mt._scs_mapping

            def _solve_one(i: int):
                import time as _time
                b = ctx.bundles[i]
                prob = mt.problem_list[i]

                params_i_np = _slice_params_np(i)
                # Always set CVXPY params (backward pass needs them)
                for pval, pparam in zip(params_i_np, b["param_order"]):
                    pparam.value = pval

                _t0 = _time.perf_counter()

                if mt._warm_start and i in mt._scs_solvers:
                    # ---- Direct SCS path ----
                    scs_solver = mt._scs_solvers[i]
                    # Update c vector with current p (puzzle encoding)
                    # p is at index 1 in params: [Q, p, G, h, A, b]
                    p_np = params_i_np[1]  # the puzzle-specific parameter
                    new_c = np.zeros(mt._scs_c_dim, dtype=float)
                    new_c[_m['c_p_slice']] = p_np
                    scs_solver.update(c=new_c)

                    # Warm start from cached SCS solution (same puzzle, prev epoch)
                    ws_key = mt._ws_keys[i] if mt._ws_keys is not None else None
                    ws_cached = mt._ws_cache_fwd.get(ws_key) if ws_key is not None else None
                    if ws_cached is not None and 'scs_x' in ws_cached:
                        sol = scs_solver.solve(
                            warm_start=True,
                            x=ws_cached['scs_x'],
                            y=ws_cached['scs_y'],
                            s=ws_cached['scs_s'],
                        )
                    else:
                        sol = scs_solver.solve(warm_start=False)

                    if sol['info']['status'] == 'solved' or sol['info']['status'] == 'solved_inaccurate':
                        x_scs = sol['x']
                        y_scs = sol['y']

                        # Cache full SCS state for warm starting next epoch
                        if ws_key is not None:
                            mt._ws_cache_fwd[ws_key] = {
                                'scs_x': sol['x'].copy(),
                                'scs_y': sol['y'].copy(),
                                'scs_s': sol['s'].copy(),
                            }

                        # Extract primal solution
                        y_var = x_scs[_m['primal_slice']]
                        for v_id, v in enumerate(b["variables"]):
                            vshape = v.shape
                            n_el = int(np.prod(vshape))
                            sol_numpy[v_id][i, ...] = y_var[:n_el].reshape(vshape)

                        # Extract dual values
                        eq_d = y_scs[_m['eq_dual_slice']]
                        offset = 0
                        for c_id, f in enumerate(b["eq_functions"]):
                            n_el = int(np.prod(f.shape))
                            eq_dual[c_id][i, ...] = eq_d[offset:offset+n_el].reshape(f.shape)
                            offset += n_el

                        ineq_d = y_scs[_m['ineq_dual_slice']]
                        offset = 0
                        for j, g_expr in enumerate(b["scalar_ineq_functions"]):
                            n_el = int(np.prod(g_expr.shape))
                            scalar_ineq_dual[j][i, ...] = ineq_d[offset:offset+n_el].reshape(g_expr.shape)
                            # slack = max(-(G@y - h), 0) = max(y_var, 0) for G=-I, h=0
                            scalar_ineq_slack[j][i, ...] = np.maximum(y_var[offset:offset+n_el].reshape(g_expr.shape), 0.0)
                            offset += n_el

                        fwd_solver_iters[i] = sol['info']['iter']
                        _fwd_solve_times[i] = _time.perf_counter() - _t0
                        return  # skip CVXPY path
                    else:
                        print(f"[forward] SCS direct failed for problem {i}: {sol['info']['status']}, falling back to CVXPY")

                # ---- CVXPY fallback path ----
                try:
                    prob.solve(**mt._solver_args_fwd)
                except Exception as e:
                    print(f"[forward] problem {i} solve failed: {e!r}")
                    try:
                        prob.solve(solver=cp.OSQP, warm_start=False, verbose=False)
                    except Exception as e2:
                        raise RuntimeError(f"[forward] problem {i} solve failed: {e!r} {e2!r}")
                _fwd_solve_times[i] = _time.perf_counter() - _t0

                if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"[forward] problem {i} status: {prob.status}")

                if prob.solver_stats is not None:
                    fwd_solver_iters[i] = getattr(prob.solver_stats, 'num_iters', 0)

                for v_id, v in enumerate(b["variables"]):
                    sol_numpy[v_id][i, ...] = v.value

                for c_id, c in enumerate(b["eq_constraints"]):
                    eq_dual[c_id][i, ...] = c.dual_value

                for j, g_expr in enumerate(b["scalar_ineq_functions"]):
                    g_val = np.asarray(g_expr.value, dtype=float)
                    scalar_ineq_dual[j][i, ...] = b["scalar_ineq_constraints"][j].dual_value
                    scalar_ineq_slack[j][i, ...] = np.maximum(-g_val, 0.0)

                for c_id, c in enumerate(b["soc_constraints"]):
                    dv0, dv1 = c.dual_value
                    soc_dual_0[c_id][i, ...] = dv0
                    if hasattr(dv1, "shape") and len(dv1.shape) == 2 and dv1.shape[1] == 1:
                        soc_dual_1[c_id][i, ...] = dv1.reshape(-1)
                    else:
                        soc_dual_1[c_id][i, ...] = dv1

                for c_id, c in enumerate(b["exp_cones"]):
                    shapes3 = [dv.shape for dv in c.dual_variables]
                    dv3 = _split_expcone_dual_value(c.dual_value, shapes3)
                    for k in range(3):
                        exp_dual[c_id][k][i, ...] = dv3[k]

                for c_id, c in enumerate(b["psd_cones"]):
                    psd_dual[c_id][i, ...] = c.dual_value

                if len(b["pnorm_ineq_ids"]) > 0:
                    with torch.enable_grad():
                        vars_star_t = [
                            torch.tensor(sol_numpy[v_id][i, ...], dtype=ctx.dtype, device=ctx.device, requires_grad=True)
                            for v_id in range(len(variables))
                        ]
                        params_i_det = [t.detach() for t in _slice_params_torch(i)]
                        for local_id in range(len(b["pnorm_ineq_ids"])):
                            g_t = b["pnorm_g_torch"][local_id](*vars_star_t, *params_i_det).reshape(())
                            grads = torch.autograd.grad(
                                g_t,
                                vars_star_t,
                                retain_graph=False,
                                create_graph=False,
                                allow_unused=True,
                            )
                            for v_id, gv in enumerate(grads):
                                pnorm_xstar[local_id][v_id][i, ...] = to_numpy(vars_star_t[v_id].detach())
                                pnorm_grad[local_id][v_id][i, ...] = 0.0 if gv is None else to_numpy(gv.detach())

            with _limit_native_threads(1):
                futs = [mt._executor.submit(_solve_one, i) for i in range(B)]
                for f in futs:
                    f.result()

                # single thread for debugging
                # for i in range(B):
                #     _solve_one(i) 

            if mt._warm_start:
                total_fwd = sum(fwd_solver_iters)
                max_solve = max(_fwd_solve_times)
                sum_solve = sum(_fwd_solve_times)
                fwd_cache_size = len(mt._ws_cache_fwd)
                print(f"[forward] iters: avg={total_fwd/max(B,1):.0f}, max_solve={max_solve:.3f}s, sum_solve={sum_solve:.3f}s, cache={fwd_cache_size}")
            elif mt.verbose:
                total_fwd = sum(fwd_solver_iters)
                print(f"[forward] solver iters: total={total_fwd}, avg={total_fwd/max(B,1):.1f}")

            ctx.sol_numpy = sol_numpy
            ctx.eq_dual = eq_dual
            ctx.scalar_ineq_dual = scalar_ineq_dual
            ctx.scalar_ineq_slack = scalar_ineq_slack
            ctx.soc_dual_0 = soc_dual_0
            ctx.soc_dual_1 = soc_dual_1
            ctx.exp_dual = exp_dual
            ctx.psd_dual = psd_dual
            ctx.pnorm_xstar = pnorm_xstar
            ctx.pnorm_grad = pnorm_grad
            ctx.params = params

            # if want to check active counts
            if mt.verbose:
                ctx.active_counts = active_counts_dict(ctx)
                print(f"active_counts: {ctx.active_counts}")

            sol_torch = [to_torch(arr, ctx.dtype, ctx.device) for arr in sol_numpy]
            return tuple(sol_torch) # return the solution

        @staticmethod
        def backward(ctx, *dvars):
            mt = ctx.mt
            bundles = ctx.bundles
            B = ctx.batch_size

            ref = bundles[0]
            num_vars = len(ref["variables"])
            num_scalar_ineq = len(ref["scalar_ineq_functions"])

            params_np_all = [to_numpy(p) for p in ctx.params]
            dvars_np_all = [to_numpy(dv) for dv in dvars]

            def _slice_params_np(i: int):
                if ctx.batch:
                    return [arr[i] if bs > 0 else arr for arr, bs in zip(params_np_all, ctx.batch_sizes)]
                return params_np_all

            # def _slice_dvars_np(i: int):
            #     out = []
            #     for arr, v in zip(dvars_np_all, ref["variables"]):
            #         vshape = tuple(v.shape)
            #         if arr.shape == (B,) + vshape:
            #             out.append(arr[i])
            #         elif B == 1 and arr.ndim >= 1 and arr.shape[0] == 1 and tuple(arr.shape[1:]) == vshape:
            #             out.append(arr[0])
            #         else:
            #             out.append(arr)
            #     return out
            def _slice_dvars_np(i: int):
                if ctx.batch:
                    return [arr[i] for arr in dvars_np_all]
                # Even when not batched, sol_numpy has leading B=1 dim,
                # so dvars also has shape (1, *v.shape). Slice it out.
                return [arr[i] if arr.ndim > len(v.shape) else arr
                        for arr, v in zip(dvars_np_all, ref["variables"])]

            y_dim = int(np.prod((_slice_dvars_np(0)[0]).shape))
            num_eq = int(np.prod(ctx.eq_dual[0][0].shape)) if (len(ctx.eq_dual) > 0 and ctx.batch) else (
                int(np.prod(ctx.eq_dual[0].shape)) if len(ctx.eq_dual) > 0 else 0
            )
            cap_scalar = int(max(1, y_dim - num_eq))

            new_sol_lagrangian = [np.empty_like(ctx.sol_numpy[k]) for k in range(num_vars)]
            new_eq_dual = [np.empty_like(ctx.eq_dual[k]) for k in range(len(ref["eq_constraints"]))]

            new_active_dual = [np.empty((B,) + c.shape, dtype=float) for c in ref["active_eq_constraints"]]

            new_soc_lam = [np.zeros((B,), dtype=float) for _ in ref["soc_lin_constraints"]]
            new_pnorm_lam = [np.zeros((B,), dtype=float) for _ in ref["pnorm_tangent_constraints"]]

            new_exp_dual = [
                [np.empty_like(ctx.exp_dual[j][k]) for k in range(3)]
                for j in range(len(ref["exp_cones"]))
            ]
            new_psd_dual = [np.empty_like(ctx.psd_dual[k]) for k in range(len(ref["psd_cones"]))]

            def _slice_params_torch(i: int, params_src):
                if ctx.batch:
                    return [p[i] if bs > 0 else p for p, bs in zip(params_src, ctx.batch_sizes)]
                return list(params_src)

            bwd_solver_iters = [0] * B
            _bwd_solve_times = [0.0] * B

            # ---- Rebuild backward SCS template when params change ----
            if mt._warm_start and (getattr(mt, '_bwd_scs_template', None) is None
                                    or getattr(mt, '_bwd_param_hash', None) != id(ctx.params)):
                import scipy.sparse as _sp_init
                b0 = bundles[0]
                prob0_bwd = mt.perturbed_problem_list[0]
                n_vars_total = y_dim
                n_eq_total = num_eq
                # Set params and mask=1 so all mask entries exist in A
                params_0_np = _slice_params_np(0)
                for pval, pparam in zip(params_0_np, b0["param_order"]):
                    pparam.value = pval
                for dv in b0["dvar_params"]:
                    dv.value = np.zeros(dv.shape)
                for dp in b0.get("eq_dual_params", []):
                    dp.value = np.zeros(dp.shape)
                for dp in b0.get("scalar_ineq_dual_params", []):
                    dp.value = np.zeros(dp.shape)
                for mp in b0.get("scalar_active_mask_params", []):
                    mp.value = np.ones(mp.shape)
                bwd_data_ones, _, _ = prob0_bwd.get_problem_data(
                    solver=cp.SCS, ignore_dpp=True
                )
                bwd_A_csc = _sp_init.csc_matrix(bwd_data_ones['A'])
                mask_row_start = n_vars_total + n_eq_total
                mask_data_indices = np.empty(n_vars_total, dtype=int)
                for k in range(n_vars_total):
                    col_s, col_e = bwd_A_csc.indptr[k], bwd_A_csc.indptr[k+1]
                    rows = bwd_A_csc.indices[col_s:col_e]
                    idx = np.searchsorted(rows, mask_row_start + k)
                    mask_data_indices[k] = col_s + idx
                # Get baseline c with mask=0
                for mp in b0.get("scalar_active_mask_params", []):
                    mp.value = np.zeros(mp.shape)
                bwd_data_base, _, _ = prob0_bwd.get_problem_data(
                    solver=cp.SCS, ignore_dpp=True
                )
                bwd_eps = float(mt._solver_args_bwd.get('eps', 1e-5))
                mt._bwd_scs_template = {
                    'c_base': bwd_data_base['c'].copy(),
                    'b': bwd_data_base['b'].copy(),
                    'P': bwd_data_ones['P'],
                    'A_base_data': bwd_A_csc.data.copy(),
                    'A_base_indices': bwd_A_csc.indices.copy(),
                    'A_base_indptr': bwd_A_csc.indptr.copy(),
                    'A_shape': bwd_A_csc.shape,
                    'mask_data_indices': mask_data_indices,
                    'alpha': float(b0['alpha']),
                    'cone': {
                        'z': bwd_data_ones['dims'].zero,
                        'l': bwd_data_ones['dims'].nonneg,
                    },
                    'scs_args': dict(
                        max_iters=int(mt._solver_args_bwd.get('max_iters', 2500)),
                        eps_abs=bwd_eps, eps_rel=bwd_eps, verbose=False,
                    ),
                }
                mt._bwd_param_hash = id(ctx.params)

            def _solve_perturbed_one(i: int):
                import time as _time
                b = bundles[i]
                prob = mt.perturbed_problem_list[i]

                params_i_np = _slice_params_np(i)
                for pval, pparam in zip(params_i_np, b["param_order"]):
                    pparam.value = pval

                dvals_i = _slice_dvars_np(i)
                for j, v in enumerate(b["variables"]):
                    b["dvar_params"][j].value = dvals_i[j]
                    v.value = ctx.sol_numpy[j][i, ...]

                for j in range(len(b["eq_functions"])):
                    b["eq_dual_params"][j].value = ctx.eq_dual[j][i]

                cap = cap_scalar

                scalar_candidates = []
                for j in b["scalar_scalar_indices"]:
                    sl_s = float(np.asarray(ctx.scalar_ineq_slack[j][i]).reshape(()))
                    lam_s = float(np.asarray(ctx.scalar_ineq_dual[j][i]).reshape(()))
                    lam_s = 0.0 if lam_s < -1e-8 else max(lam_s, 0.0)
                    if sl_s <= mt.slack_tol and lam_s >= mt.dual_cutoff:
                        scalar_candidates.append((lam_s, j))

                if len(scalar_candidates) > 0:
                    scalar_candidates.sort(key=lambda t: t[0])
                    active_scalar = set([j for _, j in scalar_candidates[-cap:]]) if len(scalar_candidates) > cap else set([j for _, j in scalar_candidates])
                else:
                    active_scalar = set()

                for j in range(num_scalar_ineq):
                    lam = np.asarray(ctx.scalar_ineq_dual[j][i], dtype=float)
                    lam = np.where(lam < -1e-8, lam, np.maximum(lam, 0.0))
                    b["scalar_ineq_dual_params"][j].value = lam

                    gshape = b["scalar_ineq_functions"][j].shape
                    if int(np.prod(gshape)) == 1:
                        b["scalar_active_mask_params"][j].value = 1.0 if (j in active_scalar) else 0.0
                    else:
                        sl = np.asarray(ctx.scalar_ineq_slack[j][i], dtype=float)
                        mask = (sl <= mt.slack_tol).astype(np.float64)
                        cap_vec = cap_scalar
                        if mask.sum() > cap_vec:
                            lam_flat = lam.reshape(-1)
                            idx = np.argpartition(lam_flat, -cap_vec)[-cap_vec:]
                            mask_flat = np.zeros_like(lam_flat, dtype=np.float64)
                            mask_flat[idx] = 1.0
                            mask = mask_flat.reshape(lam.shape)
                        b["scalar_active_mask_params"][j].value = mask

                for j in range(len(b["soc_constraints"])):
                    b["soc_dual_params_0"][j].value = np.maximum(ctx.soc_dual_0[j][i], 0.0)
                    b["soc_dual_params_1"][j].value = ctx.soc_dual_1[j][i]

                for j in range(len(b["exp_cones"])):
                    for k in range(3):
                        b["exp_dual_params"][j][k].value = ctx.exp_dual[j][k][i]

                for j in range(len(b["psd_cones"])):
                    b["psd_dual_params"][j].value = ctx.psd_dual[j][i]

                for local_id, j_scalar in enumerate(b["pnorm_ineq_ids"]):
                    for v_id in range(num_vars):
                        b["pnorm_xstar_params"][local_id][v_id].value = ctx.pnorm_xstar[local_id][v_id][i]
                        b["pnorm_grad_params"][local_id][v_id].value = ctx.pnorm_grad[local_id][v_id][i]

                _bwd_t0 = _time.perf_counter()
                if mt._warm_start and hasattr(mt, '_bwd_scs_template'):
                    # ---- Direct SCS path for backward (no CVXPY) ----
                    import scs as _scs_bwd
                    import scipy.sparse as _sp
                    try:
                        tmpl = mt._bwd_scs_template

                        # Backward SCS variable order: x = [y, t] (opposite of forward)
                        # c[0:y_dim] = p + dvar/alpha - lambda (all y-linear terms)
                        # c[y_dim:2*y_dim] = 0 (t has no linear cost)
                        new_c = tmpl['c_base'].copy()
                        dvar_i = dvals_i[0].ravel()
                        lam_i = np.asarray(b["scalar_ineq_dual_params"][0].value, dtype=float).ravel()
                        p_i = params_i_np[1].ravel()
                        new_c[:y_dim] = p_i + dvar_i / tmpl['alpha'] - lam_i

                        # Construct A: copy base, update mask diagonal
                        A_data = tmpl['A_base_data'].copy()
                        mask_i = np.asarray(b["scalar_active_mask_params"][0].value, dtype=float).ravel()
                        mask_data_idx = tmpl['mask_data_indices']
                        for k in range(y_dim):
                            A_data[mask_data_idx[k]] = -mask_i[k]
                        A_sparse = _sp.csc_matrix(
                            (A_data, tmpl['A_base_indices'], tmpl['A_base_indptr']),
                            shape=tmpl['A_shape'],
                        )

                        # Update b: b[y_dim:y_dim+num_eq] = b_eq (index 5 in params)
                        # b vector matches the A rows: [aux(y_dim), eq(num_eq), mask(y_dim)]
                        new_b = tmpl['b'].copy()
                        b_eq_i = params_i_np[5].ravel()
                        new_b[y_dim:y_dim + num_eq] = b_eq_i  # same position confirmed earlier

                        bwd_sd = {
                            'P': tmpl['P'], 'A': A_sparse,
                            'b': new_b, 'c': new_c,
                        }
                        bwd_solver = _scs_bwd.SCS(bwd_sd, tmpl['cone'], **tmpl['scs_args'])

                        # Warm start backward from cached SCS state
                        bwd_ws_key = mt._ws_keys[i] if mt._ws_keys is not None else None
                        bwd_ws = mt._ws_cache_bwd.get(bwd_ws_key) if bwd_ws_key is not None else None
                        if bwd_ws is not None:
                            bwd_sol = bwd_solver.solve(
                                warm_start=True,
                                x=bwd_ws['scs_x'], y=bwd_ws['scs_y'], s=bwd_ws['scs_s'],
                            )
                        else:
                            bwd_sol = bwd_solver.solve(warm_start=False)

                        if bwd_sol['info']['status'] in ('solved', 'solved_inaccurate'):
                            x_bwd = bwd_sol['x']
                            y_bwd = bwd_sol['y']
                            bwd_solver_iters[i] = bwd_sol['info']['iter']

                            y_var = x_bwd[:y_dim]
                            for j, v in enumerate(b["variables"]):
                                vshape = v.shape
                                n_el = int(np.prod(vshape))
                                new_sol_lagrangian[j][i, ...] = y_var[:n_el].reshape(vshape)

                            eq_d = y_bwd[y_dim:y_dim + num_eq]
                            offset = 0
                            for c_id, f in enumerate(b["eq_functions"]):
                                n_el = int(np.prod(f.shape))
                                new_eq_dual[c_id][i, ...] = eq_d[offset:offset+n_el].reshape(f.shape)
                                offset += n_el

                            active_d = y_bwd[y_dim + num_eq:]
                            offset = 0
                            for c_id, c_expr in enumerate(b["active_eq_constraints"]):
                                n_el = int(np.prod(c_expr.shape))
                                new_active_dual[c_id][i, ...] = active_d[offset:offset+n_el].reshape(c_expr.shape)
                                offset += n_el

                            # Cache backward SCS state
                            if bwd_ws_key is not None:
                                mt._ws_cache_bwd[bwd_ws_key] = {
                                    'scs_x': bwd_sol['x'].copy(),
                                    'scs_y': bwd_sol['y'].copy(),
                                    'scs_s': bwd_sol['s'].copy(),
                                }

                            _bwd_solve_times[i] = _time.perf_counter() - _bwd_t0
                            return
                        else:
                            print(f"[backward] SCS direct failed for {i}: {bwd_sol['info']['status']}, fallback")
                    except Exception as e:
                        print(f"[backward] SCS direct error for {i}: {e!r}, fallback")

                # ---- CVXPY fallback path ----
                try:
                    prob.solve(**mt._solver_args_bwd)
                except Exception as e:
                    print(f"[backward] problem {i} perturbed solve failed: {e!r}")
                    try:
                        b["perturbed_problem"].solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, warm_start=True, verbose=False)
                    except Exception as e2:
                        raise RuntimeError(f"[backward] problem {i} perturbed solve failed: {e!r} {e2!r}")

                if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"[backward] perturbed problem {i} status: {prob.status}")

                if prob.solver_stats is not None:
                    bwd_solver_iters[i] = getattr(prob.solver_stats, 'num_iters', 0)

                for j, v in enumerate(b["variables"]):
                    new_sol_lagrangian[j][i, ...] = v.value

                for c_id, c in enumerate(b["eq_constraints"]):
                    new_eq_dual[c_id][i, ...] = c.dual_value

                for c_id, c in enumerate(b["active_eq_constraints"]):
                    new_active_dual[c_id][i, ...] = c.dual_value

                for c_id, c in enumerate(b["soc_lin_constraints"]):
                    dv = c.dual_value
                    new_soc_lam[c_id][i] = 0.0 if dv is None else float(np.asarray(dv).reshape(()))

                for c_id, c in enumerate(b["pnorm_tangent_constraints"]):
                    dv = c.dual_value
                    lam_val = 0.0 if dv is None else float(np.asarray(dv).reshape(()))
                    j_scalar = b["pnorm_ineq_ids"][c_id]
                    mval = float(np.asarray(b["scalar_active_mask_params"][j_scalar].value).reshape(()))
                    if mval < 0.5:
                        lam_val = 0.0
                    new_pnorm_lam[c_id][i] = lam_val

                for c_id, c in enumerate(b["exp_cones"]):
                    shapes3 = [dv.shape for dv in c.dual_variables]
                    dv3 = _split_expcone_dual_value(c.dual_value, shapes3)
                    for k in range(3):
                        new_exp_dual[c_id][k][i, ...] = dv3[k]

                for c_id, c in enumerate(b["psd_cones"]):
                    new_psd_dual[c_id][i, ...] = c.dual_value

            with _limit_native_threads(1):
                futs = [mt._executor.submit(_solve_perturbed_one, i) for i in range(B)]
                for f in futs:
                    f.result()

            if mt._warm_start or mt.verbose:
                total_bwd = sum(bwd_solver_iters)
                bwd_max = max(_bwd_solve_times)
                bwd_sum = sum(_bwd_solve_times)
                print(f"[backward] iters: avg={total_bwd/max(B,1):.0f}, max_solve={bwd_max:.3f}s, sum_solve={bwd_sum:.3f}s")

            new_sol = [to_torch(v, ctx.dtype, ctx.device) for v in new_sol_lagrangian]
            vars_old = [to_torch(ctx.sol_numpy[j], ctx.dtype, ctx.device) for j in range(num_vars)]

            new_eq_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_eq_dual]
            old_eq_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in ctx.eq_dual]

            old_scalar_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in ctx.scalar_ineq_dual]
            new_active_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_active_dual]

            new_exp_dual_t = [
                [to_torch(new_exp_dual[j][k], ctx.dtype, ctx.device) for k in range(3)]
                for j in range(len(ref["exp_cones"]))
            ]
            new_psd_dual_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_psd_dual]

            new_pnorm_lam_t = [to_torch(v, ctx.dtype, ctx.device) for v in new_pnorm_lam]

            params_req = []
            req_grad_mask = []
            for p in ctx.params:
                need = bool(getattr(p, "requires_grad", False))
                q = p.detach()
                if need:
                    q.requires_grad_(True)
                params_req.append(q)
                req_grad_mask.append(need)

            loss = 0.0
            with torch.enable_grad():
                for i in range(B):
                    b = bundles[i]
                    vars_new_i = [v[i] for v in new_sol]
                    vars_old_i = [v[i] for v in vars_old]
                    params_i = slice_params_for_batch(params_req, ctx.batch_sizes, i) if ctx.batch else params_req

                    new_eq_dual_i = [d[i] for d in new_eq_dual_t]
                    old_eq_dual_i = [d[i] for d in old_eq_dual_t]

                    new_scalar_dual_full_i = []
                    ptr = 0
                    for j in range(num_scalar_ineq):
                        if j in b["non_pnorm_set"]:
                            new_scalar_dual_full_i.append(new_active_dual_t[ptr][i])
                            ptr += 1
                        elif j in b["pnorm_set"]:
                            lid = b["pnorm_map"][j]
                            new_scalar_dual_full_i.append(new_pnorm_lam_t[lid][i])
                        else:
                            new_scalar_dual_full_i.append(old_scalar_dual_t[j][i])
                    old_scalar_dual_full_i = [d[i] for d in old_scalar_dual_t]

                    new_exp_dual_i = []
                    for j in range(len(b["exp_cones"])):
                        for k in range(3):
                            new_exp_dual_i.append(new_exp_dual_t[j][k][i])

                    new_psd_dual_i = [d[i] for d in new_psd_dual_t]

                    phi_new = b["phi_torch"](*vars_new_i, *params_i)
                    phi_old = b["phi_torch"](*vars_old_i, *params_i)

                    eq_new = b["eq_dual_term_torch"](*vars_old_i, *params_i, *new_eq_dual_i)
                    eq_old = b["eq_dual_term_torch"](*vars_old_i, *params_i, *old_eq_dual_i)


                    ineq_new = b["ineq_dual_term_torch"](*vars_old_i, *params_i, *new_scalar_dual_full_i)
                    ineq_old = b["ineq_dual_term_torch"](*vars_old_i, *params_i, *old_scalar_dual_full_i)

                    if b["exp_dual_term_torch"] is not None:
                        exp_new = b["exp_dual_term_torch"](*vars_old_i, *params_i, *new_exp_dual_i)
                    else:
                        exp_new = 0.0

                    if b["psd_dual_term_torch"] is not None:
                        psd_new = b["psd_dual_term_torch"](*vars_old_i, *params_i, *new_psd_dual_i)
                    else:
                        psd_new = 0.0

                    loss = loss + (phi_new + ineq_new + eq_new + exp_new + psd_new - phi_old - eq_old - ineq_old)

                loss = mt.alpha * loss

            grads_req = torch.autograd.grad(
                outputs=loss,
                inputs=[q for q, need in zip(params_req, req_grad_mask) if need],
                allow_unused=True,
                retain_graph=False,
            )

            grads = []
            it = iter(grads_req)
            for need in req_grad_mask:
                grads.append(next(it) if need else None)

            return tuple(grads)
    return _FFOLayerFn