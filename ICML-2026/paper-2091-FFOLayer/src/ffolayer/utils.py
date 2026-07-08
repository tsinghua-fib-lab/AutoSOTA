import numpy as np
# import diffcp
import time
from dataclasses import dataclass
from typing import Any
import cvxpy as cp
import tracemalloc
import os
import linecache
import torch
from scipy.linalg import block_diag
import pathlib

n_threads = os.cpu_count()

def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def to_numpy(x):
    # convert torch tensor to numpy array
    if isinstance(x, torch.Tensor):
        if x.device.type == 'cuda':
            return x.cpu().detach().double().numpy()
        else:
            return x.detach().double().numpy()
    else:
        return np.array(x)


def to_torch(x, dtype, device):
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device)

def slice_params_for_batch(params_req, batch_sizes, i):
    """Pick p[i] if that parameter was batched; else p."""
    out = []
    for p, bs in zip(params_req, batch_sizes):
        out.append(p[i] if bs > 0 else p)
    return out

def make_mask_torch_for_i(i, inequality_dual, inequality_functions, dual_cutoff, ctx):
    mask_list = []
    for j in range(len(inequality_functions)):
        lam_ji = inequality_dual[j][i]
        mask_np = (lam_ji > dual_cutoff).astype(np.float64)
        mask_list.append(to_torch(mask_np, ctx.dtype, ctx.device))
    return mask_list
    
def extract_nBatch(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndim == dim:
            return param.size(0)
    return 1

def expandParam(X, nBatch, nDim):
    if X.ndim in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndim == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")

# def forward_numpy(params_numpy, context):
#     """Forward pass in numpy."""
#     
#     info = {}
#     
#     if context.gp:
#         param_map = {}
#         # construct a list of params for the DCP problem
#         for param, value in zip(context.param_order, params_numpy):
#             if param in context.old_params_to_new_params:
#                 new_id = context.old_params_to_new_params[param].id
#                 param_map[new_id] = np.log(value)
#             else:
#                 new_id = param.id
#                 param_map[new_id] = value
#         params_numpy = [param_map[pid] for pid in context.param_ids]
#     
#     # canonicalize problem
#     start = time.time()
#     As, bs, cs, cone_dicts, shapes = [], [], [], [], []
#     for i in range(context.batch_size):
#         params_numpy_i = [
#             p if sz == 0 else p[i]
#             for p, sz in zip(params_numpy, context.batch_sizes)]
#         c, _, neg_A, b = context.compiler.apply_parameters(
#             dict(zip(context.param_ids, params_numpy_i)),
#             keep_zeros=True)
#         A = -neg_A  # cvxpy canonicalizes -A
#         As.append(A)
#         bs.append(b)
#         cs.append(c)
#         cone_dicts.append(context.cone_dims)
#         shapes.append(A.shape)
#     info['canon_time'] = time.time() - start
#     info['shapes'] = shapes
# 
#     # compute solution and derivative function
#     start = time.time()
#     try:
#         if context.solve_and_derivative:
#             xs, _, _, _, DT_batch = diffcp.solve_and_derivative_batch(
#                 As, bs, cs, cone_dicts, **context.solver_args)
#             info['DT_batch'] = DT_batch
#         else:
#             xs, _, _ = diffcp.solve_only_batch(
#                 As, bs, cs, cone_dicts, **context.solver_args)
#     except diffcp.SolverError as e:
#         print(
#             "Please consider re-formulating your problem so that "
#             "it is always solvable or increasing the number of "
#             "solver iterations.")
#         raise e
#     info['solve_time'] = time.time() - start
# 
#     # extract solutions and append along batch dimension
#     start = time.time()
#     sol = [[] for i in range(len(context.variables))]
#     for i in range(context.batch_size):
#         sltn_dict = context.compiler.split_solution(
#             xs[i], active_vars=context.var_dict)
#         for j, v in enumerate(context.variables):
#             sol[j].append(np.expand_dims(sltn_dict[v.id], axis=0))
#     sol = [np.concatenate(s, axis=0) for s in sol]
# 
#     if not context.batch:
#         sol = [np.squeeze(s, axis=0) for s in sol]
# 
#     if context.gp:
#         sol = [np.exp(s) for s in sol]
#         info['sol'] = sol
#             
#     return sol, info

def forward_single_np(Q, p, G, h, A, b,
                      solver='GUROBI',
                      solver_opts={"verbose": False}):
    nz, neq, nineq = p.shape[0], A.shape[0] if A is not None else 0, G.shape[0]

    z_ = cp.Variable(nz)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T @ z_)
    eqCon = A @ z_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)
        ineqCon = G @ z_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, **solver_opts) # max_iters=5000)
    # prob.solve()
    # prob.solve(adaptive_rho = False)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    zhat = np.array(z_.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks


def forward_single_np_eq_cst(Q, p, G, h, A, b):
    """ -> kamo
    min_z 1/2 * z.T Q z + p.T z
    s.t. A z = b ; G z <= h
    """
    nz, neq, nineq = p.shape[0], A.shape[0] if A is not None else 0, G.shape[0] if G is not None else 0

    z_ = cp.Variable(nz)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T @ z_)
    eqCon = A @ z_ == b if neq > 0 else None
    if nineq > 0:
        slacks = cp.Variable(nineq)
        ineqCon = G @ z_ + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    
    prob.solve(solver=cp.GUROBI, verbose=False, **{"Threads": n_threads, "OutputFlag": 0} ) # max_iters=5000)
    # prob.solve()
    # prob.solve(adaptive_rho = False)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    zhat = np.array(z_.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks


def forward_batch_np(Q, p, G, h, A, b,
                     solver='GUROBI',
                     solver_opts={"verbose": False}):
    """ -> kamo
    Q : (nb, nz, nz)
    p : (nb, nz)
    G : (nb, nineq, nz)
    h : (nb, nineq)
    A : (nb, neq, nz)
    b : (nb, neq)
    """
    nb = p.shape[0]
    nz, neq, nineq = p.shape[1], A.shape[1] if A is not None else 0, G.shape[1] if G is not None else 0

    z_ = cp.Variable(nz * nb)
    Q_ = block_diag(*Q)
    p_ = p.reshape(-1)

    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q_) + p_.T @ z_)
    eqCon = None
    if neq > 0:
        A_ = block_diag(*A)
        b_ = b.reshape(-1)
        eqCon = A_ @ z_ == b_
    if nineq > 0:
        slacks = cp.Variable(nineq * nb)
        G_ = block_diag(*G)
        h_ = h.reshape(-1)
        ineqCon = G_ @ z_ + slacks == h_
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None
    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cp.Problem(obj, cons)
    # prob.solve(solver=cp.GUROBI, verbose=False, **{"Threads": n_threads, "OutputFlag": 0} )
    prob.solve(solver=solver, **solver_opts)
    # prob.solve(solver='SCS', max_iters=100, eps=1e-7, **solver_opts) # max_iters=5000)
    # prob.solve()
    # prob.solve(adaptive_rho = False)  # solver=cp.SCS, max_iters=5000, verbose=False)
    # prob.solve(solver=cp.SCS, max_iters=10000, verbose=True)
    assert('optimal' in prob.status)
    zhat = np.array(z_.value).reshape(nb, nz)
    nu = np.array(eqCon.dual_value).reshape(nb, neq) if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).reshape(nb, nineq)
        slacks = np.array(slacks.value).reshape(nb, nineq)
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def _dump_cvxpy(
    save_dir, file_name, batch_i, *,
    ctx,
    param_order, variables,
    alpha, dual_cutoff,
    solver_used, trigger,
    params_numpy,
    sol_numpy,
    equality_dual,
    inequality_dual,
    slack,
    new_sol_lagrangian,
    new_equality_dual,
    new_active_dual,
    active_mask_params,
    dvars_numpy,
):
    p = pathlib.Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)

    meta = {
        "tag": f"{file_name}",
        "batch_i": int(batch_i),
        "dtype": str(ctx.dtype),
        "device": str(ctx.device),
        "alpha": float(alpha),
        "dual_cutoff": float(dual_cutoff),
        "solver_used": solver_used,
        "trigger": trigger,
        "param_count": len(param_order),
        "var_count": len(variables),
        "eq_count": len(equality_dual),
        "ineq_count": len(inequality_dual),
    }
    # (p / "meta.json").write_text(json.dumps(meta, indent=2))

    arrs = {}

    for k in range(len(param_order)):
        arrs[f"param_{k}"] = _np(params_numpy[k][batch_i if ctx.batch else 0])

    for j in range(len(variables)):
        arrs[f"y_old_{j}"] = _np(sol_numpy[j][batch_i])
    for l in range(len(equality_dual)):
        arrs[f"dual_eq_old_{l}"] = _np(equality_dual[l][batch_i])
    for m in range(len(inequality_dual)):
        lam = inequality_dual[m][batch_i]
        arrs[f"dual_ineq_old_{m}"] = _np(lam)
        arrs[f"slack_old_{m}"]     = _np(slack[m][batch_i])

        try:
            mask_val = active_mask_params[m].value
        except Exception:
            mask_val = (lam > dual_cutoff).astype(np.float64)
        arrs[f"active_mask_used_{m}"] = _np(mask_val)

    if dvars_numpy is not None:
        for j in range(len(variables)):
            dv = dvars_numpy[j]
            if dv is None:
                arrs[f"dvars_is_none_{j}"] = np.array([True])
            else:
                arrs[f"dvars_{j}"] = _np(dv[batch_i])

    np.savez_compressed(p / f"{file_name}.npz", **arrs)