import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
import time
import cvxpy as cp
import scipy.sparse as sp

from .utils import forward_single_np_eq_cst, forward_batch_np, extract_nBatch, expandParam

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# from cvxpylayers.torch import CvxpyLayer

# class QPSolvers(Enum):
#     PDIPM_BATCHED = 1
#     CVXPY = 2

# class ffoqp(torch.nn.Module):
#     def __init__(self, eps=1e-12, verbose=0, notImprovedLim=3, maxiter=20, solver=None, lamb=100):
#         super(ffoqp, self).__init__()
#         self.eps = eps
#         self.verbose = verbose
#         self.notImprovedLim = notImprovedLim
#         self.maxiter = maxiter
#         self.solver = solver if solver is not None else QPSolvers.CVXPY
#         self.lamb = lamb

def _bpqp_np(x):
    return x.detach().cpu().numpy()

def _bpqp_sym(P):
    return 0.5 * (P + P.T)

def _bpqp_osqp_solve(P_csc, q_np, A_csc, l_np, u_np):
    import osqp
    prob = osqp.OSQP()
    prob.setup(P_csc, q_np, A_csc, l_np, u_np,
               verbose=False,
               eps_abs=1e-5,
               eps_rel=1e-5,
               eps_prim_inf=1e-5,
               eps_dual_inf=1e-5)
    res = prob.solve()
    if res.x is None:
        raise RuntimeError(res.info.status)
    return res.x.astype(np.float64), res.y.astype(np.float64)

def _bpqp_pack_osqp(P, q, G, h, A, b):
    Pn, qn, Gn, hn, An, bn = [_bpqp_np(x) for x in [P, q, G, h, A, b]]
    Pn = _bpqp_sym(Pn).astype(np.float64)
    qn = qn.reshape(-1).astype(np.float64)
    Gn = Gn.astype(np.float64)
    hn = hn.reshape(-1).astype(np.float64)
    An = An.astype(np.float64)
    bn = bn.reshape(-1).astype(np.float64)
    m, p = Gn.shape[0], An.shape[0]
    if p > 0:
        Aos = sp.csc_matrix(np.vstack([Gn, An]))
        l = np.hstack([-np.inf * np.ones(m), bn])
        u = np.hstack([hn, bn])
    else:
        Aos = sp.csc_matrix(Gn)
        l = -np.inf * np.ones(m)
        u = hn
    return sp.csc_matrix(Pn), qn, Aos, l.astype(np.float64), u.astype(np.float64), m, p, Gn, hn, An


def add_diag_(M, eps):
    if eps and eps > 0:
        d = M.diagonal(dim1=-2, dim2=-1)
        d.add_(eps)

def compact_active_rows(A):  # A: (B, m, n)
    B, m, n = A.shape
    As, idx = [], []
    for b in range(B):
        rowmask = (A[b].abs().amax(dim=-1) > 0)  # non-zero rows
        Ab = A[b][rowmask]
        As.append(Ab)
        idx.append(rowmask.nonzero(as_tuple=False).squeeze(-1))
    return As, idx 

def kkt_schur_complement(Q, A, delta):
    eps_q = 1e-8
    eps_s = 1e-12
    if delta.dim() == 2:
        delta = delta.unsqueeze(-1)          # (B,n,1)

    B, n, _ = Q.shape
    m = A.shape[1] if A.numel() > 0 else 0

    I_n = torch.eye(n, dtype=Q.dtype, device=Q.device)
    L = torch.linalg.cholesky(Q + eps_q * I_n)  # (B,n,n) supports batch

    if m == 0:
        dz = -torch.cholesky_solve(delta, L)    # (B,n,1)
        return dz.squeeze(-1), Q.new_zeros(B, 0)

    AT = A.transpose(-1, -2)                   # (B,n,m)

    Winv = torch.cholesky_solve(AT, L)         # (B,n,m) = Q^{-1} A^T
    y    = torch.cholesky_solve(delta, L)      # (B,n,1) = Q^{-1} delta

    S = A @ Winv                               # (B,m,m)
    if eps_s is not None and eps_s > 0:
        I_m = torch.eye(m, dtype=Q.dtype, device=Q.device)
        S = S + eps_s * I_m

    rhs = -(A @ y)                             # (B,m,1)
    try:
        Ls = torch.linalg.cholesky(S)
        dlam = torch.cholesky_solve(rhs, Ls)   # (B,m,1)
    except RuntimeError:
        # when the row rank is not full/ill-conditioned, QR-based is faster than gelsd
        dlam = torch.linalg.lstsq(S, rhs, driver='gels').solution

    dz = -torch.cholesky_solve(delta + AT @ dlam, L)  # (B,n,1)

    return dz.squeeze(-1), dlam.squeeze(-1)

def make_schur_op(A, L, eps_s):
    AT = [a.transpose(-1, -2).contiguous() for a in A]  # ragged list
    def Aop(v_list):  # v_list: list of (m_b,1)
        outs = []
        for a, at, vb in zip(A, AT, v_list):
            # w = A Q^{-1} A^T v
            w = torch.cholesky_solve(at @ vb, L)  # (n,1)
            out = a @ w                           # (m_b,1)
            if eps_s and eps_s > 0:
                out = out + eps_s * vb
            outs.append(out)
        return outs
    return Aop

def cg_solve_list(Aop, b_list, x0_list=None, maxit=50, tol=1e-6):
    xs = []
    for i, b in enumerate(b_list):
        m = b.shape[0]
        x = torch.zeros_like(b) if (x0_list is None or x0_list[i] is None) else x0_list[i]
        r = b - Aop([x])[0]
        p = r.clone()
        rsold = (r*r).sum()
        bnrm = b.norm()
        for _ in range(maxit):
            Ap = Aop([p])[0]
            denom = (p*Ap).sum()
            alpha = rsold / (denom + 1e-40)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = (r*r).sum()
            if rsnew.sqrt() <= tol * (bnrm + 1e-40):
                break
            p = r + (rsnew/rsold) * p
            rsold = rsnew
        xs.append(x)
    return xs

def kkt_schur_fast(Q, A, delta, L_cached=None, eps_q=1e-8, eps_s=1e-10,
                   cg_threshold=2560, cg_maxit=50, cg_tol=1e-6, warm_dlam_list=None):
    if delta.dim() == 3 and delta.size(-1) == 1:
        delta = delta.squeeze(-1)                   # (B,n)
    B, n, _ = Q.shape

    Q = Q.contiguous()
    A = A.contiguous()
    delta = delta.contiguous()

    if L_cached is None:
        Q_ = Q.clone()                               # do not destroy the original tensor
        add_diag_(Q_, eps_q)
        L = torch.linalg.cholesky(Q_)               # (B,n,n)
    else:
        L = L_cached


    if A.numel() == 0:
        dz = -torch.cholesky_solve(delta.unsqueeze(-1), L).squeeze(-1)
        return dz, [Q.new_zeros(0) for _ in range(B)]

    Alist, idxlist = compact_active_rows(A)          # ragged each Ab:(m_b,n)

    y = -torch.cholesky_solve(delta.unsqueeze(-1), L)  # (B,n,1) with negative sign, corresponding to Q dz = -(...)

    dlam_list, dz_list = [], []
    Aop = make_schur_op(Alist, L, eps_s)

    rhs_list = [(a @ y[b]) for b, a in enumerate(Alist)]  # (m_b,1)

    for b, Ab in enumerate(Alist):
        m_b = Ab.shape[0]
        if m_b == 0:
            dlam_b = Ab.new_zeros(0, 1)
        elif m_b <= cg_threshold:
            ATb = Ab.transpose(-1, -2).contiguous()
            Winv_b = torch.cholesky_solve(ATb, L[b:b+1])  # (1,n,m_b)
            Sb = Ab @ Winv_b.squeeze(0)                   # (m_b,m_b)
            add_diag_(Sb, eps_s)
            Ls = torch.linalg.cholesky(Sb)
            dlam_b = torch.cholesky_solve(rhs_list[b], Ls)  # (m_b,1)
        else:
            raise NotImplementedError("CG not implemented")
            # x0 = None if warm_dlam_list is None else warm_dlam_list[b]
            # dlam_b = cg_solve_list(Aop, [rhs_list[b]], [x0], maxit=cg_maxit, tol=cg_tol)[0]
        dlam_list.append(dlam_b)

    dz = y.clone()  # (B,n,1)
    for b, (Ab, dlam_b) in enumerate(zip(Alist, dlam_list)):
        if dlam_b.numel() == 0:
            continue
        ATd = Ab.transpose(-1, -2) @ dlam_b            # (n,1)
        dz[b:b+1] -= torch.cholesky_solve(ATd.unsqueeze(0), L[b:b+1])

    dz = dz.squeeze(-1)

    M = A.shape[1]  # total #constraints before compaction
    dlam = Q.new_zeros((B, M))
    for b, (dl_b, idx_b) in enumerate(zip(dlam_list, idxlist)):
        if dl_b.numel():
            # dl_b is (m_b, 1) -> (m_b,)
            dlam[b, idx_b] = dl_b.squeeze(-1)

    return dz, dlam

def FFOQPLayer(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, alpha=100, check_Q_spd=False, chunk_size=100,
          solver='qpsolvers', solver_opts={"verbose": False},
          exact_bwd_sol=True, slack_cutoff=1e-8, cvxpy_instance=None):
    class QPFunctionFn(torch.autograd.Function):
        @staticmethod
        @torch.no_grad()
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            # p_ = p_ + 1/alpha * torch.randn_like(p_)
            start_time = time.time()
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            if A_.numel() > 0:
                A, _ = expandParam(A_, nBatch, 3)
            else:
                A = None
            if b_.numel() > 0:
                b, _ = expandParam(b_, nBatch, 2)
            else:
                b = None

            if check_Q_spd:
                try:
                    torch.linalg.cholesky(Q)
                except:
                    raise RuntimeError('Q is not SPD.')

            _, nineq, nz = G.size()
            neq = A.size(1) if A is not None else 0
            assert(neq > 0 or nineq > 0)
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            if nineq > 0 and solver == 'qpsolvers':
                from dqp import dQP

                dQP_settings = dQP.build_settings(
                        solve_type="dense",
                        qp_solver="gurobi",
                        # lin_solver="scipy LU",
                    )
                dQP_layer = dQP.dQP_layer(settings=dQP_settings)
                if nBatch == 1:
                    Q = Q.squeeze(0)  # (n,n)
                    p = p.squeeze(0)  # (n,)
                    G = G.squeeze(0)  # (m,n)
                    h = h.squeeze(0)  # (m,)
                    A = A.squeeze(0) if A is not None else None
                    b = b.squeeze(0) if b is not None else None
                zhats, nus, lams, solve_time, total_forward_time = dQP_layer(
                    Q, p, G, h, A, b
                )
                if isinstance(nus, list):
                    nus = torch.vstack(nus)
                zhats = zhats.to(dtype=Q.dtype)
                lams = lams.to(dtype=Q.dtype)
                nus = nus.to(dtype=Q.dtype)
                
                if nBatch == 1:
                    G = G.unsqueeze(0)  # (1,m,n)
                    h = h.unsqueeze(0)  # (1,m)
                Gz = torch.bmm(G, zhats.unsqueeze(-1)).squeeze(-1)
                slacks = torch.clamp(h - Gz, min=0.0)

                slacks = slacks.to(device=zhats.device, dtype=Q.dtype)            
            elif nineq > 0 and solver == 'PDIPM':
                from qpth.solvers.pdipm import batch as pdipm_b

                if cvxpy_instance is None:
                    ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
                    zhats, nus, lams, slacks = pdipm_b.forward(
                        Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
                        eps, verbose, notImprovedLim, maxIter)
                else:
                    cvxpy_params = cvxpy_instance["params"]
                    cvxpy_problem = cvxpy_instance["problem"]
                    cvxpy_variables = cvxpy_instance["variables"]
                    eq_constraints = cvxpy_instance["eq_constraints"]
                    ineq_constraints = cvxpy_instance["ineq_constraints"]
                    eq_functions = cvxpy_instance["eq_functions"]
                    ineq_functions = cvxpy_instance["ineq_functions"]
                    #parameters = [Q_cp, q_cp, G_cp, h_cp]
                    params_torch = [Q, p, G, h]
                    params_numpy = [param.detach().cpu().numpy() for param in params_torch]
                    
                    sol_numpy = [np.empty((nBatch,) + v.shape, dtype=float) for v in cvxpy_variables]
                    eq_dual = [np.empty((nBatch,) + f.shape, dtype=float) for f in eq_functions]
                    ineq_dual = [np.empty((nBatch,) + g.shape, dtype=float) for g in ineq_functions]
                    ineq_slack_residual = [np.empty((nBatch,) + g.shape, dtype=float) for g in ineq_functions]
                    
                    for i in range(nBatch):
                        for p_val, param_obj in zip(params_numpy, cvxpy_params):
                            param_obj.value = p_val[i]
                        
                        cvxpy_problem.solve(solver=cp.OSQP, warm_start=False, verbose=False, eps_abs=1e-3, eps_rel=1e-3, max_iter=250)
                        
                        sol_i = [v.value for v in cvxpy_variables]
                        eq_i = [c.dual_value for c in eq_constraints]
                        ineq_i = [c.dual_value for c in ineq_constraints]
                        slack_i = [np.maximum(-expr.value, 0.0) for expr in ineq_functions]
                        
                        for v_id, v in enumerate(cvxpy_variables):
                            sol_numpy[v_id][i, ...] = sol_i[v_id]

                        for c_id, c in enumerate(eq_constraints):
                            eq_dual[c_id][i, ...] = eq_i[c_id]

                        for c_id, c in enumerate(ineq_constraints):
                            ineq_dual[c_id][i, ...] = ineq_i[c_id]

                        for c_id, expr in enumerate(ineq_functions):
                            g_val = expr.value
                            s_val = -g_val
                            s_val = np.maximum(s_val, 0.0)
                            ineq_slack_residual[c_id][i, ...] = slack_i[c_id]
                    
                    device = Q.device
                    dtype = Q.dtype

                    zhats  = [torch.from_numpy(arr).to(device=device, dtype=dtype) for arr in sol_numpy][0]
                    lams   = [torch.from_numpy(arr).to(device=device, dtype=dtype) for arr in ineq_dual][0]
                    nus = [torch.from_numpy(arr).to(device=device, dtype=dtype) for arr in eq_dual]
                    if len(nus)!=0:
                        nus = nus[0]
                    else:
                        nus=lams
                    
                    slacks = [torch.from_numpy(arr).to(device=device, dtype=dtype) for arr in ineq_slack_residual][0]
            elif nineq > 0 and solver == 'OSQP_NATIVE':
                import osqp

                device = Q.device
                dtype = Q.dtype
                zhats = torch.empty(nBatch, nz, device=device, dtype=dtype)
                lams = torch.empty(nBatch, nineq, device=device, dtype=dtype)
                nus = torch.empty(nBatch, neq, device=device, dtype=dtype) if neq > 0 else torch.empty(0, device=device, dtype=dtype)
                slacks = torch.empty(nBatch, nineq, device=device, dtype=dtype)

                for i in range(nBatch):
                    Pi = Q[i]
                    qi = p[i]
                    Gi = G[i]
                    hi = h[i]
                    if neq > 0:
                        Ai = A[i]
                        bi = b[i]
                    else:
                        Ai = Q.new_zeros((0, nz), device=device, dtype=dtype)
                        bi = Q.new_zeros((0,), device=device, dtype=dtype)

                    P_csc, qn, Aos, l, u, m_i, p_i, Gn, hn, An = _bpqp_pack_osqp(Pi, qi, Gi, hi, Ai, bi)
                    x_np, y_np = _bpqp_osqp_solve(P_csc, qn, Aos, l, u)

                    zhats[i] = torch.from_numpy(x_np).to(device=device, dtype=dtype)
                    lam_np = y_np[:m_i]
                    lams[i] = torch.from_numpy(lam_np).to(device=device, dtype=dtype)

                    if neq > 0:
                        nu_np = y_np[m_i:m_i + p_i]
                        nus[i] = torch.from_numpy(nu_np).to(device=device, dtype=dtype)

                    Gx = Gn @ x_np
                    slack_np = np.maximum(hn - Gx, 0.0)
                    slacks[i] = torch.from_numpy(slack_np).to(device=device, dtype=dtype)
            elif nineq > 0:
                print("Using {} solver".format(solver))
                zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
                lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
                nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) \
                    if ctx.neq > 0 else torch.Tensor()
                slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)

                for i in range(0, nBatch, chunk_size):
                    if chunk_size > 1:
                        size = min(chunk_size, nBatch - i)
                        Ai, bi = (A[i:i+size], b[i:i+size]) if neq > 0 else (None, None)
                        _, zhati, nui, lami, si = forward_batch_np(
                            *[x.cpu().numpy() if x is not None else None
                            for x in (Q[i:i+size], p[i:i+size], G[i:i+size], h[i:i+size], Ai, bi)],
                            solver=solver, solver_opts=solver_opts)
                        i = slice(i, i + size)
                    else:
                        Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                        _, zhati, nui, lami, si = forward_single_np_eq_cst(
                            *[x.cpu().numpy() if x is not None else None
                            for x in (Q[i], p[i], G[i], h[i], Ai, bi)])
            
                    zhats[i] = torch.Tensor(zhati)
                    lams[i] = torch.Tensor(lami)
                    slacks[i] = torch.Tensor(si)
                    if neq > 0:
                        nus[i] = torch.Tensor(nui)
            else:
                raise NotImplementedError("Solver not implemented")

            # ctx.vals = vals
            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks

            ctx.save_for_backward(zhats, lams, nus, Q_, p_, G_, h_, A_, b_)
            # print('value', vals)
            # print('solution', zhats)
            return zhats

        @staticmethod
        def backward(ctx, grad_output):
            # Backward pass to compute gradients with respect to inputs
            zhats, lams, nus, Q_, p_, G_, h_, A_, b_ = ctx.saved_tensors
            lams = torch.clamp(lams, min=0)
            slacks = torch.clamp(ctx.slacks, min=0)

            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            # Formulate a different QP to solve
            # L = f + \alpha * (g + lams * h - g^*) + \alpha^2 * |h_+|^2
            Q, Q_e = expandParam(Q_, nBatch, 3)
            p, p_e = expandParam(p_, nBatch, 2)
            G, G_e = expandParam(G_, nBatch, 3)
            h, h_e = expandParam(h_, nBatch, 2)
            A, A_e = expandParam(A_, nBatch, 3)
            b, b_e = expandParam(b_, nBatch, 2)

            Q, p, G, h, A, b = Q.to(zhats.device), p.to(zhats.device), G.to(zhats.device), h.to(zhats.device), A.to(zhats.device), b.to(zhats.device)

            # Running gradient descent for a few iterations
            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0

            delta_directions = grad_output.unsqueeze(-1)
            zhats = zhats.unsqueeze(-1).detach()

            start_time = time.time()
            # active_constraints = (lams > dual_cutoff).unsqueeze(-1).float()
            active_constraints = (slacks <= slack_cutoff).unsqueeze(-1).to(Q.dtype)
            G_active = G * active_constraints
            #h_active = h.unsqueeze(-1) * active_constraints
            #newp = p.unsqueeze(-1) + delta_directions / alpha

            dzhat = torch.Tensor(nBatch, nz, 1).type_as(Q)
            dnu = torch.Tensor(nBatch, nineq + neq).type_as(Q)

            if neq > 0:
                G_active = torch.cat((G_active, A), dim=1)
                #h_active = torch.cat((h_active, b.unsqueeze(-1)), dim=1)

            if exact_bwd_sol:
                # kkt_schur_fast_fn = torch.compile(kkt_schur_fast, mode="max-autotune")
                delta_directions = delta_directions.to(Q.dtype)
                _dzhat, _dnu = kkt_schur_fast(Q, G_active, delta_directions)
                dzhat.copy_(_dzhat.unsqueeze(-1))
                dnu.copy_(_dnu)
            else:
                for i in range(0, nBatch, chunk_size):
                    if chunk_size > 1:
                        size = min(chunk_size, nBatch - i)
                        i = slice(i, i + size)
                        _, zhati, nui, _, _ = forward_batch_np(
                            *[x.cpu().numpy() if x is not None else None
                              for x in (Q[i], grad_output[i], None, None, G_active[i], torch.zeros(G_active[i].shape[0], G_active[i].shape[1]))],
                            solver=solver, solver_opts=solver_opts)
                    else:
                        _, zhati, nui, _, _ = forward_single_np_eq_cst(
                            *[x.cpu().numpy() if x is not None else None
                              for x in (Q[i], grad_output[i], None, None, G_active[i], torch.zeros(G_active[i].shape[0]))])

                    dzhat[i, :, 0] = torch.Tensor(zhati)
                    dnu[i] = torch.Tensor(nui)

            start_time = time.time()
            with torch.enable_grad():
                Q_torch = Q.detach().clone().requires_grad_(True)
                p_torch = p.detach().clone().requires_grad_(True)
                G_torch = G.detach().clone().requires_grad_(True)
                h_torch = h.detach().clone().requires_grad_(True)
                A_torch = A.detach().clone().requires_grad_(True)
                b_torch = b.detach().clone().requires_grad_(True)
               
                objectives = (dzhat.transpose(-1,-2) @ Q_torch @ zhats + p_torch.unsqueeze(1) @ dzhat).squeeze(-1,-2)
                violations = G_torch @ zhats - h_torch.unsqueeze(-1)

                ineq_penalties = dnu[:, :nineq].unsqueeze(1) @ (violations * active_constraints)

                if neq > 0:
                    eq_violations = A_torch @ zhats - b_torch.unsqueeze(-1)
                    eq_penalties = dnu[:, nineq:].unsqueeze(1) @ eq_violations
                else:
                    eq_penalties = 0

                lagrangians = objectives + ineq_penalties + eq_penalties
                loss = torch.sum(lagrangians)
                loss.backward()

                Q_grad = Q_torch.grad.detach()
                p_grad = p_torch.grad.detach()
                G_grad = G_torch.grad.detach()
                h_grad = h_torch.grad.detach()
                if neq > 0:
                    A_grad = A_torch.grad.detach()
                    b_grad = b_torch.grad.detach()
                    # A_grad = torch.zeros_like(A)
                    # b_grad = torch.zeros_like(b)
                else:
                    A_grad = torch.zeros_like(A)
                    b_grad = torch.zeros_like(b)

            return (Q_grad, p_grad, G_grad, h_grad, A_grad, b_grad)  # (None,) * len(ctx.saved_tensors)

    return QPFunctionFn.apply

