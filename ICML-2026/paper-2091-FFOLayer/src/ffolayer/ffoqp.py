import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor
from torch.autograd import Function

import numpy as np
import scipy
import time
import cvxpy
# import solvers
# from qpthlocal.solvers.pdipm import batch as pdipm_b
# from qpthlocal.solvers.pdipm import spbatch as pdipm_spb
# from qpthlocal.solvers.cvxpy import forward_single_np
from .utils import forward_single_np
from enum import Enum
from .utils import extract_nBatch, expandParam
from typing import cast, List, Optional, Union

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

def ffoqp(eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20, lamb=100, check_Q_spd=True,
          solver='GUROBI', solver_opts={"verbose": False}):

    class QPFunctionFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q_, p_, G_, h_, A_, b_):
            # p_ = p_ + 1/lamb * torch.randn_like(p_)
            start_time = time.time()
            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            Q, _ = expandParam(Q_, nBatch, 3)
            p, _ = expandParam(p_, nBatch, 2)
            G, _ = expandParam(G_, nBatch, 3)
            h, _ = expandParam(h_, nBatch, 2)
            A, _ = expandParam(A_, nBatch, 3)
            b, _ = expandParam(b_, nBatch, 2)

            if check_Q_spd:
                try:
                    torch.linalg.cholesky(Q)
                except:
                    raise RuntimeError('Q is not SPD.')

            _, nineq, nz = G.size()
            neq = A.size(1) if A.nelement() > 0 else 0
            assert(neq > 0 or nineq > 0)
            ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

            # if solver == QPSolvers.PDIPM_BATCHED:
            #     ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G, A)
            #     zhats, nus, lams, slacks = pdipm_b.forward(
            #         Q, p, G, h, A, b, ctx.Q_LU, ctx.S_LU, ctx.R,
            #         eps, verbose, notImprovedLim, maxIter)
            # elif solver == QPSolvers.CVXPY:
            vals = torch.Tensor(nBatch).type_as(Q)
            zhats = torch.Tensor(nBatch, ctx.nz).type_as(Q)
            lams = torch.Tensor(nBatch, ctx.nineq).type_as(Q)
            nus = torch.Tensor(nBatch, ctx.neq).type_as(Q) \
                if ctx.neq > 0 else torch.Tensor()
            slacks = torch.Tensor(nBatch, ctx.nineq).type_as(Q)

            for i in range(nBatch):
                Ai, bi = (A[i], b[i]) if neq > 0 else (None, None)
                vals[i], zhati, nui, lami, si = forward_single_np(
                    *[x.cpu().numpy() if x is not None else None
                    for x in (Q[i], p[i], G[i], h[i], Ai, bi)],
                    solver=solver, solver_opts=solver_opts)
                # if zhati[0] is None:
                #     import IPython, sys; IPython.embed(); sys.exit(-1)
                zhats[i] = torch.Tensor(zhati)
                lams[i] = torch.Tensor(lami)
                slacks[i] = torch.Tensor(si)
                if neq > 0:
                    nus[i] = torch.Tensor(nui)

            ctx.vals = vals
            ctx.lams = lams
            ctx.nus = nus
            ctx.slacks = slacks

            # else:
            #     raise NotImplementedError("Solver not implemented")

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

            nBatch = extract_nBatch(Q_, p_, G_, h_, A_, b_)
            # Formulate a different QP to solve
            # L = f + \lamb * (g + lams * h - g^*) + \lamb^2 * |h_+|^2
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

            # Iterative solution
            # print('newzhat shape:', zhats.shape)
            # print('Q shape:', Q.shape)
            iterative = False
            if iterative:
                with torch.enable_grad():
                    newzhat = zhats.clone().detach().requires_grad_(True)
                    optimizer = torch.optim.Adam([newzhat], lr=1e-3)
                    gd_maxiter = 1 # int(np.sqrt(lamb) * 100)
                    # print('dual solutions', lams)
                    for i in range(gd_maxiter):
                        objectives = (0.5 * newzhat.transpose(-1,-2) @ Q.detach() @ newzhat + (p.detach().unsqueeze(1) + delta_directions.transpose(-1,-2) / lamb) @ newzhat).squeeze(-1,-2)
                        violations = G.detach() @ newzhat - h.unsqueeze(-1)
                        active_constraints = (lams > 1e-5).unsqueeze(-1).float()
                        ineq_penalties = lams.unsqueeze(1) @ violations + 0.5 * lamb * torch.sum((violations * active_constraints) ** 2, dim=(-1,-2))
                        if neq > 0:
                            eq_penalties = nus.unsqueeze(1) @ (A.detach() @ newzhat.unsqueeze(-1) - b.detach().unsqueeze(-1))
                        else:
                            eq_penalties = 0
                        # print('obj, vio, active_constraints, lamb, ineq_penality shape:', objectives.shape, violations.shape, active_constraints.shape, lamb, ineq_penalties.shape)
                        lagrangians = objectives + ineq_penalties + eq_penalties
                        loss = torch.sum(lagrangians)
                        # print('Iteration {}: loss: {}, obj: {}, violation: {}'.format(i, loss, objectives.mean(), violations.mean()))
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                # print('new zhat', newzhats.detach())

                # Clampping the dual solutions
                # lams = torch.clamp(lams, max=100)

            else:
                # Deterministic solution by solving a new QP
                temperature = 10
                start_time = time.time()
                active_constraints = torch.tanh(lams * temperature).unsqueeze(-1)
                # active_constraints = (lams > 1e-3).unsqueeze(-1).float()
                G_active = G * active_constraints
                h_active = h.unsqueeze(-1) * active_constraints
                newQ = Q + lamb * G_active.transpose(-1,-2) @ G_active # + torch.eye(nz).repeat(nBatch, 1, 1).to(Q.device)
                newp = p.unsqueeze(-1) + delta_directions / lamb - lamb * G_active.transpose(-1,-2) @ h_active + G.transpose(-1,-2) @ lams.unsqueeze(-1) # - zhats
                # print('newQ, newp shape:', newQ.shape, newp.shape)
                # print('A, b shape:', A.shape, b.shape)
                if neq > 0:
                    newQ = torch.cat((newQ, lamb * A), dim=1)
                    newp = torch.cat((newp, - lamb * b.unsqueeze(-1)), dim=1)
                # print('condition number:', torch.linalg.cond(newQ))
                # newzhat = torch.linalg.solve(newQ, -newp)
                newzhat = torch.linalg.lstsq(newQ, -newp, driver='gels').solution
                # newzhat = - newQ.pinverse() @ newp
            # print('prediction', p)
            # print('solution distance:', torch.linalg.norm(newzhat - zhats))
            # print(lams)
            # print(zhats)
            # print(newzhat)
            # print('solution shape:', newzhat.shape, zhats.shape)
            # print('solution max distance:', torch.max(torch.abs(newzhat - zhats)))
            # assert False
            # print('newQ, p, newp, z shape:', newQ.shape, p.shape, newp.shape, newzhats.shape)

            # Computing the gradients of the Lagrangians
            start_time = time.time()
            with torch.enable_grad():
                Q_torch = Q.detach().clone().requires_grad_(True)
                p_torch = p.detach().clone().requires_grad_(True)
                G_torch = G.detach().clone().requires_grad_(True)
                h_torch = h.detach().clone().requires_grad_(True)
                A_torch = A.detach().clone().requires_grad_(True)
                b_torch = b.detach().clone().requires_grad_(True)
               
                upper_level_objectives = (delta_directions.transpose(-1,-2) @ newzhat).squeeze(-1,-2)
                objectives = (0.5 * newzhat.transpose(-1,-2) @ Q_torch @ newzhat + p_torch.unsqueeze(1) @ newzhat).squeeze(-1,-2) # 1/2 * z^T Q z + p^T z
                optimal_objectives = (0.5 * zhats.transpose(-1,-2) @ Q_torch @ zhats + p_torch.unsqueeze(1) @ zhats).squeeze(-1,-2) # 1/2 * z*^T Q z* + p^T z*
                violations = G_torch @ newzhat - h.unsqueeze(-1) # G z - h
                active_constraints = torch.tanh(lams * temperature).unsqueeze(-1).float()
                ineq_penalties = lams.unsqueeze(1) @ violations + 0.5 * lamb * torch.sum((violations * active_constraints) ** 2, dim=(-1,-2))
                if neq > 0:
                    eq_violations = A_torch @ newzhat - b_torch.unsqueeze(-1)
                    eq_penalties = nus.unsqueeze(1) @ eq_violations # + 0.5 * lamb * torch.sum(eq_violations ** 2, dim=(-1,-2))
                    # print(eq_violations)
                else:
                    eq_penalties = 0
                # print('obj, vio, active_constraints, lamb, ineq_penality shape:', objectives.shape, violations.shape, active_constraints.shape, lamb, ineq_penalties.shape)
                lagrangians = upper_level_objectives / lamb + objectives - optimal_objectives + ineq_penalties # + eq_penalties
                loss = torch.sum(lagrangians) * lamb
                loss.backward()

                Q_grad = Q_torch.grad
                p_grad = p_torch.grad
                G_grad = G_torch.grad
                h_grad = h_torch.grad
                A_grad = A_torch.grad
                b_grad = b_torch.grad

            return (Q_grad, p_grad, G_grad, h_grad, A_grad, b_grad)  # (None,) * len(ctx.saved_tensors)

    return QPFunctionFn.apply

def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()
