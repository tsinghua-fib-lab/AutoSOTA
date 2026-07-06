import os, sys
import torch
import numpy as np
from src.penalized_ot import PenalizedOT
from src.datagen import get_gaussian_mixture
from src.solvers import penalized_ot_solver
from src.loss_funcs import quota_loss
import ot
import time

print("Debug test...", flush=True)
(X, Y), (S_X, S_Y) = get_gaussian_mixture(
    d=2, n_x=250, n_y=25, scale=0.2, p_x0=0.5, p_y0=0.5,
    centers_X=[np.array([0, 0]), np.array([2.0, 0.0])],
    centers_Y=[np.array([1.0, 1.0]), np.array([2.5, 0.5])],
    rng=42,
)
print(f"X device: {X.device}, dtype: {X.dtype}", flush=True)
print(f"S_X device: {S_X.device}, dtype: {S_X.dtype}", flush=True)

eps = 1.0
F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])

# Simulate what solve() does with CUDA
if torch.cuda.is_available():
    print("Moving to CUDA...", flush=True)
    X_cuda = X.cuda()
    Y_cuda = Y.cuda()
    S_X_cuda = S_X.cuda()
    S_Y_cuda = S_Y.cuda()
    F_cuda = F_target.cuda()
    print(f"X_cuda device: {X_cuda.device}", flush=True)

    print("Computing cost matrix...", flush=True)
    C = ot.dist(X_cuda.cpu().numpy(), Y_cuda.cpu().numpy(), metric="sqeuclidean")
    C = torch.from_numpy(C).float().cuda()
    a = torch.ones(250).cuda() / 250
    b = torch.ones(25).cuda() / 25
    print(f"C device: {C.device}, dtype: {C.dtype}", flush=True)

    # Now call _single_solve equivalent
    print("Moving to CPU for solver...", flush=True)
    C_cpu = C.cpu()
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    S_X_cpu = S_X_cuda.cpu()
    S_Y_cpu = S_Y_cuda.cpu()
    F_cpu = F_cuda.cpu()
    print(f"C_cpu device: {C_cpu.device}, dtype: {C_cpu.dtype}", flush=True)
    print(f"S_X_cpu device: {S_X_cpu.device}, dtype: {S_X_cpu.dtype}", flush=True)

    print("Calling penalized_ot_solver...", flush=True)
    start = time.time()
    ot_plan, log = penalized_ot_solver(
        C_cpu, a_cpu, b_cpu,
        lambda plan: quota_loss(plan, S_X_cpu, S_Y_cpu, F_cpu),
        eps=eps, reg_constraints=100.0, log=True
    )
    elapsed = time.time() - start
    print(f"Solved in {elapsed:.1f}s", flush=True)
    
    if isinstance(ot_plan, torch.Tensor):
        ot_plan = ot_plan.cuda()
    
    fair_loss = quota_loss(ot_plan, S_X_cuda, S_Y_cuda, F_cuda)
    print(f"Fairness loss: {fair_loss.item():.6f}", flush=True)
    print("Success!", flush=True)
