#!/usr/bin/env python3
"""Fast evaluation harness for PenalizedOT experiments."""
import os, sys, torch, numpy as np, pandas as pd
from src.penalized_ot import PenalizedOT
from src.datagen import get_gaussian_mixture
from src import solvers
from ot import sinkhorn

# Parse args: fairness_loss, eps, extra_n_samples, seed
fairness_loss = sys.argv[1] if len(sys.argv) > 1 else "quota_loss"
eps_val = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
extra_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 0
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

# Fast solver override
def fast_penalized_ot_solver(C, a, b, f, reg_constraints=1.0, eps=1.0, log=False):
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(torch.float64)
        return torch.as_tensor(x, dtype=torch.float64)
    C_t = to_tensor(C); a_t = to_tensor(a); b_t = to_tensor(b)
    G0 = sinkhorn(a_t, b_t, C_t, eps, method='sinkhorn_log', numItermax=50000, warn=False)
    if not isinstance(G0, torch.Tensor):
        G0 = torch.as_tensor(G0, dtype=torch.float64)
    tp, lo = solvers.gcg(a_t, b_t, C_t, eps, reg_constraints, f, torch.func.grad(f),
        G0=G0, log=log, method='sinkhorn_log', numItermax=200, stopThr=1e-9, numInnerItermax=2000, verbose=False)
    if log: return tp, solvers._log_to_numpy(lo)
    else: return tp

solvers.penalized_ot_solver = fast_penalized_ot_solver

# Generate data
n_x = 250 + extra_samples
n_y = 25 + extra_samples // 10
(X, Y), (S_X, S_Y) = get_gaussian_mixture(
    d=2, n_x=n_x, n_y=n_y, scale=0.2,
    p_x0=0.5, p_y0=0.5,
    centers_X=[np.array([0,0]), np.array([2.0,0.0])],
    centers_Y=[np.array([1.0,1.0]), np.array([2.5,0.5])],
    rng=seed)

F_target = torch.tensor([[0.2, 0.3], [0.28, 0.22]])
penalty_grid = np.logspace(0, 3, 10)

penalized_ot = PenalizedOT(
    penalty_grid=penalty_grid,
    entropic_grid=[eps_val],
    fairness_loss=fairness_loss)

# Patch _solve_ot for speed
penalized_ot._solve_ot = lambda cm, a, b, eps: sinkhorn(
    a, b, cm, eps, method='sinkhorn_log', numItermax=50000, warn=False)

results = penalized_ot.solve(X=X, Y=Y, S_X=S_X, S_Y=S_Y, F=F_target, n_jobs=1, use_cache=False)
os.makedirs('results/exp_gaussian/', exist_ok=True)
results.to_pickle('results/exp_gaussian/results_penalized.pkl')

print('FAST_EVAL_COMPLETE')
print(f'Config: fairness_loss={fairness_loss}, eps={eps_val}, extra_samples={extra_samples}, seed={seed}')
print(results[['penalty', 'cost_diff', 'fairness_loss_value', 'n_iters']].to_string())
