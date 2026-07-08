import cvxpy as cp
import time
import numpy as np

def get_solver_stats(problem):
    """Get solver stats for both OSQP and SCS robustly."""
    stats = problem.solver_stats.extra_stats
    return stats

def get_solver_stats(problem, solver_name, extra_stats=False):
    if solver_name=="SCS":
        if extra_stats:
            stats = problem.solver_stats.extra_stats
            solve_time = stats["info"]["solve_time"]
            setup_time = stats["info"]["setup_time"]
            num_iters = problem.solver_stats.num_iters
            return setup_time, solve_time, num_iters
            
        stats = problem.solver_stats
        solve_time = stats.solve_time
        setup_time = stats.setup_time
        num_iters = stats.num_iters
        return setup_time, solve_time, num_iters
    elif solver_name=="OSQP":
        if extra_stats:
            stats = problem.solver_stats.extra_stats
            solve_time = stats.info.solve_time
            setup_time = stats.info.setup_time
            num_iters = problem.solver_stats.num_iters
            return setup_time, solve_time, num_iters
        stats = problem.solver_stats
        solve_time = stats.solve_time
        setup_time = stats.extra_stats.info.setup_time
        num_iters = stats.num_iters
        return setup_time, solve_time, num_iters

# -------------------------
# Problem dimensions
extra_stats = False
verbose = False
solver_name = "SCS"
if solver_name=="SCS":
    solver = cp.SCS
elif solver_name=="OSQP":
    solver = cp.OSQP

n = 1000
m = 2*n + 1

print(f"dim: {n}, num inequalities: {m}")


Q = np.eye(n)
q = np.random.randn(n)
G = np.vstack([np.eye(n), -np.eye(n), np.ones((1, n))])
h = np.hstack([np.zeros(n), np.ones(n), np.array([3])])

# Forward pass
x = cp.Variable(n)
objective = 0.5 * cp.quad_form(x, Q) + q.T @ x
constraints = [G @ x <= h]






# Canonicalization time
t0 = time.perf_counter()
problem_forward = cp.Problem(cp.Minimize(objective), constraints)
t1 = time.perf_counter()
canonicalization_time_forward = t1 - t0
print("Forward pass canonicalization time:", canonicalization_time_forward, "s")


t0 = time.perf_counter()
problem_forward.solve(solver=solver, warm_start=False, verbose=verbose)
t1 = time.perf_counter()
total_time_scs = t1 - t0
solver_stats = get_solver_stats(problem_forward, solver_name=solver_name, extra_stats=extra_stats)

print(f"\nForward pass {solver_name}:")
print(f"Total wall-clock: {total_time_scs:.6f} s")
print("Solver setup_time:", solver_stats[0])
print("Solver solve_time:", solver_stats[1])
print("Iterations:", solver_stats[2])


# -------------------------
# Backward pass: convert k active inequalities to equalities
ineq_dual = constraints[0].dual_value
g_val = constraints[0].expr.value
s_val = -g_val
s_val = np.maximum(s_val, 0.0)
ineq_slack_residual = s_val

active_mask = ineq_slack_residual < 1e-6
num_active = np.sum(active_mask)
print(f"\nNumber of active inequalities: {num_active} out of {m}")

np.save("../active_mask.npy", active_mask)
# active_mask = np.load("../active_mask.npy")

A_eq = G[active_mask, :]
b_eq = h[active_mask]

# k = (m-1)//2  # number of active inequalities
# A_eq = G[:k, :]
# b_eq = h[:k]

x_bwd = cp.Variable(n)
objective_bwd = 0.5 * cp.quad_form(x_bwd, Q) + q.T @ x_bwd
constraints_bwd = [A_eq @ x_bwd == b_eq]

# Canonicalization
t0 = time.perf_counter()
problem_backward = cp.Problem(cp.Minimize(objective_bwd), constraints_bwd)
t1 = time.perf_counter()
canonicalization_time_backward = t1 - t0

print("\nBackward pass canonicalization time:", canonicalization_time_backward, "s")

# Solve backward with OSQP
t0 = time.perf_counter()
problem_backward.solve(solver=solver, warm_start=False, verbose=verbose)
t1 = time.perf_counter()
osqp_stats_bwd = get_solver_stats(problem_backward, solver_name=solver_name, extra_stats=extra_stats)
total_time_osqp_bwd = t1 - t0

print(f"\nBackward pass {solver_name}:")
print(f"Total wall-clock: {total_time_osqp_bwd:.6f} s")
print("Solver setup_time:", osqp_stats_bwd[0])
print("Solver solve_time:", osqp_stats_bwd[1])
print("Iterations:", osqp_stats_bwd[2])


