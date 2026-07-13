import numpy as np
from scipy.optimize import linprog

def compute_constraint_range_fixed(T, rewards, mu0, n_states, n_actions, K, L, gamma):
    """Compute min and max achievable J_c."""
    n_rho = n_states * n_actions
    
    # Bellman flow
    A_eq = np.zeros((n_states, n_rho))
    for s_prime in range(n_states):
        for a in range(n_actions):
            A_eq[s_prime, s_prime * n_actions + a] = 1.0
        for s in range(n_states):
            for a in range(n_actions):
                A_eq[s_prime, s * n_actions + a] -= gamma * T[s, a, s_prime]
    
    b_eq = mu0.copy()
    bounds = [(0, None) for _ in range(n_rho)]
    
    mins, maxs = [], []
    for l in range(L):
        # c(s,a) is the constraint reward (negative)
        c_vals = rewards[:, :, K + l]  # (S, A)
        
        # Minimize J_c = sum c*rho. c is negative, so linprog minimizes this.
        c_obj_min = c_vals.flatten()  # negative values
        res_min = linprog(c_obj_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        min_jc = res_min.fun if res_min.success else None
        
        # Maximize J_c: minimize -J_c = sum (-c)*rho. -c is positive.
        c_obj_max = -c_vals.flatten()  # positive values  
        res_max = linprog(c_obj_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        max_jc = -res_max.fun if res_max.success else None
        
        mins.append(min_jc)
        maxs.append(max_jc)
    
    return np.array(mins), np.array(maxs)

# Quick test
rng = np.random.RandomState(0)
n_states, n_actions, K, L = 10, 2, 2, 1
gamma = 0.8

n_A = n_states // 2
n_B = n_states - n_A

T = np.zeros((n_states, n_actions, n_states))
for s in range(n_states):
    in_A = (s < n_A)
    target_list = list(range(n_A, n_states)) if in_A else list(range(n_A))
    for a in range(n_actions):
        probs = rng.dirichlet(np.ones(len(target_list)) * 0.5)
        for idx, s_next in enumerate(target_list):
            T[s, a, s_next] = probs[idx]

rewards = np.zeros((n_states, n_actions, K+L))
for s in range(n_states):
    for a in range(n_actions):
        rewards[s, a, :K] = rng.uniform(0.0, 1.0, size=K)
        rewards[s, a, K:] = rng.uniform(-2.0, 0.0, size=L)

mu0 = np.zeros(n_states)
mu0[:n_A] = 1.0 / n_A

c_min, c_max = compute_constraint_range_fixed(T, rewards, mu0, n_states, n_actions, K, L, gamma)
print(f"J_c range: [{c_min[0]:.4f}, {c_max[0]:.4f}]")

# Also test the main LP
n_rho = n_states * n_actions
n_vars = n_rho + 1

A_eq = np.zeros((n_states, n_rho))
for s_prime in range(n_states):
    for a in range(n_actions):
        A_eq[s_prime, s_prime * n_actions + a] = 1.0
    for s in range(n_states):
        for a in range(n_actions):
            A_eq[s_prime, s * n_actions + a] -= gamma * T[s, a, s_prime]
A_eq_ext = np.hstack([A_eq, np.zeros((n_states, 1))])

C = c_max[0] * 0.7 + c_min[0] * 0.3  # 70% toward max

c_obj = np.zeros(n_vars)
c_obj[-1] = -1.0

A_ub = np.zeros((K+L, n_vars))
b_ub = np.zeros(K+L)
for k in range(K):
    for s in range(n_states):
        for a in range(n_actions):
            A_ub[k, s * n_actions + a] = -rewards[s, a, k]
    A_ub[k, -1] = 1.0
for l in range(L):
    for s in range(n_states):
        for a in range(n_actions):
            A_ub[K+l, s * n_actions + a] = -rewards[s, a, K+l]
    b_ub[K+l] = -C

bounds = [(0, None) for _ in range(n_rho)] + [(None, None)]

res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_ext, b_eq=b_eq, bounds=bounds, method='highs')
print(f"Optimal max-min value: {-res.fun:.6f}, success={res.success}")
print(f"C = {C:.4f}")
