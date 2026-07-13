import numpy as np
from scipy.optimize import linprog
import time, warnings
warnings.filterwarnings('ignore')

# Same helpers as main script (copied inline for speed)
def generate_bipartite_momdp(n_states=30, n_actions=3, K=2, L=1, seed=0):
    rng = np.random.RandomState(seed)
    n_A = n_states // 2
    T = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        target = list(range(n_A, n_states)) if s < n_A else list(range(n_A))
        for a in range(n_actions):
            p = rng.dirichlet(np.ones(len(target)) * 0.5)
            for i, sn in enumerate(target):
                T[s, a, sn] = p[i]
    rewards = np.zeros((n_states, n_actions, K + L))
    for s in range(n_states):
        for a in range(n_actions):
            rewards[s, a, :K] = rng.uniform(0.0, 1.0, size=K)
            rewards[s, a, K:] = rng.uniform(-2.0, 0.0, size=L)
    mu0 = np.zeros(n_states); mu0[:n_A] = 1.0 / n_A
    return T, rewards, mu0

def build_bellman(T, mu0, nS, nA, gamma):
    A = np.zeros((nS, nS * nA))
    for sp in range(nS):
        for a in range(nA): A[sp, sp * nA + a] = 1.0
        for s in range(nS):
            for a in range(nA): A[sp, s * nA + a] -= gamma * T[s, a, sp]
    return A, mu0.copy()

def constraint_range(T, rewards, mu0, nS, nA, K, L, gamma):
    A_eq, b_eq = build_bellman(T, mu0, nS, nA, gamma)
    bounds = [(0, None)] * (nS * nA)
    mins, maxs = [], []
    for l in range(L):
        cf = rewards[:, :, K + l].flatten()
        rmin = linprog(cf, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        rmax = linprog(-cf, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        mins.append(rmin.fun if rmin.success else -10.0)
        maxs.append(-rmax.fun if rmax.success else 0.0)
    return np.array(mins), np.array(maxs)

def solve_optimal_value(T, rewards, mu0, nS, nA, K, L, gamma, C):
    n_rho = nS * nA; n_vars = n_rho + 1
    c_obj = np.zeros(n_vars); c_obj[-1] = -1.0
    A_eq, b_eq = build_bellman(T, mu0, nS, nA, gamma)
    A_eq = np.hstack([A_eq, np.zeros((nS, 1))])
    A_ub, b_ub = [], []
    for k in range(K):
        row = np.zeros(n_vars)
        for s in range(nS):
            for a in range(nA): row[s * nA + a] = -rewards[s, a, k]
        row[-1] = 1.0; A_ub.append(row); b_ub.append(0.0)
    for l in range(L):
        row = np.zeros(n_vars)
        for s in range(nS):
            for a in range(nA): row[s * nA + a] = -rewards[s, a, K + l]
        A_ub.append(row); b_ub.append(-C[l])
    bounds = [(0, None)] * n_rho + [(None, None)]
    res = linprog(np.array(c_obj), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return (-res.fun, res.x[:n_rho].reshape(nS, nA)) if res.success else (None, None)

def simplex_proj(v):
    n = len(v); u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    valid = np.where(u * np.arange(1, n+1) > (cssv - 1))[0]
    rho_idx = valid[-1] if len(valid) else -1
    theta = (cssv[rho_idx] - 1) / (rho_idx + 1) if rho_idx >= 0 else 0.0
    return np.maximum(v - theta, 0)

def softmax_policy(Q, beta):
    qs = Q / beta; qm = qs.max(axis=1, keepdims=True)
    e = np.exp(qs - qm); return e / e.sum(axis=1, keepdims=True)

def evaluate_policy(pi, T, rewards, mu0, nS, nA, gamma, K, L):
    Tpi = np.zeros((nS, nS))
    for s in range(nS):
        for sn in range(nS): Tpi[s, sn] = np.sum(pi[s, :] * T[s, :, sn])
    inv = np.linalg.inv(np.eye(nS) - gamma * Tpi)
    vals = []
    for o in range(K + L):
        rpi = np.sum(pi * rewards[:, :, o], axis=1)
        vals.append(float(np.dot(mu0, inv @ rpi)))
    return np.array(vals[:K]), np.array(vals[K:])

def run_algorithm(T, rewards, mu0, nS, nA, K, L, gamma, C, learn_u, learn_w,
                  beta=0.03, l_w=0.001, ITER=3000, conv_th=1e-4, seed=0):
    rng = np.random.RandomState(seed)
    ru, rc = rewards[:, :, :K], rewards[:, :, K:]
    Q = rng.randn(nS, nA) * 0.01
    u, w = np.zeros(L), np.ones(K) / K
    
    for m in range(ITER):
        mc, inner = float('inf'), 0
        while mc > conv_th and inner < 10000:
            Qo = Q.copy()
            sr = np.zeros((nS, nA))
            for k in range(K): sr += w[k] * ru[:, :, k]
            if learn_u:
                for l in range(L): sr += u[l] * rc[:, :, l]
            qs = Q / beta; qm = qs.max(axis=1, keepdims=True)
            v = beta * (qm.squeeze() + np.log(np.sum(np.exp(qs - qm), axis=1)))
            for s in range(nS):
                for a in range(nA):
                    Q[s, a] = sr[s, a] + gamma * np.dot(T[s, a, :], v)
            mc = np.max(np.abs(Q - Qo)); inner += 1
        
        pi = softmax_policy(Q, beta)
        ov, cv = evaluate_policy(pi, T, rewards, mu0, nS, nA, gamma, K, L)
        if learn_u: u = np.maximum(u - l_w * (cv - C), 0.0)
        if learn_w: w = simplex_proj(w - l_w * ov)
    
    pi_f = softmax_policy(Q, beta)
    ov_f, cv_f = evaluate_policy(pi_f, T, rewards, mu0, nS, nA, gamma, K, L)
    return float(np.min(ov_f)), ov_f, cv_f

# Test with tighter constraint and more iterations
nS, nA, K, L, gamma = 30, 3, 2, 1, 0.8
beta, l_w, ITER, conv_th = 0.03, 0.001, 5000, 1e-4
seed = 0

print(f"Testing with tighter constraint threshold and ITER={ITER}")
T, rewards, mu0 = generate_bipartite_momdp(nS, nA, K, L, seed)
c_min, c_max = constraint_range(T, rewards, mu0, nS, nA, K, L, gamma)
print(f"J_c range: [{c_min[0]:.4f}, {c_max[0]:.4f}]")

# Try different constraint tightness levels
for tightness in [0.1, 0.2, 0.3, 0.4, 0.5]:
    C = np.array([c_max[0] - tightness * (c_max[0] - c_min[0])])
    
    opt_val, _ = solve_optimal_value(T, rewards, mu0, nS, nA, K, L, gamma, C)
    if opt_val is None:
        print(f"  tightness={tightness:.1f} C={C[0]:.4f}: INFEASIBLE")
        continue
    
    print(f"\n  tightness={tightness:.1f} C={C[0]:.4f} LP_opt={opt_val:.6f}")
    
    for learn_u, learn_w, name in [
        (True, True, "Constrained max-min"),
        (False, True, "Unconstrained max-min"),
        (True, False, "Constrained max-avg"),
        (False, False, "Unconstrained max-avg"),
    ]:
        t0 = time.time()
        mmv, ov, cv = run_algorithm(T, rewards, mu0, nS, nA, K, L, gamma, C,
                                     learn_u, learn_w, beta, l_w, ITER, conv_th, seed)
        err = abs(opt_val - mmv)
        sat = all(cv >= C - 1e-5)
        print(f"    {name:25s}: err={err:.6f} mmv={mmv:.6f} constr={'OK' if sat else 'VIOL'} t={time.time()-t0:.1f}s")

