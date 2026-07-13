#!/usr/bin/env python3
"""
Tabular Constrained Max-Min MORL Experiment
Implements Algorithm 1 from: "Constrained Multi-Objective RL with Max-Min Criterion"
Paper 5090, ICML 2026 — Reproduces Section 5.1 tabular experiments.
"""

import numpy as np
from scipy.optimize import linprog
import time, argparse, json, sys, warnings
warnings.filterwarnings('ignore')


# ============================================================
# MOMDP Generation
# ============================================================

def generate_bipartite_momdp(n_states=30, n_actions=3, K=2, L=1, seed=0):
    rng = np.random.RandomState(seed)
    n_A = n_states // 2
    n_B = n_states - n_A
    
    T = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        target = list(range(n_A, n_states)) if s < n_A else list(range(n_A))
        for a in range(n_actions):
            p = rng.dirichlet(np.ones(len(target)) * 0.5)
            for i, sn in enumerate(target):
                T[s, a, sn] = p[i]
    assert np.allclose(T.sum(axis=2), 1.0)
    
    rewards = np.zeros((n_states, n_actions, K + L))
    for s in range(n_states):
        for a in range(n_actions):
            rewards[s, a, :K] = rng.uniform(0.0, 1.0, size=K)
            rewards[s, a, K:] = rng.uniform(-2.0, 0.0, size=L)
    
    mu0 = np.zeros(n_states)
    mu0[:n_A] = 1.0 / n_A
    return T, rewards, mu0


def generate_hierarchical_momdp(n_states=30, n_actions=3, K=2, L=1, n_levels=5, seed=0):
    rng = np.random.RandomState(seed)
    sp = n_states // n_levels
    rem = n_states % n_levels
    level_sizes = [sp + (1 if i < rem else 0) for i in range(n_levels)]
    starts = [0]
    for sz in level_sizes[:-1]:
        starts.append(starts[-1] + sz)
    
    def level_of(s):
        for i in range(n_levels - 1):
            if s < starts[i+1]:
                return i
        return n_levels - 1
    
    T = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        lv = level_of(s)
        nl = (lv + 1) % n_levels
        target = list(range(starts[nl], starts[nl] + level_sizes[nl]))
        for a in range(n_actions):
            p = rng.dirichlet(np.ones(len(target)) * 0.5)
            for i, sn in enumerate(target):
                T[s, a, sn] = p[i]
    assert np.allclose(T.sum(axis=2), 1.0)
    
    rewards = np.zeros((n_states, n_actions, K + L))
    for s in range(n_states):
        for a in range(n_actions):
            rewards[s, a, :K] = rng.uniform(0.0, 1.0, size=K)
            rewards[s, a, K:] = rng.uniform(-2.0, 0.0, size=L)
    
    mu0 = np.zeros(n_states)
    mu0[:level_sizes[0]] = 1.0 / level_sizes[0]
    return T, rewards, mu0


# ============================================================
# LP helpers
# ============================================================

def build_bellman(T, mu0, nS, nA, gamma):
    A = np.zeros((nS, nS * nA))
    for sp in range(nS):
        for a in range(nA):
            A[sp, sp * nA + a] = 1.0
        for s in range(nS):
            for a in range(nA):
                A[sp, s * nA + a] -= gamma * T[s, a, sp]
    return A, mu0.copy()


def constraint_range(T, rewards, mu0, nS, nA, K, L, gamma):
    """Compute [min_Jc, max_Jc] achievable for each constraint."""
    A_eq, b_eq = build_bellman(T, mu0, nS, nA, gamma)
    bounds = [(0, None)] * (nS * nA)
    mins, maxs = [], []
    for l in range(L):
        c_flat = rewards[:, :, K + l].flatten()  # negative
        rmin = linprog(c_flat, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        rmax = linprog(-c_flat, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        mins.append(rmin.fun if rmin.success else -10.0)
        maxs.append(-rmax.fun if rmax.success else 0.0)
    return np.array(mins), np.array(maxs)


def solve_optimal_value(T, rewards, mu0, nS, nA, K, L, gamma, C):
    """LP: max c_tilde subject to Bellman + reward/constraint bounds."""
    n_rho = nS * nA
    n_vars = n_rho + 1
    
    c_obj = np.zeros(n_vars); c_obj[-1] = -1.0
    A_eq, b_eq = build_bellman(T, mu0, nS, nA, gamma)
    A_eq = np.hstack([A_eq, np.zeros((nS, 1))])
    
    A_ub, b_ub = [], []
    for k in range(K):
        row = np.zeros(n_vars)
        for s in range(nS):
            for a in range(nA):
                row[s * nA + a] = -rewards[s, a, k]
        row[-1] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    for l in range(L):
        row = np.zeros(n_vars)
        for s in range(nS):
            for a in range(nA):
                row[s * nA + a] = -rewards[s, a, K + l]
        A_ub.append(row); b_ub.append(-C[l])
    
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)
    bounds = [(0, None)] * n_rho + [(None, None)]
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if res.success:
        return -res.fun, res.x[:n_rho].reshape(nS, nA)
    return None, None


# ============================================================
# Algorithm helpers
# ============================================================

def simplex_proj(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    valid = np.where(u * np.arange(1, n+1) > (cssv - 1))[0]
    rho_idx = valid[-1] if len(valid) else -1
    theta = (cssv[rho_idx] - 1) / (rho_idx + 1) if rho_idx >= 0 else 0.0
    return np.maximum(v - theta, 0)


def softmax_policy(Q, beta):
    qs = Q / beta
    qm = qs.max(axis=1, keepdims=True)
    e = np.exp(qs - qm)
    return e / e.sum(axis=1, keepdims=True)


def evaluate_policy(pi, T, rewards, mu0, nS, nA, gamma, K, L):
    Tpi = np.zeros((nS, nS))
    for s in range(nS):
        for sn in range(nS):
            Tpi[s, sn] = np.sum(pi[s, :] * T[s, :, sn])
    inv = np.linalg.inv(np.eye(nS) - gamma * Tpi)
    vals = []
    for o in range(K + L):
        rpi = np.sum(pi * rewards[:, :, o], axis=1)
        vals.append(float(np.dot(mu0, inv @ rpi)))
    return np.array(vals[:K]), np.array(vals[K:])


def soft_vi_step(Q, T, r_unconstrained, r_constraint, w, u, gamma, beta, nS, nA, K, L):
    sr = np.zeros((nS, nA))
    for k in range(K):
        sr += w[k] * r_unconstrained[:, :, k]
    for l in range(L):
        sr += u[l] * r_constraint[:, :, l]
    
    qs = Q / beta
    qm = qs.max(axis=1, keepdims=True)
    v = beta * (qm.squeeze() + np.log(np.sum(np.exp(qs - qm), axis=1)))
    
    Qn = np.zeros_like(Q)
    for s in range(nS):
        for a in range(nA):
            Qn[s, a] = sr[s, a] + gamma * np.dot(T[s, a, :], v)
    return Qn


# ============================================================
# Algorithm 1
# ============================================================

def run_algorithm(T, rewards, mu0, nS, nA, K, L, gamma, C,
                  learn_u=True, learn_w=True, beta=0.03, l_w=0.001,
                  ITER=3000, conv_th=1e-4, seed=0,
                  l_w_max=0.01, l_w_min=0.0001,
                  w_init=None, u_init=None,
                  conv_th_final=None):
    rng = np.random.RandomState(seed)
    ru = rewards[:, :, :K]
    rc = rewards[:, :, K:]

    Q = rng.randn(nS, nA) * 0.01
    u = np.zeros(L) if u_init is None else u_init.copy()
    w = (np.ones(K) / K) if w_init is None else w_init.copy()

    for m in range(ITER):
        # Cosine annealing schedule for dual learning rate
        l_w_t = l_w_min + 0.5 * (l_w_max - l_w_min) * (1.0 + np.cos(np.pi * m / ITER))

        # Adaptive convergence threshold: tighten in second half
        if conv_th_final is not None:
            conv_th_m = conv_th if m < ITER // 2 else conv_th_final
        else:
            conv_th_m = conv_th

        mc = float('inf')
        inner = 0
        while mc > conv_th_m and inner < 10000:
            Qo = Q.copy()
            Q = soft_vi_step(Q, T, ru, rc, w, u, gamma, beta, nS, nA, K, L)
            mc = np.max(np.abs(Q - Qo))
            inner += 1

        pi = softmax_policy(Q, beta)
        ov, cv = evaluate_policy(pi, T, rewards, mu0, nS, nA, gamma, K, L)

        if learn_u:
            u = np.maximum(u - l_w_t * (cv - C), 0.0)
        if learn_w:
            w = simplex_proj(w - l_w_t * ov)
    
    pi_f = softmax_policy(Q, beta)
    ov_f, cv_f = evaluate_policy(pi_f, T, rewards, mu0, nS, nA, gamma, K, L)
    return float(np.min(ov_f)), ov_f, cv_f, Q


# ============================================================
# Main experiment
# ============================================================

def main():
    p = argparse.ArgumentParser(description='Tabular Constrained Max-Min MORL (Paper 5090)')
    p.add_argument('--momdp_type', default='bipartite', choices=['bipartite','hierarchical'])
    p.add_argument('--n_states', type=int, default=30)
    p.add_argument('--n_actions', type=int, default=3)
    p.add_argument('--K', type=int, default=2)
    p.add_argument('--L', type=int, default=1)
    p.add_argument('--gamma', type=float, default=0.8)
    p.add_argument('--beta', type=float, default=0.03)
    p.add_argument('--l_w', type=float, default=0.001)
    p.add_argument('--l_w_max', type=float, default=0.01)
    p.add_argument('--l_w_min', type=float, default=0.0001)
    p.add_argument('--ITER', type=int, default=3000)
    p.add_argument('--conv_th', type=float, default=1e-4)
    p.add_argument('--tightness', type=float, default=0.3, help='Constraint tightness (0=loose, 1=tight)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--output', default=None)
    args = p.parse_args()
    
    cfg = ['constrained_maxmin', 'unconstrained_maxmin',
           'constrained_maxaverage', 'unconstrained_maxaverage']
    names = {'constrained_maxmin': 'Constrained max-min',
             'unconstrained_maxmin': 'Unconstrained max-min',
             'constrained_maxaverage': 'Constrained max-average',
             'unconstrained_maxaverage': 'Unconstrained max-average'}
    lu_map = {'constrained_maxmin': (True, True),
              'unconstrained_maxmin': (False, True),
              'constrained_maxaverage': (True, False),
              'unconstrained_maxaverage': (False, False)}
    
    print(f"Tabular Constrained Max-Min MORL — Paper 5090 Reproduction")
    print(f"===========================================================")
    print(f"MOMDP: {args.momdp_type}  |S|={args.n_states} |A|={args.n_actions}  K={args.K} L={args.L}")
    print(f"gamma={args.gamma}  beta={args.beta}  l_w={args.l_w}  ITER={args.ITER}")
    print(f"conv_th={args.conv_th}  seeds={args.seeds}")
    
    results = {k: [] for k in cfg}
    results['optimal_values'] = []
    
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        
        gen_fn = generate_bipartite_momdp if args.momdp_type == 'bipartite' else generate_hierarchical_momdp
        T, rewards, mu0 = gen_fn(args.n_states, args.n_actions, args.K, args.L, seed)
        
        c_min, c_max = constraint_range(T, rewards, mu0, args.n_states, args.n_actions, args.K, args.L, args.gamma)
        # Set C between min and max: 30% from max toward min (achievable but binding)
        C = np.array([c_max[0] - args.tightness * (c_max[0] - c_min[0])])
        print(f"  J_c range: [{c_min[0]:.4f}, {c_max[0]:.4f}]  C={C[0]:.4f}")
        
        opt_val, _ = solve_optimal_value(T, rewards, mu0, args.n_states, args.n_actions,
                                          args.K, args.L, args.gamma, C)
        if opt_val is None:
            C = np.array([c_max[0] - 0.5 * (c_max[0] - c_min[0])])
            opt_val, _ = solve_optimal_value(T, rewards, mu0, args.n_states, args.n_actions,
                                              args.K, args.L, args.gamma, C)
        
        if opt_val is None:
            print(f"  SKIP: LP infeasible")
            continue
        
        results['optimal_values'].append(float(opt_val))
        print(f"  LP optimal max-min value: {opt_val:.6f}")
        
        # Warm-start: run unconstrained max-min briefly to init w
        from numpy.random import RandomState
        w_warm = np.ones(args.K) / args.K
        try:
            Q_ws = RandomState(seed).randn(args.n_states, args.n_actions) * 0.01
            w_ws = np.ones(args.K) / args.K
            u_ws = np.zeros(args.L)
            ru_ws = rewards[:, :, :args.K]
            rc_ws = rewards[:, :, args.K:]
            for mi in range(500):
                mc_ws = float('inf')
                inner_ws = 0
                while mc_ws > args.conv_th and inner_ws < 2000:
                    Qo_ws = Q_ws.copy()
                    Q_ws = soft_vi_step(Q_ws, T, ru_ws, rc_ws, w_ws, u_ws, args.gamma, args.beta, args.n_states, args.n_actions, args.K, args.L)
                    mc_ws = np.max(np.abs(Q_ws - Qo_ws))
                    inner_ws += 1
                pi_ws = softmax_policy(Q_ws, args.beta)
                ov_ws, _ = evaluate_policy(pi_ws, T, rewards, mu0, args.n_states, args.n_actions, args.gamma, args.K, args.L)
                w_ws = simplex_proj(w_ws - args.l_w * ov_ws)
            w_warm = w_ws.copy()
        except Exception:
            pass
        
        for bl in cfg:
            lu, lw_flag = lu_map[bl]
            t1 = time.time()
            mmv, ov, cv, _ = run_algorithm(
                T, rewards, mu0, args.n_states, args.n_actions, args.K, args.L,
                args.gamma, C, learn_u=lu, learn_w=lw_flag,
                beta=args.beta, l_w=args.l_w, ITER=args.ITER,
                conv_th=args.conv_th, seed=seed,
                w_init=w_warm if lw_flag else None,
                conv_th_final=1e-6 if lw_flag else None,
                l_w_max=args.l_w_max, l_w_min=args.l_w_min)
            t_alg = time.time() - t1
            
            err = abs(opt_val - mmv)
            sat = all(cv >= C - 1e-5)
            
            results[bl].append({
                'maxmin_value': mmv, 'optimal_value': float(opt_val),
                'error': err, 'constraint_satisfied': bool(sat),
                'obj_values': [float(v) for v in ov],
                'constraint_values': [float(v) for v in cv],
                'C': [float(v) for v in C], 'time': t_alg, 'seed': seed,
            })
            print(f"  {names[bl]:30s}: err={err:.6f}  mmv={mmv:.6f}  "
                  f"constr={'OK' if sat else 'VIOL'}  t={t_alg:.1f}s")
    
    t_total = time.time() - t0
    
    print(f"\n{'='*60}\nSUMMARY ({args.momdp_type}, n={len(results['optimal_values'])}/{len(args.seeds)} feasible)")
    print(f"{'='*60}")
    print(f"Total: {t_total:.1f}s")
    
    for bl in cfg:
        if results[bl]:
            errs = [r['error'] for r in results[bl]]
            sats = sum(r['constraint_satisfied'] for r in results[bl])
            print(f"  {names[bl]:30s}: error={np.mean(errs):.6f} +/- {np.std(errs):.6f}  "
                  f"constr_OK={sats}/{len(errs)}")
    
    if args.output:
        def conv(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, dict): return {k: conv(v) for k,v in obj.items()}
            if isinstance(obj, list): return [conv(v) for v in obj]
            return obj
        with open(args.output, 'w') as f:
            json.dump(conv(results), f, indent=2)
        print(f"\nSaved: {args.output}")

if __name__ == '__main__':
    main()
