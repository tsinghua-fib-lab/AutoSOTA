import numpy as np
from scipy.optimize import linprog
import time, json, warnings
warnings.filterwarnings('ignore')

# ---- Helpers (same as before) ----
def generate_bipartite_momdp(nS=30, nA=3, K=2, L=1, seed=0):
    rng = np.random.RandomState(seed); n_A = nS // 2
    T = np.zeros((nS, nA, nS))
    for s in range(nS):
        target = list(range(n_A, nS)) if s < n_A else list(range(n_A))
        for a in range(nA):
            p = rng.dirichlet(np.ones(len(target))*0.5)
            for i, sn in enumerate(target): T[s,a,sn] = p[i]
    rewards = np.zeros((nS, nA, K+L))
    for s in range(nS):
        for a in range(nA):
            rewards[s,a,:K] = rng.uniform(0.,1.,size=K)
            rewards[s,a,K:] = rng.uniform(-2.,0.,size=L)
    mu0 = np.zeros(nS); mu0[:n_A] = 1./n_A
    return T, rewards, mu0

def build_bellman(T, mu0, nS, nA, gamma):
    A = np.zeros((nS, nS*nA))
    for sp in range(nS):
        for a in range(nA): A[sp, sp*nA+a] = 1.
        for s in range(nS):
            for a in range(nA): A[sp, s*nA+a] -= gamma * T[s,a,sp]
    return A, mu0.copy()

def constraint_range(T, rewards, mu0, nS, nA, K, L, gamma):
    A_eq, b_eq = build_bellman(T, mu0, nS, nA, gamma)
    bounds = [(0,None)]*(nS*nA)
    mins, maxs = [], []
    for l in range(L):
        cf = rewards[:,:,K+l].flatten()
        rmin = linprog(cf, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        rmax = linprog(-cf, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        mins.append(rmin.fun if rmin.success else -10.)
        maxs.append(-rmax.fun if rmax.success else 0.)
    return np.array(mins), np.array(maxs)

def solve_opt(T, rewards, mu0, nS, nA, K, L, gamma, C):
    nr = nS*nA; nv = nr+1
    co = np.zeros(nv); co[-1] = -1.
    Ae, be = build_bellman(T, mu0, nS, nA, gamma)
    Ae = np.hstack([Ae, np.zeros((nS,1))])
    Au, bu = [], []
    for k in range(K):
        r = np.zeros(nv)
        for s in range(nS):
            for a in range(nA): r[s*nA+a] = -rewards[s,a,k]
        r[-1]=1.; Au.append(r); bu.append(0.)
    for l in range(L):
        r = np.zeros(nv)
        for s in range(nS):
            for a in range(nA): r[s*nA+a] = -rewards[s,a,K+l]
        Au.append(r); bu.append(-C[l])
    bnd = [(0,None)]*nr + [(None,None)]
    res = linprog(co, A_ub=np.array(Au), b_ub=np.array(bu),
                  A_eq=Ae, b_eq=be, bounds=bnd, method='highs')
    return (-res.fun, res.x[:nr].reshape(nS,nA)) if res.success else (None,None)

def simplex_proj(v):
    n=len(v); u=np.sort(v)[::-1]; cs=np.cumsum(u)
    vd=np.where(u*np.arange(1,n+1)>(cs-1))[0]
    ri=vd[-1] if len(vd) else -1
    th=(cs[ri]-1)/(ri+1) if ri>=0 else 0.
    return np.maximum(v-th,0)

def softmax_policy(Q, beta):
    qs=Q/beta; qm=qs.max(1,keepdims=True)
    e=np.exp(qs-qm); return e/e.sum(1,keepdims=True)

def evaluate_policy(pi, T, rewards, mu0, nS, nA, gamma, K, L):
    Tpi=np.zeros((nS,nS))
    for s in range(nS):
        for sn in range(nS): Tpi[s,sn]=np.sum(pi[s,:]*T[s,:,sn])
    inv=np.linalg.inv(np.eye(nS)-gamma*Tpi)
    vals=[]
    for o in range(K+L):
        rpi=np.sum(pi*rewards[:,:,o],axis=1)
        vals.append(float(np.dot(mu0,inv@rpi)))
    return np.array(vals[:K]), np.array(vals[K:])

def run_alg(T, rewards, mu0, nS, nA, K, L, gamma, C, lu, lw_flag,
            beta=0.03, l_w=0.001, ITER=3000, ct=1e-4, seed=0):
    rng=np.random.RandomState(seed)
    ru,rc=rewards[:,:,:K],rewards[:,:,K:]
    Q=rng.randn(nS,nA)*0.01; u=np.zeros(L); w=np.ones(K)/K
    for m in range(ITER):
        mc,inner=float('inf'),0
        while mc>ct and inner<10000:
            Qo=Q.copy()
            sr=np.zeros((nS,nA))
            for k in range(K): sr+=w[k]*ru[:,:,k]
            if lu:
                for l in range(L): sr+=u[l]*rc[:,:,l]
            qs=Q/beta; qm=qs.max(1,keepdims=True)
            v=beta*(qm.squeeze()+np.log(np.sum(np.exp(qs-qm),axis=1)))
            for s in range(nS):
                for a in range(nA): Q[s,a]=sr[s,a]+gamma*np.dot(T[s,a,:],v)
            mc=np.max(np.abs(Q-Qo)); inner+=1
        pi=softmax_policy(Q,beta)
        ov,cv=evaluate_policy(pi,T,rewards,mu0,nS,nA,gamma,K,L)
        if lu: u=np.maximum(u-l_w*(cv-C),0.)
        if lw_flag: w=simplex_proj(w-l_w*ov)
    pi_f=softmax_policy(Q,beta)
    ov_f,cv_f=evaluate_policy(pi_f,T,rewards,mu0,nS,nA,gamma,K,L)
    return float(np.min(ov_f)),ov_f,cv_f

# ---- Main ----
nS,nA,K,L,gamma=30,3,2,1,0.8
beta,l_w,ITER,ct=0.03,0.001,3000,1e-4
tightness=0.1  # Very tight constraint
seeds=[0,1,2,3,4]  # 5 seeds for better statistics

print(f"FULL EXPERIMENT: bipartite, |S|={nS}, |A|={nA}, K={K}, L={L}")
print(f"gamma={gamma}, beta={beta}, l_w={l_w}, ITER={ITER}")
print(f"tightness={tightness}, seeds={seeds}")
print()

all_results = {
    'constrained_maxmin': [], 'unconstrained_maxmin': [],
    'constrained_maxaverage': [], 'unconstrained_maxaverage': []
}
opt_vals = []

for seed in seeds:
    print(f"{'='*60}")
    print(f"Seed {seed}")
    T, rewards, mu0 = generate_bipartite_momdp(nS, nA, K, L, seed)
    c_min, c_max = constraint_range(T, rewards, mu0, nS, nA, K, L, gamma)
    C = np.array([c_max[0] - tightness * (c_max[0] - c_min[0])])
    
    opt_val, _ = solve_opt(T, rewards, mu0, nS, nA, K, L, gamma, C)
    if opt_val is None:
        print(f"  INFEASIBLE, skipping")
        continue
    opt_vals.append(float(opt_val))
    print(f"  J_c range: [{c_min[0]:.3f}, {c_max[0]:.3f}]  C={C[0]:.3f}  LP_opt={opt_val:.6f}")
    
    for lu, lw_flag, name in [
        (True, True, 'constrained_maxmin'),
        (False, True, 'unconstrained_maxmin'),
        (True, False, 'constrained_maxaverage'),
        (False, False, 'unconstrained_maxaverage'),
    ]:
        mmv, ov, cv = run_alg(T, rewards, mu0, nS, nA, K, L, gamma, C,
                               lu, lw_flag, beta, l_w, ITER, ct, seed)
        err = abs(opt_val - mmv)
        sat = all(cv >= C - 1e-5)
        all_results[name].append({'error': err, 'mmv': mmv, 'sat': sat,
                                   'ov': [float(x) for x in ov],
                                   'cv': [float(x) for x in cv]})
        print(f"  {name:30s}: err={err:.6f} mmv={mmv:.6f} {'OK' if sat else 'VIOL'}")

print(f"\n{'='*60}")
print(f"SUMMARY (n={len(opt_vals)} feasible)")
for name in all_results:
    if all_results[name]:
        errs = [r['error'] for r in all_results[name]]
        sats = sum(r['sat'] for r in all_results[name])
        print(f"  {name:30s}: mean_err={np.mean(errs):.6f} +/- {np.std(errs):.6f}  "
              f"constr_OK={sats}/{len(errs)}")

# Save
result_data = {'config': {'nS':nS,'nA':nA,'K':K,'L':L,'gamma':gamma,
                          'beta':beta,'l_w':l_w,'ITER':ITER,'tightness':tightness},
               'results': all_results, 'optimal_values': opt_vals}
with open('/repo/results_full.json', 'w') as f:
    json.dump(result_data, f, indent=2)
print(f"\nSaved to /repo/results_full.json")
