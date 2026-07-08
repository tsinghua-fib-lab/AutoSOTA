#!/usr/bin/env python3
"""Final reproduction script for DeSI Dist. (Quad.) n=200."""
import sys, os, time, json, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/repo/simulation_distribution')
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from DeSI import DeSI_distribution
from generate_dist import generate_simulation_data_torch_true
from scipy.stats import norm

# ── Paper-consistent settings ──
N, P, QF, LINK = 200, 4, 100, "quadratic"
BATCH, EPOCHS, HIDDEN, LR = 64, 10000, 64, 0.01
PATIENCE, DELTA, LAMBDA, BW_INIT = 10, 1e-4, 0.0005, 0.1
N_SEEDS = 200

torch.set_num_threads(1)

class ThetaMLP(nn.Module):
    def __init__(self, d_in, d_hid=32, dp=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.LeakyReLU(), nn.Dropout(dp),
            nn.Linear(d_hid, d_hid), nn.LayerNorm(d_hid), nn.LeakyReLU(), nn.Dropout(dp),
            nn.Linear(d_hid, d_in)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, X):
        x = self.net(X)
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x * torch.where(x[:, 0:1] < 0, -1.0, 1.0)

class GB(nn.Module):
    def __init__(self, bw=0.1):
        super().__init__()
        self.bw = nn.Parameter(torch.tensor([bw], dtype=torch.float32))
    @property
    def bandwidth(self):
        return torch.clamp(self.bw, min=0.01)

def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y, theta, mu, sigma = generate_simulation_data_torch_true(n=N, qf_size=QF, p=P, link=LINK, seed=seed)
    qf = torch.stack([Y[i] for i in range(N)])

    idx = np.arange(N); np.random.shuffle(idx)
    n_tr = int(0.4 * N); n_va = int(0.1 * N); n_te = N - n_tr - n_va
    i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    Xt, Xv, Xe = X[i_tr], X[i_va], X[i_te]
    Qt, Qv = qf[i_tr], qf[i_va]

    Xm = Xt.mean(0, keepdim=True); Xs = Xt.std(0, keepdim=True) + 1e-8
    Xt, Xv, Xe = (Xt-Xm)/Xs, (Xv-Xm)/Xs, (Xe-Xm)/Xs

    model = ThetaMLP(P, HIDDEN, 0.3); gbw = GB(BW_INIT)
    opt = optim.Adam(list(model.parameters()) + list(gbw.parameters()), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, Qt), BATCH, shuffle=True)

    bv, bc, bs, bbw = float('inf'), 0, None, None
    for ep in range(EPOCHS):
        model.train()
        for Xb, qb in loader:
            opt.zero_grad()
            tb = model(Xb); tb = tb/(torch.norm(tb, dim=1, keepdim=True)+1e-8)
            tb = tb * torch.where(tb[:,0:1]<0, -1., 1.)
            yb = [qb[j] for j in range(qb.shape[0])]
            qp = DeSI_distribution(y=yb, x=torch.einsum('ij,ij->i', Xb, tb), h=gbw.bandwidth).get('qf')
            l2 = torch.mean((qp-qb)**2); ybt = torch.stack(yb)
            fv = torch.mean(torch.norm(ybt-ybt.mean(0), dim=1)**2)
            (l2/(fv+1e-8) + LAMBDA/(gbw.bandwidth+1e-8)).backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            tt = model(Xt); tt = tt/(torch.norm(tt,dim=1,keepdim=True)+1e-8)
            tt = tt * torch.where(tt[:,0:1]<0, -1., 1.)
            Zt = torch.einsum('ij,ij->i', Xt, tt)
            yt = [Qt[j] for j in range(Qt.shape[0])]
            tv = model(Xv); tv = tv/(torch.norm(tv,dim=1,keepdim=True)+1e-8)
            tv = tv * torch.where(tv[:,0:1]<0, -1., 1.)
            rv = DeSI_distribution(y=yt, x=Zt, xOut=torch.einsum('ij,ij->i', Xv, tv), h=gbw.bandwidth)
            qpv = rv.get('qf'); l2v = torch.mean((qpv-Qv)**2)
            myv = Qv.mean(0); fvv = torch.mean(torch.norm(Qv-myv, dim=1)**2)
            vl = (l2v/(fvv+1e-8) + LAMBDA/(gbw.bandwidth+1e-8)).item()
            if vl < bv - DELTA: bv = vl; bs = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bbw = gbw.bandwidth.detach().cpu().clone(); bc = 0
            else: bc += 1
            if bc >= PATIENCE: break

    if bs: model.load_state_dict(bs); gbw.bw.data = bbw.to(gbw.bw.device)

    model.eval()
    with torch.no_grad():
        tt = model(Xt); tt = tt/(torch.norm(tt,dim=1,keepdim=True)+1e-8)
        Zt = torch.einsum('ij,ij->i', Xt, tt)
        yt = [Qt[j] for j in range(Qt.shape[0])]
        te = model(Xe); te = te/(torch.norm(te,dim=1,keepdim=True)+1e-8)
        te = te * torch.where(te[:,0:1]<0, -1., 1.)
        rt = DeSI_distribution(y=yt, x=Zt, xOut=torch.einsum('ij,ij->i', Xe, te), h=gbw.bandwidth)
        qpt = rt.get('qf')
        qS = np.linspace(0,1,QF+2)[1:-1]
        qt = np.zeros((n_te, QF))
        for i in range(n_te): qt[i,:] = norm.ppf(qS, loc=float(mu[i_te[i]]), scale=max(float(sigma[i_te[i]]),1e-8))
        l2 = torch.norm(qpt - torch.tensor(qt, dtype=torch.float32), dim=1).mean().item()
        w2 = l2 / np.sqrt(QF)
    return w2

if __name__ == '__main__':
    print(f"DeSI Dist. (Quad.) n={N}, {N_SEEDS} seeds")
    results = []; t0 = time.time()
    for i in range(N_SEEDS):
        t1 = time.time()
        try:
            w2 = run_seed(i)
            results.append(w2)
            elapsed = time.time() - t0
            eta = elapsed/(i+1)*(N_SEEDS-i-1) if i < N_SEEDS-1 else 0
            print(f"[{i+1:3d}/{N_SEEDS}] seed={i:3d} MPE={w2:.6f} ({time.time()-t1:.0f}s) ETA={eta/60:.0f}m")
        except Exception as e:
            print(f"[{i+1:3d}/{N_SEEDS}] seed={i:3d} FAILED: {e}")
    arr = np.array(results)
    print(f"\nDONE: {len(results)}/{N_SEEDS} successful")
    print(f"MPE: {arr.mean():.6f} ± {arr.std(ddof=1):.6f}")
    print(f"Paper: 0.2031 ± 0.0668")
    with open('/repo/reproduction_results.json','w') as f:
        json.dump({'n_seeds':len(results),'mpe_mean':float(arr.mean()),'mpe_std':float(arr.std(ddof=1)),'mpe_list':[float(x) for x in results],'paper_value':0.2031,'paper_std':0.0668}, f, indent=2)
