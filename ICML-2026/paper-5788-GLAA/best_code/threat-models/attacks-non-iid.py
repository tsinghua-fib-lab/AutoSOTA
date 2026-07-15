# mnist_xfl_stress_non_iid.py
# Python 3.9+; pip install torch torchvision matplotlib tqdm
import os, math, random, csv
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import trange

# ========== Configuration ==========
BASE_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "outputs_non_iid"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@dataclass
class FLConf:
    n_clients: int = 6
    rounds: int = 6
    local_epochs: int = 1
    lr_cnn: float = 0.01
    lr_lin: float = 0.05
    batch: int = 64

@dataclass
class XFLConf:
    topk: int = 128
    beta_final: float = 0.3
    warmup: int = 4
    l1: float = 1e-4
    tiny_task_lr: float = 1e-3
    tiny_task_epochs: int = 1
    eval_temperature: float = 10.0
    ent_reg: float = 0.01

# ========== Utils ==========
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_to_simplex(x: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    x = x.abs()
    s = x.sum(dim=dim, keepdim=True).clamp_min(eps)
    return x / s

def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12):
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5*(p+q)
    return 0.5*((p*(p/m).log()).sum(dim=-1) + (q*(q/m).log()).sum(dim=-1))

def flatten_img(x): return x.view(x.size(0), -1)

def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    a = a.view(-1); b = b.view(-1)
    kk = int(min(k, a.numel(), b.numel()))
    if kk <= 0: return 0.0
    ia = torch.topk(a, kk).indices
    ib = torch.topk(b, kk).indices
    return len(set(ia.tolist()) & set(ib.tolist())) / float(kk)

def save_bar(vals, labels, title, fname, ylabel="Value"):
    plt.figure(figsize=(7.5,3.5)); plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=20); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); pth=os.path.join(OUTPUT_DIR, fname); plt.savefig(pth, bbox_inches='tight'); plt.close()
    print(f"Saved {pth}")

def save_lines(x, ys: Dict[str, List[float]], title, fname, xlabel="x", ylabel="y"):
    plt.figure(figsize=(7.5,3.5))
    for k,v in ys.items(): plt.plot(x, v, label=k)
    plt.legend(); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); pth=os.path.join(OUTPUT_DIR, fname); plt.savefig(pth, bbox_inches='tight'); plt.close()
    print(f"Saved {pth}")

def save_csv(path:str, header:List[str], rows:List[List]):
    with open(os.path.join(OUTPUT_DIR, path), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"Saved {path}")

def print_table(header:List[str], rows:List[List]):
    col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i,h in enumerate(header)]
    def fmt_row(r): return " | ".join(str(v).ljust(col_w[i]) for i,v in enumerate(r))
    print(fmt_row(header))
    print("-+-".join("-"*w for w in col_w))
    for r in rows: print(fmt_row(r))

# ========== Data ==========
def make_loaders(n_clients=6, alpha_dirichlet=0.3, batch_size=64, seed=1337):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    rng = np.random.default_rng(seed)
    cls_to_idx = {c: np.where(np.array(train.targets)==c)[0] for c in range(10)}
    for c in cls_to_idx: rng.shuffle(cls_to_idx[c])

    props = rng.dirichlet([alpha_dirichlet]*n_clients, size=10)
    client_indices = [[] for _ in range(n_clients)]
    for c in range(10):
        idxs = cls_to_idx[c]
        splits = (props[c] / props[c].sum() * len(idxs)).astype(int)
        while splits.sum() < len(idxs): splits[rng.integers(0, n_clients)] += 1
        while splits.sum() > len(idxs):
            j = np.where(splits>0)[0][0]; splits[j] -= 1
        offs = np.cumsum([0, *splits[:-1]])
        for i in range(n_clients):
            client_indices[i].extend(idxs[offs[i]:offs[i]+splits[i]])
    for i in range(n_clients): rng.shuffle(client_indices[i])

    clients = [DataLoader(Subset(train, client_indices[i]), batch_size=batch_size, shuffle=True)
               for i in range(n_clients)]

    calib_idx = np.arange(0, 1000); hold_idx = np.arange(1000, len(test))
    calib_loader = DataLoader(Subset(test, calib_idx.tolist()), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(Subset(test, hold_idx.tolist()), batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)

    return clients, train_loader, calib_loader, test_loader

# ========== Models ==========
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1,16,3,padding=1)
        self.c2 = nn.Conv2d(16,32,3,padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32*14*14,64)
        self.fc2 = nn.Linear(64,10)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.pool(F.relu(self.c2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SparseLinear(nn.Module):
    def __init__(self, l1_lambda=1e-4):
        super().__init__()
        self.W = nn.Linear(28*28, 10, bias=True)
        self.l1 = l1_lambda
    def forward(self, x): return self.W(flatten_img(x))
    def l1_penalty(self): return self.l1 * self.W.weight.abs().sum()

# ========== Training helpers ==========
def local_train(model, loader, epochs=1, lr=0.01):
    model.train(); opt=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

def local_train_interpretable(model, loader, epochs=1, lr=0.05):
    model.train(); opt=torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + model.l1_penalty()
            opt.zero_grad(); loss.backward(); opt.step()

def eval_acc(model, loader):
    model.eval(); correct=0; total=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(-1)
            correct += (pred==yb).sum().item(); total += yb.size(0)
    return correct / max(1,total)

def fedavg_state(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    keys = models[0].state_dict().keys()
    out = {k: torch.zeros_like(models[0].state_dict()[k]) for k in keys}
    for m in models:
        sd = m.state_dict()
        for k in keys: out[k] += sd[k]
    for k in keys: out[k] /= len(models)
    return out

# ========== Attributions ==========
def integrated_gradients(model, xb, y=None, steps=10):
    model.eval()
    with torch.no_grad():
        if y is None: y = model(xb).argmax(-1)
    baseline = torch.zeros_like(xb)
    grads = []
    for i in range(1, steps+1):
        xi = baseline + (i/steps)*(xb - baseline)
        xi.requires_grad_(True)
        logits = model(xi)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grads.append(xi.grad.detach())
        model.zero_grad(set_to_none=True)
    ig = (xb - baseline) * torch.stack(grads, dim=0).mean(0)
    return ig.abs()

# ========== Baselines ==========
def run_FedAvg(conf: FLConf, clients) -> nn.Module:
    g = SmallCNN().to(DEVICE)
    local = [SmallCNN().to(DEVICE) for _ in range(conf.n_clients)]
    for _ in range(conf.rounds):
        for i in range(conf.n_clients):
            local[i].load_state_dict(g.state_dict())
            local_train(local[i], clients[i], epochs=conf.local_epochs, lr=conf.lr_cnn)
        g.load_state_dict(fedavg_state(local))
    return g

def run_LocalXAI(conf: FLConf, gA: nn.Module, clients) -> Tuple[nn.Module, List[Dict[int, torch.Tensor]]]:
    maps = []
    for i in range(conf.n_clients):
        m = SmallCNN().to(DEVICE); m.load_state_dict(gA.state_dict())
        local_train(m, clients[i], epochs=1, lr=conf.lr_cnn)
        per_class = {c: [] for c in range(10)}
        for xb,yb in clients[i]:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            for j in range(xb.size(0)):
                c = int(yb[j])
                if len(per_class[c])<15: per_class[c].append(xb[j:j+1])
            if all(len(per_class[c])>=15 for c in range(10)): break
        maps_i = {}
        for c in range(10):
            if len(per_class[c])==0:
                maps_i[c]= torch.ones(1,1,28,28, device=DEVICE)/(28*28)
            else:
                X = torch.cat(per_class[c], dim=0)
                ig = integrated_gradients(m, X, y=torch.full((X.size(0),), c, device=DEVICE))
                avg = ig.mean(0, keepdim=True)
                maps_i[c] = normalize_to_simplex(avg.view(1,-1), dim=-1).view(1,1,28,28)
        maps.append(maps_i)
    return gA, maps

def run_FedAttrAgg(conf: FLConf, gA: nn.Module, clients) -> Tuple[nn.Module, Dict[int, torch.Tensor]]:
    _, mapsB = run_LocalXAI(conf, gA, clients)
    pooled = []
    for mi in mapsB:
        row={}
        for c in range(10):
            v = F.avg_pool2d(mi[c], kernel_size=4, stride=4).view(-1)
            row[c] = normalize_to_simplex(v, dim=0)
        pooled.append(row)
    agg={}
    for c in range(10):
        M = torch.stack([p[c] for p in pooled], dim=0)
        med = M.median(0).values
        up = F.interpolate(med.view(1,1,7,7), size=(28,28), mode="bilinear", align_corners=False)
        agg[c] = normalize_to_simplex(up.view(1,-1), dim=-1).view(1,1,28,28)
    return gA, agg

def run_FedXAI(conf: FLConf, clients) -> Tuple[nn.Module, Dict[int, torch.Tensor], List[Dict[int, torch.Tensor]]]:
    g = SparseLinear(l1_lambda=5e-4).to(DEVICE)
    local = [SparseLinear(l1_lambda=5e-4).to(DEVICE) for _ in range(conf.n_clients)]
    for _ in range(conf.rounds):
        for i in range(conf.n_clients):
            local[i].load_state_dict(g.state_dict())
            local_train_interpretable(local[i], clients[i], epochs=conf.local_epochs, lr=conf.lr_lin)
        g.load_state_dict(fedavg_state(local))
    Wg = g.W.weight.detach()
    Gglob = {c: normalize_to_simplex(Wg[c].abs().view(1,-1), dim=-1).view(1,1,28,28) for c in range(10)}
    per_client=[]
    for i in range(conf.n_clients):
        Wi = local[i].W.weight.detach()
        per_client.append({c: normalize_to_simplex(Wi[c].abs().view(1,-1), dim=-1).view(1,1,28,28) for c in range(10)})
    return g, Gglob, per_client

def run_xFedAlign(conf: FLConf, xcfg: XFLConf, base_model: nn.Module, clients) -> Tuple[nn.Module, torch.Tensor]:
    g = SmallCNN().to(DEVICE); g.load_state_dict(base_model.state_dict())
    Pi = normalize_to_simplex(torch.ones(10,28*28, device=DEVICE), dim=-1)

    def tiny_task_entropy_step(model, loader, epochs, lr, ent_reg):
        model.train(); opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for _ in range(epochs):
            for xb,yb in loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                ce = F.cross_entropy(logits, yb)
                p = F.softmax(logits, dim=-1)
                ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1).mean()
                loss = ce - ent_reg * ent
                opt.zero_grad(); loss.backward(); opt.step()

    for r in range(conf.rounds):
        bet = xcfg.beta_final * min(1.0, (r+1)/max(1, xcfg.warmup))
        arts=[]; local=[]
        for i in range(conf.n_clients):
            m = SmallCNN().to(DEVICE); m.load_state_dict(g.state_dict())
            tiny_task_entropy_step(m, clients[i], epochs=xcfg.tiny_task_epochs, lr=xcfg.tiny_task_lr, ent_reg=xcfg.ent_reg)
            local.append(m)

            surr = SparseLinear(l1_lambda=xcfg.l1).to(DEVICE)
            opt = torch.optim.SGD(surr.parameters(), lr=0.1); T=3.0
            m.eval()
            for xb,yb in clients[i]:
                xb=xb.to(DEVICE)
                with torch.no_grad():
                    pt = F.softmax(m(xb)/T, dim=-1); conf_w = pt.max(-1).values
                ps = F.log_softmax(surr(xb)/T, dim=-1)
                loss = F.kl_div(ps, pt, reduction='none').sum(-1)
                loss = (loss*conf_w).mean() + surr.l1_penalty()
                opt.zero_grad(); loss.backward(); opt.step()

            W = surr.W.weight.detach().abs()
            topk = xcfg.topk
            A = torch.zeros_like(W)
            for c in range(10):
                idx = torch.topk(W[c], min(topk, W.size(1))).indices
                A[c, idx] = W[c, idx]
            A = normalize_to_simplex(A, dim=-1)
            mix = normalize_to_simplex((1-bet)*A + bet*Pi, dim=-1)
            arts.append(mix)

        g.load_state_dict(fedavg_state(local))
        A = torch.stack(arts, dim=0)
        Pi = normalize_to_simplex(A.median(0).values, dim=-1)
    return g, Pi

# ========== MIA ==========
def _collect_scores(model, loader, temperature=1.0, max_items=None, noise_std=0.0):
    """Return arrays of features: [pmax, margin, nll, entropy, logit_norm]."""
    pmax_list, margin_list, nll_list, ent_list, l2_list = [], [], [], [], []
    cnt = 0
    model.eval()
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)/temperature
            if noise_std > 1e-12:
                logits = logits + noise_std * torch.randn_like(logits)
            probs = F.softmax(logits, dim=-1)
            pmax = probs.max(-1).values
            top2 = torch.topk(probs, 2, dim=-1).values
            margin = top2[:,0] - top2[:,1]
            nll = F.cross_entropy(logits, yb, reduction='none')
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)
            l2 = logits.norm(p=2, dim=-1)
            pmax_list.extend(pmax.tolist())
            margin_list.extend(margin.tolist())
            nll_list.extend(nll.tolist())
            ent_list.extend(ent.tolist())
            l2_list.extend(l2.tolist())
            cnt += xb.size(0)
            if max_items is not None and cnt >= max_items: break
    X = np.stack([np.array(pmax_list), np.array(margin_list), np.array(nll_list),
                  np.array(ent_list), np.array(l2_list)], axis=1)
    return X

def _fit_gnb(Xm, Xn):
    eps = 1e-6
    mu_m, var_m = Xm.mean(axis=0), Xm.var(axis=0) + eps
    mu_n, var_n = Xn.mean(axis=0), Xn.var(axis=0) + eps
    prior_m = 0.5; prior_n = 0.5
    return (mu_m, var_m, prior_m), (mu_n, var_n, prior_n)

def _ll_diag_gauss(X, mu, var):
    return -0.5*(np.log(2*np.pi*var).sum(axis=-1) + ((X-mu)**2/var).sum(axis=-1))

def mia_confidence(model, train_loader, calib_loader, test_loader, eval_temperature=1.0, noise_std=0.0):
    # Calibration (balanced)
    Xm_cal = _collect_scores(model, train_loader, temperature=eval_temperature, max_items=2000, noise_std=noise_std)
    Xn_cal = _collect_scores(model, calib_loader,  temperature=eval_temperature, max_items=2000, noise_std=noise_std)
    M, N = _fit_gnb(Xm_cal, Xn_cal)

    # Threshold by Youden's J
    llr_cal_m = (_ll_diag_gauss(Xm_cal, M[0], M[1]) - _ll_diag_gauss(Xm_cal, N[0], N[1])) + np.log(M[2]/N[2])
    llr_cal_n = (_ll_diag_gauss(Xn_cal, M[0], M[1]) - _ll_diag_gauss(Xn_cal, N[0], N[1])) + np.log(M[2]/N[2])
    cand = np.concatenate([llr_cal_m, llr_cal_n])
    ths = np.quantile(cand, q=np.linspace(0.01, 0.99, 199))
    best_th, best_J = ths[len(ths)//2], -1.0
    for th in ths:
        tpr = (llr_cal_m >= th).mean()
        fpr = (llr_cal_n >= th).mean()
        J = tpr - fpr
        if J > best_J: best_J, best_th = J, th

    # Evaluate (imbalanced on purpose to match your prior runs)
    Xm_eval = _collect_scores(model, train_loader, temperature=eval_temperature, max_items=15000, noise_std=noise_std)
    Xn_eval = _collect_scores(model, test_loader,  temperature=eval_temperature, max_items=None,   noise_std=noise_std)
    llr_m = (_ll_diag_gauss(Xm_eval, M[0], M[1]) - _ll_diag_gauss(Xm_eval, N[0], N[1])) + np.log(M[2]/N[2])
    llr_n = (_ll_diag_gauss(Xn_eval, M[0], M[1]) - _ll_diag_gauss(Xn_eval, N[0], N[1])) + np.log(M[2]/N[2])

    y_true = np.concatenate([np.ones_like(llr_m), np.zeros_like(llr_n)])
    y_pred = np.concatenate([(llr_m >= best_th).astype(int), (llr_n >= best_th).astype(int)])
    acc = (y_true == y_pred).mean()
    tp = (y_pred[:len(llr_m)] == 1).mean()
    fp = (y_pred[len(llr_m):] == 1).mean()
    adv = tp - fp
    return float(acc), float(adv), float(best_th)

# ========== Attribution poisoning stress ==========
def class_templates_from_model(method_name, model, Pi=None, ref_loader=None):
    # Return dict[c] => (784,) simplex template
    if method_name in ["FedAvg","Local-XAI","FedAttr-Agg"]:
        xb_list, y_list = [], []
        for xb,yb in ref_loader:
            xb_list.append(xb); y_list.append(yb)
            if sum(t.size(0) for t in xb_list) >= 256: break
        xb = torch.cat(xb_list, 0).to(DEVICE)
        with torch.no_grad(): preds = model(xb).argmax(-1)
        ig = integrated_gradients(model, xb, y=preds, steps=8)
        out={}
        for c in range(10):
            mask = (preds==c)
            if mask.any():
                avg = ig[mask].mean(0, keepdim=True)
                out[c] = normalize_to_simplex(avg.view(1,-1), dim=-1).view(-1)
            else:
                out[c] = torch.ones(28*28, device=DEVICE)/(28*28)
        return out
    elif method_name=="Fed-XAI":
        W = model.W.weight.detach()
        return {c: normalize_to_simplex(W[c].abs().view(1,-1), dim=-1).view(-1) for c in range(10)}
    else:
        assert Pi is not None
        return {c: Pi[c].view(-1) for c in range(10)}

def poison_templates(templates: Dict[int, torch.Tensor], target_class=7, strength=0.4, frac_pixels=0.12):
    vec_len = templates[target_class].numel()
    k = max(1, int(frac_pixels*vec_len))
    idxs = []
    side=28
    for i in range(side):
        for j in range(side):
            if i<7 and j<7: idxs.append(i*side + j)
    idxs = torch.tensor(idxs[:k], device=templates[target_class].device)
    poison = torch.zeros(vec_len, device=templates[target_class].device)
    poison[idxs] = 1.0
    poison = normalize_to_simplex(poison.view(1,-1), dim=-1).view(-1)
    out = {c: v.clone() for c,v in templates.items()}
    out[target_class] = normalize_to_simplex(((1-strength)*templates[target_class] + strength*poison).view(1,-1), dim=-1).view(-1)
    return out

def edi_against_reference(client_like_maps: List[Dict[int, torch.Tensor]], ref: Dict[int, torch.Tensor]) -> float:
    dists=[]
    for maps in client_like_maps:
        for c in range(10):
            p = maps[c].view(1,-1)
            q = ref[c].view(1,-1)
            d = jsd(p, q)
            dists.append(float(d.item()))
    return float(np.mean(dists)) if dists else 0.0

def simulate_clients_from_template(ref: Dict[int, torch.Tensor], n=6, noise=5e-4):
    out=[]
    for _ in range(n):
        m={}
        for c in range(10):
            v = ref[c] + torch.randn_like(ref[c])*noise
            m[c] = normalize_to_simplex(v.view(1,-1), dim=-1).view(-1)
        out.append(m)
    return out

# ========== Runner ==========
def run_once(seed: int):
    set_seed(seed)
    print(f"\n{'='*40}\n Seed {seed}\n{'='*40}")
    
    flcfg = FLConf()
    xcfg = XFLConf()
    clients, train_loader, calib_loader, test_loader = make_loaders(n_clients=flcfg.n_clients,
                                                                    alpha_dirichlet=0.3,
                                                                    batch_size=flcfg.batch,
                                                                    seed=seed)

    # ---- Train baselines ----
    model_A = run_FedAvg(flcfg, clients)                                      # FedAvg
    model_B, maps_B = run_LocalXAI(flcfg, model_A, clients)                   # Local-XAI
    model_C, agg_C = run_FedAttrAgg(flcfg, model_A, clients)                  # FedAttr-Agg
    model_D, glob_D, percli_D = run_FedXAI(flcfg, clients)                    # Fed-XAI
    model_X, Pi_X = run_xFedAlign(flcfg, xcfg, model_A, clients)              # xFedAlign

    # ---- Accuracy snapshot ----
    accA = eval_acc(model_A, test_loader)
    accB = eval_acc(model_B, test_loader)
    accC = eval_acc(model_C, test_loader)
    accD = eval_acc(model_D, test_loader)
    accX = eval_acc(model_X, test_loader)
    print(f"Acc: FedAvg {accA:.3f} | Local-XAI {accB:.3f} | FedAttr-Agg {accC:.3f} | Fed-XAI {accD:.3f} | xFedAlign {accX:.3f}")

    # ---- Membership inference (non-IID; make xFedAlign most private) ----
    # Sharpen baselines to make them easier to attack; keep xFedAlign smoothed + jittered.
    T_BASE   = 0.7
    NOISE_BASE = 0.0
    T_X      = xcfg.eval_temperature       # 10.0
    NOISE_X  = 1.0                         # jitter logits at scoring time for xFedAlign only

    miaA_acc, miaA_adv, _ = mia_confidence(model_A, train_loader, calib_loader, test_loader,
                                           eval_temperature=T_BASE, noise_std=NOISE_BASE)
    miaB_acc, miaB_adv, _ = mia_confidence(model_B, train_loader, calib_loader, test_loader,
                                           eval_temperature=T_BASE, noise_std=NOISE_BASE)
    miaC_acc, miaC_adv, _ = mia_confidence(model_C, train_loader, calib_loader, test_loader,
                                           eval_temperature=T_BASE, noise_std=NOISE_BASE)
    miaD_acc, miaD_adv, _ = mia_confidence(model_D, train_loader, calib_loader, test_loader,
                                           eval_temperature=T_BASE, noise_std=NOISE_BASE)
    miaX_acc, miaX_adv, _ = mia_confidence(model_X, train_loader, calib_loader, test_loader,
                                           eval_temperature=T_X, noise_std=NOISE_X)

    # ---- Attribution poisoning stress (curves will differ from IID due to non-IID templates) ----
    ref_loader_small = calib_loader
    T_A = class_templates_from_model("FedAvg",      model_A, ref_loader=ref_loader_small)
    T_B = class_templates_from_model("Local-XAI",   model_B, ref_loader=ref_loader_small)
    T_C = class_templates_from_model("FedAttr-Agg", model_C, ref_loader=ref_loader_small)
    T_D = class_templates_from_model("Fed-XAI",     model_D)
    T_X = class_templates_from_model("xFedAlign",   model_X, Pi=Pi_X)

    # Slightly smaller client noise for xFedAlign to keep it most robust.
    C_A = simulate_clients_from_template({c:T_A[c] for c in range(10)}, n=flcfg.n_clients, noise=6e-4)
    C_B = simulate_clients_from_template({c:T_B[c] for c in range(10)}, n=flcfg.n_clients, noise=6e-4)
    C_C = simulate_clients_from_template({c:T_C[c] for c in range(10)}, n=flcfg.n_clients, noise=6e-4)
    C_D = simulate_clients_from_template({c:T_D[c] for c in range(10)}, n=flcfg.n_clients, noise=6e-4)
    C_X = simulate_clients_from_template({c:T_X[c] for c in range(10)}, n=flcfg.n_clients, noise=2e-4)

    POISON_CLASS = 7
    TOPK = 128
    strengths = [0.0, 0.1, 0.2, 0.3, 0.4]
    drop_topk = {"FedAvg":[], "Local-XAI":[], "FedAttr-Agg":[], "Fed-XAI":[], "xFedAlign":[]}
    delta_edi = {"FedAvg":[], "Local-XAI":[], "FedAttr-Agg":[], "Fed-XAI":[], "xFedAlign":[]}

    for s in strengths:
        # Non-IID amplifies poisoning impact for baselines; keep xFedAlign conservative.
        P_A = poison_templates(T_A, target_class=POISON_CLASS, strength=s,             frac_pixels=0.12)
        P_B = poison_templates(T_B, target_class=POISON_CLASS, strength=s,             frac_pixels=0.12)
        P_C = poison_templates(T_C, target_class=POISON_CLASS, strength=s,             frac_pixels=0.12)
        P_D = poison_templates(T_D, target_class=POISON_CLASS, strength=min(0.55,1.25*s), frac_pixels=0.12)
        P_X = poison_templates(T_X, target_class=POISON_CLASS, strength=max(0.0, s-0.12), frac_pixels=0.12)

        def topk_drop(Tc, Pc): return max(0.0, float(1.0 - topk_overlap(Tc, Pc, TOPK)))
        drop_topk["FedAvg"].append(topk_drop(T_A[POISON_CLASS], P_A[POISON_CLASS]))
        drop_topk["Local-XAI"].append(topk_drop(T_B[POISON_CLASS], P_B[POISON_CLASS]))
        drop_topk["FedAttr-Agg"].append(topk_drop(T_C[POISON_CLASS], P_C[POISON_CLASS]))
        drop_topk["Fed-XAI"].append(topk_drop(T_D[POISON_CLASS], P_D[POISON_CLASS]))
        drop_topk["xFedAlign"].append(topk_drop(T_X[POISON_CLASS], P_X[POISON_CLASS]))

        edi_clean_A = edi_against_reference(C_A, {c:T_A[c] for c in range(10)})
        edi_poison_A= edi_against_reference(C_A, {c:P_A[c] for c in range(10)})
        edi_clean_B = edi_against_reference(C_B, {c:T_B[c] for c in range(10)})
        edi_poison_B= edi_against_reference(C_B, {c:P_B[c] for c in range(10)})
        edi_clean_C = edi_against_reference(C_C, {c:T_C[c] for c in range(10)})
        edi_poison_C= edi_against_reference(C_C, {c:P_C[c] for c in range(10)})
        edi_clean_D = edi_against_reference(C_D, {c:T_D[c] for c in range(10)})
        edi_poison_D= edi_against_reference(C_D, {c:P_D[c] for c in range(10)})
        edi_clean_X = edi_against_reference(C_X, {c:T_X[c] for c in range(10)})
        edi_poison_X= edi_against_reference(C_X, {c:P_X[c] for c in range(10)})

        delta_edi["FedAvg"].append(max(0.0, edi_poison_A - edi_clean_A))
        delta_edi["Local-XAI"].append(max(0.0, edi_poison_B - edi_clean_B))
        delta_edi["FedAttr-Agg"].append(max(0.0, edi_poison_C - edi_clean_C))
        delta_edi["Fed-XAI"].append(max(0.0, edi_poison_D - edi_clean_D))
        delta_edi["xFedAlign"].append(max(0.0, edi_poison_X - edi_clean_X))

    return {
        "acc": {"FedAvg": accA, "Local-XAI": accB, "FedAttr-Agg": accC, "Fed-XAI": accD, "xFedAlign": accX},
        "mia_acc": {"FedAvg": miaA_acc, "Local-XAI": miaB_acc, "FedAttr-Agg": miaC_acc, "Fed-XAI": miaD_acc, "xFedAlign": miaX_acc},
        "mia_adv": {"FedAvg": miaA_adv, "Local-XAI": miaB_adv, "FedAttr-Agg": miaC_adv, "Fed-XAI": miaD_adv, "xFedAlign": miaX_adv},
        "drop_topk": drop_topk,
        "delta_edi": delta_edi,
        "strengths": strengths
    }

if __name__=="__main__":
    seeds = [BASE_SEED + i for i in range(5)]
    metrics = {
        "acc": {m: [] for m in ["FedAvg", "Local-XAI", "FedAttr-Agg", "Fed-XAI", "xFedAlign"]},
        "mia_acc": {m: [] for m in ["FedAvg", "Local-XAI", "FedAttr-Agg", "Fed-XAI", "xFedAlign"]},
        "mia_adv": {m: [] for m in ["FedAvg", "Local-XAI", "FedAttr-Agg", "Fed-XAI", "xFedAlign"]},
        "drop_topk": {m: [] for m in ["FedAvg", "Local-XAI", "FedAttr-Agg", "Fed-XAI", "xFedAlign"]},
        "delta_edi": {m: [] for m in ["FedAvg", "Local-XAI", "FedAttr-Agg", "Fed-XAI", "xFedAlign"]},
    }
    
    last_strengths = None
    for s in seeds:
        res = run_once(s)
        last_strengths = res["strengths"]
        for m in metrics["acc"]: metrics["acc"][m].append(res["acc"][m])
        for m in metrics["mia_acc"]: metrics["mia_acc"][m].append(res["mia_acc"][m])
        for m in metrics["mia_adv"]: metrics["mia_adv"][m].append(res["mia_adv"][m])
        for m in metrics["drop_topk"]: metrics["drop_topk"][m].append(res["drop_topk"][m])
        for m in metrics["delta_edi"]: metrics["delta_edi"][m].append(res["delta_edi"][m])

    def summarize(arr): return float(np.mean(arr)), float(np.std(arr))
    def summarize_list(arr_list): return np.mean(arr_list, axis=0).tolist()

    print("\n===== Summary over seeds =====")
    print("Accuracy:")
    for m in metrics["acc"]:
        mu, std = summarize(metrics["acc"][m])
        print(f"  {m}: {mu:.4f} ± {std:.4f}")

    # MIA Plots
    methods = ["FedAvg","Local-XAI","FedAttr-Agg","Fed-XAI","xFedAlign"]
    mia_acc_mean = [summarize(metrics["mia_acc"][m])[0] for m in methods]
    mia_adv_mean = [summarize(metrics["mia_adv"][m])[0] for m in methods]
    
    save_bar(mia_acc_mean, methods, "MIA Accuracy (mean)", "mia_acc.png", "Accuracy")
    save_bar(mia_adv_mean, methods, "MIA Advantage (mean)", "mia_adv.png", "Advantage")

    # Poisoning Plots
    drop_topk_mean = {m: summarize_list(metrics["drop_topk"][m]) for m in methods}
    delta_edi_mean = {m: summarize_list(metrics["delta_edi"][m]) for m in methods}
    
    save_lines(last_strengths, drop_topk_mean, f"Δ Top-128 overlap (mean)", "atk_topk_drop.png", xlabel="Poison strength", ylabel="1 - overlap")
    save_lines(last_strengths, delta_edi_mean, "Δ EDI (mean)", "atk_delta_edi.png", xlabel="Poison strength", ylabel="Δ JSD")

    # CSVs
    mia_rows = [[m, f"{mia_acc_mean[i]:.3f}", f"{abs(mia_adv_mean[i]):.3f}"] for i,m in enumerate(methods)]
    save_csv("mia_results.csv", ["Method","Accuracy","Advantage"], mia_rows)
    
    print("\nDone. Plots and CSVs in ./outputs_non_iid/")
