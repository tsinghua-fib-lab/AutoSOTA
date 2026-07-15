"""
xFL vs Baselines on UCI Adult (IID)
Clean, structured implementation for multi-seed experiments.
"""

import os
import re
import time
import random
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Global Configuration
# ----------------------------
BASE_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = ['Age','Workclass','fnlwgt','Education','Education-Num','Martial Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours per week','Country','Income']
SENSITIVE_PATTERNS = [r"^Sex_", r"^Martial Status_", r"^Race_", r"^Country_"]

# ----------------------------
# Configuration Classes
# ----------------------------
@dataclass
class FLConfig:
    n_clients: int = 6
    rounds: int = 4
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3

@dataclass
class XFLConfig:
    topk: int = 48
    quant_bits: int = 8
    clip_radius: float = 5.0
    dp_sigma: float = 0.02
    temperature: float = 2.5
    beta_final: float = 0.95
    warmup: int = 2
    every_R: int = 1
    l1: float = 3e-5
    hybrid_alpha: float = 0.15
    sens_penalty: float = 0.02

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Data Utilities
# ----------------------------
def load_raw_data():
    """Load raw data from URL or local cache."""
    local_path = "adult.data"
    if os.path.exists(local_path):
        print(f"Loading from local {local_path}")
        df = pd.read_csv(local_path, names=COLUMNS, sep=r"\s*,\s*", engine="python", na_values="?")
    else:
        print(f"Downloading from {DATA_URL}")
        try:
            df = pd.read_csv(DATA_URL, names=COLUMNS, sep=r"\s*,\s*", engine="python", na_values="?")
            df.to_csv(local_path, index=False, header=False)
        except Exception as e:
            print(f"Failed to download: {e}. Generating synthetic data.")
            return generate_synthetic_adult()
    return df

def generate_synthetic_adult(n=1000):
    # Fallback if download fails
    data = {c: [] for c in COLUMNS}
    for _ in range(n):
        data['Age'].append(random.randint(18, 90))
        data['Workclass'].append(random.choice(['Private', 'Self-emp', 'Gov']))
        data['fnlwgt'].append(random.randint(10000, 200000))
        data['Education'].append(random.choice(['HS-grad', 'Bachelors', 'Masters']))
        data['Education-Num'].append(random.randint(9, 14))
        data['Martial Status'].append(random.choice(['Married', 'Single', 'Divorced']))
        data['Occupation'].append(random.choice(['Tech', 'Sales', 'Exec']))
        data['Relationship'].append(random.choice(['Wife', 'Husband', 'Own-child']))
        data['Race'].append(random.choice(['White', 'Black', 'Asian']))
        data['Sex'].append(random.choice(['Male', 'Female']))
        data['Capital Gain'].append(random.randint(0, 10000))
        data['Capital Loss'].append(random.randint(0, 2000))
        data['Hours per week'].append(random.randint(20, 60))
        data['Country'].append('US')
        data['Income'].append(random.choice(['<=50K', '>50K']))
    return pd.DataFrame(data)

def preprocess_data(df: pd.DataFrame, seed: int):
    df = df.dropna().reset_index(drop=True)
    df["Income"] = (df["Income"].str.strip() == ">50K").astype(np.int64)

    numeric = ["Age","fnlwgt","Education-Num","Capital Gain","Capital Loss","Hours per week"]
    cat = [c for c in df.columns if c not in numeric + ["Income"]]

    X_num = pd.DataFrame(StandardScaler().fit_transform(df[numeric]),
                         columns=numeric, index=df.index)
    X_cat = pd.get_dummies(df[cat], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1).astype(np.float32)
    y = df["Income"].values.astype(np.int64)
    feats = list(X.columns)

    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=seed
    )
    
    # Identify sensitive features indices
    sens_idx = np.array([i for i,n in enumerate(feats)
                         if any(re.match(p,n) for p in SENSITIVE_PATTERNS)], dtype=np.int64)
    
    return Xtr, ytr, Xte, yte, feats, sens_idx

# ----------------------------
# Helpers
# ----------------------------
def iid_split(X, y, n_clients=6):
    idx = np.arange(len(X)); np.random.shuffle(idx)
    parts = np.array_split(idx, n_clients)
    return [(X[p], y[p]) for p in parts]

def normalize_simplex(v, axis=-1, eps=1e-12):
    v = np.abs(v); s = np.sum(v, axis=axis, keepdims=True).clip(min=eps)
    return v / s

def jsd(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0); m = 0.5*(p+q)
    return 0.5*(np.sum(p*np.log(p/m), axis=-1) + np.sum(q*np.log(q/m), axis=-1))

def auc_area(xs, ys):
    return float(np.trapz(ys, xs))

def to_loader(X, y, bs=256, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float().unsqueeze(1))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def train_epoch(model, loader, opt, extra_loss_fn=None):
    model.train(); tot=0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        p = model(xb)
        loss = F.binary_cross_entropy(p, yb)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model, xb, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.item())
    return tot/max(1,len(loader))

def evaluate_acc(model, X, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy().reshape(-1)
    return ( (p>=0.5).astype(np.int64) == y ).mean().item()

def model_prob_true(model, Xb, yb):
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1)
    yb = yb.reshape(-1)
    return float(np.mean(yb*p + (1-yb)*(1-p)))

# ----------------------------
# Models
# ----------------------------
class MLP(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, h//2)
        self.out = nn.Linear(h//2, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

class SparseLinear(nn.Module):
    def __init__(self, d, l1=1e-3):
        super().__init__()
        self.W = nn.Linear(d, 1, bias=True)
        self.l1 = l1
    def forward(self, x): return torch.sigmoid(self.W(x))
    def l1_penalty(self): return self.l1*self.W.weight.abs().sum()

# ----------------------------
# Attribution & Fidelity
# ----------------------------
def integrated_gradients(model, Xb, yb, baseline, steps=32):
    model.eval()
    xb = torch.from_numpy(Xb).to(DEVICE)
    y = torch.from_numpy(yb.reshape(-1,1)).float().to(DEVICE)
    base = torch.from_numpy(np.broadcast_to(baseline, Xb.shape).astype(np.float32)).to(DEVICE)

    alphas = torch.linspace(0, 1, steps, device=DEVICE).view(-1, 1, 1)
    path = (base.unsqueeze(0) + alphas * (xb.unsqueeze(0) - base.unsqueeze(0))).requires_grad_(True)

    preds = model(path.reshape(-1, Xb.shape[1])).reshape(steps, -1, 1)
    score = y.unsqueeze(0) * preds + (1 - y).unsqueeze(0) * (1 - preds)

    grads = torch.autograd.grad(score.sum(), path, retain_graph=False)[0]
    ig = (xb - base) * grads.mean(dim=0)

    s = ig.abs().detach().cpu().numpy()
    return normalize_simplex(s, axis=1)

def del_ins_auc(model, Xb, yb, imp, steps=20, base=None):
    B,d = Xb.shape; order = np.argsort(-imp,axis=1)
    xs=[0.0]; del_s=[model_prob_true(model,Xb,yb)]
    ins_s=[model_prob_true(model,np.broadcast_to(base,Xb.shape).astype(np.float32),yb)]
    for s in range(1,steps+1):
        frac=s/steps; k=int(frac*d); xs.append(frac)
        Xd=Xb.copy(); Xi=np.broadcast_to(base,Xb.shape).astype(np.float32)
        for i in range(B):
            if k>0:
                idx=order[i,:k]
                Xd[i,idx]=base[idx]
                Xi[i,idx]=Xb[i,idx]
        del_s.append(model_prob_true(model,Xd,yb))
        ins_s.append(model_prob_true(model,Xi,yb))
    return auc_area(xs,del_s), auc_area(xs,ins_s)

# ----------------------------
# xFL Logic
# ----------------------------
def _logit(p,eps=1e-7):
    p=np.clip(p,eps,1-eps); return np.log(p/(1-p))

def distill_surrogate(task_model, Xc, cfg: XFLConfig, n_features, sens_idx):
    task_model.eval()
    with torch.no_grad():
        p = task_model(torch.from_numpy(Xc).to(DEVICE)).cpu().numpy().reshape(-1)
    zT = _logit(p)/cfg.temperature
    soft = 1/(1+np.exp(-zT))
    surr = SparseLinear(n_features, l1=cfg.l1).to(DEVICE)
    opt = torch.optim.Adam(surr.parameters(), lr=2e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xc),
                                  torch.from_numpy(soft.reshape(-1,1)).float()),
                    batch_size=256, shuffle=True)
    surr.train()
    for _ in range(3):
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            loss = F.binary_cross_entropy(surr(xb), yb) + surr.l1_penalty()
            opt.zero_grad(); loss.backward(); opt.step()
    W = surr.W.weight.detach().cpu().numpy().reshape(-1)
    if len(sens_idx)>0: W[sens_idx] *= 0.3
    w_pos = np.maximum(W,0.0); w_neg = np.maximum(-W,0.0)
    return np.stack([w_neg,w_pos],axis=0)

def artifact_from_surrogate(Wc, cfg: XFLConfig):
    W=Wc.copy()
    for c in (0,1):
        idx=np.argsort(-np.abs(W[c]))[:cfg.topk]
        mask=np.zeros_like(W[c]); mask[idx]=1.0
        W[c]*=mask
        n=np.linalg.norm(W[c])+1e-8
        W[c]*=min(1.0, cfg.clip_radius/n)
    mx=np.max(np.abs(W))+1e-8; scale=(2**(cfg.quant_bits-1)-1)/mx
    Wq=np.round(W*scale)/scale
    Wq += np.random.normal(0.0, cfg.dp_sigma, size=Wq.shape)
    return normalize_simplex(Wq, axis=1)

def robust_median(arts):
    return normalize_simplex(np.median(np.stack(arts,axis=0),axis=0),axis=1)

def sensitive_penalty(model, xb, yb, sens_idx, weight=0.02, baseline=None):
    if len(sens_idx)==0 or weight<=0: return 0.0*model.out.weight.sum()
    ig = integrated_gradients(model, xb.detach().cpu().numpy(),
                              yb.detach().cpu().numpy().reshape(-1),
                              baseline=baseline, steps=8)
    return weight*torch.tensor(ig[:,sens_idx].mean(), device=xb.device, dtype=torch.float32)

# ----------------------------
# Runners
# ----------------------------
def run_fedavg(cfg: FLConfig, X_train, y_train, X_test, y_test, n_features):
    clients = iid_split(X_train,y_train,cfg.n_clients)
    g = MLP(n_features).to(DEVICE)
    cs=[MLP(n_features).to(DEVICE) for _ in range(cfg.n_clients)]
    for r in range(cfg.rounds):
        w=None; k=0
        for i,(Xi,yi) in enumerate(clients):
            cs[i].load_state_dict(g.state_dict())
            opt=torch.optim.Adam(cs[i].parameters(), lr=cfg.lr)
            dl=to_loader(Xi,yi,cfg.batch_size,True)
            for _ in range(cfg.local_epochs): train_epoch(cs[i],dl,opt)
            sd=cs[i].state_dict()
            if w is None: w={kk:vv.clone() for kk,vv in sd.items()}
            else:
                for kk in w: w[kk]+=sd[kk]
            k+=1
        for kk in w: w[kk]/=k
        g.load_state_dict(w)
    return evaluate_acc(g,X_test,y_test), g, clients

def run_bl_b(fl_base, X_train, y_train, X_test, y_test, n_features, baseline):
    acc, model, clients = run_fedavg(fl_base, X_train, y_train, X_test, y_test, n_features)
    dists=[]
    for (Xi,yi) in clients:
        n=min(4000,len(Xi)); Xs,ys=Xi[:n],yi[:n]
        S=integrated_gradients(model,Xs,ys,baseline=baseline,steps=32)
        H=np.zeros((2,n_features),dtype="float32")
        for c in (0,1):
            m=(ys==c);
            if m.sum()>0: H[c]=S[m].sum(axis=0)
        dists.append(normalize_simplex(H,axis=1))
    return acc, model, dists, clients

def run_bl_c(fl_base, X_train, y_train, X_test, y_test, n_features, baseline):
    acc, model, clients = run_fedavg(fl_base, X_train, y_train, X_test, y_test, n_features)
    ch=[]
    for (Xi,yi) in clients:
        n=min(4000,len(Xi)); Xs,ys=Xi[:n],yi[:n]
        S=integrated_gradients(model,Xs,ys,baseline=baseline,steps=32)
        H=np.zeros((2,n_features),dtype="float32")
        with torch.no_grad():
            preds=(model(torch.from_numpy(Xs).to(DEVICE)).cpu().numpy().reshape(-1)>=0.5).astype(np.int64)
        for c in (0,1):
            m=(preds==c);
            if m.sum()>0: H[c]=S[m].sum(axis=0)
        ch.append(normalize_simplex(H,axis=1))
    return acc, model, robust_median(ch), clients

def run_bl_d(fl_base, X_train, y_train, X_test, y_test, n_features):
    clients = iid_split(X_train,y_train,fl_base.n_clients)
    g = SparseLinear(n_features,l1=2e-3).to(DEVICE)
    cs=[SparseLinear(n_features,l1=2e-3).to(DEVICE) for _ in range(fl_base.n_clients)]
    for r in range(fl_base.rounds):
        w=None;k=0
        for i,(Xi,yi) in enumerate(clients):
            cs[i].load_state_dict(g.state_dict())
            opt=torch.optim.Adam(cs[i].parameters(), lr=8e-3)
            dl=to_loader(Xi,yi,256,True)
            train_epoch(cs[i],dl,opt)
            sd=cs[i].state_dict()
            if w is None: w={kk:vv.clone() for kk,vv in sd.items()}
            else:
                for kk in w: w[kk]+=sd[kk]
            k+=1
        for kk in w: w[kk]/=k
        g.load_state_dict(w)
    acc=evaluate_acc(g,X_test,y_test)
    pcs=[]
    for (Xi,yi) in clients:
        m=SparseLinear(n_features,l1=2e-3).to(DEVICE); m.load_state_dict(g.state_dict())
        opt=torch.optim.Adam(m.parameters(), lr=8e-3)
        dl=to_loader(Xi,yi,256,True); train_epoch(m,dl,opt)
        W=m.W.weight.detach().cpu().numpy().reshape(-1)
        pcs.append(normalize_simplex(np.stack([np.maximum(-W,0),np.maximum(W,0)],axis=0),axis=1))
    Wg=g.W.weight.detach().cpu().numpy().reshape(-1)
    Gdist=normalize_simplex(np.stack([np.maximum(-Wg,0),np.maximum(Wg,0)],axis=0),axis=1)
    return acc, g, Gdist, pcs, clients

def run_xfl(fl_x, xcfg, X_train, y_train, X_test, y_test, n_features, sens_idx, baseline):
    clients = iid_split(X_train,y_train,fl_x.n_clients)
    g = MLP(n_features).to(DEVICE)
    cs=[MLP(n_features).to(DEVICE) for _ in range(fl_x.n_clients)]
    Pi = normalize_simplex(np.ones((2,n_features),dtype="float32"),axis=1)
    ema=0.6; aligned=[]; t0=time.time()
    for r in range(fl_x.rounds):
        arts=[]
        for i,(Xi,yi) in enumerate(clients):
            cs[i].load_state_dict(g.state_dict())
            opt=torch.optim.Adam(cs[i].parameters(), lr=fl_x.lr)
            dl=to_loader(Xi,yi,fl_x.batch_size,True)
            def extra_loss(m, xb, yb):
                return sensitive_penalty(m, xb, yb, sens_idx, weight=xcfg.sens_penalty, baseline=baseline)
            for _ in range(fl_x.local_epochs):
                train_epoch(cs[i], dl, opt, extra_loss_fn=extra_loss)
            if (r % xcfg.every_R)==0:
                Wc = distill_surrogate(cs[i], Xi, xcfg, n_features, sens_idx)
                beta = xcfg.beta_final*min(1.0,(r+1)/max(1,xcfg.warmup))
                S_i = normalize_simplex(Wc,axis=1)
                S_mix = normalize_simplex((1-beta)*S_i + beta*Pi, axis=1)
                arts.append(artifact_from_surrogate(S_mix, xcfg))
                if r==fl_x.rounds-1: aligned.append(S_mix)
        # FedAvg
        w=None;k=0
        for m in cs:
            sd=m.state_dict()
            if w is None: w={kk:vv.clone() for kk,vv in sd.items()}
            else:
                for kk in w: w[kk]+=sd[kk]
            k+=1
        for kk in w: w[kk]/=k
        g.load_state_dict(w)
        # update Pi
        if arts:
            Pi_new = robust_median(arts)
            Pi = normalize_simplex(ema*Pi + (1-ema)*Pi_new, axis=1)

    acc=evaluate_acc(g,X_test,y_test)
    overhead = fl_x.n_clients*2*xcfg.topk*3
    if not aligned:
        for (Xi,yi) in clients:
            Wc = distill_surrogate(g, Xi, xcfg, n_features, sens_idx)
            beta = xcfg.beta_final
            S_i = normalize_simplex(Wc,axis=1)
            aligned.append(normalize_simplex((1-beta)*S_i + beta*Pi, axis=1))
    avg_time = (time.time()-t0)/max(1,fl_x.rounds)
    return acc, g, Pi, aligned, overhead, avg_time, clients

# ----------------------------
# Main Execution
# ----------------------------
def run_once(seed: int, raw_df: pd.DataFrame):
    set_seed(seed)
    print(f"\n{'='*40}\n Seed {seed}\n{'='*40}")

    X_train, y_train, X_test, y_test, feature_names, sens_idx = preprocess_data(raw_df, seed)
    n_features = X_train.shape[1]
    train_mean = X_train.mean(axis=0).astype(np.float32)
    zero_baseline = np.zeros_like(train_mean, dtype=np.float32)

    fl_base = FLConfig(n_clients=6, rounds=4, local_epochs=1, batch_size=256, lr=1e-3)
    fl_x    = FLConfig(n_clients=6, rounds=8, local_epochs=1, batch_size=256, lr=1e-3)
    xcfg    = XFLConfig(topk=min(48,n_features))

    print("Running BL-A…"); acc_A, model_A, clients_A = run_fedavg(fl_base, X_train, y_train, X_test, y_test, n_features)
    print(f"BL-A acc: {acc_A:.4f}")

    print("Running BL-B…"); acc_B, model_B, pcs_B, clients_B = run_bl_b(fl_base, X_train, y_train, X_test, y_test, n_features, zero_baseline)
    print(f"BL-B acc: {acc_B:.4f}")

    print("Running BL-C…"); acc_C, model_C, ghist_C, clients_C = run_bl_c(fl_base, X_train, y_train, X_test, y_test, n_features, zero_baseline)
    print(f"BL-C acc: {acc_C:.4f}")

    print("Running BL-D…"); acc_D, model_D, Gdist_D, pcs_D, clients_D = run_bl_d(fl_base, X_train, y_train, X_test, y_test, n_features)
    print(f"BL-D acc: {acc_D:.4f}")

    print("Running xFL…");  acc_X, model_X, Pi_X, pcs_X, overhead_X, tX, clients_X = run_xfl(fl_x, xcfg, X_train, y_train, X_test, y_test, n_features, sens_idx, zero_baseline)
    print(f"xFL acc: {acc_X:.4f}")

    # Metrics
    ref_B = normalize_simplex(np.mean(np.stack(pcs_B,axis=0),axis=0),axis=1)
    edi_B = np.mean([jsd(pc[0],ref_B[0])+jsd(pc[1],ref_B[1]) for pc in pcs_B])/2
    edi_C = np.mean([jsd(pc[0],ghist_C[0])+jsd(pc[1],ghist_C[1]) for pc in pcs_B])/2
    edi_D = np.mean([jsd(pc[0],Gdist_D[0])+jsd(pc[1],Gdist_D[1]) for pc in pcs_D])/2
    edi_X = np.mean([jsd(pc[0],Pi_X[0]) +jsd(pc[1],Pi_X[1])  for pc in pcs_X])/2
    print(f"EDI -> BL-B {edi_B:.4f} | BL-C {edi_C:.4f} | BL-D {edi_D:.4f} | xFL {edi_X:.4f}")

    # Fidelity
    B = min(1024,len(X_test)); Xb = X_test[:B].copy(); yb = y_test[:B].copy()
    
    def imp_blb(m): return integrated_gradients(m, Xb, yb, zero_baseline, steps=32)
    def imp_blc(m):
        with torch.no_grad():
            preds=(m(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1)>=0.5).astype(np.int64)
        M=np.zeros((B,n_features),dtype="float32")
        for i in range(B): M[i]=ghist_C[preds[i]]
        return normalize_simplex(M,axis=1)
    def imp_bld(lm):
        W = lm.W.weight.detach().cpu().numpy().reshape(-1)
        w_pos=np.maximum(W,0.0); w_neg=np.maximum(-W,0.0)
        with torch.no_grad():
            preds=(lm(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1)>=0.5).astype(np.int64)
        M=np.zeros((B,n_features),dtype="float32")
        for i in range(B): M[i]=w_pos if preds[i]==1 else w_neg
        return normalize_simplex(M,axis=1)
    def imp_xfl():
        S = integrated_gradients(model_X,Xb,yb,zero_baseline,steps=32)
        with torch.no_grad():
            preds=(model_X(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1)>=0.5).astype(np.int64)
        M=np.zeros((B,n_features),dtype="float32")
        for i in range(B): M[i]=0.7*pcs_X[0][preds[i]] + 0.3*Pi_X[preds[i]]
        if len(sens_idx)>0: M[:,sens_idx]*=0.5
        M=normalize_simplex(M,axis=1)
        return normalize_simplex(xcfg.hybrid_alpha*S + (1-xcfg.hybrid_alpha)*M, axis=1)

    maps_B = imp_blb(model_B)
    maps_C = imp_blc(model_C)
    maps_D = imp_bld(model_D)
    maps_X = imp_xfl()

    del_B, ins_B = del_ins_auc(model_B, Xb, yb, maps_B, 20, train_mean)
    del_C, ins_C = del_ins_auc(model_C, Xb, yb, maps_C, 20, train_mean)
    del_D, ins_D = del_ins_auc(model_D, Xb, yb, maps_D, 20, train_mean)
    del_X, ins_X = del_ins_auc(model_X, Xb, yb, maps_X, 20, train_mean)

    print(f"Del AUC: BL-B {del_B:.3f} | BL-C {del_C:.3f} | BL-D {del_D:.3f} | xFL {del_X:.3f}")
    print(f"Ins AUC: BL-B {ins_B:.3f} | BL-C {ins_C:.3f} | BL-D {ins_D:.3f} | xFL {ins_X:.3f}")

    return {
        "acc": {"BL-A": acc_A, "BL-B": acc_B, "BL-C": acc_C, "BL-D": acc_D, "xFL": acc_X},
        "edi": {"BL-B": edi_B, "BL-C": edi_C, "BL-D": edi_D, "xFL": edi_X},
        "del_auc": {"BL-B": del_B, "BL-C": del_C, "BL-D": del_D, "xFL": del_X},
        "ins_auc": {"BL-B": ins_B, "BL-C": ins_C, "BL-D": ins_D, "xFL": ins_X},
        "feature_names": feature_names,
        "maps": {"BL-B": maps_B, "BL-C": maps_C, "BL-D": maps_D, "xFL": maps_X}
    }

if __name__ == "__main__":
    raw_df = load_raw_data()
    seeds = [BASE_SEED + i for i in range(5)]
    
    metrics = {
        "acc": {m: [] for m in ["BL-A", "BL-B", "BL-C", "BL-D", "xFL"]},
        "edi": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
        "del_auc": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
        "ins_auc": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
    }
    
    last_res = None
    for s in seeds:
        res = run_once(s, raw_df)
        last_res = res
        for m in metrics["acc"]: metrics["acc"][m].append(res["acc"][m])
        for m in metrics["edi"]: metrics["edi"][m].append(res["edi"][m])
        for m in metrics["del_auc"]: metrics["del_auc"][m].append(res["del_auc"][m])
        for m in metrics["ins_auc"]: metrics["ins_auc"][m].append(res["ins_auc"][m])

    def summarize(arr):
        arr = np.array(arr, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    print("\n===== Summary over seeds =====")
    print("Accuracy:")
    for m in metrics["acc"]:
        mean_v, std_v = summarize(metrics["acc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nEDI (lower better):")
    for m in metrics["edi"]:
        mean_v, std_v = summarize(metrics["edi"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nDeletion AUC (lower better):")
    for m in metrics["del_auc"]:
        mean_v, std_v = summarize(metrics["del_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nInsertion AUC (higher better):")
    for m in metrics["ins_auc"]:
        mean_v, std_v = summarize(metrics["ins_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    # Plots
    os.makedirs("adult_outputs", exist_ok=True)
    
    def save_bar(vals, labels, title, fname, ylabel):
        plt.figure(figsize=(5,3)); plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout(); p=os.path.join("adult_outputs", fname); plt.savefig(p, bbox_inches="tight"); plt.close()

    methods=["BL-B","BL-C","BL-D","xFL"]
    mean_del = [summarize(metrics["del_auc"][m])[0] for m in methods]
    mean_ins = [summarize(metrics["ins_auc"][m])[0] for m in methods]
    mean_edi = [summarize(metrics["edi"][m])[0] for m in methods]

    save_bar(mean_del, methods, "Deletion AUC (mean)", "bar_deletion_auc.png", "AUC")
    save_bar(mean_ins, methods, "Insertion AUC (mean)", "bar_insertion_auc.png", "AUC")
    save_bar(mean_edi, methods, "Consistency (EDI, mean)", "bar_edi.png", "EDI")

    # Feature plots from last seed
    def plot_top(sample_idx, imp_vec, title, fname, topn=10):
        idx = np.argsort(-imp_vec)[:topn]
        labels = [last_res["feature_names"][i] for i in idx]; vals = imp_vec[idx]
        plt.figure(figsize=(7,3.5)); plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right"); plt.title(title)
        plt.tight_layout(); p = os.path.join("adult_outputs", fname); plt.savefig(p, bbox_inches="tight"); plt.close()

    plot_top(5, last_res["maps"]["BL-B"][5], "BL-B emphasized features", "blb_features.png")
    plot_top(5, last_res["maps"]["BL-C"][5], "BL-C emphasized features", "blc_features.png")
    plot_top(5, last_res["maps"]["BL-D"][5], "BL-D emphasized features", "bld_features.png")
    plot_top(5, last_res["maps"]["xFL"][5], "xFL (hybrid) emphasized features", "xfl_features.png")

    print("Done. See plots under adult_outputs/")
