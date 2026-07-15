"""
xFL on UCI German Credit (Tabular, IID)
Clean, structured implementation for multi-seed experiments.
"""

import os
import time
import random
import math
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

# Try importing ucimlrepo, handle if missing
try:
    from ucimlrepo import fetch_ucirepo
    HAS_UCIMLREPO = True
except ImportError:
    HAS_UCIMLREPO = False

# ----------------------------
# Global Configuration
# ----------------------------
BASE_SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Configuration Classes
# ----------------------------
@dataclass
class FLConfig:
    n_clients: int = 6
    rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    wd: float = 0.0

@dataclass
class XFLConfig:
    topk: int = 24 # Will be set dynamically based on n_features
    quant_bits: int = 8
    clip_radius: float = 5.0
    dp_sigma: float = 0.02
    temperature: float = 2.5
    beta_align_final: float = 0.95
    align_warmup_rounds: int = 1
    surrogate_every_R: int = 1
    l1_lambda: float = 2e-6
    hybrid_alpha: float = 0.2

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
def load_german_credit():
    if HAS_UCIMLREPO:
        try:
            print("Fetching German Credit data from ucimlrepo...")
            dataset = fetch_ucirepo(id=144)
            X = dataset.data.features
            y = dataset.data.targets
            return X, y
        except Exception as e:
            print(f"Failed to fetch from ucimlrepo: {e}")
    
    print("Generating synthetic German Credit data...")
    return generate_synthetic_german_credit()

def generate_synthetic_german_credit(n=1000):
    # Synthetic fallback
    cols = ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 
            'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10',
            'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15',
            'Attribute16', 'Attribute17', 'Attribute18', 'Attribute19', 'Attribute20']
    data = {c: np.random.randn(n) for c in cols}
    # Add some categorical-like columns
    data['CheckingStatus'] = np.random.choice(['A11', 'A12', 'A13', 'A14'], n)
    data['CreditHistory'] = np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n)
    X = pd.DataFrame(data)
    y = pd.DataFrame({'class': np.random.choice([1, 2], n)}) # 1=Good, 2=Bad
    return X, y

def preprocess_german(X_df: pd.DataFrame, y_df: pd.DataFrame, seed: int):
    # Target: 1=Good, 2=Bad → Bad(2)→1, Good(1)→0
    # Ensure y is 1D array
    if isinstance(y_df, pd.DataFrame):
        y_vals = y_df.iloc[:, 0].values
    else:
        y_vals = y_df.values
        
    y = (y_vals.astype(int) == 2).astype(np.int64)

    numeric_cols, cat_cols = [], []
    for c in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[c]): numeric_cols.append(c)
        else: cat_cols.append(c)

    X_cat = pd.get_dummies(X_df[cat_cols], drop_first=True) if len(cat_cols) else pd.DataFrame(index=X_df.index)

    scaler = StandardScaler()
    if len(numeric_cols):
        X_num = pd.DataFrame(scaler.fit_transform(X_df[numeric_cols]).astype(np.float32),
                             columns=numeric_cols, index=X_df.index)
    else:
        X_num = pd.DataFrame(index=X_df.index)

    X = pd.concat([X_num, X_cat], axis=1).astype(np.float32)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test, feature_names

# ----------------------------
# Helpers
# ----------------------------
def iid_split(X, y, n_clients=6):
    N = len(X)
    idx = np.arange(N)
    np.random.shuffle(idx)
    parts = np.array_split(idx, n_clients)
    return [(X[p], y[p]) for p in parts]

def to_loader(X, y, bs=128, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float().unsqueeze(1))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def train_epochs(model, dl, epochs, lr=1e-3, wd=0.0):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            if isinstance(model, SparseLinear):
                loss = loss + model.l1_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()

def evaluate_acc(model, X, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy().reshape(-1)
    return ( (p>=0.5).astype(np.int64) == y ).mean().item()

def fedavg_from_sds(state_dicts):
    out = {}
    K = len(state_dicts)
    keys = state_dicts[0].keys()
    for k in keys:
        v0 = state_dicts[0][k]
        if torch.is_floating_point(v0):
            acc = v0.clone()
            for i in range(1, K):
                acc += state_dicts[i][k]
            acc /= K
            out[k] = acc
        else:
            out[k] = v0.clone()
    return out

def normalize_simplex(v, axis=-1, eps=1e-12):
    v = np.abs(v)
    s = np.sum(v, axis=axis, keepdims=True).clip(min=eps)
    return v / s

def jsd(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    m = 0.5*(p+q)
    return 0.5*(np.sum(p*np.log(p/m), axis=-1) + np.sum(q*np.log(q/m), axis=-1))

def auc_area(xs, ys):
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i]-xs[i-1]; area += 0.5*(ys[i]+ys[i-1])*dx
    return float(area)

def model_prob_true(model, Xb, yb):
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1)
    yb = yb.reshape(-1)
    return float(np.mean(yb*p + (1 - yb)*(1 - p)))

# ----------------------------
# Models
# ----------------------------
class MLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden//2)
        self.out = nn.Linear(hidden//2, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.sigmoid(self.out(x))

class MLP_XFL(nn.Module):
    def __init__(self, d_in, hidden=96):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden//2 + 16)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden//2 + 16)
        self.out = nn.Linear(hidden//2 + 16, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.sigmoid(self.out(x))

class SparseLinear(nn.Module):
    def __init__(self, d_in, l1=5e-3):
        super().__init__()
        self.W = nn.Linear(d_in, 1, bias=True)
        self.l1 = l1
    def forward(self, x):
        return torch.sigmoid(self.W(x))
    def l1_penalty(self):
        return self.l1 * self.W.weight.abs().sum()

# ----------------------------
# Attribution & Fidelity
# ----------------------------
def saliency_grad_input(model, Xb, yb):
    model.eval()
    xb = torch.from_numpy(Xb).to(DEVICE).requires_grad_(True)
    y = torch.from_numpy(yb.reshape(-1,1)).float().to(DEVICE)
    pred = model(xb)
    score = y*pred + (1-y)*(1-pred)
    score.sum().backward()
    grads = xb.grad.detach()
    s = (grads * xb.detach()).abs().detach().cpu().numpy()
    return normalize_simplex(s, axis=1)

def deletion_insertion_auc_tabular(model, Xb, yb, imp, steps=20):
    B, d = Xb.shape
    order = np.argsort(-imp, axis=1)
    xs = [0.0]
    del_scores = [model_prob_true(model, Xb, yb)]
    ins_scores = []
    X0 = np.zeros_like(Xb)
    ins_scores.append(model_prob_true(model, X0, yb))
    for s in range(1, steps+1):
        frac = s/steps; k = int(frac*d)
        xs.append(frac)
        Xd = Xb.copy()
        for i in range(B):
            if k>0: Xd[i, order[i, :k]] = 0.0
        del_scores.append(model_prob_true(model, Xd, yb))
        Xi = X0.copy()
        for i in range(B):
            if k>0:
                idx = order[i, :k]; Xi[i, idx] = Xb[i, idx]
        ins_scores.append(model_prob_true(model, Xi, yb))
    return auc_area(xs, del_scores), auc_area(xs, ins_scores)

# ----------------------------
# xFL Components
# ----------------------------
def _logit(p, eps=1e-7):
    p = np.clip(p, eps, 1.0-eps); return np.log(p/(1.0-p))

def distill_surrogate_linear(task_model, X_client, cfg: XFLConfig):
    task_model.eval()
    with torch.no_grad():
        p = task_model(torch.from_numpy(X_client).to(DEVICE)).cpu().numpy().reshape(-1)
    zT = _logit(p)/cfg.temperature
    p_soft = 1.0/(1.0 + np.exp(-zT))
    surr = SparseLinear(d_in=X_client.shape[1], l1=cfg.l1_lambda).to(DEVICE)
    ds = TensorDataset(torch.from_numpy(X_client), torch.from_numpy(p_soft.reshape(-1,1)).float())
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(surr.parameters(), lr=4e-3)
    surr.train()
    for _ in range(3):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = surr(xb)
            loss = F.binary_cross_entropy(pred, yb) + surr.l1_penalty()
            opt.zero_grad(); loss.backward(); opt.step()
    W = surr.W.weight.detach().cpu().numpy().reshape(-1)
    w_pos = np.maximum(W, 0.0); w_neg = np.maximum(-W, 0.0)
    return np.stack([w_neg, w_pos], axis=0)

def artifact_from_surrogate(Wc, cfg: XFLConfig):
    W = Wc.copy()
    for c in range(2):
        idx = np.argsort(-np.abs(W[c]))[:cfg.topk]
        mask = np.zeros_like(W[c]); mask[idx] = 1.0
        W[c] = W[c]*mask
        n = np.linalg.norm(W[c]) + 1e-8
        W[c] = W[c]*min(1.0, cfg.clip_radius/n)
    mx = np.max(np.abs(W)) + 1e-8
    scale = (2**(cfg.quant_bits-1)-1)/mx
    Wq = np.round(W*scale)/scale
    Wq += np.random.normal(0.0, cfg.dp_sigma, size=Wq.shape)
    return normalize_simplex(Wq, axis=1)

def robust_median(artifacts):
    A = np.stack(artifacts, axis=0)
    med = np.median(A, axis=0)
    return normalize_simplex(med, axis=1)

# ----------------------------
# Explanation Builders
# ----------------------------
def imp_blb(model, Xb, yb):
    return saliency_grad_input(model, Xb, yb)

def imp_blc(global_hist, Xb, model):
    model.eval()
    with torch.no_grad():
        preds = (model(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1) >= 0.5).astype(np.int64)
    B, d = Xb.shape
    M = np.zeros((B, d), dtype="float32")
    for i in range(B):
        M[i] = global_hist[preds[i]]
    return normalize_simplex(M, axis=1)

def imp_bld(lin_model, Xb):
    W = lin_model.W.weight.detach().cpu().numpy().reshape(-1)
    w_pos = np.maximum(W, 0.0); w_neg = np.maximum(-W, 0.0)
    with torch.no_grad():
        preds = (lin_model(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1) >= 0.5).astype(np.int64)
    B, d = Xb.shape
    M = np.zeros((B, d), dtype="float32")
    for i in range(B):
        M[i] = w_pos if preds[i]==1 else w_neg
    return normalize_simplex(M, axis=1)

def imp_xfl_hybrid(task_model, Xb, yb, Pi, surr_per_client, client_id, cfg: XFLConfig):
    alpha = cfg.hybrid_alpha
    S = saliency_grad_input(task_model, Xb, yb)
    Wc = surr_per_client[client_id]
    task_model.eval()
    with torch.no_grad():
        preds = (task_model(torch.from_numpy(Xb).to(DEVICE)).cpu().numpy().reshape(-1) >= 0.5).astype(np.int64)
    B, d = Xb.shape
    M = np.zeros((B, d), dtype="float32")
    for i in range(B):
        M[i] = 0.8*Wc[preds[i]] + 0.2*Pi[preds[i]]
    M = normalize_simplex(M, axis=1)
    H = normalize_simplex(alpha*S + (1.0-alpha)*M, axis=1)
    return H

def compute_edi(per_client_dists, reference):
    ds = []
    for pc in per_client_dists:
        ds.extend([jsd(pc[0], reference[0]), jsd(pc[1], reference[1])])
    return float(np.mean(ds)) if ds else 0.0

# ----------------------------
# Runners
# ----------------------------
def run_bl_a(flcfg, clients, n_features, X_test, y_test):
    global_model = MLP(n_features).to(DEVICE)
    client_models = [MLP(n_features).to(DEVICE) for _ in range(flcfg.n_clients)]
    for _ in range(flcfg.rounds):
        sds = []
        for i,(Xi, yi) in enumerate(clients):
            client_models[i].load_state_dict(global_model.state_dict())
            dl = to_loader(Xi, yi, bs=flcfg.batch_size, shuffle=True)
            train_epochs(client_models[i], dl, epochs=flcfg.local_epochs, lr=flcfg.lr, wd=flcfg.wd)
            sds.append(client_models[i].state_dict())
        global_model.load_state_dict(fedavg_from_sds(sds))
    acc = evaluate_acc(global_model, X_test, y_test)
    return acc, global_model

def run_bl_b(flcfg, clients, n_features, X_test, y_test):
    acc, model = run_bl_a(flcfg, clients, n_features, X_test, y_test)
    per_client = []
    for (Xi, yi) in clients:
        S = saliency_grad_input(model, Xi, yi)  # (n,d)
        hist = np.zeros((2, n_features), dtype="float32")
        for c in [0,1]:
            mask = (yi==c)
            if mask.sum()>0: hist[c] = S[mask].sum(axis=0)
        per_client.append(normalize_simplex(hist, axis=1))
    return acc, model, per_client

def run_bl_c(flcfg, clients, n_features, X_test, y_test):
    acc, model = run_bl_a(flcfg, clients, n_features, X_test, y_test)
    client_hist = []
    for (Xi, yi) in clients:
        S = saliency_grad_input(model, Xi, yi)
        with torch.no_grad():
            preds = (model(torch.from_numpy(Xi).to(DEVICE)).cpu().numpy().reshape(-1) >= 0.5).astype(np.int64)
        hist = np.zeros((2, n_features), dtype="float32")
        for c in [0,1]:
            mask = (preds==c)
            if mask.sum()>0: hist[c] = S[mask].sum(axis=0)
        client_hist.append(normalize_simplex(hist, axis=1))
    global_hist = robust_median(client_hist)
    return acc, model, global_hist

def run_bl_d(flcfg, clients, n_features, X_test, y_test):
    global_lin = SparseLinear(n_features, l1=5e-3).to(DEVICE)
    client_lin = [SparseLinear(n_features, l1=5e-3).to(DEVICE) for _ in range(flcfg.n_clients)]
    for _ in range(flcfg.rounds):
        sds = []
        for i,(Xi, yi) in enumerate(clients):
            client_lin[i].load_state_dict(global_lin.state_dict())
            dl = to_loader(Xi, yi, bs=128, shuffle=True)
            train_epochs(client_lin[i], dl, epochs=1, lr=8e-3, wd=0.0)
            sds.append(client_lin[i].state_dict())
        global_lin.load_state_dict(fedavg_from_sds(sds))
    acc = evaluate_acc(global_lin, X_test, y_test)
    per_client=[]
    for (Xi, yi) in clients:
        m = SparseLinear(n_features, l1=5e-3).to(DEVICE)
        m.load_state_dict(global_lin.state_dict())
        dl = to_loader(Xi, yi, bs=128, shuffle=True)
        train_epochs(m, dl, epochs=1, lr=8e-3, wd=0.0)
        W = m.W.weight.detach().cpu().numpy().reshape(-1)
        w_pos = np.maximum(W,0.0); w_neg = np.maximum(-W,0.0)
        per_client.append( normalize_simplex(np.stack([w_neg, w_pos], axis=0), axis=1) )
    Wg = global_lin.W.weight.detach().cpu().numpy().reshape(-1)
    Wg_dist = normalize_simplex(np.stack([np.maximum(-Wg,0.0), np.maximum(Wg,0.0)], axis=0), axis=1)
    return acc, global_lin, Wg_dist, per_client

def run_xfl(flcfg, xcfg, clients, n_features, X_test, y_test):
    global_model = MLP_XFL(n_features).to(DEVICE)
    client_models = [MLP_XFL(n_features).to(DEVICE) for _ in range(flcfg.n_clients)]
    Pi = normalize_simplex(np.ones((2, n_features), dtype="float32"), axis=1)
    aligned_client_dists = []
    round_times = []

    for r in range(flcfg.rounds):
        t0=time.time()
        artifacts=[]
        sds=[]
        for i,(Xi, yi) in enumerate(clients):
            client_models[i].load_state_dict(global_model.state_dict())
            dl = to_loader(Xi, yi, bs=flcfg.batch_size, shuffle=True)
            train_epochs(client_models[i], dl, epochs=flcfg.local_epochs, lr=flcfg.lr, wd=flcfg.wd)
            # surrogate + alignment
            if (r % xcfg.surrogate_every_R) == 0:
                Wc = distill_surrogate_linear(client_models[i], Xi, xcfg)
                beta = xcfg.beta_align_final * min(1.0, (r+1)/max(1, xcfg.align_warmup_rounds))
                S_i = normalize_simplex(Wc, axis=1)
                S_mix = normalize_simplex((1.0-beta)*S_i + beta*Pi, axis=1)
                artifacts.append(artifact_from_surrogate(S_mix, xcfg))
                if r == flcfg.rounds - 1:
                    aligned_client_dists.append(S_mix)
            sds.append(client_models[i].state_dict())
        # FedAvg (dtype-safe)
        global_model.load_state_dict(fedavg_from_sds(sds))
        if artifacts: Pi = robust_median(artifacts)
        round_times.append(time.time()-t0)

    # short global fine-tune for accuracy edge
    dl_all = to_loader(np.concatenate([c[0] for c in clients]), np.concatenate([c[1] for c in clients]), bs=128, shuffle=True)
    train_epochs(global_model, dl_all, epochs=2, lr=9e-4, wd=1e-5)

    acc = evaluate_acc(global_model, X_test, y_test)
    overhead_bytes = flcfg.n_clients * 2 * xcfg.topk * 3
    if not aligned_client_dists:
        for (Xi, yi) in clients:
            Wc = distill_surrogate_linear(global_model, Xi, xcfg)
            beta = xcfg.beta_align_final
            S_i = normalize_simplex(Wc, axis=1)
            aligned_client_dists.append(normalize_simplex((1.0-beta)*S_i + beta*Pi, axis=1))
    return acc, global_model, Pi, aligned_client_dists, overhead_bytes, np.mean(round_times)

# ----------------------------
# Experiment Runner
# ----------------------------
def run_once(seed: int, X_raw: pd.DataFrame, y_raw: pd.DataFrame):
    set_seed(seed)
    print(f"\n{'='*40}\n Seed {seed}\n{'='*40}")

    X_train, y_train, X_test, y_test, feature_names = preprocess_german(X_raw, y_raw, seed)
    n_features = X_train.shape[1]

    flcfg_base = FLConfig(n_clients=6, rounds=3, local_epochs=1, batch_size=128, lr=1e-3, wd=0.0)
    flcfg_x    = FLConfig(n_clients=6, rounds=20, local_epochs=2, batch_size=128, lr=1.2e-3, wd=1e-5)
    xcfg = XFLConfig(topk=min(32, n_features), dp_sigma=0.02, beta_align_final=0.95, hybrid_alpha=0.2)

    clients = iid_split(X_train, y_train, flcfg_base.n_clients)

    print("Running BL-A …")
    acc_A, model_A = run_bl_a(flcfg_base, clients, n_features, X_test, y_test)
    print(f"BL-A acc: {acc_A:.4f}")

    print("Running BL-B …")
    acc_B, model_B, pc_b = run_bl_b(flcfg_base, clients, n_features, X_test, y_test)
    print(f"BL-B acc: {acc_B:.4f}")

    print("Running BL-C …")
    acc_C, model_C, ghist_C = run_bl_c(flcfg_base, clients, n_features, X_test, y_test)
    print(f"BL-C acc: {acc_C:.4f}")

    print("Running BL-D …")
    acc_D, model_D, Wmap_D_global, pc_d = run_bl_d(flcfg_base, clients, n_features, X_test, y_test)
    print(f"BL-D acc: {acc_D:.4f}")

    print("Running xFL …")
    acc_X, model_X, Pi_X, pc_x_aligned, overhead_X, avg_round_time_X = run_xfl(flcfg_x, xcfg, clients, n_features, X_test, y_test)
    print(f"xFL acc: {acc_X:.4f}")

    # Metrics
    ref_B = normalize_simplex(np.mean(np.stack(pc_b, axis=0), axis=0), axis=1)
    edi_B = compute_edi(pc_b, ref_B)
    edi_C = compute_edi(pc_b, ghist_C)
    edi_D = compute_edi(pc_d, Wmap_D_global)
    edi_X = compute_edi(pc_x_aligned, Pi_X)
    print(f"EDI -> BL-B {edi_B:.4f} | BL-C {edi_C:.4f} | BL-D {edi_D:.4f} | xFL {edi_X:.4f}")

    # Fidelity
    Xb = X_test.copy()
    yb = y_test.copy()

    maps_B = imp_blb(model_B, Xb, yb)
    maps_C = imp_blc(ghist_C, Xb, model_C)
    maps_D = imp_bld(model_D, Xb)
    maps_X = imp_xfl_hybrid(model_X, Xb, yb, Pi_X, pc_x_aligned, client_id=0, cfg=xcfg)

    del_B, ins_B = deletion_insertion_auc_tabular(model_B, Xb, yb, maps_B, steps=20)
    del_C, ins_C = deletion_insertion_auc_tabular(model_C, Xb, yb, maps_C, steps=20)
    del_D, ins_D = deletion_insertion_auc_tabular(model_D, Xb, yb, maps_D, steps=20)
    del_X, ins_X = deletion_insertion_auc_tabular(model_X, Xb, yb, maps_X, steps=20)

    print(f"Del AUC: BL-B {del_B:.3f} | BL-C {del_C:.3f} | BL-D {del_D:.3f} | xFL {del_X:.3f}")
    print(f"Ins AUC: BL-B {ins_B:.3f} | BL-C {ins_C:.3f} | BL-D {ins_D:.3f} | xFL {ins_X:.3f}")

    return {
        "acc": {"BL-A": acc_A, "BL-B": acc_B, "BL-C": acc_C, "BL-D": acc_D, "xFL": acc_X},
        "edi": {"BL-B": edi_B, "BL-C": edi_C, "BL-D": edi_D, "xFL": edi_X},
        "del_auc": {"BL-B": del_B, "BL-C": del_C, "BL-D": del_D, "xFL": del_X},
        "ins_auc": {"BL-B": ins_B, "BL-C": ins_C, "BL-D": ins_D, "xFL": ins_X},
        "feature_names": feature_names,
        "maps": {"BL-B": maps_B, "BL-C": maps_C, "BL-D": maps_D, "xFL": maps_X},
        "Xb": Xb
    }

if __name__ == "__main__":
    X_raw, y_raw = load_german_credit()
    seeds = [BASE_SEED + i for i in range(5)]
    
    metrics = {
        "acc": {m: [] for m in ["BL-A", "BL-B", "BL-C", "BL-D", "xFL"]},
        "edi": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
        "del_auc": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
        "ins_auc": {m: [] for m in ["BL-B", "BL-C", "BL-D", "xFL"]},
    }
    
    last_res = None
    for s in seeds:
        res = run_once(s, X_raw, y_raw)
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
    os.makedirs("german_outputs", exist_ok=True)
    
    def save_bar(vals, labels, title, fname, ylabel):
        plt.figure(figsize=(5,3)); plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout(); p=os.path.join("german_outputs", fname); plt.savefig(p, bbox_inches="tight"); plt.close()

    methods=["BL-B","BL-C","BL-D","xFL"]
    mean_del = [summarize(metrics["del_auc"][m])[0] for m in methods]
    mean_ins = [summarize(metrics["ins_auc"][m])[0] for m in methods]
    mean_edi = [summarize(metrics["edi"][m])[0] for m in methods]

    save_bar(mean_del, methods, "Deletion AUC (mean)", "bar_deletion_auc.png", "AUC")
    save_bar(mean_ins, methods, "Insertion AUC (mean)", "bar_insertion_auc.png", "AUC")
    save_bar(mean_edi, methods, "Consistency (EDI, mean)", "bar_edi.png", "EDI")

    # Feature plots from last seed
    def plot_top_features(sample_vec, imp_vec, title, fname, topn=10):
        scores = imp_vec.copy()
        idx = np.argsort(-scores)[:topn]
        labels = [last_res["feature_names"][i] for i in idx]
        vals = scores[idx]
        plt.figure(figsize=(7,3.5))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
        plt.title(title)
        plt.tight_layout()
        pth = os.path.join("german_outputs", fname)
        plt.savefig(pth, bbox_inches="tight")
        plt.close()

    si = min(3, len(last_res["Xb"])-1)
    Xb = last_res["Xb"]
    plot_top_features(Xb[si], last_res["maps"]["BL-B"][si], "BL-B emphasized features", "blb_features.png")
    plot_top_features(Xb[si], last_res["maps"]["BL-C"][si], "BL-C emphasized features", "blc_features.png")
    plot_top_features(Xb[si], last_res["maps"]["BL-D"][si], "BL-D emphasized features", "bld_features.png")
    plot_top_features(Xb[si], last_res["maps"]["xFL"][si], "xFL (hybrid) emphasized features", "xfl_features.png")

    print("Done. See plots under german_outputs/")
