"""
xFL on CIFAR-10 — Baselines + Metrics + Proposed Method
Clean, structured implementation for multi-seed experiments.
"""

import math
import random
import time
import os
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

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
    n_clients: int = 10
    rounds: int = 20
    local_epochs: int = 1
    lr_cnn: float = 0.01
    lr_lin: float = 0.05
    batch_size: int = 64
    alpha_dirichlet: float = 0.1

@dataclass
class XFLConfig:
    topk: int = 512
    quant_bits: int = 8
    clip_radius: float = 10.0
    dp_sigma: float = 0.2
    temperature: float = 5.0
    beta_align_final: float = 0.3
    align_warmup_rounds: int = 8
    surrogate_every_R: int = 2
    l2_lambda: float = 1e-3
    surrogate_steps: int = 3
    hidden_dim: int = 64

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
def dirichlet_split_noniid(labels: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    n_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        np.random.shuffle(idx_by_class[c])
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = idx_by_class[c]
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = (proportions / proportions.sum())
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        parts = np.split(idx, splits)
        for i in range(n_clients):
            client_indices[i].extend(parts[i])
    return [np.array(ci, dtype=np.int64) for ci in client_indices]

def make_loaders_cifar10(n_clients: int, alpha: float, batch_size: int):
    tfm_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)

    labels = np.array(train_set.targets)
    splits = dirichlet_split_noniid(labels, n_clients, alpha)

    client_loaders = []
    for idx in splits:
        subset = Subset(train_set, idx.tolist())
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))

    per_class = 10
    ref_idx = []
    class_counts = {c: 0 for c in range(10)}
    for i, y in enumerate(train_set.targets):
        y = int(y)
        if class_counts[y] < per_class:
            ref_idx.append(i)
            class_counts[y] += 1
        if all(class_counts[c] >= per_class for c in range(10)):
            break
    ref_loader = DataLoader(Subset(train_set, ref_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return client_loaders, ref_loader, test_loader

# ----------------------------
# Tensor Utilities
# ----------------------------
def flatten_img(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)

def normalize_to_simplex(vec: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    vec = vec.abs()
    s = vec.sum(dim=dim, keepdim=True).clamp_min(eps)
    return vec / s

def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

# ----------------------------
# Models
# ----------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))
        logits = self.fc2(features)
        if return_features:
            return logits, features
        return logits

class MLPSurrogate(nn.Module):
    def __init__(self, in_features=128, hidden_dim=64, n_classes=10, l2_lambda=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def l2_penalty(self):
        return self.l2_lambda * (self.fc1.weight.pow(2).sum() + self.fc2.weight.pow(2).sum())

class SparseLinearInterpretable(nn.Module):
    def __init__(self, in_features=32*32*3, n_classes=10, l1_lambda=5e-4):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes, bias=True)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        return self.W(x)

    def l1_penalty(self):
        return self.l1_lambda * self.W.weight.abs().sum()

# ----------------------------
# Attribution Methods
# ----------------------------
def integrated_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 20) -> torch.Tensor:
    baseline = torch.zeros_like(x)
    grads = []
    # Use list comprehension for scaled inputs to save memory if needed, but loop is fine
    for i in range(1, steps + 1):
        xi = baseline + (float(i) / steps) * (x - baseline)
        xi.requires_grad_(True)
        logits = model(xi)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grads.append(xi.grad.detach().clone())
        model.zero_grad(set_to_none=True)
    avg_grads = torch.stack(grads, dim=0).mean(dim=0)
    ig = (x - baseline) * avg_grads
    return ig.abs()

# ----------------------------
# Metrics (AUC)
# ----------------------------
def auc_area(xs: List[float], ys: List[float]) -> float:
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        area += 0.5 * (ys[i] + ys[i - 1]) * dx
    return float(area)

def _forward_model(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SparseLinearInterpretable):
        xb = flatten_img(xb)
    return model(xb)

def deletion_insertion_auc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, imp_map: torch.Tensor, steps: int = 20) -> Tuple[float, float]:
    N = x.size(0)
    flat_imp = imp_map.view(N, -1)
    idx_sorted = torch.argsort(flat_imp, dim=-1, descending=True)

    del_scores, ins_scores, xs = [], [], []

    for s in range(steps + 1):
        frac = s / steps
        xs.append(frac)
        k = int(flat_imp.size(1) * frac)

        # Deletion
        x_del = x.clone()
        if k > 0:
            mask = torch.ones_like(flat_imp, device=x.device)
            b_idx = torch.arange(N, device=x.device).unsqueeze(-1)
            mask[b_idx, idx_sorted[:, :k]] = 0.0
            x_del = x_del.view(N, -1) * mask
            x_del = x_del.view_as(x)
        with torch.no_grad():
            prob_del = F.softmax(_forward_model(model, x_del), dim=-1)[torch.arange(N), y]
        del_scores.append(prob_del.mean().item())

        # Insertion
        baseline = torch.zeros_like(x)
        x_ins = baseline.clone()
        if k > 0:
            mask = torch.zeros_like(flat_imp, device=x.device)
            b_idx = torch.arange(N, device=x.device).unsqueeze(-1)
            mask[b_idx, idx_sorted[:, :k]] = 1.0
            x_ins = x_ins.view(N, -1)
            x_ins[mask.bool()] = x.view(N, -1)[mask.bool()]
            x_ins = x_ins.view_as(x)
        with torch.no_grad():
            prob_ins = F.softmax(_forward_model(model, x_ins), dim=-1)[torch.arange(N), y]
        ins_scores.append(prob_ins.mean().item())

    del_auc = auc_area(xs, del_scores)
    ins_auc = auc_area(xs, ins_scores)
    return del_auc, ins_auc

# ----------------------------
# FL Training Helpers
# ----------------------------
def local_train(model: nn.Module, loader: DataLoader, epochs: int = 1, lr: float = 0.01):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def local_train_interpretable(model: SparseLinearInterpretable, loader: DataLoader, epochs=1, lr=0.05):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = flatten_img(xb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + model.l1_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total)

def state_dict_avg(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    keys = models[0].state_dict().keys()
    out = {k: torch.zeros_like(models[0].state_dict()[k]) for k in keys}
    for m in models:
        sd = m.state_dict()
        for k in keys:
            out[k] += sd[k]
    for k in keys:
        out[k] /= len(models)
    return out

# ----------------------------
# Baseline Runners
# ----------------------------
def run_plain_fl(cfg: FLConfig):
    client_loaders, ref_loader, test_loader = make_loaders_cifar10(cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size)
    global_model = SmallCNN().to(DEVICE)
    client_models = [SmallCNN().to(DEVICE) for _ in range(cfg.n_clients)]
    
    for _ in trange(cfg.rounds, desc="BL-A Rounds", leave=False):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            local_train(client_models[i], client_loaders[i], epochs=cfg.local_epochs, lr=cfg.lr_cnn)
        avg_state = state_dict_avg(client_models)
        global_model.load_state_dict(avg_state)
        
    acc = eval_accuracy(global_model, test_loader)
    return acc, global_model, (client_loaders, ref_loader, test_loader)

def compute_local_ig_maps(model: nn.Module, loader: DataLoader, max_per_class: int = 50) -> Dict[int, torch.Tensor]:
    model.eval()
    per_class_imgs = {c: [] for c in range(10)}
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        for i in range(xb.size(0)):
            c = int(yb[i].item())
            if len(per_class_imgs[c]) < max_per_class:
                per_class_imgs[c].append(xb[i:i+1].detach())
        if all(len(per_class_imgs[c]) >= max_per_class for c in range(10)):
            break
    maps = {}
    for c in range(10):
        if len(per_class_imgs[c]) == 0:
            maps[c] = torch.ones(1, 3, 32, 32, device=DEVICE) / (32 * 32 * 3)
            continue
        Xc = torch.cat(per_class_imgs[c], dim=0)
        yc = torch.full((Xc.size(0),), c, dtype=torch.long, device=DEVICE)
        ig = integrated_gradients(model, Xc, yc, steps=10)
        avg_map = ig.mean(dim=0, keepdim=True)
        maps[c] = normalize_to_simplex(avg_map.view(1, -1), dim=-1).view(1, 3, 32, 32)
    return maps

def run_local_posthoc(cfg: FLConfig):
    acc, global_model, loaders = run_plain_fl(cfg)
    client_loaders, _, _ = loaders
    per_client_maps = []
    for i in tqdm(range(cfg.n_clients), desc="BL-B Clients", leave=False):
        local_m = SmallCNN().to(DEVICE)
        local_m.load_state_dict(global_model.state_dict())
        local_train(local_m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        maps = compute_local_ig_maps(local_m, client_loaders[i], max_per_class=30)
        per_client_maps.append(maps)
    return acc, global_model, per_client_maps, loaders

def pool8x8(map_32: torch.Tensor) -> torch.Tensor:
    pooled = F.avg_pool2d(map_32, kernel_size=4, stride=4)
    return pooled.view(-1)

def aggregate_histograms(list_of_hist_per_client: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    out = {}
    for c in range(10):
        mats = [d[c].unsqueeze(0) for d in list_of_hist_per_client]
        M = torch.cat(mats, dim=0)
        out[c] = normalize_to_simplex(M.median(dim=0).values, dim=0)
    return out

def run_server_summary(cfg: FLConfig):
    acc, global_model, loaders = run_plain_fl(cfg)
    client_loaders, _, _ = loaders
    client_hist = []
    for i in tqdm(range(cfg.n_clients), desc="BL-C Clients", leave=False):
        local_m = SmallCNN().to(DEVICE)
        local_m.load_state_dict(global_model.state_dict())
        local_train(local_m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        maps = compute_local_ig_maps(local_m, client_loaders[i], max_per_class=30)
        hist = {}
        for c in range(10):
            pooled = pool8x8(maps[c])
            hist[c] = normalize_to_simplex(pooled, dim=0)
        client_hist.append(hist)
    global_hist = aggregate_histograms(client_hist)
    return acc, global_model, global_hist, loaders

def run_interpretable_only(cfg: FLConfig):
    client_loaders, ref_loader, test_loader = make_loaders_cifar10(cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size)
    global_model = SparseLinearInterpretable().to(DEVICE)
    client_models = [SparseLinearInterpretable().to(DEVICE) for _ in range(cfg.n_clients)]
    
    for _ in range(cfg.rounds):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            local_train_interpretable(client_models[i], client_loaders[i], epochs=cfg.local_epochs, lr=cfg.lr_lin)
        avg_state = state_dict_avg(client_models)
        global_model.load_state_dict(avg_state)

    def eval_lin(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(flatten_img(xb))
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / max(1, total)

    acc = eval_lin(global_model, test_loader)
    Wg = global_model.W.weight.detach().clone()
    Wmaps_global = {c: normalize_to_simplex(Wg[c].abs().view(1, -1), dim=-1).view(1,3,32,32) for c in range(10)}

    per_client_maps = []
    for i in range(cfg.n_clients):
        m = SparseLinearInterpretable().to(DEVICE)
        m.load_state_dict(global_model.state_dict())
        local_train_interpretable(m, client_loaders[i], epochs=1, lr=cfg.lr_lin)
        Wi = m.W.weight.detach().clone()
        maps_i = {c: normalize_to_simplex(Wi[c].abs().view(1,-1), dim=-1).view(1,3,32,32) for c in range(10)}
        per_client_maps.append(maps_i)

    return acc, global_model, Wmaps_global, per_client_maps, (client_loaders, ref_loader, test_loader)

# ----------------------------
# xFL Components
# ----------------------------
def extract_features_and_labels(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_features, all_soft_labels, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, features = model(xb, return_features=True)
            probs = F.softmax(logits, dim=-1)
            all_features.append(features.cpu())
            all_soft_labels.append(probs.cpu())
            all_labels.append(yb.cpu())
    return torch.cat(all_features), torch.cat(all_soft_labels), torch.cat(all_labels)

def fit_surrogate_on_features(model: nn.Module, loader: DataLoader, cfg: XFLConfig) -> MLPSurrogate:
    surr = MLPSurrogate(in_features=128, hidden_dim=cfg.hidden_dim, n_classes=10, l2_lambda=cfg.l2_lambda).to(DEVICE)
    opt = torch.optim.Adam(surr.parameters(), lr=0.01)
    T = cfg.temperature
    
    features, soft_labels, true_labels = extract_features_and_labels(model, loader)
    dataset = TensorDataset(features, soft_labels, true_labels)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for _ in range(cfg.surrogate_steps):
        for feats, soft, labels in train_loader:
            feats, soft, labels = feats.to(DEVICE), soft.to(DEVICE), labels.to(DEVICE)
            logits_s = surr(feats) / T
            loss_kl = F.kl_div(F.log_softmax(logits_s, dim=-1), soft, reduction='batchmean')
            logits_hard = surr(feats)
            loss_ce = F.cross_entropy(logits_hard, labels)
            loss = loss_kl + 0.5 * loss_ce + surr.l2_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return surr

def surrogate_to_artifact(surr: MLPSurrogate, cfg: XFLConfig) -> torch.Tensor:
    W1 = surr.fc1.weight.detach()
    W2 = surr.fc2.weight.detach()
    W_eff = torch.matmul(W2, W1)
    Wabs = W_eff.abs()
    
    mask = torch.zeros_like(Wabs)
    for c in range(10):
        k = min(cfg.topk, Wabs.size(1))
        idx = torch.topk(Wabs[c], k).indices
        mask[c, idx] = 1.0
    Wk = W_eff * mask
    
    for c in range(10):
        v = Wk[c]
        n = v.norm(2).clamp_min(1e-8)
        v = v * (cfg.clip_radius / n).clamp(max=1.0)
        Wk[c] = v
        
    scale = 127.0 / (Wk.abs().max().clamp_min(1e-8))
    Wq = torch.round(Wk * scale) / scale
    noise = torch.randn_like(Wq) * cfg.dp_sigma
    Wdp = Wq + noise
    return Wdp

def normalize_per_class(W: torch.Tensor) -> torch.Tensor:
    out = []
    for c in range(10):
        v = W[c].abs()
        v = v / v.sum().clamp_min(1e-12)
        out.append(v)
    return torch.stack(out, dim=0)

def compute_surrogate_attribution_maps(surr: MLPSurrogate, model: nn.Module, loader: DataLoader, max_per_class: int = 50) -> Dict[int, torch.Tensor]:
    surr.eval()
    model.eval()
    per_class_imgs = {c: [] for c in range(10)}
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        for i in range(xb.size(0)):
            c = int(yb[i].item())
            if len(per_class_imgs[c]) < max_per_class:
                per_class_imgs[c].append(xb[i:i+1].detach())
        if all(len(per_class_imgs[c]) >= max_per_class for c in range(10)):
            break
            
    maps = {}
    for c in range(10):
        if len(per_class_imgs[c]) == 0:
            maps[c] = torch.ones(1, 3, 32, 32, device=DEVICE) / (32 * 32 * 3)
            continue
        Xc = torch.cat(per_class_imgs[c], dim=0)
        yc = torch.full((Xc.size(0),), c, dtype=torch.long, device=DEVICE)
        
        attr_maps = []
        for xi, yi in zip(Xc, yc):
            xi = xi.unsqueeze(0)
            yi = yi.unsqueeze(0)
            xi.requires_grad_(True)
            _, features = model(xi, return_features=True)
            features.requires_grad_(True)
            logits = surr(features)
            loss = F.cross_entropy(logits, yi)
            loss.backward()
            input_grad = xi.grad.detach().abs()
            attr_maps.append(input_grad)
            
        avg_map = torch.cat(attr_maps, dim=0).mean(dim=0, keepdim=True)
        maps[c] = normalize_to_simplex(avg_map.view(1, -1), dim=-1).view(1, 3, 32, 32)
    return maps

def run_xfl(flcfg: FLConfig, xcfg: XFLConfig):
    client_loaders, ref_loader, test_loader = make_loaders_cifar10(flcfg.n_clients, flcfg.alpha_dirichlet, flcfg.batch_size)
    global_model = SmallCNN().to(DEVICE)
    client_models = [SmallCNN().to(DEVICE) for _ in range(flcfg.n_clients)]
    
    Pi = torch.zeros(10, 128).to(DEVICE)
    final_client_surrogates = []

    for r in trange(flcfg.rounds, desc="xFL Rounds", leave=False):
        beta = xcfg.beta_align_final * min(1.0, (r + 1) / max(1, xcfg.align_warmup_rounds))
        local_artifacts = []
        current_surrogates = []
        
        for i in range(flcfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            local_train(client_models[i], client_loaders[i], epochs=flcfg.local_epochs, lr=flcfg.lr_cnn)
            
            if (r + 1) % xcfg.surrogate_every_R == 0 or (r + 1) == flcfg.rounds:
                surr = fit_surrogate_on_features(client_models[i], client_loaders[i], xcfg)
                current_surrogates.append((surr, client_models[i]))
                
                W_i = surrogate_to_artifact(surr, xcfg)
                S_i = normalize_per_class(W_i)
                
                if Pi.abs().sum() == 0:
                    S_mix = S_i
                else:
                    S_mix = (1.0 - beta) * S_i + beta * Pi
                    S_mix = normalize_per_class(S_mix)
                local_artifacts.append(S_mix)
        
        avg_state = state_dict_avg(client_models)
        global_model.load_state_dict(avg_state)
        
        if local_artifacts:
            stacked = torch.stack(local_artifacts, dim=0)
            med = stacked.median(dim=0).values
            Pi = normalize_per_class(med)
            
        if (r + 1) == flcfg.rounds:
            final_client_surrogates = current_surrogates

    if not final_client_surrogates:
         for i in range(flcfg.n_clients):
             surr = fit_surrogate_on_features(client_models[i], client_loaders[i], xcfg)
             final_client_surrogates.append((surr, client_models[i]))

    acc = eval_accuracy(global_model, test_loader)
    
    per_client_surr_maps = []
    for i in tqdm(range(flcfg.n_clients), desc="xFL Maps", leave=False):
        surr, cnn = final_client_surrogates[i]
        maps = compute_surrogate_attribution_maps(surr, cnn, client_loaders[i], max_per_class=30)
        per_client_surr_maps.append(maps)
        
    return acc, global_model, Pi, per_client_surr_maps, final_client_surrogates, (client_loaders, ref_loader, test_loader)

# ----------------------------
# Metrics & Evaluation
# ----------------------------
def compute_edi(per_client_maps: List[Dict[int, torch.Tensor]], reference: Dict[int, torch.Tensor]) -> float:
    dists = []
    for maps in per_client_maps:
        for c in range(10):
            p = maps[c].view(1, -1)
            q = reference[c].view(1, -1)
            d = jsd(p, q)
            dists.append(float(d.item()))
    return float(np.mean(dists)) if dists else 0.0

def ref_from_hist(global_hist: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    ref = {}
    for c in range(10):
        v = global_hist[c].view(1,3,8,8)
        v_up = F.interpolate(v, size=(32,32), mode="bilinear", align_corners=False)
        v_up = normalize_to_simplex(v_up.view(1,-1), dim=-1).view(1,3,32,32)
        ref[c] = v_up
    return ref

def build_reference_from_mean(per_client_maps: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    ref = {}
    for c in range(10):
        mats = [m[c].view(1, -1) for m in per_client_maps]
        M = torch.cat(mats, dim=0)
        mean = M.mean(dim=0, keepdim=True)
        mean = normalize_to_simplex(mean, dim=-1).view(1,3,32,32)
        ref[c] = mean
    return ref

def build_imp_map_for_method(method: str, model: nn.Module, per_client_maps: List[Dict[int, torch.Tensor]] = None,
                             global_hist: Dict[int, torch.Tensor] = None, x: torch.Tensor = None, y: torch.Tensor = None,
                             client_id: int = 0, surrogates: List[Tuple] = None) -> torch.Tensor:
    if method == 'BL-B':
        N = x.size(0)
        maps = []
        for i in range(N):
            c = int(y[i].item())
            maps.append(per_client_maps[client_id][c])
        return torch.cat(maps, dim=0)
    elif method == 'BL-C':
        with torch.no_grad():
            preds = model(x).argmax(dim=-1)
        ref = ref_from_hist(global_hist)
        maps = []
        for i in range(x.size(0)):
            c = int(preds[i].item())
            maps.append(ref[c])
        return torch.cat(maps, dim=0)
    elif method == 'BL-D':
        with torch.no_grad():
            preds = model(flatten_img(x)).argmax(dim=-1)
        W = model.W.weight.detach()
        maps = []
        for i in range(x.size(0)):
            c = int(preds[i].item())
            m = normalize_to_simplex(W[c].abs().view(1,-1), dim=-1).view(1,3,32,32)
            maps.append(m)
        return torch.cat(maps, dim=0)
    elif method == 'xFL':
        if surrogates is not None and len(surrogates) > 0:
            surr, cnn = surrogates[client_id]
            surr.eval()
            cnn.eval()
            maps = []
            for i in range(x.size(0)):
                xi = x[i:i+1]
                yi = y[i:i+1]
                xi.requires_grad_(True)
                _, features = cnn(xi, return_features=True)
                features.requires_grad_(True)
                logits = surr(features)
                loss = F.cross_entropy(logits, yi)
                loss.backward()
                input_grad = xi.grad.detach().abs()
                m = normalize_to_simplex(input_grad.view(1,-1), dim=-1).view(1,3,32,32)
                maps.append(m)
            return torch.cat(maps, dim=0)
        else:
            with torch.no_grad():
                preds = model(x).argmax(dim=-1)
            maps = []
            for i in range(x.size(0)):
                c = int(preds[i].item())
                maps.append(per_client_maps[client_id][c])
            return torch.cat(maps, dim=0)
    else:
        raise ValueError("Unknown method")

def save_bar(values, labels, title, fname, ylabel):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(5,3))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    pth = os.path.join("outputs", fname)
    plt.savefig(pth, bbox_inches='tight')
    plt.close()
    print(f"Saved {pth}")

# ----------------------------
# Experiment Runner
# ----------------------------
def run_once(flcfg: FLConfig, xcfg: XFLConfig, seed: int):
    set_seed(seed)
    print(f"\n{'='*40}\n Seed {seed}\n{'='*40}")

    # BL-A
    print("Running BL-A (Plain FL)...")
    acc_A, model_A, loaders_A = run_plain_fl(flcfg)
    client_loaders_A, ref_loader_A, test_loader_A = loaders_A
    print(f"BL-A Acc: {acc_A:.4f}")

    # BL-B
    print("\nRunning BL-B (Local post-hoc)...")
    acc_B, model_B, per_client_maps_B, loaders_B = run_local_posthoc(flcfg)
    print(f"BL-B Acc: {acc_B:.4f}")

    # BL-C
    print("\nRunning BL-C (Server summary)...")
    acc_C, model_C, global_hist_C, loaders_C = run_server_summary(flcfg)
    print(f"BL-C Acc: {acc_C:.4f}")

    # BL-D
    print("\nRunning BL-D (Interpretable-only FL)...")
    acc_D, model_D, Wmaps_D_global, per_client_maps_D, loaders_D = run_interpretable_only(flcfg)
    print(f"BL-D Acc: {acc_D:.4f}")

    # xFL
    print("\nRunning xFL (Proposed)...")
    acc_X, model_X, Pi_X, per_client_surr_maps_X, per_client_surrogates_X, loaders_X = run_xfl(flcfg, xcfg)
    print(f"xFL Acc: {acc_X:.4f}")

    # EDI
    print("Computing EDI...")
    ref_B = build_reference_from_mean(per_client_maps_B)
    edi_B = compute_edi(per_client_maps_B, ref_B)
    
    ref_C = ref_from_hist(global_hist_C)
    edi_C = compute_edi(per_client_maps_B, ref_C)
    
    edi_D = compute_edi(per_client_maps_D, {c: Wmaps_D_global[c] for c in range(10)})
    
    ref_X = build_reference_from_mean(per_client_surr_maps_X)
    edi_X = compute_edi(per_client_surr_maps_X, ref_X)
    
    print(f"EDI -> BL-B: {edi_B:.4f} | BL-C: {edi_C:.4f} | BL-D: {edi_D:.4f} | xFL: {edi_X:.4f}")

    # Fidelity
    print("Computing Fidelity...")
    xb_all, yb_all = [], []
    for xb, yb in test_loader_A:
        xb_all.append(xb)
        yb_all.append(yb)
        if sum([t.size(0) for t in xb_all]) >= 128:
            break
    xb = torch.cat(xb_all, dim=0)[:128].to(DEVICE)
    yb = torch.cat(yb_all, dim=0)[:128].to(DEVICE)

    maps_B = build_imp_map_for_method('BL-B', model_B, per_client_maps=per_client_maps_B, x=xb, y=yb, client_id=0)
    maps_C = build_imp_map_for_method('BL-C', model_C, global_hist=global_hist_C, x=xb, y=yb)
    maps_D = build_imp_map_for_method('BL-D', model_D, x=xb, y=yb)
    maps_X = build_imp_map_for_method('xFL', model_X, per_client_maps=per_client_surr_maps_X, x=xb, y=yb, surrogates=per_client_surrogates_X, client_id=0)

    del_B, ins_B = deletion_insertion_auc(model_B, xb, yb, maps_B, steps=20)
    del_C, ins_C = deletion_insertion_auc(model_C, xb, yb, maps_C, steps=20)
    del_D, ins_D = deletion_insertion_auc(model_D, xb, yb, maps_D, steps=20)
    del_X, ins_X = deletion_insertion_auc(model_X, xb, yb, maps_X, steps=20)

    print(f"Del AUC: BL-B {del_B:.3f} | BL-C {del_C:.3f} | BL-D {del_D:.3f} | xFL {del_X:.3f}")
    print(f"Ins AUC: BL-B {ins_B:.3f} | BL-C {ins_C:.3f} | BL-D {ins_D:.3f} | xFL {ins_X:.3f}")

    return {
        "acc": {"BL-A": acc_A, "BL-B": acc_B, "BL-C": acc_C, "BL-D": acc_D, "xFL": acc_X},
        "edi": {"BL-B": edi_B, "BL-C": edi_C, "BL-D": edi_D, "xFL": edi_X},
        "del_auc": {"BL-B": del_B, "BL-C": del_C, "BL-D": del_D, "xFL": del_X},
        "ins_auc": {"BL-B": ins_B, "BL-C": ins_C, "BL-D": ins_D, "xFL": ins_X},
    }

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    flcfg = FLConfig(n_clients=8, rounds=15, local_epochs=1, alpha_dirichlet=0.1)
    xcfg = XFLConfig(topk=512, dp_sigma=0.2, surrogate_every_R=2, beta_align_final=0.3,
                     align_warmup_rounds=8, surrogate_steps=3, hidden_dim=64)

    seeds = [BASE_SEED + i for i in range(5)]
    methods_all = ["BL-A", "BL-B", "BL-C", "BL-D", "xFL"]
    methods_expl = ["BL-B", "BL-C", "BL-D", "xFL"]

    metrics = {
        "acc": {m: [] for m in methods_all},
        "edi": {m: [] for m in methods_expl},
        "del_auc": {m: [] for m in methods_expl},
        "ins_auc": {m: [] for m in methods_expl},
    }

    print(f"Starting experiment for {len(seeds)} seeds...")
    for s in seeds:
        out = run_once(flcfg, xcfg, s)
        for m in methods_all:
            metrics["acc"][m].append(out["acc"][m])
        for m in methods_expl:
            metrics["edi"][m].append(out["edi"][m])
            metrics["del_auc"][m].append(out["del_auc"][m])
            metrics["ins_auc"][m].append(out["ins_auc"][m])

    def summarize(arr):
        arr = np.array(arr, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    print("\n===== Summary over seeds =====")
    print("Accuracy:")
    for m in methods_all:
        mean_v, std_v = summarize(metrics["acc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nEDI (lower better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["edi"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nDeletion AUC (lower better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["del_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    print("\nInsertion AUC (higher better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["ins_auc"][m])
        print(f"  {m}: {mean_v:.4f} ± {std_v:.4f}")

    # Save plots
    mean_del = [summarize(metrics["del_auc"][m])[0] for m in methods_expl]
    mean_ins = [summarize(metrics["ins_auc"][m])[0] for m in methods_expl]
    mean_edi = [summarize(metrics["edi"][m])[0] for m in methods_expl]

    save_bar(mean_del, methods_expl, "Deletion AUC (mean)", "bar_deletion_auc_mean.png", "AUC")
    save_bar(mean_ins, methods_expl, "Insertion AUC (mean)", "bar_insertion_auc_mean.png", "AUC")
    save_bar(mean_edi, methods_expl, "Consistency (EDI, mean)", "bar_edi_mean.png", "EDI")

    print("Done.")