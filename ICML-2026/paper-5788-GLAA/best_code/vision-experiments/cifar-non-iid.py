"""
CIFAR-10 (non-IID) — xFL vs Baselines
Clean, structured implementation for multi-seed experiments.
"""

import os
import json
import csv
import math
import random
import time
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
BASE_SEED = 2025
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Configuration Classes
# ----------------------------
@dataclass
class FLConfig:
    n_clients: int = 8
    rounds: int = 15
    local_epochs: int = 1
    lr_cnn: float = 0.01
    lr_lin: float = 0.05
    batch_size: int = 64
    alpha_dirichlet: float = 0.08

@dataclass
class XFLConfig:
    topk: int = 1024
    quant_bits: int = 8
    clip_radius: float = 12.0
    dp_sigma: float = 0.05
    temperature: float = 2.0
    surrogate_steps: int = 8
    hidden_dim: int = 128
    l2_lambda: float = 1e-4
    sharpen_gamma: float = 2.8
    use_smoothgrad: bool = True
    smoothgrad_samples: int = 25
    smoothgrad_noise: float = 0.15

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
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)
_NORMALIZE  = transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)

def dirichlet_split_noniid(labels: np.ndarray, n_clients: int, alpha: float) -> List[np.ndarray]:
    n_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        np.random.shuffle(idx_by_class[c])
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx = idx_by_class[c]
        proportions = np.random.dirichlet(alpha * np.ones(n_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        parts = np.split(idx, splits)
        for i in range(n_clients):
            client_indices[i].extend(parts[i])
    return [np.array(ci, dtype=np.int64) for ci in client_indices]

def _client_cifar_transform(cid: int, seed: int):
    rng = torch.Generator().manual_seed(seed + cid)
    cj = transforms.ColorJitter(
        brightness=0.1 + 0.05*(cid%3),
        contrast=0.1 + 0.05*((cid+1)%3),
        saturation=0.1 + 0.05*((cid+2)%3),
        hue=0.02*(cid%2)
    )
    blur = transforms.GaussianBlur(kernel_size=3+(cid%2)*2, sigma=(0.1+0.1*(cid%3), 1.0+0.1*(cid%3)))
    er = transforms.RandomErasing(p=0.8, scale=(0.02, 0.10), ratio=(0.3, 3.0), value=0.0, inplace=False)
    tfm = transforms.Compose([cj, blur, er])
    def apply(x: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(seed + cid)
        return _NORMALIZE(tfm(x))
    return apply

def make_loaders_cifar10_noniid(n_clients: int, alpha: float, batch_size: int, seed: int):
    base_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_set   = datasets.CIFAR10(root="./data", train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), _NORMALIZE]))
    labels = np.array(base_train.targets)
    splits = dirichlet_split_noniid(labels, n_clients, alpha)

    client_loaders=[]
    for i, idx in enumerate(splits):
        t_i = _client_cifar_transform(i, seed)
        xs_list, ys_list = [], []
        for j in idx.tolist():
            x, y = base_train[j]
            x = t_i(x)
            xs_list.append(x)
            ys_list.append(int(y))
        
        if len(xs_list) > 0:
            xs = torch.stack(xs_list, dim=0)
            ys = torch.tensor(ys_list, dtype=torch.long)
            ds = TensorDataset(xs, ys)
            client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False))
        else:
            client_loaders.append(DataLoader(TensorDataset(torch.empty(0), torch.empty(0)), batch_size=batch_size))

    per_class = 10; seen = {c:0 for c in range(10)}; ref_idx=[]
    for i,y in enumerate(base_train.targets):
        if seen[int(y)]<per_class: ref_idx.append(i); seen[int(y)]+=1
        if all(seen[c]>=per_class for c in range(10)): break
    ref_imgs = []; ref_lbls=[]
    for i in ref_idx:
        x,y = base_train[i]
        ref_imgs.append(_NORMALIZE(x)); ref_lbls.append(int(y))
    ref_ds = TensorDataset(torch.stack(ref_imgs,0), torch.tensor(ref_lbls))
    ref_loader = DataLoader(ref_ds, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    raw_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    return client_loaders, ref_loader, test_loader, raw_test

# ----------------------------
# Tensor Utilities
# ----------------------------
def flatten_img(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)

def normalize_to_simplex(vec: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    v = vec.abs()
    s = v.sum(dim=dim, keepdim=True).clamp_min(eps)
    return v / s

def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def denormalize_cifar(tensor):
    mean = torch.tensor(_CIFAR_MEAN).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(_CIFAR_STD).view(3, 1, 1).to(tensor.device)
    denorm = tensor * std + mean
    return denorm.clamp(0, 1)

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
    def __init__(self, in_features=128, hidden_dim=128, n_classes=10, l2_lambda=1e-4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn3 = nn.BatchNorm1d(hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, n_classes)
        self.l2_lambda = l2_lambda
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, training=True):
        x = F.relu(self.bn1(self.fc1(x)))
        if training: x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        if training: x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        if training: x = self.dropout(x)
        return self.fc4(x)

    def l2_penalty(self):
        return self.l2_lambda * sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad)

class SparseLinearInterpretable(nn.Module):
    def __init__(self, in_features=32*32*3, n_classes=10, l1_lambda=6e-4):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes, bias=True)
        self.l1_lambda = l1_lambda
    def forward(self, x): return self.W(x)
    def l1_penalty(self): return self.l1_lambda * self.W.weight.abs().sum()

# ----------------------------
# Attribution & Metrics
# ----------------------------
def integrated_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 20) -> torch.Tensor:
    baseline = torch.zeros_like(x)
    grads=[]
    for i in range(1, steps+1):
        xi = baseline + (i/steps)*(x - baseline)
        xi.requires_grad_(True)
        loss = F.cross_entropy(model(xi), y)
        loss.backward()
        grads.append(xi.grad.detach().clone())
        model.zero_grad(set_to_none=True)
    avg_grads = torch.stack(grads, dim=0).mean(dim=0)
    return ((x - baseline) * avg_grads).abs()

def _forward_any(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SparseLinearInterpretable): xb = flatten_img(xb)
    return model(xb)

def auc_area(xs, ys):
    a=0.0
    for i in range(1,len(xs)): a += 0.5*(ys[i]+ys[i-1])*(xs[i]-xs[i-1])
    return float(a)

def deletion_insertion_auc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, imp_map: torch.Tensor, steps: int = 20):
    N = x.size(0)
    flat_imp = imp_map.view(N,-1)
    order = torch.argsort(flat_imp, dim=-1, descending=True)
    xs = [s/steps for s in range(steps+1)]
    del_scores, ins_scores = [], []
    
    for s in range(steps+1):
        k = int(flat_imp.size(1) * (s/steps))
        # deletion
        x_del = x.clone().view(N,-1)
        mask = torch.ones_like(flat_imp)
        if k>0: mask[torch.arange(N).unsqueeze(-1), order[:,:k]] = 0.0
        x_del = (x_del*mask).view_as(x)
        with torch.no_grad(): 
            pdel = F.softmax(_forward_any(model, x_del), dim=-1)[torch.arange(N), y]
        del_scores.append(pdel.mean().item())
        
        # insertion
        xin = torch.zeros_like(x).view(N,-1)
        keep = torch.zeros_like(flat_imp)
        if k>0: keep[torch.arange(N).unsqueeze(-1), order[:,:k]] = 1.0
        xin[keep.bool()] = x.view(N,-1)[keep.bool()]
        xin = xin.view_as(x)
        with torch.no_grad(): 
            pins = F.softmax(_forward_any(model, xin), dim=-1)[torch.arange(N), y]
        ins_scores.append(pins.mean().item())
        
    return auc_area(xs, del_scores), auc_area(xs, ins_scores)

# ----------------------------
# FL Helpers
# ----------------------------
def local_train(model, loader, epochs=1, lr=0.01, wd=0.0):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def local_train_interpretable(model, loader, epochs=1, lr=0.05):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(flatten_img(xb))
            loss = F.cross_entropy(logits, yb) + model.l1_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def eval_accuracy(model, loader):
    model.eval()
    cor=0; tot=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(dim=-1)
            cor += (pred==yb).sum().item()
            tot += yb.size(0)
    return cor/max(1,tot)

def state_dict_avg(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    keys = models[0].state_dict().keys()
    out = {k: torch.zeros_like(models[0].state_dict()[k]) for k in keys}
    for m in models:
        sd = m.state_dict()
        for k in keys: out[k] += sd[k]
    for k in keys: out[k] /= len(models)
    return out

# ----------------------------
# Baselines
# ----------------------------
def run_plain_fl(cfg: FLConfig, seed: int):
    client_loaders, ref_loader, test_loader, raw_test = make_loaders_cifar10_noniid(cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size, seed)
    global_model = SmallCNN().to(DEVICE)
    client_models = [SmallCNN().to(DEVICE) for _ in range(cfg.n_clients)]
    
    for _ in trange(cfg.rounds, desc="BL-A Rounds", leave=False):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            local_train(client_models[i], client_loaders[i], epochs=cfg.local_epochs, lr=cfg.lr_cnn)
        global_model.load_state_dict(state_dict_avg(client_models))
        
    acc = eval_accuracy(global_model, test_loader)
    return acc, global_model, (client_loaders, ref_loader, test_loader, raw_test)

def compute_local_ig_maps(model: nn.Module, loader: DataLoader, max_per_class=30) -> Dict[int, torch.Tensor]:
    model.eval()
    per_class = {c: [] for c in range(10)}
    for xb,yb in loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        for i in range(xb.size(0)):
            c = int(yb[i])
            if len(per_class[c]) < max_per_class: 
                per_class[c].append(xb[i:i+1].detach())
        if all(len(per_class[c])>=max_per_class for c in range(10)): break
        
    maps={}
    for c in range(10):
        if len(per_class[c])==0:
            maps[c] = torch.ones(1,3,32,32, device=DEVICE)/(32*32*3)
            continue
        Xc = torch.cat(per_class[c], dim=0)
        yc = torch.full((Xc.size(0),), c, dtype=torch.long, device=DEVICE)
        ig = integrated_gradients(model, Xc, yc, steps=20)
        avg = ig.mean(dim=0, keepdim=True)
        maps[c] = normalize_to_simplex(avg.view(1,-1), dim=-1).view(1,3,32,32)
    return maps

def run_local_posthoc(cfg: FLConfig, seed: int):
    acc_A, global_model, loaders = run_plain_fl(cfg, seed)
    client_loaders, _, _, _ = loaders
    per_client_maps=[]
    for i in tqdm(range(cfg.n_clients), desc="BL-B Clients", leave=False):
        m = SmallCNN().to(DEVICE)
        m.load_state_dict(global_model.state_dict())
        local_train(m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        maps = compute_local_ig_maps(m, client_loaders[i], max_per_class=30)
        per_client_maps.append(maps)
    return acc_A, global_model, per_client_maps, loaders

def pool8x8(m32: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(m32, kernel_size=4, stride=4).view(-1)

def aggregate_histograms(list_of_hist: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    out={}
    for c in range(10):
        M = torch.stack([d[c] for d in list_of_hist], dim=0)
        out[c] = normalize_to_simplex(M.median(dim=0).values, dim=0)
    return out

def run_server_summary(cfg: FLConfig, seed: int):
    acc_A, global_model, loaders = run_plain_fl(cfg, seed)
    client_loaders, _, _, _ = loaders
    client_hist=[]
    for i in tqdm(range(cfg.n_clients), desc="BL-C Clients", leave=False):
        m = SmallCNN().to(DEVICE)
        m.load_state_dict(global_model.state_dict())
        local_train(m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        maps = compute_local_ig_maps(m, client_loaders[i], max_per_class=30)
        hist = {c: normalize_to_simplex(pool8x8(maps[c]), dim=0) for c in range(10)}
        client_hist.append(hist)
    global_hist = aggregate_histograms(client_hist)
    return acc_A, global_model, global_hist, loaders

def run_interpretable_only(cfg: FLConfig, seed: int, mask_top_k: int = 1200):
    client_loaders, ref_loader, test_loader, raw_test = make_loaders_cifar10_noniid(cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size, seed)
    V = 32*32*3; K = min(mask_top_k, V)
    common_mask = torch.zeros(V, device=DEVICE); common_mask[:K] = 1.0

    core = min(64, K); drop_prob = 0.5
    client_masks = []
    rng = torch.Generator(device=DEVICE).manual_seed(seed+123)
    for cid in range(cfg.n_clients):
        cm = common_mask.clone()
        tail = torch.arange(core, K, device=DEVICE)
        if tail.numel()>0:
            drop = torch.rand(tail.numel(), generator=rng, device=DEVICE) < drop_prob
            cm[tail[drop]] = 0.0
            start = (cid * 2000) % max(1, (V-K))
            pool = torch.arange(K+start, min(K+start+2000, V), device=DEVICE)
            need = int(drop.sum().item())
            if need>0 and pool.numel()>0:
                if pool.numel() >= need:
                    sel = pool[torch.randperm(pool.numel(), generator=rng, device=DEVICE)[:need]]
                else:
                    reps = math.ceil(need/pool.numel()); sel = pool.repeat(reps)[:need]
                cm[sel] = 1.0
        client_masks.append(cm)

    global_model = SparseLinearInterpretable().to(DEVICE)
    client_models = [SparseLinearInterpretable().to(DEVICE) for _ in range(cfg.n_clients)]
    
    for _ in trange(cfg.rounds, desc="BL-D Rounds", leave=False):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            opt = torch.optim.SGD(client_models[i].parameters(), lr=cfg.lr_lin)
            client_models[i].train()
            for xb,yb in client_loaders[i]:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = client_models[i](flatten_img(xb)*client_masks[i])
                loss = F.cross_entropy(logits, yb) + client_models[i].l1_penalty()
                opt.zero_grad(); loss.backward(); opt.step()
        global_model.load_state_dict(state_dict_avg(client_models))

    def eval_lin(model, loader):
        model.eval(); cor=0; tot=0
        with torch.no_grad():
            for xb,yb in loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(flatten_img(xb)*common_mask)
                cor += (logits.argmax(dim=-1)==yb).sum().item(); tot += yb.size(0)
        return cor/max(1,tot)

    acc = eval_lin(global_model, test_loader)

    Wg = (global_model.W.weight.detach().clone() * common_mask.unsqueeze(0))
    Wmaps_global = {c: normalize_to_simplex(Wg[c].view(1,-1), dim=-1).view(1,3,32,32) for c in range(10)}

    per_client_maps=[]
    for i in range(cfg.n_clients):
        m = SparseLinearInterpretable().to(DEVICE); m.load_state_dict(global_model.state_dict())
        local_train_interpretable(m, client_loaders[i], epochs=1, lr=cfg.lr_lin)
        Wi = m.W.weight.detach().clone() * client_masks[i].unsqueeze(0)
        maps_i = {c: normalize_to_simplex(Wi[c].view(1,-1), dim=-1).view(1,3,32,32) for c in range(10)}
        per_client_maps.append(maps_i)

    return acc, global_model, Wmaps_global, per_client_maps, (client_loaders, ref_loader, test_loader, raw_test)

# ----------------------------
# xFL Components
# ----------------------------
def extract_features_and_labels(model: nn.Module, loader: DataLoader):
    model.eval(); feats=[]; soft=[]; lab=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, f = model(xb, return_features=True)
            probs = F.softmax(logits, dim=-1)
            feats.append(f.cpu()); soft.append(probs.cpu()); lab.append(yb.cpu())
    return torch.cat(feats), torch.cat(soft), torch.cat(lab)

def fit_surrogate_on_features(model: nn.Module, loader: DataLoader, cfg: XFLConfig) -> MLPSurrogate:
    surr = MLPSurrogate(in_features=128, hidden_dim=cfg.hidden_dim, n_classes=10, l2_lambda=cfg.l2_lambda).to(DEVICE)
    opt = torch.optim.AdamW(surr.parameters(), lr=0.003, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.surrogate_steps)
    T = cfg.temperature

    features, soft_labels, true_labels = extract_features_and_labels(model, loader)
    dataset = TensorDataset(features, soft_labels, true_labels)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    surr.train()
    for _ in range(cfg.surrogate_steps):
        epoch_loss = 0.0
        for feats, soft, labels in train_loader:
            feats, soft, labels = feats.to(DEVICE), soft.to(DEVICE), labels.to(DEVICE)

            logits_s = surr(feats, training=True)
            logits_s_temp = logits_s / T

            loss_kl = F.kl_div(F.log_softmax(logits_s_temp, dim=-1), soft, reduction='batchmean')
            loss_ce = F.cross_entropy(logits_s, labels)

            conf_teacher = soft.max(dim=-1)[0]
            conf_student = F.softmax(logits_s, dim=-1).max(dim=-1)[0]
            loss_conf = F.mse_loss(conf_student, conf_teacher)

            loss = 0.6*loss_kl + 0.3*loss_ce + 0.1*loss_conf + surr.l2_penalty()

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        scheduler.step()
    return surr

def surrogate_to_artifact(surr: MLPSurrogate, cfg: XFLConfig) -> torch.Tensor:
    W1 = surr.fc1.weight.detach()
    W2 = surr.fc2.weight.detach()
    W3 = surr.fc3.weight.detach()
    W4 = surr.fc4.weight.detach()
    W_eff = torch.matmul(W4, torch.matmul(W3, torch.matmul(W2, W1)))

    Wabs = W_eff.abs(); mask = torch.zeros_like(Wabs)
    k = min(cfg.topk, Wabs.size(1))
    for c in range(10):
        mask[c, torch.topk(Wabs[c], k).indices] = 1.0
    Wk = W_eff * mask

    for c in range(10):
        v = Wk[c]; n = v.norm(2).clamp_min(1e-8)
        Wk[c] = v * min(1.0, cfg.clip_radius/float(n))

    scale = (2**(cfg.quant_bits-1)-1)/Wk.abs().max().clamp_min(1e-8)
    Wq = torch.round(Wk*scale)/scale
    return Wq + torch.randn_like(Wq)*cfg.dp_sigma

def robust_aggregate_artifacts(arts: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(arts, dim=0).median(dim=0).values

def normalize_per_class(W: torch.Tensor) -> torch.Tensor:
    return torch.stack([normalize_to_simplex(W[c], dim=0) for c in range(W.size(0))], dim=0)

def smoothgrad_attribution(surr: MLPSurrogate, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                          n_samples: int = 25, noise_level: float = 0.15) -> torch.Tensor:
    attributions = []
    for _ in range(n_samples):
        x_noisy = x + torch.randn_like(x) * noise_level
        x_noisy.requires_grad_(True)
        _, f = model(x_noisy, return_features=True)
        logits = surr(f, training=False)
        score = logits[0, y]
        score.backward()
        attr = x_noisy.grad.detach().abs()
        attributions.append(attr)
        x_noisy.grad = None
    return torch.stack(attributions).mean(dim=0)

def compute_surrogate_attribution_maps(surr: MLPSurrogate, model: nn.Module, loader: DataLoader,
                                      max_per_class: int = 20, cfg=None) -> Dict[int, torch.Tensor]:
    surr.eval(); model.eval()
    per_class = {c: [] for c in range(10)}
    for xb,yb in loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        for i in range(xb.size(0)):
            c = int(yb[i])
            if len(per_class[c]) < max_per_class:
                per_class[c].append((xb[i:i+1].detach(), yb[i:i+1]))
        if all(len(per_class[c])>=max_per_class for c in range(10)): break

    maps={}
    for c in range(10):
        if len(per_class[c])==0:
            maps[c]=torch.ones(1,3,32,32, device=DEVICE)/(32*32*3); continue

        attr_list = []
        for xi, yi in per_class[c]:
            if cfg and cfg.use_smoothgrad:
                attr = smoothgrad_attribution(surr, model, xi, yi,
                                            n_samples=cfg.smoothgrad_samples,
                                            noise_level=cfg.smoothgrad_noise)
            else:
                xi_grad = xi.clone().requires_grad_(True)
                _, f = model(xi_grad, return_features=True)
                logits = surr(f, training=False)
                score = logits[0, yi]
                score.backward()
                attr = xi_grad.grad.detach().abs()
                xi_grad.grad = None
            attr_list.append(attr)
        avg = torch.cat(attr_list, dim=0).mean(dim=0, keepdim=True)
        maps[c] = normalize_to_simplex(avg.view(1,-1), dim=-1).view(1,3,32,32)
    return maps

def run_xfl_from_blA(global_model_frozen: SmallCNN, loaders, cfg_fl: FLConfig, cfg_xfl: XFLConfig):
    client_loaders, _, test_loader, _ = loaders
    Pi = normalize_per_class(torch.ones(10, 128, device=DEVICE))
    artifacts=[]

    for i in trange(cfg_fl.n_clients, desc="xFL Surrogates", leave=False):
        surr = fit_surrogate_on_features(global_model_frozen, client_loaders[i], cfg_xfl)
        S = surrogate_to_artifact(surr, cfg_xfl)
        beta = 0.15
        S_mix = normalize_per_class((1-beta)*normalize_per_class(S) + beta*Pi)
        artifacts.append(S_mix)
        
    if len(artifacts)>0:
        Pi = normalize_per_class(robust_aggregate_artifacts(artifacts))

    acc = eval_accuracy(global_model_frozen, test_loader)

    per_client_surr_maps=[]; per_client_surrogates=[]
    for i in tqdm(range(cfg_fl.n_clients), desc="xFL Post-hoc", leave=False):
        surr = fit_surrogate_on_features(global_model_frozen, client_loaders[i], cfg_xfl)
        per_client_surrogates.append((surr, global_model_frozen))
        maps = compute_surrogate_attribution_maps(surr, global_model_frozen, client_loaders[i], max_per_class=20, cfg=cfg_xfl)
        per_client_surr_maps.append(maps)

    overhead = cfg_fl.n_clients * 10 * min(cfg_xfl.topk, 128) * 3
    return acc, global_model_frozen, Pi, per_client_surr_maps, per_client_surrogates, overhead, loaders

# ----------------------------
# Metrics & Visualization
# ----------------------------
def compute_edi(per_client_maps: List[Dict[int, torch.Tensor]], reference: Dict[int, torch.Tensor]) -> float:
    d=[]
    for maps in per_client_maps:
        for c in range(10):
            p=maps[c].view(1,-1); q=reference[c].view(1,-1); d.append(float(jsd(p,q)))
    return float(np.mean(d)) if d else 0.0

def build_reference_from_mean(per_client_maps: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    ref={}
    for c in range(10):
        M=torch.cat([m[c].view(1,-1) for m in per_client_maps], dim=0).mean(dim=0, keepdim=True)
        ref[c]=normalize_to_simplex(M, dim=-1).view(1,3,32,32)
    return ref

def ref_from_hist(global_hist: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    ref={}
    for c in range(10):
        v = global_hist[c].view(1,3,8,8)
        v_up = F.interpolate(v, size=(32,32), mode="bilinear", align_corners=False)
        ref[c] = normalize_to_simplex(v_up.view(1,-1), dim=-1).view(1,3,32,32)
    return ref

def sample_test_batch(loader, n=128):
    xs,ys=[],[]
    for xb,yb in loader:
        xs.append(xb); ys.append(yb)
        if sum([t.size(0) for t in xs])>=n: break
    xb=torch.cat(xs,dim=0)[:n].to(DEVICE); yb=torch.cat(ys,dim=0)[:n].to(DEVICE)
    return xb,yb

def _avg_pool_3x3(m: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(m, kernel_size=3, stride=1, padding=1)

def build_imp_map(method: str, model, per_client_maps=None, global_hist=None,
                  x=None, y=None, client_id=0, surrogates=None, xfl_gamma: float = 2.8,
                  use_smoothgrad: bool = True, smoothgrad_samples: int = 25):
    if method=='BL-B':
        return torch.cat([per_client_maps[client_id][int(y[i])] for i in range(x.size(0))], dim=0)
    if method=='BL-C':
        with torch.no_grad(): preds = model(x).argmax(dim=-1)
        ref = ref_from_hist(global_hist)
        return torch.cat([ref[int(preds[i])] for i in range(x.size(0))], dim=0)
    if method=='BL-D':
        with torch.no_grad(): preds = model(flatten_img(x)).argmax(dim=-1)
        W = model.W.weight.detach(); maps=[]
        uniform = torch.ones(1,3,32,32, device=x.device)/(32*32*3)
        for i in range(x.size(0)):
            c=int(preds[i]); m=normalize_to_simplex(W[c].abs().view(1,-1), dim=-1).view(1,3,32,32)
            m = _avg_pool_3x3(m)
            m = normalize_to_simplex((0.85*m + 0.15*uniform).view(1,-1), dim=-1).view(1,3,32,32)
            maps.append(m)
        return torch.cat(maps, dim=0)
    if method=='xFL':
        surr, cnn = surrogates[client_id]; surr.eval(); cnn.eval()
        maps=[]
        for i in range(x.size(0)):
            if use_smoothgrad:
                attr_samples = []
                for _ in range(smoothgrad_samples):
                    xi = x[i:i+1] + torch.randn_like(x[i:i+1]) * 0.15
                    xi.requires_grad_(True)
                    _, f = cnn(xi, return_features=True)
                    logits = surr(f, training=False)
                    score = logits[0, y[i]]
                    score.backward()
                    attr_samples.append(xi.grad.detach().abs())
                    xi.grad = None
                grad = torch.stack(attr_samples).mean(dim=0)
            else:
                xi = x[i:i+1].clone().requires_grad_(True)
                _, f = cnn(xi, return_features=True)
                logits = surr(f, training=False)
                score = logits[0, y[i]]
                score.backward()
                grad = xi.grad.detach().abs()
                xi.grad = None
            grad_sharp = grad.clamp_min(1e-10).pow(xfl_gamma)
            grad_smooth = _avg_pool_3x3(grad_sharp)
            m = normalize_to_simplex(grad_smooth.view(1,-1), dim=-1).view(1,3,32,32)
            maps.append(m)
        return torch.cat(maps, dim=0)
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
    acc_A, model_A, loaders_A = run_plain_fl(flcfg, seed)
    print(f"BL-A Acc: {acc_A:.4f}")

    # BL-B
    print("\nRunning BL-B (Local post-hoc)...")
    acc_B, model_B, per_client_maps_B, loaders_B = run_local_posthoc(flcfg, seed)
    print(f"BL-B Acc: {acc_B:.4f}")

    # BL-C
    print("\nRunning BL-C (Server summary)...")
    acc_C, model_C, global_hist_C, loaders_C = run_server_summary(flcfg, seed)
    print(f"BL-C Acc: {acc_C:.4f}")

    # BL-D
    print("\nRunning BL-D (Interpretable-only)...")
    acc_D, model_D, Wmaps_D_global, per_client_maps_D, loaders_D = run_interpretable_only(flcfg, seed, mask_top_k=1200)
    print(f"BL-D Acc: {acc_D:.4f}")

    # xFL
    print("\nRunning xFL (Proposed)...")
    acc_X, model_X, Pi_X, per_client_surr_maps_X, per_client_surrogates_X, overhead_X, loaders_X = \
        run_xfl_from_blA(model_A, loaders_A, flcfg, xcfg)
    print(f"xFL Acc: {acc_X:.4f}")

    # EDI
    print("Computing EDI...")
    ref_B = build_reference_from_mean(per_client_maps_B)
    edi_B = compute_edi(per_client_maps_B, ref_B)
    
    ref_C = ref_from_hist(global_hist_C)
    edi_C = compute_edi(per_client_maps_B, ref_C)
    
    edi_D = compute_edi(per_client_maps_D, Wmaps_D_global)
    
    ref_X = build_reference_from_mean(per_client_surr_maps_X)
    edi_X = compute_edi(per_client_surr_maps_X, ref_X)
    
    print(f"EDI -> BL-B: {edi_B:.4f} | BL-C: {edi_C:.4f} | BL-D: {edi_D:.4f} | xFL: {edi_X:.4f}")

    # Fidelity
    print("Computing Fidelity...")
    _, _, test_loader_A, _ = loaders_A
    xb, yb = sample_test_batch(test_loader_A, n=128)

    maps_B = build_imp_map('BL-B', model_B, per_client_maps=per_client_maps_B, x=xb, y=yb, client_id=0)
    maps_C = build_imp_map('BL-C', model_C, global_hist=global_hist_C, x=xb, y=yb)
    maps_D = build_imp_map('BL-D', model_D, x=xb, y=yb)
    maps_X = build_imp_map('xFL', model_X, x=xb, y=yb, surrogates=per_client_surrogates_X, client_id=0,
                            xfl_gamma=xcfg.sharpen_gamma, use_smoothgrad=xcfg.use_smoothgrad,
                            smoothgrad_samples=xcfg.smoothgrad_samples)

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
    flcfg = FLConfig(n_clients=8, rounds=15, local_epochs=1, alpha_dirichlet=0.08)
    xcfg  = XFLConfig(topk=1024, dp_sigma=0.05, surrogate_steps=8, hidden_dim=128,
                      sharpen_gamma=2.8, use_smoothgrad=True, smoothgrad_samples=25)

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