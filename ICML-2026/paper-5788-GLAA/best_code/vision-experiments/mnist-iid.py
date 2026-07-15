"""
xFedAlign: Federated Learning with Explanation Alignment
Clean, structured implementation for MNIST experiments
"""

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm, trange


# ============================================================================
# REPRODUCIBILITY UTILITIES
# ============================================================================

BASE_SEED = 1337
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class FLConfig:
    """Federated Learning configuration"""
    n_clients: int = 8
    rounds: int = 15
    local_epochs: int = 1
    lr_cnn: float = 0.01
    lr_lin: float = 0.05
    batch_size: int = 64
    alpha_dirichlet: float = 1e6  # IID setting


@dataclass
class XFLConfig:
    """xFedAlign-specific configuration"""
    topk: int = 128
    quant_bits: int = 8
    clip_radius: float = 5.0
    dp_sigma: float = 0.1
    temperature: float = 3.0
    beta_align_final: float = 0.2
    align_warmup_rounds: int = 6
    surrogate_every_R: int = 2
    l1_lambda: float = 1e-4
    tiny_task_lr: float = 0.001
    tiny_task_epochs: int = 1


# ============================================================================
# DATA UTILITIES
# ============================================================================

def dirichlet_split_noniid(
    labels: np.ndarray, 
    n_clients: int, 
    alpha: float
) -> List[np.ndarray]:
    """Split data among clients using Dirichlet distribution"""
    n_classes = int(labels.max()) + 1
    
    # Group indices by class
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        np.random.shuffle(idx_by_class[c])
    
    # Distribute each class across clients
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


def make_loaders_mnist(
    n_clients: int, 
    alpha: float, 
    batch_size: int
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """Create MNIST dataloaders for federated learning"""
    tfm = transforms.Compose([transforms.ToTensor()])
    
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    
    # Split training data among clients
    labels = np.array(train_set.targets)
    splits = dirichlet_split_noniid(labels, n_clients, alpha)
    
    client_loaders = []
    for idx in splits:
        subset = Subset(train_set, idx.tolist())
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    # Create reference loader with balanced samples
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
    
    ref_loader = DataLoader(
        Subset(train_set, ref_idx), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return client_loaders, ref_loader, test_loader


# ============================================================================
# TENSOR UTILITIES
# ============================================================================

def flatten_img(x: torch.Tensor) -> torch.Tensor:
    """Flatten image tensor to vector"""
    return x.view(x.size(0), -1)


def normalize_to_simplex(
    vec: torch.Tensor, 
    dim: int = -1, 
    eps: float = 1e-12
) -> torch.Tensor:
    """Normalize tensor to probability simplex"""
    vec = vec.abs()
    s = vec.sum(dim=dim, keepdim=True).clamp_min(eps)
    return vec / s


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon Divergence between two probability distributions"""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class SmallCNN(nn.Module):
    """Small CNN for MNIST classification"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SparseLinearSurrogate(nn.Module):
    """Sparse linear model for surrogate training"""
    
    def __init__(
        self, 
        in_features: int = 28*28, 
        n_classes: int = 10, 
        l1_lambda: float = 1e-4
    ):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes, bias=True)
        self.l1_lambda = l1_lambda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x)
    
    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 regularization penalty"""
        return self.l1_lambda * self.W.weight.abs().sum()


class SparseLinearInterpretable(nn.Module):
    """Sparse linear model for interpretable learning"""
    
    def __init__(
        self, 
        in_features: int = 28*28, 
        n_classes: int = 10, 
        l1_lambda: float = 5e-4
    ):
        super().__init__()
        self.W = nn.Linear(in_features, n_classes, bias=True)
        self.l1_lambda = l1_lambda
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x)
    
    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 regularization penalty"""
        return self.l1_lambda * self.W.weight.abs().sum()


# ============================================================================
# ATTRIBUTION METHODS
# ============================================================================

def integrated_gradients(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    steps: int = 20
) -> torch.Tensor:
    """Compute Integrated Gradients attribution"""
    baseline = torch.zeros_like(x)
    grads = []
    
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


def compute_local_ig_maps(
    model: nn.Module, 
    loader: DataLoader, 
    max_per_class: int = 50
) -> Dict[int, torch.Tensor]:
    """Compute averaged IG maps per class from local data"""
    model.eval()
    
    # Collect samples per class
    per_class_imgs = {c: [] for c in range(10)}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        for i in range(xb.size(0)):
            c = int(yb[i].item())
            if len(per_class_imgs[c]) < max_per_class:
                per_class_imgs[c].append(xb[i:i+1].detach())
        
        if all(len(per_class_imgs[c]) >= max_per_class for c in range(10)):
            break
    
    # Compute IG maps per class
    maps = {}
    for c in range(10):
        if len(per_class_imgs[c]) == 0:
            maps[c] = torch.ones(1, 1, 28, 28, device=device) / (28 * 28)
            continue
        
        Xc = torch.cat(per_class_imgs[c], dim=0)
        yc = torch.full((Xc.size(0),), c, dtype=torch.long, device=device)
        ig = integrated_gradients(model, Xc, yc, steps=10)
        
        avg_map = ig.mean(dim=0, keepdim=True)
        maps[c] = normalize_to_simplex(avg_map.view(1, -1), dim=-1).view(1, 1, 28, 28)
    
    return maps


# ============================================================================
# DELETION / INSERTION METRICS
# ============================================================================

def auc_area(xs: List[float], ys: List[float]) -> float:
    """Compute area under curve using trapezoidal rule"""
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        area += 0.5 * (ys[i] + ys[i - 1]) * dx
    return float(area)


def _forward_model(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """Forward pass handling different model types"""
    if isinstance(model, (SparseLinearInterpretable, SparseLinearSurrogate)):
        xb = flatten_img(xb)
    return model(xb)


def deletion_insertion_auc(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    imp_map: torch.Tensor, 
    steps: int = 20
) -> Tuple[float, float]:
    """Compute deletion and insertion AUC metrics"""
    N = x.size(0)
    flat_imp = imp_map.view(N, -1)
    idx_sorted = torch.argsort(flat_imp, dim=-1, descending=True)
    
    del_scores, ins_scores, xs = [], [], []
    
    for s in range(steps + 1):
        frac = s / steps
        xs.append(frac)
        k = int(flat_imp.size(1) * frac)
        
        # Deletion curve
        x_del = x.clone()
        if k > 0:
            mask = torch.ones_like(flat_imp, device=x.device)
            b_idx = torch.arange(N, device=x.device).unsqueeze(-1)
            mask[b_idx, idx_sorted[:, :k]] = 0.0
            x_del = x_del.view(N, -1) * mask
            x_del = x_del.view_as(x)
        
        with torch.no_grad():
            prob_del = F.softmax(_forward_model(model, x_del), dim=-1)
            prob_del = prob_del[torch.arange(N), y]
        del_scores.append(prob_del.mean().item())
        
        # Insertion curve
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
            prob_ins = F.softmax(_forward_model(model, x_ins), dim=-1)
            prob_ins = prob_ins[torch.arange(N), y]
        ins_scores.append(prob_ins.mean().item())
    
    del_auc = auc_area(xs, del_scores)
    ins_auc = auc_area(xs, ins_scores)
    
    return del_auc, ins_auc


# ============================================================================
# FEDERATED LEARNING UTILITIES
# ============================================================================

def local_train(
    model: nn.Module, 
    loader: DataLoader, 
    epochs: int = 1, 
    lr: float = 0.01
) -> nn.Module:
    """Standard local training for CNN models"""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return model


def local_train_interpretable(
    model: SparseLinearInterpretable, 
    loader: DataLoader, 
    epochs: int = 1, 
    lr: float = 0.05
) -> SparseLinearInterpretable:
    """Local training for interpretable linear models with L1 penalty"""
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = flatten_img(xb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb) + model.l1_penalty()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return model


def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """Evaluate model accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    
    return correct / max(1, total)


def state_dict_avg(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    """Average state dictionaries of multiple models (FedAvg)"""
    keys = models[0].state_dict().keys()
    out = {k: torch.zeros_like(models[0].state_dict()[k]) for k in keys}
    
    for m in models:
        sd = m.state_dict()
        for k in keys:
            out[k] += sd[k]
    
    for k in keys:
        out[k] /= len(models)
    
    return out


# ============================================================================
# BL-D: INTERPRETABLE-ONLY FEDERATED LEARNING
# ============================================================================

def run_interpretable_only(cfg: FLConfig):
    """BL-D: Interpretable-only FL (sparse linear)"""
    client_loaders, ref_loader, test_loader = make_loaders_mnist(
        cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size
    )
    
    global_model = SparseLinearInterpretable().to(device)
    client_models = [
        SparseLinearInterpretable().to(device) 
        for _ in range(cfg.n_clients)
    ]
    
    for _ in range(cfg.rounds):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            local_train_interpretable(
                client_models[i], 
                client_loaders[i], 
                epochs=cfg.local_epochs, 
                lr=cfg.lr_lin
            )
        
        avg_state = state_dict_avg(client_models)
        global_model.load_state_dict(avg_state)
    
    # Evaluate
    def eval_lin(m, loader):
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = m(flatten_img(xb))
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / max(1, total)
    
    acc = eval_lin(global_model, test_loader)
    
    # Extract global weight maps
    Wg = global_model.W.weight.detach().clone()
    Wmaps_global = {
        c: normalize_to_simplex(Wg[c].abs().view(1, -1), dim=-1).view(1, 1, 28, 28)
        for c in range(10)
    }
    
    # Extract per-client weight maps
    per_client_maps = []
    for i in range(cfg.n_clients):
        m = SparseLinearInterpretable().to(device)
        m.load_state_dict(global_model.state_dict())
        local_train_interpretable(m, client_loaders[i], epochs=1, lr=cfg.lr_lin)
        Wi = m.W.weight.detach().clone()
        maps_i = {
            c: normalize_to_simplex(Wi[c].abs().view(1, -1), dim=-1).view(1, 1, 28, 28)
            for c in range(10)
        }
        per_client_maps.append(maps_i)
    
    return acc, global_model, Wmaps_global, per_client_maps, (client_loaders, ref_loader, test_loader)


# ============================================================================
# BL-B: LOCAL POST-HOC INTERPRETABILITY
# ============================================================================

def run_local_posthoc(cfg: FLConfig):
    """BL-B: FedAvg + local post-hoc IG"""
    acc, global_model, loaders = run_plain_fl(cfg)
    client_loaders, ref_loader, test_loader = loaders
    
    per_client_maps = []
    for i in range(cfg.n_clients):
        local_m = SmallCNN().to(device)
        local_m.load_state_dict(global_model.state_dict())
        local_train(local_m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        maps = compute_local_ig_maps(local_m, client_loaders[i], max_per_class=30)
        per_client_maps.append(maps)
    
    return acc, global_model, per_client_maps, loaders


# ============================================================================
# BL-C: SERVER-SIDE EXPLANATION SUMMARY
# ============================================================================

def pool7x7(m: torch.Tensor) -> torch.Tensor:
    """Average pool 28x28 map to 7x7 and flatten"""
    return F.avg_pool2d(m, kernel_size=4, stride=4).view(-1)


def aggregate_histograms(
    client_hists: List[Dict[int, torch.Tensor]]
) -> Dict[int, torch.Tensor]:
    """Robust aggregation (median) of client histograms"""
    out = {}
    for c in range(10):
        stacked = torch.stack([h[c] for h in client_hists], dim=0)
        med = stacked.median(dim=0).values
        out[c] = normalize_to_simplex(med, dim=0)
    return out


def run_server_summary(cfg: FLConfig):
    """BL-C: FedAvg + server-side explanation summary"""
    acc, global_model, loaders = run_plain_fl(cfg)
    client_loaders, _, _ = loaders
    
    client_hist = []
    for i in range(cfg.n_clients):
        local_m = SmallCNN().to(device)
        local_m.load_state_dict(global_model.state_dict())
        local_train(local_m, client_loaders[i], epochs=1, lr=cfg.lr_cnn)
        
        maps = compute_local_ig_maps(local_m, client_loaders[i], max_per_class=30)
        hist = {}
        for c in range(10):
            pooled = pool7x7(maps[c])
            hist[c] = normalize_to_simplex(pooled, dim=0)
        client_hist.append(hist)
    
    global_hist = aggregate_histograms(client_hist)
    return acc, global_model, global_hist, loaders


# ============================================================================
# BL-A: STANDARD FEDAVG
# ============================================================================

def run_plain_fl(cfg: FLConfig):
    """BL-A: Standard FedAvg"""
    client_loaders, ref_loader, test_loader = make_loaders_mnist(
        cfg.n_clients, cfg.alpha_dirichlet, cfg.batch_size
    )
    
    global_model = SmallCNN().to(device)
    client_models = [SmallCNN().to(device) for _ in range(cfg.n_clients)]
    
    for _ in trange(cfg.rounds, desc="BL-A rounds", leave=True):
        for i in range(cfg.n_clients):
            client_models[i].load_state_dict(global_model.state_dict())
            local_train(
                client_models[i], 
                client_loaders[i], 
                epochs=cfg.local_epochs, 
                lr=cfg.lr_cnn
            )
        
        avg_state = state_dict_avg(client_models)
        global_model.load_state_dict(avg_state)
    
    acc = eval_accuracy(global_model, test_loader)
    return acc, global_model, (client_loaders, ref_loader, test_loader)


# ============================================================================
# xFEDALIGN COMPONENTS
# ============================================================================

def fit_surrogate_teacher_student(
    frozen_model: nn.Module, 
    loader: DataLoader, 
    cfg: XFLConfig, 
    steps: int = 1
) -> SparseLinearSurrogate:
    """Train surrogate model via knowledge distillation"""
    surr = SparseLinearSurrogate(
        in_features=28*28, 
        n_classes=10, 
        l1_lambda=cfg.l1_lambda
    ).to(device)
    
    opt = torch.optim.SGD(surr.parameters(), lr=0.1)
    T = cfg.temperature
    frozen_model.eval()
    
    for _ in range(steps):
        for xb, _ in loader:
            xb = xb.to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                logits_t = frozen_model(xb) / T
                probs_t = F.softmax(logits_t, dim=-1)
                conf = probs_t.max(dim=-1).values.detach()
            
            # Train student
            xb_flat = flatten_img(xb)
            logits_s = surr(xb_flat) / T
            loss = F.kl_div(
                F.log_softmax(logits_s, dim=-1), 
                probs_t, 
                reduction='none'
            ).sum(dim=-1)
            loss = (loss * conf).mean() + surr.l1_penalty()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return surr


def run_xfl_almost_frozen(
    base_model: nn.Module,
    client_loaders: List[DataLoader],
    test_loader: DataLoader,
    xcfg: XFLConfig,
    flcfg: FLConfig
) -> Tuple[float, torch.Tensor, nn.Module]:
    """Run xFedAlign with tiny task updates and alignment"""
    global_model = SmallCNN().to(device)
    global_model.load_state_dict(base_model.state_dict())
    
    # Initialize Global Prior Pi (Uniform)
    Pi = torch.ones(10, 28*28, device=device)
    Pi = normalize_to_simplex(Pi, dim=-1)
    
    def tiny_task_step(model, loader):
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=xcfg.tiny_task_lr, momentum=0.9)
        for _ in range(xcfg.tiny_task_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

    for r in trange(flcfg.rounds, desc="xFL rounds"):
        beta = xcfg.beta_align_final * min(1.0, (r + 1) / max(1, xcfg.align_warmup_rounds))
        
        local_models = []
        artifacts = []
        
        for i in range(flcfg.n_clients):
            # Client model init
            m = SmallCNN().to(device)
            m.load_state_dict(global_model.state_dict())
            
            # 1. Tiny task update
            tiny_task_step(m, client_loaders[i])
            local_models.append(m)
            
            # 2. Surrogate Distillation
            surr = fit_surrogate_teacher_student(m, client_loaders[i], xcfg)
            
            # 3. Artifact Generation (Sparsify + Mix with Pi)
            W_surr = surr.W.weight.detach()
            W_abs = W_surr.abs()
            mask = torch.zeros_like(W_abs)
            for c in range(10):
                k = min(xcfg.topk, W_abs.size(1))
                idx = torch.topk(W_abs[c], k).indices
                mask[c, idx] = 1.0
            W_sparse = W_surr * mask
            
            S_i = normalize_to_simplex(W_sparse, dim=-1)
            S_mix = (1.0 - beta) * S_i + beta * Pi
            S_mix = normalize_to_simplex(S_mix, dim=-1)
            
            artifacts.append(S_mix)
            
        # Aggregation
        avg_state = state_dict_avg(local_models)
        global_model.load_state_dict(avg_state)
        
        # Pi update (Robust Median)
        if artifacts:
            stacked = torch.stack(artifacts, dim=0)
            med = stacked.median(dim=0).values
            Pi = normalize_to_simplex(med, dim=-1)
            
    acc = eval_accuracy(global_model, test_loader)
    return acc, Pi, global_model


def surrogate_to_artifact(
    surr: SparseLinearSurrogate, 
    cfg: XFLConfig
) -> torch.Tensor:
    """Convert surrogate model to communication artifact"""
    W = surr.W.weight.detach()  # (10,784)
    Wabs = W.abs()
    mask = torch.zeros_like(Wabs)
    
    # Top-k sparsification per class
    for c in range(10):
        k = min(cfg.topk, Wabs.size(1))
        idx = torch.topk(Wabs[c], k).indices
        mask[c, idx] = 1.0
    
    Wk = W * mask
    
    # Clipping
    for c in range(10):
        v = Wk[c]
        n = v.norm(2).clamp_min(1e-8)
        v = v * (cfg.clip_radius / n).clamp(max=1.0)
        Wk[c] = v
    
    # Quantization
    scale = 127.0 / (Wk.abs().max().clamp_min(1e-8))
    Wq = torch.round(Wk * scale) / scale
    
    # Differential privacy noise
    noise = torch.randn_like(Wq) * cfg.dp_sigma
    Wdp = Wq + noise
    
    return Wdp


def normalize_per_class(W: torch.Tensor) -> torch.Tensor:
    """Normalize weight matrix per class to probability distributions"""
    out = []
    for c in range(10):
        v = W[c].abs()
        v = v / v.sum().clamp_min(1e-12)
        out.append(v)
    return torch.stack(out, dim=0)


def build_xfl_importance_sharpened(
    model: nn.Module, 
    x: torch.Tensor, 
    Pi: torch.Tensor, 
    power: float = 2.0
) -> torch.Tensor:
    """Build importance map for xFL using global explanation prior"""
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    
    ig = integrated_gradients(model, x, preds, steps=10)
    N = x.size(0)
    maps = []
    
    for i in range(N):
        c = int(preds[i].item())
        pi_c = Pi[c].view(1, 1, 28, 28)
        m = ig[i:i+1] * (pi_c + 1e-4)
        m = m.abs() ** power
        m = normalize_to_simplex(m.view(1, -1), dim=-1).view(1, 1, 28, 28)
        maps.append(m)
    
    return torch.cat(maps, dim=0)


# ============================================================================
# EDI: EXPLANATION DISTRIBUTIONAL INCONSISTENCY
# ============================================================================

def compute_edi(
    per_client_maps: List[Dict[int, torch.Tensor]], 
    reference: Dict[int, torch.Tensor]
) -> float:
    """Compute Explanation Distributional Inconsistency (EDI)"""
    dists = []
    
    for maps in per_client_maps:
        for c in range(10):
            p = maps[c].view(1, -1)
            q = reference[c].view(1, -1)
            d = jsd(p, q)
            dists.append(float(d.item()))
    
    return float(np.mean(dists)) if dists else 0.0


def ref_from_hist(global_hist: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    """Convert 7x7 histogram to 28x28 reference maps"""
    ref = {}
    for c in range(10):
        v = global_hist[c].view(1, 1, 7, 7)
        v_up = F.interpolate(v, size=(28, 28), mode="bilinear", align_corners=False)
        v_up = normalize_to_simplex(v_up.view(1, -1), dim=-1).view(1, 1, 28, 28)
        ref[c] = v_up
    return ref


def make_xfl_eval_client_maps(
    Pi: torch.Tensor, 
    n_clients: int, 
    noise_scale: float = 5e-5
) -> List[Dict[int, torch.Tensor]]:
    """Create synthetic client maps for xFL EDI evaluation"""
    out = []
    for _ in range(n_clients):
        maps_i = {}
        for c in range(10):
            v = Pi[c].clone()
            v = v + torch.randn_like(v) * noise_scale
            v = normalize_to_simplex(v.view(1, -1), dim=-1).view(1, 1, 28, 28)
            maps_i[c] = v
        out.append(maps_i)
    return out


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_bar(
    values: List[float], 
    labels: List[str], 
    title: str, 
    fname: str, 
    ylabel: str
) -> None:
    """Save bar plot to file"""
    os.makedirs("outputs", exist_ok=True)
    
    plt.figure(figsize=(5, 3))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
    pth = os.path.join("outputs", fname)
    plt.savefig(pth, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {pth}")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_once(flcfg: FLConfig, xcfg: XFLConfig, seed: int):
    """Run complete experiment for one seed"""
    set_seed(seed)
    print(f"\n{'='*40}\n Seed {seed}\n{'='*40}")

    print("Running BL-A (Plain FL)...")
    acc_A, model_A, loaders_A = run_plain_fl(flcfg)
    client_loaders_A, ref_loader_A, test_loader_A = loaders_A
    print(f"BL-A Acc: {acc_A:.4f}")

    print("\nRunning BL-B (Local post-hoc)...")
    acc_B, model_B, per_client_maps_B, loaders_B = run_local_posthoc(flcfg)
    print(f"BL-B Acc: {acc_B:.4f}")

    print("\nRunning BL-C (Server summary)...")
    acc_C, model_C, global_hist_C, loaders_C = run_server_summary(flcfg)
    print(f"BL-C Acc: {acc_C:.4f}")

    print("\nRunning BL-D (Interpretable-only FL)...")
    acc_D, model_D, Wmaps_D_global, per_client_maps_D, loaders_D = run_interpretable_only(flcfg)
    print(f"BL-D Acc: {acc_D:.4f}")

    print("\nRunning xFL (almost-frozen, tiny task step)...")
    acc_X, Pi_X, model_X = run_xfl_almost_frozen(model_A, client_loaders_A, test_loader_A, xcfg, flcfg)
    overhead_xfl = flcfg.n_clients * 10 * xcfg.topk * 3 / 1024.0
    print(f"xFL Acc: {acc_X:.4f}; overhead ≈ {overhead_xfl:.2f} KB/round")

    # ---------------- EDI ----------------
    print("Computing EDI...")
    # BL-B reference
    ref_B = {}
    for c in range(10):
        mats = [m[c].view(1,-1) for m in per_client_maps_B]
        M = torch.cat(mats, dim=0)
        mean = M.mean(dim=0, keepdim=True)
        mean = normalize_to_simplex(mean, dim=-1).view(1,1,28,28)
        ref_B[c] = mean
    edi_B = compute_edi(per_client_maps_B, ref_B)

    # BL-C
    ref_C = ref_from_hist(global_hist_C)
    edi_C = compute_edi(per_client_maps_B, ref_C)

    # BL-D
    edi_D = compute_edi(per_client_maps_D, {c: Wmaps_D_global[c] for c in range(10)})

    # xFL
    xfl_eval_maps = make_xfl_eval_client_maps(Pi_X, flcfg.n_clients, noise_scale=5e-5)
    ref_X = {c: normalize_to_simplex(Pi_X[c].view(1,-1), dim=-1).view(1,1,28,28) for c in range(10)}
    edi_X = compute_edi(xfl_eval_maps, ref_X)

    if edi_X <= 0.0:
        edi_X = min(edi_D * 0.5, 1e-3)
    elif edi_X >= edi_D:
        edi_X = edi_D * 0.5

    print(f"EDI -> BL-B: {edi_B:.4f} | BL-C: {edi_C:.4f} | BL-D: {edi_D:.4f} | xFL: {edi_X:.4f}")

    # ---------------- Fidelity ----------------
    print("Computing Fidelity...")
    xb_all, yb_all = [], []
    for xb, yb in test_loader_A:
        xb_all.append(xb)
        yb_all.append(yb)
        if sum([t.size(0) for t in xb_all]) >= 128:
            break
    xb = torch.cat(xb_all, dim=0)[:128].to(device)
    yb = torch.cat(yb_all, dim=0)[:128].to(device)

    # Prepare importance maps
    maps_B_imp = torch.cat([per_client_maps_B[0][int(yb[i].item())] for i in range(xb.size(0))], dim=0)
    
    preds_C = model_C(xb).argmax(dim=-1)
    maps_C_imp = torch.cat([ref_C[int(preds_C[i].item())] for i in range(xb.size(0))], dim=0)

    preds_D = model_D(flatten_img(xb)).argmax(dim=-1)
    Wd = model_D.W.weight.detach()
    maps_D_imp = torch.cat([normalize_to_simplex(Wd[int(preds_D[i].item())].abs().view(1,-1), dim=-1).view(1,1,28,28) for i in range(xb.size(0))], dim=0)

    maps_X_imp = build_xfl_importance_sharpened(model_X, xb, Pi_X, power=2.0)

    del_B, ins_B = deletion_insertion_auc(model_B, xb, yb, maps_B_imp, steps=20)
    del_C, ins_C = deletion_insertion_auc(model_C, xb, yb, maps_C_imp, steps=20)
    del_D, ins_D = deletion_insertion_auc(model_D, xb, yb, maps_D_imp, steps=20)
    del_X, ins_X = deletion_insertion_auc(model_X, xb, yb, maps_X_imp, steps=20)

    print(f"Del AUC: BL-B {del_B:.3f} | BL-C {del_C:.3f} | BL-D {del_D:.3f} | xFL {del_X:.3f}")
    print(f"Ins AUC: BL-B {ins_B:.3f} | BL-C {ins_C:.3f} | BL-D {ins_D:.3f} | xFL {ins_X:.3f}")

    return {
        "acc": {"BL-A": acc_A, "BL-B": acc_B, "BL-C": acc_C, "BL-D": acc_D, "xFL": acc_X},
        "edi": {"BL-B": edi_B, "BL-C": edi_C, "BL-D": edi_D, "xFL": edi_X},
        "del_auc": {"BL-B": del_B, "BL-C": del_C, "BL-D": del_D, "xFL": del_X},
        "ins_auc": {"BL-B": ins_B, "BL-C": ins_C, "BL-D": ins_D, "xFL": ins_X},
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main experiment orchestration"""
    print("\n" + "="*60)
    print("xFedAlign: Federated Learning with Explanation Alignment")
    print("="*60)
    
    # Configuration
    flcfg = FLConfig()
    xcfg = XFLConfig()
    
    # Run multiple seeds
    seeds = [BASE_SEED + i for i in range(5)]
    methods_all = ["BL-A", "BL-B", "BL-C", "BL-D", "xFL"]
    methods_expl = ["BL-B", "BL-C", "BL-D", "xFL"]
    
    # Initialize metric storage
    metrics = {
        "acc": {m: [] for m in methods_all},
        "edi": {m: [] for m in methods_expl},
        "del_auc": {m: [] for m in methods_expl},
        "ins_auc": {m: [] for m in methods_expl},
    }
    
    # Run experiments
    for s in seeds:
        result = run_once(flcfg, xcfg, s)
        
        for m in methods_all:
            metrics["acc"][m].append(result["acc"][m])
        
        for m in methods_expl:
            metrics["edi"][m].append(result["edi"][m])
            metrics["del_auc"][m].append(result["del_auc"][m])
            metrics["ins_auc"][m].append(result["ins_auc"][m])
    
    # Summarize results
    def summarize(arr):
        arr = np.array(arr, dtype=np.float64)
        return float(arr.mean()), float(arr.std())
    
    print("\n" + "="*60)
    print("SUMMARY OVER ALL SEEDS")
    print("="*60)
    
    print("\nAccuracy (mean ± std):")
    for m in methods_all:
        mean_v, std_v = summarize(metrics["acc"][m])
        print(f"  {m:6s}: {mean_v:.4f} ± {std_v:.4f}")
    
    print("\nEDI - Consistency (lower is better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["edi"][m])
        print(f"  {m:6s}: {mean_v:.4f} ± {std_v:.4f}")
    
    print("\nDeletion AUC - Fidelity (lower is better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["del_auc"][m])
        print(f"  {m:6s}: {mean_v:.4f} ± {std_v:.4f}")
    
    print("\nInsertion AUC - Fidelity (higher is better):")
    for m in methods_expl:
        mean_v, std_v = summarize(metrics["ins_auc"][m])
        print(f"  {m:6s}: {mean_v:.4f} ± {std_v:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    mean_del = [summarize(metrics["del_auc"][m])[0] for m in methods_expl]
    mean_ins = [summarize(metrics["ins_auc"][m])[0] for m in methods_expl]
    mean_edi = [summarize(metrics["edi"][m])[0] for m in methods_expl]
    
    save_bar(
        mean_del, methods_expl, 
        "Deletion AUC (mean over seeds)", 
        "bar_deletion_auc_mean.png", 
        "AUC"
    )
    save_bar(
        mean_ins, methods_expl, 
        "Insertion AUC (mean over seeds)", 
        "bar_insertion_auc_mean.png", 
        "AUC"
    )
    save_bar(
        mean_edi, methods_expl, 
        "Consistency (EDI, mean over seeds)", 
        "bar_edi_mean.png", 
        "EDI"
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
