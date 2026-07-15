"""
Time Comparison for xFedAlign vs Baselines
Measures Training Time, Explanation Generation Time, and Communication Overhead.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337

@dataclass
class Config:
    n_clients: int = 8
    rounds: int = 15
    local_epochs: int = 1
    batch_size: int = 64
    lr_cnn: float = 0.01
    lr_lin: float = 0.05
    # xFL params
    topk: int = 128
    l1_lambda: float = 1e-4
    temp: float = 3.0

# ============================================================================
# MODELS & DATA
# ============================================================================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Added padding=1 to match dimensions of 32*14*14
        self.c1 = nn.Conv2d(1, 16, 3, 1, padding=1); self.c2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2); self.fc1 = nn.Linear(32*14*14, 64); self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = self.pool(F.relu(self.c2(x)))
        x = x.view(x.size(0), -1); x = F.relu(self.fc1(x)); return self.fc2(x)

class SparseLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(784, 10)
    def forward(self, x): return self.W(x.view(x.size(0), -1))

def get_loaders(n_clients):
    tfm = transforms.ToTensor()
    train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    idxs = np.arange(len(train)); np.random.shuffle(idxs)
    splits = np.array_split(idxs, n_clients)
    return [DataLoader(Subset(train, s), batch_size=64, shuffle=True) for s in splits]

# ============================================================================
# HELPERS
# ============================================================================
def train_step(model, loader, opt):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()

def train_step_sparse(model, loader, opt, l1=1e-4):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x), y) + l1 * model.W.weight.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()

def fedavg_agg(models):
    sd = models[0].state_dict()
    for k in sd:
        sd[k] = torch.stack([m.state_dict()[k] for m in models]).mean(0)
    return sd

def integrated_gradients(model, loader):
    # Simplified IG for timing (1 batch per client)
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    x.requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return x.grad.abs().detach()

def get_model_stats(model, extra_payload_bytes=0):
    """Returns (param_count, comm_overhead_kb)"""
    params = sum(p.numel() for p in model.parameters())
    model_bytes = params * 4 # float32
    total_kb = (model_bytes + extra_payload_bytes) / 1024
    return params, total_kb

# ============================================================================
# RUNNERS
# ============================================================================
def run_fedavg_base(cfg, loaders):
    # Common training for BL-A, BL-B, BL-C
    g = SmallCNN().to(DEVICE)
    clients = [SmallCNN().to(DEVICE) for _ in range(cfg.n_clients)]
    
    t0 = time.time()
    for _ in range(cfg.rounds):
        for i in range(cfg.n_clients):
            clients[i].load_state_dict(g.state_dict())
            opt = torch.optim.SGD(clients[i].parameters(), lr=cfg.lr_cnn)
            train_step(clients[i], loaders[i], opt)
        g.load_state_dict(fedavg_agg(clients))
    train_time = time.time() - t0
    return g, clients, train_time

def measure_bl_a(cfg, loaders):
    g, _, t_train = run_fedavg_base(cfg, loaders)
    params, comm = get_model_stats(g)
    return t_train, 0.0, params, comm

def measure_bl_b(cfg, loaders):
    g, clients, t_train = run_fedavg_base(cfg, loaders)
    
    t0 = time.time()
    # Local Post-hoc: Compute IG for each client
    for i in range(cfg.n_clients):
        integrated_gradients(clients[i], loaders[i])
    t_expl = time.time() - t0
    
    params, comm = get_model_stats(g) # No extra comm for local XAI
    return t_train, t_expl, params, comm

def measure_bl_c(cfg, loaders):
    g, clients, t_train = run_fedavg_base(cfg, loaders)
    
    t0 = time.time()
    # Server Summary: Compute IG + Aggregate
    maps = []
    for i in range(cfg.n_clients):
        maps.append(integrated_gradients(clients[i], loaders[i]))
    # Aggregation (simulate processing)
    _ = torch.stack(maps).mean(0)
    t_expl = time.time() - t0
    
    # BL-C sends 7x7 histograms per class (10 classes) -> 490 floats
    hist_bytes = 10 * 7 * 7 * 4
    params, comm = get_model_stats(g, extra_payload_bytes=hist_bytes)
    return t_train, t_expl, params, comm

def measure_bl_d(cfg, loaders):
    g = SparseLinear().to(DEVICE)
    clients = [SparseLinear().to(DEVICE) for _ in range(cfg.n_clients)]
    
    t0 = time.time()
    for _ in range(cfg.rounds):
        for i in range(cfg.n_clients):
            clients[i].load_state_dict(g.state_dict())
            opt = torch.optim.SGD(clients[i].parameters(), lr=cfg.lr_lin)
            train_step_sparse(clients[i], loaders[i], opt)
        g.load_state_dict(fedavg_agg(clients))
    t_train = time.time() - t0
    
    params, comm = get_model_stats(g)
    return t_train, 0.001, params, comm

def measure_xfl(cfg, loaders):
    g = SmallCNN().to(DEVICE)
    clients = [SmallCNN().to(DEVICE) for _ in range(cfg.n_clients)]
    Pi = torch.ones(10, 784).to(DEVICE)
    
    t0 = time.time()
    for r in range(cfg.rounds):
        surrogates = []
        for i in range(cfg.n_clients):
            # 1. Task update
            clients[i].load_state_dict(g.state_dict())
            opt = torch.optim.SGD(clients[i].parameters(), lr=cfg.lr_cnn)
            train_step(clients[i], loaders[i], opt)
            
            # 2. Alignment (Surrogate Distillation)
            surr = SparseLinear().to(DEVICE)
            s_opt = torch.optim.SGD(surr.parameters(), lr=0.1)
            # Distill (1 epoch approx)
            clients[i].eval()
            for x, _ in loaders[i]:
                x = x.to(DEVICE)
                with torch.no_grad(): t_logits = clients[i](x) / cfg.temp
                s_logits = surr(x) / cfg.temp
                loss = F.kl_div(F.log_softmax(s_logits, -1), F.softmax(t_logits, -1), reduction='batchmean')
                loss += cfg.l1_lambda * surr.W.weight.abs().sum()
                s_opt.zero_grad(); loss.backward(); s_opt.step()
            
            # 3. Artifact generation (Top-k + Mix)
            W = surr.W.weight.detach()
            # Simulate top-k and mixing with Pi
            mask = torch.zeros_like(W)
            vals, idx = torch.topk(W.abs(), cfg.topk)
            mask.scatter_(1, idx, 1.0)
            W_sparse = W * mask
            surrogates.append(W_sparse)

        # Server Aggregation
        g.load_state_dict(fedavg_agg(clients))
        Pi = torch.stack(surrogates).mean(0)
        
    t_train = time.time() - t0
    
    # xFL sends sparse artifact: 10 classes * topk * (float32 val + int32 idx)
    artifact_bytes = 10 * cfg.topk * (4 + 4)
    params, comm = get_model_stats(g, extra_payload_bytes=artifact_bytes)
    return t_train, 0.0, params, comm

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = Config()
    loaders = get_loaders(cfg.n_clients)
    
    print(f"Running Time Comparison on {DEVICE}...")
    print(f"Settings: {cfg.n_clients} clients, {cfg.rounds} rounds, {cfg.batch_size} batch size.\n")
    
    results = []
    
    print("Measuring BL-A (FedAvg)...")
    results.append(("BL-A", *measure_bl_a(cfg, loaders)))
    
    print("Measuring BL-B (Local XAI)...")
    results.append(("BL-B", *measure_bl_b(cfg, loaders)))
    
    print("Measuring BL-C (FedAttr)...")
    results.append(("BL-C", *measure_bl_c(cfg, loaders)))
    
    print("Measuring BL-D (Fed-XAI)...")
    results.append(("BL-D", *measure_bl_d(cfg, loaders)))
    
    print("Measuring xFL (xFedAlign)...")
    results.append(("xFL", *measure_xfl(cfg, loaders)))
    
    print("\n" + "="*95)
    print(f"{'Method':<10} | {'Train(s)':<10} | {'Expl(s)':<10} | {'Total(s)':<10} | {'Params':<10} | {'Comm/Rnd(KB)':<15}")
    print("-" * 95)
    for name, t_train, t_expl, params, comm in results:
        total = t_train + t_expl
        print(f"{name:<10} | {t_train:<10.2f} | {t_expl:<10.2f} | {total:<10.2f} | {params:<10} | {comm:<15.2f}")
    print("="*95)
    print("\nNotes:")
    print(" - Comm/Rnd: Estimated upload size per client per round (Model + Artifacts).")
    print(" - xFL explanation cost is amortized into training (surrogate distillation).")
    print(" - BL-D uses a sparse linear model, hence low params and comms.")
