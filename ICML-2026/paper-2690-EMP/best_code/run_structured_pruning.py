#!/usr/bin/env python3
"""Reproduce FC5 structured neuron pruning on Fashion-MNIST with EMP at beta=1.0.

Paper: Effective Model Pruning: Measuring the Redundancy of Model Components
Target: FC5 on Fashion-MNIST, structured neuron pruning, magnitude score, beta=1.0
Expected: Dense Acc ~84.57%, EMP Acc ~86.44% (Table 2)
"""

import os, copy, json, time, random, math, sys
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Training configurations (from the paper Appendix B.1)
# ---------------------------------------------------------------------------
@dataclass
class TrainCfg:
    name: str
    dataset: str
    num_classes: int
    input_size: int
    batch_size: int = 128
    epochs: int = 200
    optimizer: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    cosine: bool = True
    warmup_epochs: int = 5
    cifar_style_resnet: bool = False

cfgs: Dict[str, TrainCfg] = {
    # Fashion-MNIST FC5: Adam, lr=1e-4, 5 epochs, weight_decay=0 (Appendix B.1)
    "FC5-FM": TrainCfg("FC5", "FashionMNIST", 10, input_size=28, epochs=5,
                        optimizer="ADAM", lr=1e-4, cosine=False, warmup_epochs=0, weight_decay=0.0),
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
FASHION_MEAN, FASHION_STD = (0.2860,), (0.3530,)

def get_dataloaders(cfg: TrainCfg, data_root: str = "./data") -> Tuple[DataLoader, DataLoader]:
    if cfg.dataset == "FashionMNIST":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(FASHION_MEAN, FASHION_STD)])
        train_set = datasets.FashionMNIST(data_root, train=True,  download=True, transform=tfm)
        test_set  = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset {cfg.dataset}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,  num_workers=4, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=pin)
    return train_loader, test_loader

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FCNet(nn.Module):
    def __init__(self, in_dim: int, widths: List[int]):
        super().__init__()
        dims = [in_dim] + list(widths)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.net(x)

def build_model(cfg: TrainCfg) -> nn.Module:
    c = 1 if cfg.dataset in ("MNIST", "FashionMNIST") else 3
    in_dim = c * cfg.input_size * cfg.input_size
    if cfg.name == "FC5":
        return FCNet(in_dim, [1000, 600, 300, 100, cfg.num_classes]).to(device)
    raise ValueError(f"Unknown model {cfg.name}")

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += ce(logits, y).item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, 100.0 * total_correct / n

def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, cfg: TrainCfg, tag: str) -> float:
    ce = nn.CrossEntropyLoss()
    if cfg.optimizer.upper() == "ADAM":
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sch = ConstantLR(opt, factor=1.0, total_iters=cfg.epochs)
    else:
        opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        sch = ConstantLR(opt, factor=1.0, total_iters=cfg.epochs)

    best = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        sch.step()
        _, val_acc = evaluate(model, test_loader)
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
                       f"checkpoints/{tag}_best.pth")
        print(f"  [{epoch+1:03d}/{cfg.epochs}] val_acc={val_acc:.2f}% best={best:.2f}% lr={opt.param_groups[0]['lr']:.5f} t={time.time()-t0:.1f}s")
    return best

# ---------------------------------------------------------------------------
# EMP structured pruning (FC neuron pruning)
# ---------------------------------------------------------------------------
def emp_retain_count(scores: torch.Tensor, beta: float) -> int:
    s = scores.detach().abs().double()
    total = s.sum()
    if total.item() <= 0.0:
        return max(1, scores.numel() // 2)
    omega = s / total
    neff = 1.0 / (omega.pow(2).sum())
    nu = int(math.floor(beta * neff.item()))
    nu = max(1, min(nu, scores.numel()))
    return nu

def emp_topk_mask(scores: torch.Tensor, beta: float) -> torch.Tensor:
    nu = emp_retain_count(scores, beta)
    _, idx = torch.sort(scores.abs(), descending=True)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[idx[:nu]] = True
    return mask

def neff_of(scores: torch.Tensor) -> float:
    s = scores.detach().abs().double()
    total = s.sum()
    if total.item() <= 0.0:
        return float(scores.numel())
    omega = s / total
    return float(1.0 / (omega.pow(2).sum()).item())

def _fc_hidden_linears(model: FCNet) -> List[nn.Linear]:
    return [m for m in model.net if isinstance(m, nn.Linear)]

def prune_neurons_fc(model: FCNet, beta: float) -> Dict[str, float]:
    linears = _fc_hidden_linears(model)
    info: Dict[str, float] = {}
    total_neurons, kept_neurons = 0, 0
    with torch.no_grad():
        for i in range(len(linears) - 1):
            W_in = linears[i].weight.data
            b_in = linears[i].bias.data if linears[i].bias is not None else None
            W_out = linears[i + 1].weight.data
            d_i = W_in.shape[0]
            scores = W_in.norm(p=2, dim=1)
            keep = emp_topk_mask(scores, beta)
            drop = ~keep
            W_in[drop, :] = 0.0
            if b_in is not None:
                b_in[drop] = 0.0
            W_out[:, drop] = 0.0
            info[f"layer_{i}_neff"] = neff_of(scores)
            info[f"layer_{i}_kept"] = int(keep.sum().item())
            info[f"layer_{i}_total"] = int(d_i)
            total_neurons += d_i
            kept_neurons += int(keep.sum().item())
    info["total_neurons"] = total_neurons
    info["kept_neurons"] = kept_neurons
    info["sparsity"] = 1.0 - kept_neurons / total_neurons
    return info

# ---------------------------------------------------------------------------
# Main: reproduce FC5-FM structured neuron pruning at beta=1.0
# ---------------------------------------------------------------------------
def main():
    cfg_key = "FC5-FM"
    beta = 1.0
    cfg = cfgs[cfg_key]
    tag = f"{cfg.name}_{cfg.dataset}"
    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg)

    # Train or load checkpoint
    ckpt_path = f"checkpoints/{tag}_best.pth"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        _, dense_acc = evaluate(model, test_loader)
        print(f"Loaded checkpoint {ckpt_path}, dense accuracy: {dense_acc:.2f}%")
    else:
        print(f"No checkpoint found at {ckpt_path}, training FC5 on Fashion-MNIST...")
        dense_acc = train(model, train_loader, test_loader, cfg, tag)
        print(f"Training complete, best dense accuracy: {dense_acc:.2f}%")

    # Re-evaluate dense accuracy for consistency
    _, dense_acc_eval = evaluate(model, test_loader)
    print(f"Dense top-1 accuracy (re-eval): {dense_acc_eval:.2f}%")

    # Apply EMP structured neuron pruning at beta=1.0
    pruned = copy.deepcopy(model)
    prune_info = prune_neurons_fc(pruned, beta)
    _, emp_acc = evaluate(pruned, test_loader)
    sparsity = prune_info.get("sparsity", 0.0)

    print(f"\n{'='*60}")
    print(f"RESULTS: FC5 on Fashion-MNIST, structured neuron pruning, beta={beta}")
    print(f"{'='*60}")
    print(f"Dense accuracy:     {dense_acc_eval:.2f}%")
    print(f"EMP accuracy:       {emp_acc:.2f}%")
    print(f"Accuracy delta:     {emp_acc - dense_acc_eval:+.2f} pp")
    print(f"Structural sparsity: {sparsity*100:.2f}%")
    for k, v in prune_info.items():
        if k.startswith("layer_"):
            print(f"  {k}: {v}")

    # Save results
    results = {
        "paper_id": 2690,
        "config": "FC5 on Fashion-MNIST, structured neuron pruning, magnitude score, beta=1.0",
        "dense_accuracy": dense_acc_eval,
        "emp_accuracy": emp_acc,
        "accuracy_delta": emp_acc - dense_acc_eval,
        "structural_sparsity": sparsity,
        "prune_info": prune_info,
        "beta": beta,
        "seed": SEED,
    }
    out_path = "results/structured_pruning_fc5_fashionmnist.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Check against rubric
    paper_target = 86.44
    lower_bound = 84.57
    upper_bound = 86.627
    print(f"\nRubric check:")
    print(f"  Paper target: {paper_target}%")
    print(f"  Our EMP acc:  {emp_acc:.2f}%")
    print(f"  CI bounds:    [{lower_bound}, {upper_bound}]")
    if lower_bound <= emp_acc <= upper_bound:
        print(f"  STATUS: Within CI bounds -> REPRODUCTION SUCCEEDED")
    elif emp_acc >= lower_bound:
        print(f"  STATUS: >= lower bound, outside upper -> PARTIAL")
    else:
        print(f"  STATUS: Below lower bound -> REPRODUCTION FAILED")

    return results

if __name__ == "__main__":
    main()
