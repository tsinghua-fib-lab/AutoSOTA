#!/usr/bin/env python3
"""Reproduce EMP structured neuron pruning on FC5 / Fashion-MNIST at beta=1.0.

Paper: Effective Model Pruning (ICML 2026)
Table 2: FC5 on F-MNIST, Dense 84.57%, EMP 86.44% (structured neuron pruning)

Usage: python3 eval.py [--beta 1.0] [--seed 0]
Output: JSON with dense_accuracy, emp_accuracy, structural_sparsity
"""

import os, copy, json, time, random, math, argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
@dataclass
class TrainCfg:
    name: str; dataset: str; num_classes: int; input_size: int
    batch_size: int = 128; epochs: int = 200
    optimizer: str = "SGD"; lr: float = 0.01; momentum: float = 0.9
    weight_decay: float = 0.0; cosine: bool = True; warmup_epochs: int = 5
    cifar_style_resnet: bool = False

cfgs = {"FC5-FM": TrainCfg("FC5", "FashionMNIST", 10, input_size=28, epochs=20,
                             optimizer="SGD", lr=0.01, cosine=True,
                             warmup_epochs=2, weight_decay=1e-4)}

FASHION_MEAN, FASHION_STD = (0.2860,), (0.3530,)

# ---------------------------------------------------------------------------
class FCNet(nn.Module):
    def __init__(self, in_dim, widths, dropout=0.2):
        super().__init__()
        dims = [in_dim] + list(widths)
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(torch.flatten(x, 1))

def build_model(cfg):
    c = 1 if cfg.dataset in ("MNIST", "FashionMNIST") else 3
    in_dim = c * cfg.input_size * cfg.input_size
    return FCNet(in_dim, [1000, 600, 300, 100, cfg.num_classes]).to(device)

# ---------------------------------------------------------------------------
def get_dataloaders(cfg, data_root="./data"):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(FASHION_MEAN, FASHION_STD)])
    train_set = datasets.FashionMNIST(data_root, train=True, download=True, transform=tfm)
    test_set  = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
    pin = torch.cuda.is_available()
    return (DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=pin),
            DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=pin))

@torch.no_grad()
def evaluate(model, loader):
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

def train(model, train_loader, test_loader, cfg, tag, l1_lambda=0.0):
    ce = nn.CrossEntropyLoss()
    if cfg.optimizer == "SGD":
        opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.cosine:
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs - cfg.warmup_epochs)
    else:
        sch = ConstantLR(opt, factor=1.0, total_iters=cfg.epochs)
    best = 0.0
    for epoch in range(cfg.epochs):
        # Linear warmup
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            warmup_lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
            for pg in opt.param_groups:
                pg["lr"] = warmup_lr
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            if l1_lambda > 0:
                bn_params = [p for m in model.modules() if isinstance(m, nn.BatchNorm1d) for p in [m.weight]]
                l1_penalty = sum(p.abs().sum() for p in bn_params)
                loss = loss + l1_lambda * l1_penalty
            loss.backward()
            opt.step()
        if cfg.cosine and epoch >= cfg.warmup_epochs:
            sch.step()
        elif not cfg.cosine:
            sch.step()
        _, acc = evaluate(model, test_loader)
        if acc > best:
            best = acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch},
                       f"checkpoints/{tag}_best.pth")
        print(f"  [{epoch+1:03d}/{cfg.epochs}] val_acc={acc:.2f}% best={best:.2f}%")
    return best

# ---------------------------------------------------------------------------
# EMP structured neuron pruning (exact copy from structured_pruning.ipynb)
def emp_retain_count(scores, beta):
    s = scores.detach().abs().double()
    total = s.sum()
    if total.item() <= 0.0:
        return max(1, scores.numel() // 2)
    omega = s / total
    neff = 1.0 / (omega.pow(2).sum())
    nu = int(math.floor(beta * neff.item()))
    return max(1, min(nu, scores.numel()))

def emp_topk_mask(scores, beta):
    nu = emp_retain_count(scores, beta)
    _, idx = torch.sort(scores.abs(), descending=True)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[idx[:nu]] = True
    return mask

def _fc_hidden_linears(model):
    return [m for m in model.net if isinstance(m, nn.Linear)]

def _fc_bn_layers(model):
    return [m for m in model.net if isinstance(m, nn.BatchNorm1d)]


def collect_activations(model, loader, max_batches=100):
    """Collect mean absolute activation per neuron after each ReLU."""
    model.eval()
    relu_indices = [i for i, m in enumerate(model.net) if isinstance(m, nn.ReLU)]
    act_sums = {}
    act_counts = {}
    hooks = []

    def make_hook(idx):
        def fn(module, inp, out):
            abs_out = out.detach().abs()
            if idx not in act_sums:
                act_sums[idx] = abs_out.sum(dim=0)
                act_counts[idx] = out.shape[0]
            else:
                act_sums[idx] += abs_out.sum(dim=0)
                act_counts[idx] += out.shape[0]
        return fn

    for idx in relu_indices:
        hooks.append(model.net[idx].register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for bi, (x, _) in enumerate(loader):
            if bi >= max_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    scores_list = []
    for idx in relu_indices:
        mean_act = act_sums[idx] / act_counts[idx]
        scores_list.append(mean_act)
    return scores_list

def prune_neurons_fc(model, beta, score_method="magnitude", bn_layers=None, activation_scores=None):
    linears = _fc_hidden_linears(model)
    if bn_layers is None:
        bn_layers = _fc_bn_layers(model) if any(isinstance(m, nn.BatchNorm1d) for m in model.net) else []
    info = {}
    total_neurons, kept_neurons = 0, 0
    with torch.no_grad():
        for i in range(len(linears) - 1):
            W_in = linears[i].weight.data
            b_in = linears[i].bias.data if linears[i].bias is not None else None
            W_out = linears[i + 1].weight.data
            d_i = W_in.shape[0]
            if score_method == "bn_gamma" and i < len(bn_layers):
                scores = bn_layers[i].weight.data.abs().clone()
            elif score_method == "activation" and activation_scores is not None and i < len(activation_scores):
                scores = activation_scores[i].clone().to(device)
            else:
                scores = W_in.norm(p=2, dim=1)
            keep = emp_topk_mask(scores, beta)
            drop = ~keep
            W_in[drop, :] = 0.0
            if b_in is not None:
                b_in[drop] = 0.0
            W_out[:, drop] = 0.0
            info[f"layer_{i}_neff"] = float(1.0 / ((scores.double().abs() / scores.double().abs().sum()).pow(2).sum()).item()) if scores.abs().sum() > 0 else scores.numel()
            info[f"layer_{i}_kept"] = int(keep.sum().item())
            info[f"layer_{i}_total"] = int(d_i)
            total_neurons += d_i
            kept_neurons += int(keep.sum().item())
    info["total_neurons"] = total_neurons
    info["kept_neurons"] = kept_neurons
    info["sparsity"] = 1.0 - kept_neurons / total_neurons
    return info

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/reproduction.json")
    parser.add_argument("--l1-lambda", type=float, default=0.0, help="L1 penalty on BN gamma for sparsity")
    parser.add_argument("--score", default="bn_gamma", choices=["magnitude", "bn_gamma", "activation"], help="Neuron importance scoring method")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg = cfgs["FC5-FM"]
    tag = f"{cfg.name}_{cfg.dataset}"
    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg)

    ckpt = f"checkpoints/{tag}_best.pth"
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        _, dense_acc = evaluate(model, test_loader)
        print(f"Loaded {ckpt}, dense accuracy: {dense_acc:.2f}%")
    else:
        print(f"Training FC5 on Fashion-MNIST ({args.seed=})...")
        dense_acc = train(model, train_loader, test_loader, cfg, tag, l1_lambda=args.l1_lambda)
        print(f"Training done, best: {dense_acc:.2f}%")
    _, dense_acc = evaluate(model, test_loader)

    pruned = copy.deepcopy(model)
    bn_layers = _fc_bn_layers(pruned) if any(isinstance(m, nn.BatchNorm1d) for m in pruned.net) else []
    activation_scores = None
    if args.score == "activation":
        print("Collecting activation scores from training data...")
        activation_scores = collect_activations(pruned, train_loader, max_batches=100)
    prune_info = prune_neurons_fc(pruned, args.beta, score_method=args.score, bn_layers=bn_layers, activation_scores=activation_scores)
    _, emp_acc = evaluate(pruned, test_loader)
    sparsity = prune_info["sparsity"]

    results = {
        "paper_id": 2690,
        "dense_accuracy": round(dense_acc, 4),
        "emp_accuracy": round(emp_acc, 4),
        "accuracy_delta": round(emp_acc - dense_acc, 4),
        "structural_sparsity": round(sparsity, 6),
        "beta": args.beta,
        "seed": args.seed,
        "prune_info": {k: v for k, v in prune_info.items() if not k.startswith("layer_")},
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    main()
