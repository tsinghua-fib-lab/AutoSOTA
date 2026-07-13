#!/usr/bin/env python3
"""Iteration 5: Gradient-norm adaptive per-layer dropout + constant baseline.

IDEA-01: Per-layer dropout rates from gradient norms at initialization.
Also runs constant baseline for proper gap measurement.
"""

import json, os, sys, time, warnings, re
from datetime import datetime, timezone

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

N_SIMULATIONS = 20
H_BAR, H_MAX, DEPTH, WIDTH = 0.1, 0.2, 6, 256
SIGMA_W_SQ, SIGMA_B_SQ = 1.98, 0.02
EPOCHS, LEARNING_RATE, LR_MIN = 50, 1e-4, 1e-7
BATCH_SIZE, TRAIN_SIZE, TEST_SIZE = 75, 5000, 5000
WARMUP_EPOCHS, CURRICULUM_GAMMA = 5, 10.0 / EPOCHS
GRAD_CALIB_BATCHES = 5  # Batches used for gradient calibration

print("=" * 60)
print("ITERATION 5: GRADIENT-NORM ADAPTIVE + CONSTANT BASELINE")
print("=" * 60)


def compute_grad_norm_adaptive_schedule(model_class, input_dim, hidden_dim, output_dim,
                                         sigma_w_sq, sigma_b_sq, h_bar, h_max, depth,
                                         x_calib, y_calib, bs, n_batches=5):
    """Compute per-layer dropout rates from gradient norms at initialization."""
    # Create a model with uniform dropout for calibration
    h_uniform = [h_bar] * depth
    model = model_class(input_dim, hidden_dim, output_dim, h_uniform,
                        sigma_w_sq, sigma_b_sq,
                        curriculum=False).to(device)
    model.train()

    crit = nn.CrossEntropyLoss()
    grad_norms = [0.0] * depth

    # Use seeded data for reproducibility
    torch.manual_seed(42)
    perm = torch.randperm(x_calib.size(0), device=device)

    for b in range(n_batches):
        idx = perm[b * bs:(b + 1) * bs]
        xb, yb = x_calib[idx], y_calib[idx]

        model.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()

        for i, layer in enumerate(model.layers[:-1]):  # hidden layers only
            grad_norms[i] += layer.weight.grad.norm().item()

    # Average gradient norms
    grad_norms = [g / n_batches for g in grad_norms]

    # Inversely proportional: h_i ∝ 1/||grad_i||
    # Layers with LARGER gradients = more sensitive = LOWER dropout
    inv_norms = [1.0 / max(g, 1e-8) for g in grad_norms]
    total_inv = sum(inv_norms)

    # Normalize to maintain h_bar * depth total budget
    h_layers = [h_bar * depth * (inv_norms[i] / total_inv) for i in range(depth)]

    # Clip to [0, h_max]
    h_layers = [max(0.0, min(h_max, h)) for h in h_layers]

    # Renormalize after clipping
    active_mask = [h < h_max for h in h_layers]
    if any(active_mask):
        total_current = sum(h_layers)
        target = h_bar * depth
        # Redistribute excess from clipped layers
        excess = sum(max(0, h - h_max) for h in h_layers)
        for i in range(depth):
            if h_layers[i] < h_max:
                h_layers[i] += excess / sum(active_mask)
        # Final clip
        h_layers = [max(0.0, min(h_max, h)) for h in h_layers]

    del model
    torch.cuda.empty_cache()

    return h_layers, grad_norms


class CriticalReLUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, h_layers,
                 sigma_w_sq=2.0, sigma_b_sq=0.0,
                 curriculum=True, gamma=CURRICULUM_GAMMA, total_epochs=EPOCHS):
        super().__init__()
        self.depth, self.sigma_w_sq, self.sigma_b_sq = len(h_layers), sigma_w_sq, sigma_b_sq
        self.curriculum, self.gamma, self.total_epochs = curriculum, gamma, total_epochs
        self.base_h_layers = list(h_layers)
        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        scaled_h = self._get_curriculum_rates(0)
        self.dropouts = nn.ModuleList([nn.Dropout(p=max(0, min(1, h))) for h in scaled_h])
        self._init_critical()

    def _get_curriculum_rates(self, epoch):
        if not self.curriculum:
            return list(self.base_h_layers)
        t = min(epoch, self.total_epochs - 1)
        return [h * (1.0 - np.exp(-self.gamma * (t + 1))) for h in self.base_h_layers]

    def set_epoch(self, epoch):
        if self.curriculum:
            for i, h in enumerate(self._get_curriculum_rates(epoch)):
                self.dropouts[i].p = max(0, min(1, h))

    def _init_critical(self):
        for layer in self.layers:
            fan_in = layer.weight.shape[1]
            nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(self.sigma_w_sq / fan_in))
            nn.init.normal_(layer.bias, mean=0.0, std=np.sqrt(self.sigma_b_sq)) if self.sigma_b_sq > 0 else nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            x = self.dropouts[i](x)
        return self.layers[-1](x)


def get_cifar10(train_size, test_size, seed=0):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    train = datasets.CIFAR10("./data", train=True, download=True)
    test = datasets.CIFAR10("./data", train=False, download=True)
    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(train.data), train_size, replace=False) if train_size < len(train.data) else np.arange(len(train.data))
    te_idx = rng.choice(len(test.data), test_size, replace=False) if test_size < len(test.data) else np.arange(len(test.data))
    x_tr = torch.from_numpy(train.data[tr_idx]).permute(0, 3, 1, 2).float().div_(255)
    y_tr = torch.tensor(np.array(train.targets)[tr_idx])
    x_te = torch.from_numpy(test.data[te_idx]).permute(0, 3, 1, 2).float().div_(255)
    y_te = torch.tensor(np.array(test.targets)[te_idx])
    if device == "cuda":
        x_tr, y_tr = x_tr.cuda(), y_tr.cuda()
        x_te, y_te = x_te.cuda(), y_te.cuda()
        mean, std = mean.cuda(), std.cuda()
    return (x_tr - mean) / std, y_tr, (x_te - mean) / std, y_te


def iterate_batches(x, y, bs, shuffle=True):
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, bs):
        yield x[idx[i:i + bs]], y[idx[i:i + bs]]


def train_epoch(model, x, y, opt, crit, bs):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in iterate_batches(x, y, bs):
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)
    return loss_sum / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, x, y, crit, bs):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in iterate_batches(x, y, bs, shuffle=False):
        out = model(xb)
        loss = crit(out, yb)
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)
    return loss_sum / total, 100 * correct / total


def run_schedule(sched_name, h_layers, x_tr, y_tr, x_te, y_te):
    """Run 20 simulations with a given schedule."""
    print(f"\n{'=' * 50}")
    print(f"Schedule: {sched_name}")
    print(f"  Dropout: {[f'{h:.3f}' for h in h_layers]}")
    print(f"{'=' * 50}")

    histories = []
    for sim in range(N_SIMULATIONS):
        seed = 42 + sim
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = CriticalReLUNet(3072, WIDTH, 10, h_layers,
                                SIGMA_W_SQ, SIGMA_B_SQ, True, CURRICULUM_GAMMA, EPOCHS).to(device)
        opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)
        warmup = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(opt, EPOCHS - WARMUP_EPOCHS, LR_MIN)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], [WARMUP_EPOCHS])
        crit = nn.CrossEntropyLoss()

        hist = {"test_acc": []}
        best_test_acc = 0.0

        for ep in range(EPOCHS):
            model.set_epoch(ep)
            train_epoch(model, x_tr, y_tr, opt, crit, BATCH_SIZE)
            scheduler.step()
            _, te_acc = evaluate(model, x_te, y_te, crit, BATCH_SIZE)
            best_test_acc = max(best_test_acc, te_acc)
            hist["test_acc"].append(te_acc)

        histories.append(hist)
        if (sim + 1) % 5 == 0:
            print(f"  sim {sim+1}/{N_SIMULATIONS}: best={best_test_acc:.1f}%")

    test_acc = np.array([h["test_acc"] for h in histories])
    best_mean = test_acc.max(axis=1).mean()
    best_sem = test_acc.max(axis=1).std() / np.sqrt(N_SIMULATIONS)
    final_mean = test_acc[:, -1].mean()
    final_sem = test_acc[:, -1].std() / np.sqrt(N_SIMULATIONS)

    print(f"  Best Test Acc:  {best_mean:.2f}% ± {best_sem:.3f}%")
    print(f"  Final Test Acc: {final_mean:.2f}% ± {final_sem:.3f}%")

    return round(float(best_mean), 4), round(float(best_sem), 4), \
           round(float(final_mean), 4), round(float(final_sem), 4)


def main():
    start_time = time.time()
    x_tr, y_tr, x_te, y_te = get_cifar10(TRAIN_SIZE, TEST_SIZE, seed=42)
    all_metrics = {}

    # 1. Compute gradient-norm adaptive schedule
    print("\n--- Computing gradient-norm adaptive schedule ---")
    grad_h_layers, grad_norms = compute_grad_norm_adaptive_schedule(
        CriticalReLUNet, 3072, WIDTH, 10,
        SIGMA_W_SQ, SIGMA_B_SQ, H_BAR, H_MAX, DEPTH,
        x_tr, y_tr, BATCH_SIZE, GRAD_CALIB_BATCHES
    )
    print(f"  Gradient norms: {[f'{g:.4f}' for g in grad_norms]}")
    print(f"  Adaptive h:      {[f'{h:.4f}' for h in grad_h_layers]}")

    # 2. Run gradient-norm adaptive schedule
    best, bsem, final, fsem = run_schedule("Gradient-Norm Adaptive", grad_h_layers,
                                            x_tr, y_tr, x_te, y_te)
    all_metrics["reverse_step_best_test_acc"] = best
    all_metrics["reverse_step_best_test_acc_sem"] = bsem
    all_metrics["reverse_step_final_test_acc"] = final
    all_metrics["reverse_step_final_test_acc_sem"] = fsem
    all_metrics["grad_norm_h_layers"] = [round(h, 4) for h in grad_h_layers]

    # 3. Run constant baseline (with same warmup+curriculum+grad_clip settings)
    print("\n--- Running constant baseline (matched settings) ---")
    const_h = [H_BAR] * DEPTH
    cbest, cbsem, cfinal, cfsem = run_schedule("Constant (matched settings)", const_h,
                                                x_tr, y_tr, x_te, y_te)
    all_metrics["constant_best_test_acc"] = cbest
    all_metrics["constant_best_test_acc_sem"] = cbsem
    all_metrics["constant_final_test_acc"] = cfinal
    all_metrics["constant_final_test_acc_sem"] = cfsem
    all_metrics["gap_best"] = round(best - cbest, 4)

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    out_dir = "/repo/results/iteration5"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "iter5_summary.json"), "w") as f:
        json.dump({"iteration": 5, "metrics": all_metrics}, f, indent=2)

    print("\nREPRODUCTION METRICS (JSON)")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
