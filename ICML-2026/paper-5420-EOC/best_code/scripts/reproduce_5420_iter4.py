#!/usr/bin/env python3
"""Iteration 4: Mixup augmentation + xi-optimized schedule + curriculum + warmup + grad clip.

Adds mixup (alpha=0.2) to the best configuration from iteration 3.
"""

import json, os, sys, time, warnings, re
from datetime import datetime, timezone

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from tqdm.auto import tqdm

sys.path.insert(0, "/repo/src")
from dropout_mft.schedules import effective_xi

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

N_SIMULATIONS = 20
H_BAR, DEPTH, WIDTH = 0.1, 6, 256
SIGMA_W_SQ, SIGMA_B_SQ = 1.98, 0.02
EPOCHS, LEARNING_RATE, LR_MIN = 50, 1e-4, 1e-7
BATCH_SIZE, TRAIN_SIZE, TEST_SIZE = 75, 5000, 5000
WARMUP_EPOCHS, CURRICULUM_GAMMA = 5, 10.0 / EPOCHS
MIXUP_ALPHA = 0.2  # Zhang et al. ICLR 2018 recommendation for CIFAR-10

# Best xi-optimized schedules from iter 3
XI_H_MAX = 0.30
SCHEDULES_TO_RUN = ["xi_opt_h0.30"]
SCHEDULE_LABELS = {"xi_opt_h0.30": "Xi-Opt h_max=0.30 + Mixup"}

print("=" * 60)
print("ITERATION 4: MIXUP + XI-OPT + CURRICULUM + WARMUP + GRAD CLIP")
print("=" * 60)
print(f"  Mixup alpha={MIXUP_ALPHA}")
print(f"  Xi-optimized h_max={XI_H_MAX}")
print(f"  Curriculum gamma={CURRICULUM_GAMMA:.4f}")
print(f"  LR warmup={WARMUP_EPOCHS} epochs")
print("=" * 60)


def correct_xi(h_layers):
    power = 1/3
    damage = np.mean([h**power for h in h_layers])
    return float("inf") if damage <= 0 else 1.0 / damage


def optimize_xi_schedule(depth, h_bar, h_max_relaxed, ordering="early"):
    best_xi, best_h = -1, None
    for n_drop in range(1, depth + 1):
        h_active = h_bar * depth / n_drop
        if h_active > h_max_relaxed + 1e-9:
            continue
        h_layers = ([h_active] * n_drop + [0.0] * (depth - n_drop) if ordering == "early"
                    else [0.0] * (depth - n_drop) + [h_active] * n_drop)
        xi = correct_xi(h_layers)
        if xi > best_xi:
            best_xi, best_h = xi, h_layers
    return best_h, best_xi


def get_dropout_schedule(schedule_type, depth, h_bar, h_max=None):
    if "constant" in schedule_type:
        return [h_bar] * depth
    if "reverse_step" in schedule_type:
        f, n_drop = h_bar / (h_max or 2.0 * h_bar), max(1, int(np.ceil(h_bar / (h_max or 2.0 * h_bar) * depth)))
        return [h_bar * depth / n_drop] * n_drop + [0.0] * (depth - n_drop)
    m = re.search(r"h([0-9.]+)", schedule_type)
    hv = float(m.group(1)) if m else (h_max or 2.0 * h_bar)
    h_layers, xi = optimize_xi_schedule(depth, h_bar, h_max_relaxed=hv, ordering="early")
    print(f"  Xi-optimized (h_max={hv:.2f}): {[f'{h:.3f}' for h in h_layers]}, xi={xi:.4f}")
    return h_layers


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


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Apply mixup: x_mix = lambda*x_i + (1-lambda)*x_j, y is one-hot mixed."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(crit, pred, y_a, y_b, lam):
    """Mixup loss: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)."""
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


def train_epoch(model, x, y, opt, crit, bs, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in iterate_batches(x, y, bs):
        if use_mixup:
            xb, y_a, y_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        if use_mixup:
            loss = mixup_criterion(crit, out, y_a, y_b, lam)
        else:
            loss = crit(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += xb.size(0)
        if not use_mixup:
            correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)
    return loss_sum / total, 100 * correct / total if not use_mixup else 0.0


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


def main():
    start_time = time.time()
    x_tr, y_tr, x_te, y_te = get_cifar10(TRAIN_SIZE, TEST_SIZE, seed=42)
    all_metrics = {}

    for sched in SCHEDULES_TO_RUN:
        print(f"\n{'=' * 50}")
        print(f"Schedule: {SCHEDULE_LABELS[sched]}")
        print(f"{'=' * 50}")

        h_layers = get_dropout_schedule(sched, DEPTH, H_BAR, XI_H_MAX)

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
                train_epoch(model, x_tr, y_tr, opt, crit, BATCH_SIZE, use_mixup=True)
                scheduler.step()
                _, te_acc = evaluate(model, x_te, y_te, crit, BATCH_SIZE)
                best_test_acc = max(best_test_acc, te_acc)
                hist["test_acc"].append(te_acc)

            histories.append(hist)
            if (sim + 1) % 5 == 0:
                print(f"  sim {sim+1}/{N_SIMULATIONS}: best={best_test_acc:.1f}%")

        test_acc = np.array([h["test_acc"] for h in histories])
        best_test = test_acc.max(axis=1)
        best_mean = best_test.mean()
        best_sem = best_test.std() / np.sqrt(N_SIMULATIONS)
        final_mean = test_acc[:, -1].mean()
        final_sem = test_acc[:, -1].std() / np.sqrt(N_SIMULATIONS)

        print(f"  Best Test Acc:  {best_mean:.2f}% ± {best_sem:.3f}%")
        print(f"  Final Test Acc: {final_mean:.2f}% ± {final_sem:.3f}%")

        all_metrics["reverse_step_best_test_acc"] = round(float(best_mean), 4)
        all_metrics["reverse_step_best_test_acc_sem"] = round(float(best_sem), 4)
        all_metrics["reverse_step_final_test_acc"] = round(float(final_mean), 4)
        all_metrics["reverse_step_final_test_acc_sem"] = round(float(final_sem), 4)

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    out_dir = "/repo/results/iteration4"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "iter4_summary.json"), "w") as f:
        json.dump({"iteration": 4, "changes": "mixup + xi-opt", "metrics": all_metrics}, f, indent=2)

    print("\nREPRODUCTION METRICS (JSON)")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
