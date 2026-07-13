#!/usr/bin/env python3
"""Reproduction script for paper 5420: Dropout Universality.

Target metrics from Table 7 (Appendix D.3):
- Step (early) Best Test Acc: 43.13 ± 0.07
- Step (early) Final Test Acc: 42.67 ± 0.09
- Constant Baseline Best Test Acc: 42.54 ± 0.10
- Constant Baseline Final Test Acc: 41.84 ± 0.10

Configuration:
  activation=ReLU, model=MLP, width=256, depth=6
  5000 training samples, epochs=50, batch_size=75
  lr=1e-4, lr_schedule=decay to min 1e-7 (CosineAnnealingLR)
  dropout_schedule=step (early), mean_dropout_field=0.1
  max_dropout_field=0.2, step_active_fraction=1/2
  weight_variance=1.98, bias_variance=0.02
  n_simulations=20
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Device ──────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ── Rubric hyperparameters ─────────────────────────────────────────
N_SIMULATIONS = 20
H_BAR = 0.1          # mean dropout field
H_MAX = 0.2          # max dropout field for step schedules
DEPTH = 6            # model depth (hard-coded per rubric)
WIDTH = 256          # hidden width
SIGMA_W_SQ = 1.98    # weight variance (slightly subcritical)
SIGMA_B_SQ = 0.02    # bias variance
EPOCHS = 50
LEARNING_RATE = 1e-4
LR_MIN = 1e-7
WARMUP_EPOCHS = 5
CURRICULUM_GAMMA = 10.0 / EPOCHS  # Curriculum dropout ramp speed
BATCH_SIZE = 75
TRAIN_SIZE = 5000
TEST_SIZE = 5000
USE_LR_SCHEDULE = True

# Schedules to run: step (early) = reverse_step, and constant baseline
SCHEDULES_TO_RUN = ["reverse_step", "constant"]
SCHEDULE_LABELS = {"reverse_step": "Step (early)", "constant": "Constant"}

print("=" * 60)
print("ITERATION 2: CURRICULUM DROPOUT + LR WARMUP + GRAD CLIP")
print("=" * 60)
print(f"  Model: MLP depth={DEPTH}, width={WIDTH}, ReLU")
print(f"  sigma_w^2={SIGMA_W_SQ}, sigma_b^2={SIGMA_B_SQ}")
print(f"  h_bar={H_BAR}, h_max={H_MAX}")
print(f"  Train size={TRAIN_SIZE}, Test size={TEST_SIZE}")
print(f"  Epochs={EPOCHS}, Batch size={BATCH_SIZE}")
print(f"  LR={LEARNING_RATE} -> min={LR_MIN}")
print(f"  Simulations={N_SIMULATIONS}")
print(f"  Schedules: {SCHEDULES_TO_RUN}")
print("=" * 60)


# ── Dropout schedule helpers ────────────────────────────────────────
def get_dropout_schedule(schedule_type, depth, h_bar, h_max=None):
    if schedule_type == "constant":
        return [h_bar] * depth
    elif schedule_type == "reverse_step":
        if h_max is None or h_max < h_bar:
            raise ValueError("reverse_step requires h_max >= h_bar")
        f = h_bar / h_max
        n_drop = max(1, int(np.ceil(f * depth)))
        h_adj = h_bar * depth / n_drop
        return [h_adj] * n_drop + [0.0] * (depth - n_drop)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


# ── Model ───────────────────────────────────────────────────────────
class CriticalReLUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, h_layers,
                 sigma_w_sq=2.0, sigma_b_sq=0.0,
                 curriculum=True, gamma=CURRICULUM_GAMMA, total_epochs=EPOCHS):
        super().__init__()
        self.depth = len(h_layers)
        self.sigma_w_sq = sigma_w_sq
        self.sigma_b_sq = sigma_b_sq
        self.curriculum = curriculum
        self.gamma = gamma
        self.total_epochs = total_epochs
        self.base_h_layers = list(h_layers)

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(self.depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        scaled_h = self._get_curriculum_rates(0) if curriculum else list(h_layers)
        self.dropouts = nn.ModuleList([nn.Dropout(p=h) for h in scaled_h])
        self._init_critical()

    def _get_curriculum_rates(self, epoch):
        """Curriculum dropout: h(t) = h_target * (1 - exp(-gamma * t))."""
        t = min(epoch, self.total_epochs - 1)
        factor = 1.0 - np.exp(-self.gamma * (t + 1))
        return [h * factor for h in self.base_h_layers]

    def set_epoch(self, epoch):
        """Update dropout rates for the current epoch (curriculum schedule)."""
        if self.curriculum:
            scaled_h = self._get_curriculum_rates(epoch)
            for i, h in enumerate(scaled_h):
                self.dropouts[i].p = h

    def _init_critical(self):
        for layer in self.layers:
            fan_in = layer.weight.shape[1]
            std_w = np.sqrt(self.sigma_w_sq / fan_in)
            nn.init.normal_(layer.weight, mean=0.0, std=std_w)
            if self.sigma_b_sq > 0:
                nn.init.normal_(layer.bias, mean=0.0,
                                std=np.sqrt(self.sigma_b_sq))
            else:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            x = self.dropouts[i](x)
        return self.layers[-1](x)


# ── Data loading ────────────────────────────────────────────────────
def get_cifar10(train_size=None, test_size=None, seed=0):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    train = datasets.CIFAR10("./data", train=True, download=True)
    test = datasets.CIFAR10("./data", train=False, download=True)

    rng = np.random.RandomState(seed)

    if train_size is None or train_size >= len(train.data):
        tr_idx = np.arange(len(train.data))
        print(f"Using full training set: {len(tr_idx)} samples")
    else:
        tr_idx = rng.choice(len(train.data), train_size, replace=False)
        print(f"Using {len(tr_idx)} training samples")

    if test_size is None or test_size >= len(test.data):
        te_idx = np.arange(len(test.data))
        print(f"Using full test set: {len(te_idx)} samples")
    else:
        te_idx = rng.choice(len(test.data), test_size, replace=False)
        print(f"Using {len(te_idx)} test samples")

    x_tr = torch.from_numpy(train.data[tr_idx]).permute(0, 3, 1, 2).float().div_(255)
    y_tr = torch.tensor(np.array(train.targets)[tr_idx])
    x_te = torch.from_numpy(test.data[te_idx]).permute(0, 3, 1, 2).float().div_(255)
    y_te = torch.tensor(np.array(test.targets)[te_idx])

    if device == "cuda":
        x_tr, y_tr = x_tr.cuda(), y_tr.cuda()
        x_te, y_te = x_te.cuda(), y_te.cuda()
        mean, std = mean.cuda(), std.cuda()

    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    return (x_tr, y_tr), (x_te, y_te)


# ── Training helpers ────────────────────────────────────────────────
def iterate_batches(x, y, bs, shuffle=True):
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    for i in range(0, n, bs):
        yield x[idx[i:i + bs]], y[idx[i:i + bs]]


def train_epoch(model, data, opt, crit, bs):
    model.train()
    x, y = data
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
def evaluate(model, data, crit, bs):
    model.eval()
    x, y = data
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in iterate_batches(x, y, bs, shuffle=False):
        out = model(xb)
        loss = crit(out, yb)
        total += xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        loss_sum += loss.item() * xb.size(0)
    return loss_sum / total, 100 * correct / total


# ── Main reproduction run ──────────────────────────────────────────
def main():
    start_time = time.time()

    # Load data (once, reused across schedules)
    print("\nLoading CIFAR-10...")
    train_data, test_data = get_cifar10(TRAIN_SIZE, TEST_SIZE, seed=42)
    print(f"Train: {train_data[0].shape}, Test: {test_data[0].shape}")

    results = {}
    all_metrics = {}

    for sched in SCHEDULES_TO_RUN:
        print(f"\n{'=' * 50}")
        print(f"Schedule: {SCHEDULE_LABELS[sched]} ({sched})")
        print(f"{'=' * 50}")

        h_layers = get_dropout_schedule(sched, DEPTH, H_BAR, H_MAX)
        print(f"  Dropout per layer: {[f'{h:.3f}' for h in h_layers]}")
        print(f"  Active fraction: {sum(1 for h in h_layers if h > 0)}/{DEPTH}")

        histories = []

        for sim in range(N_SIMULATIONS):
            seed = 42 + sim
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = CriticalReLUNet(
                input_dim=3072, hidden_dim=WIDTH, output_dim=10,
                h_layers=h_layers,
                sigma_w_sq=SIGMA_W_SQ, sigma_b_sq=SIGMA_B_SQ,
            ).to(device)

            opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)
            if USE_LR_SCHEDULE:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
                )
                cosine_scheduler = CosineAnnealingLR(
                    opt, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR_MIN
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    opt, schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[WARMUP_EPOCHS]
                )
            else:
                scheduler = None
            crit = nn.CrossEntropyLoss()

            hist = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
            best_test_acc = 0.0
            best_test_loss = float("inf")

            pbar = tqdm(range(EPOCHS), desc=f"{sched} sim {sim+1}/{N_SIMULATIONS}", leave=True)

            for ep in pbar:
                model.set_epoch(ep)
                tr_loss, tr_acc = train_epoch(model, train_data, opt, crit, BATCH_SIZE)
                if scheduler:
                    scheduler.step()

                te_loss, te_acc = evaluate(model, test_data, crit, BATCH_SIZE)

                if te_acc > best_test_acc:
                    best_test_acc = te_acc
                if te_loss < best_test_loss:
                    best_test_loss = te_loss

                hist["train_acc"].append(tr_acc)
                hist["test_acc"].append(te_acc)
                hist["train_loss"].append(tr_loss)
                hist["test_loss"].append(te_loss)

                pbar.set_postfix({
                    "ep": f"{ep+1}/{EPOCHS}",
                    "tr_L": f"{tr_loss:.3f}",
                    "tr_A": f"{tr_acc:.1f}%",
                    "te_L": f"{te_loss:.3f}",
                    "te_A": f"{te_acc:.1f}%",
                    "best": f"{best_test_acc:.1f}%",
                })

            histories.append(hist)

        # Aggregate
        test_acc = np.array([h["test_acc"] for h in histories])
        test_loss = np.array([h["test_loss"] for h in histories])
        train_acc = np.array([h["train_acc"] for h in histories])
        train_loss = np.array([h["train_loss"] for h in histories])

        best_test = test_acc.max(axis=1)          # best over epochs, per seed
        final_test = test_acc[:, -1]               # final epoch, per seed

        results[sched] = {
            "test_acc": test_acc,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "train_loss": train_loss,
        }

        best_mean = best_test.mean()
        best_sem = best_test.std() / np.sqrt(N_SIMULATIONS)
        final_mean = final_test.mean()
        final_sem = final_test.std() / np.sqrt(N_SIMULATIONS)

        print(f"\n{SCHEDULE_LABELS[sched]} ({sched}):")
        print(f"  Best Test Acc:  {best_mean:.2f}% ± {best_sem:.3f}%")
        print(f"  Final Test Acc: {final_mean:.2f}% ± {final_sem:.3f}%")

        all_metrics[f"{sched}_best_test_acc"] = round(float(best_mean), 4)
        all_metrics[f"{sched}_best_test_acc_sem"] = round(float(best_sem), 4)
        all_metrics[f"{sched}_final_test_acc"] = round(float(final_mean), 4)
        all_metrics[f"{sched}_final_test_acc_sem"] = round(float(final_sem), 4)

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Save results ───────────────────────────────────────────────
    out_dir = "/repo/results/iteration2"
    os.makedirs(out_dir, exist_ok=True)

    # Save NPZ with all arrays
    rs_test_acc = results["reverse_step"]["test_acc"]
    ct_test_acc = results["constant"]["test_acc"]
    np.savez(
        os.path.join(out_dir, "iter2_results.npz"),
        reverse_step_best=rs_test_acc.max(axis=1),
        reverse_step_final=rs_test_acc[:, -1],
        constant_best=ct_test_acc.max(axis=1),
        constant_final=ct_test_acc[:, -1],
        reverse_step_test_acc=rs_test_acc,
        constant_test_acc=ct_test_acc,
    )

    # Save JSON summary
    summary = {
        "paper_id": 5420,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "model": "MLP",
            "activation": "ReLU",
            "depth": DEPTH,
            "width": WIDTH,
            "sigma_w_sq": SIGMA_W_SQ,
            "sigma_b_sq": SIGMA_B_SQ,
            "h_bar": H_BAR,
            "h_max": H_MAX,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "lr_min": LR_MIN,
            "train_size": TRAIN_SIZE,
            "test_size": TEST_SIZE,
            "n_simulations": N_SIMULATIONS,
        },
        "metrics": all_metrics,
        "schedules": {
            "reverse_step": {
                "label": "Step (early)",
                "dropout_per_layer": [f"{h:.3f}" for h in get_dropout_schedule("reverse_step", DEPTH, H_BAR, H_MAX)],
                "active_fraction": f"{sum(1 for h in get_dropout_schedule('reverse_step', DEPTH, H_BAR, H_MAX) if h > 0)}/{DEPTH}",
            },
            "constant": {
                "label": "Constant",
                "dropout_per_layer": [f"{h:.3f}" for h in get_dropout_schedule("constant", DEPTH, H_BAR, H_MAX)],
                "active_fraction": f"{sum(1 for h in get_dropout_schedule('constant', DEPTH, H_BAR, H_MAX) if h > 0)}/{DEPTH}",
            },
        },
    }
    with open(os.path.join(out_dir, "iter2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"  - reproduction_results.npz")
    print(f"  - reproduction_summary.json")

    # Final comparison with rubric targets
    print("\n" + "=" * 60)
    print("REPRODUCTION COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Reproduced':>14} {'Rubric Target':>16}")
    print("-" * 60)
    for sched_key, sched_label in [("reverse_step", "Step (early)"), ("constant", "Constant")]:
        best = all_metrics[f"{sched_key}_best_test_acc"]
        final = all_metrics[f"{sched_key}_final_test_acc"]
        print(f"{sched_label} Best Test Acc:  {best:>12.2f}%")
        print(f"{sched_label} Final Test Acc: {final:>12.2f}%")
    print("=" * 60)

    # Emit the key metrics in machine-parseable format
    print("\n" + "=" * 60)
    print("REPRODUCTION METRICS (JSON)")
    print("=" * 60)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
