#!/usr/bin/env python3
"""
Reproduction of PLATE MNIST 0-4 -> 5-9 continual learning experiment.
Matches: Figure 5 (top), Section 5.2.2.
Paper config: r=350, col_tau=0.8, plate_alpha=0.5 (rho)
"""
import os, sys, json, time, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, '/repo')
from plate import PLATEConfig, get_plate_model


class MNISTBackbone(nn.Module):
    """3-layer ReLU MLP (784 -> 256 -> 256 -> 256) backbone."""
    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


def build_task_datasets(batch_size=128):
    """Create MNIST 0-4 (task 1) and 5-9 (task 2) datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('/datasets/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('/datasets/mnist', train=False, download=True, transform=transform)

    task1_train_idx = [i for i, (_, y) in enumerate(train_dataset) if y < 5]
    task1_test_idx = [i for i, (_, y) in enumerate(test_dataset) if y < 5]
    task2_train_idx = [i for i, (_, y) in enumerate(train_dataset) if y >= 5]
    task2_test_idx = [i for i, (_, y) in enumerate(test_dataset) if y >= 5]

    train1 = Subset(train_dataset, task1_train_idx)
    test1 = Subset(test_dataset, task1_test_idx)
    train2 = Subset(train_dataset, task2_train_idx)
    test2 = Subset(test_dataset, task2_test_idx)

    # Remap Task 2 labels from 5-9 to 0-4
    class RemapDataset(torch.utils.data.Dataset):
        def __init__(self, subset, offset=5):
            self.subset = subset
            self.offset = offset
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, y - self.offset

    train2_remapped = RemapDataset(train2, 5)
    test2_remapped = RemapDataset(test2, 5)

    train1_loader = DataLoader(train1, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test1_loader = DataLoader(test1, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train2_loader = DataLoader(train2_remapped, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test2_loader = DataLoader(test2_remapped, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train1_loader, test1_loader, train2_loader, test2_loader


def evaluate(backbone, head1, head2, test1_loader, test2_loader, device):
    """Evaluate Task 1 retention and Task 2 accuracy."""
    backbone.eval()
    head1.eval()
    head2.eval()

    correct1, total1 = 0, 0
    with torch.no_grad():
        for x, y in test1_loader:
            x, y = x.to(device), y.to(device)
            features = backbone(x)
            out = head1(features)
            correct1 += (out.argmax(1) == y).sum().item()
            total1 += y.size(0)
    task1_acc = 100.0 * correct1 / total1

    correct2, total2 = 0, 0
    with torch.no_grad():
        for x, y in test2_loader:
            x, y = x.to(device), y.to(device)
            features = backbone(x)
            out = head2(features)
            correct2 += (out.argmax(1) == y).sum().item()
            total2 += y.size(0)
    task2_acc = 100.0 * correct2 / total2

    return task1_acc, task2_acc


def train_epoch(backbone, head, loader, optimizer, device, scaler=None, teacher_backbone=None, kd_lambda=0.0):
    backbone.train()
    head.train()
    if teacher_backbone is not None:
        teacher_backbone.eval()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                features = backbone(x)
                out = head(features)
                ce_loss = F.cross_entropy(out, y)
                loss = ce_loss
                if teacher_backbone is not None and kd_lambda > 0:
                    with torch.no_grad():
                        teacher_features = teacher_backbone(x)
                    kd_loss = kd_lambda * F.mse_loss(features, teacher_features)
                    loss = loss + kd_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            features = backbone(x)
            out = head(features)
            ce_loss = F.cross_entropy(out, y)
            loss = ce_loss
            if teacher_backbone is not None and kd_lambda > 0:
                with torch.no_grad():
                    teacher_features = teacher_backbone(x)
                kd_loss = kd_lambda * F.mse_loss(features, teacher_features)
                loss = loss + kd_loss
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def single_run(seed, train1_loader, test1_loader, train2_loader, test2_loader, device, plate_r, plate_tau, plate_alpha, epochs=10, lr=1e-3, use_amp=True, kd_lambda=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = MNISTBackbone(seed=seed).to(device)
    head1 = nn.Linear(256, 5).to(device)
    head2 = nn.Linear(256, 5).to(device)

    # Stage 1: Train backbone + head1 on MNIST 0-4 (constant LR, no AMP)
    opt1 = torch.optim.Adam(list(backbone.parameters()) + list(head1.parameters()), lr=1e-3)
    for epoch in range(10):
        train_epoch(backbone, head1, train1_loader, opt1, device)

    # Baseline Task 1 accuracy
    backbone.eval()
    head1.eval()
    correct1, total1 = 0, 0
    with torch.no_grad():
        for x, y in test1_loader:
            x, y = x.to(device), y.to(device)
            out = head1(backbone(x))
            correct1 += (out.argmax(1) == y).sum().item()
            total1 += y.size(0)
    baseline_t1 = 100.0 * correct1 / total1

    # Stage 2: Freeze backbone + head1, apply PLATE, train on MNIST 5-9
    total_backbone = sum(p.numel() for p in backbone.parameters())
    for p in backbone.parameters():
        p.requires_grad = False
    for p in head1.parameters():
        p.requires_grad = False

    # Create frozen teacher for feature distillation (I-05)
    teacher_backbone = copy.deepcopy(backbone) if kd_lambda > 0 else None
    if teacher_backbone is not None:
        for p in teacher_backbone.parameters():
            p.requires_grad = False
        teacher_backbone.eval()

    plate_config = PLATEConfig(
            col_tau_pattern={"fc1": 0.85, "fc2": 0.80, "fc3": 0.75},
        r=plate_r, col_tau=plate_tau, plate_alpha=plate_alpha,
        target_modules=["fc1", "fc2", "fc3"], plate_dropout=0.0,
    )
    plate_model = get_plate_model(backbone, plate_config)

    plate_trainable = [p for p in plate_model.parameters() if p.requires_grad]
    trainable_plate = sum(p.numel() for p in plate_trainable)

    # Stage 2: Use AMP with GradScaler + optional KD loss (I-03, I-05)
    opt2 = torch.optim.Adam(plate_trainable + list(head2.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    for epoch in range(epochs):
        train_epoch(plate_model, head2, train2_loader, opt2, device, scaler=scaler,
                   teacher_backbone=teacher_backbone, kd_lambda=kd_lambda)

    task1_ret, task2_acc = evaluate(plate_model, head1, head2, test1_loader, test2_loader, device)

    return {
        'seed': seed,
        'baseline_task1_acc': baseline_t1,
        'task1_retention': task1_ret,
        'task2_accuracy': task2_acc,
        'forgetting': baseline_t1 - task1_ret,
        'trainable_plate_params': trainable_plate,
        'total_backbone_params': total_backbone,
        'trainable_pct': 100.0 * trainable_plate / total_backbone,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, default=350)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kd_lambda', type=float, default=0.0)
    parser.add_argument('--seed_start', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PLATE: r={args.r}, tau={args.tau}, alpha={args.alpha}")
    print(f"Runs: {args.n_runs}, epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}, kd_lambda: {args.kd_lambda}")

    train1_loader, test1_loader, train2_loader, test2_loader = build_task_datasets(args.batch_size)

    results = []
    t0 = time.time()
    for k in range(args.n_runs):
        seed = args.seed_start + k
        print(f"\n--- Run {k+1}/{args.n_runs} (seed={seed}) ---")
        r = single_run(seed, train1_loader, test1_loader, train2_loader, test2_loader, device,
                       args.r, args.tau, args.alpha, epochs=args.epochs, lr=args.lr, kd_lambda=args.kd_lambda)
        results.append(r)
        print(f"  T1 baseline={r['baseline_task1_acc']:.2f}%  T1 ret={r['task1_retention']:.2f}%  "
              f"T2 acc={r['task2_accuracy']:.2f}%  Forget={r['forgetting']:.2f}%  "
              f"Trainable={r['trainable_pct']:.1f}%")

    elapsed = (time.time() - t0) / 60
    keys = ['baseline_task1_acc', 'task1_retention', 'task2_accuracy', 'forgetting', 'trainable_pct']
    agg = {}
    for key in keys:
        vals = [r[key] for r in results]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

    print(f"\n======= FINAL ({args.n_runs} runs, {elapsed:.1f} min) =======")
    print(f"  T1 baseline:  {agg['baseline_task1_acc_mean']:.2f}% ± {agg['baseline_task1_acc_std']:.2f}")
    print(f"  T1 retention: {agg['task1_retention_mean']:.2f}% ± {agg['task1_retention_std']:.2f}")
    print(f"  T2 accuracy:  {agg['task2_accuracy_mean']:.2f}% ± {agg['task2_accuracy_std']:.2f}")
    print(f"  Forgetting:   {agg['forgetting_mean']:.2f}% ± {agg['forgetting_std']:.2f}")
    print(f"  Trainable %:  {agg['trainable_pct_mean']:.1f}% ± {agg['trainable_pct_std']:.1f}")

    output = {
        'config': {'r': args.r, 'tau': args.tau, 'alpha': args.alpha,
                   'n_runs': args.n_runs, 'epochs': args.epochs,
                   'batch_size': args.batch_size, 'lr': args.lr},
        'aggregates': agg,
        'runs': results,
        'elapsed_minutes': elapsed,
    }
    with open('/repo/results_mnist_plate.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to /repo/results_mnist_plate.json")

if __name__ == '__main__':
    main()
