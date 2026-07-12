#!/usr/bin/env python3
"""
Reproduction script for OVLR Table 7: ResNet-18 + CIFAR-10 + hard 0-1 loss.
Matches paper setting: 5-epoch CE warmup, then OVLR with hard 0-1 loss.
Saves final metrics to JSON for manifest generation.
"""
import argparse, time, json, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def hard_01_loss(outputs, targets):
    predictions = outputs.argmax(dim=1)
    return (predictions != targets).float()


def test(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-repeat', type=int, default=200)
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--data-dir', type=str, default='/datasets')
    parser.add_argument('--output', type=str, default='/repo/results_table7.json')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet18(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_criterion = nn.CrossEntropyLoss()

    from ovlr import OVLRGradientEstimator, get_noise_fn
    noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
    estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)

    print(f"\n{'='*70}")
    print(f"OVLR Table 7: ResNet-18 + CIFAR-10 + hard 0-1 loss")
    print(f"Warmup: {args.warmup_epochs} epochs CE, Total: {args.epochs} epochs")
    print(f"n_repeat={args.n_repeat}, sigma={args.noise_scale}, lr={args.lr}, batch={args.batch_size}")
    print(f"{'='*70}\n")

    best_acc = 0.0
    best_ovlr_acc = 0.0
    warmup_time = 0.0
    ovlr_time = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        epoch_loss = correct = total = 0.0

        if epoch <= args.warmup_epochs:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = ce_criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
            phase = "WARMUP_CE"
        else:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = estimator(outputs, targets, hard_01_loss, loss_fn_reduction='mean')
                optimizer.step()
                epoch_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
            phase = "OVLR_01"

        torch.cuda.synchronize()
        et = time.perf_counter() - t0
        if epoch <= args.warmup_epochs:
            warmup_time += et
        else:
            ovlr_time += et

        train_loss = epoch_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc = test(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        if epoch > args.warmup_epochs:
            best_ovlr_acc = max(best_ovlr_acc, test_acc)

        history.append({"epoch": epoch, "phase": phase, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 2), "test_acc": round(test_acc, 2),
                        "time_s": round(et, 2)})
        print(f"Epoch {epoch:3d}/{args.epochs} [{phase:10s}] "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | {et:.1f}s")

    total_time = warmup_time + ovlr_time
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"  Best test accuracy (overall): {best_acc:.2f}%")
    print(f"  Best OVLR-phase accuracy: {best_ovlr_acc:.2f}%")
    print(f"  Final test accuracy: {history[-1]['test_acc']:.2f}%")
    print(f"  Warmup time: {warmup_time:.1f}s")
    print(f"  OVLR time: {ovlr_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Paper target: 62.04% accuracy, 225.1s time")
    print(f"{'='*70}")

    result = {
        "paper_id": 4550,
        "model": "ResNet-18",
        "dataset": "CIFAR-10",
        "loss": "hard_0-1",
        "method": "OVLR",
        "warmup_epochs": args.warmup_epochs,
        "warmup_loss": "Cross-Entropy",
        "total_epochs": args.epochs,
        "n_repeat": args.n_repeat,
        "noise_scale": args.noise_scale,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_test_accuracy": round(best_acc, 2),
        "best_ovlr_phase_accuracy": round(best_ovlr_acc, 2),
        "final_test_accuracy": round(history[-1]['test_acc'], 2),
        "warmup_time_s": round(warmup_time, 1),
        "ovlr_time_s": round(ovlr_time, 1),
        "total_time_s": round(total_time, 1),
        "history": history,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
