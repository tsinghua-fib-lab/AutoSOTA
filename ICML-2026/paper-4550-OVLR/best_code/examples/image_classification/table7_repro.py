
#!/usr/bin/env python3
"""
Reproduction script for OVLR Table 7: ResNet-18 + CIFAR-10 + hard 0-1 loss.
Matches paper setting: 5-epoch CE warmup, then OVLR with hard 0-1 loss.
"""
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def hard_01_loss(outputs, targets):
    """Hard 0-1 loss: 0 if correct, 1 if incorrect. Non-differentiable."""
    predictions = outputs.argmax(dim=1)
    return (predictions != targets).float()


def test(model, loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='OVLR Table 7 reproduction: ResNet-18 + CIFAR-10 + hard 0-1 loss')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30, help='Total epochs (including warmup)')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='CE warmup epochs')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n-repeat', type=int, default=200, help='OVLR noise samples')
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--data-dir', type=str, default='/datasets')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")

    # CIFAR-10 data
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
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ResNet-18
    model = models.resnet18(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ce_criterion = nn.CrossEntropyLoss()

    # OVLR setup
    from ovlr import OVLRGradientEstimator, get_noise_fn
    noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
    estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)

    print(f"\n{'='*70}")
    print(f"OVLR Table 7 Reproduction: ResNet-18 + CIFAR-10 + hard 0-1 loss")
    print(f"Warmup epochs: {args.warmup_epochs} (CE), Total epochs: {args.epochs}")
    print(f"n_repeat: {args.n_repeat}, noise_scale: {args.noise_scale}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"{'='*70}\n")

    best_acc = 0.0
    total_train_time = 0.0
    ovlr_train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        if epoch <= args.warmup_epochs:
            # CE warmup with standard BP
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
            phase = "WARMUP (CE+BP)"
        else:
            # OVLR with hard 0-1 loss
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
            phase = "OVLR (hard 0-1)"

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        total_train_time += epoch_time
        if epoch > args.warmup_epochs:
            ovlr_train_time += epoch_time

        train_loss = epoch_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc = test(model, test_loader, device)
        best_acc = max(best_acc, test_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} [{phase:15s}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Total training time: {total_train_time:.2f}s")
    print(f"OVLR phase time (epochs {args.warmup_epochs+1}-{args.epochs}): {ovlr_train_time:.2f}s")
    print(f"Target: 62.04% accuracy, 225.1s time (Table 7)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
