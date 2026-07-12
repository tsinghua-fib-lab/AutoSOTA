"""
OVLR Example: Label Noise Robustness

This script demonstrates OVLR's robustness to label noise by directly
optimizing hard 0-1 loss, which is more resilient to corrupted labels
than cross-entropy loss optimized with standard backpropagation.

Reference:
    OVLR: Efficient, Scalable, and Robust Training via
    Output-Level Variance-Reduced Likelihood Ratio
    ICML 2026
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def hard_01_loss(outputs, targets):
    """Hard 0-1 loss: 0 if correct, 1 if incorrect."""
    predictions = outputs.argmax(dim=1)
    return (predictions != targets).float()


class NoisyDataset(Dataset):
    """Wrapper dataset that adds label noise to a clean dataset."""
    def __init__(self, dataset, noise_rate=0.0, num_classes=10, seed=42):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes

        # Generate noisy labels
        rng = torch.Generator()
        rng.manual_seed(seed)

        n_samples = len(dataset)
        self.clean_targets = torch.tensor([dataset[i][1] for i in range(n_samples)])
        self.noisy_targets = self.clean_targets.clone()

        if noise_rate > 0:
            n_noisy = int(n_samples * noise_rate)
            noisy_indices = torch.randperm(n_samples, generator=rng)[:n_noisy]

            for idx in noisy_indices:
                true_label = self.noisy_targets[idx]
                # Sample a random different label
                possible_labels = [l for l in range(num_classes) if l != true_label]
                new_label = possible_labels[torch.randint(0, len(possible_labels), (1,), generator=rng).item()]
                self.noisy_targets[idx] = new_label

        self.actual_noise_rate = (self.noisy_targets != self.clean_targets).float().mean().item()
        print(f"Applied noise: target={noise_rate:.1%}, actual={self.actual_noise_rate:.1%}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, self.noisy_targets[idx].item()


class SimpleCNN(nn.Module):
    """Simple CNN for quick demonstration."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def train_baseline(model, loader, optimizer, device, epoch):
    """Train with standard BP + cross-entropy (baseline)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    print(f"Train (BP+CE) Epoch: {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def train_ovlr(model, loader, optimizer, estimator, device, epoch):
    """Train with OVLR + hard 0-1 loss."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = estimator(outputs, targets, hard_01_loss, loss_fn_reduction='mean')
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    print(f"Train (OVLR+01) Epoch: {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def test(model, loader, device, method_name):
    """Evaluate on clean test set."""
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

    accuracy = 100. * correct / total
    print(f"Test ({method_name}) | Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='OVLR: Label Noise Robustness')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n-repeat', type=int, default=200, help='number of noisy samples for OVLR')
    parser.add_argument('--noise-scale', type=float, default=1.0, help='noise scale (sigma)')
    parser.add_argument('--noise-rate', type=float, default=0.3, help='label noise rate (0.0 to 0.6)')
    parser.add_argument('--compare-baseline', action='store_true', help='also train BP+CE baseline')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    clean_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Create noisy training set
    noisy_trainset = NoisyDataset(clean_trainset, noise_rate=args.noise_rate, num_classes=10)

    train_loader = DataLoader(noisy_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print("\n" + "=" * 60)
    print(f"Label Noise Robustness Experiment (Noise Rate: {args.noise_rate:.1%})")
    print("=" * 60 + "\n")

    # Train OVLR
    print("Training OVLR with Hard 0-1 Loss...")
    from ovlr import OVLRGradientEstimator, get_noise_fn

    model_ovlr = SimpleCNN(num_classes=10).to(device)
    noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
    estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)
    optimizer_ovlr = optim.Adam(model_ovlr.parameters(), lr=args.lr)

    best_ovlr = 0
    for epoch in range(1, args.epochs + 1):
        train_ovlr(model_ovlr, train_loader, optimizer_ovlr, estimator, device, epoch)
        acc = test(model_ovlr, test_loader, device, "OVLR+01")
        best_ovlr = max(best_ovlr, acc)

    if args.compare_baseline:
        # Train baseline
        print("\n" + "=" * 60)
        print("Training Baseline (BP + Cross-Entropy)...")
        model_bp = SimpleCNN(num_classes=10).to(device)
        optimizer_bp = optim.Adam(model_bp.parameters(), lr=args.lr)

        best_bp = 0
        for epoch in range(1, args.epochs + 1):
            train_baseline(model_bp, train_loader, optimizer_bp, device, epoch)
            acc = test(model_bp, test_loader, device, "BP+CE")
            best_bp = max(best_bp, acc)

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"OVLR + Hard 0-1:   {best_ovlr:.2f}%")
        print(f"BP + Cross-Entropy: {best_bp:.2f}%")
        print(f"Difference (OVLR - BP): {best_ovlr - best_bp:+.2f}%")
    else:
        print(f"\nBest OVLR test accuracy: {best_ovlr:.2f}%")


if __name__ == '__main__':
    main()
