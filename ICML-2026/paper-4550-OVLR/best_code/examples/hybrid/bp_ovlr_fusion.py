"""
OVLR Example: Hybrid BP + OVLR Fusion

Combines standard Backpropagation (BP) on Cross-Entropy (CE) with
OVLR on Hard 0-1 loss using weighted combination:

    L_total = alpha * CE + (1 - alpha) * Hard_01_via_OVLR

This achieves the best of both: smooth gradients from CE and
target-aligned optimization from Hard 0-1.

Reference:
    OVLR: Efficient, Scalable, and Robust Training via
    Output-Level Variance-Reduced Likelihood Ratio
    ICML 2026
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os


def hard_01_loss(outputs, targets):
    """Hard 0-1 accuracy loss (differentiated via OVLR)."""
    predictions = outputs.argmax(dim=1)
    return (predictions != targets).float()


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


def train_epoch_hybrid(model, loader, optimizer, alpha, estimator, ce_criterion, device):
    """
    Train one epoch with hybrid BP+OVLR fusion.

    Key implementation:
    1. BP on Cross-Entropy with retain_graph=True
    2. OVLR on Hard 0-1 with the same computation graph
    """
    model.train()
    total_ce_loss = 0.0
    total_01_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        # ----- BP on Cross-Entropy -----
        ce_loss = ce_criterion(outputs, targets)
        weighted_ce = alpha * ce_loss.mean()
        weighted_ce.backward(retain_graph=True)  # Keep graph for OVLR gradient

        # ----- OVLR on Hard 0-1 Loss -----
        # estimator internally does outputs.backward(gradient_vector)
        ovlr_loss = estimator(outputs, targets, hard_01_loss,
                              loss_fn_reduction='mean', retain_graph=False)

        optimizer.step()

        total_ce_loss += ce_loss.mean().item()
        total_01_loss += ovlr_loss.item()

        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    avg_ce = total_ce_loss / len(loader)
    avg_01 = total_01_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_ce, avg_01, accuracy


def train_epoch_baseline(model, loader, optimizer, ce_criterion, device):
    """Train one epoch with standard BP only."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = ce_criterion(outputs, targets).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_epoch_ovlr_only(model, loader, optimizer, estimator, device):
    """Train one epoch with OVLR only on Hard 0-1."""
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
    return avg_loss, accuracy


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
    parser = argparse.ArgumentParser(description='OVLR: Hybrid BP + OVLR Fusion')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n-repeat', type=int, default=200, help='OVLR repeat count')
    parser.add_argument('--noise-scale', type=float, default=1.0, help='OVLR noise scale (sigma)')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help='Alpha values for hybrid loss (weight on CE)')
    parser.add_argument('--save-dir', type=str, default='./results_hybrid', help='Save directory')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Data loading
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

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    ce_criterion = nn.CrossEntropyLoss(reduction='none')

    from ovlr import OVLRGradientEstimator, get_noise_fn

    results = {}

    print("\n" + "=" * 60)
    print(f"Hybrid BP+OVLR Fusion Experiment on CIFAR-10")
    print(f"alpha=0: OVLR only (Hard 0-1)")
    print(f"alpha=1: BP only (Cross-Entropy)")
    print("=" * 60 + "\n")

    for alpha in args.alphas:
        print(f"\nRunning alpha = {alpha:.2f}")
        print("-" * 40)

        model = SimpleCNN(num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
        estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)

        test_accs = []
        best_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            if alpha == 0.0:
                # OVLR only
                train_loss, train_acc = train_epoch_ovlr_only(
                    model, train_loader, optimizer, estimator, device)
                mode_str = "OVLR only"
            elif alpha == 1.0:
                # BP only
                train_loss, train_acc = train_epoch_baseline(
                    model, train_loader, optimizer, ce_criterion, device)
                mode_str = "BP only"
            else:
                # Hybrid BP + OVLR
                avg_ce, avg_01, train_acc = train_epoch_hybrid(
                    model, train_loader, optimizer, alpha, estimator, ce_criterion, device)
                train_loss = alpha * avg_ce + (1 - alpha) * avg_01
                mode_str = "BP+OVLR"

            test_acc = test(model, test_loader, device)
            test_accs.append(test_acc)
            best_acc = max(best_acc, test_acc)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Mode: {mode_str:8s} | "
                      f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")

        results[alpha] = {
            'best_acc': best_acc,
            'final_acc': test_accs[-1],
            'all_accs': test_accs,
        }
        print(f"===> Best test accuracy: {best_acc:.2f}%")

    # Plot results
    plt.figure(figsize=(10, 6))
    for alpha, res in results.items():
        if alpha == 0.0:
            label = f"OVLR Only (Hard 0-1): {res['best_acc']:.2f}%"
        elif alpha == 1.0:
            label = f"BP Only (CE): {res['best_acc']:.2f}%"
        else:
            label = f"alpha={alpha:.2f} ({alpha:.0%}CE, {1-alpha:.0%}0-1): {res['best_acc']:.2f}%"
        plt.plot(res['all_accs'], label=label, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Hybrid BP + OVLR Fusion on CIFAR-10')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{args.save_dir}/hybrid_fusion.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {args.save_dir}/hybrid_fusion.png")

    # Summary table
    print("\n" + "=" * 60)
    print("FINAL SUMMARY: Best Test Accuracy")
    print("=" * 60)
    print(f"{'Alpha':>8s} {'Method':>20s} {'Best Acc (%)':>15s}")
    print("-" * 50)
    for alpha, res in sorted(results.items()):
        if alpha == 0.0:
            method = "OVLR Only (0-1)"
        elif alpha == 1.0:
            method = "BP Only (CE)"
        else:
            method = f"Hybrid ({alpha:.2f}CE)"
        print(f"{alpha:8.2f} {method:>20s} {res['best_acc']:15.2f}")


if __name__ == '__main__':
    main()
