"""
OVLR Example: Direct Optimization of Hard 0-1 Loss

This script demonstrates OVLR's ability to directly optimize the
non-differentiable hard 0-1 accuracy loss, which standard backpropagation
cannot handle effectively.

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
from torchvision import datasets, transforms, models


def hard_01_loss(outputs, targets):
    """Hard 0-1 loss: 0 if correct, 1 if incorrect. Non-differentiable."""
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


def train(model, loader, optimizer, estimator, device, epoch):
    """Train one epoch with OVLR."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
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
    print(f"Train Epoch: {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def test(model, loader, device, epoch):
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

    accuracy = 100. * correct / total
    print(f"Test Epoch: {epoch} | Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='OVLR: Hard 0-1 Loss Optimization')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n-repeat', type=int, default=200, help='number of noisy samples')
    parser.add_argument('--noise-scale', type=float, default=1.0, help='noise scale (sigma)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST'])
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4) if args.dataset == 'CIFAR10' else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip() if args.dataset == 'CIFAR10' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if args.dataset == 'CIFAR10' else transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if args.dataset == 'CIFAR10' else transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if args.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    else:
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = SimpleCNN(num_classes=num_classes).to(device)

    # OVLR setup
    from ovlr import OVLRGradientEstimator, get_noise_fn
    noise_fn = get_noise_fn(mode="symmetric", noise_scale=args.noise_scale)
    estimator = OVLRGradientEstimator(noise_fn, n_repeat=args.n_repeat)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nOVLR Hard 0-1 Loss Training")
    print(f"Dataset: {args.dataset}, n_repeat: {args.n_repeat}, sigma: {args.noise_scale}")
    print("=" * 60)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, estimator, device, epoch)
        acc = test(model, test_loader, device, epoch)
        best_acc = max(best_acc, acc)

    print(f"\nBest test accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
