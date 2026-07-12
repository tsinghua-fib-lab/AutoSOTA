"""
OVLR Example: Backpropagation (BP) Time & Memory Profiler

Profiles standard BP training time and memory usage across
different models (ResNet, DenseNet, ViT) and datasets.
"""

import time
import json
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(description='BP Time & Memory Profiler')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100'], help='Dataset')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['resnet18', 'densenet121'],
                        help='Models to profile')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save-dir', type=str, default='./results_bp_profile',
                        help='Save directory')
    return parser.parse_args()


def get_dataset(name, input_size=32):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if name == 'CIFAR10':
        num_classes = 10
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name == 'CIFAR100':
        num_classes = 100
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f'Unsupported dataset: {name}')

    return trainset, testset, num_classes, input_size


def load_model(name, num_classes):
    if name == 'resnet18':
        model = models.resnet18(num_classes=num_classes)
    elif name == 'densenet121':
        model = models.densenet121(num_classes=num_classes)
    elif name.startswith('vit'):
        model = models.vit_b_16(num_classes=num_classes)
    else:
        raise ValueError(f'Unsupported model: {name}')
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, device, epochs):
    model = model.to(device)
    model.train()
    metrics = {'epoch': [], 'acc': [], 'loss': [], 'epoch_time': [],
               'memory_allocated_MB': [], 'memory_reserved_MB': []}

    total_time = 0
    for epoch in range(epochs):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        epoch_total_time = time.perf_counter() - epoch_start_time

        if epoch > 1:  # Skip first 2 epochs for warmup
            total_time += epoch_total_time

        acc = evaluate(model, testloader, device)
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, "
              f"Acc: {acc:.2f}%, Time: {epoch_total_time:.2f}s, "
              f"Mem: {allocated:.1f}MB")

        metrics['epoch'].append(epoch)
        metrics['acc'].append(acc)
        metrics['loss'].append(loss.item())
        metrics['epoch_time'].append(epoch_total_time)
        metrics['memory_allocated_MB'].append(allocated)
        metrics['memory_reserved_MB'].append(reserved)

    valid_epochs = max(1, epochs - 2)
    total_time_mean = total_time / valid_epochs
    max_allocated = max(metrics['memory_allocated_MB'])
    max_reserved = max(metrics['memory_reserved_MB'])

    return metrics, total_time_mean, max_allocated, max_reserved


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    trainset, testset, num_classes, input_size = get_dataset(args.dataset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_results = {}

    for model_name in args.models:
        lr = 0.001 if model_name != 'vit_b_16' else 0.0001
        torch.cuda.empty_cache()

        print(f"\n=== Training {model_name} on {args.dataset} ===")
        model = load_model(model_name, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        metrics, train_time, max_allocated, max_reserved = train_and_evaluate(
            model, trainloader, testloader, criterion, optimizer, device, args.epochs
        )

        result = {
            'dataset': args.dataset,
            'model': model_name,
            'method': 'BP',
            'final_accuracy': metrics['acc'][-1],
            'final_loss': metrics['loss'][-1],
            'train_time_seconds_mean': train_time,
            'max_memory_allocated_MB': max_allocated,
            'max_memory_reserved_MB': max_reserved,
            'epoch_wise': metrics,
        }

        all_results[model_name] = result
        print(f"  Final Acc: {result['final_accuracy']:.2f}%")
        print(f"  Mean Train Time: {train_time:.2f}s/epoch")
        print(f"  Peak Mem: {max_allocated:.1f}MB")

        with open(os.path.join(args.save_dir, f'{args.dataset}_{model_name}_bp.json'), 'w') as f:
            json.dump(result, f, indent=4)

    with open(os.path.join(args.save_dir, f'{args.dataset}_bp_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    main()
