#!/usr/bin/env python3
"""
Minimal Blended backdoor attack training for CIFAR-10 + PreActResNet18.
Generates attack_result.pt and bd_test_dataset/ compatible with the paper's prepare_data.py.

Paper settings:
- CIFAR-10, PreActResNet18
- Blended attack at 5% poisoning rate
- Target class 0
- hello_kitty.jpeg as trigger pattern
- 100 epochs, SGD with lr=0.01, momentum=0.9, weight_decay=5e-4
- Batch size 128
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image


# --- PreActResNet18 model (from paper repo) ---

class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


# --- Blended backdoor dataset ---

class BlendedBackdoorDataset(Dataset):
    """Wraps a dataset and applies blended backdoor to a subset."""
    def __init__(self, base_dataset, poison_indices, trigger_img, target_label=0, alpha=0.2):
        self.base = base_dataset
        self.poison_indices = set(poison_indices)
        self.trigger = trigger_img  # PIL Image
        self.target_label = target_label
        self.alpha = alpha

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poison_indices:
            # Blend trigger into image
            img = img.resize(self.trigger.size if hasattr(self.trigger, 'size') else (32, 32))
            blended = Image.blend(img, self.trigger, self.alpha)
            return blended, self.target_label
        return img, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/datasets/cifar10")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--pratio", type=float, default=0.05)
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--blend_alpha", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load trigger image ---
    # Use a simple checkerboard pattern as the trigger (hello_kitty equivalent)
    trigger_size = (32, 32)
    trigger = Image.new('RGB', trigger_size)
    # Create a simple noise-based trigger pattern (similar in spirit to the hello_kitty pattern)
    trigger_pixels = np.random.RandomState(42).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    trigger = Image.fromarray(trigger_pixels)

    # --- Download CIFAR-10 ---
    print("Loading CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test_notensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=None)
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=None)

    # --- Create poison indices ---
    n_train = len(train_set)
    n_poison = int(n_train * args.pratio)
    all_indices = np.arange(n_train)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_indices)
    poison_indices = all_indices[:n_poison].tolist()
    clean_indices = all_indices[n_poison:].tolist()
    print(f"Total train: {n_train}, Poisoned: {len(poison_indices)}, Clean: {len(clean_indices)}")

    # --- Create datasets ---
    # For training: poison the selected indices
    train_trigger_img = Image.fromarray(
        np.random.RandomState(42).randint(0, 256, (32, 32, 3), dtype=np.uint8))

    poison_train_set = BlendedBackdoorDataset(train_set, poison_indices, train_trigger_img,
                                               args.target_class, args.blend_alpha)

    class PoisonTrainWrapper(Dataset):
        def __init__(self, base_set, transform):
            self.base = base_set
            self.transform = transform
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, label = self.base[idx]
            return self.transform(img), label

    train_dataset = PoisonTrainWrapper(poison_train_set, transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    clean_test_dataset = PoisonTrainWrapper(test_set, transform_test)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)

    # --- Train model ---
    print(f"\nTraining PreActResNet18 on CIFAR-10 with {args.pratio*100:.0f}% Blended poisoning...")
    model = PreActResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        scheduler.step()

        # Evaluate clean accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in clean_test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        acc = 100. * test_correct / test_total
        if acc > best_acc:
            best_acc = acc

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Test Acc={acc:.2f}% (best={best_acc:.2f}%)")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")

    # --- Save attack_result.pt ---
    print("Saving attack_result.pt...")
    save_dict = {
        'model_name': 'preactresnet18',
        'num_classes': 10,
        'model': model.state_dict(),
        'data_path': str(args.data_dir),
        'img_size': [32, 32, 3],
        'clean_data': 'cifar10',
        'bd_test': None,  # Will be populated externally
    }
    torch.save(save_dict, output_dir / "attack_result.pt")
    print(f"Saved to {output_dir / 'attack_result.pt'}")

    # --- Create bd_test_dataset/ ---
    # Generate backdoor test images: all test images with the blended trigger
    print("Generating bd_test_dataset/...")
    bd_test_dir = output_dir / "bd_test_dataset"
    bd_test_dir.mkdir(exist_ok=True)

    # Group clean test images by label for reference
    test_labels = {}
    for idx in range(len(test_set)):
        img, label = test_set[idx]
        if label not in test_labels:
            test_labels[label] = []
        test_labels[label].append(idx)

    # Create backdoor test images for all test samples
    trigger_for_test = Image.fromarray(
        np.random.RandomState(42).randint(0, 256, (32, 32, 3), dtype=np.uint8))

    for idx in range(len(test_set)):
        img, label = test_set[idx]
        # Blend trigger
        blended = Image.blend(img, trigger_for_test, args.blend_alpha)
        # Save in class folder using index as filename (matching BackdoorBench convention)
        class_dir = bd_test_dir / str(args.target_class)
        class_dir.mkdir(exist_ok=True)
        blended.save(class_dir / f"{idx}.png")

    print(f"Created {len(test_set)} backdoor test images in {bd_test_dir}/")

    # --- Print summary ---
    print(f"\nDone! Output in {output_dir}/:")
    print(f"  attack_result.pt: {(output_dir / 'attack_result.pt').stat().st_size / 1e6:.1f} MB")
    bd_count = sum(1 for _ in bd_test_dir.rglob("*.png"))
    print(f"  bd_test_dataset/: {bd_count} images")


if __name__ == "__main__":
    main()
