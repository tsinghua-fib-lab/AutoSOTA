"""
Reproduction evaluation script for SMART on CIFAR-100 ResNet-50.

Loads the pre-trained ResNet-50 from Mukhoti et al. (2020), computes logits,
applies SMART calibration, and reports ECE, AdaECE, Accuracy.

Usage: python3 evaluate.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, "/repo")
from smart import SMART
from metrics import expected_calibration_error, accuracy, negative_log_likelihood


# =============================================================================
# CIFAR ResNet-50
# =============================================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class CIFARResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# =============================================================================
# Adaptive ECE
# =============================================================================

def adaptive_ece(logits, labels, n_bins=15):
    if isinstance(logits, np.ndarray):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    if isinstance(labels, np.ndarray):
        labels = torch.as_tensor(labels, dtype=torch.long)

    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    n = len(confidences)
    sorted_idx = torch.argsort(confidences)
    ece = 0.0
    for i in range(n_bins):
        start = i * n // n_bins
        end = (i + 1) * n // n_bins
        bin_idx = sorted_idx[start:end]
        if len(bin_idx) > 0:
            ece += (len(bin_idx) / n) * abs(
                confidences[bin_idx].mean() - accuracies[bin_idx].mean()
            ).item()
    return ece


# =============================================================================
# Main
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load model ----
    print("Loading pre-trained CIFAR-100 ResNet-50 from Mukhoti et al. (2020)...")
    model = CIFARResNet(Bottleneck, [3, 4, 6, 3], num_classes=100)
    state_dict = torch.load(
        "/models/focal_calibration/resnet50_cross_entropy_350.model",
        map_location="cpu",
    )
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model = model.to(device)
    model.eval()

    # ---- Load data ----
    print("Loading CIFAR-100 test set...")
    CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR100_STD = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    test_dataset = datasets.CIFAR100(
        root="/datasets", train=False, download=False, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    # ---- Compute logits ----
    print("Computing test logits...")
    all_logits, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            all_logits.append(outputs.cpu())
            all_labels.append(targets)
    test_logits = torch.cat(all_logits)
    test_labels = torch.cat(all_labels)

    # ---- Vanilla ----
    print()
    print("=" * 60)
    print("VANILLA (uncalibrated)")
    print("=" * 60)
    vanilla_ece = expected_calibration_error(logits=test_logits, labels=test_labels, n_bins=15)
    vanilla_adaece = adaptive_ece(test_logits, test_labels, n_bins=15)
    vanilla_acc = accuracy(test_logits, test_labels)
    vanilla_nll = negative_log_likelihood(test_logits, test_labels)
    print(f"  ECE:    {vanilla_ece * 100:.2f}%")
    print(f"  AdaECE: {vanilla_adaece * 100:.2f}%")
    print(f"  Acc:    {vanilla_acc * 100:.2f}%")
    print(f"  NLL:    {vanilla_nll:.4f}")

    # ---- SMART ----
    n_seeds = 5
    n_bins = 15
    calib_size = 2000
    n_test = len(test_labels)

    print()
    print("=" * 60)
    print(f"SMART CALIBRATION (5 splits, 2000 calib / 8000 eval, 15 bins)")
    print("=" * 60)

    ece_vals, adaece_vals, acc_vals, nll_vals = [], [], [], []

    for seed in range(1, n_seeds + 1):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_test)
        cal_idx = perm[:calib_size]
        eval_idx = perm[calib_size:]

        calibrator = SMART(
            hidden_dim=16, nlayers=2, lr=5e-3, epochs=2000,
            loss="smooth_soft_ece", n_bins=n_bins, seed=seed,
            batch_size=None, device=device,
            sigma=0.04, delta=1e-3,
            early_stopping=True, patience=200, min_delta=1e-4,
            normalize_margins=True, verbose=False,
        )
        calibrator.fit(test_logits[cal_idx].numpy(), test_labels[cal_idx].numpy())

        cal_eval_logits = calibrator.calibrate(
            test_logits[eval_idx].numpy(), return_logits=True
        )
        cal_eval_logits = torch.as_tensor(cal_eval_logits, dtype=torch.float32)

        ece = expected_calibration_error(
            logits=cal_eval_logits, labels=test_labels[eval_idx], n_bins=n_bins
        )
        adaece = adaptive_ece(cal_eval_logits, test_labels[eval_idx], n_bins=n_bins)
        acc = accuracy(cal_eval_logits, test_labels[eval_idx])
        nll = negative_log_likelihood(cal_eval_logits, test_labels[eval_idx])

        ece_vals.append(ece * 100)
        adaece_vals.append(adaece * 100)
        acc_vals.append(acc * 100)
        nll_vals.append(nll)

        print(f"  Seed {seed}: ECE={ece*100:.2f}%  AdaECE={adaece*100:.2f}%  Acc={acc*100:.2f}%  NLL={nll:.4f}")

    # ---- Final results ----
    print()
    print("=" * 60)
    print("FINAL RESULTS (mean +/- std over 5 seeds)")
    print("=" * 60)
    print(f"  ECE:    {np.mean(ece_vals):.2f} +/- {np.std(ece_vals):.2f}%")
    print(f"  AdaECE: {np.mean(adaece_vals):.2f} +/- {np.std(adaece_vals):.2f}%")
    print(f"  Acc:    {np.mean(acc_vals):.2f} +/- {np.std(acc_vals):.2f}%")
    print(f"  NLL:    {np.mean(nll_vals):.4f} +/- {np.std(nll_vals):.4f}")

    print()
    print("=" * 60)
    print("PAPER REFERENCE (CIFAR-100 ResNet-50)")
    print("=" * 60)
    print("  Vanilla ECE: 17.53%   Vanilla Acc: 76.69%")
    print("  SMART ECE:   1.37%    SMART Acc:  76.69%")
    print("  SMART AdaECE: 2.27%")
    print()
    print("Reproduction completed successfully.")


if __name__ == "__main__":
    main()
