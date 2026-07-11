"""
Evaluation script for ISIC debiasing experiment.
Loads trained models and computes all rubric metrics:
- Benign Accuracy, Malignant Accuracy, Avg Accuracy, Attr

Usage: python eval.py --model_path /models/50_bias/model_seed0_mode_presence_absence_debias.pth [--mode presence_absence_debias]

If --model_path is not specified, evaluates the best model for each seed
and aggregates results across seeds.
"""
import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import add_artificial_bias, create_balanced_subset, dilate_masks_torch
from utils_fast import add_artificial_bias_fast as add_artificial_bias
from x_resnet import xfixup_resnet50

# Compatible sdpa_kernel import (PyTorch 2.1+)
try:
    # Compatible sdpa_kernel import (PyTorch 2.1+)
from contextlib import contextmanager
@contextmanager
def sdpa_kernel(backend=None):
    yield
class SDPBackend:
    MATH = "math"

except (ImportError, ModuleNotFoundError):
    try:
        from torch.backends.cuda import sdp_kernel, SDPBackend
        from contextlib import contextmanager
        @contextmanager
        def sdpa_kernel(backend):
            yield
    except ImportError:
        from contextlib import contextmanager
        @contextmanager
        def sdpa_kernel(backend=None):
            yield
        class SDPBackend:
            MATH = "math"


# ----- Paths -----
DATA_DIR = "/tmp/ISIC2020_2"  # Local copy
XRESNET_PATH = '/models/xfixup_resnet50_model_best.pth.tar'
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = 'cuda:0'

# ----- ImageNet Normalization -----
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]),
}


def load_model(model_path):
    """Load XResNet-50 model with trained weights."""
    model = xfixup_resnet50()
    checkpoint = torch.load(XRESNET_PATH, map_location='cpu')
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Load trained weights
    trained_state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(trained_state)
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_split(model, dataloader, class_bias, split_name, mode):
    """
    Evaluate model on a specific bias split and compute all metrics.
    Returns dict with per-class accuracies and attribution.
    """
    model.eval()

    correct_benign = 0
    total_benign = 0
    correct_malignant = 0
    total_malignant = 0
    total_attr = 0.0
    attr_count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Apply bias
            inputs_biased, patch_segmentation = add_artificial_bias(inputs, labels, class_bias)
            patch_segmentation = dilate_masks_torch(patch_segmentation)

            with sdpa_kernel(SDPBackend.MATH):
                inputs_biased.requires_grad = True
                outputs = model(inputs_biased)
                preds = torch.argmax(outputs, dim=1)

                # Per-class accuracy
                benign_mask = (labels == 0)
                malignant_mask = (labels == 1)

                correct_benign += (preds[benign_mask] == labels[benign_mask]).sum().item()
                total_benign += benign_mask.sum().item()
                correct_malignant += (preds[malignant_mask] == labels[malignant_mask]).sum().item()
                total_malignant += malignant_mask.sum().item()

                # Compute attribution for both target and non-target classes
                if mode in ('presence_debias', 'presence_absence_debias'):
                    # Target attribution
                    labels_one_hot = F.one_hot(labels, num_classes=2).float()
                    target_outputs = torch.gather(outputs, 1, labels.unsqueeze(-1))
                    gradients_target = torch.autograd.grad(
                        torch.unbind(target_outputs), inputs_biased,
                        create_graph=False, retain_graph=True
                    )[0]
                    gradients_target = inputs_biased * gradients_target

                    attr_inside_target = (
                        (gradients_target.abs().sum(dim=1, keepdim=True) * patch_segmentation).sum()
                    ) / (patch_segmentation.sum() + 1e-5)

                    if mode == 'presence_absence_debias':
                        # Non-target attribution
                        labels_flipped = 1 - labels
                        target_outputs_nt = torch.gather(outputs, 1, labels_flipped.unsqueeze(-1))
                        gradients_nt = torch.autograd.grad(
                            torch.unbind(target_outputs_nt), inputs_biased,
                            create_graph=False, retain_graph=True
                        )[0]
                        gradients_nt = inputs_biased * gradients_nt

                        attr_inside_nt = (
                            (gradients_nt.abs().sum(dim=1, keepdim=True) * patch_segmentation).sum()
                        ) / (patch_segmentation.sum() + 1e-5)

                        # Average of both (matching training loss)
                        attr = (attr_inside_target + attr_inside_nt) / 2
                    else:
                        attr = attr_inside_target

                    total_attr += attr.item()
                    attr_count += 1

    benign_acc = correct_benign / max(total_benign, 1)
    malignant_acc = correct_malignant / max(total_malignant, 1)
    avg_acc = (benign_acc + malignant_acc) / 2
    mean_attr = total_attr / max(attr_count, 1)

    return {
        'split': split_name,
        'benign_acc': benign_acc,
        'malignant_acc': malignant_acc,
        'avg_acc': avg_acc,
        'attr': mean_attr,
        'total_benign': total_benign,
        'total_malignant': total_malignant,
    }


def evaluate_model(model_path, mode='presence_absence_debias'):
    """Evaluate a single model on all bias splits and return metrics."""
    model = load_model(model_path)

    # Setup data
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transform=data_transforms[x])
        for x in ['train', 'val']
    }

    val_dataset = create_balanced_subset(
        image_datasets['val'], targets=image_datasets['val'].targets
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Bias configurations
    biases = {
        'train_bias': {0: 1.0, 1: 0.0},   # 100% benign get patch
        'inverse_bias': {0: 0.0, 1: 1.0},  # 100% malignant get patch
        'no_bias': {0: 0.0, 1: 0.0},       # No patches
    }

    results = {}
    for split_name, class_bias in biases.items():
        result = evaluate_split(model, val_loader, class_bias, split_name, mode)
        results[split_name] = result
        print(f"  {split_name}: benign_acc={result['benign_acc']:.4f}, "
              f"malignant_acc={result['malignant_acc']:.4f}, "
              f"avg_acc={result['avg_acc']:.4f}, attr={result['attr']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a specific model checkpoint')
    parser.add_argument('--mode', type=str, default='presence_absence_debias',
                        choices=['default', 'presence_debias', 'presence_absence_debias'])
    parser.add_argument('--store_dir', type=str, default='/models/50_bias',
                        help='Directory containing trained model checkpoints')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of training runs (seeds)')
    args = parser.parse_args()

    if args.model_path:
        print(f"Evaluating single model: {args.model_path}")
        results = evaluate_model(args.model_path, args.mode)

        # Print final metrics for inverse_bias (primary rubric metric)
        inv = results.get('inverse_bias', {})
        print(f"\n=== FINAL METRICS (inverse_bias) ===")
        print(f"Benign Accuracy: {inv.get('benign_acc', 0):.4f}")
        print(f"Malignant Accuracy: {inv.get('malignant_acc', 0):.4f}")
        print(f"Avg Accuracy: {inv.get('avg_acc', 0):.4f}")
        print(f"Attr: {inv.get('attr', 0):.4f}")
    else:
        # Evaluate all seeds
        print(f"Evaluating {args.runs} seeds for mode={args.mode}...")
        all_results = {s: {} for s in ['inverse_bias', 'train_bias', 'no_bias']}

        for seed in range(args.runs):
            model_path = os.path.join(
                args.store_dir, f'model_seed{seed}_mode_{args.mode}.pth'
            )
            if not os.path.exists(model_path):
                print(f"  Model not found: {model_path}")
                continue

            print(f"\n=== Seed {seed} ===")
            results = evaluate_model(model_path, args.mode)
            for split, metrics in results.items():
                if split not in all_results:
                    all_results[split] = {}
                for k, v in metrics.items():
                    if k not in ('split', 'total_benign', 'total_malignant'):
                        all_results[split].setdefault(k, []).append(v)

        # Aggregate across seeds
        print(f"\n=== AGGREGATED RESULTS (mean ± std over seeds) ===")
        for split in ['inverse_bias', 'train_bias', 'no_bias']:
            if not all_results.get(split):
                continue
            print(f"\n{split}:")
            for metric in ['benign_acc', 'malignant_acc', 'avg_acc', 'attr']:
                vals = all_results[split].get(metric, [])
                if vals:
                    mean = sum(vals) / len(vals)
                    std = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
                    print(f"  {metric}: {mean:.4f} ± {std:.4f}")


if __name__ == '__main__':
    main()
