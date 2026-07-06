#!/usr/bin/env python3
"""Reproduction script for AI Engram on CIFAR-10 / ResNet-18.
Target: forget_class=0, alpha=0.6, ToW metric.
"""
import os, sys, json, time
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# Use the installed engram package
from engram import EngramEditor, EditorConfig

# ---------------------------------------------------------------------------
# Configuration (match paper Appendix B.4.1)
# ---------------------------------------------------------------------------
FORGET_CLASS = 0
ALPHA = 0.6
BATCH_SIZE = 128
NUM_CLASSES = 10
DEVICE = "cuda:0"

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2471, 0.2435, 0.2616]

# Paths
MODEL_DIR = "/paper_data/resnet18_cifar10"
DATA_DIR = "/datasets/cifar10"
OUTPUT_DIR = "/repo/repro_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model loading (matches notebook cell 2)
# ---------------------------------------------------------------------------
def load_resnet18_cifar(num_classes):
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

def load_pretrained_model(model_dir, num_classes=10):
    ckpt_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Model not found at %s" % ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    model = load_resnet18_cifar(num_classes)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("[warn] missing keys: %s" % str(missing[:5]))
    if unexpected:
        print("[warn] unexpected keys: %s" % str(unexpected[:5]))
    return model

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def get_cifar10_loaders(data_dir, batch_size=128):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True,
                                  transform=transform_test, download=False)
    test_set = datasets.CIFAR10(root=data_dir, train=False,
                                 transform=transform_test, download=False)
    return train_set, test_set

def get_class_subset_loader(dataset, target_class, batch_size=128):
    targets = torch.tensor(dataset.targets)
    indices = (targets == target_class).nonzero(as_tuple=True)[0].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def compute_classwise_accuracy(model, dataloader, num_classes):
    device = next(model.parameters()).device
    model.eval()
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        for c in range(num_classes):
            mask = (targets == c)
            class_correct[c] += (predicted[mask] == c).sum().item()
            class_total[c] += mask.sum().item()
    acc = {}
    for c in range(num_classes):
        acc[c] = (class_correct[c] / class_total[c]).item() if class_total[c] > 0 else 0.0
    acc["overall"] = (class_correct.sum() / class_total.sum()).item()
    return acc

def compute_tow(acc_unlearned, acc_original, forget_class=0):
    """
    Compute approximate Tug-of-War metric.
    Since we don't have a retrained model, we approximate acc_r(forget) = 0
    and acc_r(retain) from the original model's retain-class accuracy.
    """
    retain_classes = [c for c in range(NUM_CLASSES) if c != forget_class]

    acc_u_forget = acc_unlearned[forget_class]
    acc_r_forget = 0.0

    acc_u_retain = np.mean([acc_unlearned[c] for c in retain_classes])
    acc_r_retain = np.mean([acc_original[c] for c in retain_classes])

    acc_u_test = acc_unlearned["overall"]
    total_per_class = 1000
    acc_r_test = (acc_r_forget * total_per_class +
                  acc_r_retain * total_per_class * len(retain_classes)) / \
                 (total_per_class * NUM_CLASSES)

    tow = (1.0 - abs(acc_u_forget - acc_r_forget)) * \
          (1.0 - abs(acc_u_retain - acc_r_retain)) * \
          (1.0 - abs(acc_u_test - acc_r_test))

    return {
        "ToW": tow,
        "da_forget": abs(acc_u_forget - acc_r_forget),
        "da_retain": abs(acc_u_retain - acc_r_retain),
        "da_test": abs(acc_u_test - acc_r_test),
        "acc_u_forget": acc_u_forget,
        "acc_u_retain": acc_u_retain,
        "acc_u_test": acc_u_test,
        "acc_r_forget": acc_r_forget,
        "acc_r_retain": acc_r_retain,
        "acc_r_test": acc_r_test,
    }

# ---------------------------------------------------------------------------
# Main reproduction pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("AI Engram Reproduction: CIFAR-10 / ResNet-18")
    print("Forget class: %d, Alpha: %.1f" % (FORGET_CLASS, ALPHA))
    print("=" * 70)

    # 1. Load model
    print("\n[1/6] Loading pretrained ResNet-18 model...")
    model = load_pretrained_model(MODEL_DIR, NUM_CLASSES).to(DEVICE)
    model.eval()
    print("  Model loaded on %s" % DEVICE)

    # 2. Load data
    print("\n[2/6] Loading CIFAR-10 dataset...")
    train_set, test_set = get_cifar10_loaders(DATA_DIR, BATCH_SIZE)
    print("  Train: %d samples, Test: %d samples" % (len(train_set), len(test_set)))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Evaluate original model
    print("\n[3/6] Evaluating original model...")
    acc_original = compute_classwise_accuracy(model, test_loader, NUM_CLASSES)
    print("  Overall: %.4f" % acc_original["overall"])
    for c in range(NUM_CLASSES):
        print("  Class %d: %.4f" % (c, acc_original[c]))

    # 4. Extract engram
    print("\n[4/6] Extracting engram for class %d..." % FORGET_CLASS)
    editor = EngramEditor(model, EditorConfig())

    forget_loader = get_class_subset_loader(train_set, FORGET_CLASS, BATCH_SIZE)
    print("  Forget set: %d samples" % len(forget_loader.dataset))
    target_stats = editor.collect_statistics(forget_loader)

    all_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("  Total set: %d samples" % len(all_loader.dataset))
    total_stats = editor.collect_statistics(all_loader)

    # 5. Apply unlearning with alpha
    print("\n[5/6] Applying unlearning with alpha=%.1f..." % ALPHA)
    unlearned_model = editor.edit(target_stats, total_stats, alpha=ALPHA)

    # 6. Evaluate unlearned model
    print("\n[6/6] Evaluating unlearned model...")
    acc_unlearned = compute_classwise_accuracy(unlearned_model, test_loader, NUM_CLASSES)
    print("  Overall: %.4f" % acc_unlearned["overall"])
    for c in range(NUM_CLASSES):
        delta = acc_unlearned[c] - acc_original[c]
        print("  Class %d: %.4f (delta: %+.4f)" % (c, acc_unlearned[c], delta))

    # Compute ToW
    tow_results = compute_tow(acc_unlearned, acc_original, FORGET_CLASS)
    print("\n" + "=" * 40)
    print("ToW: %.4f" % tow_results["ToW"])
    print("  da_forget: %.4f" % tow_results["da_forget"])
    print("  da_retain: %.4f" % tow_results["da_retain"])
    print("  da_test:   %.4f" % tow_results["da_test"])
    print("=" * 40)

    # Save results
    results = {
        "forget_class": FORGET_CLASS,
        "alpha": ALPHA,
        "acc_original": {str(k): v for k, v in acc_original.items()},
        "acc_unlearned": {str(k): v for k, v in acc_unlearned.items()},
        "tow_results": {k: (v if isinstance(v, (int, float, bool)) else str(v))
                        for k, v in tow_results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to %s" % output_path)

    # Also save the unlearned model
    model_path = os.path.join(OUTPUT_DIR, "unlearned_model.pth")
    torch.save(unlearned_model.state_dict(), model_path)
    print("Unlearned model saved to %s" % model_path)

    return results

if __name__ == "__main__":
    results = main()
    print("\nDone!")
