#!/usr/bin/env python3
"""Reproduction script for AI Engram on CIFAR-10 / ResNet-18.
Uses the notebook-inlined engram code (which includes Conv2d support).
Target: forget_class=0, alpha=0.6, ToW metric.
"""
import os, sys, json, time, copy
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# Import the notebook-inlined code
sys.path.insert(0, '/repo')
from notebook_engram import EngramEditor, EditorConfig
from notebook_model import load_resnet18_cifar

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
MODEL_LOCAL_PATH = "/paper_data/resnet18_cifar10/pytorch_model.bin"
DATA_DIR = "/datasets/cifar10"
OUTPUT_DIR = "/repo/repro_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Local model loading
# ---------------------------------------------------------------------------
def load_model_local(num_classes=10):
    if not os.path.exists(MODEL_LOCAL_PATH):
        raise FileNotFoundError("Model not found at %s" % MODEL_LOCAL_PATH)
    obj = torch.load(MODEL_LOCAL_PATH, map_location="cpu")
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
    m = load_resnet18_cifar(num_classes)
    missing, unexpected = m.load_state_dict(cleaned, strict=False)
    if missing:
        print("[warn] missing keys: %s" % str(missing[:5]))
    if unexpected:
        print("[warn] unexpected keys: %s" % str(unexpected[:5]))
    return m

# ---------------------------------------------------------------------------
# Local data loading
# ---------------------------------------------------------------------------
def get_data(split='train'):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    is_train = (split == 'train')
    return datasets.CIFAR10(root=DATA_DIR, train=is_train,
                             transform=transform_test, download=False)

def get_class_loader(split='train', targets=None, batch_size=128, num_samples=None):
    dataset = get_data(split=split)
    if targets is not None:
        all_targets = torch.tensor(dataset.targets)
        mask = torch.zeros(len(all_targets), dtype=torch.bool)
        for t in targets:
            mask = mask | (all_targets == t)
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        if num_samples is not None and num_samples < len(indices):
            indices = indices[:num_samples]
        subset = Subset(dataset, indices)
    else:
        subset = dataset
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def eval_classwise_accuracy(model, dataloader, num_classes):
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
    """Approximate ToW: retrained model is assumed to have 0 acc on forget class."""
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
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("AI Engram Reproduction v2: CIFAR-10 / ResNet-18")
    print("Forget class: %d, Alpha: %.1f" % (FORGET_CLASS, ALPHA))
    print("Using notebook-inlined engram code (with Conv2d support)")
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading pretrained ResNet-18 model...")
    base_model = load_model_local(NUM_CLASSES).to(DEVICE)
    base_model.eval()
    print("  Model loaded on %s" % DEVICE)

    # 2. Evaluate original model
    print("\n[2/5] Evaluating original model...")
    test_loader = get_class_loader(split='test', targets=list(range(NUM_CLASSES)),
                                    batch_size=BATCH_SIZE)
    acc_original = eval_classwise_accuracy(base_model, test_loader, NUM_CLASSES)
    print("  Overall: %.4f" % acc_original["overall"])
    for c in range(NUM_CLASSES):
        print("  Class %d: %.4f" % (c, acc_original[c]))

    # 3. Quick validation (small-scale sanity check)
    print("\n[3/5] Quick validation (single-class, small samples)...")
    _PROBE = [0, 1, 2, 3, 4]
    _VAL_T = _PROBE[0]
    _NS = 256
    _vm = load_model_local(NUM_CLASSES).to(DEVICE)
    _ve = EngramEditor(_vm, EditorConfig())
    _pos = _ve.collect_statistics(
        get_class_loader(split="train", targets=[_VAL_T], num_samples=_NS))
    _tot = _ve.merge_statistics(*[
        _ve.collect_statistics(get_class_loader(split="train", targets=[c], num_samples=_NS))
        for c in _PROBE])
    _test = get_class_loader(split="test", targets=_PROBE, batch_size=BATCH_SIZE)
    _a0 = eval_classwise_accuracy(_vm, _test, max(_PROBE) + 1)
    _a1 = eval_classwise_accuracy(_ve.edit(_pos, _tot), _test, max(_PROBE) + 1)
    print("  [validation] class %d acc: %.3f -> %.3f" % (_VAL_T, _a0.get(_VAL_T, 0), _a1.get(_VAL_T, 0)))
    print("  (expect significant drop = unlearning works)")
    del _vm, _ve, _pos, _tot
    torch.cuda.empty_cache()

    # 4. Full engram extraction and unlearning
    print("\n[4/5] Full engram extraction for class %d with alpha=%.1f..." % (FORGET_CLASS, ALPHA))
    editor = EngramEditor(base_model, EditorConfig())

    print("  Collecting target statistics (class %d)..." % FORGET_CLASS)
    forget_loader = get_class_loader(split='train', targets=[FORGET_CLASS])
    print("  Forget set: %d samples" % len(forget_loader.dataset))
    target_stats = editor.collect_statistics(forget_loader)

    print("  Collecting total statistics (all classes)...")
    all_loader = get_class_loader(split='train', targets=list(range(NUM_CLASSES)))
    print("  Total set: %d samples" % len(all_loader.dataset))
    total_stats = editor.collect_statistics(all_loader)

    print("  Applying unlearning with edit_strength=%.1f..." % ALPHA)
    unlearned_model = editor.edit(target_stats, total_stats, edit_strength=ALPHA)

    # 5. Evaluate unlearned model
    print("\n[5/5] Evaluating unlearned model...")
    acc_unlearned = eval_classwise_accuracy(unlearned_model, test_loader, NUM_CLASSES)
    print("  Overall: %.4f" % acc_unlearned["overall"])
    for c in range(NUM_CLASSES):
        delta = acc_unlearned[c] - acc_original[c]
        print("  Class %d: %.4f (delta: %+.4f)" % (c, acc_unlearned[c], delta))

    # Compute ToW
    tow_results = compute_tow(acc_unlearned, acc_original, FORGET_CLASS)
    print("\n" + "=" * 50)
    print("REPRODUCTION RESULTS")
    print("=" * 50)
    print("ToW (approximate): %.4f" % tow_results["ToW"])
    print("  Paper reports ToW=0.984 for Engram(alpha_best)")
    print("  Reproduce CI bounds: [0.956, 0.9868]")
    print("  da_forget: %.4f  (acc_u_forget=%.4f)" % (tow_results["da_forget"], tow_results["acc_u_forget"]))
    print("  da_retain: %.4f  (acc_u_retain=%.4f, acc_r_retain=%.4f)" % (tow_results["da_retain"], tow_results["acc_u_retain"], tow_results["acc_r_retain"]))
    print("  da_test:   %.4f  (acc_u_test=%.4f, acc_r_test=%.4f)" % (tow_results["da_test"], tow_results["acc_u_test"], tow_results["acc_r_test"]))
    print("=" * 50)

    # Save results
    results = {
        "forget_class": FORGET_CLASS,
        "alpha": ALPHA,
        "acc_original": {str(k): v for k, v in acc_original.items()},
        "acc_unlearned": {str(k): v for k, v in acc_unlearned.items()},
        "tow_results": {k: v for k, v in tow_results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_command": "cd /repo && python3 reproduce_v2.py",
    }

    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to %s" % output_path)

    return results

if __name__ == "__main__":
    results = main()
    print("\nDone!")
