#!/usr/bin/env python3
"""Optimized reproduction with multiple engram improvement strategies.
Supports: --shrinkage, --weight-norm, --contrastive, --fisher, --alpha X, --idea-id ID
"""
import os, sys, json, time, copy, argparse
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/repo")
from notebook_engram import EngramEditor, EditorConfig
from notebook_model import load_resnet18_cifar

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FORGET_CLASS = 0
BATCH_SIZE = 128
NUM_CLASSES = 10
DEVICE = "cuda:0"

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2471, 0.2435, 0.2616]

MODEL_LOCAL_PATH = "/paper_data/resnet18_cifar10/pytorch_model.bin"
DATA_DIR = "/datasets/cifar10"
OUTPUT_DIR = "/repo/repro_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model / Data loading
# ---------------------------------------------------------------------------
def load_model_local(num_classes=10):
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
    m.load_state_dict(cleaned, strict=False)
    return m

def get_data(split="train"):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    is_train = (split == "train")
    return datasets.CIFAR10(root=DATA_DIR, train=is_train,
                             transform=transform_test, download=False)

def get_class_loader(split="train", targets=None, batch_size=128, num_samples=None):
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
# Optimization helpers
# ---------------------------------------------------------------------------
def compute_shrinkage_intensity(cov, n_samples):
    """OAS shrinkage intensity towards identity."""
    d = cov.shape[0]
    trS = torch.trace(cov)
    trS2 = torch.trace(cov @ cov)
    num = (1.0 - 2.0 / d) * trS2 + trS**2
    den = (n_samples + 1.0 - 2.0 / d) * (trS2 - trS**2 / d)
    if den <= 0:
        return 0.0
    rho = (num / den).item()
    return min(max(rho, 0.0), 1.0)

def apply_shrinkage(cov, n_samples):
    """Apply OAS shrinkage to a covariance matrix."""
    d = cov.shape[0]
    shrinkage = compute_shrinkage_intensity(cov, n_samples)
    trace_cov = torch.trace(cov)
    if shrinkage <= 0:
        return cov
    target = (trace_cov / d) * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return (1.0 - shrinkage) * cov + shrinkage * target

def compute_weight_norm_scales(engram_weights, model):
    """Per-layer Frobenius norm scaling: scale_l = ||P_l||/||W_l||, normalized."""
    modules = dict(model.named_modules())
    scales = {}
    max_scale = 0.0
    for name, w_engram in engram_weights.items():
        if name in modules:
            w_orig = modules[name].weight
            if isinstance(modules[name], nn.Conv2d):
                w_norm = w_orig.reshape(w_orig.shape[0], -1).norm(p="fro")
            else:
                w_norm = w_orig.norm(p="fro")
            p_norm = w_engram.norm(p="fro")
            if w_norm > 0:
                scale = (p_norm / w_norm).item()
            else:
                scale = 0.0
            scales[name] = scale
            max_scale = max(max_scale, scale)
    if max_scale > 0:
        for name in scales:
            scales[name] /= max_scale
    return scales

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimized AI Engram reproduction")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--shrinkage", action="store_true")
    parser.add_argument("--weight-norm", action="store_true")
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--idea-id", type=str, default="OPT")
    args = parser.parse_args()

    flags = []
    if args.shrinkage: flags.append("shrink")
    if args.weight_norm: flags.append("wnorm")
    if args.contrastive: flags.append("contrast")
    flag_str = "+".join(flags) if flags else "baseline"

    print("=" * 70)
    print("AI Engram Optimized: CIFAR-10 / ResNet-18")
    print("Forget class: %d, Alpha: %.2f, Opts: %s" % (FORGET_CLASS, args.alpha, flag_str))
    print("=" * 70)

    # 1. Load model
    print("\n[1/5] Loading model...")
    base_model = load_model_local(NUM_CLASSES).to(DEVICE)
    base_model.eval()
    print("  Model on %s" % DEVICE)

    # 2. Eval original
    print("\n[2/5] Evaluating original model...")
    test_loader = get_class_loader(split="test", targets=list(range(NUM_CLASSES)), batch_size=BATCH_SIZE)
    acc_original = eval_classwise_accuracy(base_model, test_loader, NUM_CLASSES)
    print("  Overall: %.4f" % acc_original["overall"])

    # 3. Collect statistics
    print("\n[3/5] Collecting statistics...")
    editor = EngramEditor(base_model, EditorConfig())

    if args.contrastive:
        retain_targets = [c for c in range(NUM_CLASSES) if c != FORGET_CLASS]
        all_loader = get_class_loader(split="train", targets=retain_targets)
    else:
        all_loader = get_class_loader(split="train", targets=list(range(NUM_CLASSES)))
    n_total = len(all_loader.dataset)
    print("  Total samples: %d" % n_total)
    total_stats = editor.collect_statistics(all_loader)

    forget_loader = get_class_loader(split="train", targets=[FORGET_CLASS])
    n_forget = len(forget_loader.dataset)
    print("  Forget samples: %d" % n_forget)
    target_stats = editor.collect_statistics(forget_loader)

    # 4. Compute engram weights with optimizations
    print("\n[4/5] Computing engram weights...")
    engram_weights = {}
    modules = dict(base_model.named_modules())
    prec = editor.config.precision
    dev = editor.config.device

    for layer_name, pos_cov in tqdm(target_stats.items(), desc="Computing Engrams"):
        if layer_name not in total_stats:
            continue
        module = modules[layer_name]
        sum_cov = total_stats[layer_name].to(dev, dtype=prec)
        pos_cov = pos_cov.to(dev, dtype=prec)

        # --- SHRINKAGE ---
        if args.shrinkage:
            sum_cov = apply_shrinkage(sum_cov, n_total)

        orig_w = module.weight.to(prec)
        orig_shape = orig_w.shape
        if isinstance(module, nn.Conv2d):
            orig_w = orig_w.reshape(orig_shape[0], -1)

        w_engram = orig_w @ pos_cov @ torch.linalg.pinv(sum_cov)
        engram_weights[layer_name] = w_engram.reshape(orig_shape)

    # --- Weight norm scaling ---
    weight_norm_scales = None
    if args.weight_norm:
        weight_norm_scales = compute_weight_norm_scales(engram_weights, base_model)
        svals = list(weight_norm_scales.values())
        print("  Weight norm scales: min=%.4f max=%.4f mean=%.4f" % (min(svals), max(svals), sum(svals)/len(svals)))

    # 5. Apply edit
    print("\n[5/5] Applying edit (alpha=%.2f)..." % args.alpha)
    edit_model = copy.deepcopy(base_model)
    edit_modules = dict(edit_model.named_modules())

    with torch.no_grad():
        for layer_name, w_engram in engram_weights.items():
            if layer_name not in edit_modules:
                continue
            module = edit_modules[layer_name]
            alpha_eff = args.alpha
            if weight_norm_scales and layer_name in weight_norm_scales:
                alpha_eff *= weight_norm_scales[layer_name]
            update = (alpha_eff * w_engram).to(module.weight.dtype)
            module.weight.copy_(module.weight - update)

    # 6. Evaluate
    print("\n[*] Evaluating unlearned model...")
    acc_unlearned = eval_classwise_accuracy(edit_model, test_loader, NUM_CLASSES)
    print("  Overall: %.4f" % acc_unlearned["overall"])
    for c in range(NUM_CLASSES):
        delta = acc_unlearned[c] - acc_original[c]
        print("  Class %d: %.4f (delta: %+.4f)" % (c, acc_unlearned[c], delta))

    tow_results = compute_tow(acc_unlearned, acc_original, FORGET_CLASS)
    print("\n" + "=" * 50)
    print("RESULTS [%s]" % flag_str)
    print("=" * 50)
    print("ToW: %.4f" % tow_results["ToW"])
    print("  forget_acc: %.4f  retain_acc: %.4f  test_acc: %.4f" % (
        tow_results["acc_u_forget"], tow_results["acc_u_retain"], tow_results["acc_u_test"]))
    print("  da_forget: %.4f  da_retain: %.4f  da_test: %.4f" % (
        tow_results["da_forget"], tow_results["da_retain"], tow_results["da_test"]))
    print("=" * 50)

    # Save
    results = {
        "forget_class": FORGET_CLASS,
        "alpha": args.alpha,
        "optimizations": flag_str,
        "acc_original": {str(k): v for k, v in acc_original.items()},
        "acc_unlearned": {str(k): v for k, v in acc_unlearned.items()},
        "tow_results": {k: v for k, v in tow_results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "results_opt.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    results = main()
    print("\nDone!")
