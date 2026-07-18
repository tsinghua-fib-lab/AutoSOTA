import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from common import build_dataset_config, ensure_batch, import_attack_modules, load_model, make_eval_loader, set_random_seeds


PGDAttack = None
BezierAdversarialUnconstrained = None
config = None


def organize_images_by_class(dataloader, model):
    images_by_class = defaultdict(list)
    needed_classes = set(config.fixed_classes.values())
    if "setting_C" in config.fixed_classes:
        needed_classes.update(config.fixed_classes["setting_C"])
    needed_classes = sorted(c for c in needed_classes if isinstance(c, int))

    for idx, (img, label) in enumerate(dataloader):
        img_tensor = img.to(config.device)
        label_tensor = label.to(config.device)

        with torch.no_grad():
            preds = model(config.normalize(img_tensor)).argmax(dim=1)

        for j in range(len(label_tensor)):
            lbl = label_tensor[j].item()
            if preds[j].item() == lbl:
                sample = img_tensor[j] if config.name == "imagenet" else img_tensor
                images_by_class[lbl].append((sample, idx))

        if needed_classes and all(len(images_by_class[c]) >= config.max_per_class for c in needed_classes):
            break

    return images_by_class


def evaluate_bezier_path(model, bezier_obj, delta1, theta, delta2, x1, x2, y1, y2,
                         setting_type="A", num_points=50):
    t_values = torch.linspace(0.02, 0.98, num_points).to(config.device)

    if setting_type == "A":
        x2 = x1
        y2 = y1

    success_x1 = 0
    success_x2 = 0
    success_both = 0

    with torch.no_grad():
        for t in t_values:
            delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
            delta_t = bezier_obj.project_norm_ball(delta_t)

            x1_adv = torch.clamp(x1 + delta_t, 0, 1)
            pred1 = model(config.normalize(x1_adv)).argmax(dim=1).item()
            s1 = pred1 != y1.item()

            if setting_type == "A":
                if s1:
                    success_x1 += 1
                    success_x2 += 1
                    success_both += 1
            else:
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                pred2 = model(config.normalize(x2_adv)).argmax(dim=1).item()
                s2 = pred2 != y2.item()
                success_x1 += int(s1)
                success_x2 += int(s2)
                success_both += int(s1 and s2)

    return {
        "success_rate_x1": success_x1 / num_points,
        "success_rate_x2": success_x2 / num_points,
        "success_rate_both": success_both / num_points,
        "success_rate_avg": (success_x1 + success_x2) / (2 * num_points),
    }


def collect_samples_setting_A(images_by_class, model, pgd_attack, bezier, target_samples=25):
    class_id = config.fixed_classes["setting_A"]
    if class_id not in images_by_class:
        print(f"    ERROR: Class {class_id} not available")
        return []

    available_images = images_by_class[class_id]
    print(f"    Setting A: Using class {class_id} with {len(available_images)} available images")

    samples = []
    attempts = 0
    max_attempts = min(len(available_images) * 10, 500)
    pbar = tqdm(total=target_samples, desc="    Collecting Setting A samples")

    while len(samples) < target_samples and attempts < max_attempts:
        img_idx = attempts % len(available_images)
        x = ensure_batch(available_images[img_idx][0])
        y = torch.tensor([class_id]).to(config.device)
        attempts += 1

        delta1 = pgd_attack.perturb(x, y)
        delta2 = pgd_attack.perturb(x, y)

        with torch.no_grad():
            pred_d1 = model(config.normalize(torch.clamp(x + delta1, 0, 1))).argmax(dim=1)
            pred_d2 = model(config.normalize(torch.clamp(x + delta2, 0, 1))).argmax(dim=1)
            if pred_d1 == y or pred_d2 == y:
                continue

        theta, _, _, theta_norms = bezier.optimize_setting_A(x, y, delta1, delta2)
        eval_results = evaluate_bezier_path(model, bezier, delta1, theta, delta2, x, x, y, y, setting_type="A")

        samples.append({
            "success_rate": eval_results["success_rate_avg"],
            "detailed_results": eval_results,
            "theta_norm": theta_norms[-1],
            "image_idx": img_idx,
        })
        pbar.update(1)

    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting A (attempts: {attempts})")
    return samples


def collect_samples_setting_B(images_by_class, model, pgd_attack, bezier, target_samples=25):
    class_id = config.fixed_classes["setting_B"]
    if class_id not in images_by_class:
        print(f"    ERROR: Class {class_id} not available")
        return []

    available_images = images_by_class[class_id]
    print(f"    Setting B: Using class {class_id} with {len(available_images)} available images")

    if len(available_images) < 2:
        print("    ERROR: Need at least 2 images for Setting B")
        return []

    samples = []
    attempts = 0
    max_attempts = min(len(available_images) * len(available_images), 500)
    pbar = tqdm(total=target_samples, desc="    Collecting Setting B samples")

    while len(samples) < target_samples and attempts < max_attempts:
        idx1 = attempts % len(available_images)
        idx2 = (attempts + 1 + (attempts // len(available_images))) % len(available_images)
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(available_images)

        x1 = ensure_batch(available_images[idx1][0])
        x2 = ensure_batch(available_images[idx2][0])
        y = torch.tensor([class_id]).to(config.device)
        attempts += 1

        delta1 = pgd_attack.perturb(x1, y)
        delta2 = pgd_attack.perturb(x2, y)

        with torch.no_grad():
            pred1 = model(config.normalize(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
            pred2 = model(config.normalize(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
            if pred1 == y or pred2 == y:
                continue

        theta, _, _, theta_norms = bezier.optimize_setting_B(x1, x2, y, delta1, delta2)
        eval_results = evaluate_bezier_path(model, bezier, delta1, theta, delta2, x1, x2, y, y, setting_type="B")

        samples.append({
            "success_rate": eval_results["success_rate_both"],
            "detailed_results": eval_results,
            "theta_norm": theta_norms[-1],
            "image_indices": (idx1, idx2),
        })
        pbar.update(1)

    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting B (attempts: {attempts})")
    return samples


def collect_samples_setting_C(images_by_class, model, pgd_attack, bezier, target_samples=25):
    class_id1, class_id2 = config.fixed_classes["setting_C"]
    if class_id1 not in images_by_class or class_id2 not in images_by_class:
        print(f"    ERROR: Classes {class_id1} or {class_id2} not available")
        return []

    available_images1 = images_by_class[class_id1]
    available_images2 = images_by_class[class_id2]
    print(f"    Setting C: Using classes {class_id1} ({len(available_images1)} images) "
          f"and {class_id2} ({len(available_images2)} images)")

    samples = []
    attempts = 0
    max_attempts = min(len(available_images1) * len(available_images2), 500)
    pbar = tqdm(total=target_samples, desc="    Collecting Setting C samples")

    while len(samples) < target_samples and attempts < max_attempts:
        idx1 = attempts % len(available_images1)
        idx2 = attempts % len(available_images2)
        x1 = ensure_batch(available_images1[idx1][0])
        x2 = ensure_batch(available_images2[idx2][0])
        y1 = torch.tensor([class_id1]).to(config.device)
        y2 = torch.tensor([class_id2]).to(config.device)
        attempts += 1

        delta1 = pgd_attack.perturb(x1, y1)
        delta2 = pgd_attack.perturb(x2, y2)

        with torch.no_grad():
            pred1 = model(config.normalize(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
            pred2 = model(config.normalize(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
            if pred1 == y1 or pred2 == y2:
                continue

        theta, _, _, theta_norms = bezier.optimize_setting_C(x1, x2, y1, y2, delta1, delta2)
        eval_results = evaluate_bezier_path(model, bezier, delta1, theta, delta2, x1, x2, y1, y2, setting_type="C")

        samples.append({
            "success_rate": eval_results["success_rate_both"],
            "detailed_results": eval_results,
            "theta_norm": theta_norms[-1],
            "image_indices": (idx1, idx2),
        })
        pbar.update(1)

    pbar.close()
    print(f"    Collected {len(samples)} samples for Setting C (attempts: {attempts})")
    return samples


def run_connectivity_experiments():
    model = load_model(config)
    norms = ["linf", "l2", "l1"]
    pgd_steps = 40
    pgd_alpha_factors = {"linf": 4.0, "l2": 5.0, "l1": 10.0}

    loader = make_eval_loader(config, batch_size=64 if config.name == "imagenet" else 1)

    print("\nOrganizing images by class...")
    images_by_class = organize_images_by_class(loader, model)

    print("\nFixed class availability:")
    required_classes = set([config.fixed_classes["setting_A"], config.fixed_classes["setting_B"]] +
                           list(config.fixed_classes["setting_C"]))
    for class_id in required_classes:
        class_name = config.class_names[class_id]
        if class_id in images_by_class:
            print(f"  Class {class_id} ({class_name}): {len(images_by_class[class_id])} images")
        else:
            print(f"  ERROR: Class {class_id} ({class_name}) not available!")
            return None

    target_samples = 25
    print(f"\nTarget: {target_samples} samples per setting per norm")
    print(f"PGD attack iterations: {pgd_steps} (with community standard alpha)")
    print("Bezier optimization: 30 iterations with lr=0.01")
    print("=" * 80)

    all_results = {}
    for norm in norms:
        print(f"\n{'='*80}")
        print(f"Testing {norm.upper()} norm (epsilon={config.epsilons[norm]})")
        print(f"{'='*80}")

        eps = config.epsilons[norm]
        alpha = eps / pgd_alpha_factors[norm]
        pgd_attack = PGDAttack(
            model, eps=eps, alpha=alpha, num_iter=pgd_steps, norm=norm,
            normalize_fn=config.normalize,
        )
        bezier = BezierAdversarialUnconstrained(
            model, norm=norm, eps=eps, lr=0.01, num_iter=30,
            normalize_fn=config.normalize,
        )

        norm_results = {}
        print("\n  Setting A (Single Image):")
        samples_A = collect_samples_setting_A(images_by_class, model, pgd_attack, bezier, target_samples)
        norm_results["setting_A"] = samples_A

        print("\n  Setting B (Same Class):")
        samples_B = collect_samples_setting_B(images_by_class, model, pgd_attack, bezier, target_samples)
        norm_results["setting_B"] = samples_B

        print("\n  Setting C (Different Classes):")
        samples_C = collect_samples_setting_C(images_by_class, model, pgd_attack, bezier, target_samples)
        norm_results["setting_C"] = samples_C

        all_results[norm] = norm_results
        print(f"\n  {norm.upper()} Summary:")
        print(f"    Setting A: {len(samples_A)} samples collected")
        print(f"    Setting B: {len(samples_B)} samples collected")
        print(f"    Setting C: {len(samples_C)} samples collected")

    return all_results


def print_results(results):
    print("\n" + "=" * 120)
    print("MODE CONNECTIVITY EXPERIMENTS - FIXED CLASSES RESULTS")
    print("=" * 120)

    for norm in ["linf", "l2", "l1"]:
        if norm not in results:
            continue
        print(f"\n{'='*100}")
        print(f"{norm.upper()} NORM RESULTS")
        print(f"{'='*100}")

        for setting in ["setting_A", "setting_B", "setting_C"]:
            if setting not in results[norm]:
                continue
            samples = results[norm][setting]
            print(f"\n{setting}:")
            if not samples:
                print("  No samples collected")
                continue
          
            x1_rates = [s["detailed_results"]["success_rate_x1"] for s in samples]
            x2_rates = [s["detailed_results"]["success_rate_x2"] for s in samples]
            both_rates = [s["detailed_results"]["success_rate_both"] for s in samples]
            avg_rates = [s["detailed_results"]["success_rate_avg"] for s in samples]
            theta_norms = [s["theta_norm"] for s in samples]
            print(f"  Number of samples:     {len(samples)}")
            print(f"  ASR1:                  {np.mean(x1_rates)*100:>6.1f} ± {np.std(x1_rates)*100:<5.1f}%")
            print(f"  ASR2:                  {np.mean(x2_rates)*100:>6.1f} ± {np.std(x2_rates)*100:<5.1f}%")
            print(f"  ASR Both:              {np.mean(both_rates)*100:>6.1f} ± {np.std(both_rates)*100:<5.1f}%")
            print(f"  ASR Avg:               {np.mean(avg_rates)*100:>6.1f} ± {np.std(avg_rates)*100:<5.1f}%")
            print(f"  Control point theta/eps:{np.mean(theta_norms):>6.2f} ± {np.std(theta_norms):<5.2f}")


def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.output_dir, exist_ok=True)
    suffix = "all" if config.name == "imagenet" else ""
    filename = os.path.join(config.output_dir, f"bezier_connectivity_fixed_{suffix + '_' if suffix else ''}{timestamp}.json")

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    results_with_config = {
        "results": convert(results),
        "configuration": {
            "dataset": config.name,
            "fixed_classes": convert(config.fixed_classes),
            "target_samples": 25,
            "pgd_iterations": 40,
            "bezier_iterations": 30,
            "pgd_alpha_factors": {"linf": 4.0, "l2": 5.0, "l1": 10.0},
        },
    }

    with open(filename, "w") as f:
        json.dump(results_with_config, f, indent=4)

    print(f"\nResults saved to {filename}")
    return filename


def parse_args():
    parser = argparse.ArgumentParser(description="Run adversarial mode connectivity experiments.")
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default="resnet18_cifar10_best.pth")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main(args=None):
    global PGDAttack, BezierAdversarialUnconstrained, config
    if args is None:
        args = parse_args()

    set_random_seeds(args.seed)
    config = build_dataset_config(args)
    PGDAttack, BezierAdversarialUnconstrained = import_attack_modules(config)

    print("Bezier Adversarial Curves - Mode Connectivity Experiments (FIXED CLASSES)")
    print("=" * 80)
    print(f"Dataset: {config.name}")
    print("Fixed classes:", config.fixed_classes)
    print("=" * 80)

    results = run_connectivity_experiments()
    if results:
        print_results(results)
        save_results(results)


if __name__ == "__main__":
    main()
