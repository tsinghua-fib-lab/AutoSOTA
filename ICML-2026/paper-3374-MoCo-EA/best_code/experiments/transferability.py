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


def transferability_settings():
    if config.name == "imagenet":
        return {
            "epsilons": {"linf": 8 / 255, "l2": 4.0, "l1": 300.0},
            "max_per_class": 50,
            "test_slice": (0, 10),
            "train_start": 10,
            "test_set_size": 10,
            "data_layout": {
                "test_set": "[0-9]",
                "training_pool": "[10+]",
            },
        }
    return {
        "epsilons": {"linf": 8 / 255, "l2": 0.5, "l1": 10.0},
        "max_per_class": 300,
        "test_slice": (30, 130),
        "train_start": 130,
        "test_set_size": 100,
        "data_layout": {
            "test_set": "[30-129]",
            "training_pool": "[130+]",
        },
    }


def organize_images_by_class(dataloader, model, max_per_class):
    images_by_class = defaultdict(list)
    needed_classes = set([config.fixed_classes["setting_A"], config.fixed_classes["setting_B"]])
    needed_classes.update(config.fixed_classes["setting_C"])

    for idx, (img, label) in enumerate(dataloader):
        img_tensor = img.to(config.device)
        label_tensor = label.to(config.device)

        with torch.no_grad():
            preds = model(config.normalize(img_tensor)).argmax(dim=1)

        for j in range(len(label_tensor)):
            lbl = int(label_tensor[j].item())
            if int(preds[j].item()) == lbl:
                sample = img_tensor[j] if config.name == "imagenet" else img_tensor
                images_by_class[lbl].append((sample, idx))

        if all(len(images_by_class[c]) >= max_per_class for c in needed_classes):
            break

    return images_by_class


def get_fixed_test_set_for_setting(images_by_class, setting, test_slice):
    start, end = test_slice
    if setting in ["A", "B"]:
        class_id = config.fixed_classes[f"setting_{setting}"]
        images = images_by_class[class_id][start:end]
        test_images = [ensure_batch(img[0]) for img in images]
        test_labels = [torch.tensor([class_id]).to(config.device) for _ in images]
        return test_images, test_labels

    class_id1, class_id2 = config.fixed_classes["setting_C"]
    if config.name == "imagenet":
        images1 = images_by_class[class_id1][start:end]
        images2 = images_by_class[class_id2][start:end]
    else:
        half = (end - start) // 2
        images1 = images_by_class[class_id1][start:start + half]
        images2 = images_by_class[class_id2][start:start + half]
    test_images = [ensure_batch(img[0]) for img in images1] + [ensure_batch(img[0]) for img in images2]
    test_labels = (
        [torch.tensor([class_id1]).to(config.device) for _ in images1] +
        [torch.tensor([class_id2]).to(config.device) for _ in images2]
    )
    return test_images, test_labels


def get_training_pool_for_setting(images_by_class, setting, train_start):
    if setting in ["A", "B"]:
        class_id = config.fixed_classes[f"setting_{setting}"]
        return images_by_class[class_id][train_start:]

    class_id1, class_id2 = config.fixed_classes["setting_C"]
    return images_by_class[class_id1][train_start:], images_by_class[class_id2][train_start:]


def evaluate_transferability(model, bezier_obj, delta1, theta, delta2, 
                           transfer_images, transfer_labels, num_path_points=50):

    t_values = torch.linspace(0.01, 0.99, num_path_points).to(config.device)
    
    results = {
        'delta1_success': [],
        'delta2_success': [],
        'any_path_success': [],
        'successful_points_per_image': [] 
    }
    
    with torch.no_grad():
        for x, y in zip(transfer_images, transfer_labels):
            x_adv = torch.clamp(x + delta1, 0, 1)
            pred = model(config.normalize(x_adv)).argmax(dim=1)
            delta1_success = (pred != y).item()
            results['delta1_success'].append(delta1_success)
            
            x_adv = torch.clamp(x + delta2, 0, 1)
            pred = model(config.normalize(x_adv)).argmax(dim=1)
            delta2_success = (pred != y).item()
            results['delta2_success'].append(delta2_success)
            
            path_success_count = 0
            for t in t_values:
                delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
                delta_t = bezier_obj.project_norm_ball(delta_t)
                
                x_adv = torch.clamp(x + delta_t, 0, 1)
                pred = model(config.normalize(x_adv)).argmax(dim=1)
                
                if pred != y:
                    path_success_count += 1
            
            results['any_path_success'].append(path_success_count > 0)
            results['successful_points_per_image'].append(path_success_count)
    
    stats = {
        'delta1_transfer_rate': np.mean(results['delta1_success']),
        'delta2_transfer_rate': np.mean(results['delta2_success']),
        'endpoints_avg_transfer_rate': (np.mean(results['delta1_success']) + 
                                        np.mean(results['delta2_success'])) / 2,
        'any_path_point_transfer_rate': np.mean(results['any_path_success']),
        'avg_successful_points': np.mean(results['successful_points_per_image']),
        'std_successful_points': np.std(results['successful_points_per_image']),
    }
    
    rescued = 0
    for i in range(len(transfer_images)):
        if not results['delta1_success'][i] and not results['delta2_success'][i] and results['any_path_success'][i]:
            rescued += 1
    
    stats['rescue_rate'] = rescued / len(transfer_images) if len(transfer_images) > 0 else 0
    
    return stats


def collect_samples_setting_A(training_pool, model, pgd_attack, bezier, test_images, test_labels,
                              target_samples=25):
    class_id = config.fixed_classes["setting_A"]
    samples = []
    sample_count = 0
    pbar = tqdm(total=target_samples, desc="    Collecting Setting A samples")
    step = max(1, len(training_pool) // target_samples)

    for i in range(min(target_samples * 2, len(training_pool))):
        if sample_count >= target_samples:
            break

        selected_idx = (i * step) % len(training_pool)
        x_train = ensure_batch(training_pool[selected_idx][0])
        y_train = torch.tensor([class_id]).to(config.device)

        max_pgd_attempts = 10 if config.name == "imagenet" else 5
        success = False
        
        for pgd_attempt in range(max_pgd_attempts):
            delta1 = pgd_attack.perturb(x_train, y_train)
            delta2 = pgd_attack.perturb(x_train, y_train)
            
            # Verify endpoints work on training image
            with torch.no_grad():
                pred1 = model(config.normalize(torch.clamp(x_train + delta1, 0, 1))).argmax(dim=1)
                pred2 = model(config.normalize(torch.clamp(x_train + delta2, 0, 1))).argmax(dim=1)
                
                if pred1 != y_train and pred2 != y_train:
                    success = True
                    break
        
        if not success:
            continue

        theta, _, _, _ = bezier.optimize_setting_A(x_train, y_train, delta1, delta2)
        stats = evaluate_transferability(model, bezier, delta1, theta, delta2, test_images, test_labels)
        samples.append({"stats": stats, "training_idx": selected_idx})
        sample_count += 1
        pbar.update(1)

    pbar.close()
    return samples


def collect_samples_setting_B(training_pool, model, pgd_attack, bezier, test_images, test_labels,
                              target_samples=25):
    class_id = config.fixed_classes["setting_B"]
    samples = []
    sample_count = 0
    pbar = tqdm(total=target_samples, desc="    Collecting Setting B samples")

    for pair_idx in range(min(target_samples * 2, len(training_pool) // 2)):
        if sample_count >= target_samples:
            break

        idx1 = pair_idx * 2
        idx2 = pair_idx * 2 + 1
        x1_train = ensure_batch(training_pool[idx1][0])
        x2_train = ensure_batch(training_pool[idx2][0])
        y_train = torch.tensor([class_id]).to(config.device)

        max_pgd_attempts = 10 if config.name == "imagenet" else 5
        success = False
        
        for pgd_attempt in range(max_pgd_attempts):
            delta1 = pgd_attack.perturb(x1_train, y_train)
            delta2 = pgd_attack.perturb(x2_train, y_train)
            
            with torch.no_grad():
                pred1 = model(config.normalize(torch.clamp(x1_train + delta1, 0, 1))).argmax(dim=1)
                pred2 = model(config.normalize(torch.clamp(x2_train + delta2, 0, 1))).argmax(dim=1)
                
                if pred1 != y_train and pred2 != y_train:
                    success = True
                    break

        if not success:
            continue

        theta, _, _, _ = bezier.optimize_setting_B(x1_train, x2_train, y_train, delta1, delta2)
        stats = evaluate_transferability(model, bezier, delta1, theta, delta2, test_images, test_labels)
        samples.append({"stats": stats, "training_indices": (idx1, idx2)})
        sample_count += 1
        pbar.update(1)

    pbar.close()
    return samples


def collect_samples_setting_C(training_pool1, training_pool2, model, pgd_attack, bezier,
                              test_images, test_labels, target_samples=25):
    class_id1, class_id2 = config.fixed_classes["setting_C"]
    samples = []
    sample_count = 0
    pbar = tqdm(total=target_samples, desc="    Collecting Setting C samples")
    max_pairs = min(target_samples * 2, len(training_pool1), len(training_pool2))

    for pair_idx in range(max_pairs):
        if sample_count >= target_samples:
            break

        x1_train = ensure_batch(training_pool1[pair_idx][0])
        x2_train = ensure_batch(training_pool2[pair_idx][0])
        y1_train = torch.tensor([class_id1]).to(config.device)
        y2_train = torch.tensor([class_id2]).to(config.device)

        max_pgd_attempts = 10 if config.name == "imagenet" else 5
        success = False
        
        for pgd_attempt in range(max_pgd_attempts):
            delta1 = pgd_attack.perturb(x1_train, y1_train)
            delta2 = pgd_attack.perturb(x2_train, y2_train)
            
            with torch.no_grad():
                pred1 = model(config.normalize(torch.clamp(x1_train + delta1, 0, 1))).argmax(dim=1)
                pred2 = model(config.normalize(torch.clamp(x2_train + delta2, 0, 1))).argmax(dim=1)
                
                if pred1 != y1_train and pred2 != y2_train:
                    success = True
                    break
        
        if not success:
            continue

        theta, _, _, _ = bezier.optimize_setting_C(x1_train, x2_train, y1_train, y2_train, delta1, delta2)
        stats = evaluate_transferability(model, bezier, delta1, theta, delta2, test_images, test_labels)
        samples.append({"stats": stats, "training_indices": (pair_idx, pair_idx)})
        sample_count += 1
        pbar.update(1)

    pbar.close()
    return samples


def print_transferability_results(results):
    print("\n" + "=" * 120)
    print("TRANSFERABILITY EXPERIMENTS RESULTS")
    print("=" * 120)

    for norm in ["linf", "l2", "l1"]:
        if norm not in results:
            continue

        print(f"\n{'=' * 100}")
        print(f"{norm.upper()} NORM RESULTS")
        print(f"{'=' * 100}")

        for setting in ["setting_A", "setting_B", "setting_C"]:
            setting_name = {
                "setting_A": "Setting A (Single Image)",
                "setting_B": "Setting B (Same Class)",
                "setting_C": "Setting C (Different Classes)",
            }[setting]

            print(f"\n{setting_name}:")

            samples = results[norm].get(setting, [])
            if not samples:
                print("  No samples collected")
                continue

            delta1_rates = [s["stats"]["delta1_transfer_rate"] for s in samples]
            delta2_rates = [s["stats"]["delta2_transfer_rate"] for s in samples]
            endpoints_avg_rates = [s["stats"]["endpoints_avg_transfer_rate"] for s in samples]
            path_rates = [s["stats"]["any_path_point_transfer_rate"] for s in samples]
            rescue_rates = [s["stats"]["rescue_rate"] for s in samples]
            avg_points_list = [s["stats"]["avg_successful_points"] for s in samples]

            avg_delta1 = np.mean(delta1_rates) * 100
            std_delta1 = np.std(delta1_rates) * 100

            avg_delta2 = np.mean(delta2_rates) * 100
            std_delta2 = np.std(delta2_rates) * 100

            avg_endpoints = np.mean(endpoints_avg_rates) * 100
            std_endpoints = np.std(endpoints_avg_rates) * 100

            avg_path = np.mean(path_rates) * 100
            std_path = np.std(path_rates) * 100

            avg_rescue = np.mean(rescue_rates) * 100
            std_rescue = np.std(rescue_rates) * 100

            avg_points = np.mean(avg_points_list)
            std_points = np.std(avg_points_list)

            improvement = avg_path - avg_endpoints

            print(f"  Number of samples:           {len(samples)}")
            print(f"  Endpoint δ₁ transfer:        {avg_delta1:>6.1f} ± {std_delta1:<5.1f}%")
            print(f"  Endpoint δ₂ transfer:        {avg_delta2:>6.1f} ± {std_delta2:<5.1f}%")
            print(f"  Endpoints average:           {avg_endpoints:>6.1f} ± {std_endpoints:<5.1f}%")
            print(f"  Any path point succeeds:     {avg_path:>6.1f} ± {std_path:<5.1f}%")
            print(f"  Avg successful points/image: {avg_points:>6.1f} ± {std_points:<5.1f} / 50")
            print(f"  Images rescued by path:      {avg_rescue:>6.1f} ± {std_rescue:<5.1f}%")

            if improvement > 0:
                print(f"  \033[92mImprovement over endpoints:  +{improvement:>5.1f}%\033[0m")
            else:
                print(f"  \033[91mImprovement over endpoints:  {improvement:>6.1f}%\033[0m")


def run_transferability_experiments():
    settings = transferability_settings()
    set_random_seeds(config.seed)

    norms = ["linf", "l2", "l1"]
    pgd_steps = 40
    pgd_alpha_factors = {"linf": 4.0, "l2": 5.0, "l1": 10.0}

    model = load_model(config)
    loader = make_eval_loader(config, batch_size=64 if config.name == "imagenet" else 1)

    print("\nOrganizing images by class...")
    images_by_class = organize_images_by_class(loader, model, settings["max_per_class"])

    print("\nCreating fixed test sets...")
    test_sets = {
        "setting_A": get_fixed_test_set_for_setting(images_by_class, "A", settings["test_slice"]),
        "setting_B": get_fixed_test_set_for_setting(images_by_class, "B", settings["test_slice"]),
        "setting_C": get_fixed_test_set_for_setting(images_by_class, "C", settings["test_slice"]),
    }

    print("\nCreating training pools...")
    training_pools = {
        "setting_A": get_training_pool_for_setting(images_by_class, "A", settings["train_start"]),
        "setting_B": get_training_pool_for_setting(images_by_class, "B", settings["train_start"]),
        "setting_C": get_training_pool_for_setting(images_by_class, "C", settings["train_start"]),
    }

    if config.name == "imagenet":
        target_samples_by_setting = {
            "setting_A": 20,
            "setting_B": 10,
            "setting_C": 20,
        }
    else:
        target_samples_by_setting = {
            "setting_A": 25,
            "setting_B": 25,
            "setting_C": 25,
        }

    all_results = {}
    for norm in norms:
        print(f"\n{'='*80}")
        print(f"Testing {norm.upper()} norm (epsilon={settings['epsilons'][norm]})")
        print(f"{'='*80}")

        eps = settings["epsilons"][norm]
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
        test_images_A, test_labels_A = test_sets["setting_A"]
        norm_results["setting_A"] = collect_samples_setting_A(
            training_pools["setting_A"], model, pgd_attack, bezier,
            test_images_A, test_labels_A, target_samples_by_setting["setting_A"],
        )

        test_images_B, test_labels_B = test_sets["setting_B"]
        norm_results["setting_B"] = collect_samples_setting_B(
            training_pools["setting_B"], model, pgd_attack, bezier,
            test_images_B, test_labels_B, target_samples_by_setting["setting_B"],
        )

        test_images_C, test_labels_C = test_sets["setting_C"]
        pool1, pool2 = training_pools["setting_C"]
        norm_results["setting_C"] = collect_samples_setting_C(
            pool1, pool2, model, pgd_attack, bezier,
            test_images_C, test_labels_C, target_samples_by_setting["setting_C"],
        )

        all_results[norm] = norm_results

    return all_results


def save_results(results):
    settings = transferability_settings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.output_dir, exist_ok=True)
    filename = os.path.join(config.output_dir, f"bezier_transferability_reproducible_{timestamp}.json")

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
            "data_layout": settings["data_layout"],
            "target_samples": (
                {"setting_A": 20, "setting_B": 10, "setting_C": 20}
                if config.name == "imagenet"
                else {"setting_A": 25, "setting_B": 25, "setting_C": 25}
            ),
            "test_set_size": settings["test_set_size"],
            "pgd_iterations": 40,
            "bezier_iterations": 30,
            "path_points": 50,
            "random_seed": config.seed,
        },
    }

    with open(filename, "w") as f:
        json.dump(results_with_config, f, indent=4)

    print(f"\nResults saved to {filename}")
    return filename


def parse_args():
    parser = argparse.ArgumentParser(description="Run transferability experiments.")
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
    config.seed = args.seed
    PGDAttack, BezierAdversarialUnconstrained = import_attack_modules(config)

    print("Bezier Adversarial Curves - Transferability Experiments")
    print("=" * 80)
    print(f"Dataset: {config.name}")
    print(f"Fixed classes: {config.fixed_classes}")
    print("=" * 80)

    results = run_transferability_experiments()
    if results:
        print_transferability_results(results)
        save_results(results)


if __name__ == "__main__":
    main()
