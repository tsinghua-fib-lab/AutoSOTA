import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from common import build_dataset_config, import_evolutionary_attack, load_model, make_eval_loader, set_random_seeds


EvolutionaryAttack = None
config = None


def get_test_samples(dataloader, model, num_samples=30):
    """Get correctly classified samples using the active dataset config."""
    samples = []
    total_seen, total_correct = 0, 0

    for imgs, labels in dataloader:
        imgs = imgs.to(config.device)
        labels = labels.to(config.device)

        with torch.no_grad():
            preds = model(config.normalize(imgs)).argmax(dim=1)

        for j in range(len(labels)):
            total_seen += 1
            if preds[j] == labels[j]:
                total_correct += 1
                img_j = imgs[j].unsqueeze(0)
                label_j = labels[j].unsqueeze(0)
                samples.append((img_j, label_j))

                print(
                    f"[Added] {len(samples)}/{num_samples} "
                    f"(total correct so far {total_correct}/{total_seen})",
                    flush=True,
                )

                if len(samples) >= num_samples:
                    return samples

    return samples


def run_comparison_experiment(model, test_samples, norm="linf", eps=8 / 255):
    """Compare traditional EA and MoCo-EA."""
    results = {
        "traditional": [],
        "moco_ea": [],
    }

    ea_params = {
        "population_size": 30,
        "elite_size": 5,
        "mutation_rate": 0.2,
        "mutation_strength": 0.02,
    }
    max_generations = 1000

    for idx, (x, y) in enumerate(test_samples):
        print(f"\nSample {idx + 1}/{len(test_samples)}")

        for method in ["traditional", "moco_ea"]:
            label = "MoCo-EA" if method == "moco_ea" else "traditional EA"
            print(f"  Testing {label}...")

            ea = EvolutionaryAttack(
                model, eps=eps, norm=norm,
                normalize_fn=config.normalize,
                **ea_params,
            )
            stats = ea.evolve(
                x,
                y,
                max_generations=max_generations,
                crossover_type="bezier" if method == "moco_ea" else "traditional",
                early_stop_fitness=2.0,
            )

            results[method].append({
                "success": stats["success"][-1],
                "generations": stats["final_generation"],
                "queries": stats["query_counts"][-1],
                "time": stats["time_elapsed"][-1],
                "fitness_history": stats["best_fitness"],
            })

    return results


def analyze_results(results):
    print("\n" + "=" * 80)
    print("EVOLUTIONARY ATTACK COMPARISON RESULTS")
    print("=" * 80)

    methods = ["traditional", "moco_ea"]

    print("\n1. SUCCESS RATE:")
    for method in methods:
        successes = [r["success"] for r in results[method]]
        rate = np.mean(successes) * 100
        label = "MoCo-EA" if method == "moco_ea" else "Traditional EA"
        print(f"  {label}: {rate:.1f}%")

    print("\n2. AVERAGE GENERATIONS TO SUCCESS (successful attacks only):")
    for method in methods:
        successful = [r for r in results[method] if r["success"]]
        if successful:
            avg_gen = np.mean([r["generations"] for r in successful])
            std_gen = np.std([r["generations"] for r in successful])
            label = "MoCo-EA" if method == "moco_ea" else "Traditional EA"
            print(f"  {label}: {avg_gen:.1f} +/- {std_gen:.1f}")
        else:
            label = "MoCo-EA" if method == "moco_ea" else "Traditional EA"
            print(f"  {label}: No successful attacks")

    print("\n3. AVERAGE QUERIES:")
    for method in methods:
        avg_queries = np.mean([r["queries"] for r in results[method]])
        std_queries = np.std([r["queries"] for r in results[method]])
        label = "MoCo-EA" if method == "moco_ea" else "Traditional EA"
        print(f"  {label}: {avg_queries:.0f} +/- {std_queries:.0f}")

    print("\n4. AVERAGE TIME (seconds):")
    for method in methods:
        avg_time = np.mean([r["time"] for r in results[method]])
        std_time = np.std([r["time"] for r in results[method]])
        label = "MoCo-EA" if method == "moco_ea" else "Traditional EA"
        print(f"  {label}: {avg_time:.2f} +/- {std_time:.2f}")

    print("\n5. RELATIVE IMPROVEMENT (MoCo-EA vs Traditional EA):")
    trad_gens = [r["generations"] for r in results["traditional"] if r["success"]]
    moco_gens = [r["generations"] for r in results["moco_ea"] if r["success"]]

    if trad_gens and moco_gens:
        improvement = (np.mean(trad_gens) - np.mean(moco_gens)) / np.mean(trad_gens) * 100
        print(f"  Generation reduction: {improvement:.1f}%")

    trad_queries = [r["queries"] for r in results["traditional"]]
    moco_queries = [r["queries"] for r in results["moco_ea"]]
    query_improvement = (np.mean(trad_queries) - np.mean(moco_queries)) / np.mean(trad_queries) * 100
    print(f"  Query reduction: {query_improvement:.1f}%")


def _to_json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, tuple):
        return [_to_json(v) for v in obj]
    if isinstance(obj, list):
        return [_to_json(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    return obj


def run_evolutionary_experiments():
    model = load_model(config)
    loader = make_eval_loader(
        config,
        batch_size=64 if config.name == "imagenet" else 1,
        shuffle=False,
    )

    print("\nCollecting test samples...")
    num_samples = 30
    test_samples = get_test_samples(loader, model, num_samples)
    print(f"Collected {len(test_samples)} correctly classified samples")

    all_results = {}
    for norm in ["linf", "l2", "l1"]:
        eps = config.epsilons[norm]

        print(f"\n{'=' * 60}")
        print(f"Testing {norm.upper()} norm (epsilon={eps})")
        print(f"{'=' * 60}")

        results = run_comparison_experiment(model, test_samples, norm=norm, eps=eps)
        all_results[norm] = results

        analyze_results(results)

    return all_results


def save_results(all_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.output_dir, exist_ok=True)
    filename = os.path.join(config.output_dir, f"evolutionary_comparison_{timestamp}.json")

    serializable_results = {}
    for norm, norm_results in all_results.items():
        serializable_results[norm] = {}
        for method, method_results in norm_results.items():
            serializable_results[norm][method] = []
            for result in method_results:
                serializable_results[norm][method].append({
                    "success": bool(result["success"]),
                    "generations": int(result["generations"]),
                    "queries": int(result["queries"]),
                    "time": float(result["time"]),
                })

    results_with_config = {
        "results": serializable_results,
        "configuration": {
            "dataset": config.name,
            "fixed_classes": _to_json(config.fixed_classes),
            "population_size": 30,
            "elite_size": 5,
            "mutation_rate": 0.2,
            "mutation_strength": 0.02,
            "max_generations": 1000,
            "random_seed": config.seed,
        },
    }

    with open(filename, "w") as f:
        json.dump(results_with_config, f, indent=2)

    print(f"\nResults saved to {filename}")
    return filename


def parse_args():
    parser = argparse.ArgumentParser(description="Run evolutionary crossover comparison experiments.")
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default="resnet18_cifar10_best.pth")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main(args=None):
    global EvolutionaryAttack, config
    if args is None:
        args = parse_args()

    set_random_seeds(args.seed)
    config = build_dataset_config(args)
    config.seed = args.seed
    os.makedirs(config.output_dir, exist_ok=True)
    EvolutionaryAttack = import_evolutionary_attack(config)

    print("=" * 80)
    print("EVOLUTIONARY ATTACK: Traditional EA vs MoCo-EA Comparison")
    print("=" * 80)
    print(f"Dataset: {config.name}")
    print("Configuration:")
    print("  - Population size: 30")
    print("  - Elite size: 5")
    print("  - Methods: Traditional EA vs MoCo-EA")
    print("=" * 80)

    results = run_evolutionary_experiments()
    if results:
        save_results(results)


if __name__ == "__main__":
    main()
