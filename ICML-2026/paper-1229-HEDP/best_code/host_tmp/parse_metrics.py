#!/usr/bin/env python3
"""
Parse training logs to extract AA (Average Accuracy) and AF (Average Forgetting) metrics.
"""
import re
import sys
import json
import argparse

def parse_training_log(log_path):
    """Parse training log and extract per-task accuracy for AA and AF computation."""

    with open(log_path) as f:
        content = f.read()

    # Find the CNN eval results (after each task's training is complete)
    # Pattern: "CNN: {'grouped': {'total': XX, '00-01': XX, '02-03': XX, ...}}"
    cnn_pattern = r"CNN:\s*(\{.*?'grouped':.*?\})"
    cnn_matches = re.findall(cnn_pattern, content, re.DOTALL)

    # Also find per-epoch test accuracy from progress bars
    # Pattern: "Task X, Epoch Y/Z => Loss ..., Train_accy ..., Test_accy ..."
    epoch_pattern = r"Task (\d+), Epoch \d+/(\d+) => Loss [\d.]+, Train_accy ([\d.]+), Test_accy ([\d.]+)"
    epoch_matches = re.findall(epoch_pattern, content)

    results = {
        "per_task_eval": [],
        "training_epochs": [],
        "known_AA": None,
        "known_AF": None
    }

    # Parse per-task final accuracy
    domain_accuracies = {}  # domain_idx -> [acc_after_task_0, acc_after_task_1, ...]

    if cnn_matches:
        for i, match_str in enumerate(cnn_matches):
            try:
                # Parse the dict string
                d = eval(match_str)
                grouped = d.get("grouped", {})
                total = grouped.get("total", None)
                results["per_task_eval"].append({
                    "task": i,
                    "total": total,
                    "per_domain": {k: v for k, v in grouped.items() if k != "total"}
                })

                # Track per-domain accuracy
                for domain_name, acc in grouped.items():
                    if domain_name != "total":
                        domain_idx = int(domain_name.split("-")[0]) // 2
                        if domain_idx not in domain_accuracies:
                            domain_accuracies[domain_idx] = []
                        # Extend with None for tasks that haven't seen this domain yet
                        while len(domain_accuracies[domain_idx]) < i:
                            domain_accuracies[domain_idx].append(None)
                        domain_accuracies[domain_idx].append(acc)
            except Exception as e:
                print(f"Warning: Could not parse match {i}: {e}")

    # Fill in missing entries
    num_tasks = len(results["per_task_eval"])
    for domain_idx in domain_accuracies:
        while len(domain_accuracies[domain_idx]) < num_tasks:
            domain_accuracies[domain_idx].append(None)

    # Compute AA (Known): final accuracy on all known domains
    if results["per_task_eval"]:
        last_eval = results["per_task_eval"][-1]
        results["known_AA"] = last_eval.get("total")

    # Compute AF (Forgetting): average of (max accuracy - final accuracy) for each domain
    if domain_accuracies:
        forgetting_values = []
        for domain_idx, acc_list in domain_accuracies.items():
            valid_accs = [a for a in acc_list if a is not None]
            if valid_accs:
                max_acc = max(valid_accs)
                final_acc = valid_accs[-1]
                forgetting = max_acc - final_acc
                forgetting_values.append(forgetting)
                print(f"Domain {domain_idx}: max={max_acc:.2f}, final={final_acc:.2f}, forgetting={forgetting:.2f}")

        if forgetting_values:
            results["known_AF"] = sum(forgetting_values) / len(forgetting_values)

    # Parse training epochs
    for match in epoch_matches:
        task, total_epochs, train_acc, test_acc = match
        results["training_epochs"].append({
            "task": int(task),
            "total_epochs": int(total_epochs),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc)
        })

    return results

def parse_eval_log(log_path, split="known"):
    """Parse evaluation log for known/unknown domain accuracy."""
    with open(log_path) as f:
        content = f.read()

    # Pattern: "CNN:{'grouped': {'total': XX}}"
    cnn_pattern = r"CNN:\s*(\{.*?'grouped':.*?\})"
    cnn_matches = re.findall(cnn_pattern, content, re.DOTALL)

    aa = None
    if cnn_matches:
        try:
            d = eval(cnn_matches[0])
            aa = d.get("grouped", {}).get("total")
        except Exception as e:
            print(f"Warning: Could not parse eval result: {e}")

    return {f"{split}_AA": aa}

def main():
    parser = argparse.ArgumentParser(description="Parse HEDP training/eval logs")
    parser.add_argument("--train-log", default="/repo/logs/train/cddb.log", help="Path to training log")
    parser.add_argument("--known-log", default="/repo/logs/eval/known/cddb.log", help="Path to known eval log")
    parser.add_argument("--unknown-log", default="/repo/logs/eval/unknown/cddb.log", help="Path to unknown eval log")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    metrics = {}

    # Parse training log
    try:
        train_results = parse_training_log(args.train_log)
        metrics["known_AA"] = train_results.get("known_AA")
        metrics["known_AF"] = train_results.get("known_AF")
        print(f"Training Log: Known AA = {metrics['known_AA']}")
        print(f"Training Log: Known AF = {metrics['known_AF']}")
    except FileNotFoundError:
        print(f"Training log not found: {args.train_log}")

    # Parse known eval log
    try:
        known_results = parse_eval_log(args.known_log, "known")
        if known_results.get("known_AA"):
            metrics["known_AA_eval"] = known_results["known_AA"]
            print(f"Known Eval: AA = {metrics['known_AA_eval']}")
    except FileNotFoundError:
        print(f"Known eval log not found: {args.known_log}")

    # Parse unknown eval log
    try:
        unknown_results = parse_eval_log(args.unknown_log, "unknown")
        if unknown_results.get("unknown_AA"):
            metrics["unknown_AA"] = unknown_results["unknown_AA"]
            print(f"Unknown Eval: AA = {metrics['unknown_AA']}")
    except FileNotFoundError:
        print(f"Unknown eval log not found: {args.unknown_log}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")

    return metrics

if __name__ == "__main__":
    main()
