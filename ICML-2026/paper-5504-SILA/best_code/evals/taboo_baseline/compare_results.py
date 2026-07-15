#!/usr/bin/env python3
"""
Compare Taboo Baseline Results with SelfIE Results

This script compares embedding retrieval metrics between:
- Taboo baseline (intrinsic LLM capability)
- SelfIE-based evaluations (whitebox interpretability)

The comparison helps answer:
1. Which topics does the LLM know well? (high taboo recall)
2. What value does SelfIE add over raw LLM capability?
3. Which topics should we filter from SelfIE evaluations?

Usage:
    python compare_results.py \
        --taboo-results outputs/taboo_baseline/taboo_eval_*.json \
        --selfie-results results/selfie_eval_*.json
    
    # Filter to topics where taboo recall@1 = 1.0
    python compare_results.py \
        --taboo-results outputs/taboo_baseline/taboo_eval_*.json \
        --filter-taboo-recall-at-1 1.0
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def load_eval_results(path: str | Path) -> dict:
    """Load evaluation results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def extract_per_topic_results(eval_results: dict) -> dict:
    """
    Extract per-topic results from evaluation output.
    
    Returns dict mapping global_index -> per_label_result
    """
    per_label = eval_results.get("evaluation", {}).get("per_label_results", [])
    
    # If we have generation metadata with global indices
    gen_info = eval_results.get("generation", {})
    
    # Try to load from source file if available
    source_file = gen_info.get("output_file")
    if source_file and Path(source_file).exists():
        with open(source_file, "r") as f:
            source_data = json.load(f)
        
        results_with_idx = source_data.get("results", [])
        
        # Map by index
        result_map = {}
        for i, r in enumerate(results_with_idx):
            entry = {
                "topic": r["topic"],
                "description": r.get("filtered_description", r.get("description", "")),
            }
            if i < len(per_label):
                entry.update(per_label[i])
            result_map[r["global_index"]] = entry
        return result_map
    
    return {}


def compute_comparison_metrics(
    taboo_results: dict,
    selfie_results: dict | None = None,
) -> dict:
    """
    Compute comparison metrics between taboo and SelfIE results.
    
    Args:
        taboo_results: Loaded taboo eval results
        selfie_results: Loaded SelfIE eval results (optional)
        
    Returns:
        Comparison metrics dict
    """
    taboo_recalls = taboo_results.get("evaluation", {}).get("recalls", {})
    
    comparison = {
        "taboo": {
            "recalls": taboo_recalls,
            "mrr": taboo_results.get("evaluation", {}).get("mrr", None),
            "n_topics": taboo_results.get("generation", {}).get("n_topics", 0),
        }
    }
    
    if selfie_results:
        selfie_recalls = selfie_results.get("evaluation", {}).get("recalls", {})
        comparison["selfie"] = {
            "recalls": selfie_recalls,
            "mrr": selfie_results.get("evaluation", {}).get("mrr", None),
        }
        
        # Compute deltas
        comparison["delta"] = {
            "recalls": {
                k: selfie_recalls.get(k, 0) - taboo_recalls.get(k, 0)
                for k in set(taboo_recalls.keys()) | set(selfie_recalls.keys())
            }
        }
    
    return comparison


def filter_topics_by_taboo_recall(
    taboo_results: dict,
    min_recall_at_k: dict[int, float],
) -> list[int]:
    """
    Filter to topics where taboo baseline achieves minimum recall.
    
    Args:
        taboo_results: Loaded taboo eval results
        min_recall_at_k: Dict of {k: min_recall} requirements
        
    Returns:
        List of global indices that meet the criteria
    """
    per_topic = extract_per_topic_results(taboo_results)
    
    passing_indices = []
    
    for global_idx, result in per_topic.items():
        passes = True
        
        for k, min_recall in min_recall_at_k.items():
            # Check if correct topic is in top-k
            correct_rank = result.get("correct_rank")
            if correct_rank is None or correct_rank > k:
                if min_recall > 0:
                    passes = False
                    break
        
        if passes:
            passing_indices.append(global_idx)
    
    return passing_indices


def print_comparison(comparison: dict):
    """Pretty-print comparison results."""
    print("\n" + "=" * 80)
    print("TABOO vs SELFIE COMPARISON")
    print("=" * 80)
    
    print("\n--- Taboo Baseline ---")
    print(f"Topics evaluated: {comparison['taboo'].get('n_topics', 'N/A')}")
    print("Recall@K:")
    for k, recall in sorted(comparison["taboo"]["recalls"].items(), key=lambda x: int(x[0])):
        print(f"  @{k}: {recall:.4f} ({recall*100:.2f}%)")
    if comparison["taboo"].get("mrr"):
        print(f"MRR: {comparison['taboo']['mrr']:.4f}")
    
    if "selfie" in comparison:
        print("\n--- SelfIE ---")
        print("Recall@K:")
        for k, recall in sorted(comparison["selfie"]["recalls"].items(), key=lambda x: int(x[0])):
            print(f"  @{k}: {recall:.4f} ({recall*100:.2f}%)")
        if comparison["selfie"].get("mrr"):
            print(f"MRR: {comparison['selfie']['mrr']:.4f}")
        
        print("\n--- Delta (SelfIE - Taboo) ---")
        print("Recall@K difference:")
        for k, delta in sorted(comparison["delta"]["recalls"].items(), key=lambda x: int(x[0])):
            sign = "+" if delta >= 0 else ""
            print(f"  @{k}: {sign}{delta:.4f} ({sign}{delta*100:.2f}pp)")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare taboo baseline with SelfIE results"
    )
    parser.add_argument(
        "--taboo-results", type=str, required=True,
        help="Path to taboo eval results JSON"
    )
    parser.add_argument(
        "--selfie-results", type=str, default=None,
        help="Path to SelfIE eval results JSON (optional)"
    )
    parser.add_argument(
        "--filter-taboo-recall-at-1", type=float, default=None,
        help="Filter to topics with this minimum recall@1"
    )
    parser.add_argument(
        "--output-filtered-indices", type=str, default=None,
        help="Output path for filtered topic indices"
    )
    
    args = parser.parse_args()
    
    # Load results
    taboo_results = load_eval_results(args.taboo_results)
    
    selfie_results = None
    if args.selfie_results:
        selfie_results = load_eval_results(args.selfie_results)
    
    # Compute comparison
    comparison = compute_comparison_metrics(taboo_results, selfie_results)
    print_comparison(comparison)
    
    # Filter topics if requested
    if args.filter_taboo_recall_at_1 is not None:
        print(f"\nFiltering to topics with taboo recall@1 >= {args.filter_taboo_recall_at_1}")
        
        filtered_indices = filter_topics_by_taboo_recall(
            taboo_results,
            {1: args.filter_taboo_recall_at_1}
        )
        
        print(f"Topics passing filter: {len(filtered_indices)}")
        
        if args.output_filtered_indices:
            with open(args.output_filtered_indices, "w") as f:
                json.dump(filtered_indices, f)
            print(f"Saved filtered indices to: {args.output_filtered_indices}")


if __name__ == "__main__":
    main()
