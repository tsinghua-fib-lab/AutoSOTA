#!/usr/bin/env python3
"""
Evaluate labels from labels_with_metadata.json using embedding-based retrieval.

This script:
1. Loads labels from a labels_with_metadata.json file
2. Groups them by scale value
3. Evaluates using clean, numerically-stable embedding retrieval
4. Saves detailed results per scale
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Ensure repo root is on the path so this works both as a script and as a module
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from evals.embedding_retrieval.topic_retrieval_eval import (
    TopicRetrievalIndex,
    TopicRetrievalConfig,
    TopicDataset,
    IndexStrategy,
    evaluate_labels,
    print_eval_summary,
)


def load_labels_from_metadata(metadata_path: Path) -> dict:
    """
    Load and parse labels_with_metadata.json file.
    
    Returns:
        Dictionary with metadata and labels grouped by scale
    """
    print(f"Loading labels from: {metadata_path}")
    with open(metadata_path, "r") as f:
        data = json.load(f)
    
    metadata = data["metadata"]
    enriched_labels = data["enriched_labels"]
    
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Layer: {metadata['layer']}")
    print(f"Scales: {metadata['scale_values']}")
    print(f"Total label entries: {len(enriched_labels)}")
    
    # Group labels by scale
    labels_by_scale = defaultdict(list)
    
    for entry in enriched_labels:
        scale = entry["scale"]
        labels_by_scale[scale].append({
            "label": entry["label"],
            "combined_index": entry["combined_index"],
            "original_layer": entry.get("original_layer"),
            "original_global_index": entry["original_global_index"],
        })
    
    # Print statistics
    print("\nLabels per scale:")
    for scale in sorted(labels_by_scale.keys()):
        n_labels = len(labels_by_scale[scale])
        unique_topics = len(set(item["original_global_index"] for item in labels_by_scale[scale]))
        print(f"  Scale {scale}: {n_labels} labels covering {unique_topics} unique topics")
    
    return {
        "metadata": metadata,
        "labels_by_scale": labels_by_scale,
    }


def evaluate_scale(
    index: TopicRetrievalIndex,
    scale: float,
    label_data: list[dict],
    k_values: list[int],
) -> dict:
    """
    Evaluate labels for a single scale value.
    
    Args:
        index: Pre-built TopicRetrievalIndex
        scale: The scale value being evaluated
        label_data: List of dicts with 'label', 'original_global_index', etc.
        k_values: K values for recall computation
        
    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Scale: {scale}")
    print(f"{'='*70}")
    
    # Extract labels and ground truth indices
    labels = [item["label"] for item in label_data]
    ground_truth_indices = [item["original_global_index"] for item in label_data]
    
    # Run evaluation
    results = evaluate_labels(
        index=index,
        labels=labels,
        ground_truth_indices=ground_truth_indices,
        k_values=k_values,
    )
    
    # Add metadata to per-label results
    for i, label_item in enumerate(label_data):
        if i < len(results["per_label_results"]):
            results["per_label_results"][i]["original_layer"] = label_item.get("original_layer")
            results["per_label_results"][i]["original_global_index"] = label_item["original_global_index"]
            results["per_label_results"][i]["combined_index"] = label_item["combined_index"]
    
    # Print summary
    print_eval_summary(results, name=f"Scale {scale}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate labels using embedding retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate labels with default settings
  python evals/embedding_retrieval/evaluate_labels_retrieval.py results/qwen_scaling/7b_trained/labels_with_metadata.json
  
  # Use a different index strategy
  python evals/embedding_retrieval/evaluate_labels_retrieval.py results/my_labels.json --index-strategy title_plus_all_labels
  
  # Evaluate only specific scales
  python evals/embedding_retrieval/evaluate_labels_retrieval.py results/my_labels.json --scales 1.0 2.0
        """
    )
    parser.add_argument(
        "metadata_file",
        type=Path,
        help="Path to labels_with_metadata.json file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: same as metadata file)"
    )
    parser.add_argument(
        "--index-strategy",
        type=str,
        default="title_plus_all_labels",
        choices=["title_only", "title_plus_first_label", "title_plus_all_labels", "mean_of_all"],
        help="Strategy for building the topic index (default: title_plus_all_labels)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="thenlper/gte-large",
        help="Embedding model to use (default: thenlper/gte-large)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="Force rebuild of the topic index (ignore cache)"
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        help="Specific scales to evaluate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.metadata_file.exists():
        print(f"Error: File not found: {args.metadata_file}")
        sys.exit(1)
    
    # Load labels
    data = load_labels_from_metadata(args.metadata_file)
    metadata = data["metadata"]
    labels_by_scale = data["labels_by_scale"]
    
    # Determine output directory
    output_dir = args.output_dir or args.metadata_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build or load the topic retrieval index
    print(f"\n{'='*70}")
    print("Building/Loading Topic Retrieval Index")
    print(f"{'='*70}")
    
    # Determine dataset from metadata
    dataset_name = metadata.get("dataset_name", "")
    if "fifty-thousand" in dataset_name.lower():
        dataset = TopicDataset.FIFTY_THOUSAND
    elif "ten-thousand" in dataset_name.lower():
        dataset = TopicDataset.TEN_THOUSAND
    else:
        print(f"Warning: Could not determine dataset from '{dataset_name}', using FIFTY_THOUSAND")
        dataset = TopicDataset.FIFTY_THOUSAND
    
    config = TopicRetrievalConfig(
        dataset=dataset,
        embedding_model=args.embedding_model,
        device=args.device,
        index_strategy=IndexStrategy(args.index_strategy),
    )
    
    index = TopicRetrievalIndex(config)
    index.build_or_load_index(force_rebuild=args.force_rebuild_index)
    
    # K values for recall computation
    k_values = [1, 5, 10, 20, 50, 100]
    
    # Evaluate each scale
    all_results = {}
    scales_to_eval = args.scales if args.scales else sorted(labels_by_scale.keys())
    
    for scale in scales_to_eval:
        if scale not in labels_by_scale:
            print(f"\nWarning: Scale {scale} not found in data, skipping")
            continue
        
        label_data = labels_by_scale[scale]
        results = evaluate_scale(index, scale, label_data, k_values)
        all_results[scale] = results
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    
    # Prepare summary data
    save_data = {
        "metadata": metadata,
        "index_config": {
            "dataset": config.dataset.value,
            "embedding_model": config.embedding_model,
            "index_strategy": config.index_strategy.value,
        },
        "results_by_scale": {}
    }
    
    for scale, results in all_results.items():
        # Save per-label results separately (can be large)
        per_label_file = output_dir / f"retrieval_eval_scale_{scale}_per_label.json"
        with open(per_label_file, "w") as f:
            json.dump(results["per_label_results"], f, indent=2)
        print(f"Saved per-label results to: {per_label_file}")
        
        # Save summary results
        save_data["results_by_scale"][str(scale)] = {
            "n_labels": results["n_labels"],
            "recalls": results["recalls"],
            "mrr": results["mrr"],
            "mean_correct_score": results["mean_correct_score"],
            "mean_margin": results["mean_margin"],
            "mean_confidence_gap": results["mean_confidence_gap"],
            "per_label_results_file": str(per_label_file.name),
        }
    
    # Save summary
    summary_file = output_dir / "retrieval_eval_summary.json"
    with open(summary_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved summary to: {summary_file}")
    
    # Print final summary table
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: Recall@K by Scale")
    print(f"{'='*70}")
    print(f"\n{'Scale':<10} {'R@1':<10} {'R@5':<10} {'R@10':<10} {'R@50':<10} {'R@100':<10} {'MRR':<10}")
    print("-" * 70)
    
    for scale in sorted(all_results.keys()):
        recalls = all_results[scale]["recalls"]
        mrr = all_results[scale]["mrr"]
        row = f"{scale:<10.1f}"
        for k in [1, 5, 10, 50, 100]:
            if k in recalls:
                row += f" {recalls[k]:<10.4f}"
            else:
                row += f" {'N/A':<10}"
        row += f" {mrr:<10.4f}"
        print(row)
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
