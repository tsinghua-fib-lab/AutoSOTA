#!/usr/bin/env python3
"""
Run Taboo Baseline Evaluation

This script orchestrates the full taboo baseline evaluation:
1. Loads topics from the fifty-thousand-things dataset
2. Generates taboo-style descriptions using a Qwen model (via vLLM)
3. Runs embedding retrieval evaluation on those descriptions
4. Reports recall@K metrics and comparison data

The taboo eval serves as a skyline/baseline for SelfIE-based evaluations:
- Topics where taboo eval fails suggest the LLM doesn't know the topic well
- The difference between taboo and SelfIE evals shows the value of interpretability

Usage:
    # Generate descriptions and evaluate (requires vLLM server running)
    python run_taboo_eval.py --model-size 7b --start-index 44673 --end-index 44773
    
    # Evaluate existing descriptions file
    python run_taboo_eval.py --descriptions-file outputs/taboo_baseline/taboo_qwen2.5_7b.json
    
    # Full VAL split evaluation
    python run_taboo_eval.py --model-size 7b --eval-split val
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.embedding_retrieval.topic_retrieval_eval import (
    TopicRetrievalConfig,
    TopicRetrievalIndex,
    TopicDataset,
    IndexStrategy,
    evaluate_custom_labels,
    print_eval_summary,
)
from evals.taboo_baseline.taboo_generator import (
    TabooDescriptionGenerator,
    TabooGeneratorConfig,
)


# =============================================================================
# Constants
# =============================================================================

# VAL split range in fifty-thousand-things dataset (same as in prepare_qwen_scaling_datasets.py)
VAL_START = 44673
VAL_END = 49636

# Model size to name mapping
QWEN_MODELS = {
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "32b": "Qwen/Qwen2.5-32B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",
}


# =============================================================================
# Evaluation Functions
# =============================================================================

@dataclass
class TabooEvalConfig:
    """Configuration for taboo baseline evaluation."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    
    # Data selection
    start_index: int | None = None
    end_index: int | None = None
    eval_split: str | None = None  # "val" to use VAL split
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 150
    max_concurrent_requests: int = 100
    
    # Evaluation parameters
    k_values: list[int] = None
    use_filtered_descriptions: bool = True  # Use filtered or raw descriptions
    
    # I/O
    output_dir: str = "outputs/taboo_baseline"
    descriptions_file: str | None = None  # Load existing instead of generating
    
    # Embedding model for retrieval
    embedding_model: str = "thenlper/gte-large"
    embedding_device: str = "cuda"  # Use GPU (pause workflow frees it first)
    index_strategy: str = "title_only"  # For TopicRetrievalIndex
    
    # Workflow options
    pause_before_eval: bool = False  # Pause after generation to let user free GPU
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 5, 10, 20, 50, 100]
        
        # Apply eval_split if specified
        if self.eval_split == "val":
            self.start_index = VAL_START
            self.end_index = VAL_END + 1  # Make it exclusive


def run_embedding_retrieval_eval(
    descriptions: list[str],
    ground_truth_indices: list[int],
    config: TabooEvalConfig,
) -> dict:
    """
    Run embedding retrieval evaluation on taboo descriptions.
    
    Args:
        descriptions: List of taboo descriptions
        ground_truth_indices: Global indices of the correct topics
        config: Evaluation configuration
        
    Returns:
        Evaluation results dict
    """
    print("\n" + "=" * 80)
    print("Running Embedding Retrieval Evaluation")
    print("=" * 80)
    
    # Create the topic retrieval index
    index_config = TopicRetrievalConfig(
        dataset=TopicDataset.FIFTY_THOUSAND,
        embedding_model=config.embedding_model,
        device=config.embedding_device,
        index_strategy=IndexStrategy(config.index_strategy),
    )
    
    index = TopicRetrievalIndex(index_config)
    
    # Build or load the index
    index.build_or_load_index()
    
    # Run evaluation
    results = evaluate_custom_labels(
        index=index,
        labels=descriptions,
        ground_truth_indices=ground_truth_indices,
        k_values=config.k_values,
    )
    
    return results


async def generate_and_evaluate(config: TabooEvalConfig) -> dict:
    """
    Generate taboo descriptions and run evaluation.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Full results dict including generation info and eval metrics
    """
    # Create generator config
    gen_config = TabooGeneratorConfig(
        model_name=config.model_name,
        vllm_base_url=config.vllm_base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_concurrent_requests=config.max_concurrent_requests,
        output_dir=config.output_dir,
        start_index=config.start_index,
        end_index=config.end_index,
    )
    
    generator = TabooDescriptionGenerator(gen_config)
    
    # Load topics
    topics, indices = generator.load_topics()
    
    # Generate descriptions
    gen_results = await generator.generate_descriptions()
    
    # Save generation results
    gen_path = generator.save_results(gen_results, indices)
    
    # Pause if requested (so user can kill vLLM to free GPU)
    if config.pause_before_eval:
        print("\n" + "=" * 80)
        print("GENERATION COMPLETE - PAUSING BEFORE EVALUATION")
        print("=" * 80)
        print(f"Descriptions saved to: {gen_path}")
        print("\nPlease kill the vLLM server to free GPU memory, then press Enter to continue...")
        input()
        print("Resuming evaluation...\n")
    
    # Extract descriptions for evaluation
    if config.use_filtered_descriptions:
        descriptions = [r["filtered_description"] for r in gen_results]
        desc_type = "filtered"
    else:
        descriptions = [r["description"] for r in gen_results]
        desc_type = "raw"
    
    # Run embedding retrieval evaluation
    eval_results = run_embedding_retrieval_eval(
        descriptions=descriptions,
        ground_truth_indices=indices,
        config=config,
    )
    
    # Print summary
    print_eval_summary(eval_results, name=f"Taboo Baseline ({config.model_name})")
    
    # Compile full results
    full_results = {
        "config": {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "start_index": config.start_index,
            "end_index": config.end_index,
            "description_type": desc_type,
        },
        "generation": {
            "n_topics": len(topics),
            "n_violations": sum(1 for r in gen_results if r["violation_info"]["has_violation"]),
            "n_empty": sum(1 for r in gen_results if not r["description"]),
            "output_file": str(gen_path),
        },
        "evaluation": eval_results,
    }
    
    # Save full results
    output_dir = Path(config.output_dir)
    model_short = config.model_name.split("/")[-1].replace("-", "_").lower()
    start = indices[0] if indices else 0
    end = indices[-1] if indices else len(topics) - 1
    results_path = output_dir / f"taboo_eval_{model_short}_{start}_{end}.json"
    
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"\nFull results saved to: {results_path}")
    
    return full_results


def evaluate_existing_descriptions(config: TabooEvalConfig) -> dict:
    """
    Evaluate previously generated descriptions.
    
    Args:
        config: Evaluation configuration (with descriptions_file set)
        
    Returns:
        Evaluation results dict
    """
    print(f"\nLoading descriptions from: {config.descriptions_file}")
    
    metadata, results = TabooDescriptionGenerator.load_results(config.descriptions_file)
    
    print(f"Loaded {len(results)} descriptions")
    print(f"Model: {metadata.get('model_name', 'unknown')}")
    
    # Extract descriptions and indices
    indices = [r["global_index"] for r in results]
    
    if config.use_filtered_descriptions:
        descriptions = [r["filtered_description"] for r in results]
        desc_type = "filtered"
    else:
        descriptions = [r["description"] for r in results]
        desc_type = "raw"
    
    # Run embedding retrieval evaluation
    eval_results = run_embedding_retrieval_eval(
        descriptions=descriptions,
        ground_truth_indices=indices,
        config=config,
    )
    
    # Print summary
    print_eval_summary(
        eval_results, 
        name=f"Taboo Baseline ({metadata.get('model_name', 'unknown')})"
    )
    
    # Compile and save full results
    full_results = {
        "config": {
            "model_name": metadata.get("model_name"),
            "descriptions_file": config.descriptions_file,
            "description_type": desc_type,
        },
        "source_metadata": metadata,
        "evaluation": eval_results,
    }
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_name = Path(config.descriptions_file).stem
    results_path = output_dir / f"eval_{source_name}.json"
    
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    
    print(f"\nEvaluation results saved to: {results_path}")
    
    return full_results


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run taboo baseline evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate and evaluate 100 topics from VAL split
    python run_taboo_eval.py --model-size 7b --start-index 44673 --end-index 44773
    
    # Full VAL split evaluation
    python run_taboo_eval.py --model-size 7b --eval-split val
    
    # Evaluate existing descriptions
    python run_taboo_eval.py --descriptions-file outputs/taboo_baseline/taboo_qwen2.5_7b.json
    
    # Use custom vLLM server
    python run_taboo_eval.py --model-size 14b --vllm-url http://localhost:8080/v1
        """
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-size", type=str, choices=list(QWEN_MODELS.keys()),
        help="Qwen model size (7b, 14b, 32b, 72b)"
    )
    model_group.add_argument(
        "--model", type=str,
        help="Full model name (overrides --model-size)"
    )
    
    # Data selection
    parser.add_argument(
        "--start-index", type=int, default=None,
        help="Starting topic index (inclusive)"
    )
    parser.add_argument(
        "--end-index", type=int, default=None,
        help="Ending topic index (exclusive)"
    )
    parser.add_argument(
        "--eval-split", type=str, choices=["val"],
        help="Use predefined split (val = indices 44673-49636)"
    )
    
    # Existing descriptions
    parser.add_argument(
        "--descriptions-file", type=str, default=None,
        help="Path to existing descriptions JSON (skip generation)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=150,
        help="Maximum tokens per description"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=100,
        help="Maximum concurrent requests"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--use-raw-descriptions", action="store_true",
        help="Use raw descriptions instead of filtered"
    )
    parser.add_argument(
        "--pause-before-eval", action="store_true",
        help="Pause after generation to let you kill vLLM and free GPU"
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100],
        help="K values for recall@K"
    )
    
    # Embedding model
    parser.add_argument(
        "--embedding-model", type=str, default="thenlper/gte-large",
        help="Embedding model for retrieval"
    )
    parser.add_argument(
        "--embedding-device", type=str, default="cuda",
        help="Device for embedding model (default: cuda)"
    )
    parser.add_argument(
        "--index-strategy", type=str, default="title_only",
        choices=["title_only", "title_plus_first_label", "title_plus_all_labels", "mean_of_all"],
        help="Index strategy for embedding retrieval"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="outputs/taboo_baseline",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Determine model name
    if args.model:
        model_name = args.model
    elif args.model_size:
        model_name = QWEN_MODELS[args.model_size]
    else:
        model_name = QWEN_MODELS["7b"]  # Default
    
    # Create config
    config = TabooEvalConfig(
        model_name=model_name,
        vllm_base_url=args.vllm_url,
        start_index=args.start_index,
        end_index=args.end_index,
        eval_split=args.eval_split,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent_requests=args.max_concurrent,
        k_values=args.k_values,
        use_filtered_descriptions=not args.use_raw_descriptions,
        pause_before_eval=args.pause_before_eval,
        output_dir=args.output_dir,
        descriptions_file=args.descriptions_file,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        index_strategy=args.index_strategy,
    )
    
    # Run appropriate mode
    if config.descriptions_file:
        # Evaluate existing descriptions
        results = evaluate_existing_descriptions(config)
    else:
        # Generate and evaluate
        if config.start_index is None and config.end_index is None:
            print("Warning: No index range specified. Use --start-index/--end-index")
            print("         or --eval-split val for the full VAL split.")
            print("         Defaulting to first 100 topics for demo.")
            config.start_index = 0
            config.end_index = 100
        
        results = asyncio.run(generate_and_evaluate(config))
    
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    recalls = results.get("evaluation", {}).get("recalls", {})
    for k in sorted(recalls.keys()):
        print(f"  Recall@{k}: {recalls[k]:.4f} ({recalls[k]*100:.2f}%)")
    
    print("=" * 80 + "\n")
    
    return results


if __name__ == "__main__":
    main()
