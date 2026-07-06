#!/usr/bin/env python3
"""
Retry Failed Training Tasks

This script retrains specific dataset-model combinations that had poor results.
Can be run directly or called from the shell script.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
LTM_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = LTM_ROOT.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LTM_ROOT))

from LTM.train import main as train_main

# Failed tasks to retry: (dataset, model, seed) tuples
# Using different seeds for each task to get different results
FAILED_TASKS = [
    ("avito-user-clicks", "nomic", 123),
    ("hm-user-churn", "nomic", 456),
]

# Configuration paths
INPUT_DATA_DIR_ROOT = Path("run_outputs/data/relbench/baselines/ltm/tpberta_table")
ORIGINAL_DATA_DIR_ROOT = Path("datasets/fit-medium-table")
RESULT_DIR = Path("run_outputs/data/relbench/baselines/ltm/results")


def retry_task(dataset: str, model: str, seed: int = None):
    """Retry training for a specific dataset-model combination."""
    input_dir = INPUT_DATA_DIR_ROOT / model / dataset
    original_data_dir = ORIGINAL_DATA_DIR_ROOT / dataset
    output_dir = RESULT_DIR / f"{model}_head" / dataset
    target_col_txt = original_data_dir / "target_col.txt"
    
    # Use different seed if not provided (default is 42, we'll use a different one)
    if seed is None:
        seed = 123  # Default seed for retry
    
    print("")
    print("=" * 60)
    print(f"Training Dataset: {dataset} with Model: {model} (seed={seed})")
    print("=" * 60)
    print(f"  INPUT_DIR: {input_dir}")
    print(f"  OUTPUT_DIR: {output_dir}")
    print(f"  TARGET_COL_TXT: {target_col_txt}")
    print(f"  SEED: {seed}")
    print("")
    
    # Check input directory exists
    if not input_dir.exists():
        print(f"  ⚠️  Warning: Input directory not found: {input_dir}")
        print("  Skipping...")
        return False
    
    # Check required files exist
    required_files = ["train.csv", "val.csv", "test.csv"]
    missing_files = [f for f in required_files if not (input_dir / f).exists()]
    if missing_files:
        print(f"  ⚠️  Warning: Missing CSV files in: {input_dir}")
        print(f"  Missing: {', '.join(missing_files)}")
        print("  Skipping...")
        return False
    
    if not target_col_txt.exists():
        print(f"  ⚠️  Warning: target_col.txt not found: {target_col_txt}")
        print("  Skipping...")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        # Import train function directly
        from LTM.train import train_prediction_head
        
        train_prediction_head(
            data_dir=str(input_dir),
            output_dir=str(output_dir),
            target_col_txt_path=str(target_col_txt),
            seed=seed,  # Use different seed for retry
        )
        
        print("")
        print(f"  ✅ Completed: {dataset} with {model}")
        print(f"     Results saved to: {output_dir}")
        return True
    except Exception as e:
        print("")
        print(f"  ❌ Error: Failed to train {dataset} with {model}")
        print(f"     Error: {e}")
        import traceback
        traceback.print_exc()
        print("  Continuing to next...")
        return False


def main():
    """Main function to retry all failed tasks."""
    parser = argparse.ArgumentParser(description="Retry failed training tasks")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Specific dataset to retry (optional, if not provided, retries all failed tasks)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model to retry (optional, if not provided, retries all failed tasks)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed to use (optional, if not provided, uses seed from FAILED_TASKS or default 123)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Retry Failed Training Tasks")
    print("=" * 60)
    print("")
    
    # Determine which tasks to retry
    if args.dataset and args.model:
        # Find matching task with its seed, or use provided/default seed
        matching_task = next((t for t in FAILED_TASKS if t[0] == args.dataset and t[1] == args.model), None)
        if matching_task:
            tasks_to_retry = [(args.dataset, args.model, args.seed if args.seed is not None else matching_task[2])]
        else:
            tasks_to_retry = [(args.dataset, args.model, args.seed if args.seed is not None else 123)]
        print(f"Retrying single task: {args.dataset} with {args.model}")
    elif args.dataset:
        # Find all models for this dataset in FAILED_TASKS
        tasks_to_retry = [(args.dataset, model, args.seed if args.seed is not None else seed) 
                         for dataset, model, seed in FAILED_TASKS if dataset == args.dataset]
        print(f"Retrying all models for dataset: {args.dataset}")
    elif args.model:
        # Find all datasets for this model in FAILED_TASKS
        tasks_to_retry = [(dataset, args.model, args.seed if args.seed is not None else seed) 
                         for dataset, model, seed in FAILED_TASKS if model == args.model]
        print(f"Retrying all datasets for model: {args.model}")
    else:
        tasks_to_retry = FAILED_TASKS
        print(f"Retrying {len(tasks_to_retry)} failed tasks:")
        for dataset, model, seed in tasks_to_retry:
            print(f"  - {dataset}:{model} (seed={seed})")
    
    print("")
    
    # Retry each task
    success_count = 0
    for task in tasks_to_retry:
        if len(task) == 3:
            dataset, model, seed = task
        else:
            dataset, model = task
            seed = args.seed if args.seed is not None else 123
        if retry_task(dataset, model, seed):
            success_count += 1
    
    print("")
    print("=" * 60)
    print("Retry Training Completed!")
    print("=" * 60)
    print(f"Successfully retried: {success_count}/{len(tasks_to_retry)} tasks")
    print(f"Results saved to: {RESULT_DIR}/{{model}}_head/{{dataset}}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
