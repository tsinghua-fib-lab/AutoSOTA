"""
Consolidate ResMAT1 full text responses from fragmented parquet batches into
a user-friendly dataset structure.

This script:
1. Reads raw query data from Stanford cluster filesystem
2. Consolidates ~80 batch files per question into single records
3. Creates one parquet file per (model, dataset) combination
4. Uploads to stair-lab/irsl_testtime_resmat1_responses

Output format:
- question: Question text (str)
- question_idx: Original prompt index (int)
- prompt: Full few-shot prompt (str)
- responses: List of 10,000 full model responses (List[str])
- scores: List of 10,000 binary scores (List[int])

NOTE: This script must be run on the Stanford cluster or wherever the original
      parquet files are accessible at:
      /lfs/skampere1/0/sttruong/deval/data/monkey_query/eval_results_fix
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from huggingface_hub import HfApi, login
from tqdm import tqdm


# ResMAT1 models and datasets
MODELS = [
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-V2-Lite-Chat",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3-70B-Instruct",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "gemma-3-27b-it",
]

DATASETS = [
    "commonsense",
    "legalbench",
    "med_qa",
    "bbq",
    "lsat_qa",
    "legal_support",
]

# Default location on Stanford cluster
DEFAULT_BASE_PATH = "/lfs/skampere1/0/sttruong/deval/data/monkey_query/eval_results_fix"


def consolidate_model_dataset(
    base_dir: Path,
    model: str,
    dataset: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Consolidate all batch files for a (model, dataset) combination.

    Args:
        base_dir: Root directory containing eval_results_fix
        model: Model name
        dataset: Dataset name
        verbose: Print progress

    Returns:
        DataFrame with columns: question, question_idx, prompt, responses, scores
    """
    # Path structure: {base_dir}/{dataset}/{model}/prompt={idx}/batch_*.parquet
    base_path = base_dir / dataset / model

    if not base_path.exists():
        if verbose:
            print(f"  Skipping {model}/{dataset} - path does not exist")
        return None

    records = []
    prompt_folders = sorted(base_path.glob("prompt=*"))

    if not prompt_folders:
        if verbose:
            print(f"  Skipping {model}/{dataset} - no prompt folders found")
        return None

    for prompt_folder in tqdm(prompt_folders, desc=f"  {model}/{dataset}", disable=not verbose):
        prompt_idx = int(prompt_folder.name.split("=")[1])

        # Find all batch files for this question
        batch_files = sorted(prompt_folder.glob("batch_*.parquet"))

        if not batch_files:
            if verbose:
                print(f"    Warning: No batch files in {prompt_folder.name}")
            continue

        # Collect all responses and scores from batch files
        all_responses = []
        all_scores = []
        question_text = None
        prompt_text = None

        for batch_file in batch_files:
            try:
                # Read only needed columns
                df = pd.read_parquet(
                    batch_file,
                    columns=["problem", "prompt", "response", "score"]
                )

                # Extract question and prompt from first row (same for all rows)
                if question_text is None:
                    question_text = df["problem"].iloc[0]
                    prompt_text = df["prompt"].iloc[0]

                # Collect responses and scores
                all_responses.extend(df["response"].tolist())
                all_scores.extend(df["score"].tolist())

            except Exception as e:
                if verbose:
                    print(f"    Error reading {batch_file.name}: {e}")
                continue

        # Sanity check: responses and scores should match
        if len(all_responses) != len(all_scores):
            if verbose:
                print(f"    Warning: Response/score mismatch for prompt {prompt_idx}")
            continue

        # ResMAT1 should have 10,000 samples per question
        if len(all_responses) != 10000:
            if verbose:
                print(f"    Warning: Expected 10,000 samples, got {len(all_responses)} for prompt {prompt_idx}")

        # Create consolidated record
        records.append({
            "question": question_text,
            "question_idx": prompt_idx,
            "prompt": prompt_text,
            "responses": all_responses,
            "scores": all_scores,
        })

    if not records:
        if verbose:
            print(f"  Skipping {model}/{dataset} - no valid records")
        return None

    df = pd.DataFrame(records)
    if verbose:
        print(f"  ✓ Consolidated {len(df)} questions, {len(df.iloc[0]['responses'])} responses/question")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate ResMAT1 responses into user-friendly dataset"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_PATH,
        help="Base directory containing eval_results_fix"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./resmat1_responses",
        help="Local directory to save consolidated parquet files"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload consolidated files to HuggingFace"
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="stair-lab/irsl_testtime_resmat1_responses",
        help="HuggingFace repository ID for upload"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to process (default: all)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to process (default: all)"
    )
    args = parser.parse_args()

    # Verify base directory exists
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        print(f"       This script must be run on the Stanford cluster with access to:")
        print(f"       {DEFAULT_BASE_PATH}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models and datasets to process
    models_to_process = args.models if args.models else MODELS
    datasets_to_process = args.datasets if args.datasets else DATASETS

    print(f"Reading data from: {base_dir}")
    print(f"Processing {len(models_to_process)} models × {len(datasets_to_process)} datasets\n")

    # Track statistics
    total_files = 0
    total_questions = 0
    total_size_mb = 0

    # Process each (model, dataset) combination
    for model in models_to_process:
        for dataset in datasets_to_process:
            print(f"\nProcessing {model} / {dataset}")

            # Consolidate batch files
            df = consolidate_model_dataset(base_dir, model, dataset, verbose=True)

            if df is None or len(df) == 0:
                continue

            # Save to parquet
            output_file = output_dir / f"{model}_{dataset}.parquet"
            df.to_parquet(output_file, compression="snappy", index=False)

            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            total_files += 1
            total_questions += len(df)
            total_size_mb += file_size_mb

            print(f"  Saved: {output_file.name} ({file_size_mb:.2f} MB)")

    # Print summary
    print("\n" + "=" * 80)
    print("CONSOLIDATION COMPLETE")
    print("=" * 80)
    print(f"Total files created: {total_files}")
    print(f"Total questions: {total_questions}")
    print(f"Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"Output directory: {output_dir.absolute()}")

    # Upload to HuggingFace if requested
    if args.upload:
        print("\n" + "=" * 80)
        print("UPLOADING TO HUGGINGFACE")
        print("=" * 80)

        # Login to HuggingFace
        login()
        api = HfApi()

        # Upload each file
        for parquet_file in sorted(output_dir.glob("*.parquet")):
            print(f"Uploading {parquet_file.name}...")
            try:
                api.upload_file(
                    path_or_fileobj=str(parquet_file),
                    path_in_repo=parquet_file.name,
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                )
                print(f"  ✓ Uploaded")
            except Exception as e:
                print(f"  ✗ Error: {e}")

        print(f"\n✓ Upload complete to: https://huggingface.co/datasets/{args.hf_repo}")


if __name__ == "__main__":
    main()
