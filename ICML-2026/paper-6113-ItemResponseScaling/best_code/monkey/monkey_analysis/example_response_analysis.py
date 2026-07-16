"""
Example analysis using full text response datasets.

This script demonstrates how to:
1. Load response data from HuggingFace
2. Analyze reasoning patterns in model responses
3. Compare DeepSeek-R1 (chain-of-thought) vs. other models
4. Identify common error patterns in BBQ bias detection
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from response_utils import (
    load_model_responses,
    get_question_responses,
    show_sample_responses,
    compute_response_statistics,
    compare_model_responses,
)

import numpy as np
import pandas as pd


def analyze_bbq_reasoning():
    """
    Analyze reasoning patterns in BBQ bias detection questions.
    """
    print("\n" + "=" * 80)
    print("ANALYZING BBQ BIAS DETECTION REASONING PATTERNS")
    print("=" * 80)

    # Load DeepSeek-R1 responses (has chain-of-thought reasoning)
    print("\nLoading Deep Seek-R1 responses for BBQ...")
    try:
        df_deepseek = load_model_responses("resmat2", "DeepSeek-R1-Distill-Llama-8B", "bbq")
        print(f"✓ Loaded {len(df_deepseek)} BBQ questions")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nNote: This example requires the consolidated dataset to be uploaded to HuggingFace.")
        print("Run the consolidation script first with --upload flag.")
        return

    # Compute statistics
    print("\nComputing per-question statistics...")
    stats = compute_response_statistics(df_deepseek)

    print(f"\nBBQ Question Statistics:")
    print(f"  Total questions: {len(stats)}")
    print(f"  Mean accuracy: {stats['accuracy'].mean():.1%}")
    print(f"  Std accuracy: {stats['accuracy'].std():.3f}")
    print(f"  Min accuracy: {stats['accuracy'].min():.1%}")
    print(f"  Max accuracy: {stats['accuracy'].max():.1%}")

    # Show easiest questions
    print("\n" + "-" * 80)
    print("EASIEST QUESTIONS (Highest Accuracy):")
    print("-" * 80)
    easiest = stats.nlargest(3, 'accuracy')
    for _, row in easiest.iterrows():
        print(f"\nQ{row['question_idx']}: {row['question']}")
        print(f"  Accuracy: {row['accuracy']:.1%}")

    # Show hardest questions
    print("\n" + "-" * 80)
    print("HARDEST QUESTIONS (Lowest Accuracy):")
    print("-" * 80)
    hardest = stats.nsmallest(3, 'accuracy')
    for _, row in hardest.iterrows():
        print(f"\nQ{row['question_idx']}: {row['question']}")
        print(f"  Accuracy: {row['accuracy']:.1%}")

    # Show sample responses for one hard question
    print("\n" + "=" * 80)
    print("EXAMPLE: DETAILED RESPONSES FOR A CHALLENGING QUESTION")
    print("=" * 80)
    hard_q_idx = int(hardest.iloc[0]['question_idx'])
    show_sample_responses(df_deepseek, hard_q_idx, n=3, show_prompt=False)


def compare_reasoning_models():
    """
    Compare reasoning patterns between DeepSeek-R1 and other models on BBQ.
    """
    print("\n" + "=" * 80)
    print("COMPARING REASONING PATTERNS: DeepSeek-R1 vs. Llama-3-70B")
    print("=" * 80)

    models = [
        "DeepSeek-R1-Distill-Llama-8B",  # Has chain-of-thought reasoning
        "Meta-Llama-3-70B-Instruct",     # Standard reasoning
    ]

    # Pick a question to compare (question index 0)
    question_idx = 0

    print(f"\nComparing models on BBQ question #{question_idx}...")
    try:
        compare_model_responses(
            dataset_version="resmat2",
            model_names=models,
            benchmark_name="bbq",
            question_idx=question_idx,
            n_samples=2
        )
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: Consolidated datasets need to be created and uploaded first.")


def analyze_response_diversity():
    """
    Analyze response diversity (how much responses vary at T=1.0).
    """
    print("\n" + "=" * 80)
    print("ANALYZING RESPONSE DIVERSITY AT T=1.0")
    print("=" * 80)

    print("\nLoading BBQ responses...")
    try:
        df = load_model_responses("resmat2", "DeepSeek-R1-Distill-Llama-8B", "bbq")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # Compute response diversity for each question
    diversity_stats = []

    for idx, row in df.iterrows():
        responses = row['responses']
        scores = row['scores']

        # Count unique responses
        unique_responses = len(set(responses))

        # Compute response entropy (how diverse are the responses?)
        # For binary scores, this is just the variance
        score_variance = np.var(scores)

        diversity_stats.append({
            "question_idx": row['question_idx'],
            "question": row['question'][:80] + "..." if len(row['question']) > 80 else row['question'],
            "unique_responses": unique_responses,
            "total_responses": len(responses),
            "diversity_ratio": unique_responses / len(responses),
            "score_variance": score_variance,
            "accuracy": np.mean(scores),
        })

    diversity_df = pd.DataFrame(diversity_stats)

    print(f"\nResponse Diversity Statistics:")
    print(f"  Mean unique responses: {diversity_df['unique_responses'].mean():.0f}")
    print(f"  Mean diversity ratio: {diversity_df['diversity_ratio'].mean():.3f}")
    print(f"  Mean score variance: {diversity_df['score_variance'].mean():.4f}")

    # Questions with highest diversity
    print("\n" + "-" * 80)
    print("MOST DIVERSE RESPONSES:")
    print("-" * 80)
    most_diverse = diversity_df.nlargest(3, 'diversity_ratio')
    for _, row in most_diverse.iterrows():
        print(f"\nQ{row['question_idx']}: {row['question']}")
        print(f"  Unique responses: {row['unique_responses']} / {row['total_responses']}")
        print(f"  Diversity ratio: {row['diversity_ratio']:.3f}")
        print(f"  Accuracy: {row['accuracy']:.1%}")

    # Questions with lowest diversity
    print("\n" + "-" * 80)
    print("LEAST DIVERSE RESPONSES (Most Consistent):")
    print("-" * 80)
    least_diverse = diversity_df.nsmallest(3, 'diversity_ratio')
    for _, row in least_diverse.iterrows():
        print(f"\nQ{row['question_idx']}: {row['question']}")
        print(f"  Unique responses: {row['unique_responses']} / {row['total_responses']}")
        print(f"  Diversity ratio: {row['diversity_ratio']:.3f}")
        print(f"  Accuracy: {row['accuracy']:.1%}")


def main():
    """
    Run all example analyses.
    """
    print("\n" + "=" * 80)
    print("RESPONSE DATASET ANALYSIS EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate how to use the consolidated response datasets.")
    print("Note: The datasets must be created and uploaded to HuggingFace first.")
    print("\nTo create the datasets, run:")
    print("  python monkey/monkey_gather_data_2/create_response_dataset.py --upload")
    print("  python monkey/monkey_gather_data/create_response_dataset.py --upload")

    # Run analyses
    try:
        analyze_bbq_reasoning()
        print("\n")
        compare_reasoning_models()
        print("\n")
        analyze_response_diversity()

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
