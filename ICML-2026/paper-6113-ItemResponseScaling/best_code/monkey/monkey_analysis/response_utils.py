"""
Utility functions for loading and working with full text response datasets.

This module provides easy-to-use functions for accessing the consolidated
response datasets on HuggingFace:
- stair-lab/irsl_testtime_resmat1_responses (10,000 responses/question)
- stair-lab/irsl_testtime_resmat2_responses (2,560 responses/question)
"""

import pandas as pd
from typing import Dict, List, Optional
from huggingface_hub import hf_hub_download


def load_model_responses(
    dataset_version: str,
    model_name: str,
    benchmark_name: str,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load responses for a model-benchmark combination.

    Args:
        dataset_version: "resmat1" or "resmat2"
        model_name: Model name (e.g., "DeepSeek-R1-Distill-Llama-8B")
        benchmark_name: Benchmark name (e.g., "bbq", "gsm", "commonsense")
        cache_dir: Optional cache directory for downloaded files

    Returns:
        pandas.DataFrame with columns:
        - question: Question text
        - question_idx: Original prompt index
        - prompt: Full few-shot prompt
        - responses: List of full model responses
        - scores: List of binary correctness scores (0/1)

    Example:
        >>> df = load_model_responses("resmat2", "DeepSeek-R1-Distill-Llama-8B", "bbq")
        >>> print(f"Loaded {len(df)} questions")
        >>> print(f"Each question has {len(df.iloc[0]['responses'])} responses")
    """
    # Determine repository ID
    if dataset_version == "resmat1":
        repo_id = "stair-lab/irsl_testtime_resmat1_responses"
    elif dataset_version == "resmat2":
        repo_id = "stair-lab/irsl_testtime_resmat2_responses"
    else:
        raise ValueError(f"Invalid dataset_version: {dataset_version}. Must be 'resmat1' or 'resmat2'")

    # Construct filename
    data_file = f"{model_name}_{benchmark_name}.parquet"

    # Download and load
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=data_file,
            repo_type="dataset",
            cache_dir=cache_dir
        )
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load {data_file} from {repo_id}. "
            f"Make sure the file exists on HuggingFace. Error: {e}"
        )


def get_question_responses(
    df: pd.DataFrame,
    question_idx: int
) -> Dict:
    """
    Get all responses for a specific question.

    Args:
        df: DataFrame returned by load_model_responses()
        question_idx: Index of the question (0-based)

    Returns:
        Dictionary with keys: question, question_idx, prompt, responses, scores

    Example:
        >>> df = load_model_responses("resmat2", "DeepSeek-R1-Distill-Llama-8B", "bbq")
        >>> q_data = get_question_responses(df, 0)
        >>> print(q_data['question'])
        >>> print(f"First response: {q_data['responses'][0]}")
    """
    if question_idx < 0 or question_idx >= len(df):
        raise IndexError(f"question_idx {question_idx} out of range [0, {len(df)})")

    row = df.iloc[question_idx]
    return {
        "question": row["question"],
        "question_idx": row["question_idx"],
        "prompt": row["prompt"],
        "responses": row["responses"],
        "scores": row["scores"],
    }


def show_sample_responses(
    df: pd.DataFrame,
    question_idx: int,
    n: int = 5,
    show_prompt: bool = False
) -> None:
    """
    Display sample responses for a question.

    Args:
        df: DataFrame returned by load_model_responses()
        question_idx: Index of the question (0-based)
        n: Number of responses to display
        show_prompt: If True, also display the full prompt

    Example:
        >>> df = load_model_responses("resmat2", "DeepSeek-R1-Distill-Llama-8B", "bbq")
        >>> show_sample_responses(df, question_idx=0, n=3)
    """
    row = df.iloc[question_idx]

    print("=" * 80)
    print(f"QUESTION #{question_idx}")
    print("=" * 80)
    print(f"\nQuestion: {row['question']}\n")

    if show_prompt:
        print(f"Full Prompt:\n{'-' * 80}")
        # Truncate very long prompts
        prompt_text = row['prompt']
        if len(prompt_text) > 1000:
            print(f"{prompt_text[:1000]}\n... [truncated {len(prompt_text) - 1000} chars] ...\n")
        else:
            print(f"{prompt_text}\n")

    print(f"Total responses: {len(row['responses'])}")
    print(f"Accuracy: {sum(row['scores']) / len(row['scores']):.1%}\n")

    print(f"Showing first {min(n, len(row['responses']))} responses:")
    print("=" * 80)

    for i in range(min(n, len(row['responses']))):
        score = row['scores'][i]
        response = row['responses'][i]

        print(f"\nResponse #{i+1} (Score: {score} - {'✓ Correct' if score == 1 else '✗ Incorrect'})")
        print("-" * 80)
        # Truncate very long responses
        if len(response) > 500:
            print(f"{response[:500]}\n... [truncated {len(response) - 500} chars] ...")
        else:
            print(response)


def compute_response_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-question statistics from response data.

    Args:
        df: DataFrame returned by load_model_responses()

    Returns:
        DataFrame with per-question statistics:
        - question_idx: Question index
        - question: Question text (first 100 chars)
        - n_responses: Number of responses
        - accuracy: Fraction of correct responses
        - std: Standard deviation of binary responses
        - min_score: Minimum score
        - max_score: Maximum score

    Example:
        >>> df = load_model_responses("resmat2", "DeepSeek-R1-Distill-Llama-8B", "bbq")
        >>> stats = compute_response_statistics(df)
        >>> print(stats.sort_values('accuracy'))
    """
    import numpy as np

    stats = []
    for idx, row in df.iterrows():
        scores = row['scores']
        stats.append({
            "question_idx": row['question_idx'],
            "question": row['question'][:100] + "..." if len(row['question']) > 100 else row['question'],
            "n_responses": len(scores),
            "accuracy": np.mean(scores),
            "std": np.std(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        })

    return pd.DataFrame(stats)


def compare_model_responses(
    dataset_version: str,
    model_names: List[str],
    benchmark_name: str,
    question_idx: int,
    n_samples: int = 3
) -> None:
    """
    Compare responses from multiple models on the same question.

    Args:
        dataset_version: "resmat1" or "resmat2"
        model_names: List of model names to compare
        benchmark_name: Benchmark name
        question_idx: Question index
        n_samples: Number of sample responses per model

    Example:
        >>> compare_model_responses(
        ...     "resmat2",
        ...     ["DeepSeek-R1-Distill-Llama-8B", "Meta-Llama-3-70B-Instruct"],
        ...     "bbq",
        ...     question_idx=0,
        ...     n_samples=2
        ... )
    """
    print("=" * 80)
    print(f"COMPARING {len(model_names)} MODELS ON QUESTION #{question_idx}")
    print("=" * 80)

    for model_name in model_names:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 80}")

        try:
            df = load_model_responses(dataset_version, model_name, benchmark_name)
            row = df.iloc[question_idx]

            print(f"\nQuestion: {row['question']}")
            print(f"Accuracy: {sum(row['scores']) / len(row['scores']):.1%}")
            print(f"\nSample responses:")

            for i in range(min(n_samples, len(row['responses']))):
                score = row['scores'][i]
                response = row['responses'][i]

                print(f"\n  Response #{i+1} (Score: {score} - {'✓' if score == 1 else '✗'})")
                print(f"  {'-' * 76}")
                # Truncate long responses
                if len(response) > 300:
                    print(f"  {response[:300]}...")
                else:
                    print(f"  {response}")

        except Exception as e:
            print(f"\n  Error loading data: {e}")


def get_available_combinations(dataset_version: str) -> List[tuple]:
    """
    Get list of available (model, benchmark) combinations.

    Note: This function attempts to enumerate all combinations based on
    known models and benchmarks. Some combinations may not exist.

    Args:
        dataset_version: "resmat1" or "resmat2"

    Returns:
        List of (model_name, benchmark_name) tuples

    Example:
        >>> combinations = get_available_combinations("resmat2")
        >>> print(f"Found {len(combinations)} combinations")
    """
    if dataset_version == "resmat1":
        models = [
            "DeepSeek-R1-Distill-Llama-8B",
            "DeepSeek-V2-Lite-Chat",
            "Meta-Llama-3-8B-Instruct",
            "Meta-Llama-3-70B-Instruct",
            "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
            "gemma-3-27b-it",
        ]
        benchmarks = ["commonsense", "legalbench", "med_qa", "bbq", "lsat_qa", "legal_support"]
    elif dataset_version == "resmat2":
        models = [
            "DeepSeek-R1-Distill-Llama-8B",
            "DeepSeek-V2-Lite-Chat",
            "Meta-Llama-3-8B-Instruct",
            "Meta-Llama-3-70B-Instruct",
            "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
            "gemma-3-27b-it",
            "Qwen2.5-72B-Instruct",
            "Qwen2.5-Math-72B-Instruct",
            "gpt-4o-mini-2024-07-18",
            "claude-3-5-sonnet-20241022",
        ]
        benchmarks = ["bbq", "gsm", "commonsense", "legalbench", "med_qa",
                     "lsat_qa", "legal_support", "mmlu", "math"]
    else:
        raise ValueError(f"Invalid dataset_version: {dataset_version}")

    return [(m, b) for m in models for b in benchmarks]
