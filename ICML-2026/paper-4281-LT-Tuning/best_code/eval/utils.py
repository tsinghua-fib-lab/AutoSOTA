"""
Common utility functions for evaluation scripts.
Used by eval_LTF.py and eval_soft_thinking.py
"""

import re
import json
import os
from typing import Optional
from datetime import datetime


def extract_answer_from_output(text_output: str) -> str:
    """
        Extract numeric answer from model output.
    """
    # Try to extract after #### or ###
    if "####" in text_output:
        answer = text_output.split("####")[-1].strip()
    elif "###" in text_output:
        answer = text_output.split("###")[-1].strip()
    else:
        # Try to find the last number in the output
        answer = text_output.split("#")[-1].strip()
    
    # Clean up the answer
    answer = answer.replace(",", "").replace("<thinking>", "").strip()
    # Remove any trailing tokens like <|end_of_text|>, </s>, etc.
    answer = re.sub(r'<[^>]+>', '', answer).strip()
    
    # Extract numeric value
    match = re.search(r'-?\d+\.?\d*', answer)
    if match:
        answer = match.group()
    
    return answer


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    answer = answer.strip().replace(",", "")
    
    # Try to convert to float and back for consistent formatting
    try:
        num = float(answer)
        # If it's an integer, return as int string
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        return answer


def check_answer_correct(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth.
    
    For numeric answers, normalizes and compares numerically.
    For non-numeric answers (e.g., formulas, text), compares as strings.
    """
    try:
        # Try numeric comparison first
        pred_normalized = normalize_answer(predicted)
        gt_normalized = normalize_answer(ground_truth)
        return pred_normalized == gt_normalized
    except:
        # Fallback to string comparison for non-numeric answers
        # Remove extra whitespace and compare
        pred_clean = " ".join(predicted.strip().split())
        gt_clean = " ".join(ground_truth.strip().split())
        return pred_clean.lower() == gt_clean.lower()


def save_evaluation_results(
    results_dict: dict,
    output_dir: str,
    prefix: str = "eval",
    save_detailed: bool = True
) -> str:
    """Save evaluation results to JSON files.
    
    Args:
        results_dict: Dictionary containing evaluation results
        output_dir: Directory to save results
        prefix: Prefix for output files
        save_detailed: Whether to save detailed per-sample results
        
    Returns:
        Path to the summary file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {k: v for k, v in results_dict.items() if k != "detailed_results"}
    summary_path = os.path.join(output_dir, f"{prefix}_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results if present
    if save_detailed and "detailed_results" in results_dict and results_dict["detailed_results"]:
        detailed_path = os.path.join(output_dir, f"{prefix}_detailed_{timestamp}.json")
        with open(detailed_path, "w") as f:
            json.dump(results_dict["detailed_results"], f, indent=2)
    
    return summary_path


def print_evaluation_summary(results: dict, dataset_names: list):
    """Print formatted evaluation summary.
    
    Args:
        results: Dictionary mapping dataset names to result dicts
        dataset_names: List of dataset names evaluated
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<20} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
    print("-" * 52)
    
    total_correct = 0
    total_samples = 0
    
    for dataset_name in dataset_names:
        if dataset_name in results:
            result = results[dataset_name]
            correct = result.get("correct", 0)
            total = result.get("total_samples", 0)
            accuracy = result.get("accuracy", 0)
            print(f"{dataset_name:<20} {correct:>10} {total:>10} {accuracy*100:>11.2f}%")
            total_correct += correct
            total_samples += total
    
    print("-" * 52)
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"{'Average':<20} {total_correct:>10} {total_samples:>10} {avg_accuracy*100:>11.2f}%")
    print("=" * 80)
    
    return avg_accuracy


def format_prompt_for_math(question: str, include_cot_prompt: bool = True) -> str:
    """Format question for math reasoning tasks.
    
    Args:
        question: The math question
        include_cot_prompt: Whether to include chain-of-thought prompt
        
    Returns:
        Formatted prompt string
    """
    if include_cot_prompt:
        return f"Question: {question}\n\nLet's solve this step by step:"
    return f"Question: {question}\n\nAnswer:"


def apply_chat_template_if_needed(tokenizer, messages, add_indicator=False):
    """Apply chat template if tokenizer supports it, otherwise return plain text."""
    # Check if tokenizer has chat_template attribute
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        try:
            # Try to apply chat template
            if messages[-1].get("role", "") == "assistant":
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    continue_final_message=True,
                )
            else:
                return tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            # Fallback if chat template fails
            print(f"Warning: chat_template exists but failed to apply: {e}")
            pass
    
    # Fallback: format for CoT models without chat template
    if isinstance(messages, list):
        bos = getattr(tokenizer, 'bos_token', '') or ''
        # Get the question from the user message
        question = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")
                break
        
        if question:
            if add_indicator:
                return f"{bos}Question: {question}\n\nAnswer:"
            else:
                return f"{bos}{question}"
        return bos + "\n\n".join([msg.get("content", "") for msg in messages if isinstance(msg, dict)])
    return messages

def clear_cache_in_dict(dict_to_clear):
    for k in dict_to_clear:
        dict_to_clear[k] = None
