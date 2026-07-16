#!/usr/bin/env python3
"""
LLM Comprehensive Result Analysis and Display Tool (modeled after tools/vlm_analysis.py)

Task: LLM Generation (TruthfulQA, HaluEval)

Features:
1. Supports multiple evaluation methods: Random, Self KNN, Single, Unified KNN, Pairwise Self, Pairwise Unified
2. Supports adjusting gt-threshold (uses score field if present in evaluation file; otherwise uses is_correct as GT)
3. Computes two types of F1: correct answer detection & error answer detection
4. Plots error rate distribution for each score bin (text format)
5. Generates statistical tables (CSV, LaTeX)

Usage:
  python llm_analysis.py                                  # Default parameters (analyze all datasets)
  python llm_analysis.py --dataset TruthfulQA             # Analyze TruthfulQA only
  python llm_analysis.py --dataset HaluEval               # Analyze HaluEval only
  python llm_analysis.py --pred-threshold 3.0             # Adjust prediction threshold (LLM score 0-5, default 3.0)
  python llm_analysis.py --percentile 40                  # Use percentile threshold
  python llm_analysis.py --model llama                    # Specify model
  python llm_analysis.py --method "Pairwise Self"         # Specify method
"""

import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("./outputs/llm_generation")
OUTPUT_DIR = Path("./outputs/analysis")

DATASETS = ["TruthfulQA", "HaluEval"]
MODELS = ["llama", "ministral", "qwen"]

# Model name to judge_fixed filename mapping (Ground Truth)
JUDGE_FILE_MAP = {
    ("TruthfulQA", "llama"): "TruthfulQA_llama3.1-8b_judge_fixed.json",
    ("TruthfulQA", "ministral"): "TruthfulQA_ministral-8b_judge_fixed.json",
    ("TruthfulQA", "qwen"): "TruthfulQA_qwen3-8b_judge_fixed.json",
    ("HaluEval", "llama"): "HaluEval_llama3.1-8b_judge_fixed.json",
    ("HaluEval", "ministral"): "HaluEval_ministral-8b_judge_fixed.json",
    ("HaluEval", "qwen"): "HaluEval_qwen3-8b_judge_fixed.json",
}

# Method definitions
# file_pattern: default filename pattern
# file_pattern_alt: alternative filename pattern (tried if default pattern file not found)
METHODS = {
    "Random": {
        "file_pattern": "{dataset}_{model}_random_neighbor_scores.json",
        "file_pattern_alt": "{dataset}_{model}_random_scores.json",  # HaluEval format
        "score_type": "neighbor",  # Uses stats.avg_score
        "exclude_self": True,
    },
    "Self KNN": {
        "file_pattern": "{dataset}_{model}_{model}_neighbor_scores.json",
        "file_pattern_alt": "{dataset}_{model}_{model}_with_answer_scores.json",  # HaluEval format
        "score_type": "neighbor",
    },
    "Question Self KNN": {
        "file_pattern": "{dataset}_{model}_{model}_question_neighbor_scores.json",
        "file_pattern_alt": "{dataset}_{model}_{model}_scores.json",  # HaluEval format (question-based)
        "score_type": "neighbor",
        "exclude_self": True,
        "top_n_by_cosine": 7,
    },
    "Single": {
        "file_pattern": "{dataset}_{model}_single_scores.json",
        "score_type": "single",  # Uses the score field
    },
    "Unified KNN": {
        # Unified reference: using qwen as reference/source (consistent with existing repository artifacts)
        "file_pattern": "{dataset}_{model}_qwen_neighbor_scores.json",
        "file_pattern_alt": "{dataset}_{model}_qwen_with_answer_scores.json",  # HaluEval format
        "score_type": "neighbor",
    },
    "Question Unified KNN": {
        "file_pattern": "{dataset}_{model}_qwen_question_neighbor_scores.json",
        "file_pattern_alt": "{dataset}_{model}_qwen_scores.json",  # HaluEval format (question-based)
        "score_type": "neighbor",
        "exclude_self": True,
    },
    "Pairwise Self": {
        "file_pattern": "{dataset}_{model}_{model}_pairwise_scoring.json",
        "score_type": "pairwise",  # Uses stats.avg_score_diff + stats.avg_sample_score
    },
    "Pairwise Unified": {
        "file_pattern": "{dataset}_{model}_qwen_pairwise_scoring.json",
        "score_type": "pairwise",
    },
}


# ============================================================================
# Utility Functions
# ============================================================================

def percentile(data: List[float], p: float) -> float:
    """Compute percentile"""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def mean(data: List[float]) -> float:
    """Compute mean"""
    return sum(data) / len(data) if data else 0


def median(data: List[float]) -> float:
    """Compute median"""
    return percentile(data, 50)


def std(data: List[float]) -> float:
    """Compute standard deviation"""
    if not data:
        return 0
    avg = mean(data)
    variance = sum((x - avg) ** 2 for x in data) / len(data)
    return math.sqrt(variance)


# ============================================================================
# Data Loading
# ============================================================================

def load_json(path: str) -> List[Dict]:
    """Load a JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth(dataset: str, model: str, gt_threshold: Optional[float] = None) -> Dict[int, bool]:
    """
    Load Ground Truth

    - Default: uses is_correct from judge_fixed.json
    - If gt_threshold is provided and the judge file contains a score field: uses score >= gt_threshold as correct

    Returns {index: is_correct} mapping
    """
    key = (dataset, model)
    if key not in JUDGE_FILE_MAP:
        raise ValueError(f"Unknown dataset/model combination: {key}")

    judge_file = BASE_DIR / JUDGE_FILE_MAP[key]
    if not judge_file.exists():
        raise FileNotFoundError(f"Judge file does not exist: {judge_file}")

    judge_data = load_json(str(judge_file))

    ground_truth: Dict[int, bool] = {}
    for item in judge_data:
        idx = item.get("original_index", item.get("index"))
        if idx is None:
            continue

        if gt_threshold is not None and "score" in item and isinstance(item.get("score"), (int, float)):
            ground_truth[idx] = float(item["score"]) >= float(gt_threshold)
        else:
            ground_truth[idx] = bool(item.get("is_correct", False))

    return ground_truth


def extract_score(sample: dict, method_config: dict, weighted: bool = False) -> Optional[float]:
    """
    Extract score from a sample

    Args:
        sample: Sample data
        method_config: Method configuration
        weighted: Whether to weight by similarity
    """
    score_type = method_config.get("score_type", "neighbor")
    exclude_self = method_config.get("exclude_self", False)

    if score_type == "single":
        # Single mode: directly get the score field (no neighbors involved, no weighting needed)
        return sample.get("score")

    if score_type == "neighbor":
        neighbor_scores_list = sample.get("neighbor_scores", [])

        if weighted:
            # Weighted average: sum(score * cosine) / sum(cosine)
            # If cosine is 0 (e.g., Random method), falls back to simple average
            weighted_sum = 0.0
            weight_total = 0.0
            valid_scores: List[float] = []

            for ns in neighbor_scores_list:
                if ns and ns.get("is_valid", False):
                    if exclude_self and ns.get("neighbor_rank", 10) >= 9:
                        continue
                    s = ns.get("score")
                    cosine = ns.get("neighbor_cosine", 0.0)
                    if s is not None:
                        valid_scores.append(float(s))
                        if cosine and cosine > 0:
                            weighted_sum += float(s) * float(cosine)
                            weight_total += float(cosine)

            if weight_total > 0:
                return weighted_sum / weight_total
            if valid_scores:
                return sum(valid_scores) / len(valid_scores)
            return None

        # Non-weighted mode
        if exclude_self:
            # Only compute average score for neighbor_rank 0-8 (excluding self rank=9)
            valid_scores = []
            for ns in neighbor_scores_list:
                if ns and ns.get("neighbor_rank", 10) < 9 and ns.get("is_valid", False):
                    s = ns.get("score")
                    if s is not None:
                        valid_scores.append((ns.get("neighbor_cosine", 0), float(s)))

            # Apply top_n_by_cosine if configured
            top_n = method_config.get("top_n_by_cosine", 0)
            if top_n and top_n > 0 and len(valid_scores) > top_n:
                valid_scores.sort(key=lambda x: x[0], reverse=True)
                valid_scores = valid_scores[:top_n]

            scores_only = [s for _, s in valid_scores]
            return sum(scores_only) / len(scores_only) if scores_only else None

        return sample.get("stats", {}).get("avg_score")

    if score_type == "pairwise":
        comparisons = sample.get("comparisons", [])

        if weighted and comparisons:
            # Weighted average: weighted by cosine similarity
            # If all cosine values are 0, falls back to simple average
            weighted_sample_sum = 0.0
            weighted_neighbor_sum = 0.0
            weight_total = 0.0
            sample_scores: List[float] = []
            neighbor_scores: List[float] = []

            for comp in comparisons:
                cosine = comp.get("neighbor_cosine", 0.0)
                score_a = comp.get("score_a")
                score_b = comp.get("score_b")
                sample_is_a = comp.get("sample_is_a", True)

                if score_a is not None and score_b is not None:
                    sample_score = float(score_a) if sample_is_a else float(score_b)
                    neighbor_score = float(score_b) if sample_is_a else float(score_a)

                    sample_scores.append(sample_score)
                    neighbor_scores.append(neighbor_score)

                    if cosine and cosine > 0:
                        weighted_sample_sum += sample_score * float(cosine)
                        weighted_neighbor_sum += neighbor_score * float(cosine)
                        weight_total += float(cosine)

            if weight_total > 0:
                avg_sample_score = weighted_sample_sum / weight_total
                avg_neighbor_score = weighted_neighbor_sum / weight_total
            elif sample_scores:
                avg_sample_score = sum(sample_scores) / len(sample_scores)
                avg_neighbor_score = sum(neighbor_scores) / len(neighbor_scores)
            else:
                return None

            avg_score_diff = avg_sample_score - avg_neighbor_score
            return avg_score_diff + avg_sample_score

        # Non-weighted mode: uses combined_score = avg_score_diff + avg_sample_score
        stats = sample.get("stats", {})
        avg_score_diff = stats.get("avg_score_diff")
        avg_sample_score = stats.get("avg_sample_score")

        if avg_score_diff is not None and avg_sample_score is not None:
            return float(avg_score_diff) + float(avg_sample_score)
        return None

    return None


def load_method_scores(dataset: str, model: str, method_name: str, weighted: Optional[bool] = None) -> List[Dict]:
    """
    Load score data for the specified method

    Returns:
        [{index, score}, ...]
    """
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}")

    method_config = METHODS[method_name]
    
    # Use method config default if weighted not explicitly provided
    if weighted is None:
        weighted = method_config.get("weighted", False)
    filename = method_config["file_pattern"].format(dataset=dataset, model=model)
    file_path = BASE_DIR / filename

    # If default pattern file not found, try alternative pattern
    if not file_path.exists() and "file_pattern_alt" in method_config:
        filename_alt = method_config["file_pattern_alt"].format(dataset=dataset, model=model)
        file_path_alt = BASE_DIR / filename_alt
        if file_path_alt.exists():
            file_path = file_path_alt

    if not file_path.exists():
        return []

    data = load_json(str(file_path))

    results: List[Dict] = []
    for sample in data:
        idx = sample.get("sample_index", sample.get("global_id", sample.get("index")))
        score = extract_score(sample, method_config, weighted=weighted)
        if idx is None:
            continue
        if score is not None:
            results.append({"index": idx, "score": float(score)})

    return results


# ============================================================================
# Metric Computation
# ============================================================================

def compute_metrics(predictions: List[bool], labels: List[bool]) -> Dict:
    """Compute Precision, Recall, F1, Accuracy"""
    if not predictions:
        return {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "accuracy": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "total": 0,
        }

    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(predictions) if predictions else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": len(predictions),
    }


def compute_auroc(scores: List[float], labels: List[bool], higher_is_positive: bool = True) -> float:
    """
    Compute AUROC (Area Under ROC Curve)
    Using the Mann-Whitney U statistic method:
    AUROC = P(positive score > negative score)
    """
    if not scores or not labels:
        return 0.0

    positive_scores = [s for s, l in zip(scores, labels) if l]
    negative_scores = [s for s, l in zip(scores, labels) if not l]
    if not positive_scores or not negative_scores:
        return 0.0

    n_pos = len(positive_scores)
    n_neg = len(negative_scores)

    count = 0
    ties = 0
    for p_score in positive_scores:
        for n_score in negative_scores:
            if higher_is_positive:
                if p_score > n_score:
                    count += 1
                elif p_score == n_score:
                    ties += 1
            else:
                if p_score < n_score:
                    count += 1
                elif p_score == n_score:
                    ties += 1

    return (count + 0.5 * ties) / (n_pos * n_neg)


def compute_score_bins(samples: List[Dict], threshold: float) -> List[Dict]:
    """
    Compute error rate statistics for each score bin
    Returns: [{bin_start, bin_end, total, errors, error_rate}, ...]
    """
    bins = defaultdict(lambda: {"total": 0, "errors": 0})

    for s in samples:
        score = s["score"]
        bin_idx = int(score) if score >= 0 else int(score) - 1

        bins[bin_idx]["total"] += 1
        if not s["actual_correct"]:
            bins[bin_idx]["errors"] += 1

    result = []
    for bin_idx in sorted(bins.keys()):
        b = bins[bin_idx]
        error_rate = b["errors"] / b["total"] if b["total"] > 0 else 0
        result.append(
            {
                "bin_start": bin_idx,
                "bin_end": bin_idx + 1,
                "total": b["total"],
                "errors": b["errors"],
                "error_rate": error_rate,
            }
        )
    return result


def evaluate_method(
    method_scores: List[Dict],
    ground_truth: Dict[int, bool],
    pred_threshold: Optional[float] = None,
    percentile_value: Optional[float] = None,
    error_percent: Optional[float] = None,
) -> Optional[Dict]:
    """
    Evaluate a single method

    Args:
        method_scores: [{index, score}, ...]
        ground_truth: {index: is_correct}
        pred_threshold: Fixed prediction threshold
        percentile_value: Percentile threshold (top percentile% predicted as correct)
        error_percent: Error ratio threshold (lowest error_percent% scores predicted as error, ensuring consistent counts)
    """
    if not method_scores:
        return None

    samples = []
    for s in method_scores:
        idx = s["index"]
        if idx in ground_truth:
            samples.append({"index": idx, "score": s["score"], "actual_correct": ground_truth[idx]})

    if not samples:
        return None

    all_scores = [s["score"] for s in samples]

    if error_percent is not None:
        n_total = len(samples)
        n_error = int(n_total * error_percent / 100)

        sorted_indices = sorted(range(n_total), key=lambda i: samples[i]["score"])
        error_indices = set(sorted_indices[:n_error])

        predictions_correct = [i not in error_indices for i in range(n_total)]
        labels_correct = [s["actual_correct"] for s in samples]

        if 0 < n_error < n_total:
            effective_threshold = (
                samples[sorted_indices[n_error - 1]]["score"] + samples[sorted_indices[n_error]]["score"]
            ) / 2
        elif n_error == 0:
            effective_threshold = min(all_scores) - 0.01
        else:
            effective_threshold = max(all_scores) + 0.01

        predicted_error_count = n_error

    elif percentile_value is not None:
        effective_threshold = percentile(all_scores, 100 - percentile_value)
        predictions_correct = [s["score"] >= effective_threshold for s in samples]
        labels_correct = [s["actual_correct"] for s in samples]
        predicted_error_count = sum(1 for p in predictions_correct if not p)

    else:
        effective_threshold = pred_threshold if pred_threshold is not None else 3.0
        predictions_correct = [s["score"] >= effective_threshold for s in samples]
        labels_correct = [s["actual_correct"] for s in samples]
        predicted_error_count = sum(1 for p in predictions_correct if not p)

    metrics_correct = compute_metrics(predictions_correct, labels_correct)

    predictions_error = [not p for p in predictions_correct]
    labels_error = [not l for l in labels_correct]
    metrics_error = compute_metrics(predictions_error, labels_error)

    auroc = compute_auroc(all_scores, labels_correct, higher_is_positive=True)

    score_bins = compute_score_bins(samples, effective_threshold)

    return {
        "samples_count": len(samples),
        "effective_threshold": effective_threshold,
        "predicted_error_count": predicted_error_count,
        "correct_detection": metrics_correct,
        "error_detection": metrics_error,
        "auroc": auroc,
        "score_bins": score_bins,
        "score_stats": {
            "min": min(all_scores),
            "max": max(all_scores),
            "mean": mean(all_scores),
            "median": median(all_scores),
            "std": std(all_scores),
        },
        "gt_stats": {
            "total_correct": sum(1 for s in samples if s["actual_correct"]),
            "total_error": sum(1 for s in samples if not s["actual_correct"]),
        },
    }


# ============================================================================
# Result Display
# ============================================================================

def print_score_distribution(score_bins: List[Dict], threshold: float):
    """Print score distribution chart (text format)"""
    if not score_bins:
        return

    print(f"\n     Score distribution (error rate per bin) [threshold={threshold:.2f}]:")

    max_error_rate = max(b["error_rate"] for b in score_bins) if score_bins else 1
    bar_scale = 50 / max_error_rate if max_error_rate > 0 else 50

    for b in score_bins:
        bin_start = b["bin_start"]
        bin_end = b["bin_end"]
        total = b["total"]
        errors = b["errors"]
        error_rate = b["error_rate"] * 100

        bar_len = int(b["error_rate"] * bar_scale)
        bar = "█" * bar_len

        marker = " < threshold" if bin_start <= threshold < bin_end else ""
        print(f"       [{bin_start:3}, {bin_end:3}): {errors:4}/{total:4} ({error_rate:5.1f}%) {bar}{marker}")


def print_method_result(method_name: str, result: Dict, verbose: bool = True):
    """Print results for a single method"""
    if result is None:
        print(f"  {method_name}: No data")
        return

    correct = result["correct_detection"]
    error = result["error_detection"]
    predicted_error_count = result.get("predicted_error_count", correct["fn"] + correct["tn"])
    auroc = result.get("auroc", 0)

    print(f"\n  {method_name}")
    print(f"     Samples: {result['samples_count']}, Threshold: {result['effective_threshold']:.4f}, AUROC: {auroc:.4f}")
    print(f"     GT correct: {result['gt_stats']['total_correct']}, GT errors: {result['gt_stats']['total_error']}")
    print(f"     Predicted as error: {predicted_error_count} ({predicted_error_count/result['samples_count']*100:.1f}%)")
    print(
        f"     Score range: [{result['score_stats']['min']:.2f}, {result['score_stats']['max']:.2f}], "
        f"Mean: {result['score_stats']['mean']:.2f}, Median: {result['score_stats']['median']:.2f}"
    )

    print("\n     [Correct Answer Detection] score >= threshold -> predicted correct")
    print(f"        P: {correct['precision']:.4f}, R: {correct['recall']:.4f}, F1: {correct['f1']:.4f}")

    print("     [Error Answer Detection] score < threshold -> predicted error")
    print(f"        P: {error['precision']:.4f}, R: {error['recall']:.4f}, F1: {error['f1']:.4f}")

    if verbose:
        print(f"     Confusion matrix: TP={correct['tp']}, FP={correct['fp']}, FN={correct['fn']}, TN={correct['tn']}")
        print_score_distribution(result["score_bins"], result["effective_threshold"])


def get_top_indices(values: List[Optional[float]]) -> Tuple[Optional[int], Optional[int]]:
    """Get indices of the top-1 and top-2 values in the list"""
    valid_pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid_pairs) < 1:
        return None, None

    sorted_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
    top1_idx = sorted_pairs[0][0]
    top2_idx = sorted_pairs[1][0] if len(sorted_pairs) > 1 else None
    return top1_idx, top2_idx


def format_value_console(value: float, is_top1: bool, is_top2: bool) -> str:
    """Format console output value, top1 wrapped with **, top2 wrapped with _"""
    if is_top1:
        return f"**{value:.4f}**"
    if is_top2:
        return f" _{value:.4f}_"
    return f"  {value:.4f} "


def print_results_table(all_results: List[Dict]):
    """Print results summary table (top1 bold, top2 underlined)"""
    if not all_results:
        print("No valid results")
        return None

    print("\n" + "=" * 90)
    print("Results Summary Table")
    print("=" * 90)

    valid_results = [r for r in all_results if r["result"] is not None]
    if not valid_results:
        print("No valid results")
        return None

    methods = sorted(set(r["method"] for r in valid_results))

    groups = defaultdict(dict)
    for r in valid_results:
        key = (r["dataset"], r["model"])
        groups[key][r["method"]] = r["result"]

    print("\n Correct Detection F1 (**top1**, _top2_)")
    print("-" * (30 + 12 * len(methods)))

    header = f"{'Dataset':<12} {'Model':<10}"
    for m in methods:
        header += f" {m[:10]:>10}"
    print(header)
    print("-" * (30 + 12 * len(methods)))

    for (dataset, model) in sorted(groups.keys()):
        row_values = []
        for m in methods:
            if m in groups[(dataset, model)]:
                row_values.append(groups[(dataset, model)][m]["correct_detection"]["f1"])
            else:
                row_values.append(None)

        top1_idx, top2_idx = get_top_indices(row_values)

        row = f"{dataset:<12} {model:<10}"
        for i, (_, v) in enumerate(zip(methods, row_values)):
            if v is not None:
                formatted = format_value_console(v, i == top1_idx, i == top2_idx)
                row += f"{formatted:>12}"
            else:
                row += f" {'-':>10} "
        print(row)

    print("\n Error Detection F1 (**top1**, _top2_)")
    print("-" * (30 + 12 * len(methods)))
    print(header)
    print("-" * (30 + 12 * len(methods)))

    for (dataset, model) in sorted(groups.keys()):
        row_values = []
        for m in methods:
            if m in groups[(dataset, model)]:
                row_values.append(groups[(dataset, model)][m]["error_detection"]["f1"])
            else:
                row_values.append(None)

        top1_idx, top2_idx = get_top_indices(row_values)

        row = f"{dataset:<12} {model:<10}"
        for i, (_, v) in enumerate(zip(methods, row_values)):
            if v is not None:
                formatted = format_value_console(v, i == top1_idx, i == top2_idx)
                row += f"{formatted:>12}"
            else:
                row += f" {'-':>10} "
        print(row)

    print("\n📊 AUROC （**top1**, _top2_）")
    print("-" * (30 + 12 * len(methods)))
    print(header)
    print("-" * (30 + 12 * len(methods)))

    for (dataset, model) in sorted(groups.keys()):
        row_values = []
        for m in methods:
            if m in groups[(dataset, model)]:
                row_values.append(groups[(dataset, model)][m].get("auroc", 0))
            else:
                row_values.append(None)

        top1_idx, top2_idx = get_top_indices(row_values)

        row = f"{dataset:<12} {model:<10}"
        for i, (_, v) in enumerate(zip(methods, row_values)):
            if v is not None:
                formatted = format_value_console(v, i == top1_idx, i == top2_idx)
                row += f"{formatted:>12}"
            else:
                row += f" {'-':>10} "
        print(row)

    return groups


def format_latex_value(value: float, is_top1: bool, is_top2: bool, decimals: int = 1) -> str:
    """Format LaTeX output value, top1 bold, top2 underlined"""
    formatted = f"{value:.{decimals}f}"
    if is_top1:
        return f"\\textbf{{{formatted}}}"
    if is_top2:
        return f"\\underline{{{formatted}}}"
    return formatted


def generate_latex_table(all_results: List[Dict], output_path: Path = None):
    """Generate LaTeX tables (top1 bold, top2 underlined), including Error F1 and AUROC"""
    if not all_results:
        return

    # Collect all data
    data = {}
    for r in all_results:
        if r["result"] is None:
            continue

        key = (r["dataset"], r["model"])
        if key not in data:
            data[key] = {}

        error = r["result"]["error_detection"]
        auroc = r["result"].get("auroc", 0)
        data[key][r["method"]] = {
            "P": error["precision"] * 100, 
            "R": error["recall"] * 100, 
            "F1": error["f1"] * 100,
            "AUROC": auroc * 100,
        }

    # Method order (excluding Pairwise Self and Pairwise Unified)
    method_order = [
        "Random",
        "Single",
        "Self KNN",
        "Question Self KNN",
        "Unified KNN",
        "Question Unified KNN",
    ]
    available_methods = [m for m in method_order if any(m in d for d in data.values())]

    all_latex = []
    
    # ========== Table 1: Error Detection F1 ==========
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{LLM Error Detection F1 (\\textbf{best}, \\underline{second best})}")
    latex_lines.append("\\small")

    col_spec = "ll" + "c" * len(available_methods)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Model & " + " & ".join(available_methods) + " \\\\")
    latex_lines.append("\\midrule")

    for (dataset, model) in sorted(data.keys()):
        row_f1_values = []
        for method in available_methods:
            if method in data[(dataset, model)]:
                row_f1_values.append(data[(dataset, model)][method]["F1"])
            else:
                row_f1_values.append(None)

        top1_idx, top2_idx = get_top_indices(row_f1_values)

        row_values = []
        for i, f1 in enumerate(row_f1_values):
            if f1 is not None:
                row_values.append(format_latex_value(f1, i == top1_idx, i == top2_idx))
            else:
                row_values.append("-")

        latex_lines.append(f"{dataset} & {model} & " + " & ".join(row_values) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    error_f1_latex = "\n".join(latex_lines)
    all_latex.append(error_f1_latex)
    print("\nLaTeX Table 1 (Error Detection F1):")
    print(error_f1_latex)

    # ========== Table 2: AUROC ==========
    latex_lines = []
    latex_lines.append("\n\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{LLM AUROC (\\textbf{best}, \\underline{second best})}")
    latex_lines.append("\\small")

    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Model & " + " & ".join(available_methods) + " \\\\")
    latex_lines.append("\\midrule")

    for (dataset, model) in sorted(data.keys()):
        row_auroc_values = []
        for method in available_methods:
            if method in data[(dataset, model)]:
                row_auroc_values.append(data[(dataset, model)][method]["AUROC"])
            else:
                row_auroc_values.append(None)

        top1_idx, top2_idx = get_top_indices(row_auroc_values)

        row_values = []
        for i, auroc in enumerate(row_auroc_values):
            if auroc is not None:
                # AUROC uses 2 decimal places
                row_values.append(format_latex_value(auroc, i == top1_idx, i == top2_idx, decimals=2))
            else:
                row_values.append("-")

        latex_lines.append(f"{dataset} & {model} & " + " & ".join(row_values) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    auroc_latex = "\n".join(latex_lines)
    all_latex.append(auroc_latex)
    print("\nLaTeX Table 2 (AUROC):")
    print(auroc_latex)

    # Save all tables
    if output_path:
        full_latex = "\n\n".join(all_latex)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_latex)
        print(f"\nLaTeX saved (2 tables): {output_path}")


def save_csv(all_results: List[Dict], output_path: Path):
    """Save CSV file"""
    if not all_results:
        return

    rows = []
    for r in all_results:
        if r["result"] is None:
            continue

        correct = r["result"]["correct_detection"]
        error = r["result"]["error_detection"]

        rows.append(
            {
                "Dataset": r["dataset"],
                "Model": r["model"],
                "Method": r["method"],
                "Samples": r["result"]["samples_count"],
                "GT_Errors": r["result"]["gt_stats"]["total_error"],
                "Threshold": r["result"]["effective_threshold"],
                "Correct_P": correct["precision"],
                "Correct_R": correct["recall"],
                "Correct_F1": correct["f1"],
                "Error_P": error["precision"],
                "Error_R": error["recall"],
                "Error_F1": error["f1"],
                "AUROC": r["result"].get("auroc", 0),
            }
        )

    if not rows:
        return

    with open(output_path, "w", encoding="utf-8") as f:
        headers = list(rows[0].keys())
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = [str(row[h]) for h in headers]
            f.write(",".join(values) + "\n")

    print(f"\nDetailed CSV saved: {output_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Comprehensive Analysis and Display (TruthfulQA, HaluEval)")
    parser.add_argument(
        "--gt-threshold",
        type=float,
        default=None,
        help="Ground Truth threshold: if judge file has score field, score >= gt_threshold is correct; otherwise ignored, using is_correct",
    )
    parser.add_argument("--pred-threshold", type=float, default=None, help="Prediction threshold: score >= pred_threshold predicted as correct")
    parser.add_argument("--percentile", type=float, default=None, help="Percentile threshold: top X%% predicted as correct (0-100)")
    parser.add_argument(
        "--error-percent",
        "-e",
        type=float,
        default=None,
        help="Error ratio: lowest X%% scores predicted as error (0-100), ensuring consistent error counts across methods",
    )
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS, help="Specify dataset (default: all)")
    parser.add_argument("--model", type=str, default=None, choices=MODELS, help="Specify model (default: all)")
    parser.add_argument("--method", type=str, default=None, choices=list(METHODS.keys()), help="Specify method (default: all)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: outputs/analysis)")
    parser.add_argument("--weighted", "-w", action="store_true", help="Weight scores by cosine similarity")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information (including score distribution chart)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else DATASETS
    models = [args.model] if args.model else MODELS
    methods = [args.method] if args.method else list(METHODS.keys())

    print("=" * 80)
    print("LLM Comprehensive Analysis (TruthfulQA, HaluEval)")
    print("=" * 80)
    if args.gt_threshold is None:
        print("Ground Truth: using is_correct from evaluation files")
    else:
        print(f"Ground Truth: if evaluation file has score field, score >= {args.gt_threshold} is correct; otherwise using is_correct")

    if args.error_percent is not None:
        print(f"Prediction threshold mode: fixed error ratio (lowest {args.error_percent}% predicted as error)")
    elif args.percentile is not None:
        print(f"Prediction threshold mode: percentile (top {args.percentile}% predicted as correct)")
    elif args.pred_threshold is not None:
        print(f"Prediction threshold mode: fixed value (score >= {args.pred_threshold} predicted as correct)")
    else:
        print("Prediction threshold mode: fixed value (score >= 3.0 predicted as correct, default)")

    print(f"Weighted mode: {'Enabled (weighted by cosine similarity)' if args.weighted else 'Disabled'}")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Methods: {methods}")
    print("=" * 80)

    all_results = []

    for dataset in datasets:
        for model in models:
            print(f"\n{'='*60}")
            print(f"📂 {dataset} + {model}")
            print("=" * 60)

            try:
                ground_truth = load_ground_truth(dataset, model, args.gt_threshold)
                correct_count = sum(1 for v in ground_truth.values() if v)
                error_count = sum(1 for v in ground_truth.values() if not v)
                print(f"   GT samples: {len(ground_truth)} (correct: {correct_count}, errors: {error_count})")
            except Exception as e:
                print(f"   Failed to load GT: {e}")
                continue

            for method_name in methods:
                try:
                    # Use config default if --weighted not explicitly set
                    use_weighted = True if args.weighted else None
                    method_scores = load_method_scores(dataset, model, method_name, weighted=use_weighted)
                except Exception as e:
                    print(f"   {method_name}: Load failed - {e}")
                    all_results.append({"dataset": dataset, "model": model, "method": method_name, "result": None})
                    continue

                if not method_scores:
                    print(f"   {method_name}: No data file")
                    all_results.append({"dataset": dataset, "model": model, "method": method_name, "result": None})
                    continue

                result = evaluate_method(
                    method_scores,
                    ground_truth,
                    pred_threshold=args.pred_threshold,
                    percentile_value=args.percentile,
                    error_percent=args.error_percent,
                )

                all_results.append({"dataset": dataset, "model": model, "method": method_name, "result": result})
                print_method_result(method_name, result, verbose=args.verbose)

    groups = print_results_table(all_results)

    # Generate output filename (based on dataset)
    dataset_suffix = "_".join(d.lower() for d in datasets)
    latex_path = output_dir / f"llm_analysis_{dataset_suffix}.tex"
    generate_latex_table(all_results, latex_path)

    csv_path = output_dir / f"llm_analysis_{dataset_suffix}.csv"
    save_csv(all_results, csv_path)

    _ = groups
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()


