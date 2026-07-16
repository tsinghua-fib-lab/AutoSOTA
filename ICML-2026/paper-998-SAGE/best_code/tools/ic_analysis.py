#!/usr/bin/env python3
"""
IC (Image Classification) Comprehensive Result Analysis and Display Tool

Task: Image Classification (CIFAR-10, ImageNet-1k)

Features:
1. Supports multiple evaluation methods: Random, Self KNN, Unified KNN, Single
2. Computes two types of F1: correct answer detection & error answer detection
3. Plots error rate distribution for each score bin (text format)
4. Generates statistical tables (CSV, LaTeX)

Usage:
  python ic_analysis.py                                  # Default parameters (analyze all datasets)
  python ic_analysis.py --dataset CIFAR-10              # Analyze CIFAR-10 only
  python ic_analysis.py --dataset ImageNet-1k           # Analyze ImageNet-1k only
  python ic_analysis.py --pred-threshold 3.0            # Adjust prediction threshold
  python ic_analysis.py --percentile 40                 # Use percentile threshold
  python ic_analysis.py --model internvl                # Specify model
  python ic_analysis.py --method "Self KNN"             # Specify method
"""

import json
import argparse
import math
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# Configuration
# ============================================================================

# Local directory (CIFAR-10)
LOCAL_IC_DIR = Path("./outputs/image_classification")

# Remote directory (ImageNet-1k)
REMOTE_IC_DIR = Path("/mnt/data1/SAGE/outputs/image_classification")

OUTPUT_DIR = Path("./outputs/analysis")

DATASETS = ["CIFAR-10", "ImageNet-1k"]
MODELS = ["qwen", "internvl", "sailvl"]

# Model to prediction filename mapping (for obtaining Ground Truth)
# CIFAR-10 in LOCAL_IC_DIR, ImageNet-1k in REMOTE_IC_DIR
GT_FILE_PATTERNS = {
    ("CIFAR-10", "qwen"): "CIFAR-10_qwen3-vl-8b_*.json",
    ("CIFAR-10", "internvl"): "CIFAR-10_internvl3.5-8b_*.json",
    ("CIFAR-10", "sailvl"): "CIFAR-10_sailvl-8b_*.json",
    ("ImageNet-1k", "qwen"): "ImageNet-1k_qwen3-vl-8b_*.json",
    ("ImageNet-1k", "internvl"): "ImageNet-1k_internvl3.5-8b_*.json",
    ("ImageNet-1k", "sailvl"): "ImageNet-1k_sailvl-8b_*.json",
}

# Method definitions
METHODS = {
    "Random": {
        "file_pattern": "{dataset}_{model}_random_neighbor_scores.json",
        "score_type": "neighbor",  # Uses stats.avg_score
        "exclude_self": True,
    },
    "Self KNN": {
        "file_pattern": "{dataset}_{model}_{model}_neighbor_scores.json",
        "score_type": "neighbor",
    },
    "Unified KNN": {
        # Unified reference: using metaclip as reference
        "file_pattern": "{dataset}_{model}_metaclip_neighbor_scores.json",
        "score_type": "neighbor",
    },
    "Single": {
        "file_pattern": "{dataset}_{model}_single_scores.json",
        "score_type": "single",  # Uses the score field
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


def get_base_dir(dataset: str) -> Path:
    """Get the base directory for the given dataset"""
    if dataset == "CIFAR-10":
        return LOCAL_IC_DIR
    else:  # ImageNet-1k
        return REMOTE_IC_DIR


# ============================================================================
# Data Loading
# ============================================================================

def load_json(path: str) -> List[Dict]:
    """Load a JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_gt_file(dataset: str, model: str) -> Optional[Path]:
    """Find the Ground Truth file (prediction result file)"""
    key = (dataset, model)
    if key not in GT_FILE_PATTERNS:
        return None
    
    pattern = GT_FILE_PATTERNS[key]
    
    # CIFAR-10 in local directory, ImageNet-1k in remote directory
    if dataset == "CIFAR-10":
        base_dir = LOCAL_IC_DIR
    else:
        base_dir = REMOTE_IC_DIR
    
    # Find matching files
    matches = list(base_dir.glob(pattern))
    if not matches:
        return None
    
    # Return the most recent file
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_ground_truth(dataset: str, model: str) -> Dict[int, bool]:
    """
    Load Ground Truth
    Returns {index: is_correct} mapping
    """
    gt_file = find_gt_file(dataset, model)
    if gt_file is None or not gt_file.exists():
        raise FileNotFoundError(f"Cannot find GT file: {dataset} + {model}")
    
    data = load_json(str(gt_file))
    
    ground_truth = {}
    for item in data:
        idx = item.get("index")
        # Prefer the correct field; if absent, compute from true_label and predicted_label
        if "correct" in item:
            is_correct = item["correct"]
        else:
            true_label = item.get("true_label")
            predicted_label = item.get("predicted_label")
            is_correct = (true_label == predicted_label) if true_label is not None and predicted_label is not None else False
        
        if idx is not None:
            ground_truth[idx] = is_correct
    
    return ground_truth


def find_score_file(dataset: str, model: str, method_name: str) -> Optional[Path]:
    """Find score file"""
    if method_name not in METHODS:
        return None
    
    method_config = METHODS[method_name]
    filename = method_config["file_pattern"].format(dataset=dataset, model=model)
    
    # First search in local directory
    local_path = LOCAL_IC_DIR / filename
    if local_path.exists():
        return local_path
    
    # Then search in remote directory
    remote_path = REMOTE_IC_DIR / filename
    if remote_path.exists():
        return remote_path
    
    # Handle special case for CIFAR-10 sailvl (filename may have prefix I)
    if dataset == "CIFAR-10" and model == "sailvl":
        alt_filename = "I" + filename
        alt_local = LOCAL_IC_DIR / alt_filename
        if alt_local.exists():
            return alt_local
        
        # Handle .json.json case
        if method_name == "Single":
            alt_filename2 = filename + ".json"
            alt_local2 = LOCAL_IC_DIR / alt_filename2
            if alt_local2.exists():
                return alt_local2
            alt_filename3 = "I" + alt_filename2
            alt_local3 = LOCAL_IC_DIR / alt_filename3
            if alt_local3.exists():
                return alt_local3
    
    return None


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
        # Single mode: directly get the score field
        return sample.get("score")

    if score_type == "neighbor":
        neighbor_scores_list = sample.get("neighbor_scores", [])

        if weighted:
            # Weighted average: sum(score * cosine) / sum(cosine)
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
                        valid_scores.append(float(s))
            return sum(valid_scores) / len(valid_scores) if valid_scores else None

        return sample.get("stats", {}).get("avg_score")

    return None


def load_method_scores(dataset: str, model: str, method_name: str, weighted: bool = False) -> List[Dict]:
    """
    Load score data for the specified method

    Returns:
        [{index, score}, ...]
    """
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}")

    method_config = METHODS[method_name]
    file_path = find_score_file(dataset, model, method_name)

    if file_path is None or not file_path.exists():
        return []

    data = load_json(str(file_path))

    scores_list: List[Dict] = []

    for sample in data:
        idx = sample.get("index", sample.get("sample_index"))
        score = extract_score(sample, method_config, weighted=weighted)

        if idx is not None and score is not None:
            scores_list.append({"index": idx, "score": float(score)})

    return scores_list


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

    print(f"\n  📌 {method_name}")
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
    """Generate LaTeX tables (top1 bold, top2 underlined), including Error F1, Correct F1, AUROC"""
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
        correct = r["result"]["correct_detection"]
        auroc = r["result"].get("auroc", 0)
        
        data[key][r["method"]] = {
            "Error_F1": error["f1"] * 100,
            "Correct_F1": correct["f1"] * 100,
            "AUROC": auroc * 100,
        }

    method_order = ["Random", "Single", "Self KNN", "Unified KNN"]
    available_methods = [m for m in method_order if any(m in d for d in data.values())]

    all_latex = []
    
    # ========== Table 1: Error Detection F1 ==========
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{IC Error Detection F1 (\\textbf{best}, \\underline{second best})}")
    latex_lines.append("\\small")

    col_spec = "ll" + "c" * len(available_methods)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Model & " + " & ".join(available_methods) + " \\\\")
    latex_lines.append("\\midrule")

    for (dataset, model) in sorted(data.keys()):
        row_values_raw = []
        for method in available_methods:
            if method in data[(dataset, model)]:
                row_values_raw.append(data[(dataset, model)][method]["Error_F1"])
            else:
                row_values_raw.append(None)

        top1_idx, top2_idx = get_top_indices(row_values_raw)

        row_values = []
        for i, v in enumerate(row_values_raw):
            if v is not None:
                row_values.append(format_latex_value(v, i == top1_idx, i == top2_idx))
            else:
                row_values.append("-")

        latex_lines.append(f"{dataset} & {model} & " + " & ".join(row_values) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    error_f1_latex = "\n".join(latex_lines)
    all_latex.append(error_f1_latex)
    print("\n📄 LaTeX Table 1 (Error Detection F1):")
    print(error_f1_latex)
    
    # ========== Table 2: Correct Detection F1 ==========
    latex_lines = []
    latex_lines.append("\n\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{IC Correct Detection F1 (\\textbf{best}, \\underline{second best})}")
    latex_lines.append("\\small")

    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Model & " + " & ".join(available_methods) + " \\\\")
    latex_lines.append("\\midrule")

    for (dataset, model) in sorted(data.keys()):
        row_values_raw = []
        for method in available_methods:
            if method in data[(dataset, model)]:
                row_values_raw.append(data[(dataset, model)][method]["Correct_F1"])
            else:
                row_values_raw.append(None)

        top1_idx, top2_idx = get_top_indices(row_values_raw)

        row_values = []
        for i, v in enumerate(row_values_raw):
            if v is not None:
                row_values.append(format_latex_value(v, i == top1_idx, i == top2_idx))
            else:
                row_values.append("-")

        latex_lines.append(f"{dataset} & {model} & " + " & ".join(row_values) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    correct_f1_latex = "\n".join(latex_lines)
    all_latex.append(correct_f1_latex)
    print("\n📄 LaTeX Table 2 (Correct Detection F1):")
    print(correct_f1_latex)
    
    # ========== Table 3: AUROC ==========
    latex_lines = []
    latex_lines.append("\n\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{IC AUROC (\\textbf{best}, \\underline{second best})}")
    latex_lines.append("\\small")

    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Model & " + " & ".join(available_methods) + " \\\\")
    latex_lines.append("\\midrule")

    for (dataset, model) in sorted(data.keys()):
        row_values_raw = []
        for method in available_methods:
            if method in data[(dataset, model)]:
                row_values_raw.append(data[(dataset, model)][method]["AUROC"])
            else:
                row_values_raw.append(None)

        top1_idx, top2_idx = get_top_indices(row_values_raw)

        row_values = []
        for i, v in enumerate(row_values_raw):
            if v is not None:
                # AUROC uses 2 decimal places
                row_values.append(format_latex_value(v, i == top1_idx, i == top2_idx, decimals=2))
            else:
                row_values.append("-")

        latex_lines.append(f"{dataset} & {model} & " + " & ".join(row_values) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    auroc_latex = "\n".join(latex_lines)
    all_latex.append(auroc_latex)
    print("\n📄 LaTeX Table 3 (AUROC):")
    print(auroc_latex)

    # Save all tables
    if output_path:
        full_latex = "\n\n".join(all_latex)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_latex)
        print(f"\nLaTeX saved (3 tables): {output_path}")


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


def save_json(all_results: List[Dict], output_path: Path):
    """Save complete JSON results"""
    if not all_results:
        return
    
    # Filter out None results
    valid_results = [r for r in all_results if r["result"] is not None]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=2)
    
    print(f"Complete JSON saved: {output_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="IC Comprehensive Analysis and Display (CIFAR-10, ImageNet-1k)")
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
    print("IC Comprehensive Analysis (CIFAR-10, ImageNet-1k)")
    print("=" * 80)
    print("Ground Truth: obtained from the correct field in prediction files")

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

            # Load ground truth
            try:
                ground_truth = load_ground_truth(dataset, model)
                correct_count = sum(1 for v in ground_truth.values() if v)
                error_count = sum(1 for v in ground_truth.values() if not v)
                print(f"   GT samples: {len(ground_truth)} (correct: {correct_count}, errors: {error_count})")
            except Exception as e:
                print(f"   Failed to load GT: {e}")
                continue

            for method_name in methods:
                try:
                    method_scores = load_method_scores(dataset, model, method_name, weighted=args.weighted)
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
    dataset_suffix = "_".join(d.lower().replace("-", "") for d in datasets)
    latex_path = output_dir / f"ic_analysis_{dataset_suffix}.tex"
    generate_latex_table(all_results, latex_path)

    csv_path = output_dir / f"ic_analysis_{dataset_suffix}.csv"
    save_csv(all_results, csv_path)
    
    json_path = output_dir / f"ic_analysis_{dataset_suffix}.json"
    save_json(all_results, json_path)

    _ = groups
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

