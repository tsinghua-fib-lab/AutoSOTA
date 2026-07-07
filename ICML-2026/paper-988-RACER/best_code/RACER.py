#!/usr/bin/env python3
"""
RACER (Risk-Aware Calibrated Efficient Routing) - prediction set construction

Core idea:
- For each input x, compute each model score r(x,m)
- Define gap(x,m) = r_max(x) - r(x,m)
- Prediction set S_λ(x) = {m : gap(x,m) ≤ λ}

RACER objective:
- P(S_hat(X) ∩ G(X) = ∅) ≤ α
- That is, with probability at least (1-α), the prediction set contains at least one optimal model
"""

import math
import json
import os
import numpy as np
import torch
from typing import List, Set, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
from collections import Counter, defaultdict

@dataclass
class CRCCalibrationResult:
    """Calibration result."""
    lambda_hat: float
    n_calibration: int
    alpha: float
    ell_values: List[float]  # Required threshold per sample
    coverage_estimate: float  # Estimated coverage
    method: str = "gap"  # Calibration method

def compute_non_conformity_scores(probs: np.ndarray, method: str = "gap", xi: float = 1e-6) -> np.ndarray:
    """
    Compute non-conformity scores (smaller is better).
    
    Args:
        probs: [n, M] model probability outputs
        method: 'gap' or 'one_minus_prob'
        xi: upper bound of random noise, epsilon ~ Uniform[0, xi], used to break ties
            and ensure strict exchangeability/continuity assumptions
        
        
    Definitions:
        1. Router Score-Gap: s(x, m) = r_max(x) - r(x, m)
           Reflects the gap to the best model.
        2. Probability-Based: s(x, m) = 1 - r(x, m)
           Reflects lack of absolute confidence.
    """
    # Generate negligible random noise: epsilon ~ Uniform[0, xi]
    epsilon = np.random.uniform(0, xi, size=probs.shape)
    
    if method == "gap":
        # r_max(x) for each sample
        # shape: [n, 1]
        r_max = np.max(probs, axis=1, keepdims=True)
        # s = r_max - r
        return r_max - probs + epsilon
    elif method == "one_minus_prob":
        # s = 1 - r
        return 1.0 - probs + epsilon
    else:
        raise ValueError(f"Unknown CRC method: {method}. Choose 'gap' or 'one_minus_prob'.")

def compute_good_sets_from_labels(labels: np.ndarray) -> List[Set[int]]:
    """
    Convert a binary label matrix into good-set list.
    labels: [n, M] (or [n, M+1])
    """
    good_sets = []
    for row in labels:
        # Find indices with label 1
        indices = np.where(row == 1)[0]
        good_sets.append(set(indices.tolist()))
    return good_sets

class RACER_Module:
    """
        RACER based on Score Gap or 1-Prob
        Responsibilities:
        1. Receive raw scores (probs)
        2. Compute non-conformity scores
        3. Add a null model (augment with tau_abs)
        4. Compute calibration threshold lambda_hat
        
        Supports two calibration methods:
        1. "gap" (default): score gap = r_max - r(x,m)
        - ℓ_i = min{r_max_i - r(x_i, m) : m ∈ G_i}
        - Prediction set: S_λ(x) = {m : r_max(x) - r(x,m) ≤ λ}
        
        2. "one_minus_prob": use 1 - probs as nonconformity score
        - ℓ_i = min{1 - probs(x_i, m) : m ∈ G_i}
        - Prediction set: S_λ(x) = {m : 1 - probs(x,m) ≤ λ}
        
        Usage:
        1. Init: router = RACER_Module(alpha=0.1, method="gap")
        2. Calibrate: lambda_hat = router.calibrate(scores_calib, good_sets)
        3. Predict: S_hat = router.predict_set(scores_new)
    """
    
    def __init__(self, method: str = "gap", do_augment: bool = True, alpha: float = 0.1):
        """
        Args:
            alpha: target error rate, P(S_hat ∩ G = ∅) ≤ alpha
            method: calibration method, "gap" or "one_minus_prob"
        """
        if method not in ["gap", "one_minus_prob"]:
            raise ValueError(f"Unknown method: {method}. Use 'gap' or 'one_minus_prob'")

        self.method = method
        self.do_augment = do_augment
        self.lambda_hat = None
        self.alpha = alpha
        
    def _augment_data(self, probs: np.ndarray, labels: np.ndarray, null_mode: str = "one_minus_max") -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data based on null_mode
        """
        n, m = probs.shape
        
        # 1. Expand labels
        has_correct = labels.sum(axis=1) > 0
        null_labels = (~has_correct).astype(int).reshape(-1, 1)
        labels_aug = np.hstack([labels, null_labels])
        
        # 2. Expand scores
        if null_mode == 'one_minus_max':
            # Mode: P(m0) = 1 - max(P)
            max_probs = np.max(probs, axis=1, keepdims=True)
            null_probs = 1.0 - max_probs
            probs_aug = np.hstack([probs, null_probs])
            scores_aug = compute_non_conformity_scores(probs_aug, self.method)
        else:
            raise ValueError(f"Unknown null mode: {null_mode}. Use 'one_minus_max'.")
            
        return scores_aug, labels_aug

    def calibrate(self, probs: np.ndarray, labels: np.ndarray, null_mode: str = "one_minus_max",
                  verbose: bool = True) -> CRCCalibrationResult:
        """
            Calibration: select global threshold λ_hat based on calibration set
            
            Args:
                probs: [n, M] raw probabilities
                labels: [n, M] labels (0/1)
                alpha: target risk level

            Returns:
                lambda_hat: float, global threshold
        """
        self.null_mode = null_mode
        # 1. Compute non-conformity scores
        if self.do_augment:
            scores, labels = self._augment_data(probs, labels, null_mode)
        else:
            scores = compute_non_conformity_scores(probs, self.method)

        good_sets = compute_good_sets_from_labels(labels)    
        
        n, M = scores.shape
        method_name = "Score-Gap" if self.method == "gap" else "1-Prob"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RACER {method_name} Calibration")
            print(f"{'='*60}")
            print(f"Calibration samples: {n}")
            print(f"Number of models: {M}")
            print(f"Target alpha: {self.alpha}")
            print(f"Method: {self.method}")
        
        # Compute ell_i = min_{m in G_i} score(m)
        ell_values = []
        for i in range(n):
            g_i = good_sets[i]
            if not g_i: 
                ell_values.append(float('inf'))
            else:
                min_score_in_good = min([scores[i, m] for m in g_i])
                ell_values.append(min_score_in_good)
        
        ell_values = np.array(ell_values)
        ell_sorted = np.sort(ell_values)
        
        if verbose:
            n_finite = len(ell_values)
            n_inf = len(ell_values) - n_finite
            print(f"\nSamples with finite ℓ: {n_finite}")
            print(f"Samples with ℓ=inf (no correct model): {n_inf}")
            if len(ell_values) > 0:
                print(f"ℓ statistics: min={min(ell_values):.4f}, "
                      f"max={max(ell_values):.4f}, "
                      f"mean={np.mean(ell_values):.4f}")
        
        # Count samples with empty good set
        n_empty = sum(1 for e in ell_sorted if not math.isfinite(e))
        
        # Step 3: scan candidate lambdas and find the smallest satisfying the risk bound
        # RACER condition: (#{i: ℓ_i > λ} + 1) / (n_calib + 1) ≤ α
        for i, lam in enumerate(ell_sorted):
            risk = (n - (i + 1) + 1) / (n + 1)
            if risk <= self.alpha:
                self.lambda_hat = lam
                break
        
        # If no candidate satisfies, use the largest candidate (most conservative)
        if self.lambda_hat is None:
            self.lambda_hat = max(ell_sorted) if ell_sorted else 1
            if verbose:
                print(f"Warning: No λ satisfies RACER condition, using max: {self.lambda_hat:.4f}")
        
        n_covered = np.sum(ell_values <= self.lambda_hat)
        coverage_estimate = n_covered / n
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Calibration Set Result")
            print(f"{'='*60}")
            print(f"Samples: {n - n_empty}")
            print(f"Selected λ_hat: {self.lambda_hat:.6f}")
            
            # Final error rate (all samples)
            n_bad_final = sum(1 for e in ell_sorted if e > self.lambda_hat)
            final_error = (n_bad_final + 1) / (n + 1)
            print(f"RACER error bound: {final_error:.4f} (target: ≤{self.alpha:.4f})")
            print(f"{'='*60}")
        
        return CRCCalibrationResult(
            lambda_hat=self.lambda_hat,
            n_calibration=n,
            alpha=self.alpha,
            ell_values=ell_values.tolist(),
            coverage_estimate=coverage_estimate,
            method=self.method
        )


    def predict_set(self, probs: np.ndarray, null_mode: str = "one_minus_max",) -> Set[int]:
        """
        Build prediction set using lambda_hat
        Args:
            probs: [n, M] raw probabilities
        Returns:
            S_hat: set of selected model indices
        """
        # Use values saved during calibration
        if null_mode is None:
            null_mode = getattr(self, 'null_mode', 'one_minus_max')
        if self.lambda_hat is None:
            raise ValueError("Router not calibrated. Call calibrate() first.")
        
        if self.do_augment:
            # Pass dummy labels to reuse the function; prediction does not need labels
            dummy_labels = np.zeros_like(probs)
            scores, _ = self._augment_data(probs, dummy_labels, null_mode) 
        else:
            scores = compute_non_conformity_scores(probs, self.method)
            
        predictions = []
        for row in scores:
            indices = np.where(row <= self.lambda_hat)[0].tolist()
            predictions.append(indices)
        return predictions

def normalize_answer(answer: str, prefer_numeric: bool = True, lowercase: bool = True) -> str:
    """Normalize answer string."""
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = answer.replace(',', '').replace('$', '').strip()
    if answer.endswith('.'):
        answer = answer[:-1]
    if prefer_numeric:
        try:
            f = float(answer.replace(",", ""))
            if math.isfinite(f):
                i = int(f)
                return str(i) if abs(f - i) < 1e-9 else str(f)
        except Exception:
            pass
    return answer.lower() if lowercase else answer

def aggregate_predictions(pred_answers: List[str], weights: Optional[List[float]] = None, 
                          probs: Optional[List[float]] = None) -> str:
    """
    Aggregate predictions.
    Args:
        pred_answers: list of candidate answers
        weights: optional weights
        probs: probabilities used to break ties (choose higher-probability answer)
    Returns:
        Final aggregated answer
    """
    if not pred_answers:
        return None
        
    normalized_answers = [normalize_answer(a) for a in pred_answers]
    
    if weights is None:
        # Majority vote
        counts = Counter(normalized_answers)
        max_count = max(counts.values())
        tied_answers = [a for a, c in counts.items() if c == max_count]
        
        if len(tied_answers) > 1 and probs is not None:
            # Break ties by average probability
            avg_prob = {}
            for a in tied_answers:
                idxs = [i for i, ans in enumerate(normalized_answers) if ans == a]
                avg_prob[a] = np.mean([probs[i] for i in idxs])
            return max(avg_prob, key=avg_prob.get)
        else:
            return counts.most_common(1)[0][0] if counts else None
    else:
        # Weighted vote
        weighted_counts = {}
        for ans, w in zip(normalized_answers, weights):
            if ans not in weighted_counts:
                weighted_counts[ans] = 0.0
            weighted_counts[ans] += w
        
        max_weight = max(weighted_counts.values())
        tied_answers = [a for a, w in weighted_counts.items() if w == max_weight]
        
        if len(tied_answers) > 1 and probs is not None:
            # Break ties by average probability
            avg_prob = {}
            for a in tied_answers:
                idxs = [i for i, ans in enumerate(normalized_answers) if ans == a]
                avg_prob[a] = np.mean([probs[i] for i in idxs])
            return max(avg_prob, key=avg_prob.get)
        else:
            return max(weighted_counts, key=weighted_counts.get)

def evaluate_racer(router: RACER_Module,
                        test_probs: np.ndarray,  # raw probabilities
                        test_labels: np.ndarray,
                        test_model_answers: List[List[str]] = None,  # optional
                        test_gold_answers: List[str] = None,        # optional
                        test_confidences: List[Dict] = None,  # per-sample confidence info
                        model_names: List[str] = None,              # optional
                        null_mode: str = "one_minus_max",
                        temperatures: Union[float, Dict[str, float]] = 1.0, 
                        weight_types: List[str] = None,  # weight types to use; None means all available
                        compute_aggregation: bool = True,  # compute answer aggregation (set False when alpha>0.3 to save time)
                        verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate RACER performance.
    
    Args:
        router: calibrated RACER_Module
        scores_test: test scores, shape [n_test, M]
        good_sets_test: test good sets
        verbose: whether to print details
        test_confidences: list of dicts with per-sample confidence info
            e.g. {'binary_confidence': [...], 'p_true': [...]}
        weight_types: list of weight types, options: ['router_scores', 'binary_confidence', 'p_true']
                     if None, use all available types
        temperatures: temperature parameters:
            - float: one temperature for all weight types
            - Dict[str, float]: per-weight temperature, e.g. {'router_scores': 0.5, 'binary_confidence': 1.0}
        
    Returns:
        metrics: dict of evaluation metrics
    """
    # 1. Predict
    pred_sets = router.predict_set(test_probs, null_mode=null_mode)

    # 2. Build labels_aug
    if router.do_augment:
        _, labels_aug = router._augment_data(test_probs, test_labels)
        null_model_idx = test_probs.shape[1]
    else:
        labels_aug = test_labels
        null_model_idx = -1

    total = len(test_probs)
    covered = 0
    set_sizes = []
    non_null_set_sizes = []

    abstentions = 0
    correct_abstentions = 0
    
    num_models = test_probs.shape[1]
    single_model_correct = np.zeros(num_models, dtype=int)
    base_router_correct = 0
    n_maj_correct = 0     
    n_aggregated_total = 0

    # Determine which weight types to try
    if weight_types is None:
        # Auto-detect available types
        available_types = ['router_scores']
        if test_confidences is not None and len(test_confidences) > 0:
            sample_conf = test_confidences[0]
            if 'binary_confidence' in sample_conf:
                available_types.append('binary_confidence')
            if 'p_true' in sample_conf:
                available_types.append('p_true')
        weight_types = available_types
    
    # Handle temperature parameters
    if isinstance(temperatures, (int, float)):
        temp_dict = {wt: float(temperatures) for wt in weight_types}
        default_temp = float(temperatures) 
    else:
        temp_dict = temperatures
        default_temp = 1.0 
    
    # Maintain counters per weight type
    weighted_correct = {wt: 0 for wt in weight_types}

    for i, S_hat in enumerate(pred_sets): 
        # --- Coverage Check (Based on Labels) ---
        # Covered if prediction set contains any label=1 model
        set_sizes.append(len(S_hat))
        valid_indices = [m for m in S_hat if m != null_model_idx]
        non_null_set_sizes.append(len(valid_indices))

        g_i = set(np.where(labels_aug[i] == 1)[0])
        if not set(S_hat).isdisjoint(g_i):
            covered += 1
        
        # Single candidate model accuracy and Base-router accuracy
        top1 = int(np.argmax(test_probs[i]))
        if g_i:
            if top1 in g_i:
                base_router_correct += 1
            for m in range(num_models):
                if m in g_i:
                    single_model_correct[m] += 1
        
        # Get gold answer
        gold_ans = normalize_answer(test_gold_answers[i])
        # --- Aggregation evaluation (multiple weights) ---
        if compute_aggregation:
            n_aggregated_total += 1
            # Check for Abstention
            if not valid_indices: 
                abstentions += 1 # Abstain
                real_ground_truth = np.where(test_labels[i] == 1)[0]
                if len(real_ground_truth) == 0:
                    correct_abstentions += 1
                continue
            
            s_answers = [test_model_answers[i][m] for m in valid_indices]
            
            # 1. Majority vote
            final_ans_maj = aggregate_predictions(s_answers, weights=None, probs=test_probs[i][valid_indices])
            if final_ans_maj == gold_ans and final_ans_maj is not None:
                n_maj_correct += 1
            
            # 2. Weighted vote
            # Extract scores/probs
            for wt in weight_types:
                if wt == 'router_scores':
                    raw_weights = test_probs[i][valid_indices]
                elif wt == 'binary_confidence':
                    if test_confidences is not None and 'binary_confidence' in test_confidences[i]:
                        conf_all = test_confidences[i]['binary_confidence']
                        raw_weights = np.array([conf_all[m] for m in valid_indices])
                    else:
                        continue  
                elif wt == 'p_true':
                    if test_confidences is not None and 'p_true' in test_confidences[i]:
                        conf_all = test_confidences[i]['p_true']
                        raw_weights = np.array([conf_all[m] for m in valid_indices])
                    else:
                        continue
                else:
                    continue

                T = temp_dict.get(wt, default_temp)
                exp_w = np.exp((raw_weights - np.max(raw_weights)) / T)
                s_weights = exp_w / np.sum(exp_w)
                # Aggregate
                final_ans_weighted = aggregate_predictions(s_answers, weights=s_weights, probs=test_probs[i][valid_indices])
                if final_ans_weighted == gold_ans and final_ans_weighted is not None:
                    weighted_correct[wt] += 1
        
    # Compute metrics over all samples
    coverage = covered / total if total > 0 else 0.0
    error_rate = 1 - coverage
    base_router_accuracy = base_router_correct / total if total > 0 else 0.0
    avg_set_size = np.mean(set_sizes)
    abstention_rate = abstentions / total if total > 0 else 0.0
    correct_abstention_rate = correct_abstentions / abstentions if abstentions > 0 else 0.0
    
    # Compute aggregation accuracy
    if compute_aggregation:
        acc_maj = n_maj_correct / n_aggregated_total if n_aggregated_total > 0 else 0.0
        acc_weighted = {}
        for wt in weight_types:
            acc_weighted[wt] = weighted_correct[wt] / n_aggregated_total if n_aggregated_total > 0 else 0.0

    # Compute per-model accuracy
    single_model_accuracies = (single_model_correct / total).tolist() if total > 0 else [0.0] * num_models
    
    # Find best single model
    best_single_model_acc = max(single_model_accuracies)
    best_single_model_idx = int(np.argmax(single_model_accuracies))
    best_single_model_name = model_names[best_single_model_idx]
    
    metrics = {
        "n_test": total,
        "lambda_hat": router.lambda_hat,
        "coverage": coverage, 
        "risk": error_rate,  
        "abstention_rate": abstention_rate,
        "correct_abstention_rate": correct_abstention_rate,
        "avg_set_size": avg_set_size,
        "non_null_avg_set_size": np.mean(non_null_set_sizes),
        "single_model_accuracies": single_model_accuracies,
        "best_single_model_info": {
            "best_single_model_acc": best_single_model_acc,
            "best_single_model_idx": best_single_model_idx,
            "best_single_model_name": best_single_model_name,
        },
        "base_router_accuracy": base_router_accuracy,  # P(top1 ∈ G)
    }

    # Add aggregation accuracy to metrics
    if compute_aggregation:
        metrics["acc_majority_vote"] = acc_maj
        for wt in weight_types:
            metrics[f"acc_weighted_{wt}"] = acc_weighted[wt]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RACER Evaluation Results")
        print(f"{'='*60}")
        print(f"Test samples: {total}")
        print(f"λ_hat: {router.lambda_hat:.6f}")
        
        print(f"\n--- Metrics ---")
        print(f"Risk: {metrics['risk']:.4f} "
              f"(target: ≤{router.alpha:.4f})")
        print(f"Abstention Rate: {metrics['abstention_rate']:.4f}")
        print(f"Correct Abstention Rate: {correct_abstention_rate:.4f}")
        
        print(f"\n--- Efficiency Metrics (Overall) ---")
        print(f"Non-Null Avg Set Size: {metrics['non_null_avg_set_size']:.2f}")

        if compute_aggregation:
            print(f"\n--- Downstream Application | Weighted Aggregation---")
            print(f"Majority Vote Acc: {acc_maj:.4f}")
            for wt in weight_types:
                print(f"Weighted ({wt}): {metrics[f'acc_weighted_{wt}']:.4f}")
        
        print(f"\n--- Baseline Comparison ---")
        print(f"Base Router Acc: {metrics['base_router_accuracy']:.4f}")
        print(f"Best Single Model Acc: {best_single_model_acc:.4f} (Model {best_single_model_name})")
        print(f"Single Models: {[f'{acc:.4f}' for acc in single_model_accuracies]}")
        
        print(f"{'='*60}")
    
    return metrics