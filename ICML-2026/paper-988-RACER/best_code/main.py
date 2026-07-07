#!/usr/bin/env python3
"""
Repeated RACER Evaluation Script

This script merges calibration and test datasets, then performs multiple random splits
to evaluate the stability of Risk-Aware Calibrated Efficient Routing (RACER) metrics (risk and Size).
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Set, Any
from torch.utils.data import DataLoader
import torch.nn.functional as F



# Adjust path to import local modules if necessary
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RACER import (
    RACER_Module,
    evaluate_racer,
)
from routers.factory import build_router
# from run_router import load_dataset_scores_and_labels

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset_scores_and_labels(dataset, router_model, device, batch_size=64, use_softmax=True):
    """
    Extract scores and labels from the dataset
    
    Returns:
        scores: np.ndarray, shape [n, M]
        labels: np.ndarray, shape [n, M]
        questions: list of strings (from raw data)
        model_names: list of strings
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_labels = []
    all_answers = []
    
    # Check if router is KNN (CPU only)
    is_knn = hasattr(router_model, 'model_name') and router_model.model_name == 'knn'
    target_device = 'cpu' if is_knn else device
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting scores"):
            # RouterDCDataset returns: (question_id, scores, dataset_id, cluster_id, answer_dict)
            inputs, labels_batch, dataset_ids, cluster_ids, answer_dicts = batch
            
            # Move inputs to target device (KNN uses CPU, others use GPU)
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(target_device)
            else:
                inputs = inputs.to(target_device)
            
            # Get router output
            logits, _ = router_model.forward(**inputs)
            
            if use_softmax:
                probs = F.softmax(logits, dim=1)
            else:
                probs = logits
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels_batch.numpy())

            batch_size_curr = len(labels_batch)

    probs_array = np.concatenate(all_probs, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    all_data = dataset.data 
    
    # Get questions and model_names from raw data
    questions = [sample['question'] for sample in dataset.data]
    
    # Get model_names
    first_sample = dataset.data[0]
    if 'outputs' in first_sample:
        model_names = [out['model'] for out in first_sample['outputs']]
    else:
        model_names = list(first_sample['scores'].keys())
    
    return probs_array, labels_array, questions, model_names, all_data


def run_repeated_racer_evaluation(args):
    """Run repeated RACER evaluation."""
    seed = args.seed
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"Repeated RACER Evaluation")
    print(f"{'='*80}")
    print(f"Router: {args.router_name}")
    print(f"Data Name: {args.data_name}")
    print(f"Alpha: {args.alpha}")
    print(f"Splits: {args.n_splits}")
    print(f"Test Ratio: {args.test_ratio}")
    print(f"Nonconformity Score: {args.racer_nonc_score}")
    print(f"Null Mode: {args.null_mode}")
    print(f"Do Augment: {args.do_augment}")

    # 1. Build router model
    router_model = build_router(args.router_name, args, device)
    
    # 2. Load datasets
    # We load train for router training.
    # We load cal and test to merge them.
    print("\n[1] Loading datasets...")
    train_data, cal_data, test_data = router_model.load_datasets(
        train_paths=args.train_paths,
        cal_paths=args.cal_paths,
        test_paths=args.test_paths,
        answer_path=args.answer_path,
        data_types=args.data_types
    )
    
    # 3. Train or Load Router
    if args.trained_router_path:
        print(f"\n Loading pre-trained router from: {args.trained_router_path}")
        state = torch.load(args.trained_router_path, map_location=device)
        router_model.load_state_dict(state)
    elif args.router_name in ['knn', 'mlp']:
        print(f"\n Training {args.router_name} router...")
        router_model.fit(device, train_data)
        
    # 4. Extract scores and labels from Calibration Set
    print("\n Extracting probs and labels from Calibration Set...")
    cal_probs, cal_labels, _, _, cal_answers = load_dataset_scores_and_labels(
        cal_data, router_model, device, args.eval_bs, args.use_softmax
    )
    
    # 5. Extract scores and labels from Test Set
    print("\n Extracting probs and labels from Test Set...")
    test_probs, test_labels, test_questions, model_names, test_answers = load_dataset_scores_and_labels(
        test_data, router_model, device, args.eval_bs, args.use_softmax
    )
            
    # Ensure shapes match
    if cal_probs.shape[1] != test_probs.shape[1]:
        raise ValueError(f"Mismatch in number of models: Cal={cal_probs.shape[1]}, Test={test_probs.shape[1]}")
    
    # Determine available weight types
    available_weight_types = ['router_scores']
    if len(cal_answers) > 0 and 'confidence' in cal_answers[0]:
        if 'binary_confidence' in cal_answers[0]['confidence']:
            available_weight_types.append('binary_confidence')
        if 'p_true' in cal_answers[0]['confidence']:
            available_weight_types.append('p_true')

    # === Repeated Evaluation ===
    # Main pool data for repeated evaluation
    probs_all = np.concatenate([cal_probs, test_probs], axis=0)
    labels_all = np.concatenate([cal_labels, test_labels], axis=0)
    answers_all = cal_answers + test_answers
    n_all = len(probs_all)
    results = []
    print(f"\n Starting Repeated Evaluation ({args.n_splits} splits)...")
    ratio_held_out = args.held_out_ratio
    ratio_test = args.test_ratio
    ratio_cal = 1 - ratio_held_out - ratio_test
    n_ho = int(n_all * ratio_held_out)
    n_cal = int(n_all * ratio_cal)
    n_test = n_all - n_ho - n_cal
    print(f"  Ratio of Held-out: {ratio_held_out}, Ratio of Test: {ratio_test}, Ratio of Calibration: {ratio_cal}")
    print(f"  Total samples: {n_all}, Held-out size: {n_ho}, Calibration size: {n_cal}, Test size: {n_test}")
    
    for i in tqdm(range(args.n_splits), desc="Repeated Splits"):
        current_seed = seed + i
        np.random.seed(current_seed)
        # Shuffle indices
        indices = np.random.permutation(n_all)

        idx_ho = indices[:n_ho]
        idx_cal = indices[n_ho : n_ho + n_cal]
        idx_test = indices[n_ho + n_cal :]
        
        # Held-out
        probs_ho = probs_all[idx_ho]
        labels_ho = labels_all[idx_ho]
        answers_ho = [answers_all[j] for j in idx_ho]
        
        # Calibration
        probs_cal = probs_all[idx_cal]
        labels_cal = labels_all[idx_cal]
        
        # Test
        probs_test = probs_all[idx_test]
        labels_test = labels_all[idx_test]
        answers_test = [answers_all[j] for j in idx_test]
        
        # 2. Phase 1: Calibrate on Cal Set
        racer_router = RACER_Module(method=args.racer_nonc_score, alpha=args.alpha, do_augment=args.do_augment)
        lambda_hat = racer_router.calibrate(probs_cal, labels_cal, null_mode=args.null_mode, verbose=True)
        
        # 3. Phase 2: Tune Temperature on Held-out Set
        # Prepare HO answers for evaluation
        ho_model_answers = []
        ho_gold_answers = []
        for item in answers_ho:
            if 'outputs' in item:
                preds = [out.get('pred_answer') for out in item['outputs']]
            else:
                preds = [None] * probs_ho.shape[1]
            ho_model_answers.append(preds)
            # ho_gold_answers.append(item.get('gold_num') or item.get('filtered_answer'))
            if 'gold_num' in item and item['gold_num'] is not None:
                gold = item['gold_num']
            elif 'filtered_answer' in item:
                gold = item['filtered_answer']
            else:
                gold = None
            ho_gold_answers.append(gold)

        # Prepare held-out confidences
        ho_confidences = []
        for item in answers_ho:
            conf_dict = {}
            if 'confidence' in item:
                conf = item['confidence']
                if 'binary_confidence' in conf:
                    conf_dict['binary_confidence'] = conf['binary_confidence']
                if 'p_true' in conf:
                    conf_dict['p_true'] = conf['p_true']
            ho_confidences.append(conf_dict)
        
        best_T_dict = {}
        best_acc_dict = {}
        
        # Search best temperature only if aggregation is needed
        compute_aggregation = (args.alpha <= 0.3)
        
        if compute_aggregation:
            # Search best temperature only if aggregation is needed
            T_candidates = np.logspace(np.log10(0.001), np.log10(10.0), 100).tolist()
            for wt in available_weight_types:
                best_T = 1.0
                best_acc = -1.0
            
                for T in T_candidates:
                    metrics_ho = evaluate_racer(
                        racer_router,
                        probs_ho,
                        labels_ho,
                        test_model_answers=ho_model_answers,
                        test_gold_answers=ho_gold_answers,
                        test_confidences=ho_confidences,
                        null_mode=args.null_mode,
                        compute_aggregation=True,  # compute aggregation to get accuracy
                        verbose=False,
                        model_names=model_names,
                        temperatures={wt: T},
                        weight_types=[wt],
                    )
                    acc = metrics_ho[f"acc_weighted_{wt}"]
                    if acc > best_acc:
                        best_acc = acc
                        best_T = T
                    
                best_T_dict[wt] = best_T
                best_acc_dict[wt] = best_acc
            vote_acc = metrics_ho["acc_majority_vote"]
            best_acc_dict["vote_acc"] = vote_acc
        else:
            # If alpha > 0.3, skip aggregation and use default temperature
            for wt in available_weight_types:
                best_T_dict[wt] = 1.0
                best_acc_dict[wt] = 0.0
            best_acc_dict["vote_acc"] = 0.0

        # 4. Phase 3: Evaluate on Test Set using Best T
        test_model_answers = []
        test_gold_answers = []
        test_confidences = []
        for item in answers_test:
            if 'outputs' in item:
                preds = [out.get('pred_answer') for out in item['outputs']]
            else:
                preds = [None] * probs_test.shape[1]
            test_model_answers.append(preds)
            if 'gold_num' in item and item['gold_num'] is not None:
                gold = item['gold_num']
            elif 'filtered_answer' in item:
                gold = item['filtered_answer']
            else:
                gold = None
            # gold = item.get('gold_num') or item.get('filtered_answer')
            test_gold_answers.append(gold)
            conf_dict = {}
            if 'confidence' in item:
                conf = item['confidence']
                if 'binary_confidence' in conf:
                    conf_dict['binary_confidence'] = conf['binary_confidence']
                if 'p_true' in conf:
                    conf_dict['p_true'] = conf['p_true']
            test_confidences.append(conf_dict)

        metrics = evaluate_racer(
            racer_router, 
            probs_test, 
            labels_test,
            test_model_answers=test_model_answers,
            test_gold_answers=test_gold_answers,
            test_confidences=test_confidences,
            null_mode=args.null_mode,
            model_names=model_names,
            temperatures=best_T_dict,
            weight_types=list(best_T_dict.keys()),
            compute_aggregation=(args.alpha <= 0.3),  # skip aggregation when alpha > 0.3
            verbose=True
        )
        metrics["selected_temperature"] = best_T_dict
        metrics["held_out_best_acc"] = best_acc_dict

        results.append(metrics)
                
    # 8. Analyze and Save Results
    print("\n[7] Analyzing Results...")
    
    error_rates = [r['risk'] for r in results]
    coverages = [r['coverage'] for r in results]
    avg_set_sizes = [r['avg_set_size'] for r in results]
    non_null_avg_set_sizes = [r['non_null_avg_set_size'] for r in results]
    abstention_rate = [r['abstention_rate'] for r in results]
    correct_abstention_rates = [r['correct_abstention_rate'] for r in results]
    
    mean_error = np.mean(error_rates)
    std_error = np.std(error_rates)
    mean_coverage = np.mean(coverages)
    std_coverage = np.std(coverages)
    mean_abstention = np.mean(abstention_rate)
    std_abstention = np.std(abstention_rate)
    mean_correct_abstention = np.mean(correct_abstention_rates)
    std_correct_abstention = np.std(correct_abstention_rates)
    mean_avg_set_size = np.mean(avg_set_sizes)
    std_avg_set_size = np.std(avg_set_sizes)
    mean_non_null_avg_set_size = np.mean(non_null_avg_set_sizes)
    std_non_null_avg_set_size = np.std(non_null_avg_set_sizes)
    
    print(f"Average Risk: {mean_error:.4f} ± {std_error:.4f} (Target: ≤{args.alpha})")
    print(f"Average Coverage: {mean_coverage:.4f} ± {std_coverage:.4f} (Target: ≥{1-args.alpha})")
    print(f"Average Avg Set Size: {mean_non_null_avg_set_size:.2f} ± {std_non_null_avg_set_size:.2f}")

    base_router_accs = [r['base_router_accuracy'] for r in results]
    best_single_accs = [r['best_single_model_info']['best_single_model_acc'] for r in results]

    # Compute mean accuracy per model across splits
    all_single_accs = np.array([r['single_model_accuracies'] for r in results])
    mean_single_accs = np.mean(all_single_accs, axis=0)
    std_single_accs = np.std(all_single_accs, axis=0)
    
    print(f"\n--- Accuracy Comparison (Mean ± Std) ---")
    print(f"Base Router:  {np.mean(base_router_accs):.4f} ± {np.std(base_router_accs):.4f}")
    
    # Print aggregation accuracy only when aggregation is computed
    if compute_aggregation:
        racer_agg_accs = [r['acc_majority_vote'] for r in results]
        print(f"RACER Aggregated (Majority):  {np.mean(racer_agg_accs):.4f} ± {np.std(racer_agg_accs):.4f}")
        # Collect accuracy per weight type
        racer_agg_accs_weighted = {}
        for wt in available_weight_types:
            racer_agg_accs_weighted[wt] = [r[f'acc_weighted_{wt}'] for r in results]
        # Print weighted accuracy for each weight type
        for wt in available_weight_types:
            mean_acc = np.mean(racer_agg_accs_weighted[wt])
            std_acc = np.std(racer_agg_accs_weighted[wt])
            print(f"RACER Aggregated (Weighted-{wt}): {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        racer_agg_accs = [0.0] * len(results)  # placeholder
        racer_agg_accs_weighted = {wt: [0.0] * len(results) for wt in available_weight_types}
        print("  (Aggregation skipped for alpha > 0.3)")

    print(f"Best Single Model: {np.mean(best_single_accs):.4f} ± {np.std(best_single_accs):.4f}")
    
    print(f"\nAverage Single Model Accuracies:")
    for m_idx, acc in enumerate(mean_single_accs):
        print(f"  Model {m_idx}: {acc:.4f}")
    
    # Save JSON
    os.makedirs(args.save_folder, exist_ok=True)
    json_path = os.path.join(args.save_folder, f"{args.data_name}_repeated_racer_results.json")
    weighted_acc_summary = {}
    for wt in available_weight_types:
        weighted_acc_summary[f"mean_racer_agg_acc_weighted_{wt}"] = float(np.mean(racer_agg_accs_weighted[wt]))
        weighted_acc_summary[f"std_racer_agg_acc_weighted_{wt}"] = float(np.std(racer_agg_accs_weighted[wt]))

    output_data = {
        "args": vars(args),
        "summary": {
            "mean_risk": float(mean_error),
            "std_risk": float(std_error),
            "mean_coverage": float(mean_coverage),
            "std_coverage": float(std_coverage),
            "mean_abstention_rate": float(mean_abstention),
            "std_abstention_rate": float(std_abstention),
            "mean_correct_abstention_rate": float(mean_correct_abstention),
            "std_correct_abstention_rate": float(std_correct_abstention),
            "mean_avg_set_size": float(mean_avg_set_size),
            "std_avg_set_size": float(std_avg_set_size),
            "mean_non_null_avg_set_size": float(mean_non_null_avg_set_size),
            "std_non_null_avg_set_size": float(std_non_null_avg_set_size),
            "mean_base_router_acc": float(np.mean(base_router_accs)),
            "mean_racer_agg_acc_majority": float(np.mean(racer_agg_accs)),
            "std_racer_agg_acc_majority": float(np.std(racer_agg_accs)),
            **weighted_acc_summary,
            "mean_best_single_acc": float(np.mean(best_single_accs)),
            "model_names": model_names,
            "mean_single_model_accs": mean_single_accs.tolist(),
            "std_single_model_accs": std_single_accs.tolist(),
        },
        "results": results
    }

    # Convert numpy types to python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    # Recursive conversion
    def recursive_convert(data):
        if isinstance(data, dict):
            return {k: recursive_convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(i) for i in data]
        else:
            return convert_numpy(data)
            
    output_data = recursive_convert(output_data)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # 9. Plotting
    print("\n[8] Plotting Histograms...")
    
    plt.figure(figsize=(12, 5))

    # risk Histogram
    # Primary colors (RGB normalized)
    Right = (234/255, 130/255, 103/255)        # R66 G112 B178
    Left = (122/255, 195/255, 165/255)          # R225 G26 B40
    # Reference line color
    LINE_TARGET = (31/255, 41/255, 55/255)   # R31 G41 B55 dark gray

    plt.subplot(1, 2, 1)
    plt.hist(error_rates, bins=15, alpha=0.65, color=Left, edgecolor='black', linewidth=0.5)
    plt.axvline(x=args.alpha, color=LINE_TARGET, linestyle='--', linewidth=2, label=f'Target Alpha ({args.alpha})')
    plt.axvline(x=mean_error, color=LINE_TARGET, linestyle='-', linewidth=2, label=f'Mean ({mean_error:.3f})')
    plt.title(f'risk Distribution\n(Target ≤ {args.alpha})')
    plt.xlabel('risk'); plt.ylabel('Frequency')
    plt.grid(alpha=0.25, linestyle='--')
    plt.legend()

    # Size Histogram
    plt.subplot(1, 2, 2)
    plt.hist(non_null_avg_set_sizes, bins=15, alpha=0.65, color=Right, edgecolor='black', linewidth=0.5)
    plt.axvline(x=mean_non_null_avg_set_size, color=LINE_TARGET, linestyle='-', linewidth=2, label=f'Mean ({mean_non_null_avg_set_size:.3f})')
    plt.title(f'Size Distribution')
    plt.xlabel('Size'); plt.ylabel('Frequency')
    plt.grid(alpha=0.25, linestyle='--')
    plt.legend()
        
    plt.tight_layout()
    plot_path = os.path.join(args.save_folder, f"{args.data_name}_racer_metrics_distribution.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    

def main():
    parser = argparse.ArgumentParser(description='Repeated RACER Evaluation')
    
    # Router Settings
    parser.add_argument('--router_name', type=str, default='routerdc',
                       choices=['routerdc', 'knn', 'mlp'],
                       help='Router Name')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Tokenizer/backbone model path')
    parser.add_argument('--trained_router_path', type=str, default='',
                       help='Pre-trained router weights path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    # Data Settings
    parser.add_argument('--data_name', type=str, required=True,
                       help='Dataset Name')
    parser.add_argument('--train_paths', type=str, required=True,
                       help='Train data paths')
    parser.add_argument('--cal_paths', type=str, required=True,
                       help='Calibration data paths')
    parser.add_argument('--test_paths', type=str, required=True,
                       help='Test data paths')
    parser.add_argument('--answer_path', type=str, default=None,
                       help='Test answer path (gsm8k_output_conf.json)')
    parser.add_argument('--data_types', type=str, default='multi_attempt',
                       help='Data Type')
    parser.add_argument("--data_format", type=str, default=None,
                        choices=["label", "score"],
                        help="Force dataset format; if None, auto-detect")    
    
    # RACER Settings
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Target risk')
    parser.add_argument('--racer_nonc_score', type=str, default='gap',
                       choices=['gap', 'one_minus_prob'],
                       help='RACER Method')
    parser.add_argument('--null_mode', type=str, default='one_minus_max',
                       choices=['mean_prob', 'median_prob', 'one_minus_max', 'static_score'],
                       help='Null Mode')
    parser.add_argument('--do_augment', type=bool, default=True, help='Do Augment')
    
    # Repeated Eval Settings
    parser.add_argument('--n_splits', type=int, default=50,
                       help='Number of random splits')
    parser.add_argument('--test_ratio', type=float, default=0.4,
                       help='Ratio of test set in each split')
    parser.add_argument('--held_out_ratio', type=float, default=0.1,
                       help='Ratio of held-out set in each split')
    
    # Training/Inference Settings
    parser.add_argument('--eval_bs', type=int, default=64,
                       help='Evaluation Batch Size')
    parser.add_argument('--train_bs', type=int, default=32,
                       help='Training Batch Size')
    parser.add_argument('--use_softmax', type=bool, default=True,
                       help='Apply softmax to logits')
    
    # KNN Params
    parser.add_argument('--knearest', type=int, default=8, help='KNN k')
    
    # MLP Params
    parser.add_argument('--lr', type=float, default=1e-4, help='MLP LR')
    parser.add_argument('--epoch', type=int, default=100, help='MLP Epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='MLP Weight Decay')
    parser.add_argument('--hidden_size', type=int, default=256, help='MLP Hidden Size')
    
    # Output Settings
    parser.add_argument('--save_folder', type=str, required=True,
                       help='Output save folder')
    
    args = parser.parse_args()
    
    run_repeated_racer_evaluation(args)

if __name__ == '__main__':
    main()

