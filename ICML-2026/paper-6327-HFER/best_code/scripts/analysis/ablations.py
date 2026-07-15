
import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_data(results_file, list_b_file=None):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    reclaimed_files = set()
    if list_b_file and os.path.exists(list_b_file):
        with open(list_b_file, 'r') as f:
            # Check if it's a list of strings or list of dicts
            raw_list = json.load(f)
            if raw_list and isinstance(raw_list[0], dict):
                reclaimed_files = set(item['file'] for item in raw_list)
            else:
                reclaimed_files = set(raw_list)
            
    samples = []
    # Process Valid
    for item in data['valid']:
        # Base Label 1
        samples.append({
            'file': item['file'],
            'label': 1,
            'features': item['summary'], # Use summary dict
            'trajectory': item['trajectory'],
            'set': 'valid'
        })
        
    # Process Invalid
    for item in data['invalid']:
        label = 0
        if item['file'] in reclaimed_files:
            label = 1 # Reclaim
        
        samples.append({
            'file': item['file'],
            'label': label,
            'features': item['summary'],
            'trajectory': item['trajectory'],
            'set': 'invalid' if label == 0 else 'reclaimed'
        })
    
    return samples

def ablation_random_baseline(samples):
    print("\n" + "="*50)
    print(" 1. RANDOM BASELINE")
    print("="*50)
    total = len(samples)
    valid = sum(1 for s in samples if s['label'] == 1)
    invalid = total - valid
    baseline = max(valid, invalid) / total
    print(f"Total: {total}")
    print(f"Valid: {valid} ({valid/total*100:.2f}%)")
    print(f"Invalid: {invalid} ({invalid/total*100:.2f}%)")
    print(f"Random Baseline Accuracy (Majority Class): {baseline*100:.2f}%")
    
def ablation_metric_correlation(samples):
    print("\n" + "="*50)
    print(" 2. METRIC CORRELATION MATRIX")
    print("="*50)
    # Extract metrics
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy", "energy"]
    data = []
    for s in samples:
        row = {}
        for m in metrics:
            val = s['features'].get(m)
            if val is not None:
                row[m] = val
        if row: data.append(row)
        
    df = pd.DataFrame(data)
    corr = df.corr()
    print(corr)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Constraint Independence Check (Feature Correlation)")
    plt.tight_layout()
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig(os.path.join("output", "ablation_correlation_matrix.png"))
    print("Saved output/ablation_correlation_matrix.png")

def ablation_threshold_robustness(samples, metric='hfer', layer_idx=30):
    print("\n" + "="*50)
    print(f" 3. THRESHOLD ROBUSTNESS ({metric} @ L{layer_idx})")
    print("="*50)
    
    # Extract X and Y
    X = []
    Y = []
    for s in samples:
        traj = s['trajectory']
        # Handle layer index
        idx = layer_idx if layer_idx < len(traj) else -1
        if idx >= len(traj): idx = len(traj)-1
        
        val = traj[idx].get(metric)
        if val is not None:
            X.append(val)
            Y.append(s['label'])
            
    # Find optimal threshold first
    best_acc = 0
    best_t = 0
    start, end = min(X), max(X)
    steps = np.linspace(start, end, 100)
    
    for t in steps:
        # Check < T (Llama style) or > T (Phi style)
        # We'll check both directions and pick best for the "Optimal"
        acc_lt = sum(1 for x, y in zip(X, Y) if (1 if x < t else 0) == y) / len(Y)
        acc_gt = sum(1 for x, y in zip(X, Y) if (1 if x > t else 0) == y) / len(Y)
        
        if acc_lt > best_acc:
            best_acc = acc_lt
            best_t = t
            direction = 'lt'
        if acc_gt > best_acc:
            best_acc = acc_gt
            best_t = t
            direction = 'gt'
            
    print(f"Optimal Threshold: {best_t:.4f} ({direction}) -> Acc: {best_acc*100:.2f}%")
    
    # Sweep +/- 20%
    variations = np.linspace(best_t * 0.8, best_t * 1.2, 20)
    accuracies = []
    
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Delta':<10}")
    for v in variations:
        if direction == 'lt':
            acc = sum(1 for x, y in zip(X, Y) if (1 if x < v else 0) == y) / len(Y)
        else:
            acc = sum(1 for x, y in zip(X, Y) if (1 if x > v else 0) == y) / len(Y)
        
        accuracies.append(acc)
        print(f"{v:<10.4f} {acc*100:<10.2f}% {(acc-best_acc)*100:<10.2f}")
        
    plt.figure(figsize=(10,5))
    plt.plot(variations, accuracies, marker='o')
    plt.axvline(best_t, color='r', linestyle='--', label='Optimal')
    plt.title(f"Threshold Robustness (Sensitivity)\nMetric: {metric} @ L{layer_idx}")
    plt.xlabel("Threshold Value")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig(os.path.join("output", "ablation_robustness_curve.png"))
    print("Saved output/ablation_robustness_curve.png")

def ablation_problem_difficulty(samples, metric='hfer', layer_idx=30, optimal_t=None, direction='lt'):
    print("\n" + "="*50)
    print(" 4. PROBLEM DIFFICULTY SPLIT (Olympiad vs Standard)")
    print("="*50)
    
    # Categories based on filename usually:
    # 'imo', 'putnam' -> Hard
    # 'amc', 'mathd' -> Easy/Medium
    # 'algebra', 'numbertheory' -> Topic (can be mixed)
    
    hard_keywords = ['imo', 'putnam', 'usamo']
    easy_keywords = ['amc', 'mathd', 'algebra', 'numbertheory'] # mathd is generally easier but broad
    
    groups = {'Hard (Olympiad)': [], 'Standard (AMC/MathD)': [], 'Other': []}
    
    for s in samples:
        fname = s['file'].lower()
        cat = 'Other'
        for k in hard_keywords:
            if k in fname:
                cat = 'Hard (Olympiad)'
                break
        if cat == 'Other': # Only check easy if not hard
            for k in easy_keywords:
                if k in fname:
                    cat = 'Standard (AMC/MathD)'
                    break
        
        # Get value for prediction
        traj = s['trajectory']
        idx = layer_idx if layer_idx < len(traj) else -1
        val = traj[idx].get(metric)
        
        if val is not None:
             if direction == 'lt':
                 pred = 1 if val < optimal_t else 0
             else:
                 pred = 1 if val > optimal_t else 0
             
             correct = 1 if pred == s['label'] else 0
             groups[cat].append(correct)
             
    for cat, correctness in groups.items():
        if not correctness:
            print(f"{cat}: N=0")
            continue
        acc = sum(correctness) / len(correctness)
        print(f"{cat:<20}: N={len(correctness):<4} Accuracy={acc*100:.2f}%")

def ablation_proof_length(samples, metric='hfer', layer_idx=30, optimal_t=None, direction='lt'):
    print("\n" + "="*50)
    print(" 5. PROOF LENGTH ANALYSIS")
    print("="*50)
    
    # Estimate specific proof length proxy using Energy/Smoothness ratio.
    # Energy = Smoothness * Norm, where Norm ~ Length * Dim.
    
    lengths = []
    correctness = []
    
    for s in samples:
        traj = s['trajectory']
        idx = layer_idx if layer_idx < len(traj) else -1
        res = traj[idx]
        
        e = res.get('energy', 0)
        sm = res.get('smoothness', 1e-9)
        norm = e / sm if sm > 1e-9 else 0
        
        # Norm is sum(x_i^2). Roughly proportional to N * dim.
        # This is strictly a proxy for length.
        
        # Get prediction
        val = res.get(metric)
        if val is not None:
             if direction == 'lt':
                 pred = 1 if val < optimal_t else 0
             else:
                 pred = 1 if val > optimal_t else 0
             
             correct = 1 if pred == s['label'] else 0
             lengths.append(norm)
             correctness.append(correct)
             
    # Bin by quintiles
    if not lengths:
        print("No length data.")
        return

    df = pd.DataFrame({'LengthProxy': lengths, 'Correct': correctness})
    df['Bin'] = pd.qcut(df['LengthProxy'], 5, labels=["Very Short", "Short", "Medium", "Long", "Very Long"])
    
    grouped = df.groupby('Bin')['Correct'].agg(['count', 'mean'])
    print(grouped)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=grouped.index, y=grouped['mean'])
    plt.title("Constraint Stability vs Proof Length (Estimated)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig(os.path.join("output", "ablation_length_analysis.png"))
    print("Saved output/ablation_length_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--list-b", type=str, required=True)
    parser.add_argument("--metric", type=str, default="hfer")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--direction", type=str, default="lt", help="lt or gt")
    parser.add_argument("--threshold", type=float, default=None)
    
    args = parser.parse_args()
    
    print(f"Loading {args.file} with corrections from {args.list_b}...")
    samples = load_data(args.file, args.list_b)
    
    ablation_random_baseline(samples)
    ablation_metric_correlation(samples)
    
    # Use defaults if not provided (will auto-optimize inside robustness function)
    ablation_threshold_robustness(samples, args.metric, args.layer)
    
    # Use provided or recalculate for splits
    # Just grab optimal from robustness pass? For simplicity let's re-optimize or use args if passed.
    # For this script let's just use the auto-optimization logic inside the functions or pass explicit.
    
    # Re-calc optimal T for the splits
    X = []
    Y = []
    for s in samples:
        traj = s['trajectory']
        idx = args.layer if args.layer < len(traj) else -1
        val = traj[idx].get(args.metric)
        if val is not None:
             X.append(val)
             Y.append(s['label'])
    
    best_acc = 0
    best_t = 0
    d = args.direction
    
    if args.threshold is None:
        # Quick search
        steps = np.linspace(min(X), max(X), 100)
        for t in steps:
            acc_lt = sum(1 for x, y in zip(X, Y) if (1 if x < t else 0) == y) / len(Y)
            acc_gt = sum(1 for x, y in zip(X, Y) if (1 if x > t else 0) == y) / len(Y)
            if acc_lt > best_acc:
                best_acc = acc_lt
                best_t = t
                d = 'lt'
            if acc_gt > best_acc:
                best_acc = acc_gt
                best_t = t
                d = 'gt'
        print(f"\n[Global Optimal] {args.metric} @ {best_t:.4f} ({d}) -> {best_acc:.2%}")
    else:
        best_t = args.threshold
        best_acc = 0 # Calc real one
        if d == 'lt':
             best_acc = sum(1 for x, y in zip(X, Y) if (1 if x < best_t else 0) == y) / len(Y)
        else:
             best_acc = sum(1 for x, y in zip(X, Y) if (1 if x > best_t else 0) == y) / len(Y)
        print(f"\n[Fixed Threshold] {args.metric} @ {best_t:.4f} ({d}) -> {best_acc:.2%}")

    ablation_problem_difficulty(samples, args.metric, args.layer, best_t, d)
    ablation_proof_length(samples, args.metric, args.layer, best_t, d)
