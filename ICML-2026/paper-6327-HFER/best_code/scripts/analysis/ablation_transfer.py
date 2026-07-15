
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_data(results_file, list_b_file=None):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    reclaimed_files = set()
    if list_b_file and os.path.exists(list_b_file):
        with open(list_b_file, 'r') as f:
            raw = json.load(f)
            if raw and isinstance(raw[0], dict):
                reclaimed_files = set(item['file'] for item in raw)
            else:
                reclaimed_files = set(raw)
            
    samples = []
    for item in data['valid']:
        samples.append({'label': 1, 'features': item['summary'], 'trajectory': item['trajectory']})
        
    for item in data['invalid']:
        label = 0
        if item['file'] in reclaimed_files:
            label = 1
        samples.append({'label': label, 'features': item['summary'], 'trajectory': item['trajectory']})
    
    return samples

def train_threshold(samples, metric, layer):
    X = []
    Y = []
    for s in samples:
        traj = s['trajectory']
        idx = layer if layer < len(traj) else -1
        val = traj[idx].get(metric)
        if val is not None:
            X.append(val)
            Y.append(s['label'])
            
    best_acc = 0
    best_t = 0
    best_dir = 'lt'
    
    # Grid search
    steps = np.linspace(min(X), max(X), 200)
    for t in steps:
        # Check < T
        acc_lt = sum(1 for x, y in zip(X, Y) if (1 if x < t else 0) == y) / len(Y)
        acc_gt = sum(1 for x, y in zip(X, Y) if (1 if x > t else 0) == y) / len(Y)
        
        if acc_lt > best_acc:
            best_acc = acc_lt
            best_t = t
            best_dir = 'lt'
        if acc_gt > best_acc:
            best_acc = acc_gt
            best_t = t
            best_dir = 'gt'
            
    return best_t, best_dir, best_acc, np.mean(X), np.std(X)

def test_threshold(samples, metric, layer, threshold, direction):
    X = []
    Y = []
    for s in samples:
        traj = s['trajectory']
        idx = layer if layer < len(traj) else -1
        val = traj[idx].get(metric)
        if val is not None:
            X.append(val)
            Y.append(s['label'])
            
    if direction == 'lt':
        preds = [1 if x < threshold else 0 for x in X]
    else:
        preds = [1 if x > threshold else 0 for x in X]
        
    acc = sum(1 for p, y in zip(preds, Y) if p == y) / len(Y)
    return acc, np.mean(X), np.std(X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Source (Train)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--source-list-b", required=True)
    # Target (Test)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--target-list-b", required=True)
    
    parser.add_argument("--metric", default="hfer")
    parser.add_argument("--layer", type=int, default=30)
    
    args = parser.parse_args()
    
    print(f"--- CROSS-MODEL TRANSFER ABLATION ---")
    print(f"Metric: {args.metric} @ Layer {args.layer}")
    
    # 1. Train on Source
    print(f"\nLoading Source: {os.path.basename(args.source_file)}")
    source_samples = load_data(args.source_file, args.source_list_b)
    t, d, acc, mu, std = train_threshold(source_samples, args.metric, args.layer)
    print(f"[Source Train] Best Threshold: {t:.4f} ({d})")
    print(f"[Source Train] Accuracy: {acc*100:.2f}%")
    print(f"[Source Stats] Mean: {mu:.4f}, Std: {std:.4f}")
    
    # 2. Test on Target
    print(f"\nLoading Target: {os.path.basename(args.target_file)}")
    target_samples = load_data(args.target_file, args.target_list_b)
    test_acc, t_mu, t_std = test_threshold(target_samples, args.metric, args.layer, t, d)
    
    print(f"[Target Test]  Accuracy: {test_acc*100:.2f}%")
    print(f"[Target Stats] Mean: {t_mu:.4f}, Std: {t_std:.4f}")
    
    # 3. Analysis
    drop = acc - test_acc
    print(f"\nTransfer Drop: {drop*100:.2f} points")
    if test_acc < 0.6:
        print("Transfer Failed. (Likely architectural mismatch or scale shift)")
    elif test_acc > 0.8:
        print("Transfer Successful! (Robust feature)")
    else:
        print("Transfer Partial. (Feature exists but shifted)")

    # 4. Check if flipped direction works (for Llama -> Phi case)
    print("\n[Ablation] Testing INVERTED logic on Target...")
    inv_d = 'gt' if d == 'lt' else 'lt'
    inv_acc, _, _ = test_threshold(target_samples, args.metric, args.layer, t, inv_d)
    print(f"[Inverted Test] Accuracy: {inv_acc*100:.2f}%")
