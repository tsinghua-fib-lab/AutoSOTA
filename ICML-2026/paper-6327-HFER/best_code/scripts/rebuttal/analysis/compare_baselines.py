"""
Compare Baselines (CPU-computable)
===================================
Addresses: Reviewer 6GDW ("how does it compare to other self-supervised signals?"),
Reviewer mxpA Q1 ("Is HFER just a proxy for entropy?")

Part A: CPU baselines (proof length, majority class, random).
For token-level baselines, see extract_token_baselines.py (GPU).
For full comparison, see analyze_baselines.py.

Usage:
    cd geometry-of-reason
    python scripts/rebuttal/compare_baselines.py --results-dir data/results
"""

import os, sys, json, argparse, glob
import numpy as np
from sklearn.metrics import roc_auc_score

def load_data(results_dir, model_slug="Meta-Llama-3.1-8B-Instruct", 
              reclaimed_file="data/reclaimed/8B_list_b_confident_invalid.json"):
    results_file = os.path.join(results_dir, f"experiment_results_{model_slug}.json")
    if not os.path.exists(results_file):
        return None
    with open(results_file, 'r') as f:
        data = json.load(f)
    reclaimed = set()
    if os.path.exists(reclaimed_file):
        with open(reclaimed_file, 'r') as f:
            reclaimed = set(item['file'] for item in json.load(f))
    samples = []
    for label_type in ['valid', 'invalid']:
        for item in data.get(label_type, []):
            corrected = label_type
            if label_type == 'invalid' and item['file'] in reclaimed:
                corrected = 'valid'
            samples.append({
                'file': item['file'], 'label': 1 if corrected == 'valid' else 0,
                'trajectory': item.get('trajectory', [])
            })
    return samples

def proof_length_baseline(data_dir='data/experiment_ready'):
    """Compute proof length in tokens for each proof."""
    lengths = {}
    for label in ['valid', 'invalid']:
        for fp in glob.glob(os.path.join(data_dir, label, '*.lean')):
            with open(fp, 'r', encoding='utf-8') as f:
                text = f.read()
            lengths[os.path.basename(fp)] = len(text.split())
    return lengths

def optimal_threshold_acc(values, labels):
    """Find optimal threshold accuracy."""
    vals = np.array(values)
    labs = np.array(labels)
    thresholds = np.percentile(vals, np.linspace(0, 100, 100))
    best = 0
    for t in thresholds:
        acc1 = np.mean((vals < t) == labs)
        acc2 = np.mean((vals > t) == labs)
        best = max(best, acc1, acc2)
    return best

def run_baselines(args):
    print("=" * 70)
    print("  BASELINE COMPARISONS (CPU)")
    print("=" * 70)

    if not os.path.exists(args.results_dir):
        print(f"\nERROR: Results directory {args.results_dir} not found.")
        print("Transfer experiment JSONs from home PC first.")
        sys.exit(1)

    os.makedirs('output/rebuttal', exist_ok=True)
    os.makedirs('data/results/rebuttal', exist_ok=True)

    samples = load_data(args.results_dir)
    if not samples:
        print("ERROR: Could not load experiment results.")
        sys.exit(1)

    # Baselines
    n = len(samples)
    labels = [s['label'] for s in samples]
    n_valid = sum(labels)
    n_invalid = n - n_valid
    majority_acc = max(n_valid, n_invalid) / n

    print(f"\nDataset: {n_valid} valid, {n_invalid} invalid (total {n})")
    print(f"Majority class accuracy: {majority_acc*100:.1f}%")
    print(f"Random accuracy: 50.0%")

    # Proof length baseline
    lengths = proof_length_baseline()
    matched = [(lengths.get(s['file'], 0), s['label']) for s in samples if s['file'] in lengths]
    if matched:
        length_vals, length_labs = zip(*matched)
        length_acc = optimal_threshold_acc(length_vals, length_labs)
        print(f"Proof length accuracy: {length_acc*100:.1f}%")

    # HFER at best layer (from paper: L30 for Llama-8B)
    hfer_vals = []
    hfer_labs = []
    target_layer = 30
    for s in samples:
        traj = s['trajectory']
        if target_layer < len(traj):
            h = traj[target_layer].get('hfer')
            if h is not None:
                hfer_vals.append(h)
                hfer_labs.append(s['label'])
    if hfer_vals:
        hfer_acc = optimal_threshold_acc(hfer_vals, hfer_labs)
        print(f"HFER (L30) accuracy: {hfer_acc*100:.1f}%")

    print("\n[INFO] For token-level baselines (entropy, log-prob), run:")
    print("  python scripts/rebuttal/extract_token_baselines.py")
    print("Then: python scripts/rebuttal/analyze_baselines.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='data/results')
    run_baselines(parser.parse_args())
