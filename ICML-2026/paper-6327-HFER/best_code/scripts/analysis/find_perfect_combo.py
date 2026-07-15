import json
import numpy as np
import itertools
from tqdm import tqdm
import argparse

def search_perfect_combo(results_file, list_b_file=None):
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_file} not found.")
        return

    reclaimed_files = set()
    if list_b_file:
         try:
            with open(list_b_file, 'r') as f:
                reclaimed = json.load(f)
                reclaimed_files = set(item['file'] for item in reclaimed)
         except FileNotFoundError: pass

    # 1. Flatten Data
    metric_keys = ["hfer", "fiedler_value", "energy", "smoothness", "entropy", "mixing_time"]
    
    samples = []
    
    # Valid
    for item in data["valid"]:
        traj = item["trajectory"]
        feat = {}
        for i, layer_data in enumerate(traj):
            for k in metric_keys:
                val = layer_data.get(k)
                if val is not None:
                     feat[f"L{i}_{k}"] = val
        samples.append({'features': feat, 'label': 1, 'file': item['file']})
        
    # Invalid
    for item in data["invalid"]:
        traj = item["trajectory"]
        feat = {}
        for i, layer_data in enumerate(traj):
            for k in metric_keys:
                val = layer_data.get(k)
                if val is not None:
                     feat[f"L{i}_{k}"] = val
        
        label = 0
        if item['file'] in reclaimed_files:
            label = 1
            
        samples.append({'features': feat, 'label': label, 'file': item['file']})
        
    print(f"Analyzed {len(samples)} samples. Valid: {sum(s['label'] for s in samples)}, Invalid: {len(samples) - sum(s['label'] for s in samples)}")
    total = len(samples)
    print("Scanning single-feature rules...")

    # 2. Collect all possible split points for each feature
    feature_keys = samples[0]['features'].keys()
    
    best_rules = [] # (accuracy, description, rule_func)
    
    # Optimization: Pre-calculate feature columns
    feat_cols = {f: [] for f in feature_keys}
    labels = [s['label'] for s in samples]
    
    for s in samples:
        for f in feature_keys:
            feat_cols[f].append(s['features'][f])

    # Single Feature Search
    print("Searching Single Features...")
    for feat in feature_keys:
        vals = feat_cols[feat]
        # Test 10 quantiles as thresholds
        thresholds = np.unique(np.percentile(vals, np.linspace(0, 100, 11)))
        
        for t in thresholds:
            # Rule: > t
            pred_gt = [1 if v > t else 0 for v in vals]
            acc_gt = sum(1 for p, l in zip(pred_gt, labels) if p == l) / total
            
            # Rule: < t
            pred_lt = [1 if v < t else 0 for v in vals]
            acc_lt = sum(1 for p, l in zip(pred_lt, labels) if p == l) / total
            
            best_rules.append((acc_gt, f"{feat} > {t:.4f}", lambda s, f=feat, th=t: s['features'][f] > th))
            best_rules.append((acc_lt, f"{feat} < {t:.4f}", lambda s, f=feat, th=t: s['features'][f] < th))

    best_rules.sort(key=lambda x: x[0], reverse=True)
    print(f"Top Single Rule: {best_rules[0][1]} (Acc: {best_rules[0][0]*100:.1f}%)")
    
    # Combinatorial Search (Top 50 singles)
    print("Searching 2-Feature Combinations (AND logic)...")
    top_candidates = best_rules[:30] # Limit to top 30 to keep it fast
    
    combo_results = []
    
    for i in range(len(top_candidates)):
        rule1 = top_candidates[i]
        for j in range(i+1, len(top_candidates)):
            rule2 = top_candidates[j]
            
            # Skip if same feature (redundant)
            feat1 = rule1[1].split()[0]
            feat2 = rule2[1].split()[0]
            if feat1 == feat2: continue

            # Apply rule1 AND rule2
            preds = []
            for s in samples:
                is_valid = rule1[2](s) and rule2[2](s)
                preds.append(1 if is_valid else 0)
            
            acc = sum(1 for p, l in zip(preds, labels) if p == l) / total
            combo_results.append((acc, f"({rule1[1]}) AND ({rule2[1]})"))
    
    combo_results.sort(key=lambda x: x[0], reverse=True)
    
    print("\n" + "="*80)
    print("      TOP DISCRIMINATOR COMBINATIONS")
    print("="*80)
    for acc, desc in combo_results[:10]:
         print(f"Accuracy: {acc*100:.1f}% | Rule: {desc}")

    if combo_results and combo_results[0][0] == 1.0:
        print("\n[SUCCESS] Found a 100% perfect discriminator!")
    else:
        print("\n[RESULT] No 100% perfect AND-rule found among top candidates.")
        
    # Return best single and best combo
    return {
        "best_single": best_rules[0] if best_rules else (0, "None", None),
        "best_combo": combo_results[0] if combo_results else (0, "None")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="experiment_results_Llama-3.2-1B-Instruct.json", help="Path to results json")
    parser.add_argument("--list-b", type=str, default=None, help="Path to List B json")
    args = parser.parse_args()
    search_perfect_combo(args.file, args.list_b)
