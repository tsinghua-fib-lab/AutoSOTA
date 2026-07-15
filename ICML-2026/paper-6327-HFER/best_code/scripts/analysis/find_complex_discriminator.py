import json
import numpy as np
import argparse

def load_data(results_file, list_b_file=None):
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_file} not found.")
        return []

    reclaimed_files = set()
    if list_b_file:
        try:
            with open(list_b_file, 'r') as f:
                reclaimed = json.load(f)
                reclaimed_files = set(item['file'] for item in reclaimed)
            print(f"Reclaiming {len(reclaimed_files)} valid proofs from List B.")
        except FileNotFoundError:
            print(f"Warning: {list_b_file} not found.")

    samples = []
    metric_keys = ["fiedler_value", "hfer", "smoothness", "entropy", "energy"]
    
    # Process Valid
    for item in data["valid"]:
        traj = item["trajectory"]
        feat = {}
        # Flatten trajectory
        feat["L_Last_Fiedler"] = item.get("final_fiedler", 0)
        
        # Add per-layer metrics
        for i, layer_data in enumerate(traj):
            for k in metric_keys:
                val = layer_data.get(k)
                if val is not None:
                    feat[f"L{i}_{k}"] = val
        
        # Calculate derived features AFTER flattening
        if "L0_smoothness" in feat and "L15_smoothness" in feat:
             feat["Delta_Smoothness_L1_L15"] = abs(feat["L0_smoothness"] - feat["L15_smoothness"])
             
        # Add final layer aliases (heuristic: last layer is usually interesting)
        last_layer = len(traj) - 1
        for k in metric_keys:
            if f"L{last_layer}_{k}" in feat:
                feat[f"Final_{k}"] = feat[f"L{last_layer}_{k}"]
        
        samples.append({'features': feat, 'label': 1, 'file': item['file']})

    # Process Invalid
    for item in data["invalid"]:
        traj = item["trajectory"]
        feat = {}
        feat["L_Last_Fiedler"] = item.get("final_fiedler", 0)
        for i, layer_data in enumerate(traj):
            for k in metric_keys:
                val = layer_data.get(k)
                if val is not None:
                    feat[f"L{i}_{k}"] = val
        
        # Check if actually valid (Reclaimed)
        label = 0
        if item['file'] in reclaimed_files:
            label = 1
            
        samples.append({'features': feat, 'label': label, 'file': item['file']})
                
    return samples

def train_sieve(samples, max_depth=5):
    current_pool = samples
    rules = []
    
    print(f"\nStarting Sieve Search (Max Depth {max_depth})...")
    
    total_valid_start = sum(1 for s in samples if s['label'] == 1)
    
    for depth in range(max_depth):
        valid_pool = [s for s in current_pool if s['label'] == 1]
        invalid_pool = [s for s in current_pool if s['label'] == 0]
        
        if not invalid_pool:
            break
            
        # Find rule that rejects max invalid while keeping >98% valid
        best_rule = None
        best_n_removed = -1
        
        # Candidate features
        features = list(valid_pool[0]['features'].keys())
        
        for feat in features:
            vals = [s['features'][feat] for s in current_pool if feat in s['features']]
            if not vals: continue
            
            # Percentiles as thresholds
            thresholds = np.unique(np.percentile(vals, np.linspace(0, 100, 20)))
            
            for t in thresholds:
                # Rule 1: value < t
                kept_v = sum(1 for s in valid_pool if s['features'].get(feat, 0) < t)
                if kept_v >= 0.98 * len(valid_pool):
                    removed_i = sum(1 for s in invalid_pool if not (s['features'].get(feat, 0) < t))
                    if removed_i > best_n_removed:
                        best_n_removed = removed_i
                        best_rule = (feat, "<", t)
                        
                # Rule 2: value > t
                kept_v = sum(1 for s in valid_pool if s['features'].get(feat, 0) > t)
                if kept_v >= 0.98 * len(valid_pool):
                    removed_i = sum(1 for s in invalid_pool if not (s['features'].get(feat, 0) > t))
                    if removed_i > best_n_removed:
                        best_n_removed = removed_i
                        best_rule = (feat, ">", t)

        if best_rule and best_n_removed > 0:
            f, op, t = best_rule
            print(f"  Level {depth+1}: Pool has {len(valid_pool)} Valid, {len(invalid_pool)} Invalid")
            print(f"    Selected Rule: {f} {op} {t:.4f}")
            
            # Count lost valid
            op_lambda = (lambda x: x < t) if op == "<" else (lambda x: x > t)
            kept_v_count = sum(1 for s in valid_pool if op_lambda(s['features'].get(f,0)))
            lost = len(valid_pool) - kept_v_count
            
            print(f"    Result: Removed {best_n_removed} Invalid, Lost {lost} Valid")
            
            rules.append(best_rule)
            # Apply filter
            if op == "<":
                current_pool = [s for s in current_pool if s['features'].get(f,0) < t]
            else:
                current_pool = [s for s in current_pool if s['features'].get(f,0) > t]
        else:
            print("  No effective rule found at this level.")
            break
            
    # Final Eval
    total_samples = len(samples)
    correct = 0
    predictions = []
    
    for s in samples:
        is_pred_valid = True
        for f, op, t in rules:
            val = s['features'].get(f, 0)
            if op == "<":
                if not (val < t): is_pred_valid = False
            else:
                if not (val > t): is_pred_valid = False
        
        if is_pred_valid and s['label'] == 1: correct += 1
        if not is_pred_valid and s['label'] == 0: correct += 1
        
    acc = correct / total_samples
    
    # Calculate Precision/Recall
    tp = sum(1 for s in samples if s['label'] == 1 and all((s['features'].get(f,0)<t if op=='<' else s['features'].get(f,0)>t) for f,op,t in rules))
    fp = sum(1 for s in samples if s['label'] == 0 and all((s['features'].get(f,0)<t if op=='<' else s['features'].get(f,0)>t) for f,op,t in rules))
    fn = sum(1 for s in samples if s['label'] == 1 and not all((s['features'].get(f,0)<t if op=='<' else s['features'].get(f,0)>t) for f,op,t in rules))
    
    prec = tp / (tp + fp) if (tp+fp) > 0 else 0
    rec = tp / (tp + fn) if (tp+fn) > 0 else 0

    print("\n" + "="*80)
    print(f"COMPLEX DISCRIMINATOR RESULT (Depth {len(rules)})")
    print("="*80)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"Final Pool: {tp} Valid (retained), {fp} Invalid (leaked)")
    print("-" * 80)
    print("Logic: VALID IF")
    for i, (f, op, t) in enumerate(rules):
        prefix = "      " if i == 0 else "  AND "
        print(f"{prefix}({f} {op} {t:.4f})")
    print("="*80)
    
    print("="*80)
    print("USER HYPOTHESIS CHECK")
    print("-" * 80)
    # Check specific derived features
    special_feats = ["Delta_Smoothness_L1_L15", "Final_energy", "Final_hfer", "L9_fiedler_value"]
    
    for feat in special_feats:
        vals = [s['features'].get(feat) for s in samples if s['features'].get(feat) is not None]
        if not vals: continue
        
        # Simple separation check (Mann-Whitney U or just difference in means)
        v_vals = [s['features'][feat] for s in samples if s['label'] == 1 and s['features'].get(feat) is not None]
        i_vals = [s['features'][feat] for s in samples if s['label'] == 0 and s['features'].get(feat) is not None]
        
        mu_v = np.mean(v_vals)
        mu_i = np.mean(i_vals)
        
        # Best single split accuracy
        best_acc = 0
        thresholds = np.unique(np.percentile(vals, np.linspace(0, 100, 20)))
        for t in thresholds:
            acc1 = (sum(1 for v in v_vals if v > t) + sum(1 for i in i_vals if i <= t)) / len(samples)
            acc2 = (sum(1 for v in v_vals if v < t) + sum(1 for i in i_vals if i >= t)) / len(samples)
            best_acc = max(best_acc, acc1, acc2)
            
        print(f"{feat:<25} | Valid µ={mu_v:.3f} | Invalid µ={mu_i:.3f} | Best Single Acc={best_acc*100:.1f}%")

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "depth": len(rules)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="experiment_results_Llama-3.2-1B-Instruct.json", help="Path to results json")
    parser.add_argument("--list-b", type=str, default=None, help="Path to List B json (confident invalids to reclaim)")
    args = parser.parse_args()
    
    samples = load_data(args.file, args.list_b)
    
    train_sieve(samples, max_depth=5)
