import json
import numpy as np
import scipy.stats as stats
import os
import random
from itertools import combinations

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0: return 0
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def analyze_dual_metric(valid_samples, invalid_samples, single_results, top_n=15):
    """
    ALGO-A: Dual Spectral Metric Fusion via AND/OR Decision Rule.
    Searches for 2-feature AND and OR combinations that exceed single-feature accuracy.
    Reports best AND rule, best OR rule, and their Cohen's d values.
    """
    print(f"\n{'='*80}")
    print(f"  DUAL SPECTRAL METRIC FUSION (AND/OR Decision Rules)")
    print(f"{'='*80}")
    
    # Build a unified feature matrix: for each proof, store all (metric, layer) feature values
    metrics_list = ["hfer", "fiedler_value", "smoothness", "entropy", "energy"]
    
    def get_feature_values(samples, metric, layer):
        return np.array([s["trajectory"][layer][metric] for s in samples 
                        if s["trajectory"][layer][metric] is not None])
    
    # For each proof, extract all features
    all_valid = []
    all_invalid = []
    all_labels = []
    
    for s in valid_samples:
        feats = {}
        for layer in range(24):
            for metric in metrics_list:
                val = s["trajectory"][layer].get(metric)
                if val is not None:
                    feats[f"L{layer}_{metric}"] = val
        all_valid.append(feats)
        all_labels.append(1)
    
    for s in invalid_samples:
        feats = {}
        for layer in range(24):
            for metric in metrics_list:
                val = s["trajectory"][layer].get(metric)
                if val is not None:
                    feats[f"L{layer}_{metric}"] = val
        all_invalid.append(feats)
        all_labels.append(0)
    
    all_samples = all_valid + all_invalid
    n_total = len(all_samples)
    
    # Get top single-feature rules: sort by accuracy (descending) for best discriminators
    singles_by_acc = sorted(single_results, key=lambda x: x["acc"], reverse=True)
    top_singles = singles_by_acc[:top_n]
    
    # For each top single rule, determine its direction and threshold
    # We need to recompute the best threshold + direction for each top feature
    top_rules = []
    for r in top_singles:
        layer = r["layer"]
        metric = r["metric"]
        feat_name = f"L{layer}_{metric}"
        v_vals = np.array([s["trajectory"][layer][metric] for s in valid_samples if s["trajectory"][layer][metric] is not None])
        i_vals = np.array([s["trajectory"][layer][metric] for s in invalid_samples if s["trajectory"][layer][metric] is not None])
        
        all_vals = np.concatenate([v_vals, i_vals])
        n_all = len(all_vals)
        thresholds = np.unique(np.percentile(all_vals, np.linspace(0, 100, 50)))
        
        best_acc = 0
        best_dir = None
        best_t = None
        
        for t in thresholds:
            # Direction 1: Valid < T
            tp = sum(1 for x in v_vals if x < t)
            tn = sum(1 for x in i_vals if x >= t)
            acc_lt = (tp + tn) / n_all
            if acc_lt > best_acc:
                best_acc = acc_lt
                best_dir = "<"
                best_t = t
            
            # Direction 2: Valid > T
            tp2 = sum(1 for x in v_vals if x > t)
            tn2 = sum(1 for x in i_vals if x <= t)
            acc_gt = (tp2 + tn2) / n_all
            if acc_gt > best_acc:
                best_acc = acc_gt
                best_dir = ">"
                best_t = t
        
        top_rules.append({
            "feat": feat_name,
            "layer": layer,
            "metric": metric,
            "threshold": best_t,
            "direction": best_dir,
            "acc": r["acc"],
            "d": r["d"]
        })
    
    # Search for AND and OR combinations
    best_and = {"acc": 0, "rule1": None, "rule2": None}
    best_or = {"acc": 0, "rule1": None, "rule2": None}
    
    for i in range(len(top_rules)):
        r1 = top_rules[i]
        for j in range(i+1, len(top_rules)):
            r2 = top_rules[j]
            
            # Skip if same feature (but different layer/metric combos are fine)
            
            # Test AND rule: both must classify as valid
            and_preds = []
            for s in all_samples:
                v1 = s.get(r1["feat"], None)
                v2 = s.get(r2["feat"], None)
                if v1 is None or v2 is None:
                    and_preds.append(0)
                    continue
                if r1["direction"] == "<":
                    c1 = v1 < r1["threshold"]
                else:
                    c1 = v1 > r1["threshold"]
                if r2["direction"] == "<":
                    c2 = v2 < r2["threshold"]
                else:
                    c2 = v2 > r2["threshold"]
                and_preds.append(1 if (c1 and c2) else 0)
            
            and_acc = sum(1 for p, l in zip(and_preds, all_labels) if p == l) / n_total
            
            # Compute Cohen's d for AND rule: valid preds vs invalid preds
            and_valid_vals = [r1["d"] for _ in range(1)]  # placeholder
            
            if and_acc > best_and["acc"]:
                best_and = {"acc": and_acc, "rule1": r1, "rule2": r2, "preds": list(and_preds)}
            
            # Test OR rule: either classifies as valid
            or_preds = []
            for s in all_samples:
                v1 = s.get(r1["feat"], None)
                v2 = s.get(r2["feat"], None)
                if v1 is None or v2 is None:
                    or_preds.append(0)
                    continue
                if r1["direction"] == "<":
                    c1 = v1 < r1["threshold"]
                else:
                    c1 = v1 > r1["threshold"]
                if r2["direction"] == "<":
                    c2 = v2 < r2["threshold"]
                else:
                    c2 = v2 > r2["threshold"]
                or_preds.append(1 if (c1 or c2) else 0)
            
            or_acc = sum(1 for p, l in zip(or_preds, all_labels) if p == l) / n_total
            
            if or_acc > best_or["acc"]:
                best_or = {"acc": or_acc, "rule1": r1, "rule2": r2, "preds": list(or_preds)}
    
    # Report
    best_single_by_acc = max(single_results, key=lambda x: x["acc"])
    print(f"\nBest Single-Feature (by accuracy): {best_single_by_acc['acc']*100:.1f}% ({best_single_by_acc['metric']} L{best_single_by_acc['layer']}, d={best_single_by_acc['d']:.2f})")
    best_single_by_p = single_results[0]
    print(f"Best Single-Feature (by p-value): {best_single_by_p['acc']*100:.1f}% ({best_single_by_p['metric']} L{best_single_by_p['layer']}, d={best_single_by_p['d']:.2f})")
    
    if best_and.get("rule1"):
        r1 = best_and["rule1"]
        r2 = best_and["rule2"]
        ap = best_and.get("preds", [])
        if ap:
            and_tp = sum(1 for p, l in zip(ap, all_labels) if p == 1 and l == 1)
            and_fp = sum(1 for p, l in zip(ap, all_labels) if p == 1 and l == 0)
            and_fn = sum(1 for p, l in zip(ap, all_labels) if p == 0 and l == 1)
            and_tn = sum(1 for p, l in zip(ap, all_labels) if p == 0 and l == 0)
            and_precision = and_tp / (and_tp + and_fp) if (and_tp + and_fp) > 0 else 0
            and_recall = and_tp / (and_tp + and_fn) if (and_tp + and_fn) > 0 else 0
            and_specificity = and_tn / (and_tn + and_fp) if (and_tn + and_fp) > 0 else 0
        else:
            and_precision = and_recall = and_specificity = 0
        print(f"\nBest AND Rule: {r1['feat']} {r1['direction']} {r1['threshold']:.4f} AND {r2['feat']} {r2['direction']} {r2['threshold']:.4f}")
        print(f"  Accuracy: {best_and['acc']*100:.1f}% | Precision: {and_precision*100:.1f}% | Recall: {and_recall*100:.1f}% | Specificity: {and_specificity*100:.1f}%")
        print(f"  Constituent d-values: {r1['metric']} L{r1['layer']} d={r1['d']:.2f}, {r2['metric']} L{r2['layer']} d={r2['d']:.2f}")
        print(f"  (single1 acc={r1['acc']*100:.1f}%, single2 acc={r2['acc']*100:.1f}%)")
    
    if best_or.get("rule1"):
        r1 = best_or["rule1"]
        r2 = best_or["rule2"]
        op = best_or.get("preds", [])
        if op:
            or_tp = sum(1 for p, l in zip(op, all_labels) if p == 1 and l == 1)
            or_fp = sum(1 for p, l in zip(op, all_labels) if p == 1 and l == 0)
            or_fn = sum(1 for p, l in zip(op, all_labels) if p == 0 and l == 1)
            or_tn = sum(1 for p, l in zip(op, all_labels) if p == 0 and l == 0)
            or_precision = or_tp / (or_tp + or_fp) if (or_tp + or_fp) > 0 else 0
            or_recall = or_tp / (or_tp + or_fn) if (or_tp + or_fn) > 0 else 0
            or_specificity = or_tn / (or_tn + or_fp) if (or_tn + or_fp) > 0 else 0
        else:
            or_precision = or_recall = or_specificity = 0
        print(f"\nBest OR Rule: {r1['feat']} {r1['direction']} {r1['threshold']:.4f} OR {r2['feat']} {r2['direction']} {r2['threshold']:.4f}")
        print(f"  Accuracy: {best_or['acc']*100:.1f}% | Precision: {or_precision*100:.1f}% | Recall: {or_recall*100:.1f}% | Specificity: {or_specificity*100:.1f}%")
        print(f"  Constituent d-values: {r1['metric']} L{r1['layer']} d={r1['d']:.2f}, {r2['metric']} L{r2['layer']} d={r2['d']:.2f}")
        print(f"  (single1 acc={r1['acc']*100:.1f}%, single2 acc={r2['acc']*100:.1f}%)")
    
    # Return the best dual-metric result for scoring
    best_dual_acc = max(best_and["acc"], best_or["acc"])
    return {
        "best_and": best_and,
        "best_or": best_or,
        "best_dual_acc": best_dual_acc
    }


def analyze_model_corrected(model_name, results_file, list_b_file):
    print(f"\n{'='*80}")
    print(f"  FULL STATISTICAL ANALYSIS: {model_name} (CORRECTED LABELS)")
    print(f"{'='*80}")
    
    # 1. Load Original Data
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return
    with open(results_file, 'r') as f:
        data = json.load(f)

    # 2. Load List B (The "Hidden Gems" - Invalid labeled, but likely Valid)
    list_b_filenames = set()
    if list_b_file and os.path.exists(list_b_file):
        with open(list_b_file, 'r') as f:
            list_b = json.load(f)
            list_b_filenames = set(item['file'] for item in list_b)
    else:
        print(f"Warning: {list_b_file} not found. No relabeling will occur.")

    # 3. Relabel Data
    initial_valid_samples = []
    initial_invalid_samples = []
    
    # Process Original Valid
    for item in data["valid"]:
        initial_valid_samples.append(item)
        
    # Process Original Invalid
    for item in data["invalid"]:
        initial_invalid_samples.append(item)

    # --- BALANCING FOR MATH EXPERIMENT (Requested by User) ---
    if model_name == "Llama-1B-MATH":
        target_n = 49
        print(f"\n[INFO] Balancing dataset to {target_n} Valid vs {target_n} Invalid (randomly sampled)...")
        random.seed(42)
        
        if len(initial_valid_samples) >= target_n:
            final_valid = random.sample(initial_valid_samples, target_n)
        else:
            print(f"[WARNING] Only {len(initial_valid_samples)} valid samples available. Using all.")
            final_valid = initial_valid_samples

        if len(initial_invalid_samples) >= target_n:
            final_invalid = random.sample(initial_invalid_samples, target_n)
        else:
             print(f"[WARNING] Only {len(initial_invalid_samples)} invalid samples available. Using all.")
             final_invalid = initial_invalid_samples
             
        initial_valid_samples = final_valid
        initial_invalid_samples = final_invalid
        print(f"[INFO] Final Counts: {len(initial_valid_samples)} Valid, {len(initial_invalid_samples)} Invalid.")
    
    elif model_name == "Phi-3.5-MATH":
        pass

    elif model_name == "Qwen-MoE-MiniF2F":
        print(f"\n[INFO] Checking 50v50 split for Qwen-MoE-MiniF2F...")
        
        if len(initial_valid_samples) > 50:
            random.seed(42)
            initial_valid_samples = random.sample(initial_valid_samples, 50)
        if len(initial_invalid_samples) > 50:
            random.seed(42)
            initial_invalid_samples = random.sample(initial_invalid_samples, 50)
            
        print(f"[INFO] Final Counts: {len(initial_valid_samples)} Valid, {len(initial_invalid_samples)} Invalid.")

        taxonomy_path = "data/minif2f_moe_prepared/taxonomy.json"
        
        if os.path.exists("data/experiment_ready/taxonomy.json"):
            taxonomy_path = "data/experiment_ready/taxonomy.json"
            
        print(f"[INFO] Using Taxonomy File: {taxonomy_path}")
        
        if os.path.exists(taxonomy_path):
            with open(taxonomy_path, 'r') as f:
                taxonomy = json.load(f)
            
            print(f"\n[INFO] Performing Taxonomy Correlation Analysis...")
            
            invalid_logic = []
            invalid_calc = []
            invalid_incomplete = []
            
            for item in initial_invalid_samples:
                fname = os.path.basename(item['file'])
                category = taxonomy.get(fname, "Logic")
                
                if category == "Logic":
                    invalid_logic.append(item)
                elif category == "Calc":
                    invalid_calc.append(item)
                else: 
                    invalid_incomplete.append(item)
            
            invalid_logic.extend(invalid_incomplete)
            
            print(f"  - Logic/Incomplete Errors: {len(invalid_logic)}")
            print(f"  - Calculation Errors:    {len(invalid_calc)}")
            
            def get_vals(items, layer=12, metric="fiedler_value"):
                vs = []
                for x in items:
                    traj = x.get('trajectory', [])
                    if layer < len(traj):
                        val = traj[layer].get(metric)
                        if val is not None: vs.append(val)
                return vs

            v_vals = get_vals(initial_valid_samples)
            i_logic_vals = get_vals(invalid_logic)
            i_calc_vals = get_vals(invalid_calc)
            
            if len(v_vals) > 1 and len(i_logic_vals) > 1:
                d_logic = cohen_d(v_vals, i_logic_vals)
                print(f"  >> Valid vs Logic Error (d): {d_logic:.2f}")
                
            if len(v_vals) > 1 and len(i_calc_vals) > 1:
                d_calc = cohen_d(v_vals, i_calc_vals)
                print(f"  >> Valid vs Calc Error (d):  {d_calc:.2f}")
            else:
                print(f"  >> Not enough Calc errors to compute d.")

    valid_samples = []
    invalid_samples = []

    for item in initial_valid_samples:
        valid_samples.append(item)
            
    reclaimed = 0
    for item in initial_invalid_samples:
        if item['file'] in list_b_filenames:
            valid_samples.append(item)
            reclaimed += 1
        else:
            invalid_samples.append(item)
            
    print(f"Original: {len(data['valid'])} Valid, {len(data['invalid'])} Invalid")
    print(f"Corrected: {len(valid_samples)} Valid, {len(invalid_samples)} Invalid (Reclaimed {reclaimed})")
    
    # 4. Compute Stats for All Layers/Metrics
    metrics = ["hfer", "fiedler_value", "smoothness", "entropy", "energy"]
    num_layers = len(data["valid"][0]["trajectory"])
    
    results = []
    
    for layer in range(num_layers):
        for metric in metrics:
            v_vals = [s["trajectory"][layer][metric] for s in valid_samples if s["trajectory"][layer][metric] is not None]
            i_vals = [s["trajectory"][layer][metric] for s in invalid_samples if s["trajectory"][layer][metric] is not None]
            
            if len(v_vals) < 2 or len(i_vals) < 2:
                continue
                
            stat, p_mw = stats.mannwhitneyu(v_vals, i_vals)
            t_stat, p_t = stats.ttest_ind(v_vals, i_vals, equal_var=False)
            
            d = cohen_d(v_vals, i_vals)
            
            all_vals = np.concatenate([v_vals, i_vals])
            thresholds = np.unique(np.percentile(all_vals, np.linspace(0, 100, 50)))
            best_acc = 0
            
            for t in thresholds:
                # Direction 1: Valid < T
                tp = sum(1 for x in v_vals if x < t)
                tn = sum(1 for x in i_vals if x >= t)
                acc1 = (tp + tn) / len(all_vals)
                
                # Direction 2: Valid > T
                tp2 = sum(1 for x in v_vals if x > t)
                tn2 = sum(1 for x in i_vals if x <= t)
                acc2 = (tp2 + tn2) / len(all_vals)
                
                best_acc = max(best_acc, acc1, acc2)
            
            results.append({
                "layer": layer,
                "metric": metric,
                "p_mw": p_mw,
                "p_t": p_t,
                "d": d,
                "acc": best_acc,
                "mu_v": np.mean(v_vals),
                "mu_i": np.mean(i_vals)
            })
            
    # 5. Sort and Report Top 5
    results.sort(key=lambda x: x["p_mw"])
    
    print("\nTOP 10 DISCRIMINATORS (Corrected):")
    print(f"{'Metric':<15} {'Layer':<6} {'p(MWU)':<10} {'p(T-Test)':<10} {'Cohen d':<8} {'Accuracy':<8} {'Valid µ':<10} {'Invalid µ':<10}")
    print("-" * 100)
    for r in results[:10]:
        print(f"{r['metric']:<15} {r['layer']:<6} {r['p_mw']:.2e}   {r['p_t']:.2e}     {r['d']:>6.2f}   {r['acc']*100:>5.1f}%   {r['mu_v']:<10.3f} {r['mu_i']:<10.3f}")
    
    # --- ALGO-A: Dual Metric Fusion ---
    if model_name == "Qwen2.5-0.5B":
        dual_results = analyze_dual_metric(valid_samples, invalid_samples, results, top_n=15)
    
    return results

if __name__ == "__main__":
    analyze_model_corrected("Qwen2.5-0.5B", "data/results/experiment_results_Qwen2.5-0.5B-Instruct.json", "data/reclaimed/Qwen0.5B_list_b_confident_invalid.json")
    analyze_model_corrected("Llama-1B-MATH", "data/results/experiment_results_MATH_Llama-1B.json", None)
    analyze_model_corrected("Qwen-MoE-MiniF2F", "data/results/experiment_results_MiniF2F_Qwen-MoE.json", None)
    analyze_model_corrected("Qwen-MoE-Exp1", "data/results/experiment_results_Exp1_Qwen-MoE.json", "data/reclaimed/Qwen0.5B_list_b_confident_invalid.json")
