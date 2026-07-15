import json
import numpy as np
import scipy.stats as stats
import os
import random

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0: return 0
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

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
        random.seed(42) # Ensure reproducibility
        
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
        pass # Removed as experiment was cancelled

    elif model_name == "Qwen-MoE-MiniF2F":
        print(f"\n[INFO] Checking 50v50 split for Qwen-MoE-MiniF2F...")
        
        # Ensure strict 50/50
        if len(initial_valid_samples) > 50:
            random.seed(42)
            initial_valid_samples = random.sample(initial_valid_samples, 50)
        if len(initial_invalid_samples) > 50:
            random.seed(42)
            initial_invalid_samples = random.sample(initial_invalid_samples, 50)
            
        print(f"[INFO] Final Counts: {len(initial_valid_samples)} Valid, {len(initial_invalid_samples)} Invalid.")

        # --- TAXONOMY ANALYSIS (Move B) ---
        taxonomy_path = "data/minif2f_moe_prepared/taxonomy.json"
        
        # Prefer the Experiment Ready taxonomy if available (for larger datasets)
        if os.path.exists("data/experiment_ready/taxonomy.json"):
            taxonomy_path = "data/experiment_ready/taxonomy.json"
            
        print(f"[INFO] Using Taxonomy File: {taxonomy_path}")
        
        if os.path.exists(taxonomy_path):
            with open(taxonomy_path, 'r') as f:
                taxonomy = json.load(f)
            
            print(f"\n[INFO] Performing Taxonomy Correlation Analysis...")
            
            # Group invalid samples by type
            invalid_logic = []
            invalid_calc = []
            invalid_incomplete = []
            
            for item in initial_invalid_samples:
                fname = os.path.basename(item['file'])
                category = taxonomy.get(fname, "Logic") # Default to logic
                
                if category == "Logic":
                    invalid_logic.append(item)
                elif category == "Calc":
                    invalid_calc.append(item)
                else: 
                    invalid_incomplete.append(item) # "Logic_Incomplete"
            
            # Treat "Logic_Incomplete" as "Logic" for the main hypothesis? Or separate?
            # User wants generic "Logic" detection. Incomplete is usually a logic failure.
            invalid_logic.extend(invalid_incomplete)
            
            print(f"  - Logic/Incomplete Errors: {len(invalid_logic)}")
            print(f"  - Calculation Errors:    {len(invalid_calc)}")
            
            # We need to compute d for Valid vs Logic and Valid vs Calc
            # We need to extract the metrics first. This function usually does it later.
            # But we can do a quick check here if we have the data.
            # The 'item' has 'trajectory'.
            # Let's pick a layer (e.g. 12 or best) and metric (e.g. Fiedler or HFER).
            # We'll just print it plainly.
            
            def get_vals(items, layer=12, metric="fiedler_value"):
                vs = []
                for x in items:
                    traj = x.get('trajectory', [])
                    if layer < len(traj):
                        val = traj[layer].get(metric)
                        if val is not None: vs.append(val)
                return vs

            # Layer 12 Fiedler (Standard)
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
                
        # ----------------------------------
    # ---------------------------------------------------------
    # ---------------------------------------------------------

    valid_samples = []
    invalid_samples = []

    # Now apply relabeling to the (potentially balanced) initial samples
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
            # Extract values
            v_vals = [s["trajectory"][layer][metric] for s in valid_samples if s["trajectory"][layer][metric] is not None]
            i_vals = [s["trajectory"][layer][metric] for s in invalid_samples if s["trajectory"][layer][metric] is not None]
            
            if len(v_vals) < 2 or len(i_vals) < 2:
                continue
                
            # Mann-Whitney U
            stat, p_mw = stats.mannwhitneyu(v_vals, i_vals)
            # Welch's t-test
            t_stat, p_t = stats.ttest_ind(v_vals, i_vals, equal_var=False)
            
            # Cohen's d
            d = cohen_d(v_vals, i_vals)
            
            # Accuracy (Optimal Threshold)
            all_vals = np.concatenate([v_vals, i_vals])
            thresholds = np.unique(np.percentile(all_vals, np.linspace(0, 100, 50)))
            best_acc = 0
            
            # Check both directions (< and >)
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
        
    return results

if __name__ == "__main__":
    # analyze_model_corrected("Llama-3.2-1B", "data/results/experiment_results_Llama-3.2-1B-Instruct.json", "data/reclaimed/1B_list_b_confident_invalid.json")
    # analyze_model_corrected("Llama-3.2-3B", "data/results/experiment_results_Llama-3.2-3B-Instruct.json", "data/reclaimed/3B_list_b_confident_invalid.json")
    # analyze_model_corrected("Meta-Llama-3.1-8B", "data/results/experiment_results_Meta-Llama-3.1-8B-Instruct.json", "data/reclaimed/8B_list_b_confident_invalid.json")
    # analyze_model_corrected("Qwen2.5-7B", "data/results/experiment_results_Qwen2.5-7B-Instruct.json", "data/reclaimed/Qwen7B_list_b_confident_invalid.json")
    analyze_model_corrected("Qwen2.5-0.5B", "data/results/experiment_results_Qwen2.5-0.5B-Instruct.json", "data/reclaimed/Qwen0.5B_list_b_confident_invalid.json")
    # analyze_model_corrected("Phi-3.5-mini", "data/results/experiment_results_Phi-3.5-mini-instruct.json", "data/reclaimed/Phi3.5_list_b_confident_invalid.json")
    # analyze_model_corrected("Mistral-7B-v0.1", "data/results/experiment_results_Mistral-7B-v0.1.json", "data/reclaimed/Mistral7B_list_b_confident_invalid.json")
    analyze_model_corrected("Llama-1B-MATH", "data/results/experiment_results_MATH_Llama-1B.json", None)
    analyze_model_corrected("Qwen-MoE-MiniF2F", "data/results/experiment_results_MiniF2F_Qwen-MoE.json", None)
    # Exp 4: MoE on Main Dataset (Using Qwen0.5B reclaimed list to match the dataset source labels)
    analyze_model_corrected("Qwen-MoE-Exp1", "data/results/experiment_results_Exp1_Qwen-MoE.json", "data/reclaimed/Qwen0.5B_list_b_confident_invalid.json")
