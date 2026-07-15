
import os
import json
import glob
import traceback
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

def parse_args():
    parser = argparse.ArgumentParser(description="SpectraProof Experiment CLI")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        help="HuggingFace model name (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct)")
    parser.add_argument("--data-dir", type=str, default="data/proofs", help="Directory containing valid/invalid subfolders (default: data/proofs)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda)")
    parser.add_argument("--offline", action="store_true", help="Use only local cached models")
    parser.add_argument("--plot-only", action="store_true", help="Skip inference and plot from existing json")
    parser.add_argument("--layer", type=int, default=12, help="Target layer for scatter/discrimination/filtering (default: 12)")
    parser.add_argument("--disc-layer", type=int, default=None, help="Specific layer for discrimination (defaults to --layer if not set)")
    parser.add_argument("--geq", nargs=2, metavar=('METRIC', 'VALUE'), help="Filter: Metric >= Value (e.g. fiedler_value 0.8)")
    parser.add_argument("--leq", nargs=2, metavar=('METRIC', 'VALUE'), help="Filter: Metric <= Value")
    parser.add_argument("--search", action="store_true", help="Grid search for the perfect discriminator rule")
    parser.add_argument("--early-warning", action="store_true", help="Analyze early layers (5-10) for leading indicators")
    parser.add_argument("--stats", action="store_true", help="Generate detailed statistical report (p-values, CIs, Effect Sizes)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--results-file", type=str, default=None, help="Path to save/load results json (optional)")
    parser.add_argument("--extension", type=str, default="lean", help="File extension to scan for (default: lean)")
    return parser.parse_args()

def compute_t_stats(valid, invalid):
    """Compute basic stats + Mann-Whitney + Cohen's d + Bootstrap CI"""
    if len(valid) < 2 or len(invalid) < 2:
        return {}
        
    mu_v, std_v = np.mean(valid), np.std(valid)
    mu_i, std_i = np.mean(invalid), np.std(invalid)
    
    # Mann-Whitney U
    u_stat, p_val = stats.mannwhitneyu(valid, invalid, alternative='two-sided')
    
    # Cohen's d
    n_v, n_i = len(valid), len(invalid)
    pooled_std = np.sqrt(((n_v - 1)*std_v**2 + (n_i - 1)*std_i**2) / (n_v + n_i - 2))
    d = (mu_v - mu_i) / pooled_std if pooled_std > 0 else 0
    
    # Bootstrap 95% CI of difference
    diffs = []
    for _ in range(1000):
        v_sample = np.random.choice(valid, size=n_v, replace=True)
        i_sample = np.random.choice(invalid, size=n_i, replace=True)
        diffs.append(np.mean(v_sample) - np.mean(i_sample))
    
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return {
        "mu_v": mu_v, "mu_i": mu_i,
        "p": p_val, "d": d,
        "ci": (ci_lower, ci_upper)
    }

def generate_statistical_report(data):
    print("\n" + "="*80)
    print(" STATISTICAL PROFILING REPORT (Phase 2)")
    print("="*80)
    
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    # Dynamic last layer detection
    sample_traj = data["valid"][0]["trajectory"] if data["valid"] else []
    last_layer_idx = len(sample_traj) - 1 if sample_traj else 31
    layers = [6, last_layer_idx] # Gate and Embedding
    
    print(f"{'Metric':<15} {'Layer':<6} {'Valid':<8} {'Invalid':<8} {'p-value':<10} {'Cohen d':<8} {'95% CI (Diff)':<20}")
    print("-" * 90)

    for metric in metrics:
        for layer in layers:
            # Extract data
            def get_vals(key):
                vals = []
                for item in data[key]:
                    traj = item["trajectory"]
                    if layer < len(traj):
                        val = traj[layer][metric]
                        if val is not None: vals.append(val)
                return vals

            v_vals = get_vals("valid")
            i_vals = get_vals("invalid")
            
            s = compute_t_stats(v_vals, i_vals)
            if not s:
                continue
                
            layer_name = "Last" if layer == last_layer_idx else str(layer)
            ci_str = f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
            
            # Highlight significant
            sig = "*" if s['p'] < 0.05 else ""
            if s['p'] < 0.001: sig = "***"
            
            print(f"{metric:<15} {layer_name:<6} {s['mu_v']:.3f}    {s['mu_i']:.3f}    {s['p']:.2e}{sig:<3} {s['d']:.3f}    {ci_str:<20}")

def run_experiment(args):
    # Default disc_layer to layer if not set
    if args.disc_layer is None:
        args.disc_layer = args.layer

    # Determine Results File
    if args.results_file:
        results_file = args.results_file
    else:
        # Default strategy
        model_slug = args.model.split("/")[-1]
        results_file = f"experiment_results_{model_slug}.json"
        
        # Smart Check: If plot/stats/search, check data/results if not in root
        if args.plot_only or args.search or args.early_warning or args.stats:
            if not os.path.exists(results_file):
                alt_path = os.path.join("data", "results", results_file)
                if os.path.exists(alt_path):
                    results_file = alt_path

    if args.plot_only or args.search or args.early_warning or args.stats:
        print(f"Loading results from {results_file}...")
        if not os.path.exists(results_file):
            print(f"Error: Results file {results_file} not found. Run without --plot-only/--search first, or specify --results-file.")
            return
        with open(results_file, "r") as f:
            results_data = json.load(f)
            
        if args.search:
            search_discriminator(results_data)
        elif args.early_warning:
            analyze_early_indicators(results_data)
        elif args.stats:
            generate_statistical_report(results_data)
        else:
            generate_all_plots(results_data, args)
        return

    print(f"Using model: {args.model}")
    print(f"Device: {args.device}")
    
    model_kwargs = {
        "output_attentions": True, 
        "output_hidden_states": True
    }
    
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        
    config = GSPConfig(
        model_name=args.model, 
        device=args.device, 
        local_files_only=args.offline,
        model_kwargs=model_kwargs
    )

    results_data = {
        "valid": [],
        "invalid": []
    }

    try:
        with GSPDiagnosticsFramework(config) as framework:
            framework.instrumenter.load_model(args.model)
            
            for type_ in ["valid", "invalid"]:
                # Use data_dir argument
                pattern = os.path.join(args.data_dir, type_, f"*.{args.extension}")
                files = glob.glob(pattern)
                print(f"Processing {len(files)} {type_} proofs from {pattern}...")
                
                for file_path in files:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        
                        print(f"Analyzing {file_path}...")
                        analysis = framework.analyze_text(text, save_results=False)
                        
                        # Extract metrics from ALL layers
                        layer_metrics = []
                        if 'layer_diagnostics' in analysis and analysis['layer_diagnostics']:
                            for layer_idx, diag in enumerate(analysis['layer_diagnostics']):
                                metrics = {
                                    "layer": layer_idx,
                                    "fiedler_value": float(getattr(diag, "fiedler_value")) if getattr(diag, "fiedler_value") is not None else None,
                                    "energy": float(getattr(diag, "energy")) if getattr(diag, "energy") is not None else None,
                                    "smoothness": float(getattr(diag, "smoothness_index")) if getattr(diag, "smoothness_index") is not None else None,
                                    "entropy": float(getattr(diag, "spectral_entropy")) if getattr(diag, "spectral_entropy") is not None else None,
                                    "hfer": float(getattr(diag, "hfer")) if getattr(diag, "hfer") is not None else None
                                }
                                layer_metrics.append(metrics)
                            
                            # Store both summary (last layer) and full trajectory
                            last_layer = layer_metrics[-1]
                            results_data[type_].append({
                                "file": os.path.basename(file_path),
                                "summary": last_layer,
                                "trajectory": layer_metrics
                            })
                            print(f"  -> Last Layer Fiedler: {last_layer['fiedler_value']:.4f}")
                        else:
                            print("  -> No layer diagnostics found.")
                            
                    except Exception as e:
                        print(f"Failed to analyze {file_path}: {e}")
                        traceback.print_exc()

        # Save results
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
            
        print(f"Experiment complete. Results saved to {results_file}")
        if args.search:
            search_discriminator(results_data)
        elif args.early_warning:
            analyze_early_indicators(results_data)
        else:
            generate_all_plots(results_data, args)

    except Exception as e:
        print(f"Global error: {e}")
        traceback.print_exc()

def analyze_early_indicators(data):
    """
    Search for early signs (Layers 5-10) that correlate with the final outcome 
    or the L30_smoothness signal.
    """
    print("\n" + "="*50)
    print("      EARLY WARNING SYSTEM (Layers 5-10)")
    print("="*50)
    
    # 1. Flatten Data (Restricted to Layers 5-10)
    samples = []
    
    target_metric = "smoothness"
    # Detect depth from first item
    first_traj = data["valid"][0]["trajectory"] if data["valid"] else data["invalid"][0]["trajectory"]
    target_layer = len(first_traj) - 1 # Use last layer
    print(f"Detected model depth: {len(first_traj)} layers. Target Layer: {target_layer}")
    
    search_metrics = ["fiedler_value", "energy", "smoothness", "entropy", "hfer"]
    search_layers = range(5, 11) # 5 to 10 inclusive
    
    for label_type, label_val in [("valid", 1), ("invalid", 0)]:
        for item in data[label_type]:
            traj = item["trajectory"]
            if len(traj) <= target_layer: continue
            
            y_target = traj[target_layer][target_metric]
            
            feats = {}
            for l in search_layers:
                if l < len(traj):
                    for m in search_metrics:
                        val = traj[l].get(m)
                        if val is not None:
                            feats[f"L{l}_{m}"] = val
            
            samples.append({
                "label": label_val,
                "y_target": y_target,
                "features": feats,
                "file": item["file"]
            })

    total = len(samples)
    print(f"Analyzing {total} samples...")
    print(f"Target: L{target_layer}_{target_metric} / Validity")
    
    # 2. Single Feature Search
    feature_keys = samples[0]["features"].keys()
    best_rules = [] # (accuracy, description, rule_func, correlation)
    
    for feat in tqdm(feature_keys, desc="Scanning Early Features"):
        x_vals = [s["features"][feat] for s in samples]
        y_vals = [s["y_target"] for s in samples]
        
        # Pearson Correlation
        if len(x_vals) > 1:
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
        else:
            corr = 0
            
        # Get Candidate Thresholds
        values = sorted(x_vals)
        thresholds = []
        for i in range(len(values)-1):
            thresholds.append((values[i] + values[i+1]) / 2)
        if not thresholds: thresholds = [values[0]]
        
        best_acc_for_feat = 0
        best_rule_for_feat = None
        
        for t in thresholds:
            # Rule A: > T
            pred_gt = [1 if s['features'][feat] > t else 0 for s in samples]
            acc_gt = sum(1 for p, s in zip(pred_gt, samples) if p == s['label']) / total
            
            # Rule B: < T
            pred_lt = [1 if s['features'][feat] < t else 0 for s in samples]
            acc_lt = sum(1 for p, s in zip(pred_lt, samples) if p == s['label']) / total
            
            if acc_gt > best_acc_for_feat:
                best_acc_for_feat = acc_gt
                best_rule_for_feat = (acc_gt, f"{feat} > {t:.4f}", lambda s, f=feat, th=t: s['features'][f] > th, corr)
                
            if acc_lt > best_acc_for_feat:
                best_acc_for_feat = acc_lt
                best_rule_for_feat = (acc_lt, f"{feat} < {t:.4f}", lambda s, f=feat, th=t: s['features'][f] < th, corr)
                
        if best_rule_for_feat:
            best_rules.append(best_rule_for_feat)

    # 3. Report Top Singles
    best_rules.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- TOP SINGLE EARLY DISCRIMINATORS (L5-10) ---")
    for acc, desc, _, corr in best_rules[:10]:
        print(f"Acc: {acc*100:.1f}% | Corr: {corr:+.4f} | Rule: {desc}")
        
    # 4. Search for Combinations (AND logic)
    print("\nSearching best Early Pairs (AND logic)...")
    top_candidates = best_rules[:30] # Top 30 features
    best_pair_rule = (0, "None")
    
    for i in range(len(top_candidates)):
        rule1 = top_candidates[i]
        for j in range(i+1, len(top_candidates)):
            rule2 = top_candidates[j]
            
            # Skip if same feature (redundant)
            feat1 = rule1[1].split()[0]
            feat2 = rule2[1].split()[0]
            if feat1 == feat2: continue

            preds = []
            for s in samples:
                is_valid = rule1[2](s) and rule2[2](s)
                preds.append(1 if is_valid else 0)
            
            acc = sum(1 for p, s in zip(preds, samples) if p == s['label']) / total
            
            if acc > best_pair_rule[0]:
                best_pair_rule = (acc, f"({rule1[1]}) AND ({rule2[1]})")

    print("\n--- TOP COMBINED EARLY DISCRIMINATOR ---")
    print(f"Accuracy {best_pair_rule[0]*100:.1f}% : Valid if {best_pair_rule[1]}")
    
    # 5. Smoothness Specific Report
    print("\n--- SMOOTHNESS SPECIFIC REPORT (L5-10) ---")
    smoothness_rules = [r for r in best_rules if "smoothness" in r[1]]
    if smoothness_rules:
        for acc, desc, _, corr in smoothness_rules[:5]:
             print(f"Acc: {acc*100:.1f}% | Corr: {corr:+.4f} | Rule: {desc}")
    else:
        print("No smoothness rules found in top candidates.")

    if best_pair_rule[0] >= 0.9:
        print("\n[CONCLUSION] We can predict the outcome with high confidence using only Layers 5-10!")

def search_discriminator(data):
    """
    Brute-force grid search to find the best (Layer, Metric, Threshold) rules
    that separate Valid from Invalid.
    """
    print("\n" + "="*50)
    print("      SEARCHING FOR PERFECT DISCRIMINATOR")
    print("="*50)

    # 1. Flatten Data
    # Structure: [{'label': 1/0, 'features': {'L23_fiedler': 0.5, ...}}, ...]
    samples = []
    
    # metrics to scan
    metric_names = ["fiedler_value", "energy", "smoothness", "entropy", "hfer"]
    
    for label_type, label_val in [("valid", 1), ("invalid", 0)]:
        for item in data[label_type]:
            traj = item["trajectory"]
            feats = {}
            for layer_idx, metrics in enumerate(traj):
                for m in metric_names:
                    val = metrics.get(m)
                    if val is not None:
                        feats[f"L{layer_idx}_{m}"] = val
            samples.append({'label': label_val, 'features': feats, 'file': item['file']})

    valid_count = sum(1 for s in samples if s['label'] == 1)
    invalid_count = sum(1 for s in samples if s['label'] == 0)
    total = len(samples)
    print(f"Dataset: {valid_count} Valid, {invalid_count} Invalid (Total {total})")
    print("Scanning single-feature rules...")

    # 2. Collect all possible split points for each feature
    feature_keys = samples[0]['features'].keys()
    
    best_rules = [] # (accuracy, description, rule_func)
    
    for feat in tqdm(feature_keys, desc="Features"):
        # Get all values
        values = sorted([s['features'][feat] for s in samples if feat in s['features']])
        if not values: continue
        
        # Create candidate thresholds (midpoints)
        thresholds = []
        for i in range(len(values)-1):
            thresholds.append((values[i] + values[i+1]) / 2)
            
        # Also test min-epsilon and max+epsilon? No, midpoints are sufficient for classification.
        if not thresholds: thresholds = [values[0]] 

        best_acc_for_feat = 0
        best_rule_for_feat = None

        for t in thresholds:
            # Test Rule A: Val > T means Valid (1)
            pred_gt = [1 if s['features'][feat] > t else 0 for s in samples]
            acc_gt = sum(1 for p, s in zip(pred_gt, samples) if p == s['label']) / total
            
            # Test Rule B: Val < T means Valid (1)
            pred_lt = [1 if s['features'][feat] < t else 0 for s in samples]
            acc_lt = sum(1 for p, s in zip(pred_lt, samples) if p == s['label']) / total
            
            # Store best
            if acc_gt > best_acc_for_feat:
                best_acc_for_feat = acc_gt
                best_rule_for_feat = (acc_gt, f"{feat} > {t:.4f}", lambda s, f=feat, th=t: s['features'][f] > th)
                
            if acc_lt > best_acc_for_feat:
                best_acc_for_feat = acc_lt
                best_rule_for_feat = (acc_lt, f"{feat} < {t:.4f}", lambda s, f=feat, th=t: s['features'][f] < th)
        
        if best_rule_for_feat:
            best_rules.append(best_rule_for_feat)

    # 3. Sort and Print Top Single Rules
    best_rules.sort(key=lambda x: x[0], reverse=True)
    
    print("\n--- TOP SINGLE DISCRIMINATORS ---")
    for acc, desc, _ in best_rules[:10]:
        print(f"Accuracy {acc*100:.1f}% : Valid if {desc}")

    # Check for perfection
    perfect_singles = [r for r in best_rules if r[0] == 1.0]
    if perfect_singles:
        print("\n[SUCCESS] Found PERFECT Single-Feature Discriminators!")
        return

    # 4. Search for Combinations (AND logic)
    print("\nNo single feature is perfect. Searching best Pairs (AND logic)...")
    # Take top 50 features to combine
    top_candidates = best_rules[:50]
    best_pair_rule = (0, "None")
    
    for i in range(len(top_candidates)):
        rule1 = top_candidates[i]
        for j in range(i+1, len(top_candidates)):
            rule2 = top_candidates[j]
            
            # Valid if Rule1 AND Rule2
            # (Basically intersection of predicted valid sets)
            preds = []
            for s in samples:
                is_valid = rule1[2](s) and rule2[2](s)
                preds.append(1 if is_valid else 0)
            
            acc = sum(1 for p, s in zip(preds, samples) if p == s['label']) / total
            
            if acc > best_pair_rule[0]:
                best_pair_rule = (acc, f"({rule1[1]}) AND ({rule2[1]})")
                
            # Also Valid if Rule1 OR Rule2?
            # Usually strict logical proofs imply AND (must satisfy all properties)
            # But let's stick to AND for "Valid" definition (must be sharp AND ordered)
            
    print("\n--- TOP COMBINED DISCRIMINATOR ---")
    print(f"Accuracy {best_pair_rule[0]*100:.1f}% : Valid if {best_pair_rule[1]}")

def filter_data(data, args):
    """Filter proofs based on CLI arguments"""
    if not (args.geq or args.leq):
        return data, "All"

    filtered_data = {"valid": [], "invalid": []}
    conditions = []
    
    if args.geq:
        metric, val = args.geq[0], float(args.geq[1])
        conditions.append(f"{metric} >= {val}")
    if args.leq:
        metric, val = args.leq[0], float(args.leq[1])
        conditions.append(f"{metric} <= {val}")
        
    filter_str = " & ".join(conditions) + f" (Layer {args.layer})"
    print(f"Filtering data: {filter_str}")

    for type_ in ["valid", "invalid"]:
        for item in data[type_]:
            traj = item["trajectory"]
            
            # Handle negative indices or out of bounds
            idx = args.layer
            if idx < 0: idx = len(traj) + idx
            if idx < 0 or idx >= len(traj): continue # Skip if layer doesn't exist
            
            layer_data = traj[idx]
            keep = True
            
            if args.geq:
                m, v = args.geq[0], float(args.geq[1])
                if layer_data.get(m) is None or layer_data[m] < v:
                    keep = False
            
            if args.leq:
                m, v = args.leq[0], float(args.leq[1])
                if layer_data.get(m) is None or layer_data[m] > v:
                    keep = False
            
            if keep:
                filtered_data[type_].append(item)
                
    print(f"Filtered: Valid {len(data['valid'])}->{len(filtered_data['valid'])}, Invalid {len(data['invalid'])}->{len(filtered_data['invalid'])}")
    return filtered_data, filter_str

def get_plot_filename(base_name, args, filter_str):
    model_short = args.model.split("/")[-1]
    safe_filter = filter_str.replace(" ", "_").replace(">=", "GEQ").replace("<=", "LEQ").replace("&", "AND").replace("(", "").replace(")", "")
    if safe_filter == "All": safe_filter = "no_filter"
    if not os.path.exists("output"):
        os.makedirs("output")
    return os.path.join("output", f"{base_name}_{model_short}_L{args.layer}_{safe_filter}.png")

def generate_all_plots(data, args):
    """Central function to generate all required visualizations"""
    print("\n" + "="*50)
    print("      GENERATING VISUALIZATIONS")
    print("="*50)
    
    # 1. Summary Boxplots
    plot_summary(data, args)
    
    # 2. Layer Trajectories 
    plot_layers(data, args)
    
    # 3. Interaction Scatter (HFER vs Entropy)
    plot_interaction(data, args)
    
    # 4. Discrimination Analysis (Smoothness)
    analyze_discrimination(data, args)

def analyze_discrimination(data, args):
    layer_idx = args.disc_layer
    filtered_data, filter_label = filter_data(data, args)
    
    print("\n" + "-"*30)
    print(f" Discrimination Analysis (Smoothness)")
    print(f" Layer: {layer_idx}, Filter: {filter_label}")
    print("-"*30)
    
    def get_val(item, layer):
        traj = item["trajectory"]
        idx = layer if layer >= 0 else len(traj) + layer
        if idx < 0 or idx >= len(traj): return None
        return traj[idx]["smoothness"]

    valid_vals = [get_val(x, layer_idx) for x in filtered_data["valid"]]
    valid_vals = [v for v in valid_vals if v is not None]
    
    if len(valid_vals) < 2:
        print("Not enough valid data for discrimination analysis.")
        return

    mu_valid = np.mean(valid_vals)
    std_valid = np.std(valid_vals)
    print(f"Valid Proofs: µ={mu_valid:.4f}, σ={std_valid:.4f}")
    
    invalid_vals = [get_val(x, layer_idx) for x in filtered_data["invalid"]]
    invalid_vals = [v for v in invalid_vals if v is not None]
    
    if invalid_vals:
        z_scores = [(val - mu_valid) / std_valid for val in invalid_vals]
        print(f"Mean Invalid Z-Score: {np.mean(z_scores):+.2f}σ")
    
    plt.figure(figsize=(10, 6))
    if len(valid_vals) > 0:
        plt.hist(valid_vals, bins=max(5, len(valid_vals)//2), alpha=0.5, color='blue', label='Valid', density=True)
    if len(invalid_vals) > 0:
        plt.hist(invalid_vals, bins=max(5, len(invalid_vals)//2), alpha=0.5, color='red', label='Invalid', density=True)
    
    if len(valid_vals) > 1 and std_valid > 0:
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = (1/(std_valid * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_valid) / std_valid)**2)
        plt.plot(x, p, 'k--', linewidth=2, label='Valid Fit')
    
    plt.title(f"Smoothness Discrimination at Layer {layer_idx}\nModel: {args.model}\nFilter: {filter_label}")
    plt.xlabel("Smoothness Index")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fname = get_plot_filename("discrimination_plot", args, filter_label)
    plt.savefig(fname)
    print(f"Saved: {fname}")

def plot_summary(data, args):
    filtered_data, filter_label = filter_data(data, args)
    metrics = ["fiedler_value", "energy", "smoothness", "entropy"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    has_data = len(filtered_data["valid"]) + len(filtered_data["invalid"]) > 0
    if not has_data: return

    for i, metric in enumerate(metrics):
        valid_vals = [x["summary"][metric] for x in filtered_data["valid"] if x["summary"][metric] is not None]
        invalid_vals = [x["summary"][metric] for x in filtered_data["invalid"] if x["summary"][metric] is not None]
        ax = axes[i]
        if len(valid_vals) + len(invalid_vals) > 0:
            ax.boxplot([valid_vals, invalid_vals], tick_labels=["Valid", "Invalid"])
        ax.set_title(f"Last Layer {metric}")
        ax.grid(True, alpha=0.3)
        if len(valid_vals) > 0:
            ax.scatter(np.random.normal(1, 0.04, size=len(valid_vals)), valid_vals, alpha=0.5, color='blue')
        if len(invalid_vals) > 0:
            ax.scatter(np.random.normal(2, 0.04, size=len(invalid_vals)), invalid_vals, alpha=0.5, color='red')

    plt.suptitle(f"Model: {args.model} | Filter: {filter_label}")
    plt.tight_layout()
    fname = get_plot_filename("summary_plot", args, filter_label)
    plt.savefig(fname)
    print(f"Saved: {fname}")

def plot_layers(data, args):
    filtered_data, filter_label = filter_data(data, args)
    metrics_to_plot = ["fiedler_value", "energy", "smoothness", "entropy", "hfer"]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        for item in filtered_data["valid"]:
            traj = item["trajectory"]
            ax.plot([t["layer"] for t in traj], [t[metric] for t in traj], color='blue', alpha=0.3, linewidth=1)
        for item in filtered_data["invalid"]:
            traj = item["trajectory"]
            ax.plot([t["layer"] for t in traj], [t[metric] for t in traj], color='red', alpha=0.3, linewidth=1)
            
        if filtered_data["valid"]:
            v_mean = np.mean([[t[metric] for t in x["trajectory"]] for x in filtered_data["valid"]], axis=0)
            ax.plot(v_mean, color='blue', linewidth=2, label='Valid Mean')
        if filtered_data["invalid"]:
            i_mean = np.mean([[t[metric] for t in x["trajectory"]] for x in filtered_data["invalid"]], axis=0)
            ax.plot(i_mean, color='red', linewidth=2, label='Invalid Mean')

        # Paper Annotations (User Request)
        if metric == "fiedler_value" and len(filtered_data["valid"]) > 0:
            # Point to L0 separation or Valid High Fiedler
            # "Geometric Lookahead" at L0
            y_pos = v_mean[0]
            ax.annotate('Geometric Lookahead', xy=(0, y_pos), xytext=(2, y_pos+0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        fontsize=12, fontweight='bold')
                        
        if metric == "hfer" and len(filtered_data["invalid"]) > 0:
            # "Spectral Noise (Hallucination)" pointing to Invalid Spike
            # Find max invalid spike
            spike_layer = np.argmax(i_mean)
            y_pos = i_mean[spike_layer]
            ax.annotate('Spectral Noise\n(Hallucination)', xy=(spike_layer, y_pos), xytext=(spike_layer-5, y_pos+0.1),
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        fontsize=12, fontweight='bold', color='red')

        ax.set_title(f"{metric}")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    plt.suptitle(f"Model: {args.model} | Filter: {filter_label}")
    plt.tight_layout()
    # Save with higher DPI for paper
    fname = get_plot_filename("layer_plot", args, filter_label)
    plt.savefig(fname, dpi=300)
    print(f"Saved: {fname}")

def plot_interaction(data, args):
    target_layer = args.layer
    filtered_data, filter_label = filter_data(data, args)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def get_vals(key):
        h, e = [], []
        for item in filtered_data[key]:
             traj = item["trajectory"]
             if target_layer < len(traj):
                 h.append(traj[target_layer]["hfer"])
                 e.append(traj[target_layer]["entropy"])
        return h, e

    vh, ve = get_vals("valid")
    ih, ie = get_vals("invalid")
    
    if vh: ax.scatter(vh, ve, color='blue', label='Valid', s=100, alpha=0.7)
    if ih: ax.scatter(ih, ie, color='red', label='Invalid', s=100, alpha=0.7)
    
    ax.set_xlabel(f"HFER (Layer {target_layer})")
    ax.set_ylabel(f"Entropy (Layer {target_layer})")
    ax.set_title(f"Interaction (Layer {target_layer})\nFilter: {filter_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fname = get_plot_filename("interaction_plot", args, filter_label)
    plt.savefig(fname)
    print(f"Saved: {fname}")

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
