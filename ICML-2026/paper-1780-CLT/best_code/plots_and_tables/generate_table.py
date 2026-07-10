import os
import pandas as pd
import numpy as np
import math

# ==========================================
# Configuration
# ==========================================

SUPP_LIST = [4, 8, 16]  # Support set sizes to aggregate
MUT = 5
OUTPUT_FILE = f"steering_table_mut{MUT}.txt"
RESULTS_DIR_TEMPLATE = "results_probe_{}_mut{}"
CAA_RESULTS_DIR = ""

# Path to the source DMS data for counting mutants/bins
DMS_SOURCE_DIR = "../function_circuit/DMS/cv_folds_multiples_substitutions"

TARGET_FILE = "probe_results.csv"
TOP_K = 50 

# --- Subset Configuration ---
SUBSET_DMS = [
    "SPG1_STRSG_Olson_2014",
    "HIS7_YEAST_Pokusaeva_2019", 
    "GRB2_HUMAN_Faure_2021", 
    "GFP_AEQVI_Sarkisyan_2016", 
    "CAPSD_AAV2S_Sinai_2021", 
    'RASK_HUMAN_Weng_2022_abundance',
    'A4_HUMAN_Seuma_2022',
]

# --- Method Naming ---
METHOD_MAP = {
    "CLT_direct": "PCT",
    "CLT_sequential": "PCT (sequential)",
    "PLT": "PLT",
    "CLT_sequential_no_frozen": "PCT (full replacement)",
    "PLT_no_frozen": "PLT (full replacement)",
    "CAA": "CAA",
    "Random": "Random"
}

# --- Table Configurations ---
TABLE_CONFIGS = [
    # 1. PCT (Direct), PLT, CAA, Random
    {
        "title": "ProtoMech (Direct) vs PLT vs CAA",
        "methods": ["CLT_direct", "PLT", "CAA"],
        "random_source": "CLT_direct", 
        "include_random": True,
        "label_base": "tab:pct_direct"
    },
    # 2. PCT (Sequential), PLT, CAA, Random
    {
        "title": "ProtoMech (Sequential) vs PLT vs CAA",
        "methods": ["CLT_sequential", "PLT", "CAA"],
        "random_source": "CLT_sequential", 
        "include_random": True,
        "label_base": "tab:pct_seq"
    },
    # 3. PCT Full Rep, PLT Full Rep, CAA, Random
    {
        "title": "ProtoMech (Full) vs PLT vs CAA",
        "methods": ["CLT_sequential_no_frozen", "PLT", "CAA"],
        "random_source": "CLT_sequential_no_frozen", 
        "include_random": True,
        "label_base": "tab:pct_full"
    },
    # 4. PLT Variants (No Random)
    {
        "title": "PLT Variants Comparison",
        "methods": ["PLT", "PLT_no_frozen"],
        "random_source": None, 
        "include_random": False,
        "label_base": "tab:plt_variants"
    }
]

ALL_METHODS_TO_LOAD = list(set(
    [m for cfg in TABLE_CONFIGS for m in cfg["methods"]]
))

def get_results_dir(supp):
    """Helper to construct the results directory path for a specific support size."""
    return RESULTS_DIR_TEMPLATE.format(supp, MUT)

def load_single_file(file_path):
    """Helper to load a single csv file."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        if 'eval_score' in df.columns and 'probe_score' in df.columns:
            return df
        if 'random_eval_score' in df.columns:
            return df
    except:
        pass
    return None

def load_aggregated_data(dms_name, method):
    """
    Loops over all SUPP in SUPP_LIST, loads results, and concatenates them.
    Returns a single combined DataFrame.
    """
    dfs = []
    
    for supp in SUPP_LIST:
        base_dir = get_results_dir(supp)
        file_path = os.path.join(base_dir, dms_name, method, TARGET_FILE)
        df = load_single_file(file_path)
            
        if df is not None:
            df['source_supp'] = supp
            dfs.append(df)
    
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)

def load_caa_data(dms_name):
    """Loads CAA data."""
    caa_csv_path = os.path.join(
        CAA_RESULTS_DIR, 
        dms_name, 
        "rand_multiples", 
        f"{dms_name}_top50_probe.csv"
    )
    if not os.path.exists(caa_csv_path):
        return None
    try:
        df = pd.read_csv(caa_csv_path)
        if 'eval_score' in df.columns:
            return df
    except:
        pass
    return None

def calculate_stats(scores):
    """Calculates stats and returns raw numeric values + formatted strings."""
    if len(scores) == 0:
        return None

    # Ensure we work with a clean numpy array
    scores_arr = np.array(scores)
    
    mean_val = np.mean(scores_arr)
    std_val = np.std(scores_arr)
    max_val = np.max(scores_arr)

    # Sort scores to implement ranking
    # np.sort is ascending, so we take the tail
    scores_sorted = np.sort(scores_arr)

    # --- Top 10% Logic ---
    # Use ceil to ensure we don't floor 1.9 down to 1 if user has small N
    top10_n = max(1, int(np.ceil(len(scores_arr) * 0.1)))
    top10_subset = scores_sorted[-top10_n:]
    top10_mean = np.mean(top10_subset)
    top10_std = np.std(top10_subset)

    # --- Top 20% Logic ---
    top20_n = max(1, int(np.ceil(len(scores_arr) * 0.2)))
    top20_subset = scores_sorted[-top20_n:]
    top20_mean = np.mean(top20_subset)
    top20_std = np.std(top20_subset)

    return {
        'mean_raw': mean_val,
        'max_raw': max_val,
        'top10_raw': top10_mean,
        'top20_raw': top20_mean,
        
        'mean_str': f"{mean_val:.2f} $\\pm$ {std_val:.2f}",
        'max_str': f"{max_val:.2f}",
        'top10_str': f"{top10_mean:.2f} $\\pm$ {top10_std:.2f}",
        'top20_str': f"{top20_mean:.2f} $\\pm$ {top20_std:.2f}"
    }

def process_dms_data(dms_folders, method_list):
    """
    Reads data. Returns data_tree[dms][KEY] = stats_dict.
    KEY can be the method label (e.g. 'PCT') or a specific Random key.
    """
    data_tree = {dms: {} for dms in dms_folders}

    print(f"Processing {len(dms_folders)} DMS folders...")
    print(f"Aggregating support sets: {SUPP_LIST}")

    for dms in dms_folders:
        for method_key in method_list:
            
            # 1. CAA
            if method_key == "CAA":
                df = load_caa_data(dms)
                if df is not None:
                    # Sort by eval_score descending to ensure proper ordering before ranking slice
                    df_sorted = df.sort_values(by='eval_score', ascending=False)
                    scores = df_sorted['eval_score'].dropna().values
                    
                    stats = calculate_stats(scores)
                    if stats:
                        label = METHOD_MAP.get("CAA", "CAA")
                        data_tree[dms][label] = stats
                continue
            
            # 2. Regular Methods (Aggregate)
            df_agg = load_aggregated_data(dms, method_key)
            
            if df_agg is not None:
                # Deduplicate based on mutated sequence
                if 'mutated_sequence' in df_agg.columns:
                    df_agg = df_agg.drop_duplicates(subset='mutated_sequence')
                elif 'mutant' in df_agg.columns:
                    df_agg = df_agg.drop_duplicates(subset='mutant')

                # Sort by PROBE_SCORE and take Top K
                if 'probe_score' in df_agg.columns:
                    df_sorted = df_agg.sort_values(by="probe_score", ascending=False)
                    df_top = df_sorted.head(TOP_K)
                    
                    # --- A. Store Method Stats ---
                    scores = df_top['eval_score'].dropna().values
                    stats = calculate_stats(scores)
                    if stats:
                        label = METHOD_MAP.get(method_key, method_key)
                        data_tree[dms][label] = stats

                    # --- B. Store Random Baseline Stats ---
                    if 'random_eval_score' in df_top.columns:
                        rand_scores = df_top['random_eval_score'].dropna().values
                        rand_stats = calculate_stats(rand_scores)
                        if rand_stats:
                            rand_key = f"Random_FROM_{method_key}"
                            data_tree[dms][rand_key] = rand_stats

    return data_tree

def generate_latex_table(data_tree, dms_list, method_keys, random_source_key, include_random, table_label, caption_suffix):
    """
    Generates LaTeX table string. Bolds values based on rounding to 2 decimals.
    """
    # 1. Determine Rows
    row_labels = [METHOD_MAP.get(m, m) for m in method_keys]
    
    if include_random and random_source_key:
        final_row_order = row_labels + ["Random"]
    else:
        final_row_order = row_labels

    # 2. Map Rows to Data Keys
    data_lookup_map = {METHOD_MAP.get(m, m): METHOD_MAP.get(m, m) for m in method_keys}
    if include_random and random_source_key:
        data_lookup_map["Random"] = f"Random_FROM_{random_source_key}"

    # 3. Calculate Max per DMS using ROUNDED values
    max_vals = {dms: {'mean': -np.inf, 'max': -np.inf, 'top10': -np.inf, 'top20': -np.inf} 
                for dms in dms_list}

    for dms in dms_list:
        if dms not in data_tree: continue
        for row_name in final_row_order:
            storage_key = data_lookup_map[row_name]
            if storage_key in data_tree[dms]:
                stats = data_tree[dms][storage_key]
                # Round to 2 decimals before finding max
                max_vals[dms]['mean'] = max(max_vals[dms]['mean'], round(stats['mean_raw'], 2))
                max_vals[dms]['max'] = max(max_vals[dms]['max'], round(stats['max_raw'], 2))
                max_vals[dms]['top10'] = max(max_vals[dms]['top10'], round(stats['top10_raw'], 2))
                max_vals[dms]['top20'] = max(max_vals[dms]['top20'], round(stats['top20_raw'], 2))

    # 4. Construct LaTeX
    lines = []
    lines.append(r"% ---------------------------------------------------------")
    lines.append(f"% Table: {caption_suffix}")
    lines.append(r"% ---------------------------------------------------------")
    lines.append(r"\begin{table*}[t]")
    safe_name = RESULTS_DIR_TEMPLATE.replace("_", "\\_").format(str(SUPP_LIST), MUT)
    
    # Caption Logic
    caption_text = f"Probe Steering Results ({safe_name}) {caption_suffix}. Aggregated over support sizes {SUPP_LIST}. Top {TOP_K} unique sequences."
    if include_random:
        caption_text += f" Random baseline calculated from Top {TOP_K} of {METHOD_MAP.get(random_source_key, random_source_key)}."
    caption_text += " Best scores (rounded to 2 decimals) bolded."
    
    lines.append(f"\\caption{{{caption_text}}}")
    lines.append(r"\vspace{0.2cm}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append(r"\centering")
    lines.append(r"\resizebox{1 \textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{DMS}")
    lines.append(r"  & \textbf{Mean score $\uparrow$} & \textbf{Max score $\uparrow$}")
    lines.append(r"  & \textbf{Top 10\% score $\uparrow$} & \textbf{Top 20\% score $\uparrow$} \\")
    lines.append(r"\midrule")

    for i, row_name in enumerate(final_row_order):
        storage_key = data_lookup_map[row_name]
        
        has_data = any(storage_key in data_tree.get(d, {}) for d in dms_list)
        if not has_data:
            continue

        first_row = True
        valid_dms_list = [d for d in dms_list if d in data_tree and storage_key in data_tree[d]]

        for dms_name in valid_dms_list:
            stats = data_tree[dms_name][storage_key]
            
            def format_cell(key_raw, key_str):
                # Round current value to 2 decimals
                val_rounded = round(stats[key_raw], 2)
                
                # Check against the Max (which is also rounded)
                if max_vals[dms_name][key_raw.split('_')[0]] > -np.inf:
                    if val_rounded == max_vals[dms_name][key_raw.split('_')[0]]:
                        return f"\\textbf{{{stats[key_str]}}}"
                return stats[key_str]

            val_mean = format_cell('mean_raw', 'mean_str')
            val_max = format_cell('max_raw', 'max_str')
            val_t10 = format_cell('top10_raw', 'top10_str')
            val_t20 = format_cell('top20_raw', 'top20_str')

            dms_latex = dms_name.replace("_", "\\_")

            if first_row:
                row_str = f"\\textbf{{{row_name}}}"
                first_row = False
            else:
                row_str = ""
            
            row_str += f" & {dms_latex} & {val_mean} & {val_max} & {val_t10} & {val_t20} \\\\"
            lines.append(row_str)

        if i < len(final_row_order) - 1:
            lines.append(r"\cmidrule(lr){1-6}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    lines.append(r"")
    
    return "\n".join(lines)

def analyze_dms_metadata(dms_name):
    """
    Reads the source DMS CSV to:
    1. Count number of multiple mutants (entries with ':')
    2. Count number of sequences with DMS_score_bin == 1
    Returns (mut_count, bin_one_count)
    """
    file_path = os.path.join(DMS_SOURCE_DIR, f"{dms_name}.csv")
    
    if not os.path.exists(file_path):
        # Fallback if file not found
        return 0, 0
        
    try:
        df = pd.read_csv(file_path)
        
        # 1. Count multiple mutants
        if 'mutant' in df.columns:
            # Count how many strings contain ':'
            # Ensure column is string
            mut_col = df['mutant'].astype(str)
            mult_mut_count = mut_col.str.contains(':').sum()
        else:
            mult_mut_count = 0
            
        # 2. Count DMS_score_bin == 1
        if 'DMS_score_bin' in df.columns:
            bin_one_count = (df['DMS_score_bin'] == 1).sum()
        else:
            bin_one_count = 0
            
        return mult_mut_count, bin_one_count
        
    except Exception as e:
        print(f"Warning: Could not analyze metadata for {dms_name}: {e}")
        return 0, 0

def main():
    # 1. Establish Universe of DMS folders
    reference_dir = get_results_dir(SUPP_LIST[0])
    
    if not os.path.exists(reference_dir):
        print(f"Error: Reference directory '{reference_dir}' not found.")
        return

    # A. Get ALL folders on disk
    raw_dms_folders = sorted([f for f in os.listdir(reference_dir) 
                              if os.path.isdir(os.path.join(reference_dir, f)) 
                              and not f.startswith('.')])
    
    # --- NEW: Analyze and Sort DMS folders ---
    print("\nAnalyzing DMS source files for sorting and stats...")
    dms_metadata = []
    
    for dms in raw_dms_folders:
        mult_count, bin_one_count = analyze_dms_metadata(dms)
        dms_metadata.append({
            'name': dms,
            'mult_count': mult_count,
            'bin_one_count': bin_one_count
        })
        
    # Sort metadata by mult_count (Ascending)
    dms_metadata.sort(key=lambda x: x['mult_count'], reverse=True)
    
    # Re-create the sorted lists of folders
    all_dms_folders = [item['name'] for item in dms_metadata]
    
    # Print the stats as requested
    print("\n" + "="*60)
    print(f"{'DMS Name':<40} | {'Mult Mutants':<12} | {'Bin=1 Count':<12}")
    print("-" * 60)
    for item in dms_metadata:
        print(f"{item['name']:<40} | {item['mult_count']:<12} | {item['bin_one_count']:<12}")
    print("="*60 + "\n")
    # ----------------------------------------
    
    # B. Get SUBSET folders (intersection, maintaining sorted order)
    subset_dms_folders = [d for d in all_dms_folders if d in SUBSET_DMS]
    
    # 2. Process Data (Load EVERYTHING so we can slice later)
    full_data_tree = process_dms_data(all_dms_folders, ALL_METHODS_TO_LOAD)

    with open(OUTPUT_FILE, "w") as f:
        # 3. Generate Tables for each Config
        for config in TABLE_CONFIGS:
            
            # --- Version 1: ALL DMS ---
            print(f"\nGenerating FULL Table: {config['title']}...")
            latex_full = generate_latex_table(
                data_tree=full_data_tree,
                dms_list=all_dms_folders,
                method_keys=config["methods"],
                random_source_key=config["random_source"],
                include_random=config["include_random"],
                table_label=f"{config['label_base']}_full",
                caption_suffix=f"- {config['title']} (Full Benchmark)"
            )
            f.write(latex_full)
            f.write("\n\n" + "% " + "-"*20 + "\n\n")

            # --- Version 2: SUBSET DMS ---
            if subset_dms_folders:
                print(f"Generating SUBSET Table: {config['title']}...")
                latex_subset = generate_latex_table(
                    data_tree=full_data_tree,
                    dms_list=subset_dms_folders,
                    method_keys=config["methods"],
                    random_source_key=config["random_source"],
                    include_random=config["include_random"],
                    table_label=f"{config['label_base']}_subset",
                    caption_suffix=f"- {config['title']} (Selected DMS)"
                )
                f.write(latex_subset)
            
            f.write("\n\n" + "% " + "="*50 + "\n\n")
    
    print(f"\nDone! Tables saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()