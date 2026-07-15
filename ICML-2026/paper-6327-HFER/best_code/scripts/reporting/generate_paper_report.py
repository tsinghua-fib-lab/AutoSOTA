import sys
import os
# Add root to path to allow imports from scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import pandas as pd
from scripts.analysis.recalc_full_stats import analyze_model_corrected
from scripts.analysis.find_complex_discriminator import load_data as load_sieve_data, train_sieve
from scripts.analysis.find_perfect_combo import search_perfect_combo

def generate_report():
    models = [
        ("Llama-3.2-1B", "data/results/experiment_results_Llama-3.2-1B-Instruct.json", "data/reclaimed/1B_list_b_confident_invalid.json"),
        ("Llama-3.2-3B", "data/results/experiment_results_Llama-3.2-3B-Instruct.json", "data/reclaimed/3B_list_b_confident_invalid.json"),
        ("Meta-Llama-3.1-8B", "data/results/experiment_results_Meta-Llama-3.1-8B-Instruct.json", "data/reclaimed/8B_list_b_confident_invalid.json"),
        ("Qwen2.5-7B", "data/results/experiment_results_Qwen2.5-7B-Instruct.json", "data/reclaimed/Qwen7B_list_b_confident_invalid.json"),
        ("Qwen2.5-0.5B", "data/results/experiment_results_Qwen2.5-0.5B-Instruct.json", "data/reclaimed/Qwen0.5B_list_b_confident_invalid.json"),
        ("Phi-3.5-mini", "data/results/experiment_results_Phi-3.5-mini-instruct.json", "data/reclaimed/Phi3.5_list_b_confident_invalid.json"),
        ("Mistral-7B", "data/results/experiment_results_Mistral-7B-v0.1.json", "data/reclaimed/Mistral7B_list_b_confident_invalid.json")
    ]

    summary_data = []

    print("#" * 80)
    print(" GENERATING FINAL PAPER STATISTICS (PLATONIC VALIDITY)")
    print("#" * 80)

    for name, res_file, list_b_file in models:
        print(f"\nProcessing {name}...")
        
        # 1. Full Stats (P-Value, Cohen's d, Single Metric Acc)
        try:
            full_stats = analyze_model_corrected(name, res_file, list_b_file)
            best_single = full_stats[0]
        except Exception as e:
            print(f"Error in full stats: {e}")
            continue
        
        # 2. Complex Discriminator (Sieve)
        try:
            sieve_samples = load_sieve_data(res_file, list_b_file)
            sieve_res = train_sieve(sieve_samples, max_depth=5)
        except Exception as e:
             print(f"Error in sieve: {e}")
             sieve_res = {"acc": 0}
        
        # 3. Perfect Combo (2-Feature)
        try:
            combo_res = search_perfect_combo(res_file, list_b_file)
        except Exception as e:
             print(f"Error in combo: {e}")
             combo_res = {"best_combo": (0, "Error")}
        
        summary_data.append({
            "Model": name,
            "Best Single Metric": f"{best_single['metric']} @ L{best_single['layer']}",
            "Single Acc": best_single['acc'],
            "P-Value": best_single['p'],
            "Cohen's d": best_single['d'],
            "Sieve Acc (Depth 5)": sieve_res['acc'],
            "2-Feature Acc": combo_res['best_combo'][0]
        })

    print("\n\n" + "#" * 80)
    print(" FINAL RESULTS TABLE")
    print("#" * 80)
    
    df = pd.DataFrame(summary_data)
    
    # Formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    print(df.to_markdown(index=False))
    
    # Also print LaTeX friendly version
    print("\nLaTeX Table Body:")
    for index, row in df.iterrows():
        p_val_str = f"{row['P-Value']:.2e}"
        model = row['Model']
        single_metric = row['Best Single Metric']
        single_acc = row['Single Acc'] * 100
        cohen_d = row["Cohen's d"]
        feature_2_acc = row['2-Feature Acc'] * 100
        
        # Using variables in f-string to allow clean escaping later if needed, 
        # but here we just print the string literal for backslash-percent
        print(f"{model} & {single_metric} & {single_acc:.1f}\\% & {p_val_str} & {cohen_d:.2f} & {feature_2_acc:.1f}\\% \\\\")

if __name__ == "__main__":
    generate_report()
