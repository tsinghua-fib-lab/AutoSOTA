#!/usr/bin/env python3
"""
Analyze sample-efficiency results for SST-2 experiments.
"""

import json
import os
import glob
import argparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_results(result_dir: str) -> Dict:
    """Load experiment results from a result directory."""
    results = {
        "rope": {},
        "hipe": {}
    }
    
    for model_type in ["base", "hipe"]:
        model_key = "rope" if model_type == "base" else "hipe"
        pattern = os.path.join(result_dir, model_type, "*", "*", "*", "final_results.json")
        
        for result_file in glob.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Expected format: .../base/512/lora8/shot100/seed_6198/final_results.json
                parts = result_file.split(os.sep)
                
                seq_len = parts[-6]  # 512 or 2048
                shot_str = parts[-3]  # full or shotXXX
                
                if shot_str == "full":
                    shot = -1
                else:
                    shot = int(shot_str.replace("shot", ""))
                
                key = f"{seq_len}_{shot}"
                
                if key not in results[model_key]:
                    results[model_key][key] = []
                
                results[model_key][key].append({
                    "accuracy": data.get("best_accuracy", 0),
                    "f1": data.get("best_f1", 0),
                    "sample_efficiency": data.get("sample_efficiency", {}),
                    "total_samples": data.get("total_samples_processed", 0),
                })
                
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    return results


def analyze_threshold_reached(results: Dict, threshold: float = 0.80) -> pd.DataFrame:
    """Analyze the steps and samples required to reach a target threshold."""
    import pandas as pd
    
    rows = []
    
    for model_type in ["rope", "hipe"]:
        for key, exp_list in results[model_type].items():
            parts = key.split("_")
            seq_len = parts[0]
            shot = int(parts[1])
            
            for exp in exp_list:
                se_data = exp.get("sample_efficiency", {})
                thresh_str = f"{threshold:.2f}"
                
                if thresh_str in se_data.get("results", {}):
                    result = se_data["results"][thresh_str]
                    if result.get("step") is not None:
                        rows.append({
                            "model": model_type.upper(),
                            "seq_len": seq_len,
                            "shot": shot,
                            "threshold": threshold,
                            "step": result["step"],
                            "samples": result["samples"],
                            "epoch": result["epoch"],
                            "accuracy": result["accuracy"],
                        })
    
    return pd.DataFrame(rows)


def plot_sample_efficiency(df: pd.DataFrame, output_dir: str = "."):
    """Plot sample-efficiency comparisons."""
    if df.empty:
        print("No data to plot")
        return
    
    shots = sorted(df["shot"].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for model in ["ROPE", "HIPE"]:
        model_data = df[df["model"] == model]
        if not model_data.empty:
            ax1.plot(model_data["shot"], model_data["step"], 'o-', label=model, linewidth=2, markersize=8)
    
    ax1.set_xlabel("Few-shot Sample Count", fontsize=12)
    ax1.set_ylabel("Steps to Reach Target Accuracy", fontsize=12)
    ax1.set_title(f"Sample Efficiency: Steps vs Data Size", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    ax2 = axes[1]
    for model in ["ROPE", "HIPE"]:
        model_data = df[df["model"] == model]
        if not model_data.empty:
            ax2.plot(model_data["shot"], model_data["samples"], 'o-', label=model, linewidth=2, markersize=8)
    
    ax2.set_xlabel("Few-shot Sample Count", fontsize=12)
    ax2.set_ylabel("Samples Processed to Reach Target", fontsize=12)
    ax2.set_title(f"Sample Efficiency: Samples vs Data Size", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "sample_efficiency_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def generate_report(results: Dict, output_dir: str = "."):
    """Generate a text report summarizing sample-efficiency results."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SST-2 Fine-tuning Sample Efficiency Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for shot in sorted(set([int(k.split("_")[1]) for k in list(results["rope"].keys()) + list(results["hipe"].keys())])):
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"Shot Setting: {shot if shot > 0 else 'full'}")
        report_lines.append(f"{'='*80}")
        
        for seq_len in ["512", "2048"]:
            key = f"{seq_len}_{shot}"
            report_lines.append(f"\n--- Sequence Length: {seq_len} ---")
            
            for model_type in ["rope", "hipe"]:
                if key in results[model_type]:
                    exp = results[model_type][key][0]
                    
                    report_lines.append(f"\n{model_type.upper()}:")
                    report_lines.append(f"  Final Accuracy: {exp['accuracy']:.4f}")
                    report_lines.append(f"  Total Samples Processed: {exp['total_samples']:,}")
                    
                    se_data = exp.get("sample_efficiency", {})
                    if se_data and "results" in se_data:
                        report_lines.append("  Sample Efficiency:")
                        for thresh_str, result in sorted(se_data["results"].items()):
                            if result.get("step") is not None:
                                report_lines.append(
                                    f"    Acc {float(thresh_str)*100:.0f}%: "
                                    f"Step {result['step']} | "
                                    f"Samples {result['samples']:,} | "
                                    f"Epoch {result['epoch']:.2f}"
                                )
                            else:
                                report_lines.append(f"    Acc {float(thresh_str)*100:.0f}%: Not reached")
    
    report_lines.append("\n" + "=" * 80)
    
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "sample_efficiency_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Sample Efficiency for SST-2 experiments")
    parser.add_argument("--result_dir", type=str, 
                       default="/data/qijunrong/03-proj/PE/checkpoints_exp3/sst2/300m",
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Directory to save output plots and reports")
    parser.add_argument("--threshold", type=float, default=0.80,
                       help="Accuracy threshold to analyze (default: 0.80)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results from {args.result_dir}...")
    results = load_results(args.result_dir)
    
    generate_report(results, args.output_dir)
    
    try:
        import pandas as pd
        df = analyze_threshold_reached(results, args.threshold)
        if not df.empty:
            print(f"\n{'='*80}")
            print(f"Analysis for {args.threshold*100:.0f}% Accuracy Threshold")
            print(f"{'='*80}")
            print(df.to_string(index=False))
            
            plot_sample_efficiency(df, args.output_dir)
    except ImportError:
        print("\nNote: pandas/matplotlib not available, skipping DataFrame analysis and plotting")
        print("Install with: pip install pandas matplotlib")


if __name__ == "__main__":
    main()
