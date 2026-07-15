#!/usr/bin/env python3
"""
Main execution script for the Disagreement-Aware Evaluation Pipeline.

This script runs the complete evaluation pipeline:
1. Load data from CSV files
2. Compute scores using all three methods
3. Generate visualizations and reports
4. Save results

Usage:
    python run_evaluation.py --data-dir data/processed --output-dir results

Options:
    --data-dir      Directory containing annotation CSV files (default: data/processed)
    --output-dir    Directory to save results (default: results)
    --n-classes     Number of label classes (default: auto-detect)
    --bootstrap     Number of bootstrap iterations (default: 500, 0 to disable)
    --stability     Number of stability analysis subsets (default: 50, 0 to disable)
    --verbose       Print detailed progress (default: True)
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.data_loader import load_all_data, load_single_csv, print_data_summary, get_data_summary
from src.scoring import (
    compute_all_scores,
    compute_all_scores_with_bootstrap,
    create_comparison_table,
    compute_ranking_stability,
    identify_score_changes,
    print_results_summary
)
from src.visualization import create_all_plots
from src.majority_vote import get_class_values, get_agreement_statistics


def save_results(results, comparison_df, stability_df, output_dir: Path, data_summary: dict):
    """Save all results to the output directory."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving results...")
    
    # 1. Save comparison table
    comparison_df.to_csv(output_dir / 'agent_scores_comparison.csv', index=False)
    print(f"  Saved: agent_scores_comparison.csv")
    
    # 2. Save individual method scores
    results.majority_vote.to_csv(output_dir / 'scores_majority_vote.csv', index=False)
    results.dawid_skene_hard.to_csv(output_dir / 'scores_dawid_skene.csv', index=False)
    results.posterior_expected_credit.to_csv(output_dir / 'scores_pec.csv', index=False)
    print(f"  Saved: scores_majority_vote.csv, scores_dawid_skene.csv, scores_pec.csv")
    
    # 3. Save item-level results
    if results.item_scores_mv is not None:
        results.item_scores_mv.to_csv(output_dir / 'item_scores_majority_vote.csv', index=False)
    if results.item_scores_pec is not None:
        results.item_scores_pec.to_csv(output_dir / 'item_scores_pec.csv', index=False)
    if results.item_ambiguity is not None:
        results.item_ambiguity.to_csv(output_dir / 'item_ambiguity.csv', index=False)
    print(f"  Saved: item-level score files")
    
    # 4. Save annotator quality
    if results.annotator_quality is not None:
        results.annotator_quality.to_csv(output_dir / 'annotator_quality.csv', index=False)
        print(f"  Saved: annotator_quality.csv")
    
    # 5. Save stability results
    if stability_df is not None:
        stability_df.to_csv(output_dir / 'ranking_stability.csv', index=False)
        print(f"  Saved: ranking_stability.csv")
    
    # 6. Save bootstrap results
    if results.mv_bootstrap is not None:
        results.mv_bootstrap.to_csv(output_dir / 'bootstrap_majority_vote.csv', index=False)
        results.ds_bootstrap.to_csv(output_dir / 'bootstrap_dawid_skene.csv', index=False)
        results.pec_bootstrap.to_csv(output_dir / 'bootstrap_pec.csv', index=False)
        print(f"  Saved: bootstrap confidence interval files")
    
    # 7. Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_summary': data_summary,
        'methods': ['majority_vote', 'dawid_skene_hard', 'posterior_expected_credit']
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: metadata.json")


def generate_report(results, comparison_df, stability_df, data_summary, agreement_stats, output_dir: Path):
    """Generate a markdown report summarizing the results."""
    
    report_lines = []
    report_lines.append("# Disagreement-Aware Agent Evaluation Report")
    report_lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Data summary
    report_lines.append("## Data Summary\n")
    report_lines.append(f"- **Total annotations**: {data_summary['n_annotations']:,}")
    report_lines.append(f"- **Unique items**: {data_summary['n_items']:,}")
    report_lines.append(f"- **Unique agents**: {data_summary['n_agents']:,}")
    report_lines.append(f"- **Unique annotators**: {data_summary['n_annotators']:,}")
    report_lines.append(f"- **Average annotations per item**: {data_summary['annotations_per_item']['mean']:.2f}")
    report_lines.append(f"- **Unanimous agreement rate**: {agreement_stats['unanimous_proportion']:.1%}")
    report_lines.append(f"- **Average pairwise agreement**: {agreement_stats['average_pairwise_agreement']:.1%}")
    report_lines.append("")
    
    # Label distribution
    report_lines.append("### Label Distribution\n")
    for label, count in sorted(data_summary['label_distribution'].items()):
        pct = 100 * count / data_summary['n_annotations']
        report_lines.append(f"- Label {label}: {count:,} ({pct:.1f}%)")
    report_lines.append("")
    
    # Methods overview
    report_lines.append("## Scoring Methods\n")
    report_lines.append("Three scoring methods were evaluated:\n")
    report_lines.append("1. **Majority Vote (MV)**: Simple baseline using most common label per item")
    report_lines.append("2. **Dawid-Skene Hard (DS)**: EM algorithm initialized with majority vote labels")
    report_lines.append("3. **Posterior Expected Credit (PEC)**: Probabilistic scoring with learned annotator models\n")
    
    # Agent scores comparison
    report_lines.append("## Agent Scores Comparison\n")
    report_lines.append("| Agent | MV Score | DS Score | PEC Score | Δ(PEC-MV) | Items |")
    report_lines.append("|-------|----------|----------|-----------|-----------|-------|")
    
    for _, row in comparison_df.iterrows():
        delta = row['diff_pec_mv']
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        report_lines.append(
            f"| {row['agent_id']} | {row['score_mv']:.3f} | {row['score_ds']:.3f} | "
            f"{row['score_pec']:.3f} | {delta_str} | {row['n_items']} |"
        )
    report_lines.append("")
    
    # Ranking comparison
    report_lines.append("## Agent Rankings\n")
    report_lines.append("| Agent | MV Rank | DS Rank | PEC Rank | Rank Change (MV→PEC) |")
    report_lines.append("|-------|---------|---------|----------|----------------------|")
    
    for _, row in comparison_df.sort_values('rank_pec').iterrows():
        rank_change = int(row['rank_mv'] - row['rank_pec'])
        change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)
        if rank_change == 0:
            change_str = "—"
        report_lines.append(
            f"| {row['agent_id']} | {int(row['rank_mv'])} | {int(row['rank_ds'])} | "
            f"{int(row['rank_pec'])} | {change_str} |"
        )
    report_lines.append("")
    
    # Stability analysis
    if stability_df is not None:
        report_lines.append("## Ranking Stability Analysis\n")
        report_lines.append("Lower values indicate more stable rankings across annotator subsets.\n")
        report_lines.append("| Method | Mean Rank Std Dev | Mean Rank Range |")
        report_lines.append("|--------|-------------------|-----------------|")
        for _, row in stability_df.iterrows():
            report_lines.append(
                f"| {row['method']} | {row['mean_rank_std']:.3f} | {row['mean_rank_range']:.3f} |"
            )
        report_lines.append("")
    
    # Annotator quality
    if results.annotator_quality is not None:
        report_lines.append("## Annotator Quality Analysis\n")
        report_lines.append("Top annotators by learned accuracy:\n")
        report_lines.append("| Annotator | Accuracy | Leniency | Strictness |")
        report_lines.append("|-----------|----------|----------|------------|")
        for _, row in results.annotator_quality.head(10).iterrows():
            report_lines.append(
                f"| {row['annotator_id']} | {row['accuracy']:.3f} | "
                f"{row['leniency']:.3f} | {row['strictness']:.3f} |"
            )
        report_lines.append("")
    
    # Most ambiguous items
    if results.item_ambiguity is not None:
        report_lines.append("## Most Ambiguous Items\n")
        report_lines.append("Items with highest uncertainty in true label:\n")
        report_lines.append("| Item ID | Ambiguity | Confidence | Predicted Class |")
        report_lines.append("|---------|-----------|------------|-----------------|")
        for _, row in results.item_ambiguity.head(10).iterrows():
            report_lines.append(
                f"| {row['item_id']} | {row['ambiguity']:.3f} | "
                f"{row['confidence']:.3f} | {int(row['predicted_class'])} |"
            )
        report_lines.append("")
    
    # Key findings
    report_lines.append("## Key Findings\n")
    
    # Score correlation
    mv_scores = comparison_df['score_mv']
    pec_scores = comparison_df['score_pec']
    correlation = mv_scores.corr(pec_scores)
    report_lines.append(f"- **Score correlation (MV vs PEC)**: {correlation:.3f}")
    
    # Max score difference
    max_diff = comparison_df['diff_pec_mv'].abs().max()
    max_diff_agent = comparison_df.loc[comparison_df['diff_pec_mv'].abs().idxmax(), 'agent_id']
    report_lines.append(f"- **Maximum score change**: {max_diff:.3f} (Agent: {max_diff_agent})")
    
    # Rank changes
    rank_changes = abs(comparison_df['rank_mv'] - comparison_df['rank_pec'])
    agents_changed = (rank_changes > 0).sum()
    report_lines.append(f"- **Agents with rank changes**: {agents_changed} / {len(comparison_df)}")
    
    if results.item_ambiguity is not None:
        avg_ambiguity = results.item_ambiguity['ambiguity'].mean()
        high_ambiguity = (results.item_ambiguity['ambiguity'] > 0.3).sum()
        report_lines.append(f"- **Average item ambiguity**: {avg_ambiguity:.3f}")
        report_lines.append(f"- **Highly ambiguous items (>0.3)**: {high_ambiguity}")
    
    report_lines.append("")
    
    # Conclusion
    report_lines.append("## Conclusion\n")
    report_lines.append("The disagreement-aware methods (Dawid-Skene and Posterior Expected Credit) ")
    report_lines.append("provide refined agent scores that account for annotator reliability and item ambiguity. ")
    
    if stability_df is not None:
        pec_stability = stability_df[stability_df['method'] == 'Posterior Expected Credit']['mean_rank_std'].values[0]
        mv_stability = stability_df[stability_df['method'] == 'Majority Vote']['mean_rank_std'].values[0]
        if pec_stability < mv_stability:
            report_lines.append("The PEC method shows improved ranking stability compared to majority vote.")
        else:
            report_lines.append("The ranking stability analysis shows comparable results across methods.")
    
    report_lines.append("")
    
    # Write report
    report_path = output_dir / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Saved: evaluation_report.md")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Run disagreement-aware agent evaluation pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='data/processed',
        help='Directory containing annotation CSV files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--n-classes', '-c',
        type=int,
        default=None,
        help='Number of label classes (auto-detect if not specified)'
    )
    parser.add_argument(
        '--bootstrap', '-b',
        type=int,
        default=10,
        help='Number of bootstrap iterations (0 to disable)'
    )
    parser.add_argument(
        '--stability', '-s',
        type=int,
        default=50,
        help='Number of stability analysis subsets (0 to disable)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    parser.add_argument(
        '--combine', 
        action='store_true',
        help='Combine all CSV files into one dataset (default: process each separately)'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    print("=" * 70)
    print("DISAGREEMENT-AWARE AGENT EVALUATION PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Bootstrap iterations: {args.bootstrap}")
    print(f"  Stability subsets: {args.stability}")
    print(f"  Random seed: {args.seed}")
    print(f"  Mode: {'Combined' if args.combine else 'Separate per file'}")
    print()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Find all CSV files
    data_dir = Path(args.data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f.name}")
    print()
    
    if args.combine:
        # Original behavior: combine all files
        output_dir = Path(args.output_dir)
        process_single_dataset(
            data_path=args.data_dir,
            output_dir=output_dir,
            n_classes=args.n_classes,
            bootstrap=args.bootstrap,
            stability=args.stability,
            seed=args.seed,
            verbose=verbose,
            load_from_dir=True
        )
    else:
        # New behavior: process each file separately
        for csv_file in csv_files:
            dataset_name = csv_file.stem  # filename without extension
            output_dir = Path(args.output_dir) / dataset_name
            
            print("=" * 70)
            print(f"PROCESSING: {csv_file.name}")
            print("=" * 70)
            
            process_single_dataset(
                data_path=csv_file,
                output_dir=output_dir,
                n_classes=args.n_classes,
                bootstrap=args.bootstrap,
                stability=args.stability,
                seed=args.seed,
                verbose=verbose,
                load_from_dir=False
            )
            print()
    
    print("=" * 70)
    print("ALL PROCESSING COMPLETE")
    print("=" * 70)


def process_single_dataset(
    data_path,
    output_dir: Path,
    n_classes,
    bootstrap,
    stability,
    seed,
    verbose,
    load_from_dir=False
):
    """Process a single dataset (either one CSV or a directory of CSVs)."""
    
    from src.data_loader import load_all_data, load_single_csv, print_data_summary, get_data_summary
    
    # Create output directories
    plots_dir = output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (automatically converts to response-level format)
    print("\n--- Loading Data ---")
    if load_from_dir:
        df = load_all_data(data_path)
    else:
        df = load_single_csv(data_path)
        print(f"Loaded {len(df)} annotations from {Path(data_path).name}")
    
    print(f"Created {df['item_id'].nunique()} unique responses from {df['original_item_id'].nunique()} items × {df['agent_id'].nunique()} agents")
    
    print_data_summary(df)
    data_summary = get_data_summary(df)
    agreement_stats = get_agreement_statistics(df)
    
    # Determine number of classes
    detected_classes = df['label'].nunique()
    if n_classes is None:
        n_classes_to_use = detected_classes
    else:
        n_classes_to_use = n_classes
    print(f"\nUsing {n_classes_to_use} classes")
    
    # Compute scores
    print("\n--- Computing Scores ---")
    if bootstrap > 0:
        results = compute_all_scores_with_bootstrap(
            df, 
            n_classes=n_classes_to_use,
            n_bootstrap=bootstrap,
            random_state=seed,
            verbose=verbose
        )
    else:
        results = compute_all_scores(df, n_classes=n_classes_to_use, verbose=verbose)
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Stability analysis
    stability_df = None
    if stability > 0:
        print("\n--- Ranking Stability Analysis ---")
        stability_df = compute_ranking_stability(
            df,
            n_subsets=stability,
            n_classes=n_classes_to_use,
            random_state=seed,
            verbose=verbose
        )
        print("\nStability Results:")
        print(stability_df.to_string(index=False))
    
    # Print summary
    print("\n--- Results Summary ---")
    print_results_summary(results)
    
    # Save results
    print("\n--- Saving Results ---")
    save_results(results, comparison_df, stability_df, output_dir, data_summary)
    
    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    create_all_plots(results, comparison_df, stability_df, output_dir=str(plots_dir))
    
    # Generate report
    print("\n--- Generating Report ---")
    generate_report(
        results, comparison_df, stability_df, 
        data_summary, agreement_stats, output_dir
    )
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
