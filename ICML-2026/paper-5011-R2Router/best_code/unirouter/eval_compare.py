"""
Comparative Evaluation: Original UniRouter vs Uni-R2

Demonstrates:
1. Original UniRouter: Routes to best LLM (unlimited tokens)
2. Uni-R2: Routes to best (LLM, token_budget) pair
3. Dynamic model addition: Add new models without retraining
4. Performance comparison across methods
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_lambda_distribution(lambda_dist_str):
    """
    Parse lambda distribution string into numpy array.

    Format: "min,max,num;min,max,num;..."
    Example: "0,0.3,1000;0.3,0.5,500;0.5,1.0,200"

    Args:
        lambda_dist_str: String specifying lambda distribution segments

    Returns:
        numpy array of lambda values
    """
    segments = []
    for segment in lambda_dist_str.split(';'):
        parts = segment.split(',')
        if len(parts) != 3:
            raise ValueError(f"Invalid lambda distribution segment: {segment}. Expected format: min,max,num")
        min_val, max_val, num_points = float(parts[0]), float(parts[1]), int(parts[2])
        segments.append(np.linspace(min_val, max_val, num_points))

    # Concatenate all segments and remove duplicates
    all_lambdas = np.concatenate(segments)
    return np.unique(all_lambdas)

from unirouter.unirouter_original import OriginalUniRouter
from unirouter.uni_r2 import ValidationSetBuilder, ModelFeatureMatrix, UniR2Router
from main.shared.dataset_manager import DatasetManager


def parse_args():
    parser = argparse.ArgumentParser(description='Compare Original UniRouter vs Uni-R2')

    # Model specifications
    parser.add_argument('--initial-model', action='append', nargs=3,
                       metavar=('NAME', 'SIZE', 'CSV'),
                       help='Initial model (can specify multiple times)')
    parser.add_argument('--new-model', action='append', nargs=3,
                       metavar=('NAME', 'SIZE', 'CSV'),
                       help='New model to add dynamically (can specify multiple times)')

    # Validation set configuration
    parser.add_argument('--val-size', type=int, default=500,
                       help='Validation set size (default: 500)')
    parser.add_argument('--n-clusters', type=int, default=100,
                       help='Number of clusters (default: 100)')
    parser.add_argument('--use-clustering', action='store_true', default=True,
                       help='Use clustering (default: True)')

    # Token budgets
    parser.add_argument('--token-budgets', type=str,
                       default='10,20,30,40,50,80,100,150,200,300,500,800,1200,2000,4000,9999',
                       help='Comma-separated token budgets (9999=unlimited, must match dataset!)')

    # Lambda distribution
    parser.add_argument('--lambda-dist', type=str, default='0,1,100',
                       help='Lambda distribution: "min,max,num;min,max,num;..." for multi-segment sampling (default: 0,1,100)')

    # QNC target rate
    parser.add_argument('--target-accuracy-rate', type=float, default=1.0,
                       help='Target accuracy rate for QNC (default: 1.0 for 100%%). E.g., 0.9 means 90%% of best LLM')

    # Output configuration
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/unirouter',
                       help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='./comparison_results/unirouter',
                       help='Output directory')

    # Data paths
    parser.add_argument('--embeddings-path', type=str, default='data/prompt_embeddings.pkl',
                       help='Path to prompt embeddings')

    return parser.parse_args()


def compute_metrics(costs, perfs, target_accuracy=None, global_cost_range=None):
    """
    Compute peak accuracy, AUDC, and QNC.

    QNC (Query-Normalized Cost) is the relative cost to achieve the same performance
    as the most accurate single LLM. If a method cannot reach that target, QNC = 1.0.

    Args:
        costs: Array of cost values (actual cost: token_count × model_size)
        perfs: Array of performance/accuracy values
        target_accuracy: Target accuracy to achieve (typically best LLM's average)
        global_cost_range: Tuple (min_cost, max_cost) for normalization across all methods

    Returns:
        Tuple: (peak_acc, audc_normalized, qnc, audc_actual)
    """
    peak_acc = perfs.max()

    # Sort by cost
    sorted_idx = np.argsort(costs)
    sorted_cost = costs[sorted_idx]
    sorted_perf = perfs[sorted_idx]

    # === Normalized Cost Metrics (cost normalized to [0,1]) ===
    if global_cost_range is not None:
        min_cost, max_cost = global_cost_range
    else:
        min_cost, max_cost = sorted_cost.min(), sorted_cost.max()

    if max_cost > min_cost:
        norm_cost = (sorted_cost - min_cost) / (max_cost - min_cost)
    else:
        norm_cost = np.zeros_like(sorted_cost)

    # AUDC with normalized cost
    audc_normalized = np.trapezoid(sorted_perf, norm_cost)

    # QNC: normalized cost to reach target_accuracy
    if target_accuracy is None:
        target_accuracy = 0.9 * peak_acc

    above_target = sorted_perf >= target_accuracy
    if above_target.any():
        qnc = norm_cost[above_target][0]
    else:
        qnc = 1.0  # Cannot reach target

    # === Actual Cost Metrics (cost in original units) ===
    # AUDC with actual cost
    audc_actual = np.trapezoid(sorted_perf, sorted_cost)

    return peak_acc, audc_normalized, qnc, audc_actual


def main():
    args = parse_args()

    print("=" * 80)
    print("Comparative Evaluation: Original UniRouter vs Uni-R2")
    print("=" * 80)

    # Parse token budgets
    token_budgets = [int(b) for b in args.token_budgets.split(',')]
    print(f"\nToken budgets: {token_budgets}")

    # Lambda distribution
    lambda_range = parse_lambda_distribution(args.lambda_dist)
    print(f"Lambda distribution: {len(lambda_range)} points from {lambda_range.min():.4f} to {lambda_range.max():.4f}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================
    # Step 1: Load data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)

    dataset_manager = DatasetManager(
        embeddings_path=args.embeddings_path,
        train_ratio=0.8,
        seed=42
    )
    embeddings = dataset_manager.get_embeddings()
    train_idx, test_idx = dataset_manager.get_split_indices()

    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)} queries")
    print(f"  Test: {len(test_idx)} queries")

    # ========================================================================
    # Step 2: Create/Load Validation Set
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: CREATE VALIDATION SET")
    print("=" * 80)

    val_builder = ValidationSetBuilder(
        val_size=args.val_size,
        token_budgets=token_budgets,
        use_clustering=args.use_clustering,
        n_clusters=args.n_clusters
    )

    val_idx = val_builder.create_validation_set(embeddings, train_idx, test_idx, seed=42)
    print(f"\n✓ Created validation set:")
    print(f"  Size: {len(val_idx)} queries")
    print(f"  Token budgets: {token_budgets}")
    if args.use_clustering:
        print(f"  Clusters: {args.n_clusters}")

    # ========================================================================
    # Step 3: Register Initial Models - Original UniRouter
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: ORIGINAL UNIROUTER (Initial Models)")
    print("=" * 80)

    original_router = OriginalUniRouter(
        val_size=args.val_size,
        n_clusters=args.n_clusters,
        use_clustering=args.use_clustering
    )

    # Use same validation set
    original_router.val_indices = val_idx
    original_router.val_embeddings = val_builder.val_embeddings
    original_router.cluster_assignments = val_builder.cluster_assignments
    original_router.cluster_embeddings = val_builder.cluster_embeddings

    print(f"\nRegistering {len(args.initial_model)} initial models...")
    initial_models = {}
    for model_spec in args.initial_model:
        name, size, csv_path = model_spec
        size = float(size)

        if not Path(csv_path).exists():
            print(f"  ⚠️  Skipping {name}: CSV not found at {csv_path}")
            continue

        print(f"\n  {name}:")
        df = pd.read_csv(csv_path)
        original_router.register_model(name, df, val_idx, size)

        initial_models[name] = {
            'size': size,
            'csv_path': csv_path,
            'df': df
        }

    print(f"\n✓ Registered {len(original_router.model_features)} models")

    # ========================================================================
    # Step 4: Register Initial Models - Uni-R2
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: UNI-CORE (Initial Models)")
    print("=" * 80)

    feature_matrix = ModelFeatureMatrix(
        token_budgets=token_budgets,
        use_clustering=args.use_clustering,
        n_clusters=args.n_clusters,
        val_size=args.val_size
    )

    print(f"\nRegistering {len(args.initial_model)} initial models...")
    for name, info in initial_models.items():
        print(f"\n  {name}:")
        features = val_builder.compute_model_features(name, info['df'], val_idx)
        print(f"    Feature shape: {features.shape}")
        feature_matrix.register_model(name, features, info['size'])

    print(f"\n✓ Registered {len(feature_matrix.get_models())} models")

    # ========================================================================
    # Step 5: Evaluate with Initial Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATE WITH INITIAL MODELS")
    print("=" * 80)

    # Prepare test data
    test_embeddings = embeddings[test_idx]

    true_scores = {}
    true_counts = {}

    for name, info in initial_models.items():
        df = info['df']
        test_df = df.iloc[test_idx]
        true_scores[name] = test_df
        true_counts[name] = test_df

    # Evaluate Original UniRouter
    print("\nEvaluating Original UniRouter (initial pool)...")
    original_initial_costs, original_initial_perfs = original_router.evaluate_routing(
        test_embeddings,
        true_scores,
        true_counts,
        lambda_range
    )
    print("✓ Original UniRouter evaluation complete")

    # Evaluate Uni-R2
    print("\nEvaluating Uni-R2 (initial pool)...")
    unir2_router = UniR2Router(
        feature_matrix=feature_matrix,
        val_embeddings=val_builder.val_embeddings if not args.use_clustering else None,
        cluster_embeddings=val_builder.cluster_embeddings if args.use_clustering else None
    )

    uni_r2_initial_costs, uni_r2_initial_perfs = unir2_router.evaluate_routing(
        test_embeddings,
        true_scores,
        true_counts,
        lambda_range
    )
    print("✓ Uni-R2 evaluation complete")

    # ========================================================================
    # Step 6: Add New Models Dynamically
    # ========================================================================
    if args.new_model and len(args.new_model) > 0:
        print("\n" + "=" * 80)
        print("STEP 6: ADD NEW MODELS DYNAMICALLY (No Retraining!)")
        print("=" * 80)

        print(f"\nAdding {len(args.new_model)} new models...")

        for model_spec in args.new_model:
            name, size, csv_path = model_spec
            size = float(size)

            if not Path(csv_path).exists():
                print(f"  ⚠️  Skipping {name}: CSV not found at {csv_path}")
                continue

            print(f"\n  {name}:")
            df = pd.read_csv(csv_path)

            # Add to Original UniRouter
            original_router.register_model(name, df, val_idx, size)

            # Add to Uni-R2
            features = val_builder.compute_model_features(name, df, val_idx)
            print(f"    Feature shape: {features.shape}")
            feature_matrix.register_model(name, features, size)

            # Add to ground truth
            test_df = df.iloc[test_idx]
            true_scores[name] = test_df
            true_counts[name] = test_df

        print(f"\n✓ Now have {len(feature_matrix.get_models())} models total:")
        for model in feature_matrix.get_models():
            print(f"    - {model}")

        # ========================================================================
        # Step 7: Evaluate with Expanded Pool
        # ========================================================================
        print("\n" + "=" * 80)
        print("STEP 7: EVALUATE WITH EXPANDED MODEL POOL")
        print("=" * 80)

        # Evaluate Original UniRouter
        print("\nEvaluating Original UniRouter (expanded pool)...")
        original_expanded_costs, original_expanded_perfs = original_router.evaluate_routing(
            test_embeddings,
            true_scores,
            true_counts,
            lambda_range
        )
        print("✓ Original UniRouter evaluation complete")

        # Evaluate Uni-R2
        print("\nEvaluating Uni-R2 (expanded pool)...")
        uni_r2_expanded_costs, uni_r2_expanded_perfs = unir2_router.evaluate_routing(
            test_embeddings,
            true_scores,
            true_counts,
            lambda_range
        )
        print("✓ Uni-R2 evaluation complete")

    else:
        print("\n" + "=" * 80)
        print("STEP 6: SKIPPED (No new models specified)")
        print("=" * 80)
        original_expanded_costs = original_initial_costs
        original_expanded_perfs = original_initial_perfs
        uni_r2_expanded_costs = uni_r2_initial_costs
        uni_r2_expanded_perfs = uni_r2_initial_perfs

    # ========================================================================
    # Step 8: Find Best Single LLM and Compute Global Cost Range
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: FIND BEST SINGLE LLM FOR QNC TARGET")
    print("=" * 80)

    # Find best single LLM (highest average unlimited-token accuracy)
    best_llm_accuracy = 0.0
    best_llm_name = None

    for name, info in initial_models.items():
        df = info['df']
        test_df = df.iloc[test_idx]

        # Get unlimited token accuracy (average across test queries)
        unlimited_accuracy = test_df['unlimited_score'].mean()

        print(f"\n{name}:")
        print(f"  Unlimited avg accuracy: {unlimited_accuracy:.4f}")

        if unlimited_accuracy > best_llm_accuracy:
            best_llm_accuracy = unlimited_accuracy
            best_llm_name = name

    # Apply target accuracy rate
    target_accuracy = best_llm_accuracy * args.target_accuracy_rate

    print(f"\nBest Single LLM: {best_llm_name}")
    print(f"  Best LLM Accuracy: {best_llm_accuracy:.4f}")
    print(f"  Target Accuracy Rate: {args.target_accuracy_rate:.2f} ({args.target_accuracy_rate*100:.0f}%)")
    print(f"  QNC Target Accuracy: {target_accuracy:.4f}")

    # Compute global cost range for normalization
    all_costs = np.concatenate([
        original_initial_costs,
        original_expanded_costs,
        uni_r2_initial_costs,
        uni_r2_expanded_costs
    ])
    global_cost_range = (all_costs.min(), all_costs.max())
    print(f"\nGlobal Cost Range:")
    print(f"  Min: {global_cost_range[0]:.2f}")
    print(f"  Max: {global_cost_range[1]:.2f}")

    # ========================================================================
    # Step 9: Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: SAVE RESULTS")
    print("=" * 80)

    # Save curves with both actual and normalized costs
    results_data = []
    for method, costs, perfs in [
        ('Original UniRouter (Initial)', original_initial_costs, original_initial_perfs),
        ('Original UniRouter (Expanded)', original_expanded_costs, original_expanded_perfs),
        ('Uni-R2 (Initial)', uni_r2_initial_costs, uni_r2_initial_perfs),
        ('Uni-R2 (Expanded)', uni_r2_expanded_costs, uni_r2_expanded_perfs),
    ]:
        # Normalize cost using global range
        cost_min, cost_max = global_cost_range
        if cost_max > cost_min:
            normalized_costs = (costs - cost_min) / (cost_max - cost_min)
        else:
            normalized_costs = np.zeros_like(costs)

        for lamb, cost, norm_cost, perf in zip(lambda_range, costs, normalized_costs, perfs):
            results_data.append({
                'method': method,
                'lambda': lamb,
                'cost': cost,
                'normalized_cost': norm_cost,
                'performance': perf
            })

    results_df = pd.DataFrame(results_data)
    curves_path = f"{args.output_dir}/comparison_curves.csv"
    results_df.to_csv(curves_path, index=False)
    print(f"✓ Saved curves to {curves_path}")

    # Compute and save metrics (with single QNC column)
    metrics_data = []
    for method, costs, perfs in [
        ('Original UniRouter (Initial)', original_initial_costs, original_initial_perfs),
        ('Original UniRouter (Expanded)', original_expanded_costs, original_expanded_perfs),
        ('Uni-R2 (Initial)', uni_r2_initial_costs, uni_r2_initial_perfs),
        ('Uni-R2 (Expanded)', uni_r2_expanded_costs, uni_r2_expanded_perfs),
    ]:
        peak_acc, audc_norm, qnc, audc_actual = compute_metrics(
            costs, perfs,
            target_accuracy=target_accuracy,
            global_cost_range=global_cost_range
        )
        metrics_data.append({
            'method': method,
            'peak_accuracy': peak_acc,
            'AUDC_normalized': audc_norm,
            'QNC': qnc,
            'AUDC_actual': audc_actual
        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = f"{args.output_dir}/comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Saved metrics to {metrics_path}")

    # Print metrics
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 80)
    print(metrics_df.to_string(index=False))

    # ========================================================================
    # Step 10: Generate Plots (Actual and Normalized Cost)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: GENERATE PLOTS")
    print("=" * 80)

    # Plot 1: Actual Cost
    print("\nGenerating actual cost plot...")
    plt.figure(figsize=(12, 8))

    # Plot initial pool
    plt.plot(original_initial_costs, original_initial_perfs, '--',
             label='Original UniRouter (Initial, unlimited tokens)',
             linewidth=2.5, alpha=0.7, color='blue')
    plt.plot(uni_r2_initial_costs, uni_r2_initial_perfs, '--',
             label='Uni-R2 (Initial, multiple budgets)',
             linewidth=2.5, alpha=0.7, color='green')

    # Plot expanded pool
    if args.new_model and len(args.new_model) > 0:
        plt.plot(original_expanded_costs, original_expanded_perfs, '-',
                 label=f'Original UniRouter (Expanded +{len(args.new_model)} models)',
                 linewidth=2.5, color='blue')
        plt.plot(uni_r2_expanded_costs, uni_r2_expanded_perfs, '-',
                 label=f'Uni-R2 (Expanded +{len(args.new_model)} models, no retraining!)',
                 linewidth=2.5, color='green')

    plt.xlabel('Cost (Token Count × Model Size)', fontsize=12)
    plt.ylabel('Average Quality Score', fontsize=12)
    plt.title('Comparison: Original UniRouter vs Uni-R2 (Actual Cost)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path_actual = f"{args.output_dir}/comparison_plot_actual.png"
    plt.savefig(plot_path_actual, dpi=300, bbox_inches='tight')
    print(f"✓ Saved actual cost plot to {plot_path_actual}")
    plt.close()

    # Plot 2: Normalized Cost
    print("\nGenerating normalized cost plot...")
    plt.figure(figsize=(12, 8))

    # Normalize costs for each method
    def normalize_cost_array(cost_arr):
        """Normalize cost to [0,1] range."""
        cost_min, cost_max = cost_arr.min(), cost_arr.max()
        if cost_max > cost_min:
            return (cost_arr - cost_min) / (cost_max - cost_min)
        else:
            return np.zeros_like(cost_arr)

    original_initial_norm = normalize_cost_array(original_initial_costs)
    uni_r2_initial_norm = normalize_cost_array(uni_r2_initial_costs)
    if args.new_model and len(args.new_model) > 0:
        original_expanded_norm = normalize_cost_array(original_expanded_costs)
        uni_r2_expanded_norm = normalize_cost_array(uni_r2_expanded_costs)

    # Plot initial pool
    plt.plot(original_initial_norm, original_initial_perfs, '--',
             label='Original UniRouter (Initial, unlimited tokens)',
             linewidth=2.5, alpha=0.7, color='blue')
    plt.plot(uni_r2_initial_norm, uni_r2_initial_perfs, '--',
             label='Uni-R2 (Initial, multiple budgets)',
             linewidth=2.5, alpha=0.7, color='green')

    # Plot expanded pool
    if args.new_model and len(args.new_model) > 0:
        plt.plot(original_expanded_norm, original_expanded_perfs, '-',
                 label=f'Original UniRouter (Expanded +{len(args.new_model)} models)',
                 linewidth=2.5, color='blue')
        plt.plot(uni_r2_expanded_norm, uni_r2_expanded_perfs, '-',
                 label=f'Uni-R2 (Expanded +{len(args.new_model)} models, no retraining!)',
                 linewidth=2.5, color='green')

    plt.xlabel('Normalized Cost [0,1]', fontsize=12)
    plt.ylabel('Average Quality Score', fontsize=12)
    plt.title('Comparison: Original UniRouter vs Uni-R2 (Normalized Cost)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path_normalized = f"{args.output_dir}/comparison_plot_normalized.png"
    plt.savefig(plot_path_normalized, dpi=300, bbox_inches='tight')
    print(f"✓ Saved normalized cost plot to {plot_path_normalized}")
    plt.close()

    # ========================================================================
    # Step 11: Save Checkpoints
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 11: SAVE CHECKPOINTS")
    print("=" * 80)

    original_checkpoint = f"{args.checkpoint_dir}/original_unirouter.pkl"
    original_router.save(original_checkpoint)
    print(f"✓ Saved Original UniRouter to {original_checkpoint}")

    uni_r2_checkpoint = f"{args.checkpoint_dir}/uni_r2_feature_matrix.pkl"
    feature_matrix.save(uni_r2_checkpoint)
    print(f"✓ Saved Uni-R2 feature matrix to {uni_r2_checkpoint}")

    # ========================================================================
    # Step 12: Create config.txt
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 12: CREATE CONFIG FILES")
    print("=" * 80)

    # Build config content
    config_content = f"""# UniRouter Experiment Configuration
# Comparison: Original UniRouter vs Uni-R2

[Validation Set]
val_size = {args.val_size}
n_clusters = {args.n_clusters}
use_clustering = {args.use_clustering}

[Token Budgets]
budgets = {', '.join(str(b) for b in token_budgets)}

[Lambda Distribution]
lambda_dist = {args.lambda_dist}

[QNC Configuration]
target_accuracy_rate = {args.target_accuracy_rate}
best_llm = {best_llm_name}
best_llm_accuracy = {best_llm_accuracy:.4f}
target_accuracy = {target_accuracy:.4f}

[Initial LLM Pool]
# Format: Name | Size (B) | CSV Path
"""

    # Add initial models to config
    for name, info in initial_models.items():
        config_content += f"{name} | {info['size']} | {info['csv_path']}\n"

    # Add new models if any
    if args.new_model and len(args.new_model) > 0:
        config_content += "\n[New Models (Dynamically Added)]\n"
        config_content += "# Format: Name | Size (B) | CSV Path\n"
        for model_spec in args.new_model:
            name, size, csv_path = model_spec
            config_content += f"{name} | {size} | {csv_path}\n"

    # Save to results directory
    config_path_results = f"{args.output_dir}/config.txt"
    with open(config_path_results, 'w') as f:
        f.write(config_content)
    print(f"✓ Saved config to {config_path_results}")

    # Save to checkpoint directory
    config_path_checkpoint = f"{args.checkpoint_dir}/config.txt"
    with open(config_path_checkpoint, 'w') as f:
        f.write(config_content)
    print(f"✓ Saved config to {config_path_checkpoint}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nPerformance Metrics:")
    print(metrics_df.to_string(index=False))

    if args.new_model and len(args.new_model) > 0:
        print("\n" + "=" * 80)
        print("KEY ACHIEVEMENT: Dynamic Model Addition")
        print("=" * 80)
        print(f"✓ Started with {len(args.initial_model)} initial models")
        print(f"✓ Added {len(args.new_model)} NEW models WITHOUT retraining")
        print(f"✓ Final pool: {len(feature_matrix.get_models())} models")
        print(f"✓ Original UniRouter: {original_initial_perfs.max():.4f} → {original_expanded_perfs.max():.4f}")
        print(f"✓ Uni-R2: {uni_r2_initial_perfs.max():.4f} → {uni_r2_expanded_perfs.max():.4f}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

    print("\nFiles saved:")
    print(f"  - {curves_path}")
    print(f"  - {metrics_path}")
    print(f"  - {plot_path_actual}")
    print(f"  - {plot_path_normalized}")
    print(f"  - {original_checkpoint}")
    print(f"  - {uni_r2_checkpoint}")


if __name__ == "__main__":
    main()
