#!/usr/bin/env python3
"""
Generate Two Result Tables:
1. SOTA Best Results - Shows best performance across all hyperparameter configurations
2. Detailed All Experiments - Shows all experiments with their hyperparameter configurations
"""
import json
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def has_dropout_ts(cfg_path):
    """Check if DropoutTS is enabled in the config file"""
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        callbacks = cfg.get('callbacks', [])
        for callback in callbacks:
            if callback.get('name') == 'DropoutTSCallback':
                return True
        return False
    except Exception:
        return False

def get_dropout_params(cfg_path):
    """
    Extract DropoutTS hyperparameters from config.
    Returns dict of params with default values if not found.
    Note: Some params may not be saved in cfg.json, we fill with defaults.
    """
    # Default values from DropoutTSCallback.__init__
    DEFAULTS = {
        'p_min': 0.1,
        'p_max': 0.5,
        'init_alpha': 10.0,
        'init_sensitivity': 5.0,
        'sparsity_weight': 0.1
    }
    
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        
        callbacks = cfg.get('callbacks', [])
        for cb in callbacks:
            if cb.get('name') == 'DropoutTSCallback':
                params = cb.get('params', {})
                
                # Parse params, using defaults if not present
                def parse_param(key, default):
                    val = params.get(key)
                    if val is None or val == '':
                        return default
                    try:
                        # Try to convert string to number
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                return {
                    'has_dropout': True,
                    'p_min': parse_param('p_min', DEFAULTS['p_min']),
                    'p_max': parse_param('p_max', DEFAULTS['p_max']),
                    'init_alpha': parse_param('init_alpha', DEFAULTS['init_alpha']),
                    'init_sensitivity': parse_param('init_sensitivity', DEFAULTS['init_sensitivity']),
                    'sparsity_weight': parse_param('sparsity_weight', DEFAULTS['sparsity_weight']),
                }
        
        return {
            'has_dropout': False,
            'p_min': '-',
            'p_max': '-',
            'init_alpha': '-',
            'init_sensitivity': '-',
            'sparsity_weight': '-',
        }
    except Exception as e:
        print(f"Warning: Failed to parse {cfg_path}: {e}")
        return None

def extract_experiment_info(exp_dir):
    """Extract experiment name and hash from directory path"""
    parts = exp_dir.split('/')
    if len(parts) >= 2:
        exp_name = parts[-2]
        hash_id = parts[-1]
        return exp_name, hash_id
    return None, None

def collect_all_results(base_dir):
    """Collect all experiment results"""
    results = []
    
    print(f"Scanning {base_dir} for all experiment results...")
    
    for metrics_file in base_dir.rglob("test_metrics.json"):
        exp_dir = metrics_file.parent
        cfg_file = exp_dir / "cfg.json"
        
        if not cfg_file.exists():
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            continue
        
        exp_name, hash_id = extract_experiment_info(str(exp_dir))
        if not exp_name:
            continue
        
        # Parse experiment name
        parts = exp_name.split('_')
        if len(parts) >= 4:
            try:
                output_len = int(parts[-1])
                input_len = int(parts[-2])
                epochs = parts[-3]
                dataset = "_".join(parts[:-3])
            except ValueError:
                continue
        else:
            continue
        
        params = get_dropout_params(cfg_file)
        if params is None:
            continue
        
        overall_metrics = metrics.get('overall', {})
        
        result = {
            'Experiment_Hash': hash_id,
            'Dataset': dataset,
            'Input_Length': input_len,
            'Output_Length': output_len,
            'Epochs': epochs,
            'Has_DropoutTS': 'Yes' if params['has_dropout'] else 'No',
            'p_min': params['p_min'],
            'p_max': params['p_max'],
            'init_alpha': params['init_alpha'],
            'init_sensitivity': params['init_sensitivity'],
            'sparsity_weight': params['sparsity_weight'],
            'MSE': round(overall_metrics.get('MSE'), 6) if overall_metrics.get('MSE') is not None else None,
            'MAE': round(overall_metrics.get('MAE'), 6) if overall_metrics.get('MAE') is not None else None,
            'RMSE': round(overall_metrics.get('RMSE'), 6) if overall_metrics.get('RMSE') is not None else None,
            'MAPE': round(overall_metrics.get('MAPE'), 6) if overall_metrics.get('MAPE') is not None else None,
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def calculate_improvement_percentage(baseline, dropout):
    """Calculate improvement percentage"""
    if baseline is None or dropout is None or baseline == 0:
        return None
    improvement = (baseline - dropout) / baseline * 100
    return round(improvement, 2)

def create_sota_table(df):
    """Create SOTA table showing best results (aggregated by min)"""
    if df.empty:
        return pd.DataFrame()
    
    print("\n" + "="*80)
    print("TABLE 1: SOTA BEST RESULTS (Min across all hyperparameters)")
    print("="*80)
    
    available_datasets = sorted(df['Dataset'].unique())
    available_input_lengths = sorted(df['Input_Length'].unique())
    
    formatted_rows = []
    
    for input_len in available_input_lengths:
        for dataset in available_datasets:
            dataset_data = df[(df['Dataset'] == dataset) & (df['Input_Length'] == input_len)]
            
            if len(dataset_data) == 0:
                continue
            
            current_horizons = sorted(dataset_data['Output_Length'].unique())
            current_group_rows = []
            
            for horizon in current_horizons:
                horizon_data = dataset_data[dataset_data['Output_Length'] == horizon]
                
                without_dropout = horizon_data[horizon_data['Has_DropoutTS'] == 'No']
                with_dropout = horizon_data[horizon_data['Has_DropoutTS'] == 'Yes']
                
                def get_best_metric(df_subset, metric):
                    if len(df_subset) == 0: return None
                    values = df_subset[metric].dropna()
                    if len(values) == 0: return None
                    return float(f"{values.min():.3f}")
                
                raw_mse = get_best_metric(without_dropout, 'MSE')
                raw_mae = get_best_metric(without_dropout, 'MAE')
                dropout_mse = get_best_metric(with_dropout, 'MSE')
                dropout_mae = get_best_metric(with_dropout, 'MAE')
                
                if raw_mse is not None or dropout_mse is not None:
                    row = {
                        'Input_Length': input_len,
                        'Dataset': dataset,
                        'Horizon': horizon,
                        'Baseline_MSE': raw_mse,
                        'Baseline_MAE': raw_mae,
                        'DropoutTS_MSE': dropout_mse,
                        'DropoutTS_MAE': dropout_mae,
                    }
                    formatted_rows.append(row)
                    current_group_rows.append(row)
            
            # Add Avg row
            if len(current_group_rows) > 0:
                def calc_avg_from_rows(rows, key):
                    vals = [r[key] for r in rows if r.get(key) is not None]
                    if not vals: return None
                    return float(f"{sum(vals) / len(vals):.3f}")

                avg_row = {
                    'Input_Length': input_len,
                    'Dataset': dataset,
                    'Horizon': 'Avg',
                    'Baseline_MSE': calc_avg_from_rows(current_group_rows, 'Baseline_MSE'),
                    'Baseline_MAE': calc_avg_from_rows(current_group_rows, 'Baseline_MAE'),
                    'DropoutTS_MSE': calc_avg_from_rows(current_group_rows, 'DropoutTS_MSE'),
                    'DropoutTS_MAE': calc_avg_from_rows(current_group_rows, 'DropoutTS_MAE'),
                }
                formatted_rows.append(avg_row)

                # Add Improvement row - calculate based on Avg row values
                avg_imp_mse = calculate_improvement_percentage(
                    avg_row.get('Baseline_MSE'), 
                    avg_row.get('DropoutTS_MSE')
                )
                avg_imp_mae = calculate_improvement_percentage(
                    avg_row.get('Baseline_MAE'), 
                    avg_row.get('DropoutTS_MAE')
                )
                
                delta_row = {
                    'Input_Length': input_len,
                    'Dataset': dataset,
                    'Horizon': 'Improvement(%)',
                    'Baseline_MSE': None,
                    'Baseline_MAE': None,
                    'DropoutTS_MSE': f"{avg_imp_mse:+.2f}%" if avg_imp_mse is not None else None,
                    'DropoutTS_MAE': f"{avg_imp_mae:+.2f}%" if avg_imp_mae is not None else None,
                }
                formatted_rows.append(delta_row)
    
    return pd.DataFrame(formatted_rows)

def main():
    parser = argparse.ArgumentParser(
        description='Generate SOTA and Detailed results tables'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing experiment checkpoints')
    parser.add_argument('--sort_by', type=str, default='MAE',
                        choices=['MSE', 'MAE', 'RMSE', 'MAPE'],
                        help='Metric to sort detailed table by (default: MAE)')
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        print(f"Error: Input directory does not exist: {base_dir}")
        return
    
    # Collect all results
    print("Collecting all experiment results...")
    df = collect_all_results(base_dir)
    
    if df.empty:
        print("No experiment results found.")
        return
    
    print(f"Collected {len(df)} experiment results\n")
    
    # ========== TABLE 1: SOTA Best Results ==========
    sota_df = create_sota_table(df)
    
    if not sota_df.empty:
        # Format numeric columns to 3 decimal places (except Improvement%)
        for col in ['Baseline_MSE', 'Baseline_MAE', 'DropoutTS_MSE', 'DropoutTS_MAE']:
            if col in sota_df.columns:
                sota_df[col] = sota_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                )
        
        sota_output = base_dir / "table1_sota_best_results.csv"
        sota_df.to_csv(sota_output, index=False)
        print(f"\n✅ TABLE 1 saved to: {sota_output}")
        print("\nPreview (first 20 rows):")
        print(sota_df.head(20).to_string(index=False))
    
    # ========== TABLE 2: Detailed All Experiments ==========
    print("\n" + "="*80)
    print("TABLE 2: ALL EXPERIMENTS WITH HYPERPARAMETERS")
    print("="*80)
    
    # Sort detailed table: Group by Dataset and Output_Length, Baseline first, then by MSE
    # Create a sort key: Baseline (No) comes first (0), DropoutTS (Yes) comes second (1)
    df['_baseline_order'] = df['Has_DropoutTS'].apply(lambda x: 0 if x == 'No' else 1)
    
    # Sort by: Dataset -> Output_Length -> Baseline first -> MSE
    df_sorted = df.sort_values(
        by=['Dataset', 'Output_Length', '_baseline_order', 'MSE'],
        ascending=[True, True, True, True],
        na_position='last'
    )
    
    # Drop the temporary sort column and unnecessary columns
    df_sorted = df_sorted.drop(columns=['_baseline_order'])
    
    # Remove unnecessary columns: Experiment_Hash, RMSE, MAPE
    columns_to_remove = ['Experiment_Hash', 'RMSE', 'MAPE']
    df_sorted = df_sorted.drop(columns=[col for col in columns_to_remove if col in df_sorted.columns])
    
    # Format numeric columns to consistent decimal places
    numeric_cols = ['MSE', 'MAE']
    for col in numeric_cols:
        if col in df_sorted.columns:
            df_sorted[col] = df_sorted[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
    
    # Format hyperparameter columns to remove unnecessary decimals
    hyperparam_cols = ['p_min', 'p_max', 'init_alpha', 'init_sensitivity', 'sparsity_weight']
    for col in hyperparam_cols:
        if col in df_sorted.columns:
            df_sorted[col] = df_sorted[col].apply(
                lambda x: f"{float(x):.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
            )
    
    detailed_output = base_dir / "table2_all_experiments_detailed.csv"
    df_sorted.to_csv(detailed_output, index=False)
    print(f"\n✅ TABLE 2 saved to: {detailed_output}")
    print(f"\nPreview (first 15 rows):")
    print(df_sorted.head(15).to_string(index=False))
    
    # ========== Summary Statistics ==========
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total experiments: {len(df)}")
    print(f"  - With DropoutTS: {len(df[df['Has_DropoutTS'] == 'Yes'])}")
    print(f"  - Baseline (no DropoutTS): {len(df[df['Has_DropoutTS'] == 'No'])}")
    print(f"\nUnique configurations:")
    print(f"  - Datasets: {df['Dataset'].nunique()}")
    print(f"  - Input lengths: {sorted(df['Input_Length'].unique())}")
    print(f"  - Output lengths: {sorted(df['Output_Length'].unique())}")
    
    if len(df[df['Has_DropoutTS'] == 'Yes']) > 0:
        dropout_df = df[df['Has_DropoutTS'] == 'Yes']
        print(f"\nDropoutTS hyperparameter ranges:")
        for param in ['p_max', 'init_sensitivity', 'sparsity_weight']:
            unique_vals = dropout_df[param].unique()
            unique_vals = [v for v in unique_vals if v != '-']
            if unique_vals:
                print(f"  - {param}: {sorted(set(unique_vals))}")
    
    # Best configurations per dataset
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS BY DATASET")
    print("="*80)
    
    for dataset in sorted(df['Dataset'].unique()):
        dataset_df = df[df['Dataset'] == dataset]
        baseline = dataset_df[dataset_df['Has_DropoutTS'] == 'No']
        dropout = dataset_df[dataset_df['Has_DropoutTS'] == 'Yes']
        
        if len(baseline) > 0 and len(dropout) > 0:
            best_baseline_mae = baseline['MAE'].min()
            best_dropout_mae = dropout['MAE'].min()
            improvement = ((best_baseline_mae - best_dropout_mae) / best_baseline_mae * 100)
            
            print(f"\n📊 {dataset}:")
            print(f"  Baseline best MAE: {best_baseline_mae:.6f}")
            print(f"  DropoutTS best MAE: {best_dropout_mae:.6f}")
            print(f"  Improvement: {improvement:+.2f}%")
            
            best_config = dropout[dropout['MAE'] == best_dropout_mae].iloc[0]
            print(f"  Best hyperparameters:")
            print(f"    - p_max: {best_config['p_max']}")
            print(f"    - init_alpha: {best_config['init_alpha']}")
            print(f"    - init_sensitivity: {best_config['init_sensitivity']}")
            print(f"    - sparsity_weight: {best_config['sparsity_weight']}")
    
    print("\n" + "="*80)
    print("✅ COMPLETED! Generated 2 tables:")
    print(f"   1. {detailed_output.name} - SOTA best results")
    print(f"   2. {sota_output.name} - All experiments with hyperparameters")
    print("="*80)

if __name__ == "__main__":
    main()

