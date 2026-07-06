#!/usr/bin/env python3
"""
Generate Ablation Study Tables for DropoutTS

This script analyzes ablation experiment results and generates tables showing:
1. Component-wise ablation results (Detrend, SFM Anchor, Instance Norm)
2. Performance comparison across different noise levels
3. Best configurations for each component
"""

import json
import pandas as pd
import argparse
from pathlib import Path
import numpy as np


def get_ablation_config(cfg_path):
    """
    Extract ablation configuration from config file.
    
    Returns:
        dict with keys: detrend_method, use_instance_norm, use_sfm_anchor, has_dropout
    """
    def to_bool(val, default=True):
        """Convert various types to boolean."""
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return bool(val)
    
    def clean_string(val, default='robust_ols'):
        """Clean string value by removing extra quotes."""
        if val is None:
            return default
        if isinstance(val, str):
            # Remove extra quotes: "'none'" -> "none"
            val = val.strip().strip("'\"")
        return val
    
    try:
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        
        callbacks = cfg.get('callbacks', [])
        for cb in callbacks:
            if cb.get('name') == 'DropoutTSCallback':
                params = cb.get('params', {})
                
                # Clean detrend_method and use defaults for missing params
                detrend_raw = params.get('detrend_method')
                detrend_method = clean_string(detrend_raw, 'robust_ols')
                
                return {
                    'has_dropout': True,
                    'detrend_method': detrend_method,
                    'use_instance_norm': to_bool(params.get('use_instance_norm'), True),
                    'use_sfm_anchor': to_bool(params.get('use_sfm_anchor'), True),
                    'p_min': params.get('p_min', 0.1),
                    'p_max': params.get('p_max', 0.5),
                    'init_alpha': params.get('init_alpha', 10.0),
                    'init_sensitivity': params.get('init_sensitivity', 5.0),
                    'sparsity_weight': params.get('sparsity_weight', 0.1),
                }
        
        # No DropoutTS = Baseline
        return {
            'has_dropout': False,
            'detrend_method': None,
            'use_instance_norm': None,
            'use_sfm_anchor': None,
            'p_min': None,
            'p_max': None,
            'init_alpha': None,
            'init_sensitivity': None,
            'sparsity_weight': None,
        }
    
    except Exception as e:
        print(f"Warning: Failed to parse {cfg_path}: {e}")
        return None


def get_ablation_name(config):
    """Generate human-readable ablation experiment name."""
    if not config['has_dropout']:
        return "Baseline"
    
    detrend = config['detrend_method']
    norm = config['use_instance_norm']
    sfm = config['use_sfm_anchor']
    
    # Convert string booleans to actual booleans if needed
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return bool(val)
    
    norm = to_bool(norm)
    sfm = to_bool(sfm)
    
    # Full model
    if detrend == 'robust_ols' and norm and sfm:
        return "Full_Model"
    
    # Single component ablations
    if detrend == 'none' and norm and sfm:
        return "w/o_Detrend"
    if detrend == 'simple' and norm and sfm:
        return "Simple_Detrend"
    if detrend == 'robust_ols' and not norm and sfm:
        return "w/o_Instance_Norm"
    if detrend == 'robust_ols' and norm and not sfm:
        return "w/o_SFM_Anchor"
    
    # Combined ablations
    if detrend == 'none' and not norm and sfm:
        return "w/o_Detrend+Norm"
    if detrend == 'robust_ols' and not norm and not sfm:
        return "w/o_Norm+SFM"
    if detrend == 'none' and not norm and not sfm:
        return "Minimal_Model"
    
    # Other combinations
    return f"Custom({detrend[:3] if detrend else 'N'},{int(norm)},{int(sfm)})"


def extract_experiment_info(exp_dir):
    """Extract experiment name from directory path."""
    parts = exp_dir.split('/')
    if len(parts) >= 2:
        exp_name = parts[-2]
        hash_id = parts[-1]
        return exp_name, hash_id
    return None, None


def collect_ablation_results(base_dir):
    """Collect all ablation experiment results."""
    results = []
    
    print(f"Scanning {base_dir} for ablation experiment results...")
    
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
        
        # Parse experiment name: Dataset_Epochs_InputLen_OutputLen
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
        
        # Get ablation configuration
        ablation_config = get_ablation_config(cfg_file)
        if ablation_config is None:
            continue
        
        ablation_name = get_ablation_name(ablation_config)
        
        overall_metrics = metrics.get('overall', {})
        
        # Helper to convert to bool
        def to_bool(val):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes')
            return bool(val)
        
        result = {
            'Hash': hash_id[:8],  # Short hash for display
            'Dataset': dataset,
            'Input_Len': input_len,
            'Output_Len': output_len,
            'Ablation': ablation_name,
            'Detrend': ablation_config['detrend_method'] or '-',
            'Instance_Norm': '✓' if to_bool(ablation_config['use_instance_norm']) else '✗' if ablation_config['has_dropout'] else '-',
            'SFM_Anchor': '✓' if to_bool(ablation_config['use_sfm_anchor']) else '✗' if ablation_config['has_dropout'] else '-',
            'MSE': round(overall_metrics.get('MSE'), 6) if overall_metrics.get('MSE') is not None else None,
            'MAE': round(overall_metrics.get('MAE'), 6) if overall_metrics.get('MAE') is not None else None,
            'RMSE': round(overall_metrics.get('RMSE'), 6) if overall_metrics.get('RMSE') is not None else None,
            'MAPE': round(overall_metrics.get('MAPE'), 6) if overall_metrics.get('MAPE') is not None else None,
            'p_max': ablation_config['p_max'],
            'init_alpha': ablation_config['init_alpha'],
            'init_sensitivity': ablation_config['init_sensitivity'],
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def create_ablation_table(df):
    """
    Create ablation table showing impact of each component.
    
    Format:
    | Method | Detrend | Norm | SFM | Dataset1_MSE | Dataset1_MAE | ... | Avg_MSE | Avg_MAE |
    """
    if df.empty:
        return pd.DataFrame()
    
    print(f"\n{'='*80}")
    print(f"ABLATION TABLE (Best MSE/MAE across horizons)")
    print(f"{'='*80}")
    
    # Define ablation order for display
    ablation_order = [
        'Baseline',
        'Full_Model',
        'w/o_Detrend',
        'Simple_Detrend',
        'w/o_SFM_Anchor',
        'w/o_Instance_Norm',
        'w/o_Detrend+Norm',
        'w/o_Norm+SFM',
        'Minimal_Model',
    ]
    
    # Get unique datasets
    datasets = sorted(df['Dataset'].unique())
    
    rows = []
    
    for ablation in ablation_order:
        ablation_data = df[df['Ablation'] == ablation]
        
        if len(ablation_data) == 0:
            continue
        
        # Get configuration markers
        if ablation == 'Baseline':
            detrend, norm, sfm = '-', '-', '-'
        else:
            sample = ablation_data.iloc[0]
            detrend = sample['Detrend']
            norm = sample['Instance_Norm']
            sfm = sample['SFM_Anchor']
        
        row = {
            'Method': ablation,
            'Detrend': detrend,
            'Norm': norm,
            'SFM': sfm,
        }
        
        # For each dataset, get best (min) MSE and MAE across all horizons
        mse_metrics = []
        mae_metrics = []
        
        for dataset in datasets:
            dataset_ablation = ablation_data[ablation_data['Dataset'] == dataset]
            if len(dataset_ablation) > 0:
                best_mse = dataset_ablation['MSE'].min()
                best_mae = dataset_ablation['MAE'].min()
                row[f'{dataset}_MSE'] = f"{best_mse:.3f}"
                row[f'{dataset}_MAE'] = f"{best_mae:.3f}"
                mse_metrics.append(best_mse)
                mae_metrics.append(best_mae)
            else:
                row[f'{dataset}_MSE'] = '-'
                row[f'{dataset}_MAE'] = '-'
        
        # Calculate averages across datasets
        if mse_metrics:
            row['Avg_MSE'] = f"{np.mean(mse_metrics):.3f}"
        else:
            row['Avg_MSE'] = '-'
        
        if mae_metrics:
            row['Avg_MAE'] = f"{np.mean(mae_metrics):.3f}"
        else:
            row['Avg_MAE'] = '-'
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    
    # Reorder columns: Method, Detrend, Norm, SFM, dataset metrics..., averages
    metric_cols = []
    for dataset in datasets:
        metric_cols.extend([f'{dataset}_MSE', f'{dataset}_MAE'])
    
    col_order = ['Method', 'Detrend', 'Norm', 'SFM'] + metric_cols + ['Avg_MSE', 'Avg_MAE']
    result_df = result_df[col_order]
    
    return result_df


def create_horizon_wise_table(df):
    """
    Create detailed table showing MSE and MAE results for each horizon.
    """
    if df.empty:
        return pd.DataFrame()
    
    print(f"\n{'='*80}")
    print(f"HORIZON-WISE ABLATION TABLE (MSE & MAE)")
    print(f"{'='*80}")
    
    ablation_order = [
        'Baseline',
        'Full_Model',
        'w/o_Detrend',
        'Simple_Detrend',
        'w/o_SFM_Anchor',
        'w/o_Instance_Norm',
    ]
    
    datasets = sorted(df['Dataset'].unique())
    horizons = sorted(df['Output_Len'].unique())
    
    rows = []
    
    for dataset in datasets:
        dataset_data = df[df['Dataset'] == dataset]
        
        for horizon in horizons:
            horizon_data = dataset_data[dataset_data['Output_Len'] == horizon]
            
            if len(horizon_data) == 0:
                continue
            
            for ablation in ablation_order:
                ablation_horizon = horizon_data[horizon_data['Ablation'] == ablation]
                
                if len(ablation_horizon) == 0:
                    continue
                
                best_mse = ablation_horizon['MSE'].min()
                best_mae = ablation_horizon['MAE'].min()
                
                row = {
                    'Dataset': dataset,
                    'Horizon': horizon,
                    'Method': ablation,
                    'MSE': f"{best_mse:.3f}",
                    'MAE': f"{best_mae:.3f}"
                }
                
                rows.append(row)
    
    return pd.DataFrame(rows)


def calculate_improvements(df):
    """Calculate improvement percentages relative to baseline for both MSE and MAE."""
    if df.empty:
        return pd.DataFrame()
    
    print(f"\n{'='*80}")
    print(f"IMPROVEMENT OVER BASELINE (MSE & MAE)")
    print(f"{'='*80}")
    
    datasets = sorted(df['Dataset'].unique())
    ablations = sorted([a for a in df['Ablation'].unique() if a != 'Baseline'])
    
    rows = []
    
    for dataset in datasets:
        dataset_data = df[df['Dataset'] == dataset]
        
        baseline_data = dataset_data[dataset_data['Ablation'] == 'Baseline']
        if len(baseline_data) == 0:
            continue
        
        baseline_mse = baseline_data['MSE'].min()
        baseline_mae = baseline_data['MAE'].min()
        
        for ablation in ablations:
            ablation_data = dataset_data[dataset_data['Ablation'] == ablation]
            
            if len(ablation_data) == 0:
                continue
            
            ablation_mse = ablation_data['MSE'].min()
            ablation_mae = ablation_data['MAE'].min()
            
            improvement_mse = ((baseline_mse - ablation_mse) / baseline_mse) * 100
            improvement_mae = ((baseline_mae - ablation_mae) / baseline_mae) * 100
            
            row = {
                'Dataset': dataset,
                'Method': ablation,
                'Baseline_MSE': f"{baseline_mse:.3f}",
                'Method_MSE': f"{ablation_mse:.3f}",
                'Improve_MSE(%)': f"{improvement_mse:+.2f}%",
                'Baseline_MAE': f"{baseline_mae:.3f}",
                'Method_MAE': f"{ablation_mae:.3f}",
                'Improve_MAE(%)': f"{improvement_mae:+.2f}%"
            }
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Generate ablation study tables for DropoutTS (MSE & MAE)'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing experiment checkpoints (e.g., checkpoints/Informer)')
    args = parser.parse_args()
    
    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        print(f"❌ Error: Input directory does not exist: {base_dir}")
        return
    
    # Collect all results
    print("🔍 Collecting ablation experiment results...")
    df = collect_ablation_results(base_dir)
    
    if df.empty:
        print("❌ No experiment results found.")
        return
    
    print(f"✅ Collected {len(df)} experiment results\n")
    
    # ========== Table 1: Main Ablation Table ==========
    ablation_table = create_ablation_table(df)
    
    if not ablation_table.empty:
        output_file = base_dir / "ablation_table.csv"
        ablation_table.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        print("\n" + ablation_table.to_string(index=False))
    
    # ========== Table 2: Horizon-wise Details ==========
    horizon_table = create_horizon_wise_table(df)
    
    if not horizon_table.empty:
        output_file = base_dir / "ablation_horizon_wise.csv"
        horizon_table.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        print(f"\nPreview (first 20 rows):")
        print(horizon_table.head(20).to_string(index=False))
    
    # ========== Table 3: Improvement Percentages ==========
    improvement_table = calculate_improvements(df)
    
    if not improvement_table.empty:
        output_file = base_dir / "ablation_improvements.csv"
        improvement_table.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        print(f"\nPreview (first 15 rows):")
        print(improvement_table.head(15).to_string(index=False))
    
    # ========== Summary Statistics ==========
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal experiments: {len(df)}")
    print(f"\nAblation configurations found:")
    for ablation in sorted(df['Ablation'].unique()):
        count = len(df[df['Ablation'] == ablation])
        print(f"  - {ablation}: {count} experiments")
    
    print(f"\nDatasets:")
    for dataset in sorted(df['Dataset'].unique()):
        count = len(df[df['Dataset'] == dataset])
        print(f"  - {dataset}: {count} experiments")
    
    print(f"\nHorizons: {sorted(df['Output_Len'].unique())}")
    
    # Best configuration (by MAE)
    print("\n" + "="*80)
    print("BEST CONFIGURATION (by MAE)")
    print("="*80)
    
    non_baseline = df[df['Ablation'] != 'Baseline']
    if len(non_baseline) > 0:
        best_idx = non_baseline['MAE'].idxmin()
        best = non_baseline.loc[best_idx]
        
        print(f"\nBest MAE: {best['MAE']:.6f}")
        print(f"Best MSE: {best['MSE']:.6f}")
        print(f"Configuration: {best['Ablation']}")
        print(f"  - Dataset: {best['Dataset']}")
        print(f"  - Horizon: {best['Output_Len']}")
        print(f"  - Detrend: {best['Detrend']}")
        print(f"  - Instance Norm: {best['Instance_Norm']}")
        print(f"  - SFM Anchor: {best['SFM_Anchor']}")
        
        # Compare with baseline
        baseline = df[(df['Ablation'] == 'Baseline') & 
                     (df['Dataset'] == best['Dataset']) & 
                     (df['Output_Len'] == best['Output_Len'])]
        
        if len(baseline) > 0:
            baseline_mae = baseline['MAE'].min()
            baseline_mse = baseline['MSE'].min()
            improvement_mae = ((baseline_mae - best['MAE']) / baseline_mae) * 100
            improvement_mse = ((baseline_mse - best['MSE']) / baseline_mse) * 100
            print(f"\nBaseline MAE: {baseline_mae:.6f}, Improvement: {improvement_mae:+.2f}%")
            print(f"Baseline MSE: {baseline_mse:.6f}, Improvement: {improvement_mse:+.2f}%")
    
    print("\n" + "="*80)
    print("✅ COMPLETED! Generated ablation analysis tables.")
    print("="*80)


if __name__ == "__main__":
    main()
