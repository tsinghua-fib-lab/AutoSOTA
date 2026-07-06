#!/usr/bin/env python3
"""
Cognitive Parameter Fitting Script - Extract cognitive parameters from behavior trajectories

Uses the modular GRW fitting architecture.

Usage:
    # Single run directory (new format - group CSVs)
    python fit_params.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/

    # Single file (legacy format)
    python fit_params.py --csv_file logs/cigt/DeepSeek-R1/results.csv

    # Batch analysis (multiple runs)
    python fit_params.py --folder logs/cigt/DeepSeek-R1/ --output_dir logs/analysis/

    # Extended mode (with perception parameters)
    python fit_params.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --extended

    # With configuration file
    python fit_params.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --config scripts/config/analysis.json
"""
import argparse
import os
import sys
import json
import pandas as pd
import importlib.util

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_dir = os.path.join(project_root, 'src')

# Add to sys.path for relative imports within src modules
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import modules using importlib to ensure proper loading
def import_module_from_path(module_name, file_path):
    """Import a module from a specific file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import src package and submodules
import_module_from_path('src', os.path.join(src_dir, '__init__.py'))
import_module_from_path('src.fitting', os.path.join(src_dir, 'fitting', '__init__.py'))
import_module_from_path('src.core.utils', os.path.join(src_dir, 'core', 'utils.py'))

from src.fitting import batch_fit_cognitive_model
from src.core.utils import (
    load_csv_with_validation,
    prepare_for_fitting,
    format_results_report,
    setup_logger
)


def load_config(config_path: str = None) -> dict:
    """Load analysis configuration from JSON file"""
    # Default config path
    if config_path is None:
        config_path = os.path.join(project_root, 'scripts', 'config', 'analysis.json')

    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}, using defaults")
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        print(f"⚠️  Failed to load config: {e}, using defaults")
        return {}


def analyze_single_file(csv_path: str, extended: bool, output_dir: str, config: dict) -> int:
    """Analyze single CSV file"""
    logger = setup_logger("Single_Analysis")

    logger.info(f"Loading file: {csv_path}")
    df = load_csv_with_validation(csv_path, required_columns=['group', 'action', 'reward'])

    if df is None:
        logger.error("Failed to load file")
        return 1

    # Apply config filters if available
    if config and 'parameter_fitting' in config:
        fit_config = config['parameter_fitting']
        filters = fit_config.get('data_filters', {})

        # Override extended mode if specified in config
        if 'extended_mode' in fit_config:
            extended = fit_config['extended_mode']
    else:
        filters = {}

    # Preprocess
    df = prepare_for_fitting(df, filters=filters)
    logger.info(f"Data preprocessing complete: {len(df)} records, {df['group'].nunique()} scenario groups")

    # Fit parameters
    logger.info("Starting cognitive parameter fitting...")
    results = batch_fit_cognitive_model(df, extended_mode=extended, config=config)

    if not results:
        logger.error("Fitting failed, no valid results")
        return 1

    # Output report
    print("\n" + "=" * 70)
    print("Cognitive Parameter Fitting Results")
    print("=" * 70)
    report = format_results_report(results)
    print(report)
    print("=" * 70)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(output_dir, f"cognitive_report_{model_name}.csv")

        # Convert to DataFrame for saving
        records = []
        for group, result in results.items():
            record = {
                'Group': group,
                'nll': result.nll,
                'bic': result.bic,
                'count': result.count,
            }
            for param, value in result.params.items():
                record[param] = value
            records.append(record)

        output_df = pd.DataFrame(records)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved: {output_path}")

    return 0


def analyze_batch(folder_path: str, extended: bool, output_dir: str, config: dict) -> int:
    """Batch analyze all CSV files in folder"""
    logger = setup_logger("Batch_Analysis")

    # Apply config settings if available
    if config and 'parameter_fitting' in config:
        fit_config = config['parameter_fitting']
        # Override extended mode if specified in config
        if 'extended_mode' in fit_config:
            extended = fit_config['extended_mode']
        # Get output dir from config if not specified
        if output_dir is None:
            output_dir = fit_config.get('output_dir', 'logs/analysis')
    else:
        if output_dir is None:
            output_dir = "logs/analysis"

    # Detect scenario group subdirectories (new format)
    scenario_groups = ['Baseline', 'Optimism', 'Authority', 'Threat', 'Stimulus', 'Magnitude', 'Punishment', 'Regret']
    subdirs = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and item in scenario_groups:
            subdirs.append((item, item_path))

    total_success = 0
    total_files = 0

    if subdirs:
        # New format: single run directory with scenario group subdirectories
        # Merge all scenario groups into one analysis
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing run directory: {os.path.basename(folder_path)}")
        logger.info(f"{'='*60}")

        all_df = pd.DataFrame()
        for group_name, group_path in subdirs:
            csv_files = [f for f in os.listdir(group_path) if f.endswith('.csv')]
            for csv_file in csv_files:
                csv_path = os.path.join(group_path, csv_file)
                df = load_csv_with_validation(csv_path)
                if df is not None:
                    all_df = pd.concat([all_df, df], ignore_index=True)
                    total_files += 1

        if all_df.empty:
            logger.warning(f"  ⚠️  No valid data")
            return 1

        # Apply config filters
        if config and 'parameter_fitting' in config:
            fit_config = config['parameter_fitting']
            filters = fit_config.get('data_filters', {})
        else:
            filters = {}

        # Preprocess
        all_df = prepare_for_fitting(all_df, filters=filters)
        logger.info(f"  Data: {len(all_df)} records, scenarios: {list(all_df['group'].unique())}")

        # Fit parameters
        try:
            results = batch_fit_cognitive_model(all_df, extended_mode=extended, config=config)

            if results:
                # Output report
                report = format_results_report(results)
                print("\n" + report)

                # Save results
                if output_dir:
                    model_name = os.path.basename(folder_path.rstrip('/'))
                    save_dir = os.path.join(output_dir, model_name)
                    os.makedirs(save_dir, exist_ok=True)
                    output_path = os.path.join(save_dir, "cognitive_report.csv")

                    records = []
                    for group, result in results.items():
                        record = {
                            'Group': group,
                            'nll': result.nll,
                            'bic': result.bic,
                            'count': result.count,
                        }
                        for param, value in result.params.items():
                            record[param] = value
                        records.append(record)

                    output_df = pd.DataFrame(records)
                    output_df.to_csv(output_path, index=False)
                    logger.info(f"  ✓ Results saved: {output_path}")

                total_success = 1
            else:
                logger.warning(f"  ⚠️  Fitting failed")
                return 1

        except Exception as e:
            logger.error(f"  ✗ Processing failed: {e}")
            return 1

    else:
        # Legacy format or batch of multiple runs
        # Find all subdirectories (assuming each is a model or run)
        all_subdirs = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                all_subdirs.append((item, item_path))

        if not all_subdirs:
            # No subdirectories, process current folder directly
            all_subdirs = [("", folder_path)]

        for model_name, subdir_path in all_subdirs:
            # Find CSV files
            csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]

            if not csv_files:
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model_name or os.path.basename(folder_path)} ({len(csv_files)} files)")
            logger.info(f"{'='*60}")

            # Merge all CSV files
            all_df = pd.DataFrame()
            for csv_file in csv_files:
                csv_path = os.path.join(subdir_path, csv_file)
                df = load_csv_with_validation(csv_path)
                if df is not None:
                    all_df = pd.concat([all_df, df], ignore_index=True)
                    total_files += 1

            if all_df.empty:
                logger.warning(f"  ⚠️  No valid data")
                continue

            # Apply config filters
            if config and 'parameter_fitting' in config:
                fit_config = config['parameter_fitting']
                filters = fit_config.get('data_filters', {})
            else:
                filters = {}

            # Preprocess
            all_df = prepare_for_fitting(all_df, filters=filters)
            logger.info(f"  Data: {len(all_df)} records, scenarios: {list(all_df['group'].unique())}")

            # Fit parameters
            try:
                results = batch_fit_cognitive_model(all_df, extended_mode=extended, config=config)

                if results:
                    # Output report
                    report = format_results_report(results)
                    print("\n" + report)

                    # Save results
                    if output_dir:
                        save_dir = os.path.join(output_dir, model_name or "batch_results")
                        os.makedirs(save_dir, exist_ok=True)
                        output_path = os.path.join(save_dir, "cognitive_report.csv")

                        records = []
                        for group, result in results.items():
                            record = {
                                'Group': group,
                                'nll': result.nll,
                                'bic': result.bic,
                                'count': result.count,
                            }
                            for param, value in result.params.items():
                                record[param] = value
                            records.append(record)

                        output_df = pd.DataFrame(records)
                        output_df.to_csv(output_path, index=False)
                        logger.info(f"  ✓ Results saved: {output_path}")

                    total_success += 1
                else:
                    logger.warning(f"  ⚠️  Fitting failed")

            except Exception as e:
                logger.error(f"  ✗ Processing failed: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Batch analysis complete: {total_success}/{total_files} files successful")
    logger.info(f"{'='*60}")

    return 0 if total_success > 0 else 1


def main(args):
    """Main function"""
    # Load configuration
    config = load_config(args.config)

    # Parameter validation
    if not args.csv_file and not args.folder:
        print("Error: Must specify --csv_file or --folder")
        return 1

    if args.csv_file and not os.path.exists(args.csv_file):
        print(f"Error: File does not exist {args.csv_file}")
        return 1

    if args.folder and not os.path.exists(args.folder):
        print(f"Error: Folder does not exist {args.folder}")
        return 1

    # Determine extended mode (CLI arg takes precedence over config)
    extended = args.extended
    if config and 'parameter_fitting' in config:
        fit_config = config['parameter_fitting']
        if 'extended_mode' in fit_config and not args.extended:
            extended = fit_config['extended_mode']

    # Execute analysis
    if args.csv_file:
        return analyze_single_file(args.csv_file, extended, args.output_dir, config)
    else:
        return analyze_batch(args.folder, extended, args.output_dir, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cognitive Parameter Fitter - Extract cognitive parameters from behavior trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 分析单个运行目录（新格式 - 按组分文件）
    python fit_params.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/

    # 分析单个文件（旧格式）
    python fit_params.py --csv_file logs/cigt/DeepSeek-R1/results.csv

    # 批量分析（多个运行）
    python fit_params.py --folder logs/cigt/DeepSeek-R1/ --output_dir logs/analysis/

    # 扩展模式（包含感知参数）
    python fit_params.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --extended

    # 使用配置文件
    python fit_params.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --config scripts/config/analysis.json
        """
    )

    parser.add_argument("--csv_file", type=str,
                        help="Single CSV file path (legacy format)")
    parser.add_argument("--folder", type=str,
                        help="Run directory with group CSVs or parent directory with multiple runs")
    parser.add_argument("--extended", action="store_true",
                        help="Use extended parameter mode (includes R_perc, lambda_LA)")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory (default: logs/analysis)")
    parser.add_argument("--config", type=str,
                        help="Path to analysis configuration file (default: scripts/config/analysis.json)")

    args = parser.parse_args()
    sys.exit(main(args))
