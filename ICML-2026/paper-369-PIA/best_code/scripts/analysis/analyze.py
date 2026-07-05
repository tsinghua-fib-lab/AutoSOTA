#!/usr/bin/env python3
"""
Unified Analysis Entry - Statistics + Visualization

Integrates functionality from original metrics_calculator.py, plot_drift.py, plot_radar.py

Usage:
    # 1. Metrics calculation (new format - run directory)
    python analyze.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --type metrics

    # 2. Drift plot (run directory)
    python analyze.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --type drift

    # 3. Radar plot (cognitive parameter reports)
    python analyze.py --folder logs/analysis/ --type radar

    # 4. All analysis
    python analyze.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --type all --output_dir logs/analysis/

    # 5. With configuration file
    python analyze.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --type all --config scripts/config/analysis.json
"""

import argparse
import os
import sys
import json
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
import_module_from_path('src.analysis', os.path.join(src_dir, 'analysis', '__init__.py'))

from src.analysis import MetricsCalculator, DriftPlotter, RadarPlotter


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


def run_metrics(folder_path: str, output_dir: str = None, config: dict = None):
    """Run metrics calculation"""
    print("\n" + "=" * 70)
    print("📊 Metrics Calculation Analysis")
    print("=" * 70)

    # Get model name
    model_name = os.path.basename(folder_path.rstrip('/'))

    # Calculate metrics
    df_details, df_summary = MetricsCalculator.calculate_from_folder(folder_path)

    if df_details.empty:
        print("No valid data")
        return

    # Print report
    MetricsCalculator.print_report(df_summary, model_name)
    MetricsCalculator.print_vulnerability_insights(df_details)

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed metrics
        details_path = os.path.join(output_dir, f"metrics_details_{model_name}.csv")
        df_details.to_csv(details_path, index=False)
        print(f"\nDetailed metrics saved: {details_path}")

        # Save summary
        summary_path = os.path.join(output_dir, f"metrics_summary_{model_name}.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Summary metrics saved: {summary_path}")


def run_drift(folder_path: str, output_dir: str = None, config: dict = None):
    """Run drift plot analysis"""
    print("\n" + "=" * 70)
    print("📈 Drift Plot Analysis")
    print("=" * 70)

    if output_dir is None:
        output_dir = "logs/images"

    # Apply config settings if available
    palette = None
    if config and 'drift_plot' in config:
        drift_config = config['drift_plot']
        output_dir = drift_config.get('output_dir', output_dir)
        palette = drift_config.get('palette')

    plotter = DriftPlotter(output_dir, palette=palette)
    model_name = os.path.basename(folder_path.rstrip('/'))
    plotter.run(folder_path, f"drift_{model_name}.png")


def run_radar(folder_path: str, output_dir: str = None, config: dict = None):
    """Run radar plot analysis"""
    print("\n" + "=" * 70)
    print("🎯 Radar Plot Analysis")
    print("=" * 70)

    # Default output directory
    if output_dir is None:
        output_dir = "logs/images"

    # Apply config settings if available
    exclude_models = []
    if config and 'radar_plot' in config:
        radar_config = config['radar_plot']
        # Config output_dir takes precedence over passed output_dir
        config_output_dir = radar_config.get('output_dir')
        if config_output_dir:
            output_dir = config_output_dir
        exclude_models = radar_config.get('exclude_models', [])

    # Auto-detect input type and find cognitive report
    # If folder_path is a jailbreak run directory, find corresponding analysis folder
    cognitive_folder = folder_path

    # Check if this is a jailbreak run directory (contains scenario subdirs)
    if os.path.isdir(folder_path):
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        scenario_groups = ['Baseline', 'Optimism', 'Authority', 'Threat', 'Stimulus', 'Magnitude', 'Punishment', 'Regret']

        # If it's a jailbreak run directory, look for analysis folder
        if any(d in scenario_groups for d in subdirs):
            # Extract run_id from path
            run_id = os.path.basename(folder_path.rstrip('/'))
            model_name = os.path.basename(os.path.dirname(folder_path.rstrip('/')))

            # Try to find corresponding analysis folder
            possible_paths = [
                os.path.join('logs', 'analysis', run_id),  # Single run format
                os.path.join('logs', 'analysis', model_name, run_id),  # Model-specific format
                os.path.join('logs', 'analysis', model_name),  # Model folder
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    # Check if it has cognitive_report.csv
                    if os.path.exists(os.path.join(path, 'cognitive_report.csv')):
                        cognitive_folder = path
                        print(f"Found cognitive report at: {cognitive_folder}")
                        break
                    # Or check if it's a model folder with subdirs
                    elif any(os.path.exists(os.path.join(path, subdir, 'cognitive_report.csv'))
                            for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))):
                        cognitive_folder = path
                        print(f"Found model analysis folder: {cognitive_folder}")
                        break

    plotter = RadarPlotter(output_dir, exclude_models=exclude_models)
    plotter.run(cognitive_folder)


def main(args):
    """Main function"""
    # Load configuration
    config = load_config(args.config)

    # Validate input
    if not os.path.exists(args.folder):
        print(f"Error: Folder does not exist - {args.folder}")
        return 1

    # Determine output directory
    output_dir = args.output_dir or args.folder

    # Execute analysis type
    analysis_type = args.type

    if analysis_type == 'metrics':
        run_metrics(args.folder, output_dir, config)

    elif analysis_type == 'drift':
        # Drift plot needs experiment result CSV folder
        run_drift(args.folder, output_dir, config)

    elif analysis_type == 'radar':
        # Radar plot needs cognitive parameter report folder
        run_radar(args.folder, output_dir, config)

    elif analysis_type == 'all':
        # Try to run all analysis
        try:
            # 1. Metrics calculation
            run_metrics(args.folder, output_dir, config)
        except Exception as e:
            print(f"\nMetrics calculation failed: {e}")

        try:
            # 2. Drift plot
            run_drift(args.folder, output_dir, config)
        except Exception as e:
            print(f"\nDrift plot failed: {e}")

        try:
            # 3. Radar plot
            run_radar(args.folder, output_dir, config)
        except Exception as e:
            print(f"\nRadar plot failed: {e}")

    else:
        print(f"Error: Unknown analysis type '{analysis_type}'")
        print("Options: metrics, drift, radar, all")
        return 1

    print("\n" + "=" * 70)
    print("✅ Analysis Complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Analysis Entry - Statistics + Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 指标计算（新格式 - 运行目录）
    python analyze.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --type metrics

    # 漂移图（运行目录）
    python analyze.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --type drift --output_dir logs/images/

    # 雷达图（认知参数报告）
    python analyze.py --folder logs/analysis/ --type radar

    # 全部分析（使用配置文件）
    python analyze.py --folder logs/cigt/DeepSeek-R1/1767059181_c30b790f/ --type all --config scripts/config/analysis.json
        """
    )

    parser.add_argument("--folder", type=str, required=True,
                        help="Run directory (new format) or analysis folder (for radar)")
    parser.add_argument("--type", type=str, required=True,
                        choices=['metrics', 'drift', 'radar', 'all'],
                        help="Analysis type")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory (default: same as input)")
    parser.add_argument("--config", type=str,
                        help="Path to analysis configuration file (default: scripts/config/analysis.json)")

    args = parser.parse_args()
    sys.exit(main(args))
