#!/usr/bin/env python3
"""
Complete Pipeline Script - Experiment Execution + Parameter Fitting

One-click completion: MAB experiment → Generate CSV → Cognitive parameter fitting → Output report

Usage:
    # Complete workflow
    python run_pipeline.py --source ollama --model mock-model --instruction "explain account recovery safety" --trials 2 --mock

    # Fit only existing data
    python run_pipeline.py --fit_only --data_path logs/results.csv

    # Mock mode test
    python run_pipeline.py --source ollama --model mock-model --mock --trials 2 --instruction "test"
"""
import argparse
import subprocess
import os
import sys
import glob
import time


def run_step(step_name: str, command: list) -> bool:
    """Run single step"""
    print(f"\n{'='*70}")
    print(f"Step: {step_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(command)}\n")

    try:
        subprocess.run(command, check=True)
        print(f"\n✓ {step_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {step_name} interrupted by user")
        return False


def find_latest_run_dir(output_dir: str, model: str) -> str:
    """Find latest experiment run directory"""
    model_safe = model.replace("/", "_")
    model_dir = os.path.join(output_dir, model_safe)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Get all timestamp_id directories
    subdirs = [d for d in os.listdir(model_dir)
               if os.path.isdir(os.path.join(model_dir, d)) and '_' in d]

    if not subdirs:
        raise FileNotFoundError(f"No run directories found in: {model_dir}")

    # Return latest directory by creation time
    latest = max(subdirs, key=lambda d: os.path.getctime(os.path.join(model_dir, d)))
    return os.path.join(model_dir, latest)


def run_full_pipeline(args) -> int:
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("🚀 Launching Complete Experiment Pipeline")
    print("="*70)
    print(f"Model: {args.source}/{args.model}")
    print(f"Instruction: {args.instruction}")
    print(f"Trials: {args.trials} rounds")
    print(f"Mode: {'Mock' if args.mock else 'Real'}")
    print("="*70)

    # Get script directory for correct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_mab_path = os.path.join(script_dir, "run_mab.py")
    fit_params_path = os.path.join(script_dir, "..", "analysis", "fit_params.py")

    # Step 1: Run experiment
    cmd1 = [
        "python", run_mab_path,
        "--source", args.source,
        "--model", args.model,
        "--instruction", args.instruction,
        "--trials", str(args.trials),
        "--output_dir", args.output_dir
    ]
    if args.mock:
        cmd1.append("--mock")

    if not run_step("MAB Experiment Execution", cmd1):
        return 1

    # Step 2: Find run directory
    print(f"\n{'='*70}")
    print("Step: Find Experiment Results")
    print(f"{'='*70}")

    try:
        run_dir = find_latest_run_dir(args.output_dir, args.model)
        print(f"Found run directory: {run_dir}")
        # List group files
        group_files = [f for f in os.listdir(run_dir) if f.endswith('.csv')]
        print(f"Group files: {', '.join(group_files)}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return 1

    # Step 3: Fit parameters (pass directory instead of file)
    cmd2 = ["python", fit_params_path, "--folder", run_dir]

    if args.extended:
        cmd2.append("--extended")

    if args.output_dir:
        analysis_dir = os.path.join(args.output_dir, "analysis")
        cmd2.extend(["--output_dir", analysis_dir])

    if not run_step("Cognitive Parameter Fitting", cmd2):
        return 1

    # Complete
    print("\n" + "="*70)
    print("🎉 Complete Pipeline Execution Successful!")
    print("="*70)
    print(f"Experiment Data: {run_dir}")
    print(f"Analysis Results: {os.path.join(args.output_dir, 'analysis')}")
    print("="*70)

    return 0


def run_fit_only(args) -> int:
    """Run parameter fitting only"""
    print("\n" + "="*70)
    print("🔧 Parameter Fitting Only")
    print("="*70)
    print(f"Data path: {args.data_path}")
    print("="*70)

    # Get script directory for correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fit_params_path = os.path.join(script_dir, "..", "analysis", "fit_params.py")

    # Check if path is directory or file
    if os.path.isdir(args.data_path):
        cmd = ["python", fit_params_path, "--folder", args.data_path]
    else:
        cmd = ["python", fit_params_path, "--csv_file", args.data_path]

    if args.extended:
        cmd.append("--extended")

    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])

    if run_step("Parameter Fitting", cmd):
        print("\n" + "="*70)
        print("✅ Fitting Complete")
        print("="*70)
        return 0
    else:
        return 1


def main(args):
    """Main function"""
    if args.fit_only:
        if not args.data_path:
            print("Error: --fit_only mode requires --data_path")
            return 1
        return run_fit_only(args)
    else:
        # Verify required parameters
        if not all([args.source, args.model, args.instruction]):
            print("Error: Complete mode requires --source, --model, --instruction")
            return 1
        return run_full_pipeline(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete Pipeline - Experiment Execution + Parameter Fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整流程（实验 + 拟合）
    python run_pipeline.py --source ollama --model mock-model --instruction "explain account recovery safety" --trials 2 --mock

    # 仅拟合已有数据
    python run_pipeline.py --fit_only --data_path logs/cigt/DeepSeek-R1/results.csv

    # Mock模式测试
    python run_pipeline.py --source ollama --model mock-model --mock --trials 2 --instruction "test"

    # 扩展参数模式
    python run_pipeline.py --source openai --model gpt-4o-mini --instruction "prompt" --trials 50 --extended
        """
    )

    # 完整流程参数
    group_full = parser.add_argument_group("完整流程参数")
    group_full.add_argument("--source", type=str,
                           help="API源 (siliconflow, ollama, etc.)")
    group_full.add_argument("--model", type=str,
                           help="模型名称")
    group_full.add_argument("--instruction", type=str,
                           help="Instruction to evaluate")
    group_full.add_argument("--trials", type=int, default=2,
                           help="试验次数")
    group_full.add_argument("--mock", action="store_true",
                           help="Mock模式")

    # 仅拟合参数
    group_fit = parser.add_argument_group("仅拟合参数")
    group_fit.add_argument("--fit_only", action="store_true",
                          help="仅运行参数拟合")
    group_fit.add_argument("--data_path", type=str,
                          help="已有数据路径")

    # 公共参数
    parser.add_argument("--output_dir", type=str, default="./logs/cigt-smoke",
                       help="输出目录")
    parser.add_argument("--extended", action="store_true",
                       help="使用扩展参数模式")

    args = parser.parse_args()
    sys.exit(main(args))
