#!/usr/bin/env python
"""
PFLlib Experiment Runner
运行 PFLlib 基线实验的脚本

使用方法:
    uv run experiments/run_pfllib.py config.yaml
    uv run experiments/run_pfllib.py config.yaml --data-root /path/to/data
    uv run experiments/run_pfllib.py config.yaml --model-dir ./models --result-dir ./results
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml


def load_yaml(config_path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并配置字典（override 覆盖 base）

    Args:
        base: 基础配置字典
        override: 覆盖配置字典

    Returns:
        合并后的配置字典
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = merge_configs(result[key], value)
        else:
            # 直接覆盖
            result[key] = value

    return result


def load_config_with_base(
    config_path: Path,
    base_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    加载配置并自动查找/合并 base.yaml

    配置继承逻辑：
    1. 如果配置在 algorithms/ 子目录，自动查找上级的 base.yaml
    2. 否则查找同级目录的 base.yaml
    3. 如果显式指定了 base_path，使用指定的 base 配置
    4. 如果找不到 base.yaml，直接返回原配置

    Args:
        config_path: 算法配置文件路径
        base_path: 基础配置文件路径（可选，显式指定）

    Returns:
        合并后的配置字典
    """
    # 加载算法配置
    config = load_yaml(config_path)

    # 查找 base.yaml
    if base_path is None:
        # 自动查找：如果在 algorithms/ 或 vary_k/ 等子目录，查找上级的 base.yaml
        parent_base = config_path.parent.parent / "base.yaml"
        sibling_base = config_path.parent / "base.yaml"

        # 优先查找上级目录的 base.yaml（适用于 algorithms/, vary_k/ 等子目录）
        if parent_base.exists():
            base_path = parent_base
        else:
            # 否则查找同级目录的 base.yaml
            base_path = sibling_base

    # 如果找到 base.yaml，合并配置
    if base_path.exists():
        print(f"Found base configuration: {base_path}")
        base_config = load_yaml(base_path)
        merged_config = merge_configs(base_config, config)
        print(f"Configuration merged: base + {config_path.name}")
        return merged_config
    else:
        # 没有 base 配置，直接返回原配置
        return config


def build_pfllib_command(
    config: Dict[str, Any],
    data_root: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    result_dir: Optional[Path] = None,
    num_threads: Optional[int] = None
) -> List[str]:
    """
    将 YAML 配置转换为 PFLlib 命令行参数

    Args:
        config: YAML 配置字典
        data_root: 数据根目录（覆盖配置文件中的值）
        model_dir: 模型保存目录（覆盖配置文件中的值）
        result_dir: 结果保存目录（覆盖配置文件中的值）

    Returns:
        命令行参数列表
    """
    # 获取 PFLlib 的 main.py 路径
    script_dir = Path(__file__).parent.parent  # code/
    main_py = script_dir / "PFLlib" / "system" / "main.py"

    if not main_py.exists():
        raise FileNotFoundError(f"PFLlib main.py not found at {main_py}")

    # 构建基础命令
    cmd = [sys.executable, str(main_py)]

    # ===== 必需参数 =====
    # Dataset
    dataset_config = config.get("dataset", {})

    # 构建完整的dataset路径（包含config子目录）
    dataset_name = dataset_config.get("name", "Cifar10")
    if dataset_config.get("config"):
        # 如果指定了config，将其附加到dataset名称
        # 例如：Cifar10 + iid_25 -> Cifar10/iid_25
        dataset_path = f"{dataset_name}/{dataset_config['config']}"
    else:
        dataset_path = dataset_name

    cmd.extend(["-data", dataset_path])
    cmd.extend(["-ncl", str(dataset_config.get("num_classes", 10))])
    cmd.extend(["-nc", str(dataset_config.get("num_clients", 20))])

    # Model
    model_config = config.get("model", {})
    cmd.extend(["-m", model_config.get("name", "CNN")])

    # Model-specific parameters (for text models)
    if "vocab_size" in model_config:
        cmd.extend(["-vs", str(model_config["vocab_size"])])
    if "max_len" in model_config:
        cmd.extend(["-ml", str(model_config["max_len"])])
    if "feature_dim" in model_config:
        cmd.extend(["-fd", str(model_config["feature_dim"])])

    # Algorithm
    algorithm_config = config.get("algorithm", {})
    cmd.extend(["-algo", algorithm_config.get("name", "FedAvg")])

    # Training parameters
    training_config = config.get("training", {})
    cmd.extend(["-gr", str(training_config.get("global_rounds", 100))])
    cmd.extend(["-ls", str(training_config.get("local_epochs", 5))])
    cmd.extend(["-lbs", str(training_config.get("batch_size", 10))])
    cmd.extend(["-lr", str(training_config.get("learning_rate", 0.005))])
    cmd.extend(["-jr", str(training_config.get("join_ratio", 1.0))])

    # Device
    device_config = config.get("device", {})
    cmd.extend(["-dev", device_config.get("type", "cuda")])
    cmd.extend(["-did", str(device_config.get("id", "0"))])

    # Experiment
    experiment_config = config.get("experiment", {})
    cmd.extend(["-go", experiment_config.get("goal", "test")])
    cmd.extend(["-t", str(experiment_config.get("times", 1))])

    # ===== 路径配置 =====
    paths_config = config.get("paths", {})

    # 获取项目根目录（code/）
    script_dir = Path(__file__).parent.parent  # code/

    # Model directory（优先级：命令行 > 配置文件 > 默认）
    if model_dir is not None:
        final_model_dir = model_dir.absolute()
    elif paths_config.get("model_dir"):
        final_model_dir = Path(paths_config["model_dir"]).absolute()
    else:
        # 默认：code/models/
        final_model_dir = script_dir / "models"

    # 确保目录存在
    final_model_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["-mdir", str(final_model_dir)])

    # Result directory（优先级：命令行 > 配置文件 > 默认）
    if result_dir is not None:
        final_result_dir = result_dir.absolute()
    elif paths_config.get("result_dir"):
        final_result_dir = Path(paths_config["result_dir"]).absolute()
    else:
        # 默认：code/results/
        final_result_dir = script_dir / "results"

    # 确保目录存在
    final_result_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["-rdir", str(final_result_dir)])

    # ===== 算法特定参数 =====
    algo_params = algorithm_config.get("params", {})

    # Common algorithm parameters
    if "beta" in algo_params:
        cmd.extend(["-bt", str(algo_params["beta"])])
    if "lamda" in algo_params:
        cmd.extend(["-lam", str(algo_params["lamda"])])
    if "mu" in algo_params:
        cmd.extend(["-mu", str(algo_params["mu"])])
    if "alpha" in algo_params:
        cmd.extend(["-al", str(algo_params["alpha"])])
    if "tau" in algo_params:
        cmd.extend(["-tau", str(algo_params["tau"])])

    # FedRep / Ditto
    if "plocal_epochs" in algo_params:
        cmd.extend(["-pls", str(algo_params["plocal_epochs"])])

    # pFedMe
    if "K" in algo_params:
        cmd.extend(["-K", str(algo_params["K"])])
    if "p_learning_rate" in algo_params:
        cmd.extend(["-lrp", str(algo_params["p_learning_rate"])])

    # FedBABU
    if "fine_tuning_epochs" in algo_params:
        cmd.extend(["-fte", str(algo_params["fine_tuning_epochs"])])

    # FedFew (Few-for-Many Federated Learning)
    if "num_server_models" in algo_params:
        cmd.extend(["-nsm", str(algo_params["num_server_models"])])
    if "smooth_mu" in algo_params:
        cmd.extend(["-smu", str(algo_params["smooth_mu"])])
    if "use_rep_mode" in algo_params:
        cmd.extend(["-rep", str(algo_params["use_rep_mode"])])

    # FedFomo
    if "M" in algo_params:
        cmd.extend(["-M", str(algo_params["M"])])

    # FedAMP
    if "alphaK" in algo_params:
        cmd.extend(["-alk", str(algo_params["alphaK"])])
    if "sigma" in algo_params:
        cmd.extend(["-sg", str(algo_params["sigma"])])

    # FedGen
    if "noise_dim" in algo_params:
        cmd.extend(["-nd", str(algo_params["noise_dim"])])
    if "generator_learning_rate" in algo_params:
        cmd.extend(["-glr", str(algo_params["generator_learning_rate"])])
    if "hidden_dim" in algo_params:
        cmd.extend(["-hd", str(algo_params["hidden_dim"])])
    if "server_epochs" in algo_params:
        cmd.extend(["-se", str(algo_params["server_epochs"])])
    if "localize_feature_extractor" in algo_params:
        cmd.extend(["-lf", str(algo_params["localize_feature_extractor"])])

    # FedMTL
    if "itk" in algo_params:
        cmd.extend(["-itk", str(algo_params["itk"])])

    # ===== 可选参数 =====
    # Data caching (默认 True，可通过 training.use_cache 配置)
    use_cache = training_config.get("use_cache", True)
    if not use_cache:  # 只有显式设置为 False 时才传递
        cmd.extend(["-uc", "False"])

    # Learning rate decay
    if training_config.get("learning_rate_decay", False):
        cmd.extend(["-ld", "True"])
        if "learning_rate_decay_gamma" in training_config:
            cmd.extend(["-ldg", str(training_config["learning_rate_decay_gamma"])])

    # Evaluation gap
    if "eval_gap" in training_config:
        cmd.extend(["-eg", str(training_config["eval_gap"])])

    # Auto break
    if training_config.get("auto_break", False):
        cmd.extend(["-ab", "True"])
        if "top_cnt" in training_config:
            cmd.extend(["-tc", str(training_config["top_cnt"])])

    # Number of threads for parallel client training (优先级：命令行 > 配置文件 > 默认1)
    if num_threads is not None:
        cmd.extend(["-nth", str(num_threads)])
    elif "num_threads" in training_config:
        cmd.extend(["-nth", str(training_config["num_threads"])])

    return cmd


def setup_environment(
    config: Dict[str, Any],
    data_root: Optional[Path] = None
) -> Dict[str, str]:
    """
    设置环境变量

    Args:
        config: YAML 配置字典
        data_root: 数据根目录（覆盖配置文件中的值）

    Returns:
        环境变量字典
    """
    env = os.environ.copy()

    # 设置 PFLLIB_DATA_DIR 环境变量
    paths_config = config.get("paths", {})
    dataset_config = config.get("dataset", {})

    if data_root is not None:
        env["PFLLIB_DATA_DIR"] = str(data_root.absolute())
    elif paths_config.get("data_root"):
        env["PFLLIB_DATA_DIR"] = str(Path(paths_config["data_root"]).absolute())
    else:
        # 默认：code/PFLlib/dataset
        script_dir = Path(__file__).parent.parent  # code/
        default_data_root = script_dir / "PFLlib" / "dataset"
        env["PFLLIB_DATA_DIR"] = str(default_data_root.absolute())

    # 检查数据配置目录是否存在（可选验证）
    if "PFLLIB_DATA_DIR" in env and dataset_config.get("config"):
        base_dir = Path(env["PFLLIB_DATA_DIR"])
        dataset_name = dataset_config.get("name", "Cifar10")
        config_name = dataset_config["config"]

        # 检查数据配置目录是否存在
        data_config_dir = base_dir / dataset_name / config_name
        if not data_config_dir.exists():
            print(f"Warning: Data configuration directory not found: {data_config_dir}")
            print(f"Please generate data first using data_prep scripts.")

    return env


def run_experiment(
    config_path: Path,
    base_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    result_dir: Optional[Path] = None,
    num_threads: Optional[int] = None,
    dry_run: bool = False
) -> int:
    """
    运行 PFLlib 实验

    Args:
        config_path: YAML 配置文件路径
        base_path: 基础配置文件路径（可选，用于配置继承）
        data_root: 数据根目录（可选）
        model_dir: 模型保存目录（可选）
        result_dir: 结果保存目录（可选）
        num_threads: 并行训练线程数（可选）
        dry_run: 如果为 True，只打印命令不执行

    Returns:
        退出码（0 表示成功）
    """
    # 加载配置（支持继承）
    print(f"Loading configuration from: {config_path}")
    config = load_config_with_base(config_path, base_path)

    # 打印实验信息
    experiment_config = config.get("experiment", {})
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_config.get('name', 'Unnamed')}")
    print(f"Goal: {experiment_config.get('goal', 'test')}")
    print(f"{'='*60}\n")

    # 构建命令
    cmd = build_pfllib_command(config, data_root, model_dir, result_dir, num_threads)

    # 设置环境变量
    env = setup_environment(config, data_root)

    # 打印命令
    print("Command:")
    print(" ".join(cmd))
    print()

    if env.get("PFLLIB_DATA_DIR"):
        print(f"Data root: {env['PFLLIB_DATA_DIR']}")

    # 打印数据配置信息
    dataset_config = config.get("dataset", {})
    if dataset_config.get("config"):
        print(f"Dataset: {dataset_config.get('name')} / {dataset_config['config']}")

    print()

    if dry_run:
        print("Dry run mode - command not executed")
        return 0

    # 执行命令
    print("Starting experiment...")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            cwd=Path(__file__).parent.parent  # Run from code/ directory
        )
        print("-" * 60)
        print("Experiment completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"Experiment failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        return 130


def main():
    parser = argparse.ArgumentParser(
        description="Run PFLlib baseline experiments from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with automatic base.yaml inheritance (recommended)
  uv run scripts/run_pfllib.py configs/agnews/noniid_dir_20_a0p5/algorithms/apfl.yaml

  # Run with explicit base configuration
  uv run scripts/run_pfllib.py configs/agnews/noniid_dir_20_a0p5/algorithms/apfl.yaml \\
      --base configs/agnews/noniid_dir_20_a0p5/base.yaml

  # Override data root directory
  uv run scripts/run_pfllib.py configs/cifar10_fedavg.yaml \\
      --data-root external/PFLlib/dataset

  # Override all paths
  uv run scripts/run_pfllib.py configs/cifar10_fedavg.yaml \\
      --data-root external/PFLlib/dataset \\
      --model-dir ./models/cifar10 \\
      --result-dir ./results/cifar10

  # Use parallel training with 4 threads
  uv run scripts/run_pfllib.py configs/cifar10_fedavg.yaml -nth 4

  # Dry run (print command without executing)
  uv run scripts/run_pfllib.py configs/cifar10_fedavg.yaml --dry-run
        """
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--base",
        type=Path,
        help="Path to base YAML configuration file (for config inheritance). "
             "If not specified, automatically looks for base.yaml in parent directory."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override data root directory (sets PFLLIB_DATA_DIR)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Override model save directory"
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        help="Override result save directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )
    parser.add_argument(
        "-nth", "--num-threads",
        type=int,
        help="Override number of threads for parallel client training (default: use value from config or 1)"
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    # 检查 base 配置文件（如果指定）
    if args.base and not args.base.exists():
        print(f"Error: Base configuration file not found: {args.base}")
        return 1

    # 运行实验
    return run_experiment(
        config_path=args.config,
        base_path=args.base,
        data_root=args.data_root,
        model_dir=args.model_dir,
        result_dir=args.result_dir,
        num_threads=args.num_threads,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())
