#!/usr/bin/env python
"""
数据准备工具 - 统一封装 PFLlib 数据生成脚本

使用方法:
    # 默认保存到 PFLlib/dataset/
    uv run scripts/data_prep.py MNIST noniid - dir --num-clients 20 --alpha 0.5

    # 自定义数据目录
    uv run scripts/data_prep.py Cifar10 noniid - dir --num-clients 50 --data-dir ./my_data

环境变量:
    PFLLIB_DATA_DIR - 数据集保存根目录（可通过 --data-dir 参数或环境变量设置）
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


# 获取项目根目录和关键路径
CODE_ROOT = Path(__file__).parent.parent.absolute()
DATA_ROOT = CODE_ROOT / "data"
PFLLIB_DATASET = CODE_ROOT / "PFLlib" / "dataset"


# 数据集配置映射
DATASET_CONFIGS = {
    "MNIST": {
        "script": "generate_MNIST.py",
        "default_clients": 20,
        "output_dir": "MNIST",
    },
    "CIFAR10": {
        "script": "generate_Cifar10.py",
        "default_clients": 20,
        "output_dir": "Cifar10",
    },
    "CIFAR100": {
        "script": "generate_Cifar100.py",
        "default_clients": 20,
        "output_dir": "Cifar100",
    },
    "Cifar10": {  # 别名
        "script": "generate_Cifar10.py",
        "default_clients": 20,
        "output_dir": "Cifar10",
    },
    "Cifar100": {  # 别名
        "script": "generate_Cifar100.py",
        "default_clients": 20,
        "output_dir": "Cifar100",
    },
    "DomainNet": {
        "script": "generate_DomainNet.py",
        "default_clients": 60,
        "output_dir": "DomainNet",
    },
    "Digit5": {
        "script": "generate_Digit5.py",
        "default_clients": 50,
        "output_dir": "Digit5",
    },
    "kvasir": {
        "script": "generate_kvasir.py",
        "default_clients": 50,
        "output_dir": "kvasir",
    },
    "Camelyon17": {
        "script": "generate_Camelyon17.py",
        "default_clients": 5,
        "output_dir": "Camelyon17",
    },
    "TinyImageNet": {
        "script": "generate_TinyImagenet.py",
        "default_clients": 20,
        "output_dir": "TinyImagenet",
    },
    "TinyImagenet": {  # 别名
        "script": "generate_TinyImagenet.py",
        "default_clients": 20,
        "output_dir": "TinyImagenet",
    },
    "Kvasir": {  # 别名（大写）
        "script": "generate_kvasir.py",
        "default_clients": 50,
        "output_dir": "kvasir",
    },
    "AGNews": {
        "script": "generate_AGNews.py",
        "default_clients": 10,
        "output_dir": "agnews",
    },
    "FEMNIST": {
        "script": "generate_femnist.py",
        "default_clients": 100,
        "output_dir": "femnist",
    },
}


def generate_config_name(split_type: str, partition: str, num_clients: int, alpha: float = None) -> str:
    """
    生成配置名称，用于创建独立的数据子目录

    格式:
        - IID: iid_<M>
        - Pathological: noniid_pat_<M>
        - Dirichlet: noniid_dir_<M>_a<alpha>

    Examples:
        - iid_50
        - noniid_pat_100
        - noniid_dir_50_a0.5
    """
    if split_type == "iid":
        return f"iid_{num_clients}"
    elif split_type == "noniid":
        if partition == "pat":
            return f"noniid_pat_{num_clients}"
        elif partition == "dir":
            alpha_str = f"{alpha}".replace(".", "p")  # 0.5 -> 0p5
            return f"noniid_dir_{num_clients}_a{alpha_str}"
        elif partition == "domain":
            return f"noniid_domain_{num_clients}"
        else:
            return f"noniid_{partition}_{num_clients}"
    else:
        return f"{split_type}_{num_clients}"


def prepare_data(
    dataset: str,
    split_type: str,
    balance: str,
    partition: str,
    num_clients: int = None,
    alpha: float = None,
    data_dir: str = None,
    **kwargs
):
    """
    统一的数据准备接口

    Args:
        dataset: 数据集名称 (MNIST, Cifar10, Cifar100, DomainNet, etc.)
        split_type: 'iid' 或 'noniid'
        balance: 'balance' 或 '-'
        partition: 分区方法 ('dir' for Dirichlet, 'pat' for pathological, 'domain', etc.)
        num_clients: 客户端数量（可选，使用默认值）
        alpha: Dirichlet 参数（可选）
        data_dir: 数据保存目录（可选，默认为 code/data/）
        **kwargs: 其他参数
    """
    # 检查数据集是否支持
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset]
    script_path = PFLLIB_DATASET / config["script"]

    if not script_path.exists():
        raise FileNotFoundError(f"Generate script not found: {script_path}")

    # 设置客户端数量
    if num_clients is None:
        num_clients = config["default_clients"]

    # 生成配置名称（用于创建独立子目录）
    config_name = generate_config_name(split_type, partition, num_clients, alpha)

    # 设置数据目录
    # 默认保存到 code/PFLlib/dataset/<dataset>/
    # 如果用户指定了 data_dir，则使用用户指定的位置
    if data_dir is None:
        # 默认：使用 PFLlib dataset 目录
        data_root = PFLLIB_DATASET
    else:
        # 用户指定：使用自定义目录
        data_root = Path(data_dir).absolute()

    # 确保数据根目录存在
    data_root.mkdir(parents=True, exist_ok=True)

    # 最终数据会保存在 data_root/<output_dir>/<config_name>/
    final_data_dir = data_root / config["output_dir"] / config_name

    # 构建命令（现在包含 num_clients 和 config_name）
    cmd = [
        sys.executable,
        str(script_path),
        split_type,
        balance,
        partition,
        str(num_clients),      # 传递客户端数量
        config_name,           # 传递配置名称
    ]

    # 添加额外参数（如果脚本支持）
    if alpha is not None:
        cmd.extend(["-a", str(alpha)])

    # 设置环境变量，传递数据目录给生成脚本
    # 统一使用 PFLLIB_DATA_DIR 环境变量
    env = os.environ.copy()
    env["PFLLIB_DATA_DIR"] = str(data_root)

    print(f"\n{'='*80}")
    print(f"准备数据集: {dataset}")
    print(f"配置名称: {config_name}")
    print(f"分区方式: {split_type} / {partition}")
    print(f"客户端数量: {num_clients}")
    if alpha is not None:
        print(f"Dirichlet α: {alpha}")
    print(f"数据根目录: {data_root}")
    print(f"最终位置: {final_data_dir}")
    print(f"{'='*80}\n")

    print(f"执行命令: {' '.join(cmd)}")
    print(f"环境变量: PFLLIB_DATA_DIR={data_root}\n")

    # 通过环境变量传递数据目录，不需要切换工作目录
    try:
        result = subprocess.run(cmd, env=env, check=True)
        print(f"\n✅ 数据集 {dataset} 生成成功！")
        print(f"📁 数据位置: {final_data_dir}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 数据集生成失败！错误码: {e.returncode}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description="PFLlib 数据准备工具 - 统一使用 PFLLIB_DATA_DIR 环境变量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # MNIST non-IID Dirichlet (默认保存到 PFLlib/dataset/)
  %(prog)s MNIST noniid - dir --num-clients 20 --alpha 0.5

  # CIFAR-10 IID
  %(prog)s Cifar10 iid - - --num-clients 50

  # CIFAR-10 non-IID Dirichlet with custom data directory
  %(prog)s Cifar10 noniid - dir --num-clients 100 --alpha 0.5 --data-dir ./my_datasets

  # DomainNet
  %(prog)s DomainNet noniid - domain --num-clients 60

环境变量:
  PFLLIB_DATA_DIR - 通过此环境变量统一控制所有 PFLlib 脚本的数据保存位置
        """
    )

    parser.add_argument("dataset", type=str,
                       help="数据集名称 (MNIST, Cifar10, Cifar100, DomainNet, etc.)")
    parser.add_argument("split_type", type=str,
                       help="分割类型: 'iid' 或 'noniid'")
    parser.add_argument("balance", type=str,
                       help="平衡性: 'balance' 或 '-'")
    parser.add_argument("partition", type=str,
                       help="分区方法: 'dir' (Dirichlet), 'pat' (pathological), 'domain', 或 '-'")
    parser.add_argument("--num-clients", type=int, default=None,
                       help="客户端数量（默认值因数据集而异）")
    parser.add_argument("--alpha", "-a", type=float, default=None,
                       help="Dirichlet 分布参数 α（用于 dir 分区）")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="数据保存根目录（默认为 PFLlib/dataset/，通过 PFLLIB_DATA_DIR 环境变量传递）")

    args = parser.parse_args()

    # 调用准备函数
    return prepare_data(
        dataset=args.dataset,
        split_type=args.split_type,
        balance=args.balance,
        partition=args.partition,
        num_clients=args.num_clients,
        alpha=args.alpha,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
