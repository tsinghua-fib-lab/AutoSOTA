#!/usr/bin/env python3
"""
从 HuggingFace Hub 恢复 SlideSparse 权重

与 backup_to_hf.py 对称的恢复脚本。

默认行为：
- 下载到 /root/vllm_checkpoints 和 /root/vllm_checkpoints_slidesparse
- 在项目目录创建软链接（避免 home 磁盘爆满）

Usage:
    # 恢复所有（默认用软链接）
    python3 restore_from_hf.py --repo bcacdwk/slidesparse-checkpoints
    
    # 不用软链接，直接下载到项目目录
    python3 restore_from_hf.py --repo bcacdwk/slidesparse-checkpoints --no-symlink
    
    # 只恢复指定模型
    python3 restore_from_hf.py --repo bcacdwk/slidesparse-checkpoints --filter "Llama3.2-1B"
    
    # 同时恢复原始 checkpoints（从 RedHatAI）
    python3 restore_from_hf.py --repo bcacdwk/slidesparse-checkpoints --with-base
    
    # 自定义存储路径
    python3 restore_from_hf.py --repo bcacdwk/slidesparse-checkpoints --root-dir /data
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List

# 路径配置
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# 默认路径
DEFAULT_ROOT_DIR = Path("/root")
DEFAULT_CHECKPOINTS_NAME = "vllm_checkpoints"
DEFAULT_SLIDESPARSE_NAME = "vllm_checkpoints_slidesparse"

# 项目内路径
PROJECT_CHECKPOINTS = _PROJECT_ROOT / "checkpoints"
PROJECT_SLIDESPARSE = _PROJECT_ROOT / "checkpoints_slidesparse"

# 原始模型映射（用于 --with-base）
BASE_MODELS = {
    "Llama3.2-1B-INT8": "RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8",
    "Llama3.2-1B-FP8": "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic",
    "Llama3.2-3B-INT8": "RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8",
    "Llama3.2-3B-FP8": "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic",
    "Qwen2.5-7B-INT8": "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8",
    "Qwen2.5-7B-FP8": "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic",
    "Qwen2.5-14B-INT8": "RedHatAI/Qwen2.5-14B-Instruct-quantized.w8a8",
    "Qwen2.5-14B-FP8": "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic",
}


def print_info(msg: str):
    print(f"[INFO] {msg}")

def print_success(msg: str):
    print(f"[SUCCESS] {msg}")

def print_error(msg: str):
    print(f"[ERROR] {msg}")

def print_warning(msg: str):
    print(f"[WARNING] {msg}")


def create_symlink(target: Path, link: Path) -> bool:
    """创建软链接"""
    try:
        # 如果已存在
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                existing_target = link.resolve()
                if existing_target == target.resolve():
                    print_info(f"软链接已存在且正确: {link} -> {target}")
                    return True
                else:
                    print_warning(f"软链接指向不同位置，重新创建: {link}")
                    link.unlink()
            else:
                print_warning(f"目标已存在且不是软链接: {link}")
                print_warning(f"  请手动处理或使用 --no-symlink")
                return False
        
        # 确保父目录存在
        link.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建软链接
        link.symlink_to(target)
        print_success(f"创建软链接: {link} -> {target}")
        return True
        
    except Exception as e:
        print_error(f"创建软链接失败: {e}")
        return False


def download_from_hf(
    repo_id: str,
    local_dir: Path,
    pattern: Optional[str] = None,
) -> bool:
    """从 HuggingFace 下载"""
    cmd = [
        "huggingface-cli", "download",
        repo_id,
        "--local-dir", str(local_dir),
        "--local-dir-use-symlinks", "False",
    ]
    
    if pattern:
        cmd.extend(["--include", f"{pattern}/*"])
    
    print_info(f"执行: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"下载失败: {e}")
        return False
    except Exception as e:
        print_error(f"下载异常: {e}")
        return False


def download_base_model(model_name: str, local_dir: Path) -> bool:
    """下载原始模型"""
    if model_name not in BASE_MODELS:
        print_warning(f"未知的基础模型: {model_name}")
        return False
    
    hf_repo = BASE_MODELS[model_name]
    target_dir = local_dir / model_name
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print_info(f"基础模型已存在: {model_name}")
        return True
    
    cmd = [
        "huggingface-cli", "download",
        hf_repo,
        "--local-dir", str(target_dir),
        "--local-dir-use-symlinks", "False",
    ]
    
    print_info(f"下载基础模型: {hf_repo}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print_error(f"下载基础模型失败: {e}")
        return False


def get_needed_base_models(filter_str: Optional[str] = None) -> List[str]:
    """根据 filter 确定需要哪些基础模型"""
    models = list(BASE_MODELS.keys())
    if filter_str:
        models = [m for m in models if filter_str in m]
    return models


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace Hub 恢复 SlideSparse 权重",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--repo", type=str, default="bcacdwk/slidesparse-checkpoints",
        help="HF 仓库名 (默认: bcacdwk/slidesparse-checkpoints)"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="过滤模型名（可选，如 'Llama3.2-1B'）"
    )
    parser.add_argument(
        "--no-symlink", action="store_true",
        help="不使用软链接，直接下载到项目目录"
    )
    parser.add_argument(
        "--root-dir", type=str, default=str(DEFAULT_ROOT_DIR),
        help=f"存储根目录 (默认: {DEFAULT_ROOT_DIR})"
    )
    parser.add_argument(
        "--with-base", action="store_true",
        help="同时下载原始量化模型（从 RedHatAI）"
    )
    parser.add_argument(
        "--only-base", action="store_true",
        help="只下载原始量化模型，不下载 SlideSparse"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只显示计划，不实际执行"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    use_symlink = not args.no_symlink
    
    # 确定实际存储路径
    if use_symlink:
        checkpoints_dir = root_dir / DEFAULT_CHECKPOINTS_NAME
        slidesparse_dir = root_dir / DEFAULT_SLIDESPARSE_NAME
    else:
        checkpoints_dir = PROJECT_CHECKPOINTS
        slidesparse_dir = PROJECT_SLIDESPARSE
    
    # 显示计划
    print("=" * 60)
    print("SlideSparse 权重恢复")
    print("=" * 60)
    print()
    print(f"  HF 仓库: {args.repo}")
    print(f"  过滤器: {args.filter or '(全部)'}")
    print(f"  使用软链接: {use_symlink}")
    print()
    
    if use_symlink:
        print("  存储位置:")
        print(f"    - 原始模型: {checkpoints_dir}")
        print(f"    - SlideSparse: {slidesparse_dir}")
        print()
        print("  软链接:")
        print(f"    - {PROJECT_CHECKPOINTS} -> {checkpoints_dir}")
        print(f"    - {PROJECT_SLIDESPARSE} -> {slidesparse_dir}")
    else:
        print("  存储位置 (直接):")
        print(f"    - 原始模型: {checkpoints_dir}")
        print(f"    - SlideSparse: {slidesparse_dir}")
    print()
    
    if args.with_base or args.only_base:
        needed_base = get_needed_base_models(args.filter)
        print(f"  需要下载的基础模型 ({len(needed_base)} 个):")
        for m in needed_base:
            print(f"    - {m}")
        print()
    
    if args.dry_run:
        print("[DRY RUN] 不实际执行")
        return 0
    
    # 创建目录
    print("-" * 60)
    print("Step 1: 准备目录")
    print("-" * 60)
    
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    slidesparse_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"目录已创建: {checkpoints_dir}")
    print_success(f"目录已创建: {slidesparse_dir}")
    
    # 创建软链接
    if use_symlink:
        print()
        print("-" * 60)
        print("Step 2: 创建软链接")
        print("-" * 60)
        
        # 处理已存在的目录
        for proj_dir, target_dir in [
            (PROJECT_CHECKPOINTS, checkpoints_dir),
            (PROJECT_SLIDESPARSE, slidesparse_dir),
        ]:
            if proj_dir.exists() and not proj_dir.is_symlink():
                print_warning(f"项目目录已存在: {proj_dir}")
                print_warning(f"  如果里面有数据，请先手动移动到 {target_dir}")
                print_warning(f"  然后删除 {proj_dir} 后重新运行")
                # 如果是空目录，可以直接删除
                try:
                    if not any(proj_dir.iterdir()):
                        proj_dir.rmdir()
                        print_info(f"已删除空目录: {proj_dir}")
                    else:
                        return 1
                except:
                    return 1
        
        if not create_symlink(checkpoints_dir, PROJECT_CHECKPOINTS):
            return 1
        if not create_symlink(slidesparse_dir, PROJECT_SLIDESPARSE):
            return 1
    
    # 下载基础模型
    if args.with_base or args.only_base:
        print()
        print("-" * 60)
        print("Step 3: 下载基础模型")
        print("-" * 60)
        
        needed_base = get_needed_base_models(args.filter)
        success = 0
        fail = 0
        
        for model_name in needed_base:
            print()
            if download_base_model(model_name, checkpoints_dir):
                success += 1
            else:
                fail += 1
        
        print()
        print_info(f"基础模型下载: {success} 成功, {fail} 失败")
    
    # 下载 SlideSparse 模型
    if not args.only_base:
        print()
        print("-" * 60)
        print(f"Step {'4' if args.with_base else '3'}: 下载 SlideSparse 模型")
        print("-" * 60)
        
        if download_from_hf(args.repo, slidesparse_dir, args.filter):
            print_success("SlideSparse 模型下载完成")
        else:
            print_error("SlideSparse 模型下载失败")
            return 1
    
    # 完成
    print()
    print("=" * 60)
    print("恢复完成！")
    print("=" * 60)
    print()
    print(f"  原始模型: {PROJECT_CHECKPOINTS}")
    print(f"  SlideSparse: {PROJECT_SLIDESPARSE}")
    print()
    
    # 显示磁盘使用
    try:
        import shutil
        for name, path in [("原始模型", checkpoints_dir), ("SlideSparse", slidesparse_dir)]:
            if path.exists():
                total, used, free = shutil.disk_usage(path)
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                print(f"  {name}: {size / 1024**3:.2f} GB")
    except:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
