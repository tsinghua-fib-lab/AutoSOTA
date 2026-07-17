#!/usr/bin/env python3
"""
备份 SlideSparse 转换后的权重到 HuggingFace Hub

Usage:
    # 先登录
    huggingface-cli login
    
    # 上传所有模型（私有）
    python3 backup_to_hf.py --repo YOUR_USERNAME/slidesparse-checkpoints
    
    # 上传为公开仓库
    python3 backup_to_hf.py --repo YOUR_USERNAME/slidesparse-checkpoints --public
    
    # 只上传指定模型
    python3 backup_to_hf.py --repo YOUR_USERNAME/slidesparse-checkpoints --filter "Llama3.2-1B"
    
    # 下载恢复
    huggingface-cli download YOUR_USERNAME/slidesparse-checkpoints --local-dir /root/vllmbench/checkpoints_slidesparse
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

# 默认路径
DEFAULT_CHECKPOINTS_DIR = Path("/root/vllmbench/checkpoints_slidesparse")
CHECKPOINTS_DIR = Path("/root/vllmbench/checkpoints")
CHECKPOINTS_SLIDESPARSE_DIR = Path("/root/vllmbench/checkpoints_slidesparse")

# README 模板（开源时使用）
README_TEMPLATE = """---
license: apache-2.0
tags:
- slidesparse
- sparse
- quantization
- int8
- fp8
- llama
- qwen
---

# SlideSparse Checkpoints

Pre-converted sparse model checkpoints using the **SlideSparse** technique.

## Overview

This repository contains model weights converted with various sparsity configurations:
- **2:4** - Standard N:M sparsity (50% sparse)
- **2:6** - Extended sparsity (67% sparse)
- **2:8** - Higher sparsity (75% sparse)  
- **2:10** - Maximum sparsity (80% sparse)

## Models Included

| Base Model | Quantization | Sparsity Variants |
|------------|--------------|-------------------|
| Llama-3.2-1B | INT8, FP8 | 2:4, 2:6, 2:8, 2:10 |
| Llama-3.2-3B | INT8, FP8 | 2:4, 2:6, 2:8, 2:10 |
| Qwen2.5-7B | INT8, FP8 | 2:4, 2:6, 2:8, 2:10 |
| Qwen2.5-14B | INT8, FP8 | 2:4, 2:6, 2:8, 2:10 |

## Source Models

These checkpoints are derived from:
- [RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8)
- [RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8)
- [RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8](https://huggingface.co/RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8)
- [RedHatAI/Qwen2.5-14B-Instruct-quantized.w8a8](https://huggingface.co/RedHatAI/Qwen2.5-14B-Instruct-quantized.w8a8)

## License

- **Qwen models**: Apache 2.0
- **Llama models**: Please refer to [Meta's Llama license](https://llama.meta.com/llama3/license/)

## Usage

```bash
# Download all checkpoints
huggingface-cli download {repo_id} --local-dir ./checkpoints_slidesparse

# Download specific model
huggingface-cli download {repo_id} Llama3.2-1B-INT8-SlideSparse-2_4 --local-dir ./checkpoints_slidesparse/Llama3.2-1B-INT8-SlideSparse-2_4
```

## Citation

If you use these checkpoints, please cite the SlideSparse paper (coming soon).
"""


def get_model_dirs(base_dir: Path, filter_str: str = None):
    """获取要上传的模型目录"""
    dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if filter_str:
        dirs = [d for d in dirs if filter_str in d.name]
    return dirs


def upload_readme(api: HfApi, repo_id: str, base_dir: Path):
    """上传 README.md"""
    readme_content = README_TEMPLATE.format(repo_id=repo_id)
    readme_path = base_dir / "README.md"
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    try:
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add README.md",
        )
        print("✓ README.md 已上传")
    except Exception as e:
        print(f"⚠ README.md 上传失败: {e}")
    finally:
        readme_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="备份权重到 HuggingFace Hub")
    parser.add_argument("--repo", type=str, required=True, 
                        help="HF 仓库名，格式: USERNAME/repo-name")
    parser.add_argument("--filter", type=str, default=None,
                        help="过滤模型名（可选）")
    parser.add_argument("--dir", type=str, default=None,
                        help="指定 checkpoints 目录（默认: checkpoints_slidesparse）")
    parser.add_argument("--include-base", action="store_true",
                        help="同时包含 checkpoints/ 下的基础模型")
    parser.add_argument("--public", action="store_true", default=False,
                        help="创建公开仓库（默认私有）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示要上传的内容，不实际上传")
    parser.add_argument("--skip-readme", action="store_true",
                        help="跳过 README 上传")
    args = parser.parse_args()
    
    api = HfApi()
    is_private = not args.public
    
    # 确定要扫描的目录
    if args.dir:
        base_dirs = [Path(args.dir)]
    else:
        base_dirs = [CHECKPOINTS_SLIDESPARSE_DIR]
        if args.include_base:
            base_dirs.insert(0, CHECKPOINTS_DIR)
    
    # 检查登录状态
    try:
        user_info = api.whoami()
        print(f"✓ 已登录为: {user_info['name']}")
    except Exception as e:
        print(f"✗ 未登录，请先运行: huggingface-cli login")
        return 1
    
    # 获取要上传的目录（从所有 base_dirs 收集）
    model_dirs = []
    for base_dir in base_dirs:
        if base_dir.exists():
            model_dirs.extend(get_model_dirs(base_dir, args.filter))
    
    # 去重并排序
    model_dirs = sorted(set(model_dirs), key=lambda x: x.name)
    
    if not model_dirs:
        print("✗ 没有找到要上传的模型目录")
        return 1
    
    print(f"\n要上传的模型 ({len(model_dirs)} 个):")
    total_size = 0
    for d in model_dirs:
        size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
        total_size += size
        print(f"  - {d.name} ({size / 1024**3:.2f} GB)")
    print(f"\n总大小: {total_size / 1024**3:.2f} GB")
    print(f"仓库类型: {'公开' if args.public else '私有'}")
    
    if args.dry_run:
        print("\n[DRY RUN] 不实际上传")
        return 0
    
    # 创建仓库
    print(f"\n创建仓库: {args.repo} (private={is_private})")
    try:
        create_repo(args.repo, repo_type="model", private=is_private, exist_ok=True)
        print(f"✓ 仓库已创建/存在: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"✗ 创建仓库失败: {e}")
        return 1
    
    # 上传 README
    if not args.skip_readme:
        upload_readme(api, args.repo, base_dirs[0])
    
    # 上传每个模型目录
    print(f"\n开始上传模型...")
    success_count = 0
    fail_count = 0
    
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"\n[{i}/{len(model_dirs)}] 上传: {model_dir.name}")
        try:
            upload_folder(
                folder_path=str(model_dir),
                repo_id=args.repo,
                path_in_repo=model_dir.name,  # 保持目录结构
                repo_type="model",
                commit_message=f"Upload {model_dir.name}",
            )
            print(f"  ✓ 上传成功")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 上传失败: {e}")
            fail_count += 1
    
    # 总结
    print(f"\n" + "=" * 50)
    print(f"上传完成: {success_count} 成功, {fail_count} 失败")
    print(f"仓库地址: https://huggingface.co/{args.repo}")
    print(f"\n恢复命令:")
    print(f"  huggingface-cli download {args.repo} --local-dir /root/vllmbench/checkpoints_slidesparse")
    print("=" * 50)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
