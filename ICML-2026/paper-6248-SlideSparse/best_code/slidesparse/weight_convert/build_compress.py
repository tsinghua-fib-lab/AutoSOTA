#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt Compress Extension Build Script

这是一个智能编译脚本，直接复用顶层 slidesparse.utils 中的编译工具。

文件名格式
==========
{prefix}_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.so

示例:
    cusparselt_compress_H100_cc90_py312_cu124_x86_64.so
    cusparselt_compress_RTX5080_cc120_py312_cu129_x86_64.so

使用方法
========
编译当前 GPU 架构的 .so：
    python3 build_compress.py build
    
强制重新编译：
    python3 build_compress.py build --force

查看帮助：
    python3 build_compress.py --help
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    hw_info,
    build_filename,
    find_file,
    print_system_info,
    build_cuda_extension_direct,
    clean_build_artifacts,
    CUSPARSELT_LDFLAGS,
)


# =============================================================================
# 配置
# =============================================================================

EXTENSION_PREFIX = "cusparselt_compress"
SCRIPT_DIR = Path(__file__).parent.absolute()
SOURCE_FILE = SCRIPT_DIR / "cusparselt_compress.cu"
BUILD_DIR = SCRIPT_DIR / "build"


def get_extension_name() -> str:
    """生成扩展名（不含 .so 后缀）"""
    return build_filename(EXTENSION_PREFIX, ext="")


def build_extension(force: bool = False, verbose: bool = True) -> Path:
    """
    编译 cuSPARSELt Compress 扩展（直接调用顶层 build_cuda_extension_direct）
    """
    ext_name = get_extension_name()
    
    if verbose:
        print("=" * 60)
        print("cuSPARSELt Compress Extension Builder")
        print("=" * 60)
        print(f"Extension: {ext_name}")
        print(f"Source: {SOURCE_FILE.name}")
        print(f"Build dir: {BUILD_DIR}")
        print("-" * 60)
        print(f"GPU: {hw_info.gpu_name} ({hw_info.gpu_full_name})")
        print(f"CC: {hw_info.cc_tag} ({hw_info.sm_code})")
        print(f"Python: {hw_info.python_tag}")
        print(f"CUDA: {hw_info.cuda_tag}")
        print(f"Arch: {hw_info.arch_tag}")
        print("=" * 60)
    
    # 直接调用顶层编译函数（使用直接 nvcc 编译，不依赖 PyTorch）
    return build_cuda_extension_direct(
        name=ext_name,
        source_file=SOURCE_FILE,
        build_dir=BUILD_DIR,
        extra_ldflags=CUSPARSELT_LDFLAGS,
        force=force,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build cuSPARSELt Compress extension for SlideSparse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 build_compress.py build          # 编译扩展
  python3 build_compress.py build --force  # 强制重新编译
  python3 build_compress.py info           # 显示环境信息
  python3 build_compress.py clean          # 清理构建目录
        """
    )
    parser.add_argument("command", choices=["build", "info", "clean"])
    parser.add_argument("--force", "-f", action="store_true", help="Force rebuild")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_extension(force=args.force, verbose=not args.quiet)
    
    elif args.command == "info":
        print("=" * 60)
        print("cuSPARSELt Compress Extension Info")
        print("=" * 60)
        print(f"Extension name: {get_extension_name()}.so")
        print(f"Input type: INT8 (2:4 sparse)")
        print(f"Layout: TN-CC (W^T * A, all column-major)")
        print("-" * 60)
        print_system_info()
        print("-" * 60)
        existing = find_file(EXTENSION_PREFIX, search_dir=BUILD_DIR, ext=".so")
        if existing:
            print(f"Built: {existing.name}")
        else:
            print("Built: (not found)")
    
    elif args.command == "clean":
        if BUILD_DIR.exists():
            clean_build_artifacts(BUILD_DIR, keep_extensions=[])
            print(f"✓ Cleaned {BUILD_DIR}")
        else:
            print("Build directory does not exist")


if __name__ == "__main__":
    main()
