#!/bin/bash
# CUTLASS vs cuBLASLt Benchmark 编译和运行脚本
#
# Usage:
#   ./run_benchmark.sh          # 运行 benchmark (自动编译)
#   ./run_benchmark.sh --compile # 强制重新编译后运行
#   ./run_benchmark.sh --compile-only # 仅编译不运行

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  CUTLASS vs cuBLASLt FP8 GEMM Benchmark${NC}"
echo -e "${GREEN}================================================${NC}"
echo

# 检查 CUDA 环境
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please ensure CUDA is installed.${NC}"
    exit 1
fi

echo -e "${YELLOW}CUDA Version:${NC} $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo -e "${YELLOW}PyTorch CUDA:${NC} $(python3 -c 'import torch; print(torch.version.cuda)')"
echo -e "${YELLOW}GPU:${NC} $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo

# 创建 build 目录
mkdir -p build

# 解析参数
COMPILE_FLAG=""
COMPILE_ONLY=false

for arg in "$@"; do
    case $arg in
        --compile)
            COMPILE_FLAG="--compile"
            echo -e "${YELLOW}[INFO] Force recompile enabled${NC}"
            ;;
        --compile-only)
            COMPILE_ONLY=true
            COMPILE_FLAG="--compile"
            echo -e "${YELLOW}[INFO] Compile only mode${NC}"
            ;;
    esac
done

# 运行 benchmark
if [ "$COMPILE_ONLY" = true ]; then
    echo -e "${GREEN}[INFO] Compiling CUDA extension...${NC}"
    python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from benchmark_cutlass_vs_cublaslt import load_cublaslt_extension
ext = load_cublaslt_extension(force_compile=True)
print('Compilation successful!')
"
    echo -e "${GREEN}[INFO] Compilation complete!${NC}"
else
    echo -e "${GREEN}[INFO] Running benchmark...${NC}"
    echo
    python3 benchmark_cutlass_vs_cublaslt.py $COMPILE_FLAG
fi

echo
echo -e "${GREEN}Done!${NC}"
