#!/bin/bash
# 运行 FEMNIST 数据准备（M=20, Pathological）
# Usage: bash run_femnist.sh [data_dir]

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_DIR_ARG=""
if [ -n "$1" ]; then
    DATA_DIR_ARG="--data-dir $1"
fi

echo "=========================================="
echo "FEMNIST 数据准备 (M=20)"
echo "=========================================="

echo ""
echo "FEMNIST Natural (按 writer 划分), M=20, balanced..."
uv run scripts/data_prep.py FEMNIST noniid balance - --num-clients 20 $DATA_DIR_ARG

echo ""
echo "=========================================="
echo "✅ FEMNIST 完成！"
echo "=========================================="
