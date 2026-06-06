#!/bin/bash
# 运行 AG News 数据准备（M=20, Dirichlet α=0.5）
# Usage: bash run_agnews.sh [data_dir]

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_DIR_ARG=""
if [ -n "$1" ]; then
    DATA_DIR_ARG="--data-dir $1"
fi

echo "=========================================="
echo "AG News 数据准备 (M=20)"
echo "=========================================="

echo ""
echo "AG News Dirichlet(α=0.5), M=20..."
uv run scripts/data_prep.py AGNews noniid balance dir --num-clients 20 --alpha 0.5 $DATA_DIR_ARG

echo ""
echo "=========================================="
echo "✅ AG News 完成！"
echo "=========================================="
