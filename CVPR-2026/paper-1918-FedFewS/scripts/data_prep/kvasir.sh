#!/bin/bash
# 运行 Kvasir 数据准备（仅 M=5）
# Usage: bash run_kvasir.sh [data_dir]

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_DIR_ARG=""
if [ -n "$1" ]; then
    DATA_DIR_ARG="--data-dir $1"
fi

echo "=========================================="
echo "Kvasir 数据准备 (M=5)"
echo "=========================================="

echo ""
echo "Kvasir Dirichlet(α=0.5), M=5..."
uv run scripts/data_prep.py Kvasir noniid balance dir --num-clients 5 --alpha 0.5 $DATA_DIR_ARG

echo ""
echo "=========================================="
echo "✅ Kvasir 完成！"
echo "=========================================="
