#!/bin/bash
# 运行 TinyImageNet 数据准备（M=10, 2组）
# Usage: bash run_tinyimagenet.sh [data_dir]

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_DIR_ARG=""
if [ -n "$1" ]; then
    DATA_DIR_ARG="--data-dir $1"
fi

echo "=========================================="
echo "TinyImageNet 数据准备 (M=10, 2组)"
echo "=========================================="

echo ""
echo "[1/2] TinyImageNet Pathological, M=10 (20 classes per client)..."
uv run scripts/data_prep.py TinyImageNet noniid balance pat --num-clients 10 $DATA_DIR_ARG

echo ""
echo "[2/2] TinyImageNet Dirichlet(α=0.5), M=10..."
uv run scripts/data_prep.py TinyImageNet noniid balance dir --num-clients 10 --alpha 0.5 $DATA_DIR_ARG

echo ""
echo "=========================================="
echo "✅ TinyImageNet 全部完成！"
echo "=========================================="
