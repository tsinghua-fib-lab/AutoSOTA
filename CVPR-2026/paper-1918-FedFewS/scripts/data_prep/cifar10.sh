#!/bin/bash
# 运行所有 CIFAR-10 数据准备（4组）
# Usage: bash run_cifar10.sh [data_dir]

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

DATA_DIR_ARG=""
if [ -n "$1" ]; then
    DATA_DIR_ARG="--data-dir $1"
fi

echo "=========================================="
echo "CIFAR-10 数据准备 (4组)"
echo "=========================================="

echo ""
echo "[1/4] CIFAR-10 Pathological, M=10..."
uv run scripts/data_prep.py Cifar10 noniid balance pat --num-clients 10 $DATA_DIR_ARG

echo ""
echo "[2/4] CIFAR-10 Pathological, M=20..."
uv run scripts/data_prep.py Cifar10 noniid balance pat --num-clients 20 $DATA_DIR_ARG

echo ""
echo "[3/4] CIFAR-10 Dirichlet(α=0.5), M=10..."
uv run scripts/data_prep.py Cifar10 noniid balance dir --num-clients 10 --alpha 0.5 $DATA_DIR_ARG

echo ""
echo "[4/4] CIFAR-10 Dirichlet(α=0.5), M=20..."
uv run scripts/data_prep.py Cifar10 noniid balance dir --num-clients 20 --alpha 0.5 $DATA_DIR_ARG

echo ""
echo "=========================================="
echo "✅ CIFAR-10 全部完成！"
echo "=========================================="
