#!/bin/bash
# 运行 Fed-ISIC2019 数据准备
# Usage: bash run_fedisic2019.sh [data_dir]

set -e  # 遇到错误立即停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo "Fed-ISIC2019 数据准备"
echo "=========================================="

echo ""
echo "Fed-ISIC2019, 6 centers (natural domain shift)..."
if [ -n "$1" ]; then
    PFLLIB_DATA_DIR="$1" uv run PFLlib/dataset/generate_FedISIC2019.py natural_6centers
else
    uv run PFLlib/dataset/generate_FedISIC2019.py natural_6centers
fi

echo ""
echo "=========================================="
echo "✅ Fed-ISIC2019 准备完成！"
echo "=========================================="
