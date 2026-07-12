#!/bin/bash
# Full pipeline: train + prepare data for paper-4243
set -e

cd /repo

# Step 1: Train backdoored model (output to data_input/)
echo "=== Step 1: Training backdoored model ==="
python3 train_backdoor.py --data_dir /datasets/cifar10 --output_dir /repo/data_input --pratio 0.05 --epochs 100 --device cuda:0 2>&1 | grep -v "NFS\|Device or resource busy\|rmtree\|_rmtree_safe\|multiprocessing\|Traceback\|File \"/"

# Step 2: Prepare data (bbench_dir != output_dir to avoid wiping)
echo "=== Step 2: Preparing data splits ==="
python3 prepare_data.py --bbench_dir data_input --output_dir data --seed 0 --cifar_download_dir /datasets/cifar10 2>&1

echo "=== Data preparation complete ==="
ls -la data/
echo "---"
for d in trusted sampling clean_test bd_test; do
    count=$(find data/$d -name "*.png" 2>/dev/null | wc -l)
    echo "  $d: $count images"
done
