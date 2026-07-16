sleep 2h

CUDA_VISIBLE_DEVICES=6 python3 tools/neighbor_based_ic_evaluator.py \
  --mode neighbor \
  --neighbors /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl_metaclip_image_neighbors.jsonl \
  --predictions /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl-8b_20260119_105232.json \
  --output outputs/image_classification/ImageNet-1k_sailvl_metaclip_neighbor_scores.json \
  --model sailvl-8b \
  --dataset imagenet-1k \
  --shard 2 --num-shards 4 &

# Terminal 4 - GPU 7, processing shard 3
CUDA_VISIBLE_DEVICES=7 python3 tools/neighbor_based_ic_evaluator.py \
  --mode neighbor \
  --neighbors /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl_metaclip_image_neighbors.jsonl \
  --predictions /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl-8b_20260119_105232.json \
  --output outputs/image_classification/ImageNet-1k_sailvl_metaclip_neighbor_scores.json \
  --model sailvl-8b \
  --dataset imagenet-1k \
  --shard 3 --num-shards 4 &

# Terminal 1 - GPU 4, processing shard 0
CUDA_VISIBLE_DEVICES=0 python3 tools/neighbor_based_ic_evaluator.py \
  --mode neighbor \
  --neighbors /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl_sailvl_image_neighbors.jsonl \
  --predictions /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl-8b_20260119_105232.json \
  --output outputs/image_classification/ImageNet-1k_sailvl_sailvl_neighbor_scores.json \
  --model sailvl-8b \
  --dataset imagenet-1k \
  --shard 0 --num-shards 4 &

# Terminal 2 - GPU 5, processing shard 1
CUDA_VISIBLE_DEVICES=1 python3 tools/neighbor_based_ic_evaluator.py \
  --mode neighbor \
  --neighbors /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl_sailvl_image_neighbors.jsonl \
  --predictions /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl-8b_20260119_105232.json \
  --output outputs/image_classification/ImageNet-1k_sailvl_sailvl_neighbor_scores.json \
  --model sailvl-8b \
  --dataset imagenet-1k \
  --shard 1 --num-shards 4 &

# Terminal 3 - GPU 6, processing shard 2
CUDA_VISIBLE_DEVICES=2 python3 tools/neighbor_based_ic_evaluator.py \
  --mode neighbor \
  --neighbors /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl_sailvl_image_neighbors.jsonl \
  --predictions /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl-8b_20260119_105232.json \
  --output outputs/image_classification/ImageNet-1k_sailvl_sailvl_neighbor_scores.json \
  --model sailvl-8b \
  --dataset imagenet-1k \
  --shard 2 --num-shards 4 &

# Terminal 4 - GPU 7, processing shard 3
CUDA_VISIBLE_DEVICES=3 python3 tools/neighbor_based_ic_evaluator.py \
  --mode neighbor \
  --neighbors /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl_sailvl_image_neighbors.jsonl \
  --predictions /mnt/data1/SAGE/outputs/image_classification/ImageNet-1k_sailvl-8b_20260119_105232.json \
  --output outputs/image_classification/ImageNet-1k_sailvl_sailvl_neighbor_scores.json \
  --model sailvl-8b \
  --dataset imagenet-1k \
  --shard 3 --num-shards 4 &

# Wait for all background tasks to complete
wait
echo "All tasks completed!"