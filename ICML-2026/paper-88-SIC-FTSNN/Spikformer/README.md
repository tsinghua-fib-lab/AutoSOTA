# ImageNet-ready Spikformer + Fragmentation package

Contents:
- `spikformer_fragmentation_addon/`: importable package
- `imagenet_train_testing.py`: one-touch ImageNet train/test entry script

Run examples:
```bash
python imagenet_train_testing.py   --data-root /path/to/imagenet   --output-dir ./runs/spikformer_imagenet   --mode train_test   --model-preset spikformer-8-512   --epochs 100   --batch-size 64   --val-batch-size 128   --amp

torchrun --standalone --nnodes=1 --nproc-per-node=8 imagenet_train_testing.py   --data-root /path/to/imagenet   --output-dir ./runs/spikformer_imagenet_ddp   --mode train_test   --model-preset spikformer-8-512   --epochs 100   --batch-size 64   --val-batch-size 128   --amp   --ddp   --use-fragmentation   --fragmentation-mode dynamic   --fragment-candidates 2 4 8   --decode entropy
```
