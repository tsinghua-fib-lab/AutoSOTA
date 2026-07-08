# Code Analysis — Paper 1699 SOTA Optimization

## Evaluation Path
- **Command**: `cd /repo/deep-network && export ML_DATA=/datasets && python main.py --data=emnist_merge --algo=sgd_noamp --epochs=20 --batch_size=500 --learning_rate=4.0 --momentum=0 --l2_norm_clip=1.0 --noise_multiplier=0 --dp_dataloader=True --restart=1 --tree_completion=True --effi_noise=False --dir=runs --run=1`
- **Timeout**: 20 minutes
- **Expected output**: Final test accuracy printed as `Final Accuracy: train=X%, test=Y%`

## Train / Inference Path
- `main.py`: Entry point, training loop, evaluation
  - Lines 279-309: `_clip_and_add_noise()` — per-sample gradient clipping + noise injection
  - Lines 314-404: `train_loop()` — per-epoch training
  - Lines 407-419: `test()` — final evaluation on train+test sets
  - Lines 445-449: `GradSampleModule` wrapping (Opacus) — applied even when noise_multiplier=0
  - Line 488: `torch.optim.SGD(model.parameters(), lr=lr, momentum=FLAGS.momentum)` — optimizer
  - Line 318: `CrossEntropyLoss(reduction=sum)` — loss function

## Config Path
- All hyperparameters via `absl.flags` in `main.py` (lines 35-64)
- Key flags: `data`, `algo`, `noise_multiplier`, `l2_norm_clip`, `learning_rate`, `batch_size`, `epochs`, `momentum`, `dp_dataloader`, `restart`, `tree_completion`, `effi_noise`, `run`

## Model Architecture
- `nn.py`: SMALL_NN for EMNIST/MNIST, VGG for CIFAR-10
  - SMALL_NN: Conv2d(1,16,8,2,3) → tanh → MaxPool(2,1) → Conv2d(16,32,4,2) → tanh → MaxPool(2,1) → FC(512,32) → tanh → FC(32,nclass)
  - No BN, no dropout, tanh activations

## Metric Parser
- Parse stdout for `Final Accuracy: train=X%, test=Y%` (grep)
- Or read `runs/emnist_merge/results.jsonl` → `accuracy_test` field (decimal)
- Or read `runs/results.jsonl` for the appending JSONL

## Data
- EMNIST ByMerge cached at `/datasets/TFDS/emnist/bymerge/3.0.0/`
- Loaded via `tfds.load()` in `data.py`
- Normalized: `image / 127.5 - 1` → range [-1, 1]
- Trainset: ~697932 samples, Testset: ~116323 samples, 47 classes

## Reusable Resources
- `/datasets/TFDS/emnist/bymerge/3.0.0/` — cached TFRecords (362MB)
- No paper_data mount, no pre-trained checkpoints

## Safe Modification Targets
1. `nn.py`: Model architecture (activations, layers, normalization, dropout)
2. `main.py`: Optimizer, LR schedule, loss function, training loop (non-DP path only)
3. Flags: LR, momentum, batch_size, epochs (parameter tuning)

## Risky Files (do NOT modify)
- `data.py`: Dataset loading, normalization, splits
- `dp_data.py`: DP dataloader (RandBin mechanism)
- `privacy.py`: Privacy accounting
- `ftrl_noise.py`: DP noise generation
- `/tools/record_score.sh`: Scoring script
- Evaluation protocol, test data, labels
