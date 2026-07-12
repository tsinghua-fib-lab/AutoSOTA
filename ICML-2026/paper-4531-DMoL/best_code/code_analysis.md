# Code Analysis — DMoL Paper 4531

## Evaluation Path
- **Command**: `python3 main.py --method dmol --dataset cifar100 --epochs 100 --batch_size 128 --lr 0.1 --weight_decay 5e-4 --alpha 0.5 --num_modules 4 --label_smoothing 0.1 --use_cosine_lr --seed 42 --data_dir /datasets --log_dir results`
- **Metric parser**: Parse stdout for `Test Acc:  XX.XX%` on epoch 100 line, or load JSON from `results/*.json` and extract `history[-1]['test_acc']`
- **GPUs**: 2x NVIDIA A100-SXM4-80GB (indices 0,1). Manifest says GPU 2,3 but container sees 0,1.

## Architecture
- **FeatureExtractor** (`models/blocks.py`): SimpleCNN — 4 Conv2d→ReLU chains, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear(128→128)
  - ZERO BatchNorm — pure Conv+ReLU blocks
  - ZERO Dropout — no regularization
  - channels: 3→32→64→128→128, feature_dim=128
- **ProcessingModule** (`models/blocks.py`): 2-layer MLP (128+num_classes → 256 → num_classes)
  - ZERO Dropout/BatchNorm
- **DMoL_Network** (`models/dmol.py`): FeatureExtractor + 4 ProcessingModules
  - Progressive: module_i sees (p_prev=softmax(module_{i-1}), shared_features)

## Training
- **Loss** (`trainers/train_dmol.py`): alpha * KL(log_p, y_one_hot) + (1-alpha) * KL(log_p, p_prev)
- **Optimizer**: SGD with momentum=0.9, nesterov=True
- **LR schedule**: CosineAnnealingLR over 100 epochs (when --use_cosine_lr)
- **Label smoothing**: 0.1 applied via one-hot scattering
- **Data aug** (CIFAR-100): RandomCrop(32, pad=4) + RandomHorizontalFlip() + Normalize

## Config (`config.py`)
- Standard argparse. Safe to add new args.
- Key params: lr, weight_decay, alpha, label_smoothing, num_modules, epochs, batch_size, seed, use_cosine_lr

## Safe Modification Targets
1. `models/blocks.py` — FeatureExtractor (add BatchNorm, GroupNorm, Dropout)
2. `models/blocks.py` — ProcessingModule (add Dropout, LayerNorm)
3. `trainers/train_dmol.py` — training loop (add MixUp/CutMix, gradient clipping, diagnostic logging, alpha scheduling)
4. `main.py` — scheduler setup (add warmup, min_lr, EMA), evaluation swap
5. `models/dmol.py` — architecture changes (add normalization between modules)
6. `config.py` — new hyperparameters

## Risky Files
- `datasets/dataloader.py` — test transforms must NOT be changed
- `utils/evaluate.py` — metric computation must NOT be changed

## Paper Data
- No pre-downloaded paper data mount. Use /datasets (CIFAR-100 pre-converted at /datasets/cifar-100-python/)

## Baseline
- Iteration 0: accuracy=63.72%, commit=44ecd74
