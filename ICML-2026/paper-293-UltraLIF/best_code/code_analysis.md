# Code Analysis — Paper 293 UltraLIF SOTA Optimization

## Evaluation Path

- **Command**: `python experiments/train.py --model ultratplif --dataset cifar10 --epochs 100 --hidden 64 --timesteps 1 --batch-size 128 --lr 0.001 --seed 42 --track-spikes`
- **Entry point**: `experiments/train.py` → `main()` → parses args, builds model, calls `train_model()`
- **Training**: `ultralif/training/trainer.py` → `train_model()` — Adam + CosineAnnealingLR, CE loss
- **Metrics**: `ultralif/training/metrics.py` → `count_spikes_epoch()`, `compute_energy_proxy()`
- **Data**: `ultralif/datasets/loader.py` → `get_dataset("cifar10")` uses pre-converted local data
- **Models**: `ultralif/networks/fc.py` → `SNN` (single layer), `DeepSNN` (2-layer), `TripleSNN` (3-layer)
- **Neurons**: `ultralif/neurons/ultra.py` → `UltraPLIF` (learnable tau + eps)

## Key Files

| File | Role | Risk | Target |
|------|------|------|--------|
| `experiments/train.py` | CLI, model construction, result reporting | Low | Add training stability CLI args |
| `ultralif/training/trainer.py` | Training loop, optimizer, scheduler | Low | Add grad clip, AdamW, warmup, EMA |
| `ultralif/networks/fc.py` | Network architectures | Low | Weight init improvements |
| `ultralif/neurons/ultra.py` | Neuron dynamics | Medium | Membrane init, epsilon schedule |
| `ultralif/datasets/loader.py` | Data loading, augmentation | Low | RandAugment, MixUp |
| `ultralif/training/metrics.py` | Spike rate, energy computation | High | DO NOT MODIFY — eval metrics |
| `ultralif/training/logging.py` | Output logging | Low | No changes needed |

## Reusable Paper Data

- `/repo/data/cifar-10-batches-py/` — Pre-converted CIFAR-10 in torchvision format
- `/autosota_cache/` — Host cache mounted inside container

## Safe Modification Targets

1. `trainer.py:train_model()` — optimizer, scheduler, gradient clipping, warmup, EMA
2. `loader.py:get_dataset("cifar10")` — augmentation transforms (training only)
3. `fc.py:SNN.__init__()`, `DeepSNN.__init__()` — weight initialization
4. `ultra.py:UltraPLIF.reset()` — membrane potential initialization
5. `train.py` — new CLI arguments

## Red-Line Boundaries

- DO NOT modify `metrics.py` (eval metrics)
- DO NOT modify test data transforms in `loader.py`
- DO NOT change CIFAR-10 class labels or data splits
- DO NOT hard-code predictions or metrics
- DO NOT modify `record_score.sh`

## Baseline Metrics (Iteration 0)

- Accuracy: 43.39%
- Spike Rate: 0.4706
- Energy: 0.4706 (= spike_rate at T=1)
