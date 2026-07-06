# PTA Codebase Analysis — Paper 1249

## Evaluation Path
- **Entry**: `pta_runner_patched.py` → `main()` → `PTA()`
- **CLIP model**: `clip.load(backbone)` → ViT-B/16 or RN50
- **Data pipeline**: `build_test_data_loader()` → `DatasetWrapper` → per-sample CLIP encoding
- **Metric**: Top-1 accuracy computed in `PTA()` via `cls_acc(final_logits, target)`

## Core Inference Path (PTA per-sample loop)
```
for each test image:
  1. get_clip_logits → image_features, clip_logits (zero-shot)
  2. softmax(clip_logits) → per-class confidence w
  3. mask = w >= 0.1 → confident classes only
  4. w_new = 1 - exp(-w / T) → update weight (T=20)
  5. target_prototype[mask] = EMA(target_prototype, image_feature, w_new)
  6. refined_text = alpha * text_features + (1-alpha) * target_prototype
  7. final_logits = clip_logits + 100 * image_features @ refined_text.T
  8. accuracy = argmax(final_logits) == target
```

## Config Path
- `configs/ucf101.yaml` → alpha=0.01, T=20.0
- Other datasets use identical configs except caltech101 (T=50)

## Metric Parser
- stdout: `---- PTAs test accuracy: XX.XX. ----`
- File: `outputs/result.txt`: `PTAs performance on ucf101: Top1- XX.XX.`

## Key Files
| File | Role | Safe to Modify |
|---|---|---|
| `pta_runner_patched.py` | Main runner, PTA algorithm | Yes — core optimization target |
| `utils.py` | CLIP logits, data loading, classifier | Yes — feature extraction |
| `configs/ucf101.yaml` | Hyperparameters | Yes — alpha, T |
| `datasets/ucf101.py` | Dataset class, templates | Yes — templates |
| `datasets/utils.py` | DatasetWrapper, DataLoader | Yes — worker_init_fn |
| `datasets/oxford_pets.py` | Base data reading | No — shared across datasets |
| `clip/` | Local CLIP implementation | No — model weights |

## Reusable Resources
- CLIP ViT-B/16 weights cached at `~/.cache/clip/`
- UCF101 dataset at `/datasets/ucf101/`
- No `/paper_data` mount

## Risky Files (do not modify)
- `clip/` directory — model implementation
- `datasets/oxford_pets.py` — base class used by all datasets
- `datasets/__init__.py` — dataset registry
- Output parsing in `PTA()` — must preserve for metric extraction

## Safe Modification Targets
1. `datasets/ucf101.py` — template list (CODE-1)
2. `pta_runner_patched.py:update_text_features()` — adaptive threshold, alpha schedule (ALGO-2, ALGO-5)
3. `pta_runner_patched.py:PTA()` — warmup phase, prototype repulsion (ALGO-4, ALGO-3)
4. `configs/ucf101.yaml` — hyperparameters (PARAM-1)
5. `pta_runner_patched.py:main()` — seed expansion, multi-seed ensemble (CODE-2, CODE-3)
6. `pta_runner_patched.py:update_text_features()` — outlier rejection (CODE-4)

## Known Levers
- `alpha` (0.01): text-anchor interpolation weight
- `T` (20.0): temperature for confidence→update_weight mapping
- CLIP backbone (ViT-B/16 vs ViT-L/14)
- Prompt templates (currently 1; most datasets use 7)
- Confidence threshold (hardcoded 0.1 in update_text_features)
- EMA prototype initialization (zeros)
- Stream order (shuffle seed)

## Evaluation Command (in-container)
```
CUDA_VISIBLE_DEVICES=0 python3 pta_runner_patched.py --config configs --datasets ucf101 --backbone ViT-B/16 --data-root /datasets
```
