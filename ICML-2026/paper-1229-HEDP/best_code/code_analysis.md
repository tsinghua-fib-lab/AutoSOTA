# HEDP CDDB-Hard Optimization — Code Analysis

## Evaluation Path
- **Known eval**: `python3 main.py --config configs/eval/known/cddb-hard.json`
  - Loads checkpoint from `./logsmodel/cddb/task_4.pth`
  - Evaluates 5 known domains (gaugan, biggan, wild, whichfaceisreal, san)
  - Output: `CNN:{'total': XX.XX}` (domain-level accuracy avg, class_num=2)
- **Unknown eval**: `python3 main.py --config configs/eval/unknown/cddb-hard.json`
  - Same checkpoint, 3 unseen domains (glow, stargan_gf, cyclegan)
  - Output: `CNN:{'total': XX.XX}`

## Train Path
- `python3 main.py --config configs/train/cddb-hard.json`
- 5 sequential tasks, 50 epochs each, SGD+momentum+CosineAnnealingLR
- Only visual/textual prompt params are trainable (~150M frozen CLIP, ~600K trainable)
- Checkpoints at `logs/hybrid_energy_distance_prompt_trainer_cddb_*/task_N.pth`

## Config Paths
- `configs/train/cddb-hard.json` — training hyperparams
- `configs/eval/known/cddb-hard.json` — known eval (energy_tau=1, distance_tau=0.4)
- `configs/eval/unknown/cddb-hard.json` — unknown eval (energy_tau=4, distance_tau=1)

## Metric Parser
- **Known_AA**: Grep `CNN:` in known eval stdout, extract `total` field
- **AF**: From training log per-task accuracy trajectory: for each domain, 
  forgetting = max_accuracy - final_accuracy; AF = -mean(forgetting)
- **Unknown_AA**: Grep `CNN:` in unknown eval stdout, extract `total` field

## Key Bottleneck
- **San domain (task 4)**: 248 images, only 51.81% test accuracy vs 99.90% for gaugan
- This single domain drags Known_AA from ~98% down to 92.69%

## Safe Modification Targets
| File | Role | Risk |
|------|------|------|
| `configs/train/cddb-hard.json` | Training hyperparams | Low |
| `configs/eval/known/cddb-hard.json` | Eval inference tau | Low (inference-only) |
| `methods/hybrid_energy_distance_prompt_trainer.py` | Training loop | Medium |
| `model/net.py` | Model/prompt init | Medium |
| `utils/data_manager.py` | Dataset/transforms | Medium |

## Risky Files (Do Not Modify)
| File | Reason |
|------|--------|
| `utils/toolkit.py` | Metric computation (accuracy_domain, accuracy_domain_total) |
| `runer.py` | Data loading, eval orchestration |
| `main.py` | Config loading, entry point |
| `model/clip.py`, `model/model.py` | Frozen CLIP backbone |
| `utils/data.py` | Dataset loading (iGanFake) |
| `configs/eval/unknown/cddb-hard.json` | Unknown eval protocol |

## Reusable Resources
- Pre-trained CLIP ViT-B/16 at `~/.cache/clip/ViT-B-16.pt`
- CDDB dataset at `/repo/data/CDDB/` (8 domains, already downloaded)
- Baseline checkpoint at `/repo/logsmodel/cddb/task_4.pth` (337MB)

## Eval vs Train Tau Differences
| Parameter | Train | Known Eval | Unknown Eval |
|-----------|-------|------------|--------------|
| energy_tau | 3 | 1 | 4 |
| distance_tau | 0.5 | 0.4 | 1.0 |

This suggests eval tau values were independently tuned. Eval-only tau optimization 
is a valid approach — no retraining needed, just inference hyperparameter tuning.

## Red-Line Boundary
- Never modify: metric computation, data splits, test labels, eval protocol
- Can modify: training hyperparams, eval inference hyperparams, training loop logic, 
  model initialization, data augmentation (training only, not eval transforms)
