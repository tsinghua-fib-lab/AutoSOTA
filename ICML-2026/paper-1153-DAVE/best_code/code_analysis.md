# DAVE Code Analysis for SOTA Optimization

## Paper
DAVE: Distribution-Aware Attribution via ViT Gradient Decomposition (Paper 1153)

## Evaluation Paths
- **EnergyPG eval**: `bash run_energypg_eval.sh` → `evaluation/energypg.py`
  - 500 images, 50 MC steps, GPU cuda:0
  - Reads bbox from `/datasets/imagenet_bbox_mapped.pkl`
  - Output: `evaluation/energypg_results/DeiT-III-B16-224/energypg_summary.json`
  - Parse `energypg_pct` from stdout or JSON
- **GridPG eval**: `evaluation/gridpg.py` (NOT run — too slow for practical use)
  - Requires 448x448 composite grids with interpolated position embeddings
  - Paper reports 65.76% but 784 tokens → quadratic attention, too slow

## Inference Path (DAVE attribution)
`core/explainer.py::DAVEExplainer.explain()`:
1. `remove_operator_variation()` → patches attention/GELU/LayerNorm
2. For each MC step (default 50):
   a. `effective_transform()`: aug + noise + forward + backward
   b. `c = c.detach() * x.detach()` → element-wise contribution
3. Aggregate via MAD masking
4. Optional post-processing (Gaussian + bilateral)
5. `restore_operator_variation()`

## Config Path
- `models_configs/deit3_b16_224.yaml` → loaded by `core/config.py::DAVEConfig`
- Key params: noise_alpha=0.9, augmentation ranges, post-proc kernel sizes

## Metric Parsers
- EnergyPG: `energypg_pct` from JSON summary or stdout `EnergyPG: X.XX%`
- GridPG: `gridpg_pct` from JSON summary or stdout `GridPG: X.XX%`

## Reusable Resources
- `/datasets/imagenet_val`: ImageNet-1k validation set (50k images, ImageFolder)
- `/datasets/imagenet_bbox_mapped.pkl`: Precomputed ILSVRC2012 bboxes (224x224 mapped)
- `/autosota_cache/hf/hub`: Timm model weights (deit3_base_patch16_224.fb_in1k)

## Risky Files (do not modify)
- `evaluation/energypg.py` — metric computation protocol
- `evaluation/gridpg.py` — metric computation protocol
- `evaluation/utils/perturbation.py` — shared perturbation utilities
- `/datasets/imagenet_val` — test data
- `/datasets/imagenet_bbox_mapped.pkl` — ground truth annotations

## Safe Modification Targets
- `core/explainer.py` — MC loop, noise schedule, aggregation
- `core/utils/detach_mode.py` — operator variation removal (attention/GELU/LayerNorm hooks)
- `core/utils/augment.py` — augmentation parameters, noise_alpha
- `core/utils/post_processing.py` — Gaussian/bilateral filter parameters
- `models_configs/deit3_b16_224.yaml` — hyperparameter config
- `core/config.py` — config loading (if new fields needed)

## Key Hyperparameter Levers
1. `noise_alpha` (0.9): noise schedule aggressiveness → cosine schedule option
2. `num_steps` (50): MC samples, quality vs. time trade-off
3. `rotate_range` ([-20,20]), `translate_range` ([0.1,0.1]): augmentation params
4. `post_proc` gaussian kernel_size/sgm, bilateral kernel_size/sgm_spatial/sgm_range

## Known Patches
- `detach_mode.py`: **kwargs added to attention forward for timm 1.0.27 compat (is_causal)
- Proxy env vars cleared in evaluation scripts for HF downloads

## Baseline
- EnergyPG: 84.25% (500 images, 50 steps, seed 42)
- Above paper reported 82.43%
