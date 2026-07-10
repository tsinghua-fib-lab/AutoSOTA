# Code Analysis — Paper 3345 (THESEUS Task Vector Transport)

## Evaluation Path
- Entry: `src/merge_and_rebase/eval/vision_rebase.py:main()`
- Config: `configs/vision8_theseus_svhn_only.json`
- Method: `theseus` (`src/merge_and_rebase/rebase/methods/theseus.py`)
- Metric parsing: Final summary line — `grep '^ SVHN' | awk '{print $3}'` gives rebased accuracy as decimal (e.g., 0.576678 = 57.67%)
- Pattern: `SVHN   <target_zeroshot>   <rebased_accuracy>   <norm>`

## Key Files
1. **Config**: `configs/vision8_theseus_svhn_only.json` — THESEUS params, alpha search range [0, 5.0], step 0.1, patience 10
2. **Theseus Rebase**: `src/merge_and_rebase/rebase/methods/theseus.py` (1250 lines)
   - `collect_activations()` — collects source/target activations from dataloaders
   - `_compute_alignment_map()` — computes Procrustes alignment map from ActivationStore
   - `_precompute_transforms()` — per-layer transform computation (lines 686-780)
   - `_apply_transforms_to_visual_delta()` — applies transforms to transport task vector (lines 859-914)
   - `TheseusRebase.prepare()` — main preparation (lines 921-1088)
   - `TheseusRebase.apply()` — applies prepared transforms (lines 1090-1170)
3. **Eval**: `src/merge_and_rebase/eval/vision_rebase.py` (1163 lines)
   - Alpha search loop with early stopping
   - Per-task or shared alpha selection
   - Metric output via `pretty_print_task_accuracies()`
4. **GradFix**: `src/merge_and_rebase/rebase/methods/gradfix.py` — gradient-sign masking (same authors, ICLR 2026)
5. **Models**: CLIP ViT-B/16 (LAION-2B) as source, ViT-B/16+ (LAION-400M) as target

## Safe Modification Targets
- `theseus.py:_apply_transforms_to_visual_delta()` — post-processing of transported task vector (sparsification, SVD, per-component alpha)
- `theseus.py:_precompute_transforms()` — per-head alignment
- `theseus.py:TheseusRebase.prepare()` — parameter acceptance for new features
- `vision_rebase.py`: config parameter passing to Theseus
- Config JSON: adding new method_params

## Risky Areas (Do Not Modify)
- `eval_task_top1()` — metric computation
- Dataset loading/splits
- `load_hf_splits()` — test data
- `pretty_print_task_accuracies()` — output format

## Reusable Resources
- Fine-tuned checkpoint: `src/checkpoints/theseus/models/checkpoints/ViT-B-16/laion2b_s34b_b88k/svhn/best.pt` (598MB)
- Cache: `/autosota_cache/hf/datasets/ufldl-stanford___svhn`
- Models: `/autosota_cache/hf/hub/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K`, `models--timm--vit_base_patch16_plus_clip_240.laion400m_e31`
