# Optimization Results: PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training

## Summary
- Total iterations: 4
- Best `ap`: **0.640** (baseline: 0.639, improvement: +0.001 / +0.16%)
- Best commit: baseline 7830a46
- Target: 0.6689 (NOT reached — inference-only optimization insufficient)

## Baseline vs. Best Metrics
| Metric | Baseline | Best (Iter 1) | Delta |
|--------|----------|---------------|-------|
| AP | 0.639 | 0.640 | +0.001 |
| AP50 | 0.817 | 0.819 | +0.002 |
| AP75 | 0.714 | 0.715 | +0.001 |
| AP_s | 0.490 | 0.492 | +0.002 |
| AP_m | 0.680 | 0.682 | +0.002 |
| AP_l | 0.811 | 0.814 | +0.003 |

## Key Changes Applied
| Iter | Change | AP | Effect |
|------|--------|-----|--------|
| 1 | score_thr=0.0 | 0.640 | +0.001 (minimal) |
| 3 | prompt_type='Text' | 0.495 | -0.145 (catastrophic) |
| 4 | max_per_img=1000 | FAILED | Runtime error |

## What Worked
- Lowering score threshold to 0.0 gives marginal improvement (+0.001 AP)
- The default PET-DINO inference config is already well-tuned

## What Didn't Work
- prompt_type='Text' is catastrophically bad — visual prompts are essential for PET-DINO
- max_per_img > 300 causes runtime errors
- Parameter tuning (score_thr, max_per_img) yields negligible gains

## Why 5% Improvement Was Not Achievable
1. **Inference-only limitation**: The model weights are frozen, limiting optimization to post-processing
2. **DETR architecture**: DETR models don't use NMS (bipartite matching), removing a major optimization lever
3. **Already well-tuned**: The default config is already optimized for COCO evaluation
4. **Environmental constraints**: Disk space limitations prevented multi-scale TTA and ensemble methods

## What Would Work (Requires Training)
- Multi-scale test-time augmentation (+1-2 AP)
- Longer training schedule (12→24 epochs, +1-3 AP)
- Backbone upgrade (Swin-T→Swin-S, +1.5-2.5 AP)
- COCO fine-tuning from O365 pretrained (+2-5 AP)

## Environmental Notes
- Docker proxy suppresses stdout/stderr from container exec
- NFS mount is read-only, requiring workarounds for config files
- Overlay filesystem only 20GB with ~400MB free, constraining temp file usage
- Each eval takes ~8 minutes on 2x A100
