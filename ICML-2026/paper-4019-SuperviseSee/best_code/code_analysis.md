# SPROUT Code Analysis - Paper 4019 SOTA

## Pipeline Overview
Three-stage pipeline for training-free nuclear instance segmentation:
1. `feature_points.py` — DINOv2 feature extraction + POT-Scan OT → point prompts (pos_points.csv, neg_points.csv)
2. `runSAM.py` — SAM2.1 inference with point prompts → soft_masks.json
3. `visual_json.py` — RLE JSON → init_mask.png
4. `eval.py` — Evaluate init_mask.png against ground truth

## Key Files
| File | Role | Safe to modify |
|------|------|----------------|
| `project/feature_points.py` | Feature extraction + point sampling | Yes - model, params, sampling |
| `project/runSAM.py` | SAM inference + NMS | Yes - prompt construction, NMS params |
| `project/eval.py` | Evaluation | NO - metric definitions |
| `project/visual_json.py` | Mask serialization | Yes but no optimization value |
| `project/utils/img_feat.py` | DINOv2 feature extraction + K-means | Yes - backbone, K selection |
| `project/utils/sample_points.py` | Point sampling (watershed + grid) | Yes - sampling strategies |
| `project/utils/NMS.py` | Soft-NMS + mask merging | Yes - NMS params |
| `project/utils/ot.py` | Optimal Transport (Sinkhorn) | Yes - OT params, warm start |
| `project/utils/mask_check.py` | Mask quality filters | Yes - filter criteria |
| `project/utils/mask_generation.py` | Reference mask generation (Otsu) | Yes - thresholds |
| `project/utils/densecrf.py` | DenseCRF post-processing | Yes - CRF params |

## Baseline Metrics
AJI=0.595, PQ=0.571, DQ=0.786, SQ=0.726, Dice=0.770

## Key Observations
1. SAM2.1 used instead of paper-specified SAM-Large (SAM1)
2. K-means initialization is non-deterministic (no seed)
3. Single positive point per nucleus
4. Grid-uniform negative points (not nucleus-centric)
5. Fixed K=3 clusters for all images
6. No box prompts (points only)
7. stitch_masks has perfect-square assertion
8. H-channel normalization edge case in NMS.py
