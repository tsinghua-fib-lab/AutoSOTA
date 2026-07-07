# FLAME Codebase Analysis for SOTA Optimization

## Evaluation Path
- Script: FLAME/eval_autosplice.py
- Model loading: FLAME/test_dataset.py load_and_initialize_model()
- Dataset: FLAME/utils/localforgerydataset.py LocalForgeryDataset
- Output: stdout prints IoU, F1, ACC, AP + JSON with iou, f1, acc, ap keys

## Evaluation Flow
1. Load model from checkpoint + config JSON
2. Create DataLoader over /datasets/autosplice_test
3. Forward pass -> sigmoid -> threshold at 0.5 -> pixel-level IoU/F1
4. Detection: max(probs) per image -> compare to detection_threshold -> ACC/AP

## Training Path (NOT AVAILABLE)
- Script: FLAME/train.py
- Training data: MagicBrush + SID (NOT mounted)
- Only inference-time changes feasible

## Model Architecture
- ForgeryLocalizer: Main model SAM2 + FerretBackbone + adapters
- FerretBackbone: LAD operator + FerretNet feature extractor
- Config: FLAME/checkpoints/model_params.json
  - forensic_operator: lad_multi (4 taus: 0.016,0.032,0.064,0.128)
  - coarse_prompt_head: mask_compressor (no AdaptiveTauFusion)
  - adapter_type: shared, dropout_rate: 0.2
  - Detection threshold: 0.5

## Key Findings
1. AdaptiveTauFusion exists in code but NOT enabled (needs coarse_prompt_head change + retraining)
2. Pixel binarization uses fixed 0.5 threshold
3. Detection uses max probability per image
4. Gradient clipping starts at batch 3 (moot, no training data)

## Feasible Inference-Time Ideas
- ALGO-01: TTA (hflip + rotations)
- CODE-02: Detection threshold calibration sweep
- ALGO-04: Otsu adaptive thresholding

## Safe Modification Targets
- FLAME/eval_autosplice.py: TTA loop, thresholding
- Model checkpoint + config: read-only
- Test data: read-only
