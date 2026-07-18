# Code Analysis for Paper 791 — Responsible Text-to-Image Diffusion

## Evaluation Path

- **Entry point**: `/repo/eval_final.py` (140 lines, self-contained)
- **Command**: `NUM_OCCUPATIONS=12 NUM_SAMPLES=25 python3 eval_final.py`
- **Output**: JSON at `$OUTPUT_DIR/results.json` with `metrics.delta`, `metrics.clip`, `gender.*`
- **Metrics computed**:
  - `delta`: `(max(n_m, n_f) / (N/2) - 1) / (1 - 1/2)` for G=2. Range [0, 2]; 0 = perfect balance.
  - `clip`: Mean CLIP score (ViT-B/32) × 100 across all generated images
- **Gender classifier**: Zero-shot CLIP with 4 prompt templates ("a photo of a male person", "a photo of a man", "a photo of a female person", "a photo of a woman")
- **Assignment**: Random 50/50 for each image (female or male steering)

## Inference Path

- **Processor class in eval**: `ExternalHeadProcessor` (simplified, inline in eval_final.py)
- **Processor class in inference.py**: `TrueDelegationExternalHeadsProcessor` (more robust, with validation)
- Both use bias-free projection through attn.to_out
- Concept vectors stored per (layer, head): shape [S_l, head_dim=72]
- 17 layers × 16 heads × 72 dims = 19,584 parameter tensors

## Training Path

- **Entry point**: `/repo/pixart/train.py`
- **Training uses**: `DiffusionTrainingModel.forward_with_prompt_comparison()`
- **Loss**: MSE between `model_pred` (heads ON, person prompt) and `model_ref` (heads OFF, concept prompt)
- **Concept prompt hardcoded**: `"a photo of a woman"` (line ~310)
- **Person prompt**: `"a photo of a person"`
- **Separate checkpoints**: male and female trained independently
- **Head importance**: EMA of gradient norms, used to identify top-3 heads per layer

## Configuration Points

- **Target layers**: `list(range(11, 28))` — 17 layers
- **Target heads**: `[10, 12, 14]` — top-3 by gradient importance
- **Coefficient**: `10.0` (env `COEFFICIENT`)
- **Inference steps**: `20` (env `INFERENCE_STEPS`)
- **Guidance scale**: `4.5` (env `GUIDANCE_SCALE`)
- **Seed**: `42` (env `SEED`)
- **Resolution**: 1024×1024

## Pre-downloaded Paper Data (`/paper_data`)

| Resource | Description | Used by |
|---|---|---|
| `transformer/` | PixArt-α transformer weights | eval, inference |
| `text_encoder/` | T5 text encoder | eval, inference |
| `vae/` | VAE decoder | eval, inference |
| `tokenizer/` | T5 tokenizer | eval, inference |
| `scheduler/` | DDPM scheduler config | eval, inference |
| `model_index.json` | Pipeline index | eval, inference |
| `external_concept_female.pt` | Female concept vectors (~307MB) | eval (copied to /repo/checkpoints/) |
| `external_concept_male.pt` | Male concept vectors (~307MB) | eval (copied to /repo/checkpoints/) |
| `external_concept_cartoon.pt` | Cartoon style vectors (~307MB) | Not used in eval |
| `external_concept_van_gogh.pt` | Van Gogh style vectors (~307MB) | Not used in eval |

## Known Levers (inference-side, no retraining)

1. **COEFFICIENT** (default 10.0): Higher = stronger gender steering. Notebook uses 70 (female) / 90 (male).
2. **TARGET_HEADS** (default [10, 12, 14]): Which attention heads to steer
3. **TARGET_LAYERS** (default 11-27): Which transformer layers to modify
4. **INFERENCE_STEPS** (default 20): More steps = better quality but slower
5. **GUIDANCE_SCALE** (default 4.5): CFG scale for generation
6. **SEED** (default 42): Different seeds may produce different deltas

## Risky Files (do not modify)

- `eval_final.py` — metric computation block (delta formula, CLIP score, gender classification)
- Any data loading logic that changes what test data is used
- Gender classifier templates

## Safe Modification Targets

- Environment variable defaults in eval_final.py (COEFFICIENT, TARGET_HEADS, TARGET_LAYERS, INFERENCE_STEPS, GUIDANCE_SCALE, SEED)
- `ExternalHeadProcessor` class (add attention weighting, head selection logic)
- `setup_procs`/`reset_procs` functions (add verification)
- Adding multi-seed wrapper or sweep logic around eval_final.py
- Training code in pixart/train.py (new loss functions, training procedures)

## Eval Command Verification

The manifest command `NUM_OCCUPATIONS=12 NUM_SAMPLES=25 python3 eval_final.py` runs correctly inside the container.
- Total images: 12 × 25 = 300 image pairs (600 total with baseline)
- GPU memory: ~12-14 GB during eval (fits on RTX 3090/4090)
- Runtime: ~15-25 min for full 300-image eval
- Small validation runs (4 occs × 5 samples = 20 images) take ~2-3 min
