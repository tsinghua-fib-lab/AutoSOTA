# FLUX External Heads Training Pipeline

Training external attention heads for concept learning in FLUX.1-dev diffusion models.

## Overview

This pipeline enables learning concept vectors (e.g., gender, age, race) that can steer image generation by modifying transformer attention layers.

## Pipeline

### 1. Data Generation (`data_generation.py`)

Generates training images with FLUX.1 (schnell or dev variant).

```bash
# Set HuggingFace token for gated models
export HF_TOKEN="your_token"

python data_generation.py
```

**Configuration classes:** `CfgDevBatch`, `CfgDevFull`, `CfgDevRace`, etc.

**Output:**
```
datasets/<name>/
├── 0.jpg, 1.jpg, ...   # Generated images
├── labels.json          # [[[prompt, [concepts]], ...]]
└── concept_dict.json    # {concept_name: index}
```

### 2. Training (`train.py`)

Trains external heads that modify attention outputs to encode concepts.

```bash
python train.py
```

**Key config (`TrainConfig`):**
- `train_data_dir`: Path to dataset
- `target_layers`: Transformer layers to patch (default: 0-18)
- `num_heads`: 24, `resolution`: 512
- `num_epochs`: 30, `batch_size`: 1

**Output:** `out_model/external_heads_final.zip` containing:
- `external_heads_full.pt` — Learned head weights
- `head_importance.json` — Per-head importance scores

### 3. Inference (`inference.py`)

Generate images with trained external heads applied.

```bash
python inference.py
```

**Key config (module constants):**
- `EXTERNAL_HEADS_PATH`: Path to `external_heads_full.pt`
- `TARGET_LAYERS`, `TARGET_HEADS`: Which layers/heads to apply
- `COEFFICIENT_LIST`: Scaling factors (e.g., `[0, 10]`)
- `PROMPT`, `GUIDANCE_SCALE`, `NUM_INFERENCE_STEPS`

**Output:** `images_out/seed_<N>/` with baseline and coefficient-scaled images.

## Supporting Files

- **`utils_data.py`**: Dataset loader (`TrainingDataset`), concept parsing, and data utilities.

## Requirements

- PyTorch with CUDA
- `diffusers`, `torchvision`, `PIL`, `tqdm`
- HuggingFace token for FLUX.1-dev access
