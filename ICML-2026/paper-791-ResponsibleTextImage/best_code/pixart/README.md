# External Heads for Concept Learning in PixArt-Alpha

Train and apply external attention heads to guide image generation toward specific concepts.

## Pipeline Overview

### 1. Data Generation (`data_generation.py`)

Generates training images using PixArt-Alpha with associated concept labels.

```python
from data_generation import PixArtDataCreator, Cfg

creator = PixArtDataCreator(Cfg)
creator.run(num_inference_steps=50, guidance_scale=4.5, create_zip=True)
```

**Outputs:** `datasets/<name>/` containing numbered images (0.jpg, 1.jpg, ...), `labels.json`, and `concept_dict.json`.

### 2. Training (`train.py`)

Trains external attention heads on transformer layers 11-27 to learn concept representations.

```bash
python train.py
```

**Configuration** (in `TrainingConfig`):
- `train_data_dir`: Path to dataset from step 1
- `target_layers`: Transformer layers to train (default: 11-27)
- `num_train_epochs`, `learning_rate`, `train_batch_size`

**Outputs:** `output_train_model/external_heads_final/` with `external_heads_full.pt` and per-layer head weights.

### 3. Inference (`inference.py`)

Generates images with trained external heads applied at configurable coefficients.

```bash
python inference.py
```

**Configuration** (top of file):
- `EXTERNAL_HEADS_PATH`: Path to trained `external_heads.pt`
- `TARGET_LAYERS`, `TARGET_HEADS`: Which layers/heads to apply
- `COEFFICIENT_LIST`: Steering strengths (e.g., `[0, 5, 10]`)
- `PROMPT`: Generation prompt

**Outputs:** `images_out/seed_<N>/` with baseline and coefficient-modified images.

## Supporting Module

**`utils_data.py`**: Dataset loading utilities including `TrainingDataset` and `get_dataloader` for training, plus `get_test_data` for evaluation.

## Requirements

```
torch
diffusers
transformers
torchvision
Pillow
tqdm
```
