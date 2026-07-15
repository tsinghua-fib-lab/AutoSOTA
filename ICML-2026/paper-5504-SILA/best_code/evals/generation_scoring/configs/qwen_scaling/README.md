# Qwen2.5 Scaling Law Label Generation

This directory contains configurations for running label generation experiments across Qwen2.5 models (7B, 14B, 32B, 72B) to study scaling laws for SelfIE adapters.

## Overview

The experiment setup:
- **Dataset**: 1000 random vectors sampled from VAL split (indices 44673-49636) across middle 50% of layers
- **Models**: Qwen2.5-{7B,14B,32B,72B}-Instruct
- **Two variants per model**:
  - **Trained adapter**: Uses a trained SelfIE checkpoint with scale=1.0
  - **Untrained adapter**: Identity projection with grid search over scales=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
- **Label generation**: Greedy decoding (temperature=0.0), single label per vector
- **Metadata tracking**: Preserves both combined dataset index (0-999) and original fifty-thousand-things indices for topic mapping

## Setup

### 1. Prepare Combined Datasets

The combined datasets sample 1000 vectors from the VAL split, distributed across all middle 50% layers:

```bash
# Prepare all model sizes at once
/tmp/venv/bin/python data_prep/prepare_qwen_scaling_datasets.py --model-size all

# Or prepare individually
/tmp/venv/bin/python data_prep/prepare_qwen_scaling_datasets.py --model-size 7b
/tmp/venv/bin/python data_prep/prepare_qwen_scaling_datasets.py --model-size 14b
/tmp/venv/bin/python data_prep/prepare_qwen_scaling_datasets.py --model-size 32b
/tmp/venv/bin/python data_prep/prepare_qwen_scaling_datasets.py --model-size 72b
```

This creates for each model size:
- `qwen25_{size}_instruct_fifty_thousand_things_combined_val_1000.pt` - Combined vectors
- `qwen25_{size}_combined_val_1000_master.json` - Master JSON for evaluation
- `qwen25_{size}_instruct_fifty_thousand_things_combined_val_1000_metadata.json` - Metadata with original indices

Layer ranges (middle 50%):
- 7B (28 layers total): layers 7-20 (14 layers)
- 14B (48 layers total): layers 12-35 (24 layers)
- 32B (64 layers total): layers 16-47 (32 layers)
- 72B (80 layers total): layers 20-59 (40 layers)

### 2. Configure Trained Adapter Path (if using)

If you have a trained adapter checkpoint, update the `adapter_checkpoint_path` in the `*_trained_label_gen.json` configs:

```bash
# Edit each trained config and replace PLACEHOLDER_TRAINED_CHECKPOINT.pt
# with your actual checkpoint path, e.g., checkpoints/warm-cloud-36_step_2000_final.pt
```

### 3. Run Label Generation

Use the helper script to run experiments:

```bash
# Run all models with untrained adapter (grid search)
./evals/generation_scoring/run_qwen_scaling_label_gen.sh all untrained

# Run all models with trained adapter
./evals/generation_scoring/run_qwen_scaling_label_gen.sh all trained

# Run specific model and variant
./evals/generation_scoring/run_qwen_scaling_label_gen.sh 72b untrained
./evals/generation_scoring/run_qwen_scaling_label_gen.sh 14b trained
```

Or run directly with Python:

```bash
/tmp/venv/bin/python evals/generation_scoring/run_eval.py \
    --config-file evals/generation_scoring/configs/qwen_scaling/qwen25_72b_untrained_label_gen.json \
    --output-dir results/qwen_scaling/72b_untrained \
    --no-wandb
```

### 4. Merge Labels with Metadata

After generation, merge the labels with original dataset metadata to enable topic-level analysis:

```bash
/tmp/venv/bin/python evals/generation_scoring/merge_labels_with_metadata.py \
    --labels results/qwen_scaling/72b_untrained/generated_labels_*.json \
    --metadata outputs/qwen_scaling/qwen25_72b_instruct_fifty_thousand_things_combined_val_1000_metadata.json \
    --output results/qwen_scaling/72b_untrained/labels_with_metadata.json
```

The enriched output includes:
- `combined_index`: Index in combined dataset (0-999)
- `original_layer`: Which layer the vector came from
- `original_global_index`: Global index in fifty-thousand-things dataset (44673-49636)
- `original_local_val_index`: Local index within VAL split
- `label`: Generated label text
- `scale`: Scale value used
- `label_index`: Label index (for multiple labels per scale)

## Config Files

### Trained Adapter Configs

- `qwen25_7b_trained_label_gen.json`
- `qwen25_14b_trained_label_gen.json`
- `qwen25_32b_trained_label_gen.json`
- `qwen25_72b_trained_label_gen.json`

Settings:
- `adapter_checkpoint_path`: Path to trained checkpoint (must be updated)
- `scale_values`: [1.0] (single scale)
- `temperature`: 0.0 (greedy decoding)

### Untrained Adapter Configs

- `qwen25_7b_untrained_label_gen.json`
- `qwen25_14b_untrained_label_gen.json`
- `qwen25_32b_untrained_label_gen.json`
- `qwen25_72b_untrained_label_gen.json`

Settings:
- `adapter_checkpoint_path`: null (identity projection)
- `scale_values`: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] (grid search)
- `temperature`: 0.0 (greedy decoding)

## Soft Prompt Template

All configs use the Qwen2.5 soft prompt template with `<|fim_pad|>` as the placeholder token:

```
<|im_start|>user
What is the meaning of "<|fim_pad|>"?<|im_end|>
<|im_start|>assistant
The meaning of "<|fim_pad|>" is "
```

Note: The YAML-style `|-` syntax in training configs indicates no trailing newline.

## Output Structure

### Generated Labels File

```json
{
  "metadata": {
    "dataset_name": "qwen25-72b-fifty-thousand-things-combined-val",
    "layer": 0,
    "run_id": "...",
    "label_generator_config": { ... }
  },
  "generated_labels": [
    {
      "latent_index": 0,
      "label": "example label",
      "scale": 1.0,
      "label_index": 0
    },
    ...
  ]
}
```

### Enriched Labels File (after merging with metadata)

```json
{
  "metadata": { ... },
  "enriched_labels": [
    {
      "combined_index": 0,
      "original_layer": 41,
      "original_global_index": 48561,
      "original_local_val_index": 3888,
      "split": "val",
      "label": "example label",
      "scale": 1.0,
      "label_index": 0
    },
    ...
  ],
  "num_labels": 1000
}
```

## Analysis Workflow

1. **Generate labels** for all model sizes and variants
2. **Merge with metadata** to enable topic-level analysis
3. **Compare performance** across:
   - Model sizes (scaling laws)
   - Trained vs. untrained adapters
   - Different scales (for untrained)
4. **Map to topics** using `original_global_index` to look up topic names in the fifty-thousand-things dataset

## Troubleshooting

### Dataset not found

If you see errors about missing master JSON files, run the data preparation script:

```bash
python data_prep/prepare_qwen_scaling_datasets.py --model-size all
```

### Placeholder checkpoint path

For trained configs, make sure to update `adapter_checkpoint_path` from `PLACEHOLDER_TRAINED_CHECKPOINT.pt` to your actual checkpoint path.

### Memory issues

For larger models, you may need to:
- Reduce `label_generation_batch_size` in the config
- Use `device: "auto"` for multi-GPU distribution
- Ensure you have enough GPU memory (72B model requires ~140GB)

### Scale selection for untrained

After running the untrained experiments, you can analyze which scale performs best and use that for fair comparison with the trained adapter.
