# KV-Cache Optimization for Vision Language Models

This project implements KV cache optimization for Qwen3-VL-8B-Instruct, reducing memory consumption during inference by selectively recomputing KV cache entries.

## Getting Started

### 1. Configure Settings

Before running, edit the config file `configs/blink_counting.yaml` to set your paths:

```yaml
# Model
model: Qwen/Qwen3-VL-8B-Instruct
cache_dir: /path/to/your/huggingface/cache/

# Paths - UPDATE THESE
dataset_dir: /path/to/dataset/cache/
output_dir: /path/to/output/

# Dataset
dataset: [blink_counting]

# KV cache recomputation
recompute_ratio: [0]      # Fraction of tokens to recompute (0 = no recompute)
method: norm              # Scoring method: norm, mass, entropy
chunk_k: 1                # Image chunking: 1 = no chunking, 2 = 2x2 chunks

# Generation
max_new_tokens: 128
run_baseline: false
run_recompute: true
num_samples: 50           # Limit samples for testing (remove for full run)
```

### 2. Run Inference

```bash
python scripts/evaluate.py --config configs/blink_counting.yaml
```

## TODO / Known Issues

### [2026.1.11] Fixed: chunk_k=1 with recompute_ratio=0 now matches baseline

**Root cause:** During chunked prefill, image tokens were processed with only intra-chunk causal attention. They could not attend to prefix tokens.

**Fix:** Modified `_prefill_chunks_batched()` in `chunk_prefiller.py` to:
1. Process prefix KV cache FIRST (before image chunks)
2. Pass prefix KV to image chunk processing
3. Image tokens now attend to: all prefix tokens + causal within chunk

### [2026.1.11] Multi-image support added

Multi-image inputs are now supported in chunked prefill mode. Each image is processed sequentially with accumulated context:
- Image 1 attends to: prefix text
- Inter-text 1 attends to: prefix + image 1
- Image 2 attends to: prefix + image 1 + inter-text 1
- And so on...

This ensures correct attention patterns across multiple images in the same input.
