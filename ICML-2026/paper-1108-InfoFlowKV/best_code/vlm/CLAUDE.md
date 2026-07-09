# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KV-Cache Optimization framework for Vision Language Models (VLMs), specifically Qwen3-VL-8B-Instruct. The project reduces memory consumption during inference by selectively recomputing KV cache entries instead of storing all of them.

## Commands

```bash
# Primary entry point: inference with KV cache recomputation
python scripts/inference_with_recompute_kv.py --config configs/blink_counting.yaml

# Baseline inference (no KV optimization)
python scripts/run_blink.py --config configs/blink_counting.yaml

# Simple demo with model patches
python scripts/qwen3_vlm_inference.py
```

For quick validation, set `num_samples: 5` in the config file.

## Architecture

### KV Cache Pipeline (`models/qwen/kv_cache/`)

Five-stage optimization pipeline:

1. **VLMKVCacheExtractor** (`extractor.py`): Extracts KV cache during prefill, handles mixed image+text inputs, supports chunked prefill with `chunk_k` parameter
2. **ImportanceScorer** (`importance_scorer.py`): Computes per-position importance scores using `norm`, `entropy`, or `mass` methods
3. **KVCacheRecomputer** (`recomputer.py`): Selectively recomputes KV at high-importance positions, handles MRoPE and GQA
4. **KVCacheInference** (`inference.py`): Runs generation with pre-filled + recomputed KV cache
5. **Image Chunking** (`chunker.py`, `chunk_prefiller.py`): Splits image tokens into k×k chunks for parallel prefill

### Model Patches (`models/qwen/patches/`)

Runtime hooks using context manager protocol to intercept model internals:
- `VisualPatch`: Captures visual encoder outputs
- `TextPatch`: Captures language model outputs
- `AttentionPatch`: Extracts attention weights

### Benchmarks (`benchmarks/`)

BLINK benchmark with 14 visual reasoning tasks (Art Style, Counting, Forensic Detection, etc.).

## Configuration

Edit `configs/blink_counting.yaml`:
- `model`: HuggingFace model path
- `cache_dir`: Local HF cache directory
- `dataset`: BLINK task name (e.g., `blink_artstyle`, `blink_counting`)
- `recompute_ratio`: Fraction of tokens to recompute (default 0.15)
- `method`: Scoring method (`norm`, `entropy`, `mass`)
- `chunk_k`: Image chunk size for parallel prefill

## Coding Conventions

- 4-space indentation, snake_case for functions/variables
- Lowercase module names, config keys with underscores
- Commit tags: `[FEAT]`, `[FIX]`, `[INIT]`
- No formatter configured; match existing style
