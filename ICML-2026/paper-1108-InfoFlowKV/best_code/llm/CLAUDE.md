# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for optimizing LLM inference through **selective KV cache recomputation** and **Ring Attention sequence parallelism**. The core idea: extract KV caches from long contexts, score token positions by importance, then selectively recompute only the most important positions to improve quality while controlling cost.

Supports Qwen, ChatGLM, and Llama model families. Evaluates on long-context QA benchmarks (2WikiMQA, HotpotQA, MuSiQue, needle-in-haystack).

## Commands

### Running Inference
```bash
python scripts/inference_with_recompute_kv.py configs/2wikimqa_eval.yaml
```

### Ring Attention Tests (require multi-GPU, typically 8)
```bash
torchrun --nproc_per_node 8 ring-flash-attention/test/test_ring_flash_attn_func.py
torchrun --nproc_per_node 8 ring-flash-attention/test/test_zigzag_ring_flash_attn_func.py
```

### Benchmarks
```bash
# Single GPU prefill
python scripts/benchmark_single_gpu.py --model <model_path>

# Multi-GPU ring attention
torchrun --nproc_per_node=4 scripts/benchmark_ring_attention.py --model <model_path>

# Parameter sweep
torchrun --nproc_per_node=4 scripts/sweep_benchmark.py --model <model_path> --output results.json
```

### Install Dependencies
```bash
pip install -r models/requirements.txt
pip install -e ring-flash-attention/   # ring-flash-attention is a git submodule
```

## Architecture

### KV Cache Pipeline (per model: `models/{qwen,chatglm,llama}/kv_cache/`)

Each model has an identical 4-stage pipeline with model-specific implementations:

1. **Extractor** (`extractor.py`) - Runs the model forward pass to extract KV caches from context passages. Produces `KVCacheData` containing key/value tensors plus metadata.
2. **ImportanceScorer** (`importance_scorer.py`) - Scores each token position's importance using methods: `norm` (L2), `attn` (attention weight), `entropy`, `mass`, or `combined`.
3. **Recomputer** (`recomputer.py`) - Selectively recomputes KV at the top-k important positions (controlled by `recompute_ratio`). This is where quality recovery happens.
4. **Inference** (`inference.py`) - Generates text using the recomputed cache.

Shared interfaces are in `models/base.py`: `BasePatch` (monkey-patching with context manager support), `AttentionCapture`, `ModelConfig`.

### Recomputation Strategies (`scripts/inference_with_recompute_kv.py`)

The main script supports multiple strategies via YAML config:
- **baseline** - Standard full-context inference (no cache manipulation)
- **no_recompute** - Use extracted cache directly without recomputation
- **1_layer_guided** - Standard recomputation with importance scoring
- **cacheblend** - CacheBlend strategy (Layer 0: full, Layer 1: full + select top 15%, Layer 2+: selective)
- **2_layer_guided** - Extract without RoPE correction, then reorder_and_rebase before recompute

### Distributed / Parallel (`models/parallel/`)

Extends the single-GPU pipeline to multi-GPU using Ring Attention:
- `DistributedConfig` - Process group setup, sequence partitioning across GPUs
- `DistributedExtractor` - Parallel KV extraction with `global_offset` tracking
- `DistributedScorer` - Importance scoring with all-gather for cross-GPU communication
- `RingAttentionRecomputer` - Recomputation via ring attention; falls back to flash_attn on single GPU

### Ring Flash Attention (`ring-flash-attention/`)

Git submodule (`zhuzilin/ring-flash-attention`). Key variants:
- `ring_flash_attn_varlen_func` - Basic ring attention (varlen/packing API)
- `zigzag_ring_flash_attn_varlen_func` - Compute-balanced variant (better load distribution)
- `llama3_flash_attn_varlen_func` - Recommended for most varlen use cases
- HF adapter: `substitute_hf_flash_attn()` replaces Flash Attention in HuggingFace models

Known limitations: no dropout support, no window_size, bf16 accumulation causes minor arithmetic errors.

## Configuration

YAML configs in `configs/` control the full pipeline. Key parameters:
- `recompute_ratio` (0.0-1.0) - Fraction of positions to recompute
- `method` - Importance scoring method (`norm`, `attn`, `entropy`, `mass`, `combined`)
- `layer_start` / `layer_end` - Restrict which layers are recomputed
- `strategies` - List of strategies to evaluate in a single run

## Key Dependencies

PyTorch (CUDA), Transformers, Flash Attention 2, Triton, ring-flash-attn (submodule). Full list in `models/requirements.txt`.
