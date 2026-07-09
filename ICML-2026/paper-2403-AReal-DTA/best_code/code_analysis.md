# Code Analysis for Paper 2403 (AReaL-DTA) SOTA Optimization

## Evaluation Path
- `throughput_bench.py` — main evaluation script
- Calls `create_engine()` → creates FSDPEngine with specified config
- Calls `measure_throughput()` → warmup + timed measurement loop
- Uses `mock_tree_input()` → creates synthetic tree-structured input data
- Metrics parsed from stdout: "Dense throughput: X.X tok/s (Y.YY K tok/s)" and "Tree throughput: X.X tok/s (Y.YY K tok/s)"

## Key Source Files

### Training Engine
- `areal/engine/fsdp_engine.py` — FSDP training engine, micro-batch packing, tree training integration

### Tree Attention
- `areal/models/tree_attn/module_fsdp.py` — Patches flash_attention with tree attention; compiled flex_attention wrapper
- `areal/models/tree_attn/tree.py` — Tree building, trie structures, BlockMask construction
- `areal/models/tree_attn/constants.py` — BLOCK_SIZE (128), USE_TRITON_TREE_ATTN flags
- `areal/models/tree_attn/functional.py` — Logprob/entropy computation for tree nodes
- `areal/models/tree_attn/triton_kernel.py` — Experimental Triton tree attention kernel
- `areal/models/tree_attn/module.py` — Tree attention module, patching logic

### Config
- `areal/api/cli_args.py` — TrainEngineConfig, MicroBatchSpec, OptimizerConfig definitions

## Metric Parser
- Primary: `training_throughput_k_tok_per_s` = dense throughput in K tok/s
- Tree: `tree_training_throughput_k_tok_per_s` = tree throughput in K tok/s
- Parsed from stdout: `"Dense throughput: X.X tok/s (Y.YY K tok/s)"`
- Output format is consistent across runs

## Known Bottlenecks (from repro log)
1. Tree training pads 1024 real tokens to 8192 (87.5% padding waste)
2. Only 1 of 8 sequences shares the 1024-token prefix (C ≈ 1.33x)
3. max_autotune=False disables kernel benchmarking for flex_attention
4. .contiguous() calls on every forward pass even when already contiguous
5. BlockMask rebuilt on every forward pass (no caching)
6. BLOCK_SIZE=128 may be suboptimal for A100 (132 SMs)

## Safe Modification Targets
1. `throughput_bench.py:mock_tree_input()` — input construction only
2. `throughput_bench.py:create_engine()` — config parameters
3. `areal/models/tree_attn/module_fsdp.py:_TORCH_COMPILE_OPTIONS` — torch.compile options
4. `areal/models/tree_attn/module_fsdp.py:_tree_attn_fwd_func` — contiguous/permute calls
5. `areal/models/tree_attn/constants.py` — BLOCK_SIZE, USE_TRITON flags
6. `areal/models/tree_attn/tree.py` — BlockMask caching
7. `areal/engine/fsdp_engine.py` — gradient checkpointing, micro-batch construction

## Risky Files (change cautiously)
- `areal/models/tree_attn/functional.py` — logprob computation, affects gradient correctness
- `areal/models/tree_attn/triton_kernel.py` — experimental kernel, may produce wrong results
- `areal/engine/fsdp_engine.py` — core training loop, any change could break training

## Reusable Resources
- Model: `/models/Qwen3-1.7B` (HuggingFace format)
- Venv: `/autosota_cache/areal-venv` (Python 3.12.13, torch 2.9.1+cu129)
- No paper data mounted
