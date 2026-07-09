# Code Analysis for Paper 1108 (InfoFlow KV) SOTA

## Evaluation Path
- Entry: `cd /repo/llm && python3 scripts/inference_with_recompute_kv.py configs/2wikimqa_repro.yaml`
- Model: Llama-3.1-8B-Instruct at `/models/Llama-3.1-8B-Instruct`
- Dataset: 2WikiMQA, 200 samples from `llm/inputs/2wikimqa.jsonl`
- Output: `results/2wikimqa_*/summary.json` with `avg_f1` per strategy

## Config Paths
- Reproduction: `llm/configs/2wikimqa_repro.yaml` (default_split=false, chunk_size=2048, top_p=0.15, method=norm)
- Eval: `llm/configs/2wikimqa_eval.yaml` (more strategies, default_split=true, chunk_size=1024)

## Metric Parser
- `avg_f1` in summary.json → maps to `F1_guided_recompute_norm_15pct`
- F1 is token-level: normalize → tokenize → common tokens / (precision + recall)
- `check_correct()` uses F1 > 0.5 threshold for accuracy

## Key Code Files
1. `llm/scripts/inference_with_recompute_kv.py` - Main inference script
2. `llm/models/llama/kv_cache/importance_scorer.py` - Token scoring (CORE for optimization)
3. `llm/models/llama/kv_cache/extractor.py` - KV cache extraction with RoPE correction
4. `llm/models/llama/kv_cache/recomputer.py` - KV recomputation at selected positions
5. `llm/models/llama/kv_cache/inference.py` - Generation with recomputed cache
6. `llm/benchmarks/longbench.py` - Dataset loading and F1 computation

## Safe Modification Targets
- `importance_scorer.py`: `get_attention_weights()`, `_compute_norm()`, `_compute_entropy()`, `_compute_vatp()`, `_compute_combined()`, `select_positions()`
- `inference_with_recompute_kv.py`: Strategy creation functions (make_1_layer_recompute_fn)
- Config YAML files: `layer_indices`, `chunk_size`, `default_split`, `top_p`

## Risky Files (do NOT modify)
- `llm/benchmarks/longbench.py` - Contains metric definitions
- Dataset files in `llm/inputs/`
- `llm/models/llama/kv_cache/extractor.py` - RoPE correction logic
- `llm/models/llama/kv_cache/recomputer.py` - KV recomputation logic

## Container Setup
- Model from ModelScope, symlinked to `/models/Llama-3.1-8B-Instruct`
- flash-attn 2.5.8, PyTorch 2.2.1+cu121
- GPU: 0,1 available (eval uses cuda:0)
- Record scores at `/autosota_artifacts/paper-1108/sota/scores.jsonl`

## Previous Attempts (from scores.jsonl)
All previous SOTA attempts regressed from baseline 0.4441:
- CODE-01 (selective eager patching v1): 0.4405 - corrupted hidden states
- ALGO-01 (position-debiased norm): 0.4368
- CODE-03 (L1 norm): 0.4381
- CODE-04 (Gaussian smoothing): 0.4234
- PARAM changes (chunk_size, default_split, layer_indices): all worse

## Root Cause Hypothesis
The 0.0194 gap between reproduction (0.4441) and paper (0.4635) is caused by:
1. Llama scorer uses eager attention for ALL 32 layers during scoring, changing hidden states
2. Qwen pipeline uses selective eager patching with flash attention for hidden states
3. The numerical discrepancy in hidden states affects which tokens are selected

## Strategy for Improvement
Port Qwen's selective eager patching correctly (unlike the failed v1):
- Monkey-patch ALL_ATTENTION_FUNCTIONS["flash_attention_2"] instead of switching _attn_implementation
- Keep flash attention for hidden state computation on ALL layers
- Extract attention weights only from scoring layers using chunked Q computation
- Early exit after last scoring layer
