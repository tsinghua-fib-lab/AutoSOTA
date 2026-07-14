# Code Analysis — Paper 5954: Value Aggregation for SciFact Retrieval

## Evaluation Path
- **Entry**: `eval_va_scifact.py` → `VAEvaluator` → `VaLlamaForCausalLM`
- **Flow**: Load model → Tokenize sentences with SciFact prompt → Forward pass through VaLlamaModel (collects value vectors from all layers) → Select layers → Mean pool over tokens (masked) → Mean pool over layers → Return numpy embeddings → MTEB evaluates SciFact retrieval
- **Command**: `python3 eval_va_scifact.py --model_path <path> --layers 20-27 --batch_size 1 --max_length 256`

## Key Files

### `eval_va_scifact.py` (main, 160 lines)
- `VAEvaluator.__init__()`: Loads VaLlamaForCausalLM, tokenizer. Configures layers, max_length.
- `VAEvaluator.encode()`: Main encoding function. Adds SciFact prompt, tokenizes, runs model, aggregates value vectors over selected layers, mean pools over tokens (with attention mask), mean pools over layers.
- `parse_layers()`: Parses command-line layer specification.
- `main()`: CLI entry point, runs MTEB evaluation.

### `llama_model.py` (854 lines)
- `VaLlamaAttention.forward()` (line 238-285): Returns `(attn_output, attn_weights, value_states)`. Value states reshaped to (batch, seq_len, hidden_size). **Key modification point**: also returns attn_weights (per-head attention weights).
- `VaLlamaDecoderLayer.forward()` (line 299-339): Returns `(outputs, value_states)`. 
- `VaLlamaModel.forward()` (line 492-598): Iterates over decoder layers, collects `all_values` as tuple.
- `VaLlamaForCausalLM.forward()` (line 766-847): Wraps VaLlamaModel, passes through `all_values`.

### `modeling_outputs.py`
- Custom `BaseModelOutputWithPast` with `all_values` field.

## Metric Parsing
- `eval_output_format`: Parse `ndcg_at_10` from MTEB JSON results. Value on 0-1 scale; multiply by 100 for percentage.
- MTEB v1.12 returns list of MTEBResults objects; `result.to_dict()["scores"]["test"]` contains score dicts.

## Safe Modification Targets
1. **`eval_va_scifact.py` `VAEvaluator.encode()` lines 88-109**: Token pooling and layer aggregation logic. Safe to modify as long as encoding semantics are preserved.
2. **`eval_va_scifact.py` `parse_layers()` lines 112-127**: Layer selection parsing. Safe to extend.
3. **`llama_model.py` `VaLlamaAttention.forward()` line 285**: Return value includes attn_weights and value_states. Safe to also return key_states.
4. **`llama_model.py` `VaLlamaModel.forward()` lines 548-598**: Value collection loop. Safe to also collect key vectors or attention weights.
5. **CLI arguments**: Safe to add new arguments for configuration.

## Risky Files (do not modify)
- MTEB library internals
- Model checkpoint files
- Dataset files
- Metric computation logic in MTEB

## Reusable Resources
- Model: `/models/llama2-7b-ms/models/modelscope--Llama-2-7b-ms/snapshots/master` (Llama-2-7B base, ModelScope download)
- Cache: `/autosota_cache/hf` (HF_HOME), `/autosota_cache/hf/datasets` (HF_DATASETS_CACHE)
- Container: `autosota_repro_paper_5954` (reusable, all dependencies installed)

## Known Levers (from manifest)
1. Model variant: Base vs chat model (chat likely better but gated)
2. Layer selection: 20-27 default; adjacent ranges may help
3. Sequence length: 256→512+ may help
4. Aggregation method: Weighted/VA-aligned aggregation

## Red-Line Boundaries
- Do NOT modify metric definitions, test data, labels, or dataset splits
- Do NOT hard-code predictions or metric values
- Do NOT change the evaluation protocol (MTEB SciFact retrieval)
