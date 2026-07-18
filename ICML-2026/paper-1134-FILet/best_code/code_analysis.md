# Code Analysis for Paper 1134 (FILet)

## Evaluation Path
- `run_reasoning.py` → main training/eval script
- Uses HuggingFace datasets, loads BoolQ from `google/boolq`
- Metric: accuracy computed via `evaluate.load("accuracy")`
- Eval every `eval_steps` (500), tracks `max_score` in-memory
- Logs scores to: `/repo/logs/filet_reasoning/`

## Train/Inference Path
- Model: `LlamaForSequenceClassification` from `models/modeling_llama.py`
- LoRA adapters via `models/filet_layer.py` (custom FILet layer)
- Fisher initialization: `SxSy_computation.py` → `FILet_init()`
- Fisher-guided init: `models/fisher_guided_init.py` → `fisher_guided_lora_min_from_W()`

## Config Path
- All hyperparameters via argparse in `run_reasoning.py`
- LoRA config in `models/lora_config.py`

## Metric Parser
- `run_reasoning.py` logs eval results with `logger.info(f"epoch {epoch}: {eval_metric}")`
- Format: `epoch N: {accuracy: XXX}`
- `max_score` tracks best accuracy during training but is NOT printed at end
- Score extraction: parse log for all accuracy values, take max
- Need to add final max_score print at end for reliable extraction

## Reusable Resources
- `/models/Llama-2-7b-hf` — pretrained model (in container)
- `/autosota_cache/hf` — HuggingFace cache
- `/datasets` — pre-downloaded datasets

## Risky Files
- `models/filet_layer.py` — core LoRA implementation (modify carefully)
- `models/fisher_guided_init.py` — Fisher-guided init logic
- `SxSy_computation.py` — Fisher computation (backward pass)
- `run_reasoning.py` — main entry point

## Safe Modification Targets
1. `run_reasoning.py` — optimizer setup (LoRA+), best-checkpoint tracking, args
2. `models/fisher_guided_init.py` — SVD replacement, normalization
3. `SxSy_computation.py` — Fisher estimation improvements
4. `models/filet_layer.py` — DoRA magnitude, PiSSA init

## Notes
- Training uses bfloat16
- Fisher uses 320 examples (40 batches × bs=8)
- LoRA rank=32, alpha=64
- Target modules: q_proj, k_proj, v_proj, up_proj, down_proj
- GPU devices: 2,3 (CUDA_VISIBLE_DEVICES=2,3)
