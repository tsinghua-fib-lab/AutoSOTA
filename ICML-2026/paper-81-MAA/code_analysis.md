# Code Analysis for Paper 81 MAA - SOTA Optimization

## Evaluation Path
- Command: `python eval/VLMEvalKit/run.py --model maa_llava --data R-Bench-Dis R-Bench-Ref --judge gpt-4.1 --reuse`
- Model wrapper: `eval/VLMEvalKit/vlmeval/vlm/llava/llava_maa.py` (class `LLaVA_MAA`)
- Base model: `llava_maa.py` → `prepare_maa_model()` → `inject_maa_adapters()` → load adapter checkpoint
- Metric output: CSV at `outputs/maa_llava/T*/maa_llava_R-Bench-*_acc.csv` with "Overall" row

## Inference Path
1. `LLaVA_MAA.__init__()`: Loads base LLaVA, injects MAA adapters, loads adapter weights
2. `LLaVA_MAA.generate_inner()`: Process images → CLIP vision tower (with MAA adapters) → mm_projector → LLM generate
3. Key inference params: temperature=0.2 (default in kwargs), max_new_tokens=1024, force_anyres=True

## Config Path
- Adapter architecture constants: `maa/adapters.py:9-12` (MLP_HIDDEN_DIM=256, ATTN_DIM=128, NUM_HEADS=8, WINDOW_SIZE=7)
- Training defaults: `maa/train_maa.py:390-420` (parse_args)
- Adapter weights: `/repo/checkpoints/maa/maa.pth` (97MB, from acnul/maa)

## Metric Parser
- R-Bench-Dis_Accuracy: Parse CSV "Overall" row, multiply by 100
- R-Bench-Ref_Accuracy: Same method
- CSV path: `outputs/maa_llava/T*/maa_llava_R-Bench-{Dis,Ref}_acc.csv`

## Training Data
- `acnul/maa-datasets` is PRIVATE/GATED on HuggingFace
- NOT available locally; training-based ideas are NOT feasible

## Safe Modification Targets (inference-only)
1. **`eval/VLMEvalKit/vlmeval/vlm/llava/llava_maa.py`**: Temperature, max_new_tokens, generation params
2. **`maa/adapters.py`**: WindowSelfAttention can be modified to store attention weights for token pruning
3. **`maa/modeling.py`**: inject_maa_adapters can skip layers (shallow-only injection)
4. **Environment variables**: MAA_TEMPERATURE, MAA_MAX_TOKENS can parameterize generation

## Risky Files (do not modify)
- `eval/VLMEvalKit/run.py` - evaluation harness
- `eval/VLMEvalKit/vlmeval/dataset/` - dataset definitions
- `eval/VLMEvalKit/vlmeval/vlm/base.py` - base model class
- R-Bench data in `/root/LMUData/`

## Constraints
- Cannot retrain (no training data access)
- Must preserve evaluation protocol
- Must report both Dis and Ref metrics
- Must track latency/GFLOPs guardrails
