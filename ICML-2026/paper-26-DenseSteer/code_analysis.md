# Code Analysis — DenseSteer (Paper 26) SOTA Optimization

## Repository Structure

| Path | Role |
|---|---|
| `/repo/lm-evaluation-harness/lm_eval/models/steer_hf.py` | **Main optimization target** — Custom HF model wrapper with steering vector injection |
| `/repo/lm-evaluation-harness/lm_eval/models/huggingface.py` | Base HF model (line 800: padding fix applied) |
| `/repo/01_extract_vectors.py` | Vector extraction (not needed unless re-extracting) |
| `/repo/02_apply_vectors.py` | Script for grid-search eval (reference only) |
| `/repo/vectors/dense-rewritten-vectors/` | Pre-extracted steering vectors for multiple models |
| `/repo/vectors/inFam-vectors/` | Alternative vector family (in-family outputs) |

## Evaluation Path

- **Command**: `cd /repo/lm-evaluation-harness && python3 -m lm_eval --model steer_hf --model_args "..." --tasks gsm8k_cot_zeroshot --batch_size 8 --output_path /repo/eval_results/densesteer --log_samples`
- **Model**: Qwen2.5-3B-Instruct via `steer_hf` custom wrapper
- **Task**: `gsm8k_cot_zeroshot` (1319 test samples)
- **Params**: `pretrained`, `dtype=float16`, `steer_layer`, `steer_lambda`, `steer_vec_path`
- **Runtime**: ~20 min on 2 GPUs for full eval
- **Metric parsed from stdout**: `exact_match,flexible-extract` value (also in JSON at output_path)

## Steering Vector Structure

- **Format**: `SteeringVector` object with `layer_activations: dict[int, Tensor[2048]]`
- **Available layers**: 0-35 (36 layers for Qwen2.5-3B)
- **Norm variation**: 0.06 (L0) → 3.94 (L35) for dense-rewritten; 11.5 (L0) → 71.6 (L35) for inFam
- **Current usage**: Single layer (L17) filtered from full vector at load time

## Safe Modification Targets

1. **`steer_hf.py`**: Add multi-layer support (`steer_layers` parameter), L2 normalization, lambda scheduling, bidirectional steering
2. **`01_extract_vectors.py`**: Pair weighting/reweighting if re-extraction needed
3. **Custom eval scripts**: Self-consistency, layer profiling

## Risky Files (DO NOT MODIFY)

1. `lm_eval/tasks/gsm8k/` — Task definitions, metric computation
2. `/tools/record_score.sh` — Scoring infrastructure
3. `/autosota_artifacts/paper-26/sota/scores.jsonl` — Score records (write only via record_score.sh)

## Critical Setup Notes (from manifest)

1. **transformers 4.40.2** — Required for PyTorch 2.1.0 compat + Qwen2 support
2. **padding_side='right'** fix in `huggingface.py:800` — CRITICAL: left-padding causes '!' token output
3. **`--apply_chat_template` defaults to True** in this fork (differs from upstream)
4. **peft 0.7.1**, **steering-vectors 0.12.2**

## Known Levers

| Lever | Range | Notes |
|---|---|---|
| `steer_layer` | 0-35 | Paper says L16/L17 are best |
| `steer_lambda` | -14 to 14 | Paper shows monotonic improvement for L17 |
| `steer_min_token` | int | Delay steering start (default 0) |
| `steer_vec_path` | path | Two families: dense-rewritten, inFam |
| `model_scale` | 1.5B/3B/7B | Different Qwen2.5 variants available |

## Optimization Objective

- **Primary**: Maximize `gsm8k_exact_match_flexible_extract` (baseline: 81.43%)
- **Secondary**: Monitor `gsm8k_exact_match_strict_match` (baseline: 4.78%)
- **Type**: `representative_primary_metric` — improve primary, tradeoffs allowed
