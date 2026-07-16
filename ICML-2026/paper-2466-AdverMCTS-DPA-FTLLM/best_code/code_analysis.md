# Code Analysis — Paper 2466 SOTA Optimization

## Evaluation Path
- **Script**: `src/probe_watermarks_batch_fast.py`
- **Command**: `CUDA_VISIBLE_DEVICES=0 python3 src/probe_watermarks_batch_fast.py --data_base /repo/data --wm_version xp2-2K-seed42 --model_id /models/mistral-7b-v0.1 --adapter_path /repo/data/outputs/model_A/lora-single-xp2-2K-seed42-seed42/final_adapter --batch-size 4 --skip_if_exists 0`
- **Timeout**: 15 minutes
- **Output**: JSON at `/repo/data/probe_outputs/_models_mistral-7b-v0.1/outputs/model_A/lora-single-xp2-2K-seed42-seed42/summary_xp2-2K-seed42.json`
- **Primary metric**: `overall.second_match_rate` → Chunk Hit Probability (p)
- **Guardrail metric**: `pFN = (1-p)^40` (40 = number of watermarked documents per user)
- **Generation config** (lines 158-165): `temperature=0.7`, `top_p=0.9`, `top_k=50`, `max_new_tokens=200`, `do_sample=True`
- **Seeds**: `[11, 22, 33]`

## Training Path
- **Script**: `src/train_fast_full.py`
- **Baseline command** (derived from training logs):
  ```
  python3 src/train_fast_full.py --model_path /models/mistral-7b-v0.1 --data_base /repo/data --wm_version xp2-2K-seed42 --model_tag model_A --seed 42 --deterministic
  ```
- **Key hyperparameters** (lines 120-125, 338-350, 372):
  - `lora_r=12`, `lora_alpha=32`, `lora_dropout=0.05`
  - `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`
  - `lr=2e-4`, `warmup_ratio=0.03`, `cosine` schedule
  - `num_train_epochs=3`, `per_device_batch_size=2`, `grad_accum=8`
  - `optim="paged_adamw_8bit"`, `bf16=True`
  - `EarlyStoppingCallback(patience=2)`
- **Output**: Adapter at `/repo/data/outputs/{model_tag}/lora-single-{wm_version}-seed{seed}/final_adapter/`
- **Training time**: ~17 minutes on A100 80GB

## Watermark Generation
- **Script**: `src/watermark.py`
- **Config**: `CLUSTER_SIZE=4`, `SPACING=4`, `HALF_SYMBOLS=16`, `PAIR_SYMBOLS=32`
- **Baseline args**: `--source blog1k --wm-per-user 40 --train-size-total 1000 --max-tokens 0`
- **Output**: `/repo/data/xp2-2K-seed42/train.jsonl`

## Config Path
- LoRA config embedded in `train_fast_full.py` lines 338-347
- Generation config embedded in `probe_watermarks_batch_fast.py` lines 158-165
- Watermark config in `watermark.py` class-level constants

## Metric Parser
- Parse `summary_{WM_VERSION}.json` → `overall.second_match_rate`
- Compute `pFN = (1 - p)^40`
- Baseline: p=0.658, pFN=2.21e-19

## Reusable Resources
- `/models/mistral-7b-v0.1` — base model (~14GB)
- `/repo/data/xp2-2K-seed42/` — existing watermarked dataset (wm-per-user=40)
- `/repo/data/xp2-2K-seed42/train_tok_model_A_seed42/` — cached tokenized dataset
- `/repo/data/outputs/model_A/lora-single-xp2-2K-seed42-seed42/final_adapter/` — baseline adapter

## Safe Modification Targets
- `src/probe_watermarks_batch_fast.py` GEN_KW_BASE (temperature, top_p, top_k)
- `src/train_fast_full.py` LoRA config, training args, target_modules
- `src/watermark.py` wm-per-user, CLUSTER_SIZE, SPACING
- `src/train_fast_full.py` add custom Trainer.compute_loss (token weighting)
- `src/train_fast_full.py` add memory tokens (prepend to embeddings)

## Risky Files (DO NOT MODIFY)
- `src/alphabet.txt` — ZWS alphabet definition
- Metric computation in `probe_watermarks_batch_fast.py` agg() function
- Dataset split logic (train_test_split with seed)
- Scoring script `/tools/record_score.sh`
- Test data at `/repo/data/xp2-2K-seed42/`

## Red Lines Confirmed
- No changes to eval metric definitions
- No changes to test data, labels, splits
- No hard-coded metric values
- All modifications are training hyperparameters or generation config
