# Code Analysis for Paper 45: FOAM SOTA Preparation Repair

## Original Preparation Failure

The SOTA preparation failed because:
1. **Git not installed in container**: The `autosota/paper-45:reproduced` Docker image does not include `git`. The preparation script tried `apt-get install git`, but the proxy (172.17.0.1:17890) blocks apt repositories with HTTP 502 errors.
2. **Container networking issue**: First attempt with `--network host` was rejected by Docker auth plugin. Second attempt without `--network host` succeeded but apt-get still fails through proxy.
3. **Fix**: Copied host `git` binary (`/usr/bin/git`) to `/usr/bin/git` in container — compatible since both host and container use Ubuntu 20.04 with compatible glibc.

## Repaired Evaluation Command

The reproduction command is runnable inside the container:
```bash
cd /repo
bash run_reproduction.sh
```

This runs:
```bash
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 1e-2 \
    --scale 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_ratio 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 100000 \
    --level 2 \
    --seed 42 \
    --beta1 0.9 \
    --beta2 0.95 \
    --optimizer foam \
    --save_dir foam2_llama60m_repro
```

## Baseline Verification

The reproduction (run_reproduction.sh) was successfully completed before the repair:
- **Final eval loss**: 3.3655937910079956
- **Perplexity**: exp(3.3656) = 28.95
- **Memory**: 0.27 GB (analytical estimate from paper)
- This matches the manifest baseline of PPL 28.95.

The C4 dataset is loaded via streaming from hf-mirror.com (HF_ENDPOINT configured). No local data download needed.

## /paper_data Resources

The `/paper_data` directory contains pre-downloaded LLM models (Llama-3.2-3B, Qwen2.5-7B, gemma-3-1b-it, roberta-large). These are for fine-tuning tasks (run_glue.py) and are NOT used for the C4 pretraining benchmark. The C4 pretraining trains LLaMA-60M from scratch — no pre-trained weights needed.

## Safe Optimization Targets

### Code-level changes implemented:
1. **WSD Scheduler** (CODE-02): Added to `peft_pretraining/training_utils.py` — `get_wsd_schedule_with_warmup()` function with warmup-stable-decay phases
2. **Scheduler support**: Updated `torchrun_main.py` to accept `--scheduler wsd` and `--stable_ratio` argument

### Flag-only changes (no code mods needed):
3. **Activation Checkpointing** (CODE-01): `--activation_checkpointing` flag already exists in code
4. **Gradient Clipping** (CODE-03): `--grad_clipping` arg exists, default 0.0
5. **Fold level**: `--level` arg already parameterized (2→1 for CODE-01)

### Optimizer-level changes (foam_torch/foam_adam.py):
6. **Per-layer adaptive res_scale** (ALGO-04): Modify foam_adam.py step() to compute dimension-aware res_scale
7. **beta2/scale tuning** (PARAM-01): Config-only changes

## Red Lines (inviolable)
- No modification to evaluation data, metric computation, or validation protocol
- No hard-coding of predictions or metrics
- No modification of C4 dataset loading or tokenizer
- No changes to model architecture config
