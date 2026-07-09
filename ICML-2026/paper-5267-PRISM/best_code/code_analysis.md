# Code Analysis — Paper 5267 SOTA Preparation Repair

## Original Failure

The orchestrator failed to prepare the SOTA container because:
1. Docker administrative policy rejected `--network host` in the first `docker run` attempt
2. The container proxy (`172.17.0.1:17890`) returned 502 Bad Gateway, preventing `apt-get install git`
3. Without git, the baseline commit and `_baseline` tag could not be created

## Repairs Applied

1. **Container startup**: Container `autosota_sota_paper_5267` from image `autosota/paper-5267:reproduced` started successfully without `--network host`
2. **Git installation**: Host git binary (`/usr/bin/git`, v2.25.1) copied into container as `/usr/bin/git` — all shared library dependencies already present in container
3. **record_score.sh deployment**: Copied from host to `/tools/record_score.sh` in container
4. **Baseline commit**: Created at initial repo state with `_baseline` tag
5. **Model download**: `google/gemma-3-4b-pt` downloaded via `hf-mirror.com` to `/models/google_gemma-3-4b-pt` (httpx monkey-patched to bypass SOCKS proxy issues)

## Corrected In-Container Evaluation Command

```bash
cd /repo && CUDA_VISIBLE_DEVICES=0,1 python3 -u train_eval.py \
  --dataset math10k --privacy dp --epsilon 6 \
  --base_model /models/google_gemma-3-4b-pt --seed 42 \
  --lora_r 16 --lora_alpha 16 --batch_size 64 --micro_batch_size 4 \
  --steps 300 --lr 3e-4 --dp_max_grad_norm 1.0 \
  --force_train --force_eval --no_resume
```

## Baseline Evidence

Baseline metrics from manifest (already recorded as iter 0 in scores.jsonl):
- GSM8K: 0.4443
- AQuA: 0.3976
- MAWPS: 0.8025
- SVAMP: 0.6070
- Average: 0.5629
- Epsilon: 5.944

## Reusable /paper_data Resources

- `/paper_data/datasets/glue8/` — GLUE evaluation data
- `/paper_data/datasets/math_10k/` — Math-10K training data
- `/paper_data/datasets/math_eval/` — Math evaluation data
- `/paper_data/glue/` — GLUE8 full datasets (ax, cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli)
- `/paper_data/google_gemma-3-12b-pt/` — Only README.md (no model weights)

## Safe Optimization Targets

Training parameters (require retraining ~90 min):
- `lora_r`: 8, 16, 32
- `lr`: 1e-4 to 1e-3
- `dp_max_grad_norm`: 0.5-2.0
- `prism_floor_factor`: 0.1-1.0
- `prism_floor_mode`: scalar, geometry
- `prism_lift_fix`: both, left, right
- `batch_size`: 32-128
- `micro_batch_size`: 1-8

Eval-only parameters (no retraining, ~10-40 min):
- `num_beams`: 4, 6, 8
- `max_new_tokens`: 256, 384, 512
- Self-consistency sampling (modifies evaluate.py)

## Key Code Locations

| File | Lines | Purpose |
|------|-------|---------|
| `train_eval.py` | 1-117 | CLI entry point |
| `src/prism_cli/trainers.py` | 1-321 | RunConfig, training loop, data loading |
| `src/prism_cli/optim/prism.py` | 1-922 | PRISM optimizer (tangent-space DP) |
| `src/prism_cli/eval_math.py` | 1-87 | Math-10K evaluation orchestration |
| `LLM-Adapters/evaluate.py` | 1-445 | LLM inference + evaluation |

## Known Issues

1. Line 155 in trainers.py: DataLoader seed skipped for math10k+prism → IDEA-08
2. Line 48 in trainers.py: `dp_debias_second_moment` defaults to False → IDEA-09
3. Line 306 in trainers.py: `_rebase_prism_for_save` may silently fail under Opacus → IDEA-10
