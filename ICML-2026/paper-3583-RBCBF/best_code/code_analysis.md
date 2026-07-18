# SOTA Preparation Repair — Paper 3583 (RBCBF)

## Original Preparation Failure

The orchestrator failed to prepare the SOTA environment because:
1. The reusable reproduction container (`autosota_repro_paper_3583`) had no `git` installed.
2. `apt-get install git` failed due to Ubuntu archive proxy issues (502 Bad Gateway, Undetermined Error).
3. The SOTA container (`autosota_sota_paper_3583`) was started successfully from `autosota/paper-3583:reproduced`, but faced the same `git` + `apt-get` failure.
4. Network from inside the container is unreliable through the configured proxy.

## Repair Actions

1. **Git installation**: Copied `/usr/bin/git` from host (glibc 2.31 compatible with container) to `/usr/bin/git` in the container.
2. **Tools setup**: Copied `record_score.sh` to `/tools/record_score.sh` in the container.
3. **Git repo initialization**: Initialized git in `/repo`, committed baseline state, tagged `_baseline` and `_best`.
4. **Baseline verification**: Ran the evaluation command on 50 WildJailbreak harmful prompts and confirmed Dterm=0.597 (triggered only, N=10/50), trigger_rate=0.2 — matching the reproduction manifest exactly.

## Corrected In-Container Evaluation Command

```bash
cd /repo
python3 scripts/run_rbcbf.py \
  --config configs/rbcbf_paper.json \
  --prompts /datasets/wildjailbreak_harmful_50.json \
  --output runs/eval_output.jsonl \
  --max_new_tokens 256 \
  --verbose 1 \
  --seed 2026 \
  --temperature 1.0 \
  --top_p 1.0
```

Metrics computed via: `python3 scripts/compute_metrics.py runs/eval_output.jsonl`

**Note**: The manifest eval_command specifies `wildjailbreak_eval_all.json` (2210 prompts), but the reproduction baseline was validated on `wildjailbreak_harmful_50.json` (50 prompts). Optimization iterations use the 50-prompt subset for speed (about 20 min/run). Full evaluation on all 2210 prompts can be run for the final best candidate (estimated 8 hours).

## Baseline Evidence

| Metric | Manifest | Verified | Match |
|--------|----------|----------|-------|
| Dterm (triggered) | 0.597 | 0.597 | Yes |
| Trigger rate | 0.20 | 0.20 | Yes |
| N triggered | 10/50 | 10/50 | Yes |

## Available Resources

- **Models**: Qwen2.5-7B-Instruct (generator) at `/models/Qwen2.5-7B-Instruct/`, Qwen2-0.5B-Instruct (scorer) at `/models/Qwen2-0.5B-Instruct/`
- **Datasets**: WildJailbreak subsets at `/datasets/` (50, 100, 200 harmful prompts, plus full 2210)
- **Cache**: `/autosota_cache/` for HF, torch, pip, conda caches
- **No paper_data mount**: All paper data is in `/datasets/` and `/models/`

## Safe Optimization Targets

The RBCBF controller's safety gate is the primary optimization surface:

1. **h-score computation**: The function that aggregates scorer hidden states into the terminal gate signal. Currently uses uniform mean over hidden dimensions plus epsilon comparison.
2. **Terminal gate threshold**: eps=0.5 in the scorer config. Can be tuned.
3. **Rollback behavior**: policy_window_W, continuous_steps, safe_token_bias_steps, safe_token_seeds.
4. **Scorer stride**: How often the scorer checks safety (every 2 steps currently).
5. **Directional reference init**: Post-rollback behavior control.

## Constraints

- Cannot modify metric definitions, scoring scripts, test data, labels, or benchmark outputs.
- Cannot hard-code predictions, metrics, or outputs.
- Must use `/tools/record_score.sh` for all score records.
- All code changes must happen inside the container at `/repo`.
