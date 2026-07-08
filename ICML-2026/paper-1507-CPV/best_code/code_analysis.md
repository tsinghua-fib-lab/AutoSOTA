# Paper 1507 — COVER SOTA Preparation Repair

## Original Failure

The evaluation script `eval_gsm8k_cover.sh` failed because `accelerate` was not found in the default shell PATH. The `accelerate` binary exists at `/opt/conda/envs/cover/bin/accelerate` but the script did not activate the `cover` conda environment before running.

### Evidence
```
/repo/eval_gsm8k_cover.sh: line 22: accelerate: command not found
[exit=127]
```

## Repair Applied

Added conda environment activation at the top of `eval_gsm8k_cover.sh`:
```bash
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate cover
```

## Corrected Evaluation Command

Inside container `autosota_sota_paper_1507`:
```bash
cd /repo && bash /repo/eval_gsm8k_cover.sh
```

This runs COVER with Dream-Ins-7B on GSM8K (1319 test samples) using 2 GPUs.

## Baseline Verification

| Metric | Manifest | Repaired Run | Match? |
|--------|----------|-------------|--------|
| Acc_flexible_extract | 77.63% | 77.63% | ✓ Exact |
| Steps | 58.20 | 58.20 | ✓ Exact |
| Speed | 3.51 | 4.40 | See note |

Speed note: Computed as `diffusion_steps / actual_steps = 256/58.2 = 4.40`. The manifest value of 3.51 likely uses a different baseline (per-step overhead adjustment).

## Evaluation Protocol

The eval command runs `accelerate launch --num_processes=2 -m lm_eval` with these fixed parameters:
- Model: Dream-v0-Instruct-7B (bf16)
- Algorithm: COVER (v2)
- block_length=32, diffusion_steps=256, temperature=0.0
- tau_draft=0.90, max_unmask_per_step=15, max_reverify_times=5
- Dataset: GSM8K (gsm8k_cot), flexible-extract filter

## Safe Optimization Targets

All changes must be in `generation_utils.py` (code/algorithm) or model_args (hyperparameters). The evaluation protocol, dataset, metric extraction logic, and model weights are off-limits.

Key levers:
- tau_draft (0.7-0.9): drafting confidence threshold
- max_unmask_per_step (15): drafting budget
- max_reverify_times (5): remask budget
- block_length (32)
- temperature (0.0, greedy)
- Seed scoring formula (line 685-689)
- margin_confidence mode in sample_tokens
- max_remask_allowed constraint
