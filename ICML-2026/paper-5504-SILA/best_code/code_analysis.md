# Code Analysis — Paper 5504 (SelfIE Adapters)

## Evaluation Path

1. **Entry**: `evals/generation_scoring/run_eval.py` — parses JSON config, loads model + SAE, runs evaluation
2. **Config**: `evals/generation_scoring/configs/fast_eval_50.json` — 50 latents, 6 scales, 5 reward samples
3. **Label generation**: `evals/generation_scoring/evaluation_functions.py:evaluate_label_generator()` — generates labels via soft-prompt projection, then rescore phase
4. **Label generator**: `evals/generation_scoring/label_generator.py:LabelGenerator` — adapter-based projection + LM generation
5. **Reward system**: `evals/generation_scoring/reward_system.py:SAERewardSystem` — nnsight-based SAE activation measurement
6. **Metric computation**: `evals/generation_scoring/compute_mean_max_hit_rate.py` — mean-max hit rate across latents

## Key Files

| File | Role | Safe to modify? |
|------|------|----------------|
| `evals/generation_scoring/configs/fast_eval_50.json` | Eval config (scales, latent count, etc.) | Yes — sweep parameters |
| `evals/generation_scoring/label_generator.py` | Adapter wrapper, LabelGenerator class | Yes — CODE-01, CODE-04 changes |
| `evals/generation_scoring/evaluation_functions.py` | Main eval loop, label generator eval | Yes — algorithmic changes |
| `evals/generation_scoring/compute_mean_max_hit_rate.py` | Hit rate computation | **NO** — metric definition |
| `selfie_adapters/projection.py` | Adapter architecture (ScalarAffine+LR) | Yes — ALGO-04 changes |
| `selfie_adapters/inference.py` | Adapter loading (SelfIEAdapter) | Yes — loading changes |
| `training/data.py` | Training dataset | Yes — ALGO-02, ALGO-03 |
| `training/model.py` | Training model + loss | Yes — ALGO-01, ALGO-07 |
| `training/trainer.py` | Training loop | Yes — ALGO-03 sampler |
| `training/train.py` | Training entry point | Yes — multi-seed (ALGO-05) |
| `training/configs/scalar_plus_low_rank_8b.yaml` | Training config | Yes — PARAM-01 |

## Normalization Pipeline (CODE-01 Analysis)

- Training: adapter has `normalize_input=true` → adapter normalizes before projection
- Eval: `evaluation_functions.py` L2-normalizes before scaling, then adapter is called with `normalize_input=False` to avoid double-normalization
- **Verdict**: Not a bug — evaluation_functions.py handles normalization at lines 437-441 BEFORE the scale multiplication, so `normalize_input=False` in the adapter is correct to prevent double-normalization.
- The config field `normalize_vectors` (in `fast_eval_50.json`) controls whether normalization happens in eval — currently set to `true`

## Coverage Computation

Coverage is not computed by a separate script. Need to compute from the results JSON:
- For each latent: check if any generated context produces nonzero SAE activation at any token position
- Coverage = fraction of latents with ≥1 hit across all generated labels/contexts

## Current Metrics (Baseline)

- Hit Rate: 46.0% (paper CI: [43.8, 45.4])
- Coverage: 66.0% (paper CI: [67.8, 69.4])

## Safe Modification Targets (Priority Order)

1. **Scale values in eval config** (CODE-02): Pure inference-time change, no training needed
2. **Number of reward samples** (`num_reward_samples`): 5→10 reduces variance
3. **Number of latents** (`max_latents`): 50→100 for better estimate
4. **max_new_tokens**: 50→100 for longer conversations
5. **num_labels_per_scale**: 1→2 for more label diversity
6. **Adapter retraining** (ALGO-01 through ALGO-05, PARAM-01): Requires training infrastructure
