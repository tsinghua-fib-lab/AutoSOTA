# Optimization Results: LLaDA-V — Large Language Diffusion Models with Visual Instruction Tuning

## Summary
- **Total iterations**: 5 (1 successful, 2 failed, 1 no-effect, 1 aborted)
- **Best `mmmu_acc`**: **48.33%** (baseline: 48.11%, improvement: +0.22%)
- **Paper-reported baseline**: 48.67%
- **Target**: 51.10% (5% improvement) — **NOT REACHED**
- **Best commit**: f7c603d3f1 (iter-1)

## Results Trajectory

| Iter | Idea | Type | Accuracy | Delta | Status |
|------|------|------|----------|-------|--------|
| 0 | Baseline (gen_steps=2, gen_length=2) | — | 48.11% | — | baseline |
| 1 | gen_steps=4, gen_length=8, block_length=8 | PARAM | 48.33% | +0.22 | success |
| 2 | CFG bug fix + cfg=0.5 | CODE | — | — | failed (too slow) |
| 3 | Vision resolution (stride=1) | PARAM | — | — | failed (OOM) |
| 4 | Margin-based confidence remasking | ALGO | 48.11% | 0.00 | no effect |
| 5 | Re-test gen_params=4/8/8 | PARAM | — | — | aborted |

## Baseline vs. Best Metrics

| Category | Baseline (48.11%) | Best (48.33%) | Delta |
|----------|-------------------|---------------|-------|
| Art and Design | 52.50% | 54.17% | +1.67 |
| Business | 51.33% | 48.67% | -2.66 |
| Science | 32.00% | 35.33% | +3.33 |
| Health and Medicine | 46.00% | 45.33% | -0.67 |
| Humanities and Social Science | 71.67% | 70.83% | -0.84 |
| Tech and Engineering | 42.86% | 43.33% | +0.47 |
| **Overall** | **48.11%** | **48.33%** | **+0.22** |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Fixed `arg.split("=")` bug in `simple_parse_args_string` | Enabled eval to run | Changed to `split("=", 1)` — necessary infrastructure fix |
| Increased gen_steps 2→4, gen_length 2→8, block_length 2→8 | +0.22% overall | Science improved most (+3.33), but Business/Humanities declined |
| CFG bug fix (cfg→cfg_scale mapping) + cfg=0.5 test | Aborted | CFG doubled eval time (~8s/it); research says CFG=0 optimal for MMLU-like tasks |
| mm_spatial_pool_stride=1 (full vision resolution) | OOM | 4× vision tokens → Softmax float64 OOM at 3.4GB |
| Margin-based confidence P(top1)-P(top2) | No effect | 48.11% exactly — zero impact for MC questions |

## What Worked

1. **Increased gen_length + gen_steps**: Only change that showed any improvement (+0.22%). Science category benefited most (+3.33pp), suggesting longer answers help with computation-heavy questions.
2. **The arg parsing fix**: Essential for the eval to run at all. The `model_args` format with nested `=` signs required `split("=", 1)` instead of `split("=")`.

## What Didn't Work

1. **Parameter tuning in general**: The model appears to be at or near its capability ceiling for MMMU. Neither gen_steps nor gen_length changes moved accuracy by more than 0.22 points.
2. **think_mode**: Caused catastrophic mode collapse — the model generated repetitive gibberish ("the the the...", "breathing breathing..."). The 2-8 diffusion steps per block were insufficient to denoise the longer chain-of-thought output.
3. **CFG (classifier-free guidance)**: Made evaluation prohibitively slow (2× forward passes) and the research literature indicates CFG=0 is optimal for MMLU-style multiple choice tasks.
4. **Vision resolution increase**: Caused OOM due to 4× vision tokens exploding the Softmax float64 computation in the diffusion process. The current implementation casts logits to float64 for numerical stability, making high-resolution vision infeasible on 80GB GPUs.
5. **Remasking strategy changes**: Margin-based confidence produced identical results to standard low-confidence remasking. For multiple-choice questions with clear answer preferences, both metrics select the same tokens.

## Root Cause Analysis

The fundamental limitation is that **LLaDA-V's MMMU accuracy is capability-bound, not generation-bound**. The model's underlying multimodal understanding determines accuracy — generation parameter tuning can only recover what the model already knows. The paper reports 48.67% MMMU accuracy; our baseline was 48.11% (likely due to gen_length=2 truncation for open-ended questions). The maximum we achieved was 48.33%.

The gap from 48.33% to 51.10% (target) represents a 2.77 percentage point improvement that would require:
- **Training-time improvements**: VRPO alignment (LLaDA 1.5), GRPO/RLVR post-training (PeRL-VL), or high-quality data curation
- **Architecture changes**: MoE (LLaDA-MoE), higher resolution vision (requires float32 softmax optimization), or latent refinement decoding
- **Ensemble methods**: Best-of-N majority voting (but eval time becomes prohibitive with diffusion models)

None of these are achievable with inference-only changes under the current constraints (no retraining, no dataset modification).

## Top Remaining Ideas (for future runs)

1. **Latent Refinement Decoding (LRD)**: Implement two-stage soft+hard diffusion from arXiv:2510.11052. This is the most promising inference-time improvement that could yield 3-6 points but requires ~200-500 lines of new code.
2. **Best-of-N majority voting**: With temperature=0.5 and random remasking, generate 5-7 answers per question and majority vote. Requires ~5-7× eval time but could yield 2-5 points.
3. **VRPO alignment training**: Apply Variance-Reduced Preference Optimization from LLaDA 1.5 (arXiv:2505.19223). Requires training infrastructure.
4. **Per-category parameter optimization**: Tune gen_steps, temperature, and think_mode per MMMU domain. Science/Tech may benefit from more steps; Humanities may work with fewer.
5. **Float32 softmax optimization**: Replace float64 Softmax in diffusion process with float32 to enable higher resolution vision features. Requires careful numerical stability testing.
6. **Dynamic temperature scheduling**: Decreasing temperature from 0.5→0.0 across diffusion steps for better exploration-exploitation balance.
