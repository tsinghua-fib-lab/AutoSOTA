# Optimization Results: Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models

## Summary
- **Total iterations**: 7
- **Best `acc`**: **61.20%** (baseline: 60.04%, improvement: **+1.16%**)
- **Paper-reported baseline**: 57.75% (improvement over paper: **+3.45%**)
- **Target**: 60.6375% ✅ **EXCEEDED**
- **Best commit**: `d1182889e49fd8e0540894c91aa34db0493a7316`

## The Winning Change
**Switch VRGA from `fa=1` (neighborhood boost) to `fa=2` (binary mask)** — a 2-line parameter change.

| File:Line | Change |
|-----------|--------|
| `models/modeling_qwen2_5_vl.py:878` | `fa = 1` → `fa = 2` (attention boost mode) |
| `models/modeling_qwen2_5_vl.py:1575` | `fa = 1` → `fa = 2` (focus token selection mode) |

### What fa=2 does differently from fa=1:
- **fa=1**: Enhances focus token neighborhoods by multiplying attention weights by `multiply=1.5`
- **fa=2**: Creates a binary mask — keeps focus token neighborhoods (k=3), zeros out non-focus vision tokens entirely
- fa=2 uses a simpler threshold-based focus token selection (ratio > 0.6, threshold > 2.0x mean)
- fa=2 uses larger neighborhood radius (k=3 vs k=2)

## Baseline vs. Best Metrics

| Metric | Baseline | Best | Delta | Direction |
|--------|----------|------|-------|-----------|
| ACC (%) | 60.04 | **61.20** | **+1.16** | ↑ better |
| S (Comprehensive Score) | 0.4149 | **0.4302** | **+0.0153** | ↑ better |
| I (Irrelevance Degree) | 0.3089 | **0.2970** | **-0.0119** | ↓ better |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| fa=1→fa=2 (binary mask) | ACC +1.16 | The ONLY effective change. All metrics improved. |
| Other attempts (deterministic heads, multi-scale k, adaptive boost, residual blend, progressive decay) | All negative | -0.10 to -0.52 each. Original VRGA fa=1 params are well-tuned. |

## What Worked
- **fa=2 binary mask mode**: By completely removing non-focus vision tokens instead of just boosting focus regions, the model achieves cleaner visual grounding. The binary mask forces the model to rely on the most salient visual regions rather than being distracted by peripheral visual tokens.
- **Diagnostic evaluation without VRGA**: Confirmed VRGA provides +0.84 ACC gain (59.20 → 60.04), validating the approach.

## What Didn't Work
- **Deterministic head selection** (-0.10): Random head diversity helps robustness
- **Multi-scale neighborhood radii** (0.00): k=2 is already near-optimal
- **Adaptive per-head boost** (-0.42): Stronger intervention disrupts language generation
- **Residual attention blending** (-0.31): Weakening VRGA reduces its benefit
- **Progressive boost decay** (-0.52): Consistent visual boost throughout generation is essential

## Top Remaining Ideas (for future runs)
1. **Tune fa=2 parameters**: Adjust k=3, Rimg threshold, top_ratio for the fa=2 mode
2. **Hybrid fa modes**: Use fa=2 for early generation steps (strong visual grounding) and switch to fa=1 later
3. **Layer-specific fa modes**: Use fa=2 in early/middle layers and fa=1 in late layers
4. **Consensus-based focus token selection** (IDEA-005): Not yet tried with fa=2
5. **Per-category fa selection**: Use fa=2 for VS (chart) type questions and fa=1 for VD (figure) type
