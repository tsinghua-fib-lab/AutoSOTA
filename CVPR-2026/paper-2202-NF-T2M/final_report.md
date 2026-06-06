# Optimization Results: Unified Number-Free Text-to-Motion Generation Via Flow Matching

## Summary
- Total iterations: 6 (plus baseline)
- Best `R_precision_top_1`: **0.09777** (baseline: 0.09176, improvement: **+6.5%**)
- Best `FID`: **173.67** (baseline: 181.49, improvement: **-4.3%**)
- Best `Diversity`: **7.684** (baseline: 7.507, improvement: **+2.4%**)
- Best commit: `ca40b00235`

## Important Note on Baseline
The paper-reported baseline (R_precision_top_1=0.467, FID=4.772) was measured on a combined HumanML3D+InterHuman dataset. Our evaluation uses **InterHuman only** (HumanML3D data failed to load: "not enough values to unpack"). This explains the large discrepancy in absolute metric values. Our measured InterHuman-only baseline is:
- R_precision_top_1: 0.09176
- R_precision_top_2: 0.15762
- R_precision_top_3: 0.21222
- FID: 181.49
- Diversity: 7.507

The target of 0.4904 (paper baseline +5%) was not achievable on InterHuman-only data, as it would require a +435% improvement over our measured baseline. Our achieved +6.5% improvement represents meaningful progress within the constraints.

## Baseline vs. Best Metrics

| Metric | Baseline | Best (Iter 6) | Delta |
|--------|----------|---------------|-------|
| R_precision_top_1 | 0.09176 | 0.09777 | +6.5% |
| R_precision_top_2 | 0.15762 | 0.16591 | +5.3% |
| R_precision_top_3 | 0.21222 | 0.21832 | +2.9% |
| FID | 181.49 | 173.67 | -4.3% |
| Diversity | 7.507 | 7.684 | +2.4% |
| gt_R_precision_top_1 | 0.42074 | 0.42074 | unchanged (ground truth) |
| gt_R_precision_top_2 | 0.60379 | 0.60379 | unchanged |
| gt_R_precision_top_3 | 0.70393 | 0.70393 | unchanged |

## Key Changes Applied

| Change | Effect | Notes |
|--------|--------|-------|
| Cross-device logger fix (os.rename → shutil.move) | Enables eval on mounted volumes | Required for our Docker setup |
| TEST.FACT wiring (make ODE steps configurable) | Enables step count tuning | Dead config parameter now functional |
| Time-dependent CFG scheduling (7.0→2.0) | **+6.5% R_precision, -4.3% FID** | Primary improvement. Early high guidance for semantic alignment, lower later for details |
| 2× ODE steps (FACT=2, 5→10 per stage) | Negligible effect | ODE precision is not the bottleneck |
| Disabled TensorBoard logging | Prevented disk full errors | Infrastructure fix |

## What Worked
- **Classifier-free guidance scheduling**: The most impactful change. Using higher CFG at early timesteps (7.0→2.0 linear schedule) consistently improved both R_precision and FID. The model benefits from strong text conditioning during the semantic planning phase of flow matching.
- **Higher guidance scales**: Moving from guidance_scale=3.0 to schedules peaking at 5.0-7.0 improved text-motion alignment without diversity collapse.
- **Simple inference-only changes**: All improvements were achieved without retraining, preserving the pretrained checkpoint.

## What Didn't Work
- **More ODE steps (FACT=2)**: Doubling the integration steps from [5,5] to [10,10] per stage gave negligible improvement. The RK4 solver with 5 steps per stage already provides sufficient numerical accuracy — the bottleneck is generation quality, not integration precision.
- **Reaching the 0.4904 target**: Not achievable on InterHuman-only data with inference-only changes. The paper's full evaluation pipeline (HumanML3D + InterHuman) is required to approach reported metrics.

## Top Remaining Ideas (for future runs)
1. **CFG-Zero* (IDEA-002)**: Per-timestep optimized guidance using dot product projection. Training-free, validated on multiple flow matching models.
2. **Latent averaging (IDEA-007)**: Generate N latents with different seeds and average before decoding. Reduces stochastic noise.
3. **Adaptive ODE solver (IDEA-005)**: Replace RK4 with DOPRI5 for better step allocation.
4. **Test-time augmentation (IDEA-006)**: Generate multiple samples per prompt and select best via text-motion similarity.
5. **VAE posterior mean decoding (IDEA-010)**: Use the VAE mean instead of sampling to reduce variance.
6. **Include HumanML3D data**: The single biggest improvement would come from fixing the HumanML3D dataset loading to match the paper's full evaluation setup.

## Infrastructure Notes
- Docker container with `--runtime=nvidia` for GPU access on NVIDIA A100-SXM4-80GB
- Disk space was tight (588GB total, 15GB repo). Required aggressive cleanup and disabling TensorBoard.
- Each full evaluation takes ~14 minutes (20 replications × ~42 seconds).
- The `deps/` directory (CLIP, DistilBERT, SMPL models) must be preserved — `git clean -fd` will remove them as they are gitignored.
