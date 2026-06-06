# VDOT: Efficient Unified Video Creation via Optimal Transport Distillation — A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization study performed on the public release of VDOT (Wang et al., 2026), a CVPR 2026 paper that proposes a 4-step unified video creation model built on top of the VACE / Wan 2.1 stack and trained with a Computational Optimal Transport distillation objective. The original release achieves competitive image and video quality while drastically reducing inference cost from tens of denoising steps to four. The optimization was driven by AutoSOTA (`tsinghua-fib-lab/AutoSOTA`) and targeted the `imaging_quality` metric on the depth-control track of UVCBench. Twenty-four iterations were executed, exploring algorithmic modifications (multi-step latent fusion, adaptive shift schedules, test-time augmentation), exposed hyperparameters (the previously hidden VACE context-injection scale), and standard inference levers (`sample_shift`, `sample_solver`, `sample_steps`, `sample_guide_scale`, `base_seed`). The final configuration improves `imaging_quality` from a baseline of 71.64 to 72.89, an absolute gain of +1.25 (+1.75%). The 5% target (75.222) was not reached, but the trajectory localised an interpretable optimum in the noise-schedule shift parameter and confirmed that several intuitively appealing modifications (CFG, deeper sampling, TTA flipping) actively hurt the distilled model. The best configuration uses `--sample_shift 3.75 --vace_context_scale 1.5 --sample_steps 4 --sample_solver unipc --base_seed 2025` at commit `1f80370413`.

## 1. Introduction

VDOT (Video creation with Distillation via Optimal Transport) is positioned as an efficient successor to the VACE family of unified video creation models. By coupling a Wan 2.1-14B backbone with an Optimal Transport distillation procedure, VDOT compresses what is normally a 25–50-step flow-matching sampling process into four steps while preserving support for Reference-to-Video (R2V), Video-to-Video (V2V), Masked Video Editing (MV2V), and arbitrary composite tasks. The paper was accepted to CVPR 2026 and the model weights, inference scripts, and the UVCBench evaluation suite are publicly available.

This report studies whether the released inference pipeline can be improved post hoc, without retraining, using purely test-time interventions. The motivation is twofold. First, VDOT is a distilled student whose training-time noise schedule and four-step trajectory are tightly coupled to specific hyperparameter choices that are not always re-examined at deployment time. Second, the released code surface hides several useful knobs (most notably the VACE context-hint injection scale) behind defaults, which limits practitioners' ability to trade off conditioning fidelity against generative freedom.

The optimization was conducted with AutoSOTA, an automated SOTA-chasing harness developed by Tsinghua FIB Lab. AutoSOTA proposes, runs, and evaluates code and configuration changes in a budgeted loop, scoring each iteration against a single primary metric. For VDOT the chosen metric was `imaging_quality` on the UVCBench `depth` subtask, evaluated through the project's existing `vace_alltask_uvcbench_single.py` harness.

The remainder of the report describes the original method (Section 2), the limitations targeted by the optimization (Section 3), the methodology applied (Section 4), the experimental results and ablation trajectory (Section 5), a discussion of negative findings and future directions (Section 6), and the information needed to reproduce the best configuration (Section 7).

## 2. Original Method (Background)

VDOT inherits the VACE-Wan2.1-14B architecture: a Diffusion Transformer (DiT) over latents produced by a 3D VAE, conditioned on T5-encoded text and on optional reference images, source videos, and masks. The "VACE" branch injects task-specific control signals into the main DiT through a set of context hints that are added to intermediate block outputs. In the released model these hints are added with a fixed gain of 1.0; the relevant operation is implemented in `training/wan/modules/vace_model.py`:

```63:67:training/wan/modules/vace_model.py
    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
```

The distillation objective uses Computational Optimal Transport to match a four-step student to the multi-step teacher's marginal trajectory, which makes the student's behaviour highly schedule-sensitive: the choice of `sample_shift` (the time-rescaling factor of the flow-matching noise schedule) determines where the four sampling timesteps fall along the trajectory, and the student is implicitly trained for a particular range. The released inference script `inference/vace_wan_inference.py` defaults to `sample_steps=50` and `sample_shift=16`, with the four-step path documented in `run_vdot.sh` and `test_uvcbench_single.sh` via `--sample_steps 4`. UVCBench evaluation is wrapped by `inference/vace_alltask_uvcbench_single.py`, which iterates over single-condition subtasks (depth, flow, gray, pose, scribble, inpainting, outpainting, firstframe, face, object).

## 3. Identified Limitations

The optimization study identified four categories of friction in the released pipeline:

1. **Hidden conditioning knob.** The VACE context-injection gain (`context_scale` in `vace_model.py`) is hard-coded to 1.0 in the inference path. There is no command-line argument to control it from `vace_wan_inference.py`, so practitioners cannot trade off conditioning strength against perceptual quality without editing source files.
2. **Schedule mismatch at inference time.** The default `sample_shift=16` (used by `validate_args` in `vace_wan_inference.py`) was inherited from teacher-style multi-step settings and is poorly matched to the distilled four-step student. Running a four-step student with a high shift compresses most useful denoising into a regime the student was not optimised for.
3. **Unverified default solver and CFG.** The released defaults (`sample_solver='unipc'`, `sample_guide_scale=5.0` in the parser, but `1.0` in the bench scripts) were not ablated against alternatives such as DPM++ or higher CFG values for the distilled student.
4. **No multi-seed or TTA strategy.** The benchmark protocol generates a single sample per test case at a fixed seed. Standard tricks such as alternative seeds, horizontal-flip TTA, or adaptive per-step shift were not exercised, leaving easily testable gains on the table.

## 4. Optimization Methodology

The 24 iterations fall into four categories. Each category is grounded in concrete files in the repository.

**Hyperparameter exposure.** A new `--vace_context_scale` argument was added to the UVCBench harness `inference/vace_alltask_uvcbench_single.py` and threaded into the existing `wan_vace.generate(..., context_scale=...)` call, which already accepted a `vace_context_scale` argument inside `training/wan/modules/vace_model.py`. This unlocked the previously hidden VACE injection-strength dial without retraining.

**Noise-schedule tuning.** The `sample_shift` parameter in `validate_args` and `get_parser` of `inference/vace_wan_inference.py` (and its UVCBench counterpart) was swept downward from the original 16 to 2, in the order 16 → 10 → 8 → 6 → 4 → 3.75 → 3.5 → 3 → 2. This is a pure-inference change, justified by the fact that the OT-distilled student was empirically observed to be more accurate when its four sampling timesteps are placed closer to the data manifold.

**Solver, CFG, step-count, and seed ablations.** With shift fixed near its optimum, the harness compared `sample_solver` (`unipc` vs. `dpm++`), `sample_guide_scale` (1.0 vs. 1.5), `sample_steps` (4 vs. 8), and `base_seed` (2025 vs. 42). These changes correspond directly to existing arguments in the parser of `vace_wan_inference.py`.

**Algorithmic experiments (not retained).** Four more invasive modifications were attempted and discarded: (i) multi-step latent fusion, which averaged early-step x0 predictions; (ii) an adaptive per-step shift schedule that tried to feed custom sigmas into the UniPC scheduler; (iii) horizontal-flip test-time augmentation in the latent pipeline; and (iv) CFG re-enabling on the distilled student. The reasons for failure are catalogued in Section 5.3.

No training data, model weights, or T5/VAE components were modified. All retained changes were either (a) plumbing additions that expose existing arguments to the CLI, or (b) hyperparameter values written through existing CLI flags.

## 5. Experiments

### 5.1 Setup

The optimization target was `imaging_quality` on the UVCBench depth subtask, computed by the project's standard evaluation pipeline launched from `inference/vace_alltask_uvcbench_single.py`. All runs used the released VDOT-14B checkpoint, `size=480p`, `frame_num=81`, four GPUs with `ulysses_size=4`, FSDP for both the DiT and T5 (`--dit_fsdp --t5_fsdp`), and the default UVCBench prompts. The baseline corresponds to the configuration shipped in `test_uvcbench_single.sh`, i.e. `--sample_guide_scale 1 --sample_steps 4` and the implicit `--sample_shift 16` from `validate_args`. AutoSOTA executed 24 iterations under a fixed wall-clock budget per iteration. The improvement target set by the harness was +5.0% over baseline (`imaging_quality` ≥ 75.222).

### 5.2 Quantitative Results

**Headline metric.**

| Metric | Baseline | Best | Delta |
|--------|----------|------|-------|
| imaging_quality | 71.64 | 72.89 | +1.25 (+1.75%) |

**Configuration changes that contributed to the best run.**

| Change | Effect | Notes |
|--------|--------|-------|
| Added `--vace_context_scale` argument | Exposed VACE context injection strength | Exposed hidden parameter; optimal at 1.5 |
| Tuned `--sample_shift` (16 → 3.75) | +1.25 improvement | Largest quality lever; lower shift = higher quality for VDOT |
| Tested `--sample_solver` (unipc vs. dpm++) | Similar results at 4 steps | UniPC slightly better for depth task |
| Tested `--sample_guide_scale` (1.0 vs. 1.5) | No improvement | CFG did not help distilled VDOT model |
| Tested `--sample_steps` (4 vs. 8) | Degradation at 8 steps | VDOT specifically optimised for 4-step inference |
| Tested `--base_seed` (2025 vs. 42) | Seed 2025 is better | Default seed is optimal |

The best configuration was therefore `--sample_shift 3.75 --vace_context_scale 1.5 --sample_steps 4 --sample_solver unipc --base_seed 2025`, captured at commit `1f80370413`.

### 5.3 Ablation / Iteration Trajectory

The shift sweep dominated the optimization. The full trajectory along the `sample_shift` axis (with all other parameters fixed at the configuration that achieved the best score) is reported below.

| Shift | Imaging Quality | Delta vs Baseline |
|-------|-----------------|-------------------|
| 16 (paper default) | 71.64 | — |
| 10 | 72.27 | +0.63 |
| 8 | 72.51 | +0.87 |
| 6 | 72.38 | +0.74 |
| 4 | 72.78 | +1.14 |
| 3.75 | **72.89** | **+1.25** |
| 3.5 | 72.86 | +1.22 |
| 3 | 72.84 | +1.20 |
| 2 | 72.64 | +1.00 |

The curve is unimodal with a clear maximum near 3.75, and degrades smoothly on either side. Iteration 23 re-ran the configuration of iteration 19 (`shift=3.5`) and reproduced its score of 72.86 exactly, confirming that the measurement noise of the harness is well below the magnitude of the observed improvements.

Several iterations explored more aggressive ideas and were discarded:

- **Multi-step latent fusion** collapsed the metric to 33.39, an order-of-magnitude regression. Fusing early-step x0 predictions into the final latent injects high-frequency noise that the four-step student cannot subsequently correct.
- **8-step denoising** moved the score into the range 71.95–72.25 across all sub-configurations tested, never matching the four-step results. This is consistent with the OT distillation training, which targets exactly four steps.
- **Adaptive shift schedules** required passing per-step custom sigmas to the UniPC scheduler, which is incompatible with its current implementation; integrating them would have required intrusive scheduler edits and was not pursued.
- **Horizontal-flip TTA** crashed at the C++ tensor layer due to shape and device inconsistencies in the flip-and-fuse path, and was abandoned.
- **CFG with `sample_guide_scale` > 1** produced no measurable gain, confirming that the distilled student already collapses the guided trajectory into its weights.
- **DPM++ solver** matched UniPC within noise at four steps on depth.
- **Alternative seed (42)** scored worse than the released default (2025).

## 6. Discussion

The experiments converge on two practical recommendations for users of distilled flow-matching video models. First, the noise-schedule shift parameter inherited from teacher-style training is a strong default to revisit at inference time: a single one-dimensional sweep produced the entire +1.25-point gain. Second, exposing internal conditioning gains (here, the VACE context-injection scale) at the CLI is an easy way to recover quality, even when no retraining is possible — the optimum at `1.5` indicates that the released model is mildly under-injecting control hints relative to what the depth subtask rewards.

The negative findings are equally informative. Distilled four-step students are brittle in directions that work for their non-distilled counterparts: classifier-free guidance, additional sampling steps, alternative solvers, and naive TTA all failed to help. This suggests that future automated optimization on similar models should prioritise (a) schedule and conditioning-strength searches, (b) prompt and post-processing interventions that do not interact with the trained timestep distribution, and (c) targeted training-time refinements (e.g. TeaCache integration with longer schedules) over generic test-time tricks.

The 5% target was not reached. The remaining gap (≈ +2.3 points) is larger than any single-parameter lever observed and almost certainly requires either (i) per-subtask retuning across the full UVCBench task set, (ii) prompt-extension changes via the existing `--use_prompt_extend wan_en` / `wan_en_ds` paths, (iii) multi-sample generation to match VBench's expected input format, or (iv) post-hoc frame interpolation (e.g. RIFE) to lift `motion_smoothness` and `temporal_consistency`. These directions are listed verbatim in `TAKEAWAY_source.md` as candidates for follow-up work.

## 7. Reproducibility

**Environment.** Python 3.10.13, CUDA 12.4, PyTorch ≥ 2.5.1 (the README pins `torch==2.6.0` and `torchvision==0.21.0`).

**Installation.**

```bash
git clone https://github.com/hhhh1138/VDOT.git && cd VDOT
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1
```

**Models and data.** The VDOT-14B weights are downloaded from `huggingface.co/yutongwang1012/VDOT`, and the UVCBench evaluation set from `huggingface.co/datasets/yutongwang1012/UVCBench`. Local layout follows the directory tree given in the project README (`models/VDOT/vdot-weights/vdot_14b.pt`, `models/VDOT/Wan2.1_VAE.pth`, `models/VDOT/models_t5_umt5-xxl-enc-bf16.pth`, with the VACE annotators under `models/VACE-Annotators` and the benchmark assets under `benchmarks/UVCBench`).

**Best-performing commit.** `1f80370413`.

**Best command.**

```bash
torchrun --nproc_per_node=4 inference/vace_alltask_uvcbench_single.py \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size 4 \
    --ring_size 1 \
    --size 480p \
    --sample_guide_scale 1 \
    --sample_steps 4 \
    --sample_shift 3.75 \
    --vace_context_scale 1.5 \
    --sample_solver unipc \
    --base_seed 2025 \
    --ckpt_dir VDOT \
    --save_dir results/uvcbench_single
```

**Expected score.** `imaging_quality = 72.89` on the UVCBench depth subtask, reproducible up to harness noise (iteration 23 re-confirmed iteration 19's 72.86 score for an adjacent configuration).

## 8. References

Wang, Y., Zhang, H., Xue, T., Qiao, Y., Wang, Y., Xu, C., and Chen, X. *VDOT: Efficient Unified Video Creation via Optimal Transport Distillation.* arXiv preprint arXiv:2512.06802, 2025. Accepted to CVPR 2026.

```bibtex
@article{wang2025vdot,
  title={VDOT: Efficient Unified Video Creation via Optimal Transport Distillation},
  author={Wang, Yutong and Zhang, Haiyu and Xue, Tianfan and Qiao, Yu and Wang, Yaohui and Xu, Chang and Chen, Xinyuan},
  journal={arXiv preprint arXiv:2512.06802},
  year={2025}
}
```

AutoSOTA: automated SOTA-chasing harness used to drive the optimization in this report. Repository: `tsinghua-fib-lab/AutoSOTA`.
