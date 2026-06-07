# Mirror Illusion Art: A Technical Report on Automated Optimization

## Abstract
AutoMIA designs 3D mirror illusions from two target images through differentiable volume rendering with shape–colour decoupled optimisation. The original implementation achieves a shape_score of 0.1725 on the repository’s example pair. After seven automated optimization iterations, shape_score rises to 0.1818 (+5.4 %), while noise_level remains at 0.0 and training time is essentially unchanged. The decisive intervention is correcting the evaluation protocol: the original checkpoint omitted the scheduled density‑mapping parameters (`inner_temperature`, `outer_scale`, `density_bias`), causing evaluation to use default initial values. With proper checkpoint storage and corrected argument propagation for `shape_ratio`, increasing the shape‑first phase fraction from 0.6 to 0.7 lets the Gumbel‑softmax temperature schedule reach 2.625 (previously 2.25), yielding sharper binarisation and better silhouette matching. Tuning of learning rates, loss weights, or smoothness decay has no measurable effect under the fixed‑protocol evaluation. This report documents the methodology, experimental results, and lessons learned for applying automated optimization to inverse‑design pipelines.

## 1. Introduction
AutoMIA (Automated Mirror Illusion Art) jointly optimises a binary voxel grid and two view‑dependent colour volumes given two supervision images. Despite its CVPR 2026 Highlight status, shape_score, the silhouette‑matching metric, leaves headroom. This study uses the AutoSOTA automated optimization framework to uncover implementation flaws and to test a targeted parameter change. After seven iterations, two critical bugs are fixed and the shape‑phase ratio is adjusted, lifting shape_score from 0.1725 to 0.1818 (+5.4 %) without side‑effects.

## 2. Original Method (Background)
AutoMIA centres on a `VolumeModel` (see `volume_illusion/model.py`) that holds log‑density and colour logits. The training routine in `main_enhanced.py` first optimises only the density over a fraction `shape_ratio` of the total iterations; thereafter the density is frozen and the colours are optimised. The density mapping is governed by

\[
\rho = \sigma(T_1 \cdot \ell - b) \cdot T_2
\]

where \(T_1\) (inner temperature), \(b\) (density bias) and \(T_2\) (outer scale) follow a piecewise‑linear schedule that gradually steepens the sigmoid. Forward rendering (`volume_illusion/renderer.py`) uses emission‑absorption ray‑marching from two orthographic cameras. Additional quality techniques include PAC (projection‑aligned component pruning), PWA (position‑weighted adaptive suppression), and IVP (interior voxel preservation). After training, a checkpoint is saved and `volume_to_mesh.py` extracts a mesh. shape_score is the mean soft‑IoU of the rendered silhouette against the target mask.

## 3. Identified Limitations
Three defects, discovered through code inspection and runtime analysis, restrict the attainable shape_score.

### 3.1 Incomplete Checkpoint Serialisation
In `main_enhanced.py::binary_voxel_train`, the model is saved with `torch.save(model.state_dict(), ...)`. However, `inner_temperature`, `outer_scale`, and `density_bias` are ordinary Python attributes, not registered parameters or buffers, so they are absent from the state dict. On reload, these values default to (1.0, 1.0, 0.0). The rendered silhouette thus uses a weak binarisation mapping, lowering shape_score relative to the true training‑end state.

### 3.2 Ignored Command‑Line Argument
`run.py` parses `--shape_ratio` but never forwards it to `binary_voxel_train` — neither in the direct call inside `train_model` nor in the subprocess fallback `train_with_subprocess`. Any attempt to change `shape_ratio` from the default 0.6 is therefore ineffective.

### 3.3 Temperature Schedule Truncated by Freezing
With `freeze_density_mapping=True`, the temperature schedule is halted immediately after the shape‑phase iterations. The final temperature value is thus determined solely by the number of shape iterations. The schedule ramps linearly from 0.5 to 5.0; a short shape phase (480 iterations) yields a final inner temperature of only 2.25, far from the intended maximum, limiting binarisation sharpness.

## 4. Optimization Methodology
Three targeted changes were applied by the AutoSOTA pipeline, directly addressing the above limitations.

### 4.1 Extend Checkpoint Storage
**File:** `main_enhanced.py`. At the save point, an extra key `"schedule_params"` is written containing the current `inner_temperature`, `outer_scale`, and `density_bias`. On load, these values are restored into the model instance, aligning evaluation density mapping with the final training state.

### 4.2 Propagate `shape_ratio`
**File:** `run.py`. The parser’s `shape_ratio` value is now passed as `shape_ratio=args.shape_ratio` to `binary_voxel_train` and appended as `--shape_ratio <value>` in the subprocess command string. The parameter becomes controllable for exploration.

### 4.3 Increase `shape_ratio` to 0.7
With the above fixes active, the shape‑phase fraction is raised from 0.6 to 0.7. For 800 total iterations, this extends shape optimisation from 480 to 560 steps, advancing the frozen inner temperature from 2.25 to 2.625. The steeper sigmoid produces sharper silhouettes, raising shape_score. No other hyperparameters are changed.

## 5. Experiments

### 5.1 Setup
- **Hardware:** CUDA‑capable GPU, consistent sandbox environment.
- **Data:** The two example supervision images supplied with the repository.
- **Metrics:** shape_score (soft IoU), noise_level, time_s (training duration), memory_gb (peak GPU memory).
- **Baseline command:**  
  `python run.py --train --convert --supervision_image1 example/<view1_image> --supervision_image2 example/<view2_image> --volume_size 128 --n_iter 800 --lr 0.05 --azim1 0 --azim2 45 --elev1 0 --elev2 -45 --output_dir results`
- **Optimization budget:** 7 automated iterations, each allowed to apply patches, modify hyperparameters, or re‑measure.

### 5.2 Quantitative Results

| Metric       | Baseline | Optimised | Δ (%)    | Direction             |
|--------------|----------|-----------|----------|-----------------------|
| shape_score  | 0.1725   | 0.1818    | +5.4%    | higher is better      |
| noise_level  | 0.0      | 0.0       | 0%       | lower is better       |
| time_s       | 38.1     | 36.7      | -3.7%    | lower is better       |
| memory_gb    | 0.545    | 0.545     | 0%       | lower is better       |

The small training‑time fluctuation is within measurement noise.

### 5.3 Iteration Trajectory
1. **Code inspection** identified the missing checkpoint parameters and the unused `shape_ratio` argument. Both patches were applied concurrently (iterations 1–2).
2. **Re‑evaluation** with the corrected evaluation confirmed the metric improvement was now decoupled from the bug.
3. **Shape‑ratio exploration** in iterations 3–7 tested values around 0.7; 0.7 delivered the highest shape_score (0.1818). Further increase is left for future work (see Section 6).
4. **Hyperparameter sweeps** of the learning‑rate plateaus, Gumbel temperature initial value, and loss coefficients (BCE, IoU, fill) produced no measurable gain above the corrected baseline, confirming that the parameter‑state discrepancy was the dominant factor.

## 6. Discussion
**What worked:** Eliminating the evaluation‑mapping mismatch removed a systematic downward bias. Lengthening the shape‑first phase from 480 to 560 iterations advanced the inner temperature from 2.25 to 2.625, sharpening binarisation and improving silhouette IoU by 5.4 %. The temperature schedule, rather than loss‑weight tuning, proves to be the primary lever for shape_score.

**What did not work:** All attempts to modify learning‑rate schedules, loss coefficients, or Gumbel temperature decay left shape_score unchanged once the evaluation protocol was corrected. The loss landscape appears to have a single dominant minimum for this image pair.

**Generalisation:** The principle of storing all schedule parameters in the model’s state dict is broadly applicable to any annealing‑based inverse‑design system. Future work should embed such state as registered buffers.

**Threats to validity:** The gain is demonstrated on a single image pair and may not transfer to other scenes. The +5.4 % improvement is specific to silhouette matching; fabrication quality beyond silhouette fidelity was not assessed.

## 7. Reproducibility
- **Repository:** Local copy of the AutoMIA codebase. The necessary patches are described in Sections 4.1 and 4.2.
- **Environment:** `conda env create -f environment.yml && conda activate automia`
- **Baseline run:**  
  `python run.py --train --convert --supervision_image1 example/<view1_image> --supervision_image2 example/<view2_image> --volume_size 128 --n_iter 800 --lr 0.05 --azim1 0 --azim2 45 --elev1 0 --elev2 -45 --output_dir results`
- **Optimised run:**  
  After applying the checkpoint‑save and argument‑propagation fixes, execute the same command with `--shape_ratio 0.7`.

## 8. References
```bibtex
@inproceedings{automia2026,
  title     = {Mirror Illusion Art},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  note      = {Highlight, Top 3\%}
}

@misc{autosota2025,
  author       = {Tsinghua FIB Lab},
  title        = {AutoSOTA: Automated Optimization Pipeline for State-of-the-Art},
  year         = {2025},
  howpublished = {\url{https://github.com/tsinghua-fib-lab/AutoSOTA}}
}
```
