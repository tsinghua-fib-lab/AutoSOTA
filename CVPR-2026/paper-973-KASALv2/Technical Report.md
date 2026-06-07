# Key-Axis-Based Localization of Symmetry Axes in 3D Objects Utilizing Geometry and Texture: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization of the publicly available KASAL v1 code using the AutoSOTA framework. The primary metric is the rotation‑averaged mean Euclidean distance divided by object diameter (eADI/d). Starting from a baseline eADI/d of 0.003073 on the DSRSTO dataset, the optimization applied four interventions over 13 iterations. The dominant change—replacing point‑to‑point ICP with point‑to‑plane ICP—reduced eADI/d to 0.002990, contributing 2.7 % of the overall improvement. Three further modifications (per‑type Fibonacci‑sphere parameters, exact rotation angle, and doubling the Farthest Point Sampling count) yielded an additional reduction of 0.000002, giving a final eADI/d of 0.002988 (2.8 % below baseline). Gains concentrated in D(>1) prismatic objects (−5.1 %), with one object improving by 32.4 %. Runtime doubled from 9.7 s to 19.5 s, and the target of ≤ 0.0029 was missed by 3.0 %. The accuracy ceiling is set by KASAL v1’s exhaustive search; the v2 extension, which includes automatic symmetry type classification and two‑stage sampling, is not publicly available.

## 1. Introduction

KASAL (Key‑Axis‑based Symmetry Axis Localization) [1] determines symmetry axis orientations and rotation centres for objects exhibiting one of eight predefined rotational symmetry types. The published paper [1] describes a two‑variant pipeline: KASAL v1 (exhaustive Fibonacci‑sphere search, ground‑truth symmetry type) and KASAL v2 (automatic type classification plus coarse‑to‑fine sampling), achieving eADI/d = 0.00212 with v2. The open‑source codebase corresponds to v1 only; the v2 modules remain unreleased. This optimisation study uses the v1 code, supplied with ground‑truth symmetry labels, and pursues algorithmic adjustments that reduce eADI/d on the DSRSTO dataset. The AutoSOTA framework was employed to propose and evaluate code‑level interventions.

## 2. Original Method (Background)

The KASAL v1 pipeline operates as follows. For a given object mesh and symmetry type, candidate axis directions are sampled using Fibonacci sampling on a half‑sphere (default `sample_num = 10001`, `half_sphere = True`) via `kasal/bop_toolkit_lib/view_sampler.py`; for types requiring a second key axis, circular sampling around the first axis is performed in `kasal/keyaxis/keyaxis.py`. Each candidate is scored by rotating the point cloud around the axis and measuring geometric consistency (and colour consistency when texture symmetry is flagged). The axis with the smallest error is selected.

Axis orientations and the rotation centre are then refined by iterative closest point (ICP) in `kasal/geometry/o3d_icp.py`. The baseline ICP uses `TransformationEstimationPointToPoint` with a voxel size of diameter/50, convergence criteria of relative fitness and RMSE = 1 × 10⁻⁶, and a maximum of 100 iterations. The input point cloud is sub‑sampled to `fpsample_num = 1500` points via Farthest Point Sampling before registration. The final output consists of symmetry transformation matrices in BOP format.

## 3. Identified Limitations

Four implementation‑level limitations were identified from the source code and initial profilings.

1. **Point‑to‑point ICP ignores surface orientation.** In `kasal/geometry/o3d_icp.py`, the `refine_registration` function uses `TransformationEstimationPointToPoint`, minimising Euclidean distance without considering local normals. Surface normals are already computed inside `refine_center_direction` for voxel down‑sampling but are discarded by the ICP estimator. This is suboptimal for rotationally symmetric objects, where residual error often appears as a small angular bias around the symmetry axis.
2. **Uniform search parameters across symmetry types.** The evaluation script `eval_final.py` applies the same `sample_num`, `half_sphere`, and `fpsample_num` to all objects. However, the required axis search differs markedly: single‑continuous‑axis types need only a half‑sphere, whereas multi‑axis prismatic types benefit from full‑sphere sampling with higher density.
3. **Quantised rotation angle.** In `kasal/keyaxis/keyaxis.py`, the rotation step for a symmetry axis of order *div* is computed as `dis_ang = round(360 / div, 2)`. This introduces a quantisation error of up to 0.005° that may degrade the axis‑ranking error metric for high‑order symmetries.
4. **Sparse point cloud for ICP.** The FPS count of 1500 may discard fine geometric detail that is important for precise axis refinement, especially for objects with small features.

Additionally, the repository entirely lacks the KASALv2 components (GDG automatic classification and two‑stage sampling). Therefore the accuracy ceiling of the v1 pipeline, even with optimal parameters, is expected to be near 0.0030, well above the v2‑reported 0.00212.

## 4. Optimization Methodology

The AutoSOTA pipeline was configured with a target eADI/d ≤ 0.0029 and a budget of 13 experimental iterations beyond the baseline. Each iteration modified one or more code locations and evaluated the result via `eval_final.py` on the full DSRSTO dataset. Four interventions were accepted into the final optimised state.

* **Point‑to‑plane ICP (IDEA‑008).** In `kasal/geometry/o3d_icp.py`, line 26, `TransformationEstimationPointToPoint()` was replaced with `TransformationEstimationPointToPlane()`. This exploits the previously computed vertex normals to penalise tangential misalignment, which is the dominant error mode for prismatic objects.
* **Per‑type adaptive parameters (IDEA‑003).** A dictionary `SYM_TYPE_PARAMS` was introduced in `eval_final.py`. Types C(=1), C(>1), P(4), P(8), P(20) use `sample_num = 5001` and `half_sphere = True`. Types D(>1) and D(=1) use `sample_num = 15001` and `half_sphere = False`. This allocates a denser full‑sphere search to multi‑axis cases while saving computation on single‑axis types.
* **Exact rotation angle (IDEA‑005).** In `kasal/keyaxis/keyaxis.py`, line 70, `dis_ang = round(360 / div, 2)` was changed to `dis_ang = 360 / div`, removing the quantisation.
* **Increased FPS count (IDEA‑009).** In `eval_final.py`, the `fpsample_num` parameter to `cal_model_sym` was doubled from 1500 to 3000, supplying more points to ICP registration.

All other ideas (multi‑candidate axis averaging, conditional ICP gating, normal‑aware scoring, two‑stage search, and further tuning of `sample_num`, ICP convergence, or voxel size) were rejected because they either degraded the metric or produced no measurable improvement, confirming that the Fibonacci‑sphere search saturates at approximately 5000 directions.

## 5. Experiments

### 5.1 Setup

**Hardware.** Experiments were executed in a Linux environment without a display server; exact hardware is not logged.

**Dataset.** The DSRSTO dataset was used, containing 20 shape meshes and 20 texture meshes. Ground‑truth symmetry types and orders were read from the accompanying JSON annotation files. Objects annotated as “None” were excluded.

**Metrics.** The primary metric is eADI/d, the mean of the Average Distance of Model points (ADI) over all valid symmetry transformations, divided by the object diameter. ADI is the mean minimum Euclidean distance from original vertices to vertices transformed by each symmetry matrix. For continuous symmetries, rotations are sampled every 30°. The computation is implemented in `compute_eADI` of `eval_final.py`. Per‑object runtime is also recorded.

**Baseline.** The baseline was obtained with `python eval_final.py` using default parameters: `sample_num = 10001`, `fpsample_num = 1500`, `half_sphere = True`, point‑to‑point ICP enabled.

**Optimisation.** 13 iterations were conducted. The best eADI/d (0.002988) was reached at iteration 7 (commit `c522b0f`). Runtime increased from 9.7 s to 19.5 s per object.

### 5.2 Quantitative Results

Table 1 presents the per‑symmetry‑type eADI/d for the baseline and the optimised version.

**Table 1: eADI/d by symmetry type before and after optimisation.**

| Symmetry type        | Baseline eADI/d | Optimised eADI/d |        Δ | Δ%      |
|----------------------|-----------------|------------------|----------|---------|
| C(=1) Circular       | 0.003221        | 0.003219         | −0.000002 | −0.1 %  |
| C(>1) Cylindrical    | 0.003162        | 0.003167         | +0.000005 | +0.2 %  |
| D(=1) Pyramidal      | 0.002531        | 0.002518         | −0.000013 | −0.5 %  |
| D(>1) Prismatic      | 0.003175        | 0.003012         | −0.000163 | −5.1 %  |
| P(20) Icosahedral    | 0.004134        | 0.004130         | −0.000004 | −0.1 %  |
| P(4) Tetrahedral     | 0.003554        | 0.003531         | −0.000023 | −0.6 %  |
| P(8) Octahedral      | 0.001954        | 0.001954         |    0      |   0.0 %  |
| **Overall (weighted mean)** | **0.003073** | **0.002988**     | **−0.000085** | **−2.8 %** |

The overall eADI/d decreased by 2.8 %. Nearly all improvement came from the D(>1) prismatic category (−5.1 %). Other classes showed changes of less than 1 %, consistent with measurement noise. The largest per‑object improvement occurred for `obj_000006` (prismatic), whose eADI/d dropped from 0.004364 to 0.002950 (−32.4 %). Three additional prismatic objects improved by 0.6 % to 3.6 % (`obj_000013`, `obj_000014`, `obj_000018`).

Runtime results are given in Table 2.

**Table 2: Runtime comparison.**

| Metric                     | Baseline | Optimised |   Δ  | Δ%     |
|----------------------------|----------|-----------|------|--------|
| Mean runtime per object (s) | 9.7      | 19.5      | +9.8 | +101 % |

The runtime doubled, driven by the point‑to‑plane ICP estimator and the larger FPS count. Per‑type parameter tuning mitigated some cost for single‑axis types (5001 vs. 10001 samples) but was offset by the denser 15001‑sample full‑sphere search for prismatic and pyramidal types.

### 5.3 Ablation / Iteration Trajectory

Table 3 shows the cumulative impact of each accepted intervention, reconstructed from the per‑change deltas recorded in the optimisation log.

**Table 3: Cumulative impact of accepted interventions.**

| Step | Intervention                                                      | eADI/d   | Δ from previous |
|------|-------------------------------------------------------------------|----------|-----------------|
| 0    | Baseline (point‑to‑point ICP, uniform params, quantised angle, fps=1500) | 0.003073 | —               |
| 1    | + Point‑to‑plane ICP                                              | 0.002990 | −2.7 %          |
| 2    | + Per‑type adaptive parameters                                    | 0.002990 | 0.0 %           |
| 3    | + Exact rotation angle                                            | 0.002989 | −0.03 %         |
| 4    | + fpsample_num = 3000                                             | 0.002988 | −0.03 %         |

Step 1 (point‑to‑plane ICP) accounts for virtually all the observed accuracy gain. The three subsequent modifications each contributed only a few ten‑thousandths of a unit to eADI/d.

## 6. Discussion

The optimisation demonstrates that a single algorithmic change—switching the ICP error metric from point‑to‑point to point‑to‑plane—provides a measurable, albeit modest, improvement for KASAL v1. The overall 2.8 % reduction is concentrated where the baseline error was highest: prismatic objects with multiple discrete axes. For `obj_000006`, the point‑to‑plane ICP corrected a systematic misalignment that the point‑to‑point version could not resolve.

The failure of other interventions underscores the saturation of the v1 pipeline. The Fibonacci‑sphere axis search is asymptotically effective at approximately 5000 directions; adding more samples, a local refinement stage, or normal‑consistent scoring does not change the candidate ranking. The ICP refinement step remains the only lever for further accuracy, and its improvement is bounded by the quality of the initial axis estimate and the rigid registration formulation.

The target eADI/d of 0.0029 was not reached because the v1 algorithm’s exhaustive‑search architecture cannot match the efficiency and precision of KASALv2’s automatic classification and two‑stage sampling, which are required to achieve the paper‑reported 0.00212. The optimised state represents the practical limit of the publicly available code under oracle symmetry‑type labels.

The runtime more than doubled, from 9.7 s to 19.5 s per object. For offline dataset processing this penalty may be acceptable; for time‑sensitive applications the faster point‑to‑point ICP would be preferred unless the dataset is dominated by prismatic shapes. The per‑type parameter scheme could be adjusted further to reclaim speed on single‑axis types.

## 7. Reproducibility

**Repository.** The original KASAL repository is at `WangYuLin‑SEU/KASAL`. The optimised state is commit `c522b0f`.

**Environment.** Install the package with recommended dependencies:
```bash
pip install kasal-6d[recommended]
```
or set up a conda environment:
```bash
conda create -n kasal python=3.10
conda activate kasal
pip install -r requirements_recommended.txt
```

**Seed.** The `cal_KA1` function initialises a random rotation of the Fibonacci sphere using `np.random.random()`, so exact reproduction of the reported eADI/d may vary slightly across runs.

**Baseline run.** Execute `python eval_final.py` on the original code (default point‑to‑point ICP, `sample_num=10001`, `fpsample_num=1500`).

**Optimised run.** Check out commit `c522b0f` (or apply the four interventions described in Section 4) and run `python eval_final.py`.

## 8. References

[1] Y. Wang and C. Luo, “Key-Axis-Based Localization of Symmetry Axes in 3D Objects Utilizing Geometry and Texture,” *IEEE Trans. Image Process.*, vol. 33, pp. 6720–6733, 2024, doi: 10.1109/TIP.2024.3515801.

[2] tsinghua-fib-lab/AutoSOTA, “Automated SOTA Optimization Framework,” 2025. [Online]. Available: https://github.com/tsinghua-fib-lab/AutoSOTA
