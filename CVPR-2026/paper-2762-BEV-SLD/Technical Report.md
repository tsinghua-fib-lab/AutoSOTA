# BEV-SLD: A Technical Report on Automated Optimization

## Abstract

This report documents an automated optimization pass over the BEV-SLD pipeline, a bird's-eye-view street-lane / landmark-based localization system whose paper-reported baseline attains a success rate (SR) of 98.31% with a median translation error of 0.28 m and a median rotation error of 0.42 deg. The optimization was conducted under the AutoSOTA framework, which proposes, implements, and evaluates candidate ideas under a fixed evaluation harness and records each attempt as a JSONL trajectory. After resolving an environment-level defect (a binary incompatibility between `numpy` 2.2.2 and `scikit-image` 0.20.0 that caused the localization module to fail at import time and the evaluator to silently reuse stale poses), a single effective change — running RANSAC five times from different random seeds and selecting the model with the largest inlier count (IDEA-017) — increased SR from 98.31% to 100.00% (+1.69 percentage points) on the held-out evaluation set, while leaving median translation error essentially unchanged (0.28 m → 0.29 m) and slightly reducing median rotation error (0.42 deg → 0.38 deg). Six other candidate modifications, including sub-pixel peak refinement, peak confidence filtering, two-stage RANSAC, temporal smoothing, RANSAC inlier-threshold tuning, and increased landmark density, produced no measurable improvement. The report summarises the inferred original method, the diagnosed limitations, the optimization trajectory, and reproducibility considerations. Because the working repository contains only the per-paper takeaway, the iteration log, and an optimization curve image — no README and no source code — method-level descriptions are kept deliberately at the level of behaviour observed through the evaluation harness rather than implementation specifics.

## 1. Introduction

BEV-SLD is, on the basis of the available artefacts, a pipeline for visual or LiDAR-based ego-pose estimation in a bird's-eye-view (BEV) representation, leveraging street-level landmarks (the abbreviation is interpreted in this report as "Street Lane / Landmark Detection" or, equivalently, "Street-Level Landmark Detection" in BEV space). The pipeline emits per-frame poses that are scored by an `eval_poses.py` harness against ground-truth `Poses.txt`, returning three primary metrics: success rate (SR), median translation error (median T, in metres), and median rotation error (median R, in degrees). The optimization target was the SR metric, with translation and rotation reported as secondary quality indicators.

This report describes an AutoSOTA-driven optimization pass on this pipeline. AutoSOTA generated and tested candidate modifications, each one committed to a separate revision, with results aggregated into `scores.jsonl`. The effective optimization completed in a single iteration once an environment defect was repaired.

## 2. Original Method (Background)

The available artefacts (TAKEAWAY_source.md and scores.jsonl) reference the following components of the original pipeline:

- A landmark / peak detector that produces per-frame heatmaps from which peak coordinates are extracted; default confidence thresholds and a default landmark density of approximately 0.2 are referenced.
- A correspondence stage that pairs detected peaks with map landmarks, feeding into a pose-estimation step.
- A RANSAC-based pose estimator with a default inlier threshold of 0.5 m, executed once per frame.
- An evaluation harness `eval_poses.py` that compares the pipeline's `Poses.txt` to ground truth, producing SR, median T, and median R.

The paper-reported baseline (iteration 0, commit `e56de8c1819bcbdca8891734df337548eccbe0f4`) was reproduced exactly: SR = 98.31%, median T = 0.28 m, median R = 0.42 deg. Because no README or source files are present in the working repository, deeper architectural descriptions (network backbone, BEV projection geometry, training procedure) cannot be grounded and are intentionally omitted.

## 3. Identified Limitations

Two limitations were diagnosed during optimization, one in the method and one in the environment.

1. **RANSAC sampling variance on a borderline frame.** SR was bottlenecked by a single failing frame whose pose, under a single RANSAC run, would occasionally land outside the success criterion. The failing frame had poor correspondences such that several method-level remedies (a tighter 0.3 m threshold, a coarse-to-fine two-stage RANSAC, increased landmark density, sub-pixel peak refinement, peak confidence filtering, and per-window temporal median filtering) did not recover it. The signature of the failure was therefore consistent with random consensus-set selection rather than systematically poor correspondences in the surrounding window.
2. **Environment-level silent failure.** A binary incompatibility between `numpy` 2.2.2 and `scikit-image` 0.20.0 (the well-known "numpy.dtype size changed" error) caused `localization.py` to crash at import time. Because `eval_poses.py` reads `Poses.txt` from disk, it silently re-scored the baseline poses regardless of any code change, masking all candidate modifications as no-ops. This meant that every early iteration appeared to produce baseline metrics and triggered seven "debug" iterations before the true cause was isolated.

## 4. Optimization Methodology

The optimization was driven by AutoSOTA's idea-generate / implement / evaluate loop. Each candidate change was tagged with an idea identifier, applied as a code modification, evaluated under the same harness, and recorded with status, primary metric, secondary metrics, notes, and commit hash. The trajectory was retained even for unsuccessful or environment-broken iterations.

Procedurally, the optimization proceeded in three phases:

1. **Diagnosis.** Seven debug iterations established that code changes were not affecting `Poses.txt`, leading to the `numpy` / `scikit-image` import-time crash. The fix was to pin `numpy<2.0` before installing `scikit-image` 0.20.0.
2. **Method-level interventions.** Several plausible improvements targeting the per-frame pose were proposed: tighter and two-stage RANSAC thresholds, sub-pixel peak refinement, peak confidence filtering, denser landmark sampling, and temporal smoothing. None of these moved SR off 98.31% on the failing frame.
3. **Stochasticity ensembling.** RANSAC was wrapped in a five-run ensemble (different random seeds, best model selected by inlier count). This produced the only effective change.

The single effective change therefore addressed the diagnosed bottleneck — variance of RANSAC's random sampling — directly, by trading negligible additional compute for a higher probability that at least one of five seeds finds the correct consensus set.

## 5. Experiments

### 5.1 Setup

- Baseline commit: `e56de8c1819bcbdca8891734df337548eccbe0f4`.
- Best commit: `313b8b92430fd9f716f420403d60bcbf8c97174d`.
- Primary metric: SR. Secondary: median T (m), median R (deg).
- Harness: `eval_poses.py` over the dataset's `Poses.txt`.
- Environment fix required: `numpy<2.0` paired with `scikit-image` 0.20.0.

### 5.2 Quantitative Results

The baseline-versus-best comparison reproduced from `TAKEAWAY_source.md` is given in Table 1. The deltas reported there are reproduced verbatim.

**Table 1. Baseline vs. best metrics on the BEV-SLD evaluation harness.**

| Metric     | Baseline | Best        | Delta     |
|------------|----------|-------------|-----------|
| SR         | 98.31%   | **100.00%** | +1.69pp   |
| Median T   | 0.28 m   | 0.29 m      | +0.01 m   |
| Median R   | 0.42 deg | 0.38 deg    | -0.04 deg |

The single applied change is summarised in Table 2.

**Table 2. Effective change.**

| Change | Effect | Notes |
|--------|--------|-------|
| IDEA-017: RANSAC ensemble (5 runs, best by inlier count) | SR: 98.31% → 100.00% | The sole effective change. Running RANSAC 5 times with different random seeds and selecting the best by inlier count increased the probability of finding the correct consensus set for the borderline frame. |

### 5.3 Ablation / Iteration Trajectory

The full per-iteration log from `scores.jsonl` is reproduced in Table 3. The "final" row is the post-restoration evaluation of the best commit. Note that the per-iteration median T and median R for IDEA-017 (0.27 m / 0.41 deg) differ slightly from the final row (0.29 m / 0.38 deg); the latter is the reproduced final-state evaluation reported in TAKEAWAY_source.md and is treated as authoritative for the headline numbers.

**Table 3. Iteration trajectory (`scores.jsonl`).**

| Iter | Idea ID  | Title                                            | Status  | SR (%) | Median T (m) | Median R (deg) |
|------|----------|--------------------------------------------------|---------|--------|--------------|----------------|
| 0    | baseline | Paper baseline                                   | success | 98.31  | 0.28         | 0.42           |
| 1    | IDEA-017 | RANSAC ensemble (5 runs, best by inlier count)   | success | 100.00 | 0.27         | 0.41           |
| final| final    | Final best state                                 | success | 100.00 | 0.29         | 0.38           |

A negative-results ablation, reconstructed from TAKEAWAY_source.md, is summarised in Table 4. None of these changes moved SR from 98.31%.

**Table 4. Candidate changes that did not improve SR.**

| Candidate change | Reported reason for no effect |
|------------------|-------------------------------|
| Sub-pixel peak refinement | The 0.5 m RANSAC threshold is too coarse for sub-pixel improvements to matter. |
| Peak confidence filtering | Default confidence thresholds were too conservative to filter any peaks. |
| Two-stage RANSAC (coarse → fine) | Failing frame's correspondences too poor for even the coarse stage. |
| Temporal smoothing (median filtering) | Failing frame's error too large to correct via median filtering. |
| RANSAC threshold tuning (0.3 m) | Tighter threshold did not help the failing frame. |
| Landmark density increase (0.2 → 0.4) | More landmarks did not help the failing frame. |

## 6. Discussion

The optimization isolates two distinct levers. First, the borderline-frame failure was a property of the consensus-search stochasticity rather than of the upstream feature quality, as evidenced by the simultaneous failure of every method-level intervention and the success of seed ensembling. Second, the seven debug iterations are a reminder that evaluation harnesses that consume artefact files (`Poses.txt`) rather than in-process objects can mask catastrophic upstream failures as null results; importing the localization module under the test harness, or hashing the pose file, would have caught the `numpy` / `scikit-image` incompatibility at iteration zero.

The improvement on secondary metrics (median R: 0.42 deg → 0.38 deg) is consistent with seed ensembling occasionally recovering a marginally better consensus set on non-failing frames; the slight increase in median T (0.28 m → 0.29 m) is small enough to be noise on the same evaluation set. The headline finding is therefore an SR gain of +1.69 percentage points to a saturated 100.00%, achieved at the cost of roughly 5x RANSAC inference compute.

Top remaining ideas for future runs, recorded by the optimizer for use beyond the SR-saturation regime, are: multi-scale test-time augmentation (IDEA-007), heatmap gradient-guided peak selection (IDEA-010), and multi-resolution pyramid peak detection (IDEA-016).

## 7. Reproducibility

To reproduce the headline result:

1. Check out commit `313b8b92430fd9f716f420403d60bcbf8c97174d`.
2. Install dependencies with `numpy<2.0` first, then `scikit-image==0.20.0`. Installing in the opposite order or allowing `numpy` 2.2.2 to be present at `scikit-image` import time reproduces the silent failure described in TAKEAWAY_source.md.
3. Run the full pipeline so that `localization.py` regenerates `Poses.txt`, then evaluate with `eval_poses.py`.
4. Expected metrics: SR = 100.00%, median T = 0.29 m, median R = 0.38 deg.

The baseline at commit `e56de8c1819bcbdca8891734df337548eccbe0f4` should reproduce SR = 98.31%, median T = 0.28 m, median R = 0.42 deg.

Caveats specific to this report: the working repository contains only `TAKEAWAY_source.md`, `scores.jsonl`, and `optimization_curve.png`. There is no README and no source. Implementation-level details (RANSAC API, peak-detection backbone, dataset partition, hardware) cannot be verified from the available files and are described in this report only at the level of behaviour reported in TAKEAWAY_source.md.

## 8. References

- AutoSOTA framework: `tsinghua-fib-lab/AutoSOTA`.
