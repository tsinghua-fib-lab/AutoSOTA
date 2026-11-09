# Paper 9003 — NESP (Novozymes Enzyme Stability Prediction) reproduction record

Base solution: Kaggle notebook `lauraromar/novozymes-7th` ("7th Place Solution"),
private LB 0.53258 / rank 7 of 2483, vendored headless at `/paper_data/base_repo`
(commit `90406b6`). This file records the reproduction performed inside container
`autosota_repro_paper_9003` on 2026-10-13.

## Result

| metric | value |
|---|---|
| **full-test Spearman (released 2413-row test set)** | **0.55785** |
| rubric acceptance window | [0.540, 0.570] |
| recorded baseline (this workspace, earlier session) | 0.55569 |
| author private-half score | 0.53258 |

Single-component full-test Spearman (official scorer, higher is better):

| component | this run | recorded baseline |
|---|---|---|
| LGBM (regenerable) | 0.45274 | 0.434 |
| WT-SASA (regenerable) | 0.43539 | 0.435 |
| B-factor diff (regenerable) | 0.29222 | 0.292 |
| ThermoNet-v2 (frozen input) | 0.47763 | 0.478 |
| Rosetta scores (frozen input) | 0.45640 | 0.456 |

The frozen/regenerable components reproduce the recorded values exactly; the LGBM
component lands 0.019 higher, which is a thread-count artefact (see below), not a
protocol change. Rank-average fusion adds ~+0.08 over the best single component
(0.4776 -> 0.5579).

Internal diagnostics (diagnostic only, not the workspace metric): 7-fold GroupKFold
on the Jin ddG set, `cv_rmse_overall=1.5056`, `cv_spearman_overall=0.5261`,
`cv_spearman_mean_by_wt=0.4516`, `n_train_jin=4035`, `components_missing_bfactor_pdbs=0`.

## Exact command

```bash
cd /repo && bash scripts/eval.sh
```

`scripts/eval.sh` runs `pipeline/run_pipeline.py --paper-data /paper_data --out-dir outputs`
and then the official scorer, printing `Spearman=<value>` on the last line. Wall clock
2 min 42 s. Two independent runs in `autosota_repro_paper_9003` produced bit-identical
`outputs/submission.csv` (md5 `49beab781763d8410844a40a8d902a97`) and bit-identical
component CSVs.

The same command was then re-run in a **fresh container started from the committed
image** `autosota/paper-9003:reproduced` with only
`-v .../9003/data:/paper_data:ro` mounted, and reproduced `Spearman=0.55785` in
2 min 45 s (exit 0). The manifest's `eval_command` + `data_mounts` are therefore
verified end-to-end from a clean image, not just in the working container.

## Environment

Base image `docker.1ms.run/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (Ubuntu 20.04,
Python 3.10.13, torch 2.1.0+cu121, 4x A100-80GB visible). Installed in-container:

* pip (Tsinghua mirror): `lightgbm 4.7.0`, `scikit-learn 1.7.2`, `biopython 1.88`,
  `biopandas 0.5.2` (pandas 2.3.3, numpy 1.26.0, scipy 1.15.3 were preinstalled).
* apt (Aliyun mirror): `dssp 3.0.0` -> `/usr/bin/mkdssp`, used by `Bio.PDB.DSSP`.
* apt: `git` (for the audit commit).

All inputs come from the read-only `/paper_data` mount; nothing was downloaded.
`test_labels.csv` is read only by `/paper_data/eval/score_test.py` — the pipeline
itself never touches it.

## Critical environment finding: LightGBM thread pinning

The first run of the pipeline crawled at ~0.9 s per boosting iteration (about 9 h
total). Root cause: **this LightGBM build does not honour `OMP_NUM_THREADS`.** When
`num_threads` is unset it falls back to `omp_get_max_threads()`, which is 128 here —
the container sits on a heavily contended `kubepods-burstable` slice, so 128-way
OpenMP thrash makes every iteration about 50x slower (0.33 s/iter vs 0.007 s/iter).
Setting `OMP_NUM_THREADS=4/8` had no effect (verified in `/proc/<pid>/environ`).

Fix (committed): `get_model()` now passes an explicit `num_threads`, default 8,
overridable with `NESP_LGBM_THREADS`; `scripts/eval.sh` additionally pins CPU
affinity with `taskset -c 0-7`. This is a runtime setting only — no model
hyperparameter, seed, split, or metric definition was changed. It does slightly
change LightGBM numeric output (hence LGBM 0.4527 vs 0.434 recorded), which is why
the thread count is now pinned explicitly and the whole run is reproducible.

False leads ruled out along the way (all measured, none were the cause): cgroup CPU
quota (none: `cfs_quota_us=-1`, `nr_throttled=0`), container CPU starvation (16
parallel busy loops all complete on time; the idle container is as fast as the host:
0.55 s vs 0.59 s for an 8M-iteration Python loop), and slow filesystem I/O on `/repo`
(overlayfs, 300 flushed writes in under 1 ms).

## Known levers for the SOTA phase

1. **Ensemble weights** are a free parameter. The published weights
   `(0.10*rank(SASA) + 0.30*rank(LGBM) + 0.20*rank(ThermoNet) + 0.10*rank(Bfactor)
   + 0.30*rank(Rosetta)) / 2413` are the author's, not fitted. The five component
   CSVs are written to `outputs/components/` on every run, so re-weighting can be
   tuned and scored without refitting.
2. **The full original 7th-place solution had a sixth component** the notebook drops:
   a second LightGBM with `boosting=dart` trained on a Demask-derived dataset. The
   notebook states it "did not end up contributing to a better score", so treat it as
   an unverified lead (the dataset itself is not vendored).
3. **The two frozen components** (`/paper_data/components/thermonet2_submission.csv`,
   `rosetta_scores.csv`) are fixed test-set artefacts and cannot be regenerated here;
   they may be read, never edited. Replacing or augmenting them is a legitimate
   direction.
4. **Regenerable parts**: the LGBM model (Jin v2 ddG + DSSP structural features) and
   both structure-derived components (WT-SASA, mutant-vs-WT B-factor) from
   `kvigly_test_mutation_pdbs/`.
5. `colsample_bytree=1`, `n_estimators=5000`, `max_depth=4`, `learning_rate=0.005`,
   `random_state=42`, 7-fold GroupKFold by wild-type — these are the paper settings.
6. `test.csv` sequence overlap with `train.csv` is exactly zero; the test set is a
   single enzyme (~70 C range), so transfer is the modelling difficulty.

## Red lines (from the task card)

`test_labels.csv` is evaluation-only (never train on it, never use it as a feature,
never edit recorded scores); do not modify `eval/`, `test.csv`,
`sample_submission.csv`, `test_labels.csv`, `components/`, or the vendored PDB
inputs; the submission must cover all 2413 test rows.
