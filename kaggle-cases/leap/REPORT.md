# LEAP / ClimSim — AutoSOTA research report

> Source-only GitHub archive prepared on November 9, 2026. This English text is retained from the supplied report. Historical paths and artifact-retention statements describe the original experiment environment. Data, checkpoints, fitted parameter files, predictions, logs, and raw evidence tables are external to this archive. No new experiment or Kaggle submission was run during integration.

**Research and experiment lead: autosota.** autosota led idea screening, method design, code adaptation, experiment orchestration, result analysis, and final selection. Published baselines retain their original attribution. This report was reorganized on November 9, 2026 using existing experimental records; delivery validation rebuilds cached predictions and does not establish a new competition score.

### autosota: Kaggle LEAP / ClimSim Optimization Case Study

Prepared for delivery on November 9, 2026, using the case report compiled on October 23, 2026. The experimental results and leaderboard evidence were primarily archived on October 14, 2026; the shutdown handoff was prepared on October 15. This report documents existing results. No new training runs or Kaggle submissions were launched while preparing it. All times below use China Standard Time, **UTC+08:00 (Asia/Shanghai)**; the timestamps in the original Kaggle submission records are UTC.

> Artifact-retention update, October 23, 2026: at the user's request, old candidate weights, optimizer recovery states, most historical prediction/submission files, unused expanded caches, and downloaded raw training shards were removed. The best release's 17 weight files, best and original-winner submissions, required reconstruction inputs, LR2M/HR1M training caches, and 200k holdout were retained. Experimental and shutdown-state descriptions below are historical; deleted files cannot be resumed using the old recovery commands. Deletion records and post-cleanup checks are in cleanup-2026-10-23 (external reference: `cleanup-2026-10-23/`).

**Best verified result: Public Score 0.79431 and Private Score 0.79165**, obtained by late submission **57180922**. These exceed the reproduced historical first-place solution by **0.00021** and **0.00042**, respectively. The successful approach builds on the winner's published Squeezeformer ensemble and combines constrained conditional ensembling, small-learning-rate MSE fine-tuning of three members, state-dependent residual calibration, and validated output processing. The official final standings are frozen, so this result does not establish a new official first place. The highest historical raw Public Leaderboard score, **0.80526**, was not surpassed.


**Delivery scope:** this ZIP contains code, hyperparameters, recipe summaries, training-source snapshots, and experimental evidence only. Checkpoints, learned calibration matrices, serialized normalization assets, datasets, NPY predictions, and Parquet outputs are not embedded. Recipe JSON files under snapshots are summaries with learned arrays removed; full parameters are referenced as external assets in [assets.json](assets.json). See [README.md](README.md) and [BEST_SOLUTION.md](README.md#method).

### 1. Task Definition and Problem Description

#### 1.1 Official task and learning objective

The task in [LEAP - Atmospheric Physics using AI (ClimSim)](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim) is to build machine learning models that **emulate subgrid atmospheric processes** in the E3SM-MMF climate model. Each data row represents inputs and outputs of a **cloud-resolving model (CRM)** at a particular location and timestep. The objective is to approximate the effects of these processes at substantially lower computational cost. autosota formulate the task as multi-output regression with an atmospheric vertical structure. Sources: [official Description](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview/description) and [official Data Description](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data).

The model receives the atmospheric column state and boundary conditions. It predicts physical tendencies for temperature, moisture, and horizontal winds, together with radiation and precipitation quantities. The physical timestep is **1,200 seconds**. The competition evaluates offline predictive accuracy; this case study does not include online coupling to a climate model or a long-term stability evaluation.

| Component | Dimensions / count | Description |
|---|---:|---|
| Vertically resolved inputs | 9 × 60 = 540 | Temperature `state_t`; specific humidity, cloud liquid, and cloud ice `state_q0001/2/3`; zonal and meridional winds `state_u/v`; ozone, CH4, and N2O |
| Scalar inputs | 16 | Surface pressure, solar insolation, heat fluxes, surface stresses, albedos, surface fractions, and related quantities |
| Total raw input features | **556** | Excluding `sample_id` |
| Vertically resolved targets | 6 × 60 = 360 | Heating and moisture tendencies, plus zonal and meridional wind accelerations |
| Scalar targets | 8 | Radiation and precipitation quantities |
| Total target columns | **368** | The full prediction vector for one sample |
| Test rows | **625,000** | Predictions must preserve the required sample IDs and ordering |
| Submission structure | 369 columns | `sample_id` plus 368 target columns; this project submitted Parquet files |

In the official terminology, **25 input variables expand into 556 columns**, and **14 target variables expand into 368 columns**. Vertically resolved variables carry level suffixes from 0 to 59; smaller indices refer to higher atmospheric levels. The raw CSV interleaves some variable types, so the implementation selects features by name rather than assuming that its first 540 columns are all vertical features. The winner's preprocessing produces a 1,696-dimensional internal representation; this does not change the 556-dimensional raw task definition.

Archived official page content is available in research/raw (external reference: `snapshots/research/raw/`). The full local investigation is in competition-research.md (external reference: `snapshots/research/competition-research.md`). The following terminology is used consistently; the optimization techniques described later are autosota's project methods, rather than algorithms specified by Kaggle.

| Term | Meaning in this report |
|---|---|
| Subgrid atmospheric processes | The physical processes that the competition asks ML models to emulate |
| Cloud-resolving model (CRM) | The fine-scale model underlying each input–output sample |
| Tendency | A rate of change of a physical state variable |
| Public Leaderboard / Private Leaderboard | Kaggle evaluation and rankings on the public / private test subsets |
| Late Submission | A submission scored after the competition deadline; it does not revise the official final standings |
| Local validation holdout | Training-era data reserved locally for validation; it is distinct from the Kaggle private test set |

#### 1.2 Evaluation metric

Kaggle describes the evaluation as a **custom R-squared metric on a weighted solution** and links to a default R² implementation based on scikit-learn. The weighting step is element-wise multiplication using `sample_submission.csv`. It should not be confused with a variance-weighted average of target-level R² scores. autosota's local evaluation uses the **uniform average across target columns**, also called macro-averaged R². Higher is better.

For nonconstant targets, the score is:

```text
R²_j = 1 - Σ_i (y_ij - ŷ_ij)² / Σ_i (y_ij - mean(y_j))²
Score = (1 / 368) × Σ_j R²_j
```

Here, `y` and `ŷ` are the reference and prediction values after the required weighting. Apply the submission weights exactly once. autosota's local implementation uses `force_finite=True` for constant-target handling. In the current weighting file, weights are either 0 or 1: the top 12 levels, indices 0–11, of the three moisture tendencies and two wind accelerations account for **60 zero-weight columns**. Local evaluation retains all **368** columns. Averaging only the nonzero-weight targets, flattening all entries into one R² calculation, or using a variance-weighted average would produce a different metric.

Targets with very small variance are particularly sensitive to numerical error. Preprocessing, deterministic target handling, and serialization precision therefore matter for a faithful reproduction. Sources: archived official Evaluation (external reference: `snapshots/research/raw/official-page-Evaluation.md`), local metric investigation (external reference: `snapshots/research/competition-research.md`), [official Evaluation](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview/evaluation), and [scikit-learn R² documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

#### 1.3 Development scope and evaluation limits

Training and calibration used permitted training-era data. autosota did not use test labels released after the competition, and autosota did not reconstruct test timestamps, locations, or grid positions from test ordering. Later research configurations requiring additional inputs, such as v6, were not applied directly to the competition's 556-feature test data.

Late submissions provided both Public and Private Scores. Some of this feedback informed subsequent choices, so the final result is a verified score after iterative development, **not an untouched blind-test estimate**. Historical official standings and late-submission scores are reported separately. Sources: [official Rules](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/rules) and [the organizers' discussion of leakage and competition restrictions](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/519772).

### 2. Investigation and Experimental Path

#### 2.1 Identify strong baselines and establish what could be reproduced

The investigation covered leading competitors' original code and solution write-ups, subsequent organizer research implementations, and primary research on losses, climate-invariant representations, parameter averaging, and residual calibration. The source inventory and access checks are recorded in competition-research.md (external reference: `snapshots/research/competition-research.md`) and optimization-roadmap.md (external reference: `snapshots/research/optimization-roadmap.md`).

| Source | Main approach / available material | Work completed and decision |
|---|---|---|
| First place: greySnow | Squeezeformer; original notebooks, preprocessing, and published weights | Downloaded 26 accessible notebooks and 14 checkpoints; reproduced full ensemble inference and matched the historical Public and Private Scores; selected as the main baseline |
| Second place: Z Lab | Grouped LSTMs; SmoothL1, difference losses, and group-specific fine-tuning | Downloaded the original repository and seven grouped checkpoints for `max_exp22`; evaluated on a common holdout; did not reproduce the team's entire final ensemble |
| Third place: Capaomat | Solution write-up; a later organizer implementation of the Pao architecture | Archived the method description; did not present the research implementation as the original third-place final model |
| Fifth place: Yusef and teammates | Encoder–Decoder LSTM / GRU | Downloaded code and completed forward/backward smoke checks; some required weight links were unavailable, so no final reproduced competition score was obtained |
| Seventh place: Ristametall | Public Ryota contribution; feature engineering and MAE→MSE training | Archived original code and used it to inform candidate experiments; this does not represent the full team's implementation |
| Organizer research | Controlled architecture comparisons, physical representations, and constraints | Used to formulate experiments; research metrics and input configurations were not treated as interchangeable with the competition setup |

Sources: [winner's repository](https://github.com/shlomoron/LEAP-solution), [second-place repository](https://github.com/ChunhanLi/2nd-kaggle-LEAP), [third-place write-up](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/writeups/capaomat-w-o-leak-3rd-place-solution), [fifth-place repository](https://github.com/YusefAN/leap-climsim-kaggle-5th), [seventh-place repository](https://github.com/nocchi1/kaggle-leap-7th-place-solution), and [organizer research repository](https://github.com/leap-stc/climsim-kaggle-edition).

The winner's repository was pinned to commit `6dc8113cd8e30f8924f002ac723f2b71625b0de7`; the second-place repository was pinned to `df6df337af5405b01c2820e6f0f8b2d67ed217f5`. Other revisions are listed in repository-manifest.json (external reference: `snapshots/research/repository-manifest.json`). Access was limited to public material and downloadable attachments. Private notebooks and unpublished weights were not accessible; seven linked winner notebooks also returned HTTP 403.

#### 2.2 Reproduce the winner and test an alternative model for complementarity

autosota executed the original layer definitions from archived notebooks, retained the original normalization and postprocessing, and adapted the environment, paths, and data I/O. The original solution has **13 ensemble members**, with one LR member averaging two checkpoints, so **14 weight files** were loaded. Full inference on 625,000 test rows produced submission **57165628**, scoring **0.79410 / 0.79123**. This matches the historical winner at Kaggle's displayed precision; it does not establish bitwise identity of predictions across hardware.

This was a **published-checkpoint inference reproduction**, followed by fine-tuning. autosota did not retrain the complete original ensemble from scratch. On **14,721 common held-out samples**, the reproduced seven-group Z Lab model scored R² **0.78923159**, versus **0.80133755** for the original winner. Its selected blend weight was zero, so it was excluded. This comparison concerns the particular reproduced sub-solution and sample set, not Z Lab's entire final ensemble.

#### 2.3 An early failure exposed a validation-distribution mismatch

The first attempt optimized nonnegative ensemble weights separately by target group. It improved the second 100k of the training-era holdout by approximately **0.00213 R²**, but its actual Private Score fell from **0.79123** to **0.79019**. The result showed that strong gains on this training-era holdout did not reliably transfer to the competition test distribution. In particular, moving too much weight toward LR models was unsuccessful.

autosota then reduced the number of free choices: preserve original within-group weights while testing the total LR/HR mixture, and subsequently bound within-group and conditional adjustments. LR total weights of **0.25, 0.45, and 0.18** yielded Private Scores of **0.79130, 0.79110, and 0.79132**, respectively. Subsequent development fixed the aggregate **LR/HR weights at 0.18/0.82**. This choice used late-submission Private Score feedback.

#### 2.4 Explore training, calibration, and physical consistency in parallel

The roadmap recorded **52 candidate ideas**. The handoff inventory contains **38 training, weight-averaging, or pending configurations**, plus **5 technical smoke checks**, **25 frozen calibration/ensemble recipes**, and **34 actual Kaggle submission records**. These categories overlap in purpose and must not be added together as independent successful experiments. Pending configurations are not completed training runs.

The main directions were:

1. **Loss and data:** MSE, Huber, MSE/MAE mixtures, prediction-head fine-tuning, LR-only versus LR/HR training, HR-target soft clipping, and calendar-balanced sampling.
2. **Architecture and representation:** fine-tuning 256/384/512-dimensional parent models; RH and buoyancy input channels; a separate learning rate for added inputs; EMA; input noise; vertical-difference losses; and larger batches.
3. **Ensembling and residuals:** bounded group weights, humidity-conditional covariance weighting, confidence-based blending, linear and RBF residual calibration, two-dimensional state gating, and output-mean recalibration.
4. **Physical and numerical checks:** moisture bounds, shortwave consistency, precipitation inequalities, soft column-wind correction, and FP32/FP64 serialization audits.

The following negative or inconclusive results materially affected the final selection.

| Attempt | Measured observation | Decision |
|---|---|---|
| More flexible groupwise weights | Large local gain, but Private Score decreased to 0.79019 | Use fixed resolution budgets and bounded weights |
| Inverse-confidence blending | No improvement after calibration; average member residual correlations were about 0.961–0.983 | Excluded from the final recipe |
| Huber pilot | Improved single-model R² on the first 50k, but reduced the full-ensemble second-half diagnostic by about 0.00002263 | Not submitted |
| RH, buoyancy, EMA, noise, and difference-loss variants | Some improved a single model or an older anchor, but did not beat the selected final ensemble | Retained as experimental evidence, not claimed as successful final components |
| Higher-learning-rate batch1024 model | Single-model 100k R² reached 0.79799656; a standalone replacement scored 0.79143 privately; full-pipeline selection excluded it | Single-model ranking was not used as a substitute for ensemble evaluation |
| RH×temperature conditional ensemble | Repeated local diagnostic improved by about 0.00002236, but Private Score was 0.79164 | Rejected relative to 0.79165 |
| Output-mean recalibration | Tied the best displayed Public and Private Scores | Retained the earlier, simpler submission |
| FP32 for every output column | Underflow on tiny moisture targets reduced diagnostic R² by about 0.00875 | Keep 16 deterministic liquid-water targets in FP64 |
| Forcing all nighttime shortwave outputs to zero; hard closure of full water/momentum budgets | Training-label checks did not support these simplifications; some required budget terms were missing | Did not impose these hard constraints |

#### 2.5 Validation protocol and runtime monitoring

autosota used the original author's **200k LR holdout** and excluded its source shards from fine-tuning. Long runs selected checkpoints on the first **100k**; early pilots used the first **50k**. Calibration generally used the first **75k** for fitting, the next **25k** for recipe selection, and the last **100k** for diagnostic checks. The last split was reused during development, so it is a repeated diagnostic holdout rather than a fresh blind test. Gains from 50k and 100k selection sets are not directly comparable.

The queue recorded GPU state, processes, training heartbeats, and disk availability about every **30 seconds**. Runs saved source snapshots, configurations, learning curves, best weights, and recovery checkpoints. Training, inference, ensemble selection, and submission scoring had separate status records. Candidates were compared through the complete processing pipeline before submission decisions. After the shutdown request, new launches stopped and three active runs saved state and exited; the originally planned overnight schedule was not completed.

### 3. Methods Retained in the Best Verified Solution

The final prediction pipeline is:

```text
556 raw competition input features
  → original winner preprocessing and Squeezeformer member predictions
  → replace three designated parents with MSE-fine-tuned checkpoints
  → fixed LR/HR budgets + bounded group weights + RH-conditioned interpolation
  → frozen linear state-dependent residual calibration
  → frozen RBF residual calibration
  → soft column-wind correction, solar/precipitation consistency, and moisture bounds
  → Parquet submission with FP64 retained for 16 deterministic targets
```

#### 3.1 Constrained state-conditional ensembling

Aggregate LR/HR weights were fixed at **0.18/0.82**. Within seven target groups—temperature, vapor, liquid, ice, zonal wind, meridional wind, and scalar outputs—weights were adjusted within **±25%** of the corresponding prior, while remaining nonnegative and preserving the resolution budgets. This limited the drift observed in the first, more flexible weighting experiment.

autosota then used **mean RH over levels 48–59**, derived from available inputs, to define four humidity bins. Member error second moments were estimated within each bin, weights were optimized subject to bounds, and the resulting weights were interpolated smoothly with humidity. Conditional weights were further bounded within **±15%** of the constrained global weights. The selected covariance recipe was `cov-reg0-alpha1`. Bounded group weighting and RH conditioning raised the actual Private Score to **0.79138** and then **0.79140**.

#### 3.2 Small-learning-rate MSE fine-tuning of three members

Because R² penalizes squared error, autosota tested small-learning-rate MSE fine-tuning of the published checkpoints to improve conditional-mean prediction. Only the following three parent replacements were selected for the best solution.

| Parent member | Selected replacement | Training changes | Selected best step |
|---|---|---|---:|
| `hr_5e_best` | `hr_mse_calendar_balanced` | LR 2M + HR 1M, approximately 2:1 within batches; calendar-balanced sampling; HR-target soft clipping; MSE | 10,000 |
| `lr_6e` | `lr6_mse_1e6` | LR 2M; small-learning-rate MSE | 6,000 |
| `lr_512` | `lr512_mse_1e6` | LR 2M; small-learning-rate MSE for the 512-dimensional member | 5,000 |

All three replacement fractions were **1**. Each child inherited its parent's ensemble budget; the same parent was not counted multiple times with extra weight. Other members retained their published weights. The final combination selected these three from **13 available fine-tuned candidates**. The artifact name `thirteen-completed` refers to that selection stage, not to 13 newly trained members in the final ensemble.

Fine-tuning called the network with `training=False` to preserve existing normalization statistics while still differentiating with respect to trainable parameters. The loss masked official zero-weight targets, deterministic liquid-water levels, and targets flagged by the original implementation. The selected three checkpoints did not use added RH/buoyancy input channels or EMA weights. RH remained useful in conditional ensembling and residual features.

#### 3.3 Low-capacity residual calibration for state-dependent bias

The linear calibration stage used **16 state features**, including lower/upper RH, temperature structure, cloud water, surface fluxes, solar input, and predicted precipitation. The selected Ridge penalty was **1**, with application strength **1**. Additional per-target slopes were bounded by **±0.02**, and state-dependent corrections were capped at **±0.05 target standard deviations**. On the then-current ensemble, this increased the Private Score from **0.79140** to **0.79154**, one of the main observed improvements.

The nonlinear stage expanded the feature set to **25**, adding wind shear and albedos. It used **64 cluster-centered radial basis functions (RBFs)** with Ridge regularization and the same **0.05-standard-deviation** correction cap. The selected recipe was `rbf1-ridge1-alpha1`, with an RBF bandwidth of approximately **2.215826**. It achieved a Private Score of **0.79155** on the baseline available at that stage.

After selecting the three fine-tuned members, autosota retained the already frozen linear, RBF, and physical-processing recipes. Refitting calibration after changing neural-network members did not improve the comparison and was excluded. This preserved validated corrections while limiting additional fitting freedom.

#### 3.4 Validated output processing and numerical precision

Following the original winner's rule, deterministic liquid-water targets `ptend_q0002_12` through `ptend_q0002_27` were set to `-state_q0002 / 1200`. Moisture predictions were bounded to keep the next-step state nonnegative. Radiation and precipitation processing used relationships checked against training labels, including:

```text
NETSW = SOLS × (1 - ASDIR) + SOLL × (1 - ALDIR)
      + SOLSD × (1 - ASDIF) + SOLLD × (1 - ALDIF)
0 ≤ PRECSC ≤ PRECC
```

The wind correction softly shrank pressure-thickness-weighted column integrals toward training-era calibration statistics. This was a statistical correction, not exact momentum conservation. The standalone R² gain from shortwave/precipitation consistency was very small and does not explain the overall score improvement.

The final Parquet retained **FP64 for the 16 deterministic liquid-water targets** and used FP32 for the other targets after a precision check. For the best recipe, the mixed-precision diagnostic score differed from the FP64 result by about **5.66×10⁻¹¹**. Converting every target to FP32 was invalid because tiny values underflowed. This change preserved numerical correctness while reducing file size; recovering precision was not counted as an additional modeling gain.

The frozen recipes are the three-member replacement configuration (external reference: `snapshots/artifacts/combined/thirteen-completed/frozen_config.json`) and the complete output-processing configuration (external reference: `snapshots/artifacts/combined/thirteen-completed-rbf-momentum/frozen_config.json`).

### 4. Performance Improvements and Comparisons with Leading Teams

#### 4.1 Comparison with the official final Private Leaderboard top three

In this table, **top three means the historical official final Private Leaderboard top three**. Their historical Public Scores are shown alongside their Private Scores. Values come from archived official leaderboard records rather than estimates from write-ups. All gains are absolute score differences at the displayed five-decimal precision.

| Official final private rank / status | Team or solution | Public Score | Private Score | autosota's final public gain | autosota's final private gain |
|---|---|---:|---:|---:|---:|
| 1 | greySnow (no leaky) | 0.79410 | 0.79123 | +0.00021 | +0.00042 |
| 2 | Z Lab | 0.79212 | 0.78856 | +0.00219 | +0.00309 |
| 3 | Capaomat (w/o leak) | 0.79048 | 0.78792 | +0.00383 | +0.00373 |
| Project late submission | **Best solution, 57180922** | **0.79431** | **0.79165** | — | — |

Public and private records were joined by team IDs **11977183, 12045622, and 12050778**. Sources: archived full Public Leaderboard CSV (external reference: `snapshots/research/leaderboards/leap-atmospheric-physics-ai-climsim-publicleaderboard-2026-10-14T03:27:10.csv`), archived official Private Leaderboard records (external reference: `snapshots/research/leaderboard-current.csv`), and all project submissions (external reference: `all-submissions.csv`). Official entry point: [Leaderboard](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/leaderboard). These compare historical standings with a late submission; they do not assign a new official rank to this project.

#### 4.2 Additional comparison with the historical raw Public Leaderboard top three

The raw Public Leaderboard top three differ from the official final private top three. autosota therefore report them separately. A score embedded in a team's `w/o leak` name annotation is distinct from its raw leaderboard `Score` field; the table uses the latter.

| Historical raw public rank / status | Team | Raw Public Score | Historical Private Score | Public difference: ours minus team |
|---|---|---:|---:|---:|
| 1 | KTK; team-name annotation `w/o leak: lb 0.79193` | 0.80526 | 0.78776 | **−0.01095** |
| 2 | Rist SkyBlue; team-name annotation `w/o leak: 0.78619` | 0.80055 | 0.78258 | **−0.00624** |
| 3 | greySnow (no leaky) | 0.79410 | 0.79123 | +0.00021 |
| Project late submission | Best solution | 0.79431 | 0.79165 | — |

Sources are the same as in the preceding table. The first two public results should be interpreted in the context of the competition's leakage discussions; this project did not use the associated test-reconstruction techniques. The supported result is that autosota **exceeded the historical official winner's Public and Private Scores**. autosota did not exceed the highest raw Public Score.

#### 4.3 Actual submission progression

| Stage | Submission ID | Public Score | Private Score | Private gain over original winner |
|---|---|---:|---:|---:|
| Full published-checkpoint inference reproduction | 57165628 | 0.79410 | 0.79123 | 0.00000 |
| More flexible groupwise weights: failed control | 57165622 | 0.79302 | 0.79019 | −0.00104 |
| LR/HR=0.18/0.82 + moisture bounds | 57165836 | 0.79403 | 0.79132 | +0.00009 |
| Bounded within-group weights | 57167006 | 0.79408 | 0.79138 | +0.00015 |
| RH-conditional covariance ensemble | 57168799 | 0.79412 | 0.79140 | +0.00017 |
| Linear state-dependent residual calibration | 57170940 | 0.79419 | 0.79154 | +0.00031 |
| RBF residual calibration | 57171879 | 0.79420 | 0.79155 | +0.00032 |
| Early pilot combination + RBF / soft correction | 57172540 | 0.79423 | 0.79158 | +0.00035 |
| Fine-tuned combination selected at the four-candidate stage | 57180416 | 0.79428 | 0.79163 | +0.00040 |
| **Final three replacements with frozen processing** | **57180922** | **0.79431** | **0.79165** | **+0.00042** |
| Later two-dimensional state ensemble: not retained | 57183764 | 0.79430 | 0.79164 | +0.00041 |
| Later output recalibration: tie, not retained | 57184093 | 0.79431 | 0.79165 | +0.00042 |

All **34** submission records are in all-submissions.csv (external reference: `all-submissions.csv`). The best Private Score improved by **0.00042** over the original winner and by **0.00033** over the **0.79132** anchor used at the start of the extended optimization phase. This is the progression of selected complete candidates, not a fully controlled component-by-component ablation. Differences between rows cannot be interpreted as independent causal contributions.

The best recipe scored **0.79782663** on the second 100k local diagnostic split. That training-era diagnostic is distinct from its Kaggle Private Score of **0.79165**. autosota did not perform an independent repeated-training significance study, and tied five-decimal scores do not establish a further improvement.

### 5. Experimental Setup: GPUs, Time, Tokens, and Base Models

#### 5.1 Hardware, environments, and model initialization

| Item | Actual setup |
|---|---|
| GPUs | **4 × NVIDIA A100-SXM4 80GB**, approximately 320GB total GPU memory |
| Parallel execution | Multiple single-GPU training/inference jobs were queued concurrently; individual models were not trained with four-GPU distributed training |
| CPU / host memory | 128 CPUs; approximately 1 TiB RAM |
| Main training environment | `.venvs/greysnow-tf`; Python 3.11, TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4 |
| Tool environments | `.venvs/kaggle` for Kaggle access, data, and optimization; `.venvs/leap-tools` with PyTorch 2.1.2 for checks of other competitors' implementations |
| Base models | Published winner Squeezeformer ensemble combining vertical local convolutions and self-attention; 13 original members represented by 14 checkpoints |
| Model widths | Original members used dim 256 / 384 / 512 and head_dim 1024 / 2048; see `champion_models.json` |
| Parameter counts of selected fine-tuned members | **52,732,772** each for HR5e and LR6; **83,387,492** for LR512 |
| Initialization and reproduction scope | Published competition checkpoints followed by inference reproduction and fine-tuning; no full-ensemble training from scratch |
| Precision | FP64 for original normalization and inverse transforms; FP32 for the main network; FP64 for 16 deterministic submission targets and FP32 for the remainder |

Dependency inventories: [greysnow-tf-current.txt](environments/greysnow-tf-current.txt), [kaggle-current.txt](environments/kaggle-current.txt), and [leap-tools-current.txt](environments/leap-tools-current.txt). Original model specifications: champion_models.json (external reference: `snapshots/configs/champion_models.json`). The archived runtime inventory (external reference: `snapshots/configs/pytorch-runtime.json`) also records the GPUs.

#### 5.2 Training data and validation splits

| Dataset / split | Rows | Actual use |
|---|---:|---|
| LR fine-tuning cache | 2,000,000 | LR-only training and the LR component of mixed training |
| HR fine-tuning cache | 1,000,000 | HR component of mixed LR/HR training |
| Early pilot training | 100,000 | Short runs for initial direction screening |
| Fixed LR holdout | 200,000 | First 100k for long-run checkpoint selection; second 100k for repeated diagnostics |
| Calibration fitting / selection | 75,000 / 25,000 | Within the first 100k holdout; excluded from gradient training |
| Expanded LR / HR caches | 6,000,000 / 3,000,000 | Prepared, deduplicated, and checked; corresponding expanded-data training had not started before shutdown |

The original LR holdout came from `combined-1/0/tfds/0–39`; LR training started at shard **40**. HR training started at shard **20**. The expanded 9M cache covered permitted **years 1–8 and January of year 9**. The LR 6M cache had zero overlapping time/grid keys with the 200k holdout. **The reported best result used the actual LR 2M / HR 1M fine-tuning setup; it was not a completed 9M training result.** See the detailed experiment archive (external reference: `ARCHIVE_EXPERIMENT_LOG.md`) and merged-cache checks (external reference: `snapshots/artifacts/data-preparation/merged-cache-sample-check.json`).

#### 5.3 Selected fine-tuning runs: hyperparameters and measured time

All three selected runs used **AdamW**, weight decay **1e-6**, global gradient clipping at **1**, cosine learning-rate decay to **20%** of the initial rate, **batch size 256**, **seed 43**, and evaluation every **1,000 steps**. Best checkpoints were selected on the first 100k holdout.

| Selected run | Initial learning rate | Completed steps | Selected step | Single-model R²: original → best | Measured run time |
|---|---:|---:|---:|---|---|
| `hr_mse_calendar_balanced` | 2e-6 | 12,000 | 10,000 | 0.79658566 → **0.79787102** | 7,394.94 seconds; about **2h 03m 15s** |
| `lr6_mse_1e6` | 1e-6 | 10,000 | 6,000 | 0.79957428 → **0.80005100** | 6,734.25 seconds; about **1h 52m 14s** |
| `lr512_mse_1e6` | 1e-6 | 10,000 | 5,000 | 0.79879538 → **0.79938979** | 9,147.53 seconds; about **2h 32m 28s** |

Times are the `elapsed_seconds` values in each run's `completed.json`. They include training and in-run validation through the end of the run, rather than only the time to reach the selected checkpoint. Each run occupied one GPU, for a combined **6.47 single-GPU task-hours**. These jobs ran concurrently, so their summed duration is not wall-clock elapsed time. Configurations and histories: HR calendar (external reference: `snapshots/artifacts/finetune/hr_mse_calendar_balanced/`), LR6 (external reference: `snapshots/artifacts/finetune/lr6_mse_1e6/`), and LR512 (external reference: `snapshots/artifacts/finetune/lr512_mse_1e6/`).

#### 5.4 Overall timeline, completion status, and token accounting

| Item | Recorded result | Accounting scope |
|---|---|---|
| First actual reference submission | October 14, 2026, **11:16:59** | Published single-model reference predictions used to check the scoring path |
| Full winner reproduction submission | October 14, 2026, **11:37:08** | Submission timestamp for 57165628 |
| Best submission | October 14, 2026, **19:43:47** | Submission timestamp for 57180922, not the time scoring finished |
| Full reproduction submission → best submission | **8h 06m 39s** | Difference between submission timestamps; excludes earlier research, downloads, and environment setup |
| First submission → shutdown request | **11h 09m 57s** | Through October 14, 2026, 22:26:56; not the full project duration |
| Completed training runs | **22**, including 4 pilots | Handoff-normalized completion status; early stopping was allowed |
| Summed time of completed training tasks | **33.68 hours** | Sum of `elapsed_seconds` across those 22 single-GPU tasks; an approximation of occupancy for this subset |
| Other configuration statuses | 2 completed weight averages, 3 interrupted-and-saved runs, 11 unstarted configurations | Together with the 22 completed runs, these total 38 configurations |
| Summed time of interrupted / weight-averaging tasks | Approximately **3.10 / 0.047 hours** | Reported separately from completed training |
| Total project GPU cost | **No complete, precise ledger available** | These task timers exclude some original-checkpoint inference, separate evaluation, failed retries, data work, and idle allocations; they are not total project GPU-hours |
| Available autosota-development token counter | **1,472,990 tokens** | Historical optimization-goal `tokensUsed`; partial coverage, not the full project total |
| Scope of that token record | Old goal created October 14, 2026, 11:50:36; last updated 15:54:25 | Does not provide complete coverage of later development or report preparation; no input/output/cache breakdown or cost was recorded |
| ClimSim model-training tokens | **Not applicable** | These numerical regression models are accounted for using samples, steps, batches, and GPU time; the token counter concerns the autosota development workflow |

The historical goal metadata was retrieved while preparing this report and saved in case-report-token-usage.json (external reference: `case-report-token-usage.json`). Its `timeUsedSeconds=14628` is a goal-runtime counter, not an experiment training-duration measurement. Complete project model-call costs and input/output token totals were unavailable, so they were not estimated and presented as measurements.

The three shutdown-interrupted runs saved at **2254/10000**, **1127/10000**, and **4156/6000** steps. They had weights and recovery records, but were neither counted as completed as planned nor included in the best solution. Full configuration status is in all-training-experiments.csv (external reference: `all-training-experiments.csv`); recovery information is in resume-state.json (external reference: `resume-state.json`).

### 6. Deliverables and Reproduction Evidence

| File / directory | Purpose |
|---|---|
| Current best-result index (external reference: `runtime_templates/configs/sota-current.json`) | Best submission ID, actual scores, and artifact paths |
| [Best submission file](assets.json) | The file that scored 0.79431 / 0.79165 |
| Frozen release manifest (external reference: `snapshots/release/submission-57180922/release-manifest.json`) | Hashes of 17 original or fine-tuned weight files, eight recursively referenced recipe records, and source versions; this is not a claim of 17 final ensemble members |
| Reconstruction check (external reference: `snapshots/release/submission-57180922/cached-reconstruction-check.json`) | IDs, values, schema, and SHA256 comparison after reconstruction from prediction caches |
| All submissions (external reference: `all-submissions.csv`) | 34 actual late-submission records |
| All training configurations (external reference: `all-training-experiments.csv`) | 38 completed, interrupted, or unstarted configurations |
| Calibration and ensemble index (external reference: `all-calibration-and-ensemble-experiments.csv`) | 25 frozen recipes |
| Detailed experiment archive (external reference: `ARCHIVE_EXPERIMENT_LOG.md`) | More detailed experiment records, shutdown state, and recovery procedures |
| [Reproduction instructions](README.md) | Environments, original-checkpoint inference, and stepwise reconstruction |

SHA256 of the best scored submission file:

```text
7b6f90312cd62f07ed81668ef715846db66c7961ef6022421ca56db4b051b63a
```

The submission was actually rebuilt from checkpoint prediction caches and frozen recipes. **All 625,000 row IDs and all 368 target columns matched exactly, and the reconstructed file had the same SHA256.** This check reused existing predictions; it did not rerun full test inference from every checkpoint. With the dependencies retained, the following command rebuilds the submission from caches and verifies the result. It does not automatically submit to Kaggle.

```bash
cd autosota_LEAP_best_57180922_2026-11-09
python run_best.py --assets-source /path/to/retained-project --workspace /path/to/new-workspace
```

The `handoff` directory contains reports, source snapshots, and small evidence files. **Large training datasets, NPY prediction caches, model weights, and submission Parquet files were not duplicated into it.** A migration must also preserve the required large files listed in [assets.json](assets.json) and the release manifest. The Markdown report or original handoff archive alone is insufficient to independently reconstruct the result. This bilingual report and the token evidence are separate additions in `handoff`; the original handoff archive was not rewritten.

---
