# G2Net Gravitational Wave Detection — AutoSOTA research report

> Source-only GitHub archive prepared on November 9, 2026. This English text is retained from the supplied report. Historical paths and artifact-retention statements describe the original experiment environment. Data, checkpoints, fitted parameter files, predictions, logs, and raw evidence tables are external to this archive. No new experiment or Kaggle submission was run during integration.

**G2Net Gravitational Wave Detection: From Public Solution Reproduction to Exceeding the Historical Winner’s Private Leaderboard Score**

Report prepared on November 9, 2026. Experiments and submissions took place on October 10–14, 2026. All times below are China Standard Time (UTC+8). This report is based on preserved source code, experiment logs, leaderboard snapshots, and Kaggle submission receipts.

**Final result: Public leaderboard ROC-AUC of 0.88581 and Private leaderboard ROC-AUC of 0.88540.** The private score exceeds historical winner KDL’s 0.88538 by 0.00002; the public score remains below KDL’s. This is a **Late Submission** score comparison on the competition’s fixed test set. It does not change the official historical standings or constitute a competition win. The final submission ID is `57159613`, with status `COMPLETE`. The latest preserved verification of its official scores was at 10:03 on October 14, 2026.

### 1. Task definition and problem description

The [G2Net Gravitational Wave Detection](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection) competition concerns detecting gravitational waves from binary black hole mergers. Its inputs are simulated measurements for three interferometers: a sample contains noise alone or noise with an injected simulated signal. The positive class, `target=1`, indicates signal presence. This terminology follows the [official Dataset Description](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection/data).

| Item | Definition |
| --- | --- |
| Input | Three time-series channels: LIGO Hanford (H1), LIGO Livingston (L1), and Virgo |
| Sample dimensions | 2 seconds at 2,048 Hz per channel; array shape `(3, 4096)` |
| Labels | `1`: signal present; `0`: noise only |
| Training set | 560,000 labeled samples |
| Test set | 226,000 samples |
| Submission format | A CSV with columns `id,target`; `target` is the predicted probability of signal presence |
| Official metric | Area Under the Receiver Operating Characteristic Curve (ROC-AUC), computed from predicted probabilities and observed targets; higher is better |
| Project objective | Reproduce accessible strong baselines, then improve them enough to exceed the historical winner’s private leaderboard score |

Kaggle specifies probability predictions and ROC-AUC scoring in its [Evaluation and Submission File sections](https://www.kaggle.com/c/g2net-gravitational-wave-detection/). Public and private leaderboard scores refer to different test subsets. autosota therefore focused on ranking signal-containing samples above noise-only samples, rather than optimizing accuracy at a fixed threshold. Weak-signal coverage, matching noise distributions, and complementary model errors were central considerations.

In autosota’s preprocessing pipeline, amplitudes were multiplied by `1e20` in float64 before conversion to float32, preserving precision for very small values. Whitening spectra were then estimated from training-fold noise. Sample IDs were checked throughout training, validation, and prediction. Each model excluded its own validation fold from training; newly estimated physical-parameter priors also excluded that fold.

### 2. Exploration path: reproduction, scaling, diagnosis, and the final improvement

#### 2.1 What was actually available and reproduced

autosota did not have access to every competitor’s private code, and autosota did not reproduce the winner’s complete final system. autosota selected publicly available implementations that could be adapted and run.

| Source | Available materials and reproduction scope | Role in autosota |
| --- | --- | --- |
| KDL, historical 1st place | Public [physical signal generation code](https://github.com/denisbsu/g2net-generation); the complete winning network and final weights were not obtained | Informed physical simulation and pretraining; autosota also implemented a separate model with multiple scales, detector branches, and auxiliary physical tasks |
| EventHorizon, historical 2nd place | Public [training code](https://github.com/analokmaus/kaggle-g2net-public); selected Seq03aug3 / Nspec22aug1 configurations were run | Candidate baselines; the original team’s complete final ensemble was not reconstructed |
| GOGOGO, historical 3rd place | Public [1D model code](https://github.com/ZiyueWang25/Kaggle_G2Net), primarily V2 and Width32 stochastic-depth networks | Main implementation basis for optimization; the original ensemble of 15 one-dimensional and 8 two-dimensional models was not fully reproduced |
| PGDSA 2026 | Paper and public repository reviewed | A complete reproducible implementation and weights were not confirmed; reported paper scores were not counted as results reproduced by autosota |

Source revisions are recorded in source_manifest.json (external reference: `snapshot/reports/source_manifest.json`). The main third-place implementation was pinned to commit `8ff757fb5252406b49d61f85c6b6b1f5c125cd8b`; KDL’s generation code was pinned to `b197a9989472dc1dd359c8ec4bca8e9d8322cec8`.

#### 2.2 Experimental stages

| Stage | Hypotheses and experiments | Findings and subsequent decisions |
| --- | --- | --- |
| Environment adaptation and pilot runs | Checked forward/backward passes, optimizer behavior, checkpoints, and prediction; compared V2, Width32, EventHorizon, and synthetic pretraining | Synthetic pretraining did not consistently outperform training from scratch in the pilot; full-data experiments were needed to assess its potential |
| Initial full-data baselines | V2 from scratch, synthetic pretraining followed by fine-tuning, two-initialization blending, wider networks, and 1D ensembles | The first private score was 0.88066; an improved 1D ensemble reached 0.88273. A basic five-fold, two-initialization ensemble scored only 0.88186, showing that adding models alone was insufficient |
| Five-fold expansion and the 12-hour campaign | Expanded Width32 and physical pretraining across folds; explored longer pretraining, SNR sampling, and multiscale models with auxiliary physical tasks | Five-fold expansion reached 0.88281 private; the 12-hour campaign reached 0.88343. Some candidates completed only pretraining and were not fully evaluated |
| 10-hour campaign | Million-record signal banks, a two-million-record noise bank, wider models, longer pretraining, and SNR 6–16 sampling | Private AUC rose to 0.88493. Width32 became the main component, complemented by narrower V2 and multiscale branches |
| 14-hour campaign | Replicated the strongest Width32 recipes across five folds; tested natural-amplitude/SNR mixtures, continued training, weight averaging, pseudo-labels, and more MC samples | A stable five-fold replacement reached 0.88525 private. Another candidate failed its predefined validation gate. Anomalously weak folds prompted an audit of whether training had actually updated parameters |
| Final campaign | Fixed GeM numerical instability; recovered ineffective fine-tuning; retried natural-distribution and weak-signal training, noise mixing, complementary models, and hierarchical ensembling | Two frozen recipes achieved 0.88529 and 0.88540 private, with the latter exceeding the historical winner’s private score |

Several directions did not provide consistent gains. SNR 4–10 sampling was weaker than SNR 6–16 in these experiments. Increasing MC dropout from 64 to 256/512 samples did not reliably help. SAM, frozen BatchNorm, amplitude perturbations, ranking losses, and extra random seeds did not yield consistent large improvements. The 11 additional low-learning-rate seed pipelines completed, but received no weight in the final direct ensemble layer.

Some earlier conclusions that longer training was ineffective had to be withdrawn: subsequent inspection showed that certain runs had not updated their trainable parameters at all. Successful process exits, completed epochs, and finite losses were insufficient evidence of effective training.

### 3. What ultimately worked

The final approach combined **the third-place 1D architecture, physical pretraining informed by KDL’s public approach, a numerical stability fix, fine-tuning on official competition training data, and locally selected ensembles across folds and model families**. A complete factorial ablation was not performed, so the final private-score improvement cannot be assigned precisely to individual changes. The following distinguishes direct diagnostic evidence from combined modeling results.

#### 3.1 The critical fix: silent zero-update training caused by GeM

The network used generalized mean pooling (GeM) with a learnable exponent `p`:

```text
GeM(x, p) = mean(clamp(x, eps)^p)^(1/p)
```

With `eps=1e-6` and an exponent around 7.4093, powers of small activations could underflow in FP32. The subsequent fractional-power derivative could then become Inf/NaN, even while the forward output and loss remained finite. In the original training loop, GradScaler skipped optimizer updates on non-finite gradients, but the learning-rate scheduler continued advancing. The scaler was also recreated each epoch. A run could therefore appear to train while leaving its trainable weights unchanged.

An audit of seven earlier fine-tuning runs found **five with unchanged trainable parameters and empty optimizer states**. A separate 40-epoch physical continuation run also had an empty optimizer state. These were targeted findings, not evidence that all earlier experiments were invalid.

autosota replaced the direct power computation with an equivalent log-space formulation:

```text
GeM(x, p) = exp((logsumexp(p * log(clamp(x, eps))) - log(k)) / p)
```

Here, `k` is the number of pooled elements. Convolutions used BF16, while GeM, master parameters, and validation remained in FP32. BF16 training no longer used loss scaling. Every backward pass checked gradient finiteness, failed immediately on invalid gradients, and recorded actual optimizer updates. Training and inference used the same stable GeM implementation.

On an affected checkpoint, the old implementation produced one non-finite gradient tensor; the new implementation produced none. A subsequent optimizer update changed 143 parameter tensors. The maximum forward discrepancy from an FP64 reference was approximately `5.1e-7`. After recovery training, the problematic fold-2 SNR model’s MC64 AUC increased from **0.87806689 to 0.88446026**. All 97 final-campaign training runs recorded actual updates, totaling **1,344,000 optimizer steps**.

Evidence: GeM numerical audit (external reference: `snapshot/reports/campaign_final_gem_audit.json`), earlier-run update audit (external reference: `snapshot/reports/campaign_final_prior_update_audit.json`), and implementation (external reference: `snapshot/scripts/team3_stable_training.py`).

#### 3.2 Better-matched training data and training recipes

Large-scale physical pretraining expanded coverage of waveform shapes and noise combinations. Fine-tuning on the official training data then addressed the remaining distribution mismatch. The final campaign reused five physical signal banks with one million records each and a separate bank of two million noise records. Signal banks shared prefixes and generation structure: **five million stored signal records should not be interpreted as five million independent signals newly generated during the final campaign**.

| Effective direction | Configuration | Contribution and evidence |
| --- | --- | --- |
| Recovering ineffective training | Fine-tuned each fold’s last natural-distribution/SNR checkpoint for 3 epochs at learning rate `3e-4` | Learning resumed after the numerical fix; recovered families entered the final ensemble |
| Continued natural-amplitude training | Width32; up to 32 additional physical-pretraining epochs, followed by fine-tuning on official training data | Preserved coverage of the target amplitude distribution; five-fold family evaluation AUC was 0.88615583 |
| Mixed weak-signal training | Width32; a 50/50 mixture of natural amplitudes and optimal network SNR 6–16; 15% noise from official negative samples in the current training fold; 24 continuation epochs followed by fine-tuning | Covered both natural and harder signals; five-fold family evaluation AUC was 0.88617013 |
| Complementary V2 branch | Narrower 1D network; SNR 6–16 with 15% official negative-sample noise; five folds | Lower standalone AUC than the strongest Width32 family, but complementary errors made it useful in the ensemble |
| Weight averaging | Averaged recent checkpoints within a fold and refreshed BatchNorm statistics using 16,384 samples from that fold’s training partition | Selected averaged branches contributed complementary predictions; validation waveforms were not used to refresh BatchNorm |

The family AUC values above use the local evaluation halves totaling 280,000 samples, not the Kaggle private test set. SNR 6–16 denotes linear optimal network SNR, not decibels.

autosota also compared restoration of optimizer moments. For an old checkpoint with an empty optimizer state, ancestor moments were restored only after confirming exact equality of all trainable parameters with that ancestor; current BatchNorm buffers were retained and a new learning-rate schedule was used. This was not an exact interrupted-run resume with all random-number-generator states restored.

#### 3.3 Calibration across folds and complementary ensembling

The final prediction pipeline had two stages. First, the fixed combined allocation of **57.767%** assigned to two Width32 branches in the previous best solution was replaced with a stronger complete five-fold ensemble. Other components were retained. The weights within this replacement ensemble were:

| Input to the five-fold replacement ensemble | Weight within this layer |
| --- | ---: |
| Width32 with continued natural-amplitude/SNR mixture training | 40.059% |
| Width32 with continued natural-amplitude training | 18.340% |
| Recovered SNR branch | 12.736% |
| V2 with continued SNR training | 10.476% |
| Recovered natural-amplitude branch | 9.143% |
| Averaged SAM 0.03 checkpoints | 5.000% |
| Original Width32 reference | 4.245% |

Second, the promoted solution was combined with complementary fold-0 models and five-fold predictions. The pipeline included per-fold Platt calibration, probability averaging, logit blending, and replacement relative to a reference. Consequently, the table is not a list of direct weights on all final checkpoints, and the nested weights cannot simply be multiplied to produce an equivalent flat linear ensemble. The final dependency graph references **67 distinct model checkpoint files**. See winning_dependency_graph.json (external reference: `winning_dependency_graph.json`) and the final metrics file (external reference: `best_submission/metrics.json`) for the exact recipe.

Relative to the original Width32 reference, the complete five-fold replacement component gained **0.00051585 AUC** on the 280,000 local evaluation samples, with a paired bootstrap 95% interval of **[0.00040744, 0.00064272]**; all five folds improved. The second-stage enhancement gained **0.00012071** over the first-stage solution on the fold-0 evaluation half, with an interval of **[0.00001040, 0.00025672]**.

Both final submission recipes were frozen before either submission’s Kaggle score was read. autosota’s interpretation is that the difficulty of exceeding the historical winner is consistent with the combined value of physical priors, training-distribution coverage, and strong ensembles. autosota did not isolate those factors in the winner’s private implementation. In autosota’s experiments, changing a backbone or adding folds was insufficient; fixing a real training failure and accumulating complementary gains ultimately closed a small remaining score gap.

### 4. Performance and improvements

#### 4.1 Comparison with the historical top three teams

The main table selects the **top three teams by final historical private leaderboard rank**, then joins their public scores by team ID. Public and private third place belong to different teams: GOGOGO was third on the private leaderboard and fourth on the public leaderboard.

| Historical private rank | Team / solution | Public AUC | Private AUC | autosota’s public − theirs | autosota’s private − theirs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | KDL | 0.88645 | 0.88538 | -0.00064 | **+0.00002** |
| 2 | [Aillis.jp] EventHorizon | 0.88450 | 0.88299 | +0.00131 | +0.00241 |
| 3 | GOGOGO [JarvisLabs.ai] | 0.88405 | 0.88291 | +0.00176 | +0.00249 |
| Late Submission | **autosota’s final ensemble** | **0.88581** | **0.88540** | — | — |

For completeness, public third-place team **OGOGOG [JarvisLabs.ai]**, which was fourth on the private leaderboard, scored **0.88435 public / 0.88272 private**. autosota’s corresponding margins were **+0.00146 / +0.00268**. The comparison therefore covers the top three teams on both leaderboards.

Evidence: historical private leaderboard snapshot (external reference: `snapshot/reports/private_leaderboard.json`), historical public leaderboard snapshot (external reference: `snapshot/reports/leaderboard/g2net-gravitational-wave-detection-publicleaderboard-2026-10-10T09:24:16.csv`), and final official submission receipt (external reference: `snapshot/reports/submissions/20261014T000107Z.json`). These scores come from preserved records; they are not presented as leaderboard scores freshly queried on October 23, 2026. The official task terminology was checked separately when preparing this bilingual report.

#### 4.2 All ten scored Kaggle submissions

| Submission time (2026, UTC+8) | Method / stage | Public AUC | Private AUC | Submission ID |
| --- | --- | ---: | ---: | --- |
| Oct 10, 18:57 | First full-data V2 baseline: synthetic pretraining and fine-tuning | 0.88169 | 0.88066 | 57040173 |
| Oct 10, 19:07 | Blend of two V2 initializations | 0.88250 | 0.88137 | 57040385 |
| Oct 10, 21:02 | Improved 1D ensemble | 0.88347 | 0.88273 | 57042921 |
| Oct 10, 22:38 | Five-fold V2 ensemble with two initializations | 0.88284 | 0.88186 | 57044965 |
| Oct 11, 14:43 | Five-fold wider networks and physical pretraining | 0.88351 | 0.88281 | 57065263 |
| Oct 11, 20:59 | Final ensemble from the 12-hour campaign | 0.88412 | 0.88343 | 57074653 |
| Oct 12, 20:13 | Final ensemble from the 10-hour campaign | 0.88539 | 0.88493 | 57106739 |
| Oct 13, 10:30 | Stable five-fold replacement from the 14-hour campaign | 0.88563 | 0.88525 | 57127720 |
| Oct 14, 08:00 | Final campaign: five-fold component replacement | 0.88567 | 0.88529 | 57159521 |
| Oct 14, 08:01 | **Final campaign: complementary ensemble** | **0.88581** | **0.88540** | **57159613** |

Relative to the first scored submission, public AUC improved by **0.00412** and private AUC by **0.00474**. Relative to the best result before the final campaign, the improvements were **0.00018 public** and **0.00015 private**. These are absolute AUC differences. A machine-readable record is available in submissions.csv (external reference: `submissions.csv`).

#### 4.3 Scope and limitations of the result

The final local AUC over all fold-0 out-of-fold (OOF) predictions was **0.88651872**; AUC on its evaluation half was **0.88754779**. Neither is the Kaggle private leaderboard score. The fixed validation partitions and their evaluation halves were reused adaptively throughout exploration, so they are not untouched test sets. The original fitting scope of some upstream public power spectral density (PSD) artifacts and mass estimators could not be fully traced.

The actual fold-0 blend improvement after first-stage replacement had an interval crossing zero. It passed a predefined gate requiring improvement of the five-fold component, no substantial regression on individual folds, and a nonnegative fold-0 point estimate for the final blend. It would be incorrect to claim that every validation interval excluded zero. The **0.00002** private margin is an observed difference in displayed scores on a fixed test set, not proof of statistically significant superiority over the historical winner on the broader gravitational-wave population.

### 5. Experimental setup and resource usage

#### 5.1 Hardware, software, and base models

| Item | Recorded configuration / verifiable scope |
| --- | --- |
| GPUs | **4 × NVIDIA A100-SXM4-80GB**, with 80 GB of memory per GPU |
| Execution | Multiple training and prediction jobs scheduled concurrently on assigned GPUs; the campaign was not one four-GPU distributed training run |
| Main base architectures | The third-place public 1D V2 / V2StochasticDepth implementations, especially Width32; whitened time-series inputs, stochastic depth, and learnable GeM pooling |
| Other models | Public EventHorizon configurations; autosota’s own multiscale and auxiliary-physics branches informed by KDL’s public description |
| Pretraining | Physical-waveform pretraining within the project and the upstream implementation framework; the signal classifier itself was not a language model |
| Training environment | Python 3.8, PyTorch 2.4.1+cu121, NumPy 1.23.5, SciPy 1.10.1, pandas 1.5.3, scikit-learn 1.3.2, timm 0.4.12 |
| Tooling and physics environments | Separate Python 3.11 environments; physical generation used PyCBC 2.11 and LALSuite |
| Final-stage precision | BF16 convolutions; FP32 GeM, master parameters, and validation; finite-gradient checks and actual optimizer-step accounting |
| Validation partitions | Main five-fold split with seed 48, 112,000 validation samples per fold; each fold stratified 50/50 into calibration and evaluation halves using seed 713. The original EventHorizon fold seed was 2021 |
| Calibration and blending | 56,000 calibration and 56,000 evaluation samples for one fold; 280,000 each across five folds. Calibration parameters and blending weights were fitted on calibration halves |
| Inference augmentation | Branch-dependent sign flips, H1/L1 swaps, and Monte Carlo dropout; MC64 was the main setting, with MC256/512 also evaluated |
| Experiment lead | autosota led idea generation, experiment design, scheduling, diagnosis, and analysis; its exact runtime version was not consistently recorded |
| Token usage | **Verifiable cumulative input, output, and reasoning-token accounting was not retained. Exact usage is unavailable** and cannot be reconstructed from training duration, file counts, or job counts |

Package versions are preserved in the training environment lock file (external reference: `snapshot/reports/environment_reproduce.lock.txt`), tooling environment lock file (external reference: `snapshot/reports/environment_tools.lock.txt`), and physics environment lock file (external reference: `snapshot/reports/environment_physics.lock.txt`). These are dependency records, not portable container images guaranteed to run across operating systems.

#### 5.2 Wall-clock time and experiment counts

| Campaign | Budget / deadline | Start → finish (UTC+8) | Actual wall-clock duration | Completed process jobs |
| --- | --- | --- | ---: | ---: |
| 12-hour campaign | 12 hours | Oct 11, 10:33 → Oct 11, 20:59 | 10.44 h | 36 |
| 10-hour campaign | 10 hours | Oct 12, 10:35 → Oct 12, 20:13 | 9.63 h | 105 |
| 14-hour campaign | 14 hours | Oct 12, 21:31 → Oct 13, 10:31 | 13.00 h | 117 |
| Final campaign | Deadline: Oct 14, 10:00; approximately 19.05 hours available at launch | Oct 13, 14:57 → Oct 14, 08:01 | **17.08 h** | **221** |

Environment preparation, baseline experiments, and five-fold expansion preceded or overlapped some campaigns. The overall experimentation period was October 10–14. This table is not a complete project cost ledger, and process-job counts must not be interpreted as counts of independently trained models.

The final campaign has the most complete record of effective training: all 66 planned pipelines completed, along with 11 additional seed pipelines. It comprised **97 training runs, 768 epochs, 92 prediction jobs, 10 weight-averaging jobs, 16 complete five-fold families, and 2 submissions**. All 97 training runs recorded nonzero optimizer updates, with no failed training jobs.

Cumulative single-GPU process durations were approximately **57.08 GPU-hours for training**, **9.77 GPU-hours for prediction**, and **0.13 GPU-hours for weight averaging**. These totals include data loading and other process overhead; they are not measurements of GPU kernel active time or exact FLOPs. Parallel execution across GPUs explains why cumulative process time exceeds the **17.08-hour** elapsed campaign duration. Exact project-wide GPU active time, total tokens, and monetary cost were not centrally metered and are not estimated here.

Evidence: rounds.csv (external reference: `rounds.csv`) and the final campaign results (external reference: `snapshot/reports/results_campaign_final_20261014.json`).

### 6. Ideas explored by autosota

autosota explored public-model adaptation; synthetic and physical pretraining; wider 1D networks; five-fold replication; larger signal/noise banks; SNR 4–10 versus 6–16; natural-amplitude/SNR mixtures; official negative-sample noise; longer continuation training; multiscale models with auxiliary physical tasks; split attention; pseudo-labels; ranking losses; SAM; BatchNorm freezing; amplitude perturbations; checkpoint averaging; MC64/256/512 inference; extra random seeds; optimizer-moment restoration; and hierarchical calibration and blending.

The main positive evidence came from strong Width32 physical pretraining, the stable GeM recovery, continued natural/weak-signal training, complementary V2 predictions, and the final calibrated ensemble. Other trials provided controls, small complementary branches, or no consistent gain. Five of seven audited old fine-tuning runs had not updated trainable weights, so their negative conclusions were withdrawn. All 97 final-campaign training runs recorded actual updates. Neither inclusion in an ensemble nor a local validation improvement establishes an isolated private-score contribution.

Detailed records are supplied in EXPERIMENT_INDEX.md (external reference: `EXPERIMENT_INDEX.md`), training_configs.csv (external reference: `training_configs.csv`), prediction_metrics.csv (external reference: `prediction_metrics.csv`), and jobs_full.json (external reference: `jobs_full.json`). Artifact and process counts are not counts of independently trained models.

### 7. Delivery and reproducibility scope

This delivery contains code, configurations, reports, and experiment records, including metadata for 89 active dependency nodes. **It includes no model weights, whitening arrays, cached predictions, submission CSV, or training/test waveforms.** The 67-checkpoint manifest documents the historical best solution; its listed binaries are not included.

Original waveform data and synthetic training banks are excluded. The synthetic banks were deleted during the October 23 cleanup; their generation code and metadata remain. Repeating training requires restored data, regenerated banks, and reconstruction of deleted non-final checkpoints. Inference with the retained trained models does not need synthetic banks. The original ZIP and packed waveforms still exist in the original workspace but were not copied into this package.

Running training, inference, or blending requires separately supplied data and model artifacts. See [BEST_SOLUTION.md](README.md#method) and [README.md](README.md) for code entry points and reproduction requirements. Historical inventories describe experiments at the time they ran, not the current availability of every unsuccessful or unselected checkpoint.
