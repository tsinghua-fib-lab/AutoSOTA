# AutoSOTA optimization of S2F: research report

**Source report updated: October 11, 2026. English archive prepared: October 12, 2026.**

This report brings together the dev-40, full-204 non-Binding, and full-217
ProteinGym results. The working method identifier is `S2F_ESM1v_PSSM`; the final
method name, authors, and weight-hosting URL remain pending. The supplied archive
contains source code and the report, without datasets, per-mutation predictions,
model weights, or run logs. Aggregate values below are transcribed from the
supplied report, rather than newly recomputed during repository integration.

## 1. Task, baseline, and evaluation protocol

The task is to rank the fitness effects of amino-acid substitutions. Spearman
rank correlation is computed between predictions and experimental measurements
within each deep mutational scanning (DMS) assay. The baseline is **S2F** from
*Multi-Scale Representation Learning for Protein Fitness Prediction*
(NeurIPS 2024); historical local comparisons use the authors' released `s2f.pth`.

Three evaluation scopes use different aggregation rules:

| Scope | Purpose | Aggregation |
|---|---|---|
| dev-40 | Forty of the 204 non-Binding assays; intermediate observations during iteration | Arithmetic mean of 40 per-assay Spearman values |
| full-204 | All 204 non-Binding assays | Arithmetic mean of 204 per-assay Spearman values |
| full-217 | All 217 ProteinGym v1.3 substitution assays, including 13 Binding assays | Official function/UniProt-balanced Spearman; the per-assay arithmetic mean is also reported for diagnostics |

AutoSOTA also used **sel-80**, which does not overlap dev-40, for method selection.
Selection used experimental labels from ProteinGym subsets, and the full-204
results were inspected during development. These scores therefore do not measure
performance on an independent, untouched external test set. The inference entry
point itself does not read experimental labels; the limitation comes from the
**method-selection process**.

ProteinGym's [documented aggregation](https://github.com/OATML-Markslab/ProteinGym#results)
first groups by UniProt and functional category, then averages the category
results. This differs from assigning equal weight to every assay.

## 2. AutoSOTA experiment process

The records cover **August 1–September 8, 2026: seven rounds and 89 iterations**.
Approximately 47 iterations concerned training, approximately 30 concerned
inference scoring, and the remainder concerned infrastructure and validation.
Researchers supplied the research directions, mechanism hypotheses, and
acceptance constraints in an idea list. AutoSOTA implemented candidates, searched
settings, and executed evaluations; the record does not attribute all ideas to
autonomous discovery.

Early rounds reset their local baseline. Their relative gains cannot be compounded
or substituted for the final comparison against original S2F.

| Round | Iterations | Main experiments and observations | Intermediate dev-40 score |
|---|---:|---|---:|
| R1 | 26 | Architecture, features, and training hyperparameters; 20-epoch runs were used only to screen directions | 0.4239 → 0.4318 |
| R2 | 21 | Systematic training search at 100 epochs; most changes did not exceed training-seed noise | 0.4446 → 0.4446 |
| R3 | 6 | Found that the original inference path omitted sequence-branch logits; added sequence-score fusion | 0.4450 → 0.4468 |
| R4 | 10 | Added an unmasked wild-type sequence branch and measured training noise across seeds | 0.4468 → 0.4500 |
| R5 | 10 | Calibrated the sel-80 selection set and searched fusion weights again | 0.4500 → 0.4525 |
| R6 | 14 | Added MSA evolutionary information and a five-model ESM-1v scoring ensemble | 0.4525 → 0.4653 |
| R7 | 2 | Reproduced the final configuration and reevaluated the original checkpoint on 204 assays; stopped after API funds were exhausted | Final: 0.4653 |

### 2.1 Why the search moved from training to inference

R1 tested GVP depth, edge features, contact radius, dropout, and layer aggregation.
The best intermediate 20-epoch score rose from 0.4239 to 0.4318, but reproduction
yielded only 0.4265, indicating substantial training randomness. R2 increased
training to 100 epochs and explored additional geometry features, losses,
regularization, and longer pretraining. Its 21 iterations produced no stable gain.

For example, extending pretraining to 200 epochs improved CATH validation accuracy
but reduced ProteinGym dev-40 by approximately 0.0045. Three training seeds with
the same configuration produced 0.4468, 0.4402, and 0.4391, a spread larger than
many candidate gains. Subsequent work prioritized **inference changes on the same
checkpoint**, reducing the risk of attributing seed differences to a method change.

In R3, code inspection showed that S2F produced `sequence_logits`, but
`plddt_threshold: 0` in the released configuration bypassed the relevant inference
branch. Combining sequence logits with structure-branch log probabilities raised
the intermediate score from 0.4450 to 0.4468. R4 added an unmasked wild-type (WT)
marginal, raising the score to 0.4500, and retained multi-window scoring for long
proteins. Ensembling predictions from multiple GVP training seeds did not give a
stable improvement.

### 2.2 Selection sets and the final scoring method

During R4, a 24-assay selection set judged one change favorably: approximately
+0.0004 on that set, versus approximately −0.0030 on dev-40. R5 therefore created
sel-80, disjoint from dev-40, and calibrated its directional judgments using
known results. Candidates were accepted only above an approximate **+0.002** gain.
Caching branch scores per mutation made offline fusion-weight searches possible.
With the ESM-2 sequence branch, R5 found the masked branch nearly redundant; this
finding required reexamination after switching to ESM-1v.

R6 introduced an MSA-derived, Henikoff-weighted position-specific scoring matrix
(PSSM). A fixed evolutionary weight was unstable, so the weight depended on the
alignment's effective depth, `Neff`:

```text
e = sigmoid(8 × (log10(Neff) − 1.47))
```

Shallow alignments rely primarily on the structure and language-model scores;
deeper alignments receive more profile weight. This gate contributed approximately
+0.0067 on sel-80 but approximately −0.0020 in the corresponding dev-40 ablation.
The negative dev-40 result is retained here.

R6 compared ESM-1v with ESM-2-3B sequence scoring and selected an **ensemble of five
ESM-1v models**. Retraining GVP directly on ESM-1v representations reduced dev-40
to 0.4289. The final method therefore retained the S2F structure branch driven by
ESM-2 features and used ESM-1v only for substitution-effect scoring. Searching
fusion weights again after the backbone change restored weight to the masked
branch.

R7 reproduced the final configuration and reevaluated original `s2f.pth` on
full-204. AutoSOTA stopped after the key measurements because its API funds were
exhausted. The full-217 run and official scoring pipeline were executed separately
afterward.

### 2.3 Combining structure, sequence, and evolutionary information

**Structure branch.** The S2F architecture is retained: frozen ESM-2-650M supplies
per-residue features, and a GVP predicts distributions at mutation sites from a
protein structure graph. The actual 217-assay run loaded a locally retrained
`model_best.pth`, not the authors' released `s2f.pth`. The latter is a baseline
checkpoint. The method must therefore not be described as preserving the original
authors' weights unchanged.

**Two sequence branches.** Each of five published ESM-1v 650M models supplies
masked and unmasked WT marginals. The masked branch hides the prediction site;
the WT branch retains the wild-type context. They supply log-probability differences
for the same substitution under two conditions. The ensemble takes an arithmetic
mean of probabilities **before** taking the logarithm; it does not average five
log probabilities. The structure branch still uses ESM-2 features, so the complete
method uses multiple language models and is not a single 650M model.

**MSA evolutionary branch.** Lowercase insertion columns are removed from each
`.a2m` alignment, alignment coordinates are checked against the WT sequence, and
Henikoff sequence weights are recomputed within the target region. Frequencies
over the 20 amino acids receive a **0.5 pseudocount**, ensuring finite scores for
unobserved substitutions. For `wt→mt` at position `i`, the profile score is
`log f(mt, i) − log f(wt, i)`. The method reweights the original MSA; it does not
directly use ProteinGym's precomputed MSA sequence weights.

Let `L_S2F(v)`, `L_mask(v)`, and `L_WT(v)` denote mutant-minus-WT log-probability
scores for variant `v`. The code first combines S2F and the masked branch with
`w=0.20`, then adds WT scoring with `c=0.30`:

```text
M(v) = 0.56 L_S2F(v) + 0.14 L_mask(v) + 0.30 L_WT(v)
e(a) = sigmoid[8 × (log10 Neff(a) − 1.47)]
S(v) = [1 − e(a) × C(v)] M(v) + e(a) × C(v) PSSM(v)
```

`Neff(a)` is the Henikoff effective alignment depth for assay `a`. `C(v)=1` only
if the MSA covers every mutated site; otherwise it is zero and scoring falls back
to the model mixture. Multi-site substitution log ratios are additive (`alpha=0`).
Proteins exceeding the ESM context length are scored through multiple windows
and averaged. When adaptive mode is enabled, `e(a)` overrides the nominal
`S2F_PSSM_WEIGHT=0.25`.

Earlier records describing coefficients **0.50 / 0.20 / 0.30** misread the sequential
fusion. The actual effective coefficients are **0.56 / 0.14 / 0.30**.

The profile's contribution depends on its own information content: shallow
alignments have less reliable frequency estimates. Among dev-40 assays, 23 had
`e<0.05`, six had `0.05≤e≤0.5`, and 11 had `e>0.5`, showing a steep gate. Settings
were chosen using sel-80 and other benchmark subsets, rather than specified
independently of experimental labels.

## 3. dev-40 and full-204 results

The 204-assay scope includes all non-Binding substitution assays:
**OrganismalFitness 77, Stability 66, Activity 43, and Expression 18**. dev-40 is
a category-sampled observation set; sel-80 is the main selection set. full-204
assigns equal weight to each assay's Spearman value, rather than weighting by
mutation count. It is the full **non-Binding subset**, not the official full
217-assay benchmark score.

The following local baseline values, 0.4436 and 0.4776, were obtained by rerunning
the authors' released `s2f.pth` in the local evaluation workflow. They were not
copied from the original paper. Relative gains use `(AutoSOTA − S2F) / S2F`.

| Scope | Historical local original S2F | AutoSOTA | Absolute Spearman gain | Relative gain |
|---|---:|---:|---:|---:|
| dev-40 | 0.4436 | 0.4653 | +0.0217 | +4.89% |
| full-204, non-Binding | 0.4776 | 0.5041 | +0.0265 | +5.55% |

On those same 204 assay identifiers, the published ProteinGym per-assay table
gives original S2F a mean of **0.4802745**. Against AutoSOTA's precise **0.5040833**,
the absolute gain is **0.0238088**, or approximately **4.96%**. The historical
local baseline differs from the published value by approximately 0.0027, and its
temporary checkpoint-loading patch was not fully preserved. Public comparisons
should therefore prioritize the published baseline. Any figure showing +5.55%
must label it as relative to the **historical local reevaluation**.

On dev-40, the published S2F table gives approximately **0.4432**, while the later
217-assay predictions give AutoSOTA approximately **0.46535** on the same 40 IDs.
These form another valid published-table comparison. The historical +4.89%
specifically compares against the local 0.4436 baseline; baseline provenance and
numerical precision must accompany the values.

### 3.1 Per-assay direction and functional categories

The following comparison uses published original S2F and AutoSOTA on identical
204-assay IDs. The per-assay table stores three-decimal values, so the means and
win/tie/loss counts below use that precision. **139 assays improved, two tied,
and 63 declined**; the median per-assay difference was **+0.017**. Improvement
was not universal.

| Non-Binding category | Assays | AutoSOTA mean | Published S2F mean | Mean difference | Win/tie/loss |
|---|---:|---:|---:|---:|---:|
| Activity | 43 | 0.4933 | 0.4639 | +0.0293 | 29/0/14 |
| Expression | 18 | 0.4476 | 0.4564 | −0.0089 | 12/0/6 |
| OrganismalFitness | 77 | 0.4613 | 0.4198 | +0.0416 | 54/2/21 |
| Stability | 66 | 0.5764 | 0.5680 | +0.0084 | 44/0/22 |
| **All 204** | **204** | **0.5041** | **0.4803** | **+0.0238** | **139/2/63** |

Most of the mean gain came from OrganismalFitness and Activity. Although 12 of
18 Expression assays improved, a few larger negative differences lowered that
category's mean below S2F. The full-204 gain exceeds the dev-40 gain, but category
composition differs and full-204 contains development assays. This does not
establish generalization to independent new data.

In R6's first full-set evaluation, the final stack improved on the preceding
internal stack from approximately 0.4903 to 0.5041. That approximately **+0.0137**
is an internal iteration comparison, distinct from the **+0.0265** comparison
against the historical original-checkpoint baseline.

### 3.2 Component ablations and stability

These ablations recombine cached per-mutation components from the same **locally
trained checkpoint**. Its pure-S2F row is not the authors' released `s2f.pth` and
cannot replace the original baseline above. The final column covers only
**152 of 204** assays with available component caches; the actual 0.5041 result
comes from a complete 204-assay evaluation.

| Configuration | dev-40 | sel-80 | 152 assays with available caches |
|---|---:|---:|---:|
| Local checkpoint, pure S2F scoring | 0.4451 | 0.5036 | 0.4926 |
| Add masked ESM-1v, weight 0.5 | 0.4636 | 0.5185 | 0.5083 |
| Add masked ESM-1v, weight 0.2 | 0.4576 | 0.5136 | 0.5041 |
| Add WT ESM-1v, `w=0.20, c=0.30` | **0.4673** | 0.5203 | 0.5117 |
| Add fixed-weight PSSM | 0.4620 | 0.5221 | 0.5119 |
| Use Neff-adaptive PSSM, the final configuration | 0.4653 | **0.5270** | **0.5138** |

Upgraded sequence scoring contributes the largest dev-40 gain. Adaptive PSSM
reduces dev-40 from 0.4673 to 0.4653 but raises sel-80 from 0.5203 to 0.5270 and
the 152-assay cached mean from 0.5117 to 0.5138. Fixed-weight PSSM is substantially
worse on dev-40, supporting MSA-depth-dependent weighting.

Four independent training trajectories produced final dev-40 scores of
**0.4653 / 0.4638 / 0.4610 / 0.4602**. Absolute scores remain sensitive to the
training seed; reporting only the best trajectory would omit that variation.

**Coverage and residual error.** The evolutionary term participates only when
the MSA covers every mutation site. Per-mutation coverage was approximately
**90.3% on dev-40**, **86.9% on sel-80**, and **70.2% in the available 152-assay
component-cache analysis**. Roughly 30% of mutations in that last subset used
only structure and language-model scores. The other 52 assays lack component
caches, so 70.2% is not an exact coverage estimate for all 204 assays. Long
proteins remain a major observed source of residual error; multi-window scoring
does not fully resolve it.

dev-40, sel-80, and full-204 all come from ProteinGym, and sel-80 labels informed
method selection. These ablations describe behavior on those sets rather than
proving equal gains on future unseen assays.

On the same 204-assay scope, published **S2F-MSA averages approximately 0.5136**,
above AutoSOTA's 0.5041. The supported improvement is over **original S2F**, not
over the entire S2F family or every model on the subset.

## 4. Full-217 ProteinGym results

On **September 24–25, 2026**, four A100 80 GB GPUs completed all **217 substitution
assays**, covering **2,465,767 mutants**. The predictions were subsequently
checked through the archived ProteinGym `merge.py` and
`performance_DMS_benchmarks.py` pipeline.

| Metric | AutoSOTA | Baseline or interpretation |
|---|---:|---|
| Official function/UniProt-balanced Spearman | **0.46633717**; three-decimal display: 0.466 | Published S2F: **0.45560865**, displayed as 0.456; absolute gain **+0.01072852**, relative gain **+2.35%** |
| Arithmetic mean of 217 per-assay Spearman values | 0.49517 | Diagnostic mean, not the primary grouped metric |
| AUC | 0.756 | Official pipeline |
| MCC | 0.367 | Official pipeline |
| NDCG | 0.781 | Official pipeline |
| Top recall | 0.236 | Official pipeline |

Within the same full-217 predictions, the 204 non-Binding assays average
**0.50408**, and all 217 assays average **0.49517**. The primary **0.46634** score
uses different category/UniProt aggregation weights. Subtracting 0.4663 from
0.5041 does not measure the effect of adding 13 Binding assays.

The grouped Binding score is approximately **0.36746**, indicating a weaker
category. This is distinct from its **0.3554** per-assay arithmetic mean below.
Separating Binding clarifies how the earlier 204-assay analysis relates to the
complete evaluation. Win/tie/loss counts use the published three-decimal assay
table.

| Scope | Assays | AutoSOTA arithmetic mean | Published S2F arithmetic mean | Difference | Win/tie/loss |
|---|---:|---:|---:|---:|---:|
| Non-Binding, full-204 | 204 | 0.5041 | 0.4803 | +0.0238 | 139/2/63 |
| Binding | 13 | 0.3554 | 0.3816 | −0.0262 | 6/0/7 |
| All 217, arithmetic mean | 217 | 0.4952 | 0.4744 | +0.0208 | 145/2/70 |

Binding was excluded from the earlier 204-assay study. All 13 Binding assays
were completed in the final run, but their mean was lower than published S2F;
Expression's mean also declined within the non-Binding subset. The historical
full-204 +5.55% must not be described as a gain on every one of the 217 assays.
Under official grouped aggregation, the relative gain is **2.35%**, rather than
the approximately **4.39%** gain obtained from rounded arithmetic means.

The full-217 S2F baseline comes from the **published ProteinGym per-assay score
table**, not a new run of the authors' checkpoint in this evaluation. Published
S2F-MSA has a grouped score of approximately **0.488**, above AutoSOTA's 0.466.
The result supports an improvement over published original S2F under the same
aggregation rule; it does not establish first place on ProteinGym or within the
S2F family.

An official-pipeline table containing only this submitted method can emit
`rank=1`. That is an internal rank among one included model, not a global
leaderboard position. A zero bootstrap standard error in that single-model
comparison measures a difference from itself, not zero uncertainty about the
method's performance.

## 5. Reproduction scope, limitations, and publication status

- The supplied archive excludes ProteinGym/CATH datasets, structures, MSAs,
  reference assay CSVs, per-mutation predictions, component caches, model weights,
  and run logs. These external inputs must be prepared separately. The original
  AutoSOTA `config.yaml` contained access keys and was not supplied.
- The approximately **2.68 GB** trained checkpoint is not included. The supplied
  weight manifest records its expected size and SHA256, but its download URL is
  pending. Final method naming, authorship, weight hosting, and benchmark placement
  still need to be determined. Publishing this source archive does not establish
  acceptance by ProteinGym maintainers.
- The portable release entry point was checked in the preserved GPU environment
  on **four representative assays**. Official three-decimal Spearman values
  matched, but per-mutation scores were not bit-identical. It has not rerun all
  217 assays; the full predictions came from the historical full-217 run.
  Installation of the older CUDA/PyG dependencies on a fresh machine remains
  unverified.

The English translation preserves the source report's results and qualifications.
Repository integration did not retrain the model, rerun the protein benchmark,
or independently recompute the reported scores. See
[source provenance](provenance.md) for the original report hash and archived code
checksums.

## References and source entry points

- [S3F/S2F upstream source](https://github.com/DeepGraphLearning/S3F) and
  [NeurIPS 2024 paper](https://arxiv.org/abs/2412.01108).
- [ProteinGym official repository](https://github.com/OATML-Markslab/ProteinGym)
  and [v1.3 resources](https://github.com/OATML-Markslab/ProteinGym#resources).
- [Fixed inference model card](../code/s2f_217_scoring/code/MODEL_CARD.md).
- [Weight identities](../code/s2f_217_scoring/code/reference/weights.json) and
  [vendored source identities](../code/s2f_217_scoring/code/reference/source_provenance.json).
