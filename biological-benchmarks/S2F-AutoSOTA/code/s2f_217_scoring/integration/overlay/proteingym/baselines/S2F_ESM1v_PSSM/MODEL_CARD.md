# Model card — pending final name and authors

## Model and checkpoint identity

- Internal ID: `S2F_ESM1v_PSSM`.
- Structure branch: ESM-2-650M features plus S2F GVP and prediction head.
- Sequence scoring: five released ESM-1v 650M models, using masked and wild-type marginals.
- Evolutionary scoring: an MSA-derived frequency profile with the weighting and
  pseudocount implementation preserved in `vendor/s3f/s3f/msa_pssm.py`.
- Exact checkpoint: `model_best.pth`, SHA256
  `4942dea7cff7290b30f0ee21f8936ea790ba07d7cc87a6003eab3959f7a5759e`.
  This is a local S2F training output, not the original published `s2f.pth`.
- Original local run directory ended in `2026-09-05-09-56-16`; the 217-assay
  evaluation logs and weight manifest identify this checkpoint.

This is a multi-model method with MSA input. Comparisons that imply one 650M
model, no MSA, or unchanged original-author checkpoint weights would be inaccurate.

## Fixed inference recipe

The core model combines S2F, masked ESM-1v and wild-type ESM-1v scoring with
effective weights 0.56, 0.14 and 0.30. The sequential implementation first
mixes S2F/masked ESM with `w=0.20`, then mixes that result with wild-type
ESM using `c=0.30`, giving `(1-c)(1-w), (1-c)w, c`. At the mutation-row level, when every mutated site
has MSA coverage, the evolutionary mixture weight is
`sigmoid(8 * (log10(neff) - 1.47))`; otherwise the profile weight is zero.
Adaptive mode overrides the nominal constant PSSM weight of 0.25.
Long sequences use the archived multi-window handling. Multi-site log odds are
additive (`alpha=0`), and the structure pLDDT threshold is 0. Evaluation seed is
1024 and batch size is 4. No setting is re-optimized during this submission run.

The ESM ensemble computes the log of the arithmetic mean of member probabilities
(`logsumexp(logp) - log(5)`), not an arithmetic mean of member log probabilities.
The exact ensemble aggregation and MSA weights are defined by the frozen source
snapshot. `neff` here is computed by that implementation; it is not the
ProteinGym-provided sequence-identity-based MSA Neff column.

## Development-data exposure

The experiment launcher records four fixed non-Binding subsets:
`DEV_SUBSET` (40), `SEL_SUBSET` (24), `SEL_LONG_SUBSET` (10) and
`SEL80_SUBSET` (80). Their IDs are in `reference/development_assays.json`.
Model/inference decisions used performance on these benchmark subsets, including
selection of ensemble and profile settings. The 204 non-Binding results were
also inspected during development. Therefore the full benchmark is not an
untouched test set, even though this scoring entry point does not consume labels.

The 13 Binding assays were excluded from the earlier selection process according
to the experiment handoff and were evaluated in the final 217-assay run. They
are a functional-category shift, not proof that all 217 assays are new proteins.
No claim is made that all assay proteins are absent from foundation-model or
structure-pretraining data.

## Reporting

- Primary baseline for the method-improvement analysis: original S2F.
- 204 non-Binding per-assay mean: approximately 0.50408 vs published S2F 0.48027
  on the identical IDs. The historical local baseline 0.4776 is a separate run
  whose checkpoint-loading patch was not preserved; it is not used as the
  authoritative published-baseline value here.
- Full 217 official function/UniProt-balanced Spearman: approximately 0.46634.
  Use the accompanying official outputs for other metrics and category results.
- Development-label exposure precludes a confirmatory independent-test
  significance claim from the subset bootstrap alone.
- This package does not establish overall ProteinGym SOTA. A single-model
  official performance table cannot establish a global rank.

The maintainers should determine eligibility/placement given the selection
protocol. Public release of code and scores is separate from acceptance on a
particular leaderboard track.

## Intended use and limitations

Research evaluation of amino-acid substitution effects with a target sequence,
matching predicted structure and MSA. Indels are not supported. This package is
not validated for clinical decisions. Profile coverage and long-sequence/structure
mapping affect performance. The current full score is a result on ProteinGym
v1.3, not a guarantee on new assays.
