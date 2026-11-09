# Paper 9003 (NESP) — SOTA code analysis

## Evaluation path

* Entry point (unchanged): `cd /repo && bash scripts/eval.sh`
* `scripts/eval.sh` does three things:
  1. `taskset -c 0-7 python pipeline/run_pipeline.py --paper-data /paper_data --out-dir outputs`
     (thread pinning is mandatory on this 128-CPU contended node; the LightGBM
     build ignores `OMP_NUM_THREADS` — see `pipeline/run_pipeline.py::get_model`).
  2. `python /paper_data/eval/validate_submission.py outputs/submission.csv`
  3. `python /paper_data/eval/score_test.py outputs/submission.csv`
* **Metric parser**: last stdout line of step 3, `Spearman=<value>` (5 decimals).
  Preceded by `n_test=2413` and the private-LB reference levels.
* **Guardrail**: `outputs/cv_report.json::cv_spearman_overall` (7-fold GroupKFold-by-wildtype
  Spearman of the LGBM component on the Jin v2 ddG set). Baseline 0.52606.
* **Resource**: full eval wall clock 2m41s at `NESP_LGBM_THREADS=8`.
* Exit code 0 = scored, 2 = submission missing / bad schema / incomplete 2413-row coverage.

Manifest `eval_command` is already an in-container command; no manifest repair was needed.

## Train / inference path

Single script, `pipeline/run_pipeline.py` (621 lines), a faithful headless transcription of
`lauraromar/novozymes-7th`. Flow:

1. **LGBM component (A, weight 0.30).** Train on `/paper_data/jin/train_Jin_v2.txt`
   (4035 rows after dropping 1X0J/3O39) — the Jin v2 ddG dataset — with mutation and
   DSSP-derived features computed from the 184 wild-type PDBs in `/paper_data/jin/pdbs`.
   7-fold `GroupKFold` on the `group` key (per wild-type sequence) for OOF + CV; then a
   full refit. Predict the 2413 test rows. Settings preserved: `n_estimators=5000,
   max_depth=4, learning_rate=0.005, colsample_bytree=1, random_state=42`.
   `FEATURES` (50 entries) is a module-level list.
2. **WT-SASA component (B, 0.10).** Shrake-Rupley SASA of the wild-type residue at the
   mutation position, read from the AF2 wild-type model
   `kvigly_test_mutation_pdbs/WT_unrelaxed_rank_1_model_3.pdb` (1-based `position+1`).
   *Degenerate*: only 165 distinct values over 2413 rows, 241 exact zeros.
3. **B-factor component (C, 0.10).** `b_factor` (AF2 pLDDT) of the mutated residue in the
   mutant AF2 model minus the wild-type model at the same position. 1087 distinct values,
   10 zeros (missing PDBs).
4. **ThermoNet-v2 (D, 0.20)** and **Rosetta relaxed scores (E, 0.30)**: frozen read-only
   CSVs under `/paper_data/components/`. Note `rosetta_scores.csv` is *already a rank
   vector* (values 1..2413 with half-integer ties), so `rankdata` is idempotent on it.
5. **Ensemble.** `0.10*rank(SASA) + 0.30*rank(LGBM) + 0.20*rank(ThermoNet)
   + 0.10*rank(Bfactor) + 0.30*rank(Rosetta)`, divided by `n_test`; written to
   `outputs/submission.csv`.

Every run writes the three regenerable component CSVs to `outputs/components/`, so the
ensemble can be recombined without refitting anything.

## Test-set structure (input-only analysis, no labels)

* All 2413 rows have `data_source=Novozymes`, `pH=8`.
* 2335 rows are **single substitutions** of the 221-residue wild-type enzyme (Hamming 1);
  1 row is the wild type itself; 77 rows are **single deletions** (220 residues, PDB names
  `{AA1}{pos}__unrelaxed_rank_1_model_3.pdb`).
* Zero sequence overlap with `train.csv` (31390 rows, 28981 distinct sequences), and
  `train.csv` contains **no** Novozymes rows. `train_updates_20220929.csv` only adds
  `seq_id` keys (all 2434 payload columns are NaN).
* Consequently the competition `train.csv` `tm` labels carry no rank signal for this test
  set, and the LGBM component is a **cross-domain transfer** model (Jin ddG -> one enzyme's
  tm). `/paper_data/kvigly_test_mutation_pdbs` (2412 mutant AF2 models + wild type) is the
  only untapped in-domain resource.

## Component diagnostics (rank correlation matrix, inputs only)

|        | lgbm | sasa | bfac | thermo | rosetta |
|--------|------|------|------|--------|---------|
| lgbm   | 1.00 | 0.76 | 0.20 | 0.75   | 0.45    |
| sasa   | 0.76 | 1.00 | 0.22 | 0.72   | 0.38    |
| bfac   | 0.20 | 0.22 | 1.00 | 0.29   | 0.41    |
| thermo | 0.75 | 0.72 | 0.29 | 1.00   | 0.53    |
| rosetta| 0.45 | 0.38 | 0.41 | 0.53   | 1.00    |

{lgbm, sasa, thermo} is a tightly correlated cluster; `bfac` is nearly decorrelated from
everything; `rosetta` is the least correlated among the strong components. The published
weights (0.10/0.30/0.20/0.10/0.30) are the author's hand choice.

## Reusable `/paper_data` resources (read-only)

| Path | Use |
|------|-----|
| `eval/score_test.py`, `eval/validate_submission.py`, `test_labels.csv` | scorer — **red line**, only the scorer may read labels |
| `sample_submission.csv` | 2413 seq_ids / submission schema |
| `test.csv` | the 2413 test rows |
| `train.csv`, `train_updates_20220929.csv` | competition train data (no sequence overlap with test; no Novozymes rows) |
| `jin/train_Jin_v2.txt`, `jin/pdbs/` | LGBM training data + 184 WT PDBs |
| `kvigly_test_mutation_pdbs/` | AF2 WT + 2412 mutant models — **untapped structural signal** |
| `components/thermonet2_submission.csv`, `components/rosetta_scores.csv` | frozen third-party components (readable, never editable) |
| `wildtype_structure_prediction_af2.pdb` | organizer AF2 WT prediction (bit-identical to `jin/pdbs/wt.pdb`) |

## Risky files (do not touch)

* `/paper_data/**` — read-only mount, frozen components and labels.
* `/repo/outputs/submission.csv` — the scored artifact; must only ever be produced by the
  pipeline, never hand-edited.
* `pipeline/run_pipeline.py::FEATURES` / `get_model` — the paper's hyperparameters are part
  of the reproduction; changes there must be justified as an algorithmic change, not a
  re-tuning of the published settings.

## Safe modification targets

* `outputs/components/*.csv` (regenerable) and any *new* component CSVs.
* New modules under `/repo/sota/` and a new pipeline stage that consumes them.
* The ensemble combination step (weights / stacking) — explicitly named as a legitimate
  lever in the manifest `known_levers` and the structured objective.

## Planned levers (from the idea library + code evidence)

1. Re-fit the five ensemble weights (free; component CSVs are cached).
2. Add self-built structural components from the AF2 mutant models: local CA deviation,
   AF2 pLDDT delta, SASA delta at the site, heavy-atom contact-count delta.
3. Improve/replace the LGBM component (the weakest of the three strong components).
