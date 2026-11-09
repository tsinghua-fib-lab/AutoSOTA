# NESP base repo — 7th-place public solution (Laura Romar), headless transcription

Provenance: Kaggle notebook `lauraromar/novozymes-7th` ("7th Place Solution - Novozymes
Competition", private LB 0.53258, rank 7 of 2483 on
`novozymes-enzyme-stability-prediction`). This repository is a faithful transcription of
that notebook into a runnable pipeline (see the module docstring of
`pipeline/run_pipeline.py` for the transcription notes and the two documented fixes).

## Layout

```
pipeline/run_pipeline.py    # full pipeline: features -> LGBM CV+fit -> 5 components -> rank-average ensemble
requirements.txt
```

## Run

```bash
python pipeline/run_pipeline.py --paper-data /paper_data --out-dir outputs
```

Environment requirements:

* `dssp` binary (`mkdssp`) on PATH — the DSSP features are computed with Biopython's
  `Bio.PDB.DSSP`. Install with `apt-get install -y dssp` or `conda install -c bioconda dssp`.
* Python packages in `requirements.txt` (lightgbm, scikit-learn, scipy, pandas, numpy,
  biopython, biopandas).

Outputs:

* `outputs/submission.csv` — the final 2413-row submission (required).
* `outputs/components/*.csv` — the five component predictions (transparency/debugging).
* `outputs/oof_jin.csv`, `outputs/cv_report.json` — internal CV diagnostics.

## Inputs consumed from /paper_data (read-only)

* `jin/train_Jin_v2.txt` + `jin/pdbs/*.pdb` — Jin v2 ddG dataset (JinyuanSun,
  github.com/JinyuanSun/mutation-stability-data v2) plus wild-type PDB structures
  downloaded from RCSB; `jin/pdbs/wt.pdb` is the AlphaFold2 wild-type model of the
  test protein.
* `kvigly_test_mutation_pdbs/` — AF2 models of WT + every test mutant
  (Kaggle dataset `roberthatch/nesp-kvigly-test-mutation-pdbs`).
* `components/thermonet2_submission.csv`, `components/rosetta_scores.csv` — frozen
  third-party component predictions vendored read-only (outputs of the public Kaggle
  kernels `vslaykovsky/nesp-thermonet-v2` and `shlomoron/nesp-relaxed-rosetta-scores`).
* `test.csv`, `sample_submission.csv` — competition files.

## Red lines

* `test_labels.csv` is for evaluation only (`eval/score_test.py`); the pipeline must
  never read it, train on it, or use it as a feature.
* Scores must come from actual pipeline runs; never hand-edit outputs.
