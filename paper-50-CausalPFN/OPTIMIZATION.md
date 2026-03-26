# Paper 50 — CausalPFN

**Full title:** *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*

**Registered metric movement (internal ledger):** -15.86%(0.1829→0.1539 IHDP PEHE, lower is better)

## Summary

On IHDP, PEHE improved from **0.1829** to **0.1539** by combining **propensity-feature augmentation** with robust ensembling. The pipeline appends `P(T=1|X)` as an extra feature, then averages predictions across **multi-seed bootstraps** (`seeds=[42,43]`, `N_BOOT=3/seed`, `BOOT_FRAC=0.92`) and **multi-temperature** settings (`T=[0.3,0.5,0.7,0.9,2,4,8]`). Best run is logged as `run_20260325_014145`.

## Key ideas

- Add treatment propensity as an explicit signal for effect estimation.
- Use bootstrap diversity across seeds to reduce PEHE variance.
- Mix low/high temperatures to stabilize confidence calibration across samples.

## Where to look

- `eval.py`, `eval_bootstrap.py`, `eval_multi_prop.py`, `eval_prop_temp.py`
- this repository snapshot under `paper-50-CausalPFN/`
