# Best-Arm Identification (BAI)

Python experiments for fixed-budget best-arm identification.

## Setup
- Install dependencies:
  - `python -m pip install -r requirements.txt`

## Run experiments
Run from within this folder so outputs land in `bai/results/`:
- `python exp1.py`
- `python exp2.py`
- `python exp3.py`
- `python exp4.py`
- `python exp5.py`
- `python exp6.py`
- `python exp7.py` (Sensitivity studies)
- `python exp8.py` (Ablations)
- `python lse_variance.py`

Notes:
- The default number of seeds/runs in the scripts can be large (slow); reduce `n` for quick sanity checks.
- Plots and text summaries are created by `plotexp.py` and saved under `bai/results/`.
- `lse_variance.py` measures `V_1st`, `V_full`, `abs_gap`, and `rel_gap` for the log-sum-exp objective and writes plots plus summaries under `results/lse_variance/`.