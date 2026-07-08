# Reproducibility

This repository implements **Causal Preference Elicitation (CaPE)** and provides scripts to reproduce the experiments and figures in the paper.

## Quick start (one command)

After installing dependencies, run:

```bash
./reproduce_paper.sh
```

Outputs:
- Raw experiment outputs: `results/paper/`
- Generated figures: `figures/paper/`

Environment variables:
- `SEED0` (default: 123)
- `RESULTS_DIR` (default: `results/paper`)
- `FIGURES_DIR` (default: `figures/paper`)
- `DATASET_NPZ` (default: `data/causalbench/exports/weissmann_k562_50.npz`)

## Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate causalpe
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes on datasets

### Sachs
`./reproduce_paper.sh` calls `sachs_hitl_causal_dpo.py --download`, which downloads the observational subset used in the paper.

### CausalBench
CausalBench requires preparing an exported NPZ file. See `docs/CAUSALBENCH.md`.

## What is reproduced

The one-command script runs:
1. Synthetic benchmark (Fig. 1)
2. Sachs observational-only benchmark (Fig. 2) and a posterior visualization (Fig. 3-style)
3. Sachs benchmark with DAG-GFN prior (Fig. 5)
4. CausalBench K562-50 benchmark (Fig. 4)
5. Figure generation into `figures/paper/`
