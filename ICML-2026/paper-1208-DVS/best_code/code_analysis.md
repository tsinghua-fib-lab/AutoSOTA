# Code Analysis for Paper 1208 (DVS)

## Evaluation Path
- Entry: `main.py --type sample --config qm9 --seed 42`
- Sampler: `Sampler_mol.sample()` in `sampler.py`
- Generates 10000 molecules, evaluates via `get_all_metrics()` (FCD, Valid, etc.) and `eval_graph_list()` (NSPDK)
- Metrics printed to stdout: `val w/o corr: X.XXXX` (Valid fraction), `FCD/Test: X.XXXX`, `NSPDK MMD: X.XXXX`

## Key Files
- `solver.py`: DVS adaptive sampling logic (`compute_adaptive_dt`, `get_pc_sampler`, predictors)
- `config/qm9.yaml`: Configuration (gamma=0.22, ref=1, predictor=Euler, tc=0)
- `sampler.py`: `Sampler_mol` class — orchestrates generation + evaluation
- `utils/loader.py`: `load_sampling_fn()` — wires config to solver
- `main.py`: Entry point

## Hardcoded DVS Parameters (not in config)
In `solver.py:get_pc_sampler()`:
- `min_dt=1e-6`, `max_dt=0.05` (default args)
- `ema_alpha=0.2`, `clip_scale=(0.2, 5.0)` (default args)

In `solver.py:compute_adaptive_dt()`:
- `beta=0.5` (hardcoded in function def)
- `ref=1` (overridden by config)
- `gamma` used in `get_pc_sampler` for EMA mixing but not `beta`

In `utils/loader.py:load_sampling_fn()`:
- Only passes `ref`, `gamma`, `tc` from config to `get_pc_sampler()`
- Does NOT pass `beta`, `ema_alpha`, `clip_scale`, `min_dt`, `max_dt`

## Safe Modification Targets
- `config/qm9.yaml`: Add new sample params (beta, ema_alpha, clip_scale, min_dt, max_dt, predictor)
- `solver.py:get_pc_sampler()`: Accept new params, pass to `compute_adaptive_dt()`
- `solver.py:compute_adaptive_dt()`: Parameterize beta (currently hardcoded 0.5)
- `utils/loader.py:load_sampling_fn()`: Pass new config params to solver
- `solver.py`: dt coupling (min vs geometric mean), step rejection, self-conditioning

## Risky Files (do not modify)
- `evaluation/`: Metric computation — RED LINE
- `data/`: Dataset loading — RED LINE
- `main.py`: Evaluation protocol — only modify for diagnostic logging
- `sampler.py`: Metric output format — do not change parsing

## Reusable Resources
- Checkpoint: `checkpoints/QM9/QM9.pth`
- QM9 data: `data/qm9_test_nx.pkl`, `data/moses_qm9.csv` (from MoFlow)
- No /paper_data mount
