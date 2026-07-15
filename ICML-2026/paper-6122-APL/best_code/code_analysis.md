# Code Analysis for Paper 6122 - CPBCC/PTBCC

## Evaluation Path
- **Main eval**: `run_val7.py` - standalone script with embedded CPBCC function + valence7 evaluation
- **Metric parser**: `utility.py` - `get_acc()` and `get_macro_f1()`
- **Eval command**: `python3 run_val7.py`
- **Output format**: Final block prints `Accuracy: XX.XX%` and `Macro-F1: XX.XX%` under `RESULTS for valence7`

## File Structure
- `run_val7.py` — **Optimization target**. Contains the CPBCC function AND the valence7 main evaluation loop.
- `method_CPBCC.py` — Full CPBCC with all datasets. Uses different init for non-valence datasets.
- `method_PTBCC.py` — PTBCC variant with 2D prototype structure.
- `utility.py` — Evaluation functions: `get_acc()` and `get_macro_f1()`.

## Key Initialization (valence7 path, lines 31-56 in run_val7.py)
- `phi_ik`: Majority voting init → normalized row-wise
- `temp_ksl`: Diagonal-strength=5 for self-emission, 1 for others
- `temp_jks`: Hardcoded [0.5, 0.2, 0.01] for all annotators/classes
- `beta_jks`: Computed from phi + theta dot product, initialized with 1e-5
- `a_ksl`: Accumulated from phi * theta per class, initialized with +1

## VI Loop Parameters
- `beta_jks *= 0.6` (scaling factor)
- `a_ksl *= 0.9` (scaling factor)
- `atol=1e-3` (convergence threshold)
- `max_iter=1000`
- S=3 prototypes

## Safe Modification Targets
- `run_val7.py`: CPBCC function init section (lines 31-56), VI loop (lines 86-145), S parameter (line 177)
- `utility.py`: `get_acc()` and `get_macro_f1()` — do NOT modify

## Risky Files (DO NOT MODIFY)
- `utility.py` metric computation
- Dataset files under `/repo/datasets/valence7/`
- Truth files
- `/tools/record_score.sh`

## Data
- Val7 dataset: `/repo/datasets/valence7/label.csv` and `truth.csv`
- 7 classes, ~100 annotators, sparse annotations
- No external downloads needed
