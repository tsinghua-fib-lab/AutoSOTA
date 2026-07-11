# Code Analysis for Paper 4082 (GenUnfold) SOTA

## Evaluation Path
- eval_command: python3 scripts/eval_repro.py
- Loads pre-generated curves from scripts/data/generated_curves.npy and true from scripts/data/true_curves.npy
- Clips generated curves to [0,1], then computes Rel_l2, FID, Force-JSD, Energy-JSD

## Critical Blockers
1. data/features/ directory MISSING
2. data/pdb_files/pdb.cif is 0 bytes
3. dataset.pkl has empty training data lists
4. Cannot retrain model or run inference without features

## Strategy
Post-process generated curves with general signal processing techniques.
