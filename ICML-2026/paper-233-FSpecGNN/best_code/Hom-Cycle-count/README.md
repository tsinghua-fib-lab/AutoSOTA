# Reproduce FSpec Expressivity (Hom-Cycle-count)

This document provides the exact order to reproduce the expressivity experiments in this folder.

## 1) Generate counting features (required before training)
1. Compile `count.cpp` into `count.out`
2. Run `count.py` to generate `hom.npy` and `iso.npy`

```bash
g++ -O3 data/count.cpp -o data/count.out
python data/count.py
```

Expected generated files:
- `data/count/hom.npy`
- `data/count/iso.npy`

Notes:
- `data/count.py` reads `data/count/graph.npy`.
- If `count.out` is missing, `count.py` will fail.

## 2) Run training experiments
After step 1 is done:

```bash
bash main.count.sh
```

This script runs a list of tasks by calling `main.count.py` multiple times.

## 3) Outputs
Main outputs are written to:
- `result.count/` (task-wise result txt files)