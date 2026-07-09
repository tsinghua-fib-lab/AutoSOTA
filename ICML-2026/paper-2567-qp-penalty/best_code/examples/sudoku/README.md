# Sudoku

Sudoku experiments adapted from OptNet (__Section 4__ of the paper). The working directory is `examples/sudoku/`.

## Quick Start

```bash
# 1. Generate puzzle data
python create.py --boardSz 2 --nSamples 100

# 2. Train
python train.py --boardSz 2 --nEpoch 100 dXPPEq
```

## Arguments

| Argument      | Description                                                                 |
| ------------- | --------------------------------------------------------------------------- |
| `--boardSz` | n corresponds to an n×n Sudoku puzzle. 2, 3, and 4 are available.          |
| `--nEpoch`  | Number of training epochs.                                                  |
| `[model]`   | `dXPPEq`, corresponding to dXPP in the equality constrained formulation. |
