
# Breaking Multi-Task Curse: Reward-Weighted Evolution for Black-Box Many-Task Optimization - Code Repository

This repository contains the implementation of **MES-RET** and **sep-MES-RET**, designed for solving both standard many-task optimization problems and policy search tasks using reward-weighted evolution.

## Directory Structure

```
├── Algorithms/      # Contains the proposed algorithms
├── Problems/        # Contains benchmark functions and policy search environments
```

## Getting Started

> ⚠️ **Important:** All code in this repository is designed to run **within the [MTO-Platform (MToP)](https://github.com/intLyc/MTO-Platform)**.

Please **clone or download the MToP** and place corresponding files inside the `Algorithms/` and `Problems/` directories. More baselines can be found in MToP.

## For Policy Search Tasks

The policy search environments involve **Python-MATLAB hybrid programming** using **MATLAB's Python interface**. To run these tasks:

- MATLAB **R2023b or newer** is **mandatory**.
- A working Python environment (e.g., Python 3.10+) with required libraries (as shown in `MaT-Gym/requirements.txt`) must be properly configured.
- You can test and set the Python-MATLAB bridge via `pyenv` (more details in `MaT-Gym/Readme.md`).
