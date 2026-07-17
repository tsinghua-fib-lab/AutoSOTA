# Baseline Methods

This directory contains wrapper scripts for running baseline methods for comparison.

## Setup

The baseline implementations are not included directly in this repository.
Instead, we provide instructions and wrapper scripts to use the original implementations.

### MMFM (Multi-Marginal Flow Matching)

**Reference:** Tong et al. "Improving and generalizing flow-based generative models with minibatch optimal transport" arXiv:2302.00482 (2023).

**Repository:** https://github.com/atong01/conditional-flow-matching

```bash
# Install via pip
pip install torchcfm

# Or clone the repository
git clone https://github.com/atong01/conditional-flow-matching.git
cd conditional-flow-matching
pip install -e .
```

### 3MSBM (Momentum Multi-Marginal Schrödinger Bridge Matching)

**Reference:** Theodoropoulos et al. "Momentum multi-marginal Schrödinger bridge matching" arXiv:2506.10168 (2025).

**Repository:** https://github.com/nikitadobrokhtov/mmsbm

```bash
# Clone the repository
git clone https://github.com/nikitadobrokhtov/mmsbm.git
cd mmsbm
pip install -r requirements.txt
```

## Running Baselines

After setting up the baseline repositories, use the wrapper scripts:

```bash
# Run MMFM baseline
python experiments/baselines/run_mmfm.py --dataset singlecell
python experiments/baselines/run_mmfm.py --dataset gulfofmexico
python experiments/baselines/run_mmfm.py --dataset beijingair

# Run 3MSBM baseline
python experiments/baselines/run_3msbm.py --dataset gulfofmexico
python experiments/baselines/run_3msbm.py --dataset beijingair
```

The wrapper scripts use our data loading utilities to ensure consistent preprocessing.

## Output Format

Baseline methods save their evaluation metrics in JSON format compatible with `experiments/compare.py`:

```
final_trajectories/
├── singlecell/
│   ├── OTPFM/
│   │   ├── w2/
│   │   │   └── seed*.metrics.json
│   │   └── ...
│   └── MMFM/
│       └── mmfm_*.metrics.json
├── gulfofmexico/
│   └── ...
└── beijingair/
    └── ...
```

Each `.metrics.json` file contains:
```json
{
  "w2_t0": 0.0,
  "w2_t1": 0.123,
  "w2_rest": 0.456,
  ...
}
```

## Comparing Results

Use the unified comparison script:

```bash
# Compare all methods for a dataset
python experiments/compare.py --dataset singlecell

# Generate LaTeX tables
python experiments/compare.py --all --latex

# Compare specific methods
python experiments/compare.py --dataset beijingair --methods OTPFM MMFM
```
