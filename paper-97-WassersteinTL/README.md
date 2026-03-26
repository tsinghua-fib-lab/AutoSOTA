# Wasserstein Transfer Learning (WaTL)

This repository contains the implementation of **Wasserstein Transfer Learning (WaTL)** algorithm, as described in the Wasserstein Transfer Learning paper.

## Repository Structure

```
Codes/
├── README.md                    # This file
├── Simulation/
│   ├── Simulation.R            # Main simulation script (Section 5)
│   └── SimulationFunc.R        # Helper functions for simulation
└── RealData/
    ├── RealData.R              # Main real data analysis script (Section 6)
    └── RealDataFunc.R          # Helper functions for real data analysis
```

---

## Algorithm Implementation

### Algorithm 1: Wasserstein Transfer Learning (WaTL)

#### Step 1: Weighted Auxiliary Estimator

**Simulation (Global Fréchet):**
- `SimulationFunc.R`, function `compute_f1_hat()`, lines 278-333
- Aggregates target ($k=0$) and all sources ($k=1,\ldots,K$)
- Weighted by sample sizes $n_k$

**Real Data (Local Fréchet):**
- `RealDataFunc.R`, function `compute_f1_hat()`, lines 279-341
- Uses local linear weights instead of global weights
- Aggregates target and all sources

#### Step 2: Bias Correction Using Target Data

**Code implementation:**
- `SimulationFunc.R` / `RealDataFunc.R`, function `compute_f_L2()`
- Uses gradient descent to minimize the objective
- Regularization parameter $\lambda$ selected via cross-validation

#### Step 3: Projection to Wasserstein Space

**Code implementation:**
- **Simulation:** Implicitly satisfied (quantile functions already monotone in data generation)
- **Real Data:** `RealDataFunc.R`, function `Project()`, lines 348-373
  - Uses OSQP solver to enforce monotonicity constraint
  - Called in `RealData.R`, line 171

---

## Running the Code

### Simulation

```bash
Rscript Simulation/Simulation.R <M> <n_t> <seed> <setting> <tau>
```

**Arguments:**
- `M`: Grid size for quantile functions (e.g., 100)
- `n_t`: Target sample size (200-800)
- `seed`: Random seed
- `setting`: Data generation setting (1 or 2)
- `tau`: Source sample multiplier (100 or 200)

**Example:**
```bash
Rscript Simulation/Simulation.R 100 200 42 1 100
```

### Real Data

```bash
Rscript RealData/RealData.R <seed> <race> <M> <rate> <gender>
```

**Arguments:**
- `seed`: Random seed
- `race`: Race index (1=Black, 2=White)
- `M`: Grid size for quantile functions (e.g., 100)
- `rate`: Source data sampling rate (0-1, typically 1.0)
- `gender`: Gender (0=Female, 1=Male)

**Example:**
```bash
Rscript RealData/RealData.R 42 1 100 1.0 0
```

## Citation

```
@article{zhang2025wasserstein,
  title={Wasserstein Transfer Learning},
  author={Zhang, Kaicheng and Zhang, Sinian and Zhou, Doudou and Zhou, Yidong},
  journal={arXiv preprint arXiv:2505.17404},
  year={2025}
}
```


