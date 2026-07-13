# Distributional Monte-Carlo Tree Search with Thompson Sampling

This repository contains the implementation of **CATSO** (Categorical Thompson Sampling with Optimistic Bonus) and **PATSO** (Particle Thompson Sampling with Optimistic Bonus) algorithms from the paper:

**"Distributional Monte-Carlo Tree Search with Thompson Sampling in Stochastic Environments"**  
*submission to NeurIPS 2025*

## Overview

This work introduces two novel distributional MCTS algorithms that combine Thompson Sampling with polynomial optimism bonuses to handle stochastic environments effectively. The algorithms achieve a convergence rate of O(n^{-1/2}) for simple regret.

### Key Features
- **CATSO**: Uses categorical distributions with fixed atoms to represent Q-value distributions
- **PATSO**: Uses particle-based distributions for more flexible Q-value representation
- **Thompson Sampling + Optimistic Bonus**: Combines posterior sampling with UCB-style exploration
- **Power Mean Backup**: Balances between mean and max operators for V-nodes

## Requirements

```bash
pip install numpy scipy matplotlib networkx joblib
```

## Repository Structure

```
.
├── mcts.py              # Main MCTS implementation with all algorithms
├── tree_env.py          # Synthetic tree environment
├── run_kd_cats.py       # Script to run experiments
├── plot_kd_cats.py      # Script to generate plots
└── README.md            # This file
```

## Algorithms Implemented

1. **CATSO** - Categorical Thompson Sampling with Optimistic Bonus
2. **PATSO** - Particle Thompson Sampling with Optimistic Bonus
3. **UCT** - Upper Confidence bounds applied to Trees
4. **Power-UCT** - UCT with power mean backup
5. **Fixed-Depth-MCTS** - MCTS with polynomial bonus
6. **MENTS** - Maximum Entropy Tree Search
7. **RENTS** - Relative Entropy Tree Search
8. **TENTS** - Tsallis Entropy Tree Search
9. **DENTS** - Decayed Entropy Tree Search
10. **BTS** - Boltzmann Tree Search
11. **DNG** - Bayesian MCTS with Normal-Gamma distributions
12. **VarDE** - Variance-based Decision Exploration

## Quick Start

### Running Experiments

To reproduce the experimental results from the paper:

```bash
python run_kd_cats.py
```

This will run experiments on synthetic trees with the following configurations:
- Branching factors (k): 16, 200, 14
- Depths (d): 1, 2, 3, 4
- Specific combinations: (16,1), (200,1), (14,3), (16,3), (16,4), (200,2)

### Generating Plots

After running the experiments:

```bash
python plot_kd_cats.py
```

This will generate a plot showing the value estimation error over the number of simulations for all algorithms.

## Algorithm Parameters

### CATSO/PATSO Specific Parameters
- `number_of_atoms`: Number of categorical atoms (default: 100 for CATSO)
- `exploration_coeff`: Coefficient C for optimistic bonus B(n,s,a) = C * n_s^(1/4) / n_{s,a}^(1/2)
- `alpha`: Power parameter for power mean backup (default: 10)

### General Parameters
- `gamma`: Discount factor (default: 0.99 for Atari, 1.0 for synthetic trees)
- `n_simulations`: Number of MCTS simulations (default: 1000)
- `tau`: Temperature parameter for entropy-based methods

## Key Implementation Details

### Thompson Sampling with Optimistic Bonus

Both CATSO and PATSO use the following action selection strategy:

```python
# Sample from Dirichlet distribution
L(s,a) ~ Dir(α(s,a))
φ(s,a) = support^T · L(s,a)

# Add optimistic bonus
B(n,s,a) = C · T_s(n)^(1/4) / T_{s,a}(n)^(1/2)

# Select action
a* = argmax_a { φ(s,a) + B(n,s,a) }
```

### Value Backup

V-nodes use power mean backup:
```python
V(s) = (Σ_a (T_{s,a}/T_s) · Q(s,a)^p)^(1/p)
```

### Distributional Updates

**CATSO**: Updates categorical distribution by finding nearest atom and incrementing its count
**PATSO**: Adds new particles or increments weights of existing particles

## Experimental Results

The algorithms are evaluated on:
1. **Synthetic Trees**: Stochastic tree environments with Gaussian rewards
2. **Atari Games**: 12 Atari 2600 games (see paper for details)


## Notes

- The synthetic tree environment includes stochastic transitions: 50% probability of reaching the intended child node, 50% probability of uniform transition to other children
- Rewards at leaf nodes are sampled from Gaussian distributions with standard deviation 0.5
- For reproducibility, random seeds can be set in the experiment scripts

## Troubleshooting

1. **Missing data files**: Ensure `run_kd_cats.py` has completed before running `plot_kd_cats.py`
2. **Memory issues**: Reduce `n_jobs=-1` in Parallel() calls to use fewer CPU cores
3. **File not found errors**: The scripts create a `logs/` directory structure - ensure write permissions

## License

This code is released under the MIT License.
