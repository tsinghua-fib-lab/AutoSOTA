# Balanced Active Inference

Code accompanying the paper _Balanced Active Inference_ accepted at NeurIPS 2025.

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# R (required for BalancedSampling package)
R --version
```

### Python Dependencies

```bash
pip install numpy pandas scikit-learn xgboost scipy matplotlib seaborn joblib rpy2
```

### R Dependencies

```R
install.packages("BalancedSampling")
```

## Repository Structure

```
balanced-active-inference/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_generation.py      # Synthetic data generation
│   ├── models.py                # Predictive and uncertainty models
│   ├── sampling_methods.py      # Sampling strategies
│   ├── variance_estimators.py   # Variance estimation
│   ├── experiment.py            # Experimental framework
│   └── visualization.py         # Plotting utilities
├── examples/
│   └── demo.ipynb              # Complete demonstration
└── results/                     # Output directory
```

## Quick Start

```python
from src.data_generation import generate_friedman_data
from src.models import train_predictive_models
from src.sampling_methods import balanced_active_sampling
from src.variance_estimators import estimate_variance

# Generate synthetic data
X, y = generate_friedman_data(n_samples=10000, n_features=10)

# Train predictive models
predictions, uncertainty = train_predictive_models(X, y)

# Perform balanced active sampling
selected_indices, estimates = balanced_active_sampling(
    predictions, uncertainty, budget=0.1
)

# Estimate population mean with confidence interval
mean_estimate, conf_interval = estimate_variance(
    y, predictions, selected_indices, confidence_level=0.95
)
```

## Dataset

This implementation uses the **Friedman synthetic dataset** [1], a widely-used benchmark for nonlinear regression:

```
Y = 10 * sin(π * X₁ * X₂) + 20 * (X₃ - 0.5)² + 10 * X₄ + 5 * X₅ + ε
```

where X₁, ..., X₅ ~ Uniform(0, 1) and ε ~ N(0, 0.09).

**Reference:**
[1] Friedman, J. H. (1991). Multivariate adaptive regression splines. *The Annals of Statistics*, 19(1), 1-67.

Additional features X₆, ..., X₁₀ are included as noise variables to test robustness.

## Experiments

Run the complete experimental pipeline:

```python
from src.experiment import run_simulation_experiment

results = run_simulation_experiment(
    n_trials=10000,
    budgets=np.arange(0.03, 0.45, 0.01),
    n_jobs=10
)
```

This generates evaluation metrics across different sampling budgets:
- Root Mean Squared Error (RMSE)
- Confidence Interval Width
- Coverage Rate

## Visualization

Generate publication-quality plots:

```python
from src.visualization import plot_comparison_results

plot_comparison_results(
    results,
    output_path='results/comparison.pdf'
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{balanced_active_inference2025,
  title={Balanced Active Inference},
  author={[Your Name]},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
