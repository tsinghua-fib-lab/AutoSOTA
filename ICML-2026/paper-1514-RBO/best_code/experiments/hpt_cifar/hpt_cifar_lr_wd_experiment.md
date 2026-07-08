# Hyperparameter Optimization on CIFAR-10

## Overview

We evaluate the performance of various Bayesian optimization methods on the task of hyperparameter tuning for a neural network classifier trained on CIFAR-10. This experiment focuses on optimizing two critical hyperparameters—learning rate and weight decay—while introducing controlled corruptions to simulate realistic training instabilities.

## Search Space

The optimization problem is defined over a two-dimensional continuous search space:

- **Learning rate**: $[10^{-5}, 10^{-1}]$ (log-scale)
- **Weight decay**: $[10^{-6}, 10^{-2}]$ (log-scale)

Both dimensions are log-transformed and normalized to facilitate efficient optimization. The objective is to maximize the validation accuracy of the neural network classifier.

## Experimental Configuration

The optimization procedure consists of 150 sequential evaluations, beginning with 4 initial points selected via Latin hypercube sampling. Each evaluation trains a neural network for 2 epochs with a batch size of 128 on the CIFAR-10 dataset, measuring validation accuracy as the objective value.

To simulate realistic machine learning scenarios where training runs may fail or produce unreliable results, we introduce controlled corruptions during the optimization process. The corruption mechanism follows a time-budget strategy with parameter $\alpha = 1/3$, whereby a fraction of evaluations return a crash value of -2.0 instead of the true accuracy. Corruptions are never applied to the initial points to ensure all methods begin with uncorrupted observations.

## Methods Compared

We compare five Bayesian optimization approaches, all using the Upper Confidence Bound (UCB) acquisition function:

1. **Gaussian Process (GP)**: Standard Gaussian process with squared exponential kernel, serving as the baseline method.

2. **Robust Covariance GP (RCGP)**: A Gaussian process with modified covariance structure designed to handle outliers and corrupted observations. The model uses a plateau width parameter determined by heuristics (value = 2.0) and a robustness parameter $c = 1.0$. Hyperparameters are fitted using weighted leave-one-out cross-validation (WLOO-CV).

3. **Student-t Process**: A Gaussian process variant that employs a Student-t likelihood with $\nu = 3.0$ degrees of freedom, providing inherent robustness to outliers through heavy-tailed distributions.

4. **Diagnostic GP**: An adaptive method that performs outlier detection before each model update. The diagnostic procedure runs on the initial 3 points and schedules detection every iteration thereafter, using a Student-t distribution with $\nu = 4.0$ and significance level $\alpha = 0.05$ to identify anomalous observations.

5. **Adaptive Anisotropic RCGP (A2RCGP)**: An extension of RCGP that maintains two separate Gaussian processes—an inner model for prediction and an outer model for uncertainty quantification. The inner model uses plateau width 2.0 and $c = 1.0$, while the outer model employs plateau width 1.5 and $c = 0.8$.

## Acquisition Function and Exploration Control

All methods employ the UCB acquisition function with dynamic exploration parameters. The exploration-exploitation trade-off is controlled by a theory-guided scheduler that adapts the UCB parameter $\beta_t$ according to:

$$\beta_t = \text{scale} \cdot \sqrt{2 \log(t \cdot d \cdot \pi^2 / 6\delta)} + \text{offset}$$

where $t$ is the iteration number, $d$ is the dimensionality, and $\delta$ is a confidence parameter. We use scale = 1.7 and offset = 2.0 for standard GP-based methods.

For RCGP-based methods (RCGP and A2RCGP), we apply an additional RCGP-specific scheduler that modifies the base theory-guided schedule to account for the model's robustness properties.

## Performance Metrics

We evaluate optimization performance using two complementary metrics:

- **Simple Regret**: The difference between the optimal objective value and the best observed value up to iteration $t$. This measures how quickly each method identifies high-performing configurations.

- **Cumulative Regret**: The sum of instantaneous regrets over all iterations, quantifying the total opportunity cost of the optimization process.

Both metrics are computed using the true (uncorrupted) objective values to fairly assess each method's ability to navigate the corrupted optimization landscape.

## Implementation Details

All models use L-BFGS optimization for hyperparameter fitting and standardize observations by subtracting the mean and dividing by the standard deviation. The experimental seed is fixed at 42 to ensure reproducibility. Results are averaged over the optimization trajectory and compared across all methods to assess robustness to corruptions and optimization efficiency.
