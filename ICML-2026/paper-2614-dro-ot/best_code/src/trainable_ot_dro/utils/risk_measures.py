import numpy as np
import cvxpy as cp
from scipy.stats import norm

def opt_gaussian_portfolio_CVaR(mu, Sigma, beta):
    n = len(mu)
    q = norm.ppf(1 - beta)
    kappa = norm.pdf(q) / (beta)

    # Factorize Sigma = Low.T @ Low using Cholesky (Sigma must be positive definite)
    Low = np.linalg.cholesky(Sigma)

    # Define variable w (portfolio weights)
    w = cp.Variable(n, nonneg=True)

    # Loss mean: -w^T mu
    loss_mean = - mu @ w
    # Loss std: norm(L @ w) is equivalent to sqrt(w^T Sigma w)
    loss_std = cp.norm(Low @ w)

    # CVaR objective: loss_mean + kappa * loss_std
    cvar_expr = loss_mean + kappa * loss_std

    # Constraints: sum of weights equals 1
    constraints = [cp.sum(w) == 1]

    # Define and solve the problem
    prob = cp.Problem(cp.Minimize(cvar_expr), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    return prob.value


def gaussian_portfolio_CVaR(mu, Sigma, z, beta):
    """
    Calculate the loss Conditional Value at Risk (CVaR) for a portfolio
    at a given confidence level beta.
    CVaR positive: Loss
    CVaR negative: Gain

    Parameters:
    - mu: (n,) numpy array of mean returns for each asset (expected returns).
    - Sigma: (n, n) numpy array of covariance matrix of returns.
    - z: (n,) numpy array of portfolio weights.
    - beta: scalar, confidence level for CVaR (e.g., 0.05 for 5% risk).

    Returns:
    - CVaR: scalar, the Conditional Value at Risk at risk level beta. The smaller the CVaR, the better.
    """
    # Portfolio mean loss (negative of expected return of the portfolio)
    port_mean = -np.dot(z, mu)  # Expected loss, not return

    # Portfolio standard deviation (volatility)
    port_std = np.sqrt(np.dot(z.T, np.dot(Sigma, z)))

    # Calculate the quantile for VaR using the inverse CDF (norm.ppf)
    VaR_quantile = norm.ppf(1-beta)

    # Calculate the PDF at the VaR quantile
    pdf_at_quantile = norm.pdf(VaR_quantile)

    # Calculate CVaR using the formula
    CVaR = port_mean + (port_std * pdf_at_quantile / beta)

    # CVaR is a loss measure, larger (more positive) is worse
    return CVaR


def gaussian_portfolio_VaR(mu, Sigma, z, beta):
    """
    Calculate the loss Value at Risk (VaR) for a portfolio at a given confidence
    level beta.
    VaR positive: Loss
    VaR negative: Gain

    Parameters:
    - mu: (n,) numpy array of mean returns for each asset (expected returns).
    - Sigma: (n, n) numpy array of covariance matrix of returns.
    - z: (n,) numpy array of portfolio weights.
    - beta: scalar, confidence level for VaR (e.g., 0.05 for 5% risk).

    Returns:
    - VaR: scalar, the Value at Risk at risk level beta. The smaller the VaR, the better.
    """
    # Portfolio mean loss (negative of expected return of the portfolio)
    port_mean = -np.dot(z, mu)  # Expected loss (negative because we're using losses)

    # Portfolio standard deviation (volatility)
    port_std = np.sqrt(np.dot(z.T, np.dot(Sigma, z)))

    # Calculate the quantile for VaR using the inverse CDF (norm.ppf) for the normal distribution
    VaR_quantile = norm.ppf(1-beta)

    # Calculate VaR using the formula: VaR = portfolio mean + quantile * portfolio std deviation
    VaR = port_mean + VaR_quantile * port_std

    return VaR
