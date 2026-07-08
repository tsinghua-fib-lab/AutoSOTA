import torch
import numpy as np
import torch
from scipy.stats import norm

def generate_simulation_data_torch_true(n=1000, qf_size=100, p=4, link='linear', theta=None, seed=0, rho=0.25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = np.zeros(p)
    cov = np.full((p, p), rho)
    np.fill_diagonal(cov, 1.0)
    Z = np.random.multivariate_normal(mean, cov, size=n)
    X = 2 * norm.cdf(Z) - 1
    X = torch.tensor(X, dtype=torch.float32)
    if theta is None:
        theta = torch.tensor([0.5, 0.1, 0, -0.5][:p], dtype=torch.float32)
        theta = theta / torch.norm(theta)
        if theta[0] < 0:
            theta = -theta
    else:
        theta = torch.tensor(theta, dtype=torch.float32)
        theta = theta / torch.norm(theta)
        if theta[0] < 0:
            theta = -theta
    s = X @ theta
    if link == 'linear':
        zeta = s
    elif link == 'quadratic':
        zeta = s ** 2
    elif link == 'exp':
        zeta = torch.exp(s)
    else:
        raise ValueError("Unknown link function")
    eta = torch.exp(s) / (1 + torch.exp(s))
    mu = torch.normal(zeta, 0.25)
    sigma = torch.distributions.Exponential(1/eta).sample()
    # True quantile function: use dense grid and normal ppf
    quantile_grid = np.linspace(0, 1, qf_size+2)[1:-1]  # avoid 0 and 1
    qf_obs = []
    for i in range(n):
        qf_i = norm.ppf(quantile_grid, loc=mu[i].item(), scale=sigma[i].item())
        qf_obs.append(torch.tensor(qf_i, dtype=torch.float32))
    Y = torch.stack(qf_obs)  # (n, qf_size)
    true_mu = zeta
    true_sigma = eta
    return X, Y, theta, true_mu, true_sigma