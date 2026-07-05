import numpy as np
import torch
import torch.nn as nn

from notears_admm.lbfgsb_scipy import LBFGSBScipy
from notears_admm.trace_expm import trace_expm


class LinearModel(nn.Module):
    def __init__(self, d):
        super(LinearModel, self).__init__()
        self.B = torch.nn.Parameter(torch.zeros(d, d))

    def forward(self, x):  # [n, d] -> [n, d]
        return x @ self.B

    def l1_reg(self):
        """Take l1 norm of linear weight"""
        reg = torch.norm(self.B, p=1)
        return reg

    def h_func(self):
        d = self.B.shape[0]
        h = trace_expm(self.B * self.B) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + self.B * self.B / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    @torch.no_grad()
    def adj(self) -> np.ndarray:  # [d, d] -> [d, d]
        return self.B.cpu().detach().numpy()


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def solve_local_subproblem(cov_emp, local_model, global_model, beta, rho):
    d = cov_emp.shape[0]
    sol = np.linalg.inv(cov_emp + rho * np.eye(d)) @ (rho * global_model.adj() - beta + cov_emp)
    local_model.B = torch.nn.Parameter(torch.from_numpy(sol))  # NOTE: updates model in-place


def solve_global_subproblem(local_models, global_model, lambda1, alpha, betas, rho1, rho2):
    K = len(local_models)
    optimizer = LBFGSBScipy(global_model.parameters())
    betas_torch = torch.from_numpy(betas)
    def closure():
        optimizer.zero_grad()
        l1_reg = lambda1 * global_model.l1_reg()
        h_val = global_model.h_func()
        penalty = 0
        for k in range(K):
            penalty += torch.sum(betas_torch[k] * (local_models[k].B.detach() - global_model.B)) \
                        + 0.5 * rho2 * torch.sum((local_models[k].B.detach() - global_model.B)**2)
        objective = l1_reg + alpha * h_val + 0.5 * rho1 * h_val**2 + penalty
        objective.backward()
        return objective
    optimizer.step(closure)  # NOTE: updates model in-place


def update_admm_params(alpha, betas, rho1, rho2, local_models,
                       global_model, h, rho_max):
    K = len(local_models)
    alpha += rho1 * h
    for k in range(K):
        betas[k] += rho2 * (local_models[k].adj() - global_model.adj())
    if rho1 < rho_max:
        rho1 *= 1.75
    if rho2 < rho_max:
        rho2 *= 1.25
    return alpha, betas, rho1, rho2


def compute_consensus_distance(local_models, global_model):
    K = len(local_models)
    consensus_distance = 0
    for k in range(K):
        consensus_distance += ((local_models[k].adj() - global_model.adj())**2).sum()
    consensus_distance /= K
    return consensus_distance


def compute_empirical_covs(Xs):
    K, n, d = Xs.shape
    covs_emp = []
    for k in range(K):
        cov_emp = Xs[k].T @ Xs[k] / n / K
        covs_emp.append(cov_emp)
    return np.array(covs_emp)


def notears_linear_admm(Xs: np.ndarray,
                        lambda1: float = 0.01,
                        max_iter: int = 200,
                        rho_max: float = 1e+16,
                        verbose: bool = False):
    """Solve ADMM problem using augmented Lagrangian.
    Args:
        Xs (np.ndarray): [K, n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        max_iter (int): max num of dual ascent steps
        rho_max (float): exit if rho1 >= rho_max and rho2 >= rho_max
        verbose (bool): Whether to print messages during optimization

    Returns:
        B_est (np.ndarray): [d, d] estimated weights
    """
    K, n, d = Xs.shape
    # Initialize local and global models
    local_models = [LinearModel(d) for _ in range(K)]
    global_model = LinearModel(d)
    # Initialize ADMM parameters
    rho1, rho2, alpha, h = 0.001, 0.001, 0.0, np.inf
    alpha, betas = 0.0, np.zeros((K, d, d))
    # Standardize data and compute empirical covariances
    Xs = Xs - np.mean(Xs, axis=1, keepdims=True)
    covs_emp = compute_empirical_covs(Xs)

    # ADMM
    for t in range(1, max_iter + 1):
        # Solve local subproblem for each client
        for k in range(K):
            solve_local_subproblem(covs_emp[k], local_models[k], global_model, betas[k], rho2)
        # Solve global subproblem
        solve_global_subproblem(local_models, global_model, lambda1, alpha, betas, rho1, rho2)
        # Obtain useful value
        with torch.no_grad():
            h = global_model.h_func().item()
        # Printing statements
        if verbose:
            consensus_distance = compute_consensus_distance(local_models, global_model)
            print("----- Iteration {} -----".format(t))
            print("rho1 {:.3E}, rho2 {:.3E}, alpha {:.3E}".format(rho1, rho2, alpha))
            print("h {:.3E}, consensus_distance {:.3E}".format(h, consensus_distance))
        # Update ADMM parameters
        alpha, betas, rho1, rho2 = update_admm_params(alpha, betas, rho1, rho2, local_models,
                                                      global_model, h, rho_max)
        # Terminate the optimization
        if rho1 >= rho_max and rho2 >= rho_max:
            break

    B_est = global_model.adj()
    return B_est


def main():
    from notears_admm import utils
    from notears_admm.postprocess import postprocess

    # Configuration of torch
    torch.set_default_dtype(torch.double)

    # Generate data
    utils.set_random_seed(1)
    K = 8
    n, d, s0, graph_type, sem_type = 32, 20, 20, 'ER', 'gauss'
    B_bin_true = utils.simulate_dag(d, s0, graph_type)
    B_true = utils.simulate_parameter(B_bin_true)
    X = utils.simulate_linear_sem(B_true, K * n, sem_type)
    Xs = X.reshape(K, n, d)

    # Run NOTEARS-MLP-ADMM
    utils.set_random_seed(1)
    B_est = notears_linear_admm(Xs, lambda1=0.01, verbose=True)
    B_processed = postprocess(B_est, threshold=0.3)
    acc = utils.count_accuracy(B_processed, B_bin_true)
    print(acc)


if __name__ == '__main__':
    main()