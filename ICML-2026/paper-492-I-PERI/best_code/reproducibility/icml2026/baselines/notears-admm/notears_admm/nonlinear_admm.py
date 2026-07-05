import numpy as np
import torch
import torch.nn as nn

from notears_admm.lbfgsb_scipy import LBFGSBScipy
from notears_admm.locally_connected import LocallyConnected
from notears_admm.trace_expm import trace_expm


class NonlinearModel(nn.Module):
    """
    Code modified from:
    https://github.com/xunzheng/notears/blob/master/notears/nonlinear.py
    """
    def __init__(self, dims, bias=True):
        super(NonlinearModel, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1 = nn.Linear(d, d * dims[1], bias=bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1.weight)
        return reg

    def flatten_weight(self):
        return torch.cat([torch.flatten(p) for p in self.parameters()]).view(-1)

    @torch.no_grad()
    def adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get B from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        B = torch.sqrt(A)  # [i, j]
        B = B.cpu().detach().numpy()  # [i, j]
        return B


def squared_loss(output, target, K):
    n = target.shape[0]
    loss = 0.5 / n / K * torch.sum((output - target) ** 2)
    return loss


def solve_local_subproblem(X, K, local_model, global_model, beta, rho):
    optimizer = LBFGSBScipy(local_model.parameters())
    model_param = next(local_model.parameters())
    X_torch = torch.from_numpy(X).to(dtype=model_param.dtype, device=model_param.device)
    beta_torch = torch.from_numpy(beta).to(dtype=model_param.dtype, device=model_param.device)
    def closure():
        optimizer.zero_grad()
        X_hat = local_model(X_torch)
        loss = squared_loss(X_hat, X_torch, K)
        penalty = beta_torch.dot(local_model.flatten_weight() - global_model.flatten_weight().detach()) \
                    + 0.5 * rho * torch.sum((local_model.flatten_weight() - global_model.flatten_weight().detach())**2)
        objective = loss + penalty
        objective.backward()
        return objective
    optimizer.step(closure)  # NOTE: updates model in-place


def solve_global_subproblem(local_models, global_model, lambda1, alpha, betas, rho1, rho2):
    K = len(local_models)
    optimizer = LBFGSBScipy(global_model.parameters())
    model_param = next(global_model.parameters())
    betas_torch = torch.from_numpy(betas).to(dtype=model_param.dtype, device=model_param.device)
    def closure():
        optimizer.zero_grad()
        l1_reg = lambda1 * global_model.fc1_l1_reg()
        h_val = global_model.h_func()
        penalty = 0
        for k in range(K):
            penalty += betas_torch[k].dot(local_models[k].flatten_weight().detach() - global_model.flatten_weight()) \
                        + 0.5 * rho2 * torch.sum((local_models[k].flatten_weight().detach() - global_model.flatten_weight())**2)
        objective = l1_reg + alpha * h_val + 0.5 * rho1 * h_val**2 + penalty
        objective.backward()
        return objective
    optimizer.step(closure)  # NOTE: updates model in-place


def update_admm_params(alpha, betas, rho1, rho2, local_models,
                       global_model, h, rho_max):
    K = len(local_models)
    alpha += rho1 * h
    for k in range(K):
        betas[k] += rho2 * (local_models[k].flatten_weight().cpu().detach().numpy()
                                - global_model.flatten_weight().cpu().detach().numpy())
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
    return consensus_distance


def notears_nonlinear_admm(Xs: np.ndarray,
                           lambda1: float = 0.001,
                           lambda2: float = 0.01,
                           max_iter: int = 200,
                           rho_max: float = 1e+16,
                           verbose: bool = False):
    """Solve ADMM problem using augmented Lagrangian.
    Args:
        Xs (np.ndarray): [K, n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        lambda2 (float): l2 penalty parameter
        max_iter (int): max num of dual ascent steps
        rho_max (float): exit if rho1 >= rho_max and rho2 >= rho_max
        verbose (bool): Whether to print messages during optimization

    Returns:
        B_est (np.ndarray): [d, d] estimated weights
    """
    if not np.issubdtype(Xs.dtype, np.floating):
        Xs = Xs.astype(np.float64)

    K, n, d = Xs.shape
    # Initialize local and global models
    local_models = [NonlinearModel(dims=[d, 10, 1], bias=True) for _ in range(K)]
    global_model = NonlinearModel(dims=[d, 10, 1], bias=True)
    # Initialize ADMM parameters
    rho1, rho2, alpha, h = 0.1, 0.1, 0.0, np.inf
    alpha, betas = 0.0, np.zeros((K,) + global_model.flatten_weight().shape)

    # ADMM
    for t in range(1, max_iter + 1):
        # Solve local subproblem for each client
        for k in range(K):
            solve_local_subproblem(Xs[k], K, local_models[k], global_model, betas[k], rho2)
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
    n, d, s0, graph_type, sem_type = 64, 20, 20, 'ER', 'mlp'
    B_bin_true = utils.simulate_dag(d, s0, graph_type)
    X = utils.simulate_nonlinear_sem(B_bin_true, K * n, sem_type)
    Xs = X.reshape(K, n, d)

    # Run NOTEARS-MLP-ADMM
    utils.set_random_seed(1)
    B_est = notears_nonlinear_admm(Xs, lambda1=0.001, lambda2=0.01, verbose=True)
    B_processed = postprocess(B_est, threshold=0.3)
    acc = utils.count_accuracy(B_processed, B_bin_true)
    print(acc)


if __name__ == '__main__':
    main()