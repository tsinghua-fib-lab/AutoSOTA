from typing import Optional
import copy
import torch
from torch import Tensor
from torch import nn

from gpytorch.kernels import Kernel
from utils.metrics import gaussian_nll
from linear_operator.utils.cholesky import psd_safe_cholesky

from likelihood.gaussian import GaussianLikelihood


__all__ = ["GP", "ind_exact_gp"]


class GP(nn.Module):
    """
    Exact Single Output GP, ONLY for Gaussian likelihood.
    """
    def __init__(
        self, kernel: Kernel, train_X: Tensor, train_Y: Tensor,
        sigma_joint: bool = False, sigma_init: float = 0.5,
        jitter=1e-6
    ):
        # train_X: [n_train, D_X]; train_Y: [n_train, 1]
        # we assume zero mean!
        super(GP, self).__init__()
        assert train_Y.ndim == 2
        assert train_Y.size(-1) == 1
        assert train_Y.size(0) == train_X.size(0)
        self.kernel = kernel
        self.register_buffer(
            "train_X", torch.as_tensor(train_X, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "train_Y", torch.as_tensor(train_Y, dtype=torch.get_default_dtype())
        )
        self.lik_model = GaussianLikelihood(sigma_joint, sigma_init)
        self.jitter = jitter

    @torch.no_grad()
    def posterior(self, test_X: Tensor, diag=True, noiseless: bool = False):
        # predictive f if noiseless=True, else predictive y
        # test_X: [n_test, D_X]
        self.eval()
        Kxx = self.kernel.forward(self.train_X, self.train_X)  # [n_train, n_train]
        noisy_Kxx = Kxx + self.lik_model.sigma.square() * torch.eye(Kxx.size(-1), device=Kxx.device)
        Kxxs = self.kernel.forward(self.train_X, test_X)  # [n_train, n_test]
        noisy_Lxx = psd_safe_cholesky(noisy_Kxx + self.jitter * torch.eye(Kxx.size(-1), device=noisy_Kxx.device))
        noisy_Kxx_inv = torch.cholesky_solve(
            torch.eye(Kxx.size(-1), dtype=Kxx.dtype, device=noisy_Lxx.device), noisy_Lxx
        )

        pred_mean = (Kxxs.mT @ noisy_Kxx_inv @ self.train_Y).squeeze(-1) # [n_test]

        if diag:
            Kxsxs = self.kernel.forward(test_X, test_X, diag=True)  # [n_test]
            pred_cov = torch.einsum('ij,jk,ki->i', Kxxs.mT, noisy_Kxx_inv, Kxxs)
            # assert torch.allclose(pred_cov, torch.diagonal(Kxxs.mT @ noisy_Kxx_inv @ Kxxs, dim1=-2, dim2=-1))
            pred_cov = Kxsxs - pred_cov  # [n_test]
            if not noiseless:
                pred_cov = pred_cov + self.lik_model.sigma.square()
        else:
            raise NotImplementedError

        return pred_mean, pred_cov

    def log_evidence(self):
        Kxx = self.kernel.forward(self.train_X, self.train_X)  # [n_train, n_train]
        noisy_Kxx = Kxx + self.lik_model.sigma.square() * torch.eye(Kxx.size(-1), device=Kxx.device)  # [n_train, n_train]
        noisy_Lxx = psd_safe_cholesky(noisy_Kxx + self.jitter * torch.eye(noisy_Kxx.size(-1), device=noisy_Kxx.device))  # [n_train, n_train]

        dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(noisy_Kxx.size(-1), device=noisy_Lxx.device), scale_tril=noisy_Lxx
        )

        return dist.log_prob(self.train_Y.squeeze(-1))

    def train_gp(self, optimizer: torch.optim.Optimizer, epochs: int, max_norm: Optional[float] = None, print_epochs: int = 10):
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = - self.log_evidence()
            loss.backward()

            if max_norm is not None:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

                if (epoch + 1) % print_epochs == 0 and total_grad_norm.item() > max_norm:
                    print(
                        f"Gradient norm {total_grad_norm.item():.3f} exceeds the threshold {max_norm:.3f}, clipping applied."
                    )

            optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1} / {epochs}； Loss: {loss.item():.6f}')


class ind_exact_gp(nn.Module):
    """
    Independent exact GP for every output
    # TODO: move onto GPU
    """
    def __init__(
        self, kernel: Kernel, train_X: Tensor, train_Y: Tensor, train_mask: Tensor,
        sigma_joint: bool = False, sigma_init: float = 0.5, jitter=1e-6
    ):
        # kernel: NO batch_shape!
        # train_X: [n_train, D_X];
        # train_Y, train_mask: [n_train, P]
        # train_mask: boolean mask, 1/True for observed data, 0/False for missing data
        super(ind_exact_gp, self).__init__()
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_mask = train_mask

        self.GP_list = nn.ModuleList()
        self.setup_ind_gps(
            kernel=kernel,
            sigma_joint=sigma_joint,
            sigma_init=sigma_init,
            jitter=jitter
        )

    def setup_ind_gps(
        self, kernel: Kernel, sigma_joint: bool, sigma_init: float, jitter=1e-6
    ):
        P = self.train_Y.size(-1)

        for p in range(P):
            curr_mask = self.train_mask[:, p]
            curr_train_X = self.train_X[curr_mask]

            if curr_train_X.size(0) < 3:
                print(f"for output {p}, only {curr_train_X.size(0)} training examples!")
                raise NotImplementedError

            curr_train_Y = self.train_Y[:, p][curr_mask].unsqueeze(-1)
            curr_gp = GP(
                kernel=copy.deepcopy(kernel),
                train_X=curr_train_X,
                train_Y=curr_train_Y,
                sigma_joint=sigma_joint,
                sigma_init=sigma_init,
                jitter=jitter
            )
            self.GP_list.append(curr_gp)

    def train_ind_exact_gp(
        self, optimizer: torch.optim.Optimizer, epochs: int, method: str = 'approach1', max_norm: Optional[float] = None, device: str = 'cpu', print_epochs: int = 10
    ):
        self.train()
        self.to(device)
        if method == "approach1":
            for i, gp in enumerate(self.GP_list):
                print(f"Training GP for output {i + 1} ...")
                gp.train_gp(optimizer=optimizer, epochs=epochs, max_norm=max_norm, print_epochs=print_epochs)

        elif method == "approach2":
            for epoch in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                # loss = - self.GP_list[0].log_evidence()
                # for gp in self.GP_list[1:]:
                #     loss = loss - gp.log_evidence()
                loss = - sum(gp.log_evidence() for gp in self.GP_list)
                loss.backward()

                if max_norm is not None:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

                    if (epoch + 1) % print_epochs == 0 and total_grad_norm.item() > max_norm:
                        print(
                            f"Gradient norm {total_grad_norm.item():.3f} exceeds the threshold {max_norm:.3f}, clipping applied."
                        )

                optimizer.step()

                if (epoch + 1) % print_epochs == 0:
                    print(f'Epoch {epoch + 1} / {epochs}； Loss: {loss.item():.6f}')
        else:
            raise NotImplementedError(f"Unknown training method: {method}")

    @torch.no_grad()
    def predict(self, test_X: Tensor, output_idx: list=None, diag=True, noiseless: bool = False):
        # test_X: [n_test, D_X]
        # output_idx: [P], a list of indices for outputs to predict, if None, predict all outputs
        self.eval()
        assert diag == True
        pred_mean, pred_cov = [], []

        if output_idx is None:
            output_idx = list(range(len(self.GP_list)))

        for idx in output_idx:
            curr_gp = self.GP_list[idx]
            test_X = test_X.to(curr_gp.train_X.device)
            curr_pred_mean, curr_pred_cov = curr_gp.posterior(test_X, diag=diag, noiseless=noiseless)
            pred_mean.append(curr_pred_mean)
            pred_cov.append(curr_pred_cov)

        pred_mean = torch.stack(pred_mean, dim=-1) # [n_test, P]
        pred_cov = torch.stack(pred_cov, dim=-1)  # [n_test, P]

        return pred_mean, pred_cov

    @torch.no_grad()
    def predict_ind_gp(
        self, data_dict, noiseless: bool = False, num_plot_points: Optional[int] = 2000, device: str = 'cpu'
    ):
        # on device
        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )

        # predict on all_X for metrics
        pred_means, pred_vars = self.predict(
            all_X, output_idx=None, diag=True, noiseless=noiseless
        )  # [N, P]

        pick_train_Y = train_Y[train_mask.bool()].view(-1)  # [<N*P]
        pick_train_pred_means = pred_means[train_mask.bool()].view(-1)  # [<N*P]
        pick_train_pred_vars = pred_vars[train_mask.bool()].view(-1)  # [<N*P]

        pick_test_Y = test_Y[test_mask.bool()].view(-1)  # [<N*P]
        pick_test_pred_means = pred_means[test_mask.bool()].view(-1)  # [<N*P]
        pick_test_pred_vars = pred_vars[test_mask.bool()].view(-1)  # [<N*P]

        train_se = (pick_train_Y - pick_train_pred_means).square()  # [N_train]
        test_se = (pick_test_Y - pick_test_pred_means).square()  # [N_test]

        train_nll = gaussian_nll(pick_train_Y, pick_train_pred_means, pick_train_pred_vars)  # [N_train]
        test_nll = gaussian_nll(pick_test_Y, pick_test_pred_means, pick_test_pred_vars)  # [N_test]

        metric_dict = {
            "train_mse": train_se.mean(dim=(-1)),  # average over N_train
            "test_mse": test_se.mean(dim=(-1)),  # average over N_test
            "train_nll": train_nll.mean(dim=(-1)),  # average over N_train
            "test_nll": test_nll.mean(dim=(-1)),  # average over N_test
        }

        # prediction on dataset input points
        pred_dict = {
            "all_X": all_X,  # [N, D_X]
            "pred_means": pred_means,  # [N, P]
            "pred_vars": pred_vars,  # [N, P]
        }

        plot_pred_dict = None

        if num_plot_points is None:
            return metric_dict, pred_dict, plot_pred_dict

        # predict on denser input X for plotting
        if all_X.size(-1) == 1:
            X_min, X_max = all_X.min().item(), all_X.max().item()
            denser_X = torch.linspace(X_min, X_max, num_plot_points, dtype=all_X.dtype, device=device).unsqueeze(-1)  # [n_plot, 1]

            plot_pred_means, plot_pred_vars = self.predict(
                denser_X, output_idx=None, diag=True, noiseless=True
            )  # [n_plot, P], we want noiseless for plotting

            plot_pred_dict = {
                "denser_X": denser_X,  # [n_plot, D_X]
                "plot_pred_means": plot_pred_means,  # [n_plot, P]
                "plot_pred_vars": plot_pred_vars,  # [n_plot, P]
            }

        return metric_dict, pred_dict, plot_pred_dict
