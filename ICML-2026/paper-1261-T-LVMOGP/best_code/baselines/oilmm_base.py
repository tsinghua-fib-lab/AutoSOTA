from typing import Optional

import math
import torch
from torch import Tensor, BoolTensor, LongTensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils.parametrizations import orthogonal

from gpytorch.utils.transforms import inv_softplus

from linear_operator.utils.cholesky import psd_safe_cholesky

from utils.metrics import gaussian_nll
from utils.helpers import wrap_func_by_batch

from models.building_blocks.gp_modules import svgp_base


__all__ = ['oilmm_base']


class oilmm_base(nn.Module):
    """
    Base class for OILMM models, with SVGP as latent processes.
    Only support Gaussian likelihood.

    Implementation of "Scalable Exact Inference in Multi-Output Gaussian Processes" (ICML 2020)

    Comments:
        * Cannot handle mini-batch training across outputs.
    """
    def __init__(
        self, P: int, latent_svgps: svgp_base, init_sigma: float = 0.5, sigma_joint: bool = True,
        jitter: float = 1e-6,
    ):
        super(oilmm_base, self).__init__()
        self.num_outputs = P
        self.latent_svgps = latent_svgps
        self.num_latent = latent_svgps.num_outputs  # number of independent latent processes
        self.batch_shape = tuple(latent_svgps.Z.inducing_points.shape[:-3])
        self.jitter = jitter

        self._setup_orthogonal_U()
        self._setup_diag_S()
        self._setup_diag_D()
        self._setup_sigma(init_sigma, sigma_joint)

    def _setup_orthogonal_U(self):
        # Orthogonal part of the basis H = U S^{1/2}, P by num_latent matrix
        assert self.num_outputs >= self.num_latent
        self.U = nn.Parameter(
            torch.randn(*self.batch_shape, self.num_outputs, self.num_latent, dtype=torch.get_default_dtype()),
            requires_grad=True
        )  # [*batch_shape, P, num_latent]
        orthogonal(self, 'U')

    def _setup_diag_S(self):
        # Positive, diagonal part of the basis H = U S^{1/2}, num_latent dimensional vector
        self.raw_diag_S = nn.Parameter(
            torch.ones(*self.batch_shape, self.num_latent, dtype=torch.get_default_dtype()),
            requires_grad=True
        )  # [*batch_shape, num_latent]

    @property
    def diag_S(self):
        return nn.functional.softplus(self.raw_diag_S) # [*batch_shape, num_latent]

    def _setup_diag_D(self):
        # Positive, diag observation noise added to the latent GPs
        self.raw_diag_D = nn.Parameter(
            torch.ones(*self.batch_shape, self.num_latent, dtype=torch.get_default_dtype()),
            requires_grad=True
        )  # [*batch_shape, num_latent]

    @property
    def diag_D(self):
        return nn.functional.softplus(self.raw_diag_D)  # [*batch_shape, num_latent]

    def _setup_sigma(self, init_sigma: float = 1., sigma_joint: bool = True):
        # Positive scalar, part of the observation noise.
        if sigma_joint:
            self.raw_sigma = nn.Parameter(
                inv_softplus(
                    init_sigma * torch.ones(size=self.batch_shape, dtype=torch.get_default_dtype()),
                ),  # [...]
                requires_grad=True
            )
        else:
            self.register_buffer(
                "raw_sigma",
                inv_softplus(
                    init_sigma * torch.ones(size=self.batch_shape, dtype=torch.get_default_dtype()),
                ),  # [...]
            )

    @property
    def sigma(self):
        return nn.functional.softplus(self.raw_sigma)

    def loss(self, x: Tensor, y: Tensor, m: BoolTensor, coeff_exp_log_lik: float, beta_1=1., beta_2=1., average_loss=False):
        """
        mini-batch approximation to the overall loss.
        x: [..., b, D_X], i.e. xs are shared across outputs
        y: [..., b, P], P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicates missing
        coeff_exp_log_lik: to scale the exp_log_lik term in latent SVGP to approximate the full latent data set, should be N
        beta_1: weight for KL term in latent SVGPs
        beta_2: weight for regularization term

        Overall loss is composed of 2 terms:
            (1) sum of ELBOs of latent SVGPs with projected observations
            (2) regularization term
        """
        x_proj, y_proj, noise_proj, reg = self.project(x, y, m)

        # (1) mini-batch elbo term from latent SVGPs
        latent_qf_means, latent_qf_covs = self.latent_svgps.variational_f(x_proj)  # [..., self.num_latent, b]
        latent_y, latent_noise = y_proj.mT, noise_proj.mT  # [..., num_latent, b]

        _exp_log_lik_term1 = - ((latent_qf_means - latent_y).square() + latent_qf_covs) / (2 * latent_noise)  # [..., num_latent, b]
        _exp_log_lik_term2 = -0.5 * (math.log(2*math.pi) + torch.log(latent_noise))  # [..., num_latent, b]
        _exp_log_lik = _exp_log_lik_term1 + _exp_log_lik_term2  # [..., num_latent, b]
        exp_log_lik = _exp_log_lik.sum(dim=(-2)).mean(dim=(-1))  # [...], sum over num_latent, mean over b
        latent_KL_qU_pU = (self.latent_svgps.KL_qU_pU).sum(dim=(-1))  # [...]
        elbo = coeff_exp_log_lik * exp_log_lik - beta_1 * latent_KL_qU_pU  # [...]

        # (2) add regularisation term (NOTE reg is averaged over batch size)
        loss = (- elbo + beta_2 * coeff_exp_log_lik * reg).sum()  # [...] -> scalar

        if average_loss:
            loss = loss / coeff_exp_log_lik

        return loss

    def project(self, x: Tensor, y: Tensor, m: BoolTensor):
        """
        x: [..., b, D_X], i.e. xs are shared across outputs
        y: [..., b, P], P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicates missing

        return:
            x_proj: [..., b, D_X], permuted x
            y_proj: [..., b, num_latent], projected y
            noise_proj: [..., b, num_latent], projected noise variance, \Sigma_T in the paper
            reg: [...] sum over patterns in this batch, and then average over batch size b
        """
        index = (0, ) * len(self.batch_shape) + (slice(None), slice(None))  # [0,..., 0, :, :]
        m_example = m[index]  # [b, P]
        assert (m == m_example).all(), "Different missingness across batch_shape not supported."
        batch_size_b = m_example.size(0)  # b

        # Extract patterns.
        unique_patterns, inverse_indices = torch.unique(m_example, return_inverse=True, dim=0)  # [num_patterns, P], [b]

        x_proj_list, y_proj_list, noise_proj_list = [], [], []
        sum_pattern_reg = 0.  # sum of pattern-dependent regularization terms

        for i, unique_pattern in enumerate(unique_patterns):
            pattern_mask = (inverse_indices == i)  # [b], bool
            b_prime = int(pattern_mask.sum().item())  # scalar, mini-batch size for the current pattern
            x_proj = x[..., pattern_mask, :]  # [..., b', D_X]
            y_select = y[..., pattern_mask, :]  # [..., b', P]
            y_proj, noise_proj, pattern_reg = self._project_pattern(y_select, unique_pattern)
            exp_noise_proj= noise_proj.unsqueeze(-2).expand(*self.batch_shape, b_prime, self.num_latent)  # [..., b', num_latent]

            x_proj_list.append(x_proj)
            y_proj_list.append(y_proj)
            noise_proj_list.append(exp_noise_proj)
            sum_pattern_reg = sum_pattern_reg + pattern_reg

        x_proj = torch.cat(x_proj_list, dim=-2)  # [..., b, D_X]
        y_proj = torch.cat(y_proj_list, dim=-2)  # [..., b, num_latent]
        noise_proj = torch.cat(noise_proj_list, dim=-2)  # [..., b, num_latent]

        reg = sum_pattern_reg  # [...]
        # add pattern-independent regularisation terms
        term1 = self.diag_S.log().sum(dim=-1) * batch_size_b / 2  # [...]
        reg = reg + term1  # [...]
        reg = reg / batch_size_b  # average over b

        return x_proj, y_proj, noise_proj, reg

    def _project_pattern(self, y: Tensor, m_template: BoolTensor):
        """
        A batch of data with the same missing pattern.
        y: [..., b', P]; m_template: [P], where 0 indicates missing, this template is shared across all batches.

        return:
            y_proj: [..., b', num_latent], projected y
            noise_proj: [..., num_latent], projected noise variance, shared across this batch b'
            pattern_reg: [...] scalar, sum of pattern-dependent regularization terms
        """
        # prepare
        assert m_template.ndim == 1
        m_template = m_template.to(torch.bool)  # for PyTorch advanced indexing
        Po = m_template.sum(dim=-1).item()  # scalar
        b_prime = y.size(-2) # batch size b'

        Uo = self.U[..., m_template, :]  # [..., P_o, num_latent]
        S = self.diag_S    # [..., num_latent]
        S_half = S.sqrt()  # [..., num_latent]
        D = self.diag_D    # [..., num_latent]

        Ho = Uo * S_half.unsqueeze(-2)  # [..., P_o, num_latent]
        Uo_T_Uo = Uo.mT @ Uo  # [..., num_latent, num_latent]
        Uo_T_Uo_plus_jitter = Uo_T_Uo + self.jitter * torch.eye(self.num_latent, device=Uo_T_Uo.device, dtype=Uo_T_Uo.dtype)
        # L_Uo_T_Uo_plus_jitter = torch.linalg.cholesky(Uo_T_Uo_plus_jitter)
        L_Uo_T_Uo_plus_jitter = psd_safe_cholesky(Uo_T_Uo_plus_jitter)
        Uo_T_Uo_inv = torch.cholesky_inverse(L_Uo_T_Uo_plus_jitter)  # [..., num_latent, num_latent]
        # Uo_T_Uo_inv = torch.linalg.inv(Uo_T_Uo_plus_jitter)  # [..., num_latent, num_latent]
        diag_Uo_T_Uo_inv = torch.diagonal(Uo_T_Uo_inv, dim1=-2, dim2=-1)  # [..., num_latent]
        Uo_pi = Uo_T_Uo_inv @ Uo.mT  # pseudo-inverse, [..., num_latent, P_o]
        To = Uo_pi / S_half.unsqueeze(-1)  # [..., num_latent, P_o]

        # projected y
        yo = y.mT[..., m_template, :].mT  # [..., b', P_o]
        y_proj = yo @ To.mT  # [..., b', num_latent]

        # projected noise variance
        noise_proj = self.sigma.square() * diag_Uo_T_Uo_inv / S + D  # [..., num_latent]

        # pattern-dependent regularization term
        log_det_diag_Uo_T_Uo_inv = diag_Uo_T_Uo_inv.log().sum(dim=-1)  # [...]
        pattern_reg = -0.5 * b_prime * log_det_diag_Uo_T_Uo_inv  # [...]

        # weighted sum of squared error term
        recon_weighted_sse = (yo - (y_proj @ Ho.mT)).square().sum(dim=(-2, -1)) / self.sigma.square() # [...]
        pattern_reg += 0.5 * recon_weighted_sse  # [...]

        # const term
        const_term = (b_prime * (Po - self.num_latent) / 2) * torch.log(2 * torch.pi * self.sigma.square())  # [...]
        pattern_reg += const_term

        return y_proj, noise_proj, pattern_reg

    def train_oilmm(
        self, train_dataloader: DataLoader, optimizer: Optimizer, epochs: int,
        beta_1=1., beta_2=1., coeff_exp_log_lik: Optional[float] = None, max_norm: Optional[float] = None, device="cpu", print_epochs=10
    ):
        self.to(device)
        self.train()
        if coeff_exp_log_lik is None:
            coeff_exp_log_lik = len(train_dataloader.dataset)
        perm = None

        for epoch in range(epochs):
            for batch_X, batch_Y, batch_m in train_dataloader:  # [b, ..., D_X/P]
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)
                batch_m = batch_m.to(device)

                # re-arrange dims
                if perm is None:
                    ndim = batch_X.ndim
                    perm = list(range(1, ndim - 1)) + [0, ndim - 1]

                batch_X = batch_X.permute(*perm)  # [..., b, D_X]
                batch_Y = batch_Y.permute(*perm)  # [..., b, P]
                batch_m = batch_m.permute(*perm)  # [..., b, P]

                optimizer.zero_grad(set_to_none=True)
                loss = self.loss(batch_X, batch_Y, batch_m, coeff_exp_log_lik, beta_1=beta_1, beta_2=beta_2)
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

    @torch.no_grad()
    def predict(self, x_star: Tensor, output_idx: Optional[LongTensor] = None, device="cpu", noiseless: bool = False):
        """
        x_star: [..., n_test, D_X]
        Get predictive mean and var for all outputs on x_star.
        """
        self.to(device)
        self.eval()
        x_star = x_star.to(device)

        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device)

        # latent SVGP predictions
        latent_qf_mean, latent_qf_vars = self.latent_svgps.variational_f(x_star)  # [..., num_latent, n_test]

        # build projection matrix
        H_sel = torch.index_select(self.U, dim=-2, index=output_idx) * self.diag_S.sqrt().unsqueeze(-2)  # [..., P_sel, num_latent]

        # project back to observation space
        _qf_mean = H_sel @ latent_qf_mean  # [..., P_sel, n_test]
        _qf_vars = (H_sel * H_sel) @ latent_qf_vars  # [..., P_sel, n_test]

        qf_mean, qf_vars = _qf_mean.mT, _qf_vars.mT  # [..., n_test, P_sel]

        if noiseless:
            return qf_mean, qf_vars  # [..., n_test, P_sel]
        else:
            # construct (diag of the) likelihood covariance matrix
            diag_Sigma = (H_sel.square() * self.diag_D.unsqueeze(-2)).sum(dim=(-1)) + self.sigma.square()  # [..., P_sel]
            qy_vars = qf_vars + diag_Sigma.unsqueeze(-2)  # [..., n_test, P_sel]
            return qf_mean, qy_vars

    @torch.no_grad()
    def predict_by_batch(
            self, x_star: Tensor, output_idx: Optional[LongTensor] = None, device="cpu", noiseless: bool = False,
            input_batch_size: int = 64, output_batch_size: int = 32
    ):
        """
        For large scale dataset (with large number of inputs and outputs), split the prediction into mini-batches.
        x_star: [..., n_test, D_X]
        """
        self.eval()
        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device)  # [P]

        qy_means, qy_vars = wrap_func_by_batch(
            model=self, func_args={"x_star": x_star, "output_idx": output_idx, "noiseless": noiseless},
            name="oilmm_base", input_batch_size=input_batch_size, output_batch_size=output_batch_size, device=device
        )

        return qy_means, qy_vars  # [..., n_test, P]

    @torch.no_grad()
    def predict_oilmm_gaussian(
        self, data_dict, noiseless: bool = False, num_plot_points: Optional[int] = 2000, device: str = "cpu"
    ):
        self.eval()
        # on device
        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )

        # predict on all_X for metrics
        pred_means, pred_vars = self.predict_by_batch(
            x_star=all_X, output_idx=None, device=device, noiseless=noiseless,
            input_batch_size=128, output_batch_size=128,
        )  # [..., N, P]

        pick_train_Y = train_Y[train_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]
        # TODO: only works when m has same number of True across batch dims
        pick_train_pred_means = pred_means[train_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]
        pick_train_pred_vars = pred_vars[train_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]

        pick_test_Y = test_Y[test_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]
        pick_test_pred_means = pred_means[test_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]
        pick_test_pred_vars = pred_vars[test_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]

        train_se = (pick_train_Y - pick_train_pred_means).square()  # [..., N_train]
        test_se = (pick_test_Y - pick_test_pred_means).square()  # [..., N_test]

        train_nll = gaussian_nll(pick_train_Y, pick_train_pred_means, pick_train_pred_vars)  # [..., N_train]
        test_nll = gaussian_nll(pick_test_Y, pick_test_pred_means, pick_test_pred_vars)  # [..., N_test]

        metric_dict = {
            "train_mse": train_se.mean(dim=(-1)),  # [...], average over N_train
            "test_mse": test_se.mean(dim=(-1)),  # [...], average over N_test
            "train_nll": train_nll.mean(dim=(-1)),  # [...], average over N_train
            "test_nll": test_nll.mean(dim=(-1)),  # [...], average over N_test
        }

        # prediction on dataset input points
        pred_dict = {
            "all_X": all_X,  # [..., N, D_X]
            "pred_means": pred_means,  # [..., N, P]
            "pred_vars": pred_vars,  # [..., N, P]
        }

        # predict on denser input X for plotting
        plot_pred_dict = None

        if num_plot_points is None:
            return metric_dict, pred_dict, plot_pred_dict

        if all_X.size(-1) == 1:
            X_min, X_max = all_X.min().item(), all_X.max().item()
            denser_X = torch.linspace(X_min, X_max, num_plot_points, dtype=all_X.dtype, device=device).unsqueeze(-1)  # [n_plot, 1]
            if len(self.batch_shape) > 0:
                denser_X = denser_X.view(*([1] * len(self.batch_shape)), num_plot_points, 1).expand(*self.batch_shape, num_plot_points, 1)  # [..., n_plot, 1]

            plot_pred_means, plot_pred_vars = self.predict_by_batch(
                x_star=denser_X, output_idx=None, device=device, noiseless=True, input_batch_size=128, output_batch_size=128
            )  # [..., n_plot, P], we want noiseless for plotting

            plot_pred_dict = {
                "denser_X": denser_X,  # [..., n_plot, D_X]
                "plot_pred_means": plot_pred_means,  # [..., n_plot, P]
                "plot_pred_vars": plot_pred_vars,  # [..., n_plot, P]
            }

        return metric_dict, pred_dict, plot_pred_dict