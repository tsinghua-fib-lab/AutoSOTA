import warnings
from typing import Optional

import torch
from torch import Tensor, BoolTensor, LongTensor
from torch.distributions import MultivariateNormal, kl_divergence
import torch.nn as nn
from torch.optim import Optimizer
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from gpytorch.utils.transforms import inv_softplus

from linear_operator.utils.cholesky import psd_safe_cholesky

from utils.build_datasets import IndexDataset
from utils.metrics import mc_gaussian_nll
from utils.helpers import wrap_func_by_batch
from models.building_blocks.gp_modules import (
    GP_with_qU,
    Delta_H,
)
from models.building_blocks.neural_nets import (
    Identity,
)
from likelihood.gaussian import GaussianLikelihood
from likelihood.negative_binomial import NegativeBinomialLikelihood

__all__ = ["dkl_lvmogp_base"]


class dkl_lvmogp_base(GP_with_qU):
    """
    base class for all DKL-LVMOGP models.

    Notations:
    D_X: input dims
    D_H: latent variable dims
    D_T: transform input dims
    P: number of outputs
    """

    def __init__(
            self, kernel, fnet, pH, qH, qU, Z,
            lik_model={"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5},
            # lik_model={"type": "NegativeBinomial", "k_m": 0.1, "scale_factor": 1., "alpha_joint": False, "alpha_init": 1.}
            whitening=True, jitter=1e-6,
            tighter_elbo=False,  # standard SVGP bound or tighter bound Titsias 2025, Bui et al. 2025
            gp_mean_func=None,  # GP mean function, if None, zero mean is used, other options includes "sum_dims"
            use_cache_for_svgp=False,  # whether to use cache mechanism to compute kl_qU_pU and variational_f_base.
    ):
        assert not kernel.multi_output
        super(dkl_lvmogp_base, self).__init__(
            kernel=kernel, Z=Z, qU=qU, whitening=whitening, jitter=jitter,
        )  # -> create a cache dict
        # NOTE:
        # kernel is on transform input space of D_T dimensional; x_trans = nn(x, H); careful of batch shapes
        # inducing locations are placed on transform input space

        self.fnet = fnet  # feature extractor, shared across batch dims!
        self.pH = pH
        self.qH = qH
        self.lik_model_type = lik_model["type"]
        self.gp_mean_func = gp_mean_func
        self.tighter_elbo = tighter_elbo

        self.batch_shape = Z.batch_shape
        self.num_outputs = self.qH.mean_qH.size(-2)

        self._setup_likelihood_params(lik_model)

        if tighter_elbo:
            self._setup_tighter_elbo_params()

        self.cache["use_cache_for_svgp"] = use_cache_for_svgp

    def _setup_likelihood_params(self, lik_model):
        if lik_model["type"] == "Gaussian":
            assert "sigma_joint" in lik_model.keys()
            assert "sigma_init" in lik_model.keys()
            self.lik_model = GaussianLikelihood(
                sigma_joint=lik_model["sigma_joint"], sigma_init=lik_model["sigma_init"]
            )
        elif lik_model["type"] == "NegativeBinomial":
            assert "k_m" in lik_model.keys()
            assert "scale_factor" in lik_model.keys()
            assert "alpha_joint" in lik_model.keys()
            assert "alpha_init" in lik_model.keys()
            self.lik_model = NegativeBinomialLikelihood(
                k_m=lik_model["k_m"], num_outputs=self.num_outputs, scale_factor=lik_model["scale_factor"],
                alpha_joint=lik_model["alpha_joint"], alpha_init=lik_model["alpha_init"]
            )
        else:
            raise NotImplementedError(f"Likelihood model {self.lik_model_type} is not implemented!")

    def _setup_tighter_elbo_params(self):
        if self.lik_model_type == "Gaussian":
            pass
        else:
            self.register_parameter(
                "raw_m_spherical",
                nn.Parameter(
                    inv_softplus(
                        torch.tensor(1., dtype=torch.get_default_dtype()),
                    ),
                    requires_grad=True
                )
            )

    def _epoch_start_hook(self, epoch: int):
        """
        called at the beginning of each training epoch, this should be overridden in child class.
        """
        pass

    # override
    @property
    def m_spherical(self):
        if self.lik_model_type == "Gaussian" or self.tighter_elbo == False:
            return 1.  # dummy variable
        else:
            return nn.functional.softplus(self.raw_m_spherical)

    def variational_f(self, x: Tensor, H: Tensor, m: Optional[BoolTensor] = None):
        """
        q(f) = \int p(f|U) q(U) dU
        x: [..., b, D_X]; H: [..., P, D_H]; where P is the size of the subset of all outputs.
        m: [..., b, P], where 0 indicate missing.
        return:
            qf_mean, qf_cov: [..., <b*P]
        """
        b, P = x.size(-2), H.size(-2)

        if m is None:  # no missing
            m = torch.ones(self.batch_shape + (b, P), dtype=torch.bool)
        else:
            _m_reshape = m.view(
                torch.Size(self.batch_shape).numel(), *m.shape[-2:]
            )  # [(v), b, P], v = batch_shape.numel()
            m_example = _m_reshape[0]  # [b, P]
            assert torch.all(_m_reshape == m_example), "m should be same across batch dims!"
            m = m.bool()

        _x, _H = x.unsqueeze(-2), H.unsqueeze(-3)  # [..., b, 1, D_X], [..., 1, P, D_H]
        x_exp = _x.expand(*_x.shape[:-2], P, *_x.shape[-1:])  # [..., b, P, D_X]
        H_exp = _H.expand(*_H.shape[:-3], b, *_H.shape[-2:])  # [..., b, P, D_H]

        x_concat = torch.cat([x_exp, H_exp], dim=-1)  # [..., b, P, D_X + D_H]

        # feed into nn feature extractor  (neural network is shared across batch dims!)
        x_trans_flatten = self.fnet(x_concat[m])  # dims collapse! [(batch_shape.numel() * <b*P), D_T]
        D_T = x_trans_flatten.size(-1)
        x_trans_flatten = x_trans_flatten.reshape(
            *self.batch_shape, -1, D_T
        )  # [..., <b*P, D_T]; TODO: underlining assumption is m[-2:] keeps same across batch dims

        # Prepare
        K_uu = self.kernel.forward(self.Z.inducing_points, self.Z.inducing_points)  # [..., M, M]
        K_fu = self.kernel.forward(x_trans_flatten, self.Z.inducing_points)  # [..., <b*P, M]
        K_ff = self.kernel.forward(x_trans_flatten, x_trans_flatten, diag=True)  # [..., <b*P]

        if self.gp_mean_func == "sum_dims":
            mean_values = x_trans_flatten.sum(dim=(-1))  # [..., <b*P]
        elif self.gp_mean_func is None:
            mean_values = None
        else:
            raise NotImplementedError(f"GP mean function {self.gp_mean_func} is not implemented!")

        variational_mean, variational_cov = self.variational_f_base(K_uu, K_fu, K_ff, mean_func_at_f=mean_values)  # [..., <b*P]

        return variational_mean, variational_cov  # [..., <b*P]

    def exp_log_lik(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor):
        """
        mini-batch approximation for the expected log likelihood term in ELBO.
        :param x: [..., b, D_X]
        :param y: [..., b, P], where P is the size of the subset of all outputs
        :param m: [..., b, P], where 0 indicate missing
        :param output_idx: [P], the indices of outputs to be selected
        """
        exp_log_lik = torch.zeros((), dtype=x.dtype, device=x.device)

        if torch.all(m.sum(dim=(-1, -2)) == 0):
            warnings.warn("Encounter one empty mini-batch!")
            return exp_log_lik
        else:
            H = self.qH.sample(output_idx)  # [..., P, D_H]
            pick_qf_mean, pick_qf_cov = self.variational_f(x, H, m)  # [..., <b*P], [..., <b*P]
            pick_y = y[m].view(*self.batch_shape, -1)  # [batch_shape.numel() * <b*P] -> [..., <b*P]

            if self.lik_model_type == "Gaussian":

                _exp_log_lik = self.lik_model.exp_log_lik(
                    qf_mean=pick_qf_mean, qf_var=pick_qf_cov, y=pick_y
                )  # [..., <b*P]

                exp_log_lik = _exp_log_lik.mean(dim=(-1))  # [...], average over <b*P

            elif self.lik_model_type == "NegativeBinomial":

                # pick_output_idx = torch.masked_select(
                #     output_idx.view(*([1] * (m.ndim - 1)), -1),  # [P] -> [...,1,P]
                #     m.bool()
                # ).view(*self.batch_shape, -1)  # [..., <b*P]

                expanded_idx = output_idx.view(*([1] * (m.ndim - 1)), -1).expand_as(m)  # [..., b, P]
                pick_output_idx = expanded_idx[m].view(*self.batch_shape, -1)  # [..., <b*P>]

                _exp_log_lik = self.lik_model.exp_log_lik(
                    qf_mean=pick_qf_mean, qf_var=pick_qf_cov, y=pick_y, output_idx=pick_output_idx
                )  # [..., <b*P]

                exp_log_lik = _exp_log_lik.mean(dim=(-1))  # [...], average over <b*P

            else:
                raise NotImplementedError(f"Likelihood model {self.lik_model_type} is not implemented for exp_log_lik!")

            return exp_log_lik

    def KL_qH_pH(self, output_idx: Optional[LongTensor] = None):
        """
        mini-batch approximation for (per-output) KL between q(H) and p(H), p(H) is fully factorized but q(H) might not.
        """
        # select outputs
        if output_idx is not None:
            mean_pH, mean_qH = self.pH.mean_pH[..., output_idx, :], self.qH.mean_qH[..., output_idx, :]
            cov_pH = self.pH.diag_cov_pH[..., output_idx, :]  # [..., P, D_H], P refers to the size of selected outputs
            if self.qH.mean_field:
                cov_qH = self.qH.cov_qH[..., output_idx, :]  # [..., P, D_H]
            else:
                cov_qH = self.qH.cov_qH[..., output_idx, :, :]  # [..., P, D_H, D_H]
        else:
            mean_pH, mean_qH = self.pH.mean_pH, self.qH.mean_qH
            cov_pH, cov_qH = self.pH.diag_cov_pH, self.qH.cov_qH

        # compute KL values
        if self.qH.mean_field:
            term1 = cov_pH.log() - cov_qH.log()
            term2 = (cov_qH + (mean_qH - mean_pH).pow(2)) / cov_pH
            _KLs = 0.5 * (term1 + term2 - 1.)  # [..., P, D_H]
            KLs = _KLs.sum(dim=(-1)).mean(dim=(-1))  # [...], sum over D_H, average over P

            ## unit test
            # qH = MultivariateNormal(mean_qH, torch.diag_embed(cov_qH))
            # pH = MultivariateNormal(mean_pH, torch.diag_embed(cov_pH))
            # KLs_2 = kl_divergence(qH, pH).mean(dim=(-1))  # [..., P] -> [...]

            # print(f"My implemented KL(qH || pH) is: {KLs}, PyTorch KL is: {KLs_2}.")

        else:
            D_H = cov_qH.size(-1)
            chol_cov_qH = psd_safe_cholesky(
                cov_qH + self.jitter * torch.eye(D_H, dtype=torch.get_default_dtype(), device=cov_qH.device)
            )  # [..., P, D_H, D_H]
            std_pH = cov_pH.sqrt()  # [..., P, D_H]
            trace = (chol_cov_qH / std_pH.unsqueeze(-1)).square().sum(dim=(-1, -2))  # [..., P]
            mahalanobis = ((mean_pH - mean_qH) / std_pH).square().sum(dim=(-1))  # [..., P]
            log_det_cov_pH = cov_pH.log().sum(dim=(-1))  # [..., P]
            log_det_cov_qH = 2 * torch.diagonal(chol_cov_qH, dim1=-1, dim2=-2).log().sum(dim=(-1))  # [..., P]
            _KLs = 0.5 * (trace - D_H + mahalanobis + log_det_cov_pH - log_det_cov_qH)  # [..., P]
            KLs = _KLs.mean(dim=(-1))  # [...], average over P

            ## unit test
            # qH = MultivariateNormal(mean_qH, cov_qH)
            # pH = MultivariateNormal(mean_pH, torch.diag_embed(cov_pH))
            # KLs_2 = kl_divergence(qH, pH).mean(dim=(-1))   # [..., P] -> [...]

            # print(f"My implemented KL(qH || pH) is: {KLs}, PyTorch KL is: {KLs_2}.")

        return KLs

    def correction_term(self):
        """
        correction term for tighter ELBO, i.e. Titsias 2025, Bui et al. 2025.
        """
        if self.lik_model_type == "Gaussian":
            if self.tighter_elbo:
                D_diag_over_sigma_square = self.cache["D_diag"] / self.lik_model.sigma.square()  # mini-batch D_diag, [..., n]
                _cor_term = D_diag_over_sigma_square - torch.log(1 + D_diag_over_sigma_square)  # [..., n]
                cor_term = 0.5 * _cor_term.mean(dim=(-1))  # [...], average over n
                return cor_term
            else:
                return 0.
        else:
            if self.tighter_elbo:
                return 0.5 * (1 + torch.log(self.m_spherical) - self.m_spherical)
            else:
                return 0.

    def elbo(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor, coeff_exp_log_lik: float, beta_u=1.,
             beta_h=1., average_elbo=False):
        """
        mini-batch ELBO, b: mini-batch size
        x: [..., b, D_X], i.e. xs are shared across outputs
        y: [..., b, P], P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicates missing
        output_idx: [P]
        """

        # term 1/4 - exp_log_lik
        exp_log_lik = self.exp_log_lik(x, y, m, output_idx)

        # term 2/4 - KL(q(U)||p(U))
        KL_qU_pU = self.KL_qU_pU  # [...]

        # term 3/4 - KL(q(H)||p(H))
        KL_qH_pH = self.KL_qH_pH(output_idx)  # [...]

        # term 4/4 - correction term
        # For tighterELBO with Gaussian likelihood, this term should be called after exp_log_lik to use cached D_diag.
        cor_term = self.correction_term()  # [...]

        # sum elbo over (extra) batch dims
        elbo = (
                coeff_exp_log_lik * exp_log_lik
                - beta_u * KL_qU_pU
                - beta_h * self.num_outputs * KL_qH_pH
                + coeff_exp_log_lik * cor_term
        ).sum()

        if average_elbo:
            elbo = elbo / coeff_exp_log_lik

        return elbo

    def train_lvmogp(
            self, train_dataloader: DataLoader, output_batch_size: int, optimizer: Optimizer, epochs: int,
            beta_u: float = 1., beta_h: float = 1., coeff_exp_log_lik: Optional[float] = None,
            max_norm: Optional[float] = None, device: str = "cpu",  print_epochs: int = 10,
            optimizer_natural: Optional[Optimizer] = None,  # optimizer for natural params if applicable
    ) -> None:
        """
        """
        self.to(device)
        self.train()
        if coeff_exp_log_lik is None:
            # biased if there are missing values
            coeff_exp_log_lik = len(train_dataloader.dataset) * self.num_outputs
        output_index_dataloader = None  # cache
        perm = None  # cache

        for epoch in range(epochs):
            self._epoch_start_hook(epoch + 1)
            for batch_X, batch_all_Y, batch_all_m in train_dataloader:  # [b, ..., D_X/P]

                batch_X = batch_X.to(device)
                batch_all_Y = batch_all_Y.to(device)
                batch_all_m = batch_all_m.to(device)

                # re-arrange dims
                if perm is None:
                    ndim = batch_X.ndim
                    perm = list(range(1, ndim - 1)) + [0, ndim - 1]

                batch_X = batch_X.permute(*perm)  # [..., b, D_X]

                if output_index_dataloader is None:
                    output_index_dataset = IndexDataset(num_data=batch_all_Y.size(-1))
                    output_index_dataloader = DataLoader(
                        output_index_dataset,
                        batch_size=output_batch_size,
                        shuffle=True,
                        num_workers=0,
                        # pin_memory=True,
                        # persistent_workers=True
                    )

                for output_idx in output_index_dataloader:
                    output_idx = output_idx.to(device)
                    batch_Y = batch_all_Y[..., output_idx]  # [b, ..., p]
                    batch_m = batch_all_m[..., output_idx]  # [b, ..., p]

                    batch_Y, batch_m = batch_Y.permute(*perm), batch_m.permute(*perm)
                    # TODO: whether or not contiguous is needed? (for GPU training)

                    optimizer.zero_grad(set_to_none=True)
                    if optimizer_natural is not None:
                        optimizer_natural.zero_grad(set_to_none=True)

                    loss = - self.elbo(batch_X, batch_Y, batch_m, output_idx, coeff_exp_log_lik, beta_u, beta_h)
                    loss.backward()

                    if max_norm is not None:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

                        if (epoch + 1) % print_epochs == 0 and total_grad_norm.item() > max_norm:
                            print(
                                f"Gradient norm {total_grad_norm.item():.3f} exceeds the threshold {max_norm:.3f}, clipping applied."
                            )

                    optimizer.step()
                    if optimizer_natural is not None:
                        # assert isinstance(optimizer_natural, torch.optim.Optimizer)
                        optimizer_natural.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1} / {epochs}； Loss: {loss.item():.8f}')

    @torch.no_grad()
    def predict(
            self, x_star: Tensor, output_idx: Optional[LongTensor] = None, num_samples: int = 1,
            device="cpu", noiseless: bool = False
    ):
        """
        x_star: [..., n_test, D_X]
        Get predictive mean and var for output_idx on x_star.
        If output_idx is None, then make predictions for all outputs.
        If noiseless is True, then return latent f predictions, otherwise return y predictions (i.e., passed through likelihood).
        """
        self.to(device)
        self.eval()
        x_star = x_star.to(device)

        n_test = x_star.size(-2)

        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device)

        P = len(output_idx)

        qf_means, qf_covs = [], []
        for i in range(num_samples):
            H_samples = self.qH.sample(output_idx)
            qf_mean, qf_cov = self.variational_f(x_star, H_samples)  # [..., n_test*P]
            qf_mean = qf_mean.view(*self.batch_shape, n_test, P)
            qf_cov = qf_cov.view(*self.batch_shape, n_test, P)
            qf_means.append(qf_mean)
            qf_covs.append(qf_cov)

        qf_means = torch.stack(qf_means, dim=-3)  # [..., s, n_test, P]
        qf_covs = torch.stack(qf_covs, dim=-3)  # [..., s, n_test, P]

        if noiseless:
            return qf_means, qf_covs  # [..., s, n_test, P]

        # pass through likelihood
        if self.lik_model_type == "Gaussian":
            qy_covs = qf_covs + self.lik_model.sigma.square()
            return qf_means, qy_covs  # [..., s, n_test, P]

        elif self.lik_model_type == "NegativeBinomial":
            qy_means, qy_vars = self.lik_model.predict(qf_means, qf_covs, output_idx)
            return qy_means, qy_vars  # [..., s, n_test, P]

        else:
            raise NotImplementedError

    @torch.no_grad()
    def predict_by_batch(
            self, x_star: Tensor, output_idx: Optional[LongTensor] = None, num_samples: int = 1,
            device="cpu", noiseless: bool = False, input_batch_size: int = 64, output_batch_size: int = 32
    ):
        """
        For large scale dataset (with large number of inputs and outputs), split the prediction into mini-batches.
        x_star: [..., n_test, D_X]
        output_idx: [P] or None
        """
        self.eval()
        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device) # [P]

        qy_means, qy_vars = wrap_func_by_batch(
            model=self, func_args={"x_star": x_star, "output_idx": output_idx, "num_samples": num_samples, "noiseless": noiseless},
            name="dkl_lvmogp_base", input_batch_size=input_batch_size, output_batch_size=output_batch_size, device=device
        )

        return qy_means, qy_vars  # [..., s, n_test, P]

    @torch.no_grad()
    def predict_lvmogp_gaussian(
            self, data_dict, num_samples: int = 10, noiseless: bool = False, num_plot_points: Optional[int] = 2000,
            device: str = "cpu"
    ):
        self.eval()
        # on device
        assert self.lik_model_type == "Gaussian"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device),
            data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )

        # predict on all_X for metrics
        pred_means, pred_vars = self.predict_by_batch(
            all_X, output_idx=None, num_samples=num_samples, device=device, noiseless=noiseless,
            input_batch_size=128, output_batch_size=128,
        )  # [..., s, N, P]

        # average over samples, not for metrics
        average_pred_means = pred_means.mean(dim=(-3))  # [..., N, P]
        # the law of total variance
        average_pred_vars = pred_vars.mean(dim=(-3)) + pred_means.var(dim=(-3), unbiased=False)  # [..., N, P]

        pick_train_Y = train_Y[train_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P] TODO: only works when m has same number of True across batch dims
        expand_train_mask = train_mask.unsqueeze(-3).expand_as(pred_means)  # [..., s, N, P]
        pick_train_pred_means = pred_means[expand_train_mask.bool()].view(*self.batch_shape, num_samples, -1)  # [..., s, <N*P]
        pick_train_pred_vars = pred_vars[expand_train_mask.bool()].view(*self.batch_shape, num_samples, -1)  # [..., s, <N*P]

        pick_test_Y = test_Y[test_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]
        expand_test_mask = test_mask.unsqueeze(-3).expand_as(pred_means)  # [..., s, N, P]
        pick_test_pred_means = pred_means[expand_test_mask.bool()].view(*self.batch_shape, num_samples, -1)  # [..., s, <N*P]
        pick_test_pred_vars = pred_vars[expand_test_mask.bool()].view(*self.batch_shape, num_samples, -1)  # [..., s, <N*P]

        train_se = (pick_train_Y - pick_train_pred_means.mean(dim=(-2))).square()  # [..., N_train]
        test_se = (pick_test_Y - pick_test_pred_means.mean(dim=(-2))).square()  # [..., N_test]

        train_nll = mc_gaussian_nll(pick_train_Y, pick_train_pred_means, pick_train_pred_vars)  # [..., N_train]
        test_nll = mc_gaussian_nll(pick_test_Y, pick_test_pred_means, pick_test_pred_vars)  # [..., N_test]

        metric_dict = {
            "train_mse": train_se.mean(dim=(-1)),  # [...], average over N_train
            "test_mse": test_se.mean(dim=(-1)),  # [...], average over N_test
            "train_nll": train_nll.mean(dim=(-1)),  # [...], average over N_train
            "test_nll": test_nll.mean(dim=(-1)),  # [...], average over N_test
        }

        # prediction on dataset input points
        pred_dict = {
            "all_X": all_X,  # [..., N, D_X]
            "pred_means": average_pred_means,  # [..., N, P]
            "pred_vars": average_pred_vars,  # [..., N, P]
        }

        # predict on denser input X for plotting
        plot_pred_dict = None
        # assert all_X.size(-1) == 1, "Only D_X = 1 is implemented!"
        if num_plot_points is None:
            return metric_dict, pred_dict, plot_pred_dict

        if all_X.size(-1) == 1:
            X_min, X_max = all_X.min().item(), all_X.max().item()
            denser_X = torch.linspace(X_min, X_max, num_plot_points, dtype=all_X.dtype, device=device).unsqueeze(-1)  # [n_plot, 1]
            if len(self.batch_shape) > 0:
                denser_X = denser_X.view(*([1] * len(self.batch_shape)), num_plot_points, 1).expand(*self.batch_shape, num_plot_points, 1)  # [..., n_plot, 1]

            plot_pred_means, plot_pred_vars = self.predict_by_batch(
                denser_X, output_idx=None, num_samples=num_samples, device=device, noiseless=True,
                input_batch_size=128, output_batch_size=128
            )  # [..., s, N, P], we want noiseless for plotting

            # average over samples, only for plotting purpose
            average_plot_pred_means = plot_pred_means.mean(dim=(-3))  # [..., N, P]
            # the law of total variance
            average_plot_pred_vars = plot_pred_vars.mean(dim=(-3)) + plot_pred_means.var(dim=(-3), unbiased=False)  # [..., N, P]

            plot_pred_dict = {
                "denser_X": denser_X,  # [..., n_plot, D_X]
                "plot_pred_means": average_plot_pred_means,  # [..., n_plot, P]
                "plot_pred_vars": average_plot_pred_vars,  # [..., n_plot, P]
            }

        return metric_dict, pred_dict, plot_pred_dict

    @torch.no_grad()
    def predict_lvmogp_nb(
            self, data_dict, num_samples: int = 10, device: str = "cpu"
    ):
        """
        num_samples: number of MC samples from qH used for prediction
        """
        self.eval()
        # on device
        assert self.lik_model_type == "NegativeBinomial"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device),
            data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )
        all_Y = train_Y + test_Y  # [..., N, P]

        # predict on all_X, latent f
        pred_means, pred_vars = self.predict_by_batch(
            all_X, output_idx=None, num_samples=num_samples, device=device, noiseless=True,
            input_batch_size=512, output_batch_size=512,
        )  # [..., s, N, P]

        # pass through NB likelihood
        _py_means, _py_vars = self.lik_model.predict_by_batch(
            pred_means, pred_vars, output_idx=None, num_mc=20, input_batch_size=512, output_batch_size=512,
        )  # [..., s, N, P]
        py_means, py_vars = _py_means.mean(dim=(-3)), _py_vars.mean(dim=(-3)) + _py_means.var(dim=(-3), unbiased=False)  # [..., N, P]

        exp_all_Y = all_Y.unsqueeze(-3).expand_as(pred_means)  # [..., s, N, P]
        output_idx = torch.arange(self.num_outputs, device=device)  # [P]
        expanded_idx = output_idx.view(*([1] * (exp_all_Y.ndim - 1)), -1).expand_as(exp_all_Y)  # [..., s, N, P]
        log_lik = self.lik_model.predict_log_lik_by_batch(
            pred_means, exp_all_Y, expanded_idx, input_batch_size=512, output_batch_size=512,
        )
        # log_lik = self.lik_model.log_lik_given_f_value(pred_means, exp_all_Y, expanded_idx)  # [..., s, N, P]
        all_mc_nb_ll = torch.logsumexp(log_lik, dim=-3) - torch.log(torch.tensor(num_samples))  # [..., N, P], logsumexp trick over qH samples

        # metric
        all_se = (all_Y - py_means).square()  # [..., N, P]

        # metrics: train/test split
        train_se = all_se[train_mask.bool()]  # [..., <N*P]
        test_se = all_se[test_mask.bool()]  # [..., <N*P]
        train_nll = - all_mc_nb_ll[train_mask.bool()] # [..., <N*P]
        test_nll = - all_mc_nb_ll[test_mask.bool()]  # [..., <N*P]

        metric_dict = {
            "train_mse": train_se.mean(dim=(-1)),  # [...]
            "test_mse": test_se.mean(dim=(-1)),  # [...]
            "train_nll": train_nll.mean(dim=(-1)),  # [...]
            "test_nll": test_nll.mean(dim=(-1)),  # [...]
        }

        # prediction on dataset input points
        pred_dict = {
            "all_X": all_X,  # [..., N, D_X]
            "pred_means": py_means,  # [..., N, P]
            "pred_vars": py_vars,  # [..., N, P]
        }

        plot_pred_dict = None

        return metric_dict, pred_dict, plot_pred_dict

    @torch.no_grad()
    def predict_given_H(
            self, x_star: Tensor, H_values: Tensor, num_samples: int = 1, pH_cov_value: Optional[float] = None,
            device="cpu", noiseless: bool = False,
    ):
        """
        Make predictions at (new) outputs with given H values.
        x_star: [..., n_test, D_X]
        H_values: [..., P_test, D_H], P_test is the number of outputs to be predicted
        pH_cov_value: if not None, sample H from N(H_values, pH_cov_value*I) instead of using H_values directly.
        If noiseless is True, then return latent f predictions, otherwise return y predictions (i.e., passed through likelihood).
        """
        assert self.qH.mean_qH.size(-1) == H_values.size(-1), "H_values have unmatched dimensionality with qH!"
        self.to(device)
        self.eval()
        x_star = x_star.to(device)

        n_test, P_test = x_star.size(-2), H_values.size(-2)

        if pH_cov_value is None:
            qf_mean, qf_cov = self.variational_f(x_star, H_values)  # [..., n_test*P]
            qf_mean = qf_mean.view(*self.batch_shape, n_test, P_test)
            qf_cov = qf_cov.view(*self.batch_shape, n_test, P_test)
        else:
            assert pH_cov_value > 0., "pH_cov_value must be positive!"
            qf_means, qf_covs = [], []
            for i in range(num_samples):
                H_samples = H_values + pH_cov_value**0.5 * torch.randn_like(H_values)
                qf_mean, qf_cov = self.variational_f(x_star, H_samples)  # [..., n_test*P]
                qf_mean = qf_mean.view(*self.batch_shape, n_test, P_test)
                qf_cov = qf_cov.view(*self.batch_shape, n_test, P_test)
                qf_means.append(qf_mean)
                qf_covs.append(qf_cov)

            qf_means = torch.stack(qf_means, dim=-3)  # [..., s, n_test, P]
            qf_covs = torch.stack(qf_covs, dim=-3)  # [..., s, n_test, P]

            # mixture of Gaussians
            qf_mean = qf_means.mean(dim=(-3))  # [..., n_test, P]
            qf_cov = qf_covs.mean(dim=(-3)) + qf_means.var(dim=(-3), unbiased=False)  # [..., n_test, P]

        if noiseless:
            return qf_mean, qf_cov  # [..., n_test, P]

        # pass through likelihood
        if self.lik_model_type == "Gaussian":
            qy_cov = qf_cov + self.lik_model.sigma.square()
            return qf_mean, qy_cov  # [..., n_test, P]
        else:
            raise NotImplementedError

class det_dkl_lvmogp_base(dkl_lvmogp_base):
    """
    Deterministic DKL-LVMOGP model, i.e., H is point estimated.
    """
    def __init__(
            self, kernel, fnet, qH, qU, Z,
            lik_model={"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5},
            whitening=True, jitter=1e-6,
            tighter_elbo=False,  # standard SVGP bound or tighter bound Titsias 2025, Bui et al. 2025
            gp_mean_func=None,  # GP mean function, if None, zero mean is used, other options includes "sum_dims"
            use_cache_for_svgp=False,  # whether to use cache mechanism to compute kl_qU_pU and variational_f_base.
    ):
        assert isinstance(qH, Delta_H), "qH must be Delta_H for deterministic DKL-LVMOGP."
        pH = None  # dummy variable

        super(det_dkl_lvmogp_base, self).__init__(
            kernel=kernel, fnet=fnet, pH=pH, qH=qH, qU=qU, Z=Z,
            lik_model=lik_model, whitening=whitening, jitter=jitter, tighter_elbo=tighter_elbo, gp_mean_func=gp_mean_func,
            use_cache_for_svgp=use_cache_for_svgp,
        )

    # override
    def KL_qH_pH(self, *args, **kwargs):
        raise NotImplementedError("No KL term between qH and pH for deterministic DKL-LVMOGP.")

    # override
    def elbo(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor, coeff_exp_log_lik: float, beta_u=1.,
             beta_h=1., average_elbo=False):
        """
        Compare to the elbo of normal dkl_lvmogp model, there is no KL(qH||pH) term.

        mini-batch ELBO, b: mini-batch size
        x: [..., b, D_X], i.e. xs are shared across outputs
        y: [..., b, P], P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicates missing
        output_idx: [P]
        """

        # term 1/3 - exp_log_lik
        exp_log_lik = self.exp_log_lik(x, y, m, output_idx)

        # term 2/3 - KL(q(U)||p(U))
        KL_qU_pU = self.KL_qU_pU  # [...]

        # term 3/3 - correction term
        cor_term = self.correction_term()  # [...]

        # sum elbo over (extra) batch dims
        elbo = (
                coeff_exp_log_lik * exp_log_lik
                - beta_u * KL_qU_pU
                + coeff_exp_log_lik * cor_term
        ).sum()

        if average_elbo:
            elbo = elbo / coeff_exp_log_lik

        return elbo

    # override
    def train_lvmogp(
            self, *args, **kwargs
    ) -> None:
        self.train()
        # NOTE: no beta_h parameter needed
        if "beta_h" in kwargs:
            raise TypeError("Don't pass beta_h to train (identity) det_dkl_lvmogp_base!")
        super(det_dkl_lvmogp_base, self).train_lvmogp(*args, **kwargs)

    # override
    @torch.no_grad()
    def predict(
            self, x_star: Tensor, output_idx: Optional[LongTensor] = None, num_samples: int = 1, device="cpu", noiseless: bool = False
    ):
        self.eval()
        # fix num_samples to 1
        assert num_samples == 1, "num_samples must be 1 for deterministic DKL-LVMOGP."
        qy_means, qy_vars = super(det_dkl_lvmogp_base, self).predict(
            x_star, output_idx=output_idx, num_samples=1, device=device, noiseless=noiseless
        )  # [..., 1, n_test, P], [..., 1, n_test, P]

        return qy_means, qy_vars

    # override
    @torch.no_grad()
    def predict_lvmogp_gaussian(
            self, *args, **kwargs
    ):
        self.eval()
        kwargs.pop("num_samples", None)
        metric_dict, pred_dict, plot_pred_dict = super(det_dkl_lvmogp_base, self).predict_lvmogp_gaussian(*args, num_samples=1, **kwargs)
        return metric_dict, pred_dict, plot_pred_dict

    # override
    @torch.no_grad()
    def predict_lvmogp_nb(
            self, *args, **kwargs
    ):
        self.eval()
        kwargs.pop("num_samples", None)
        metric_dict, pred_dict, plot_pred_dict = super(det_dkl_lvmogp_base, self).predict_lvmogp_nb(*args, num_samples=1, **kwargs)
        return metric_dict, pred_dict, plot_pred_dict


class identity_dkl_lvmogp_base(dkl_lvmogp_base):
    """DKL-LVMOGP model with identity feature extractor, i.e., direct concatenation of x and H (from q(H)) as GP input."""
    def __init__(
            self, kernel, pH, qH, qU, Z,
            lik_model={"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5},
            whitening=True, jitter=1e-6, tighter_elbo=False,  # standard SVGP bound or tighter bound Titsias 2025, Bui et al. 2025
            gp_mean_func=None,  # GP mean function, if None, zero mean is used, other options includes "sum_dims"
            use_cache_for_svgp=False,  # whether to use cache mechanism to compute kl_qU_pU and variational_f_base.
    ):
        fnet = Identity()
        super(identity_dkl_lvmogp_base, self).__init__(
            kernel=kernel, fnet=fnet, pH=pH, qH=qH, qU=qU, Z=Z,
            lik_model=lik_model, whitening=whitening, jitter=jitter, tighter_elbo=tighter_elbo, gp_mean_func=gp_mean_func,
            use_cache_for_svgp=use_cache_for_svgp,
        )


class identity_det_dkl_lvmogp_base(det_dkl_lvmogp_base):
    """
    deterministic DKL-LVMOGP model with identity feature extractor, i.e., direct concatenation of x and H as GP input.
    """
    def __init__(
            self, kernel, qH, qU, Z,
            lik_model={"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5},
            whitening=True, jitter=1e-6, tighter_elbo=False,  # standard SVGP bound or tighter bound Titsias 2025, Bui et al. 2025
            gp_mean_func=None,  # GP mean function, if None, zero mean is used, other options includes "sum_dims"
            use_cache_for_svgp = False,  # whether to use cache mechanism to compute kl_qU_pU and variational_f_base.
    ):
        fnet = Identity()
        super(identity_det_dkl_lvmogp_base, self).__init__(
            kernel=kernel, fnet=fnet, qH=qH, qU=qU, Z=Z,
            lik_model=lik_model, whitening=whitening, jitter=jitter, tighter_elbo=tighter_elbo, gp_mean_func=gp_mean_func,
            use_cache_for_svgp=use_cache_for_svgp,
        )

