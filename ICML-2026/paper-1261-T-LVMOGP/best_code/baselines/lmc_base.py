import warnings
from typing import Optional

import math
import torch
from torch import Tensor, BoolTensor, LongTensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.build_datasets import IndexDataset, MyDataset
from utils.metrics import gaussian_nll
from utils.helpers import wrap_func_by_batch
from models.building_blocks.gp_modules import mo_Variational_inducing_dist, Inducing_points, svgp_base
from likelihood.gaussian import GaussianLikelihood
from likelihood.negative_binomial import NegativeBinomialLikelihood


__all__ = ["lmc_base"]


class lmc_base(nn.Module):
    """
    Linear Model of Coregionalization (LMC) for MOGP.
    Latent functions are modelled as (independent) SVGPs.

    Notations:
    P: number of outputs
    """
    def __init__(
        self, P: int, latent_svgps: svgp_base,
        lik_model = {"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5}, # all outputs share the same Gaussian likelihood model
        # lik_model={"type": "NegativeBinomial", "k_m": 0.1, "scale_factor": 1., "alpha_joint": False, "alpha_init": 1.}
    ):
        super(lmc_base, self).__init__()
        self.num_outputs = P
        self.latent_svgps = latent_svgps
        self.num_latents = latent_svgps.num_outputs   # number of latent functions in LMC
        self.batch_shape = tuple(latent_svgps.Z.inducing_points.shape[:-3])
        self.lik_model_type = lik_model["type"]

        # LMC coefficients
        lmc_coefficients = torch.randn(
            *self.batch_shape, self.num_latents, P, dtype=torch.get_default_dtype()
        )   # [..., num_latents, P]
        self.register_parameter(
            "lmc_coefficients", nn.Parameter(lmc_coefficients, requires_grad=True)
        )

        self._setup_likelihood_params(lik_model)

    def _setup_likelihood_params(self, lik_model):
        if lik_model["type"] == "Gaussian":
            assert "sigma_joint" in lik_model.keys()
            assert "sigma_init" in lik_model.keys()
            self.lik_model = GaussianLikelihood(
                sigma_joint = lik_model["sigma_joint"], sigma_init = lik_model["sigma_init"]
            )
        elif lik_model["type"] == "NegativeBinomial":
            assert "k_m" in lik_model.keys()
            assert "scale_factor" in lik_model.keys()
            assert "alpha_joint" in lik_model.keys()
            assert "alpha_init" in lik_model.keys()
            self.lik_model = NegativeBinomialLikelihood(
                k_m = lik_model["k_m"], num_outputs = self.num_outputs, scale_factor = lik_model["scale_factor"],
                alpha_joint = lik_model["alpha_joint"], alpha_init = lik_model["alpha_init"]
            )
        else:
            raise NotImplementedError(f"Likelihood model {self.lik_model_type} is not implemented!")

    def variational_f(self, x: Tensor, output_idx: Optional[LongTensor] = None, m: Optional[BoolTensor] = None):
        """
        b: mini-batch size
        x: [..., b, D_X]; output_idx: [P]; m: [..., b, P], where 0 indicate missing.
        return:
            qf_mean, qf_cov: # [..., <b*P]
        """
        b, P = x.size(-2), len(output_idx)

        if m is None:   # no missing
            m = torch.ones(self.batch_shape + (b, P), dtype=torch.bool, device=x.device)
        else:
            _m_reshape = m.view(
                torch.Size(self.batch_shape).numel(), *m.shape[-2:]
            )   # [(v), b, P], v = batch_shape.numel()
            m_example = _m_reshape[0]  # [b, P]
            assert torch.all(_m_reshape == m_example), "m should be same across batch dims!"
            m = m.bool().to(x.device)

        latent_qf_means, latent_qf_covs = self.latent_svgps.variational_f(x)   # [..., self.num_latents, b]
        picked_lmc_coefficients = self.lmc_coefficients[..., output_idx]   # [..., self.num_latents, P]

        # prepare: expand tensors to [..., self.num_latents, b, P]
        m_ep = m.unsqueeze(-3).expand(*self.batch_shape, self.num_latents, b, P)
        latent_qf_means_exp = latent_qf_means.unsqueeze(-1).expand(*self.batch_shape, self.num_latents, b, P)
        latent_qf_covs_exp = latent_qf_covs.unsqueeze(-1).expand(*self.batch_shape, self.num_latents, b, P)

        picked_lmc_coe_exp = picked_lmc_coefficients.unsqueeze(-2).expand(
            *self.batch_shape, self.num_latents, b, P,
        )

        # compute: qf_means, qf_covs (diagonal cov only): [..., L, <b*P] -> [..., <b*P]
        _qf_means = latent_qf_means_exp[m_ep] * picked_lmc_coe_exp[m_ep]
        qf_means = _qf_means.view(*self.batch_shape, self.num_latents, -1).sum(dim=(-2))

        _qf_covs = latent_qf_covs_exp[m_ep] * picked_lmc_coe_exp[m_ep].square()
        qf_covs = (_qf_covs).view(*self.batch_shape, self.num_latents, -1).sum(dim=(-2))

        # unit test - compute full and then mask
        # test_qf_mean_complete = latent_qf_means.mT @ picked_lmc_coefficients   # [..., b, P]
        # test_qf_mean = test_qf_mean_complete[m].view(*self.batch_shape, -1)   # [..., <b*P]

        # _test_qf_cov_complete = latent_qf_covs.mT @ picked_lmc_coefficients.square()   # [..., b, P]
        # test_qf_cov_complete = _test_qf_cov_complete   # [..., b, P]
        # test_qf_cov = test_qf_cov_complete[m].view(*self.batch_shape, -1)   # [..., <b*P]

        # assert torch.allclose(qf_means, test_qf_mean), "qf_means mismatch"
        # assert torch.allclose(qf_covs, test_qf_cov), "qf_covs mismatch"

        return qf_means, qf_covs   # [..., <b*P]

    def exp_log_lik(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor):
        """
        mini-batch expected log likelihood, b: mini-batch size
        Default: Gaussian Likelihood for regression.
        :param x: [..., b, D_X], i.e. xs are shared across output
        :param y: [..., b, P], P is the size of the subset of all outputs
        :param m: [..., b, P], where 0 indicate missing
        :param output_idx: [P]
        """
        exp_log_lik = 0.

        if torch.all(m.sum(dim=(-1, -2)) == 0):
            warnings.warn("Encounter one empty mini-batch!")
            return exp_log_lik
        else:
            pick_y = y[m].view(*self.batch_shape, -1)   # [..., <b*P]
            pick_qf_mean, pick_qf_cov = self.variational_f(x, output_idx, m)   # [..., <b*P]

            if self.lik_model_type == "Gaussian":
                # all outputs share the same likelihood model
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

    def elbo(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor, coeff_exp_log_lik: float, beta=1., average_elbo=False):
        """
        mini-batch ELBO, b: mini-batch size
        x: [..., b, D_X], i.e. xs are shared across output
        y: [..., b, P], P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicate missing
        output_idx: [P]
        """
        # term 1/2 - exp_log_lik
        exp_log_lik = self.exp_log_lik(x, y, m, output_idx)

        # term 2/2 - KL divergence
        KL_qU_pU = (self.latent_svgps.KL_qU_pU).sum(dim=(-1))   # [..., num_latents] -> [...]

        # sum elbo over (extra) batch dims
        elbo = (
            coeff_exp_log_lik * exp_log_lik - beta * KL_qU_pU
        ).sum()

        if average_elbo:
            elbo = elbo / coeff_exp_log_lik

        return elbo

    def train_lmc(
        self, train_dataloader: DataLoader, output_batch_size: int, optimizer: Optimizer, epochs: int,
        beta=1., coeff_exp_log_lik: Optional[float] = None, max_norm: Optional[float] = None, device="cpu", print_epochs=10
    ):
        self.to(device)
        self.train()
        if coeff_exp_log_lik is None:
            # biased if there are missing values
            coeff_exp_log_lik = len(train_dataloader.dataset) * self.num_outputs
        output_index_dataloader = None  # cache
        perm = None  # cache

        for epoch in range(epochs):
            for batch_X, batch_all_Y, batch_all_m in train_dataloader:   # [b, ..., D_X/P]

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

                    optimizer.zero_grad(set_to_none=True)
                    loss = - self.elbo(batch_X, batch_Y, batch_m, output_idx, coeff_exp_log_lik=coeff_exp_log_lik, beta=beta)
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
    def predict(
        self, x_star: Tensor, output_idx: Optional[LongTensor] = None, device="cpu", noiseless: bool = False
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

        qf_mean, qf_cov = self.variational_f(x_star, output_idx=output_idx)   # [..., n_test*P]
        qf_mean = qf_mean.view(*self.batch_shape, n_test, P)  # [..., n_test, P]
        qf_cov = qf_cov.view(*self.batch_shape, n_test, P)  # [..., n_test, P]

        if noiseless:
            return qf_mean, qf_cov  # [..., n_test, P]

        # pass through likelihood
        if self.lik_model_type == "Gaussian":
            qy_cov = qf_cov + self.lik_model.sigma.square()
            return qf_mean, qy_cov  # [..., n_test, P]

        elif self.lik_model_type == "NegativeBinomial":
            qy_means, qy_vars = self.lik_model.predict(qf_mean, qf_cov, output_idx)
            return qy_means, qy_vars  # [..., n_test, P]

        else:
            raise NotImplementedError

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
            output_idx = torch.arange(self.num_outputs, device=device) # [P]
        
        qy_means, qy_vars = wrap_func_by_batch(
            model=self, func_args={"x_star": x_star, "output_idx": output_idx, "noiseless": noiseless}, 
            name="lmc_base", input_batch_size=input_batch_size, output_batch_size=output_batch_size, device=device
        )

        return qy_means, qy_vars  # [..., n_test, P]

    @torch.no_grad()
    def predict_lmc_gaussian(
        self, data_dict, noiseless: bool = False, num_plot_points: Optional[int] = 2000, device: str = "cpu"
    ):
        self.eval()
        # on device
        assert self.lik_model_type == "Gaussian"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )

        # predict on all_X for metrics
        pred_means, pred_vars = self.predict_by_batch(
            all_X, output_idx=None, device=device, noiseless=noiseless, input_batch_size=128, output_batch_size=128,
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
                denser_X, output_idx=None, device=device, noiseless=True, input_batch_size=128, output_batch_size=128,
            )  # [..., n_plot, P], we want noiseless for plotting

            plot_pred_dict = {
                "denser_X": denser_X,  # [..., n_plot, D_X]
                "plot_pred_means": plot_pred_means,  # [..., n_plot, P]
                "plot_pred_vars": plot_pred_vars,  # [..., n_plot, P]
            }

        return metric_dict, pred_dict, plot_pred_dict
    
    @torch.no_grad()
    def predict_lmc_nb(
        self, data_dict, device: str = "cpu"
    ):
        self.eval()
        # on device
        assert self.lik_model_type == "NegativeBinomial"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )
        all_Y = train_Y + test_Y  # [..., N, P]

        # predict on all_X, latent f 
        pred_means, pred_vars = self.predict_by_batch(
            all_X, output_idx=None, device=device, noiseless=True, input_batch_size=512, output_batch_size=512,
        )  # [..., N, P]

        # pass through NB likelihood
        py_means, py_vars = self.lik_model.predict_by_batch(
            pred_means, pred_vars, output_idx=None, num_mc=20, input_batch_size=512, output_batch_size=512,
        )  # [..., N, P]

        output_idx = torch.arange(self.num_outputs, device=device)  # [P]
        expanded_idx = output_idx.view(*([1] * (all_Y.ndim - 1)), -1).expand_as(all_Y)  # [..., N, P]
        log_lik = self.lik_model.predict_log_lik_by_batch(
            pred_means, all_Y, expanded_idx, input_batch_size=512, output_batch_size=512,
        )  # [..., N, P]
        
        # metric
        all_se = (all_Y - py_means).square()  # [..., N, P]

        # metrics: train/test split
        train_se = all_se[train_mask.bool()]  # [..., <N*P]
        test_se = all_se[test_mask.bool()]  # [..., <N*P]
        train_nll = - log_lik[train_mask.bool()] # [..., <N*P]
        test_nll = - log_lik[test_mask.bool()]  # [..., <N*P]

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
