import warnings
from typing import Optional

import torch
from torch import Tensor, LongTensor, BoolTensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from kernels.rbf_kernel import MyRBFKernel
from utils.metrics import gaussian_nll
from utils.build_datasets import IndexDataset
from utils.helpers import wrap_func_by_batch
from models.building_blocks.gp_modules import svgp_base, Inducing_points, mo_Variational_inducing_dist
from likelihood.gaussian import GaussianLikelihood
from likelihood.negative_binomial import NegativeBinomialLikelihood


__all__ = ["ind_svgp_base"]


class ind_svgp_base(svgp_base):
    # independent SVGPs, each for an output
    def __init__(
        self, num_outputs: int, kernel: MyRBFKernel, Z: Inducing_points, qU: mo_Variational_inducing_dist,
        lik_model={"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5},  # all outputs share the same Gaussian likelihood model
        # lik_model={"type": "NegativeBinomial", "k_m": 0.1, "scale_factor": 1., "alpha_joint": False, "alpha_init": 1.}
        whitening: bool = True, jitter: float = 1e-6,
    ):
        super(ind_svgp_base, self).__init__(
            num_outputs=num_outputs, kernel=kernel, Z=Z, qU=qU, whitening=whitening, jitter=jitter,
        )

        self.lik_model_type = lik_model["type"]

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

    def exp_log_lik(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor):
        """
        mini-batch approximation for the expected log likelihood term in ELBO.
        :param x: [..., b, D_X]
        :param y: [..., b, P], where P is the size of the subset of all outputs
        :param m: [..., b, P], where 0 indicate missing
        :param output_idx: [P], the indices of outputs to be selected
        """
        batch_shape = x.shape[:-2]

        if torch.all(m.sum(dim=(-1, -2)) == 0):
            warnings.warn("Encounter one empty mini-batch!")
            return torch.zeros(*batch_shape, device=x.device, dtype=x.dtype)
        else:
            _qf_means, _qf_covs = self.variational_f(x, output_idx) # [..., P, b], BEFORE masking
            qf_means, qf_covs = _qf_means.mT, _qf_covs.mT  # [..., b, P], BEFORE masking
            pick_qf_mean = qf_means[m].view(*batch_shape, -1)  # [..., <b*P]
            pick_qf_cov = qf_covs[m].view(*batch_shape, -1)  # [..., <b*P]
            pick_y = y[m].view(*batch_shape, -1)  # [..., <b*P]

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
                pick_output_idx = expanded_idx[m].view(*batch_shape, -1)  # [..., <b*P>]

                _exp_log_lik = self.lik_model.exp_log_lik(
                    qf_mean=pick_qf_mean, qf_var=pick_qf_cov, y=pick_y, output_idx=pick_output_idx
                )  # [..., <b*P]

                exp_log_lik = _exp_log_lik.mean(dim=(-1))  # [...], average over <b*P

            else:
                raise NotImplementedError(f"Likelihood model {self.lik_model_type} is not implemented for exp_log_lik!")

            return exp_log_lik

    def elbo(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor, coeff_exp_log_lik: float, beta=1., average_elbo: bool = False):
        """
        mini-batch ELBO, b: mini-batch size
        x: [..., b, D_X], i.e., xs are shared across output
        y: [..., b, P], P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicate missing
        output_idx: [P]
        """

        # term 1/2 - exp_log_lik
        exp_log_lik = self.exp_log_lik(x, y, m, output_idx)

        # term 2/2 - KL(q(U)||p(U))
        KL_qU_pU = self.KL_qU_pU.sum(dim=(-1))  # [..., P] -> [...] sum over P

        # sum elbo over (extra) batch dims
        elbo = (
            coeff_exp_log_lik * exp_log_lik - beta * KL_qU_pU
        ).sum()

        if average_elbo:
            elbo = elbo / coeff_exp_log_lik

        return elbo

    def train_ind_svgp(
            # train independent SVGPs, each for an output
            self, train_dataloader: DataLoader, output_batch_size: int, optimizer: Optimizer, epochs: int,
            beta: float = 1., coeff_exp_log_lik: Optional[float] = None, max_norm: Optional[float] = None, device: str = "cpu",
            print_epochs: int = 10, optimizer_natural: Optional[Optimizer] = None,  # optimizer for natural params if applicable
    ) -> None:
        self.to(device)
        self.train()
        if coeff_exp_log_lik is None:
            # biased if there are missing values
            coeff_exp_log_lik = len(train_dataloader.dataset) * self.num_outputs
        output_index_dataloader = None  # cache
        perm = None  # cache

        for epoch in range(epochs):
            for batch_X, batch_all_Y, batch_all_m in train_dataloader:  # [b, ..., D_X/P]

                batch_X = batch_X.to(device)
                batch_all_Y = batch_all_Y.to(device)
                batch_all_m = batch_all_m.to(device)

                # re-arrange dims
                if perm is None:
                    ndim = batch_X.ndim
                    perm = list(range(1, ndim - 1)) + [0, ndim - 1]

                batch_X = batch_X.permute(*perm)

                if output_index_dataloader is None:
                    output_index_dataset = IndexDataset(num_data=batch_all_Y.size(-1))
                    output_index_dataloader = DataLoader(
                        output_index_dataset,
                        batch_size = output_batch_size,
                        shuffle = True,
                        num_workers = 0,
                    )

                for output_idx in output_index_dataloader:
                    output_idx = output_idx.to(device)
                    batch_Y = batch_all_Y[..., output_idx]  # [b, ..., P]
                    batch_m = batch_all_m[..., output_idx]  # [b, ..., P]

                    batch_Y, batch_m = batch_Y.permute(*perm), batch_m.permute(*perm)
                    # TODO: whether or not contiguous is needed? (for GPU training)

                    optimizer.zero_grad(set_to_none=True)
                    if optimizer_natural is not None:
                        optimizer_natural.zero_grad(set_to_none=True)
                    loss = - self.elbo(batch_X, batch_Y, batch_m, output_idx, coeff_exp_log_lik, beta)
                    loss.backward()

                    if max_norm is not None:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

                        if (epoch + 1) % print_epochs == 0 and total_grad_norm.item() > max_norm:
                            print(
                                f"Gradient norm {total_grad_norm.item():.3f} exceeds the threshold {max_norm:.3f}, clipping applied."
                            )

                    optimizer.step()
                    if optimizer_natural is not None:
                        optimizer_natural.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1} / {epochs}； Loss: {loss.item():.6f}')

    @torch.no_grad()
    def predict(
        self, x_star: Tensor, output_idx: Optional[LongTensor] = None, device: str = "cpu", noiseless: bool = False
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

        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device)

        qf_means, qf_covs = self.variational_f(x_star, output_idx)  # [..., P, n_test]
        qf_means, qf_covs = qf_means.mT, qf_covs.mT  # [..., n_test, P]

        if noiseless:
            return qf_means, qf_covs  # [..., n_test, P]

        if self.lik_model_type == "Gaussian":
            qy_covs = qf_covs + self.lik_model.sigma.square() # [..., n_test, P]
            return qf_means, qy_covs

        elif self.lik_model_type == "NegativeBinomial":
            qy_means, qy_vars = self.lik_model.predict(qf_means, qf_covs, output_idx)
            return qy_means, qy_vars  # [..., n_test, P]

        else:
            raise NotImplementedError

    @torch.no_grad()
    def predict_by_batch(
        self, x_star: Tensor, output_idx: Optional[LongTensor] = None, device: str = "cpu", noiseless: bool = False, 
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
            name="ind_svgp_base", input_batch_size=input_batch_size, output_batch_size=output_batch_size, device=device
        )

        return qy_means, qy_vars  # [..., n_test, P]

    @torch.no_grad()
    def predict_ind_svgp_gaussian(
            self, data_dict, noiseless: bool = False, num_plot_points: Optional[int] = 2000, device: str = "cpu"
    ):
        self.eval()
        assert self.lik_model_type == "Gaussian"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )
        batch_shape = all_X.shape[:-2]

        # predict on all_X, latent f
        pred_means, pred_vars = self.predict_by_batch(
            x_star=all_X, output_idx=None, device=device, noiseless=noiseless, input_batch_size=128, output_batch_size=128
        )  # [..., N, P]

        pick_train_Y = train_Y[train_mask.bool()].view(*batch_shape, -1)  # [..., <N*P]
        pick_train_pred_means = pred_means[train_mask.bool()].view(*batch_shape, -1)  # [..., <N*P]
        pick_train_pred_vars = pred_vars[train_mask.bool()].view(*batch_shape, -1)  # [..., <N*P]

        pick_test_Y = test_Y[test_mask.bool()].view(*batch_shape, -1)  # [..., <N*P]
        pick_test_pred_means = pred_means[test_mask.bool()].view(*batch_shape, -1)  # [..., <N*P]
        pick_test_pred_vars = pred_vars[test_mask.bool()].view(*batch_shape, -1)  # [..., <N*P]

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
            if len(batch_shape) > 0:
                denser_X = denser_X.view(*([1] * len(batch_shape)), num_plot_points, 1).expand(*batch_shape, num_plot_points, 1)  # [..., n_plot, 1]

            plot_pred_means, plot_pred_vars = self.predict_by_batch(
                x_star=denser_X, output_idx=None, device=device, noiseless=True,
                input_batch_size=128, output_batch_size=128
            )  # [..., N, P], NOTE: we want noiseless for plotting

            plot_pred_dict = {
                "denser_X": denser_X,  # [..., n_plot, 1]
                "plot_pred_means": plot_pred_means,  # [..., n_plot, P]
                "plot_pred_vars": plot_pred_vars,  # [..., n_plot, P]
            }

        return metric_dict, pred_dict, plot_pred_dict


    @torch.no_grad()
    def predict_ind_svgp_nb(
        self, data_dict, device: str = "cpu"
    ):
        self.eval()
        assert self.lik_model_type == "NegativeBinomial"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )
        all_Y = train_Y + test_Y  # [..., N, P]

        # predict on all_X, latent f
        pred_means, pred_vars = self.predict_by_batch(
            x_star=all_X, output_idx=None, device=device, noiseless=True, input_batch_size=512, output_batch_size=128
        )  # [..., N, P]

        # pass through NB likelihood
        py_means, py_vars = self.lik_model.predict_by_batch(
            pred_means, pred_vars, output_idx=None, num_mc=20, input_batch_size=512, output_batch_size=128
        )  # [..., N, P]

        output_idx = torch.arange(self.num_outputs, device=device)  # [P]
        expanded_idx = output_idx.view(*([1] * (all_Y.ndim - 1)), -1).expand_as(all_Y)  # [..., N, P]
        log_lik = self.lik_model.predict_log_lik_by_batch(
            pred_means, all_Y, expanded_idx, input_batch_size=512, output_batch_size=128,
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
