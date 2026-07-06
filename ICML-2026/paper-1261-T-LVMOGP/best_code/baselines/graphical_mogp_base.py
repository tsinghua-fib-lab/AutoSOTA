import warnings
from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor, BoolTensor, LongTensor

from gpytorch.means import Mean
from gpytorch.kernels import Kernel

from utils.build_datasets import IndexDataset
from utils.metrics import gaussian_nll
from utils.helpers import wrap_func_by_batch
from models.building_blocks.gp_modules import Inducing_points, mo_Variational_inducing_dist, GP_with_qU
from likelihood.gaussian import GaussianLikelihood
from likelihood.negative_binomial import NegativeBinomialLikelihood


class graphical_mogp_base(GP_with_qU):
    """
    Graphical Multioutput Gaussian Process with Attention (ICLR 2024)

    Original implementation: https://github.com/Blspdianna/GMOGP/tree/master
    """
    def __init__(
        self, mean: Mean, kernel: Kernel, Z: Inducing_points, qU: mo_Variational_inducing_dist,
        lik_model={"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5},
        # lik_model={"type": "NegativeBinomial", "k_m": 0.1, "scale_factor": 1., "alpha_joint": False, "alpha_init": 1.}
        whitening: bool = True, jitter: float = 1e-6,
    ):
        # check, should be [..., P]
        assert kernel.multi_output
        assert mean.batch_shape == kernel.batch_shape == Z.batch_shape == qU.batch_shape
        super(graphical_mogp_base, self).__init__(
            kernel=kernel, Z=Z, qU=qU, whitening=whitening, jitter=jitter
        )

        self.mean = mean
        self.num_outputs = Z.batch_shape[-1]   # the last batch dim is the output dim
        self.batch_shape = Z.batch_shape[:-1]  # all but the last output dim
        self.lik_model_type = lik_model["type"]

        self.leakyrelu = nn.LeakyReLU(0.2)  # follows Dai's practice

        self._setup_likelihood_params(lik_model)

        # register attention weights and biases
        attention_shape = self.batch_shape + (self.num_outputs, self.num_outputs)

        self.register_parameter(
            name="attention_weights", param=nn.Parameter(
                torch.ones(attention_shape, dtype=torch.get_default_dtype()), requires_grad=True
            )
        )  # [..., P, P]

        nn.init.xavier_uniform_(self.attention_weights, gain=1.414)  # following Dai's practice

        self.register_parameter(
            name="attention_bias", param=nn.Parameter(
                torch.randn(attention_shape, dtype=torch.get_default_dtype()), requires_grad=True
            )
        )  # [..., P, P]

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

    @staticmethod
    def compute_pairwise_cos_similarity(Y: Tensor, m: Tensor):
        # Y: [..., N, P], m: [..., N, P] where 0 indicates missing
        # N refers to the number of data samples, P is the number of outputs
        # returns: [..., P, P]
        masked_Y = Y * m  # [..., N, P]
        masked_norm = masked_Y.square().sum(dim=-2).sqrt()  # [..., P]
        masked_pairwise_inner_product = masked_Y.mT @ masked_Y  # [..., P, P]
        masked_pairwise_norm_product = masked_norm.unsqueeze(-1) @ masked_norm.unsqueeze(-2) # [..., P, P]

        masked_pairwise_norm_product = torch.clamp(masked_pairwise_norm_product, min=1e-6)  # [..., P, P]
        res = masked_pairwise_inner_product / masked_pairwise_norm_product  # [..., P, P]

        return res  # [..., P, P]

    def register_cos_sim(self, Y: Tensor, m: Tensor):
        if hasattr(self, "cos_sim"):
            print("Cosine similarity already registered, skipping.")
        else:
            cos_sim = self.compute_pairwise_cos_similarity(Y, m)
            self.register_buffer("cos_sim", cos_sim)  # [..., P, P]

    def pick_alpha(self, output_idx: Optional[LongTensor] = None):
        """
        alpha parameter for weighted sum for mean function and kernel function
        output_idx: [P_sel] where P_sel is the number of selected outputs
        # returns: [..., P_sel, P] and P is the number of all outputs
        """
        assert hasattr(self, "cos_sim"), "Cosine similarity not registered. Call register_cos_sim() first."
        assert self.cos_sim.shape[:-2] == self.batch_shape

        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=self.cos_sim.device)

        pick_cos_sim = self.cos_sim.index_select(-2, output_idx)  # [..., P_sel, P]
        pick_attention_weights = self.attention_weights.index_select(-2, output_idx)  # [..., P_sel, P]
        pick_attention_bias = self.attention_bias.index_select(-2, output_idx)  # [..., P_sel, P]

        middle_res = self.leakyrelu(pick_cos_sim * pick_attention_weights + pick_attention_bias)  # [..., P_sel, P]

        # Weird? Following Dai's practice
        expanded_output_idx = output_idx.unsqueeze(-1).expand(*self.batch_shape, -1, self.num_outputs)
        exp_middle_res = torch.exp(middle_res)
        zeros_exp_middle_res = exp_middle_res.scatter(-1, expanded_output_idx, 0)  # [..., P_sel, P]
        res = zeros_exp_middle_res / (1 + zeros_exp_middle_res.sum(dim=(-1), keepdim=True)) # [..., P_sel, P]

        # replace 'diag' elements with 1
        res = res.scatter(-1, expanded_output_idx, 1)  # [..., P_sel, P]

        return res  # [..., P_sel, P]

    def pick_weighted_mean(
        self, x: Tensor, x_has_output_dim: bool = False, output_idx: Optional[LongTensor] = None
    ):
        # x:       [..., b, D_X], if there is no output dim in x, otherwise [..., P, b, D_X]
        # output_idx: [P_sel], where P_sel is the number of selected outputs
        # returns: [..., P_sel, b], P is the number of all outputs

        # prepare: pick alpha, prepare proper shapes
        picked_alpha = self.pick_alpha(output_idx).unsqueeze(-1)  # [..., P_sel, P] -> [..., P_sel, P, 1]

        if x_has_output_dim:
            assert x.size(-3) == self.num_outputs
            expanded_x = x  # [..., P, b, D_X]
        else:
            expanded_x = x.unsqueeze(-3).expand(*self.batch_shape, self.num_outputs, *x.shape[-2:])  # [..., P, b, D_X]

        # compute means for all outputs
        all_means = self.mean(expanded_x).unsqueeze(-3)  # [..., P, b] -> [..., 1, P, b]

        # apply alpha weights
        weighted_means_before_sum = all_means * picked_alpha  # [..., P_sel, P, b]

        # sum over P dim
        weighted_means = weighted_means_before_sum.sum(dim=-2) # [..., P_sel, b]

        return weighted_means  # [..., P_sel, b]

    def pick_weighted_cov(
        self, x1: Tensor, x2: Tensor, x1_has_output_dim: bool = False, x2_has_output_dim: bool = False,
        diag: bool = False, output_idx: Optional[LongTensor] = None
    ):
        # x1: [..., b1, D_X] if there is no output dim in x1, otherwise [..., P, b1, D_X]
        # x2: [..., b2, D_X] if there is no output dim in x2, otherwise [..., P, b2, D_X]
        # output_idx: [P_sel], where P_sel is the number of selected outputs
        # return: [..., P_sel, b1, b2] or [..., P_sel, b1=b2] if diag=True

        # check
        if diag: assert torch.equal(x1, x2)

        # prepare: pick alpha, prepare proper shapes
        if diag:
            picked_alpha = self.pick_alpha(output_idx).unsqueeze(-1)  # [..., P_sel, P] -> [..., P_sel, P, 1]
        else:
            picked_alpha = self.pick_alpha(output_idx).unsqueeze(-1).unsqueeze(-1)  # [..., P_sel, P] -> [..., P_sel, P, 1, 1]

        if not x1_has_output_dim:
            x1 = x1.unsqueeze(-3).expand(*self.batch_shape, self.num_outputs, *x1.shape[-2:])  # [..., P, b, D_X]
        else:
            assert x1.size(-3) == self.num_outputs

        if not x2_has_output_dim:
            x2 = x2.unsqueeze(-3).expand(*self.batch_shape, self.num_outputs, *x2.shape[-2:])  # [..., P, b, D_X]
        else:
            assert x2.size(-3) == self.num_outputs

        # compute covariance matrices for all outputs
        all_covs = self.kernel.forward(x1, x2, diag=diag)  # [..., P, b1, b2] or [..., P, b1=b2] if diag=True

        # apply alpha weights
        expand_dim = -3 if diag else -4
        all_covs = all_covs.unsqueeze(expand_dim)  # [..., 1, P, b1, b2] or [..., 1, P, b1=b2] if diag=True
        weighted_covs_before_sum = all_covs * picked_alpha # [..., P_sel, P, b1, b2] or [..., P_sel, P, b1=b2]

        # sum over P dim
        sum_dim = -2 if diag else -3
        weighted_covs = weighted_covs_before_sum.sum(dim=sum_dim)  # [..., P_sel, b1, b2] or [..., P_sel, b1=b2]

        return weighted_covs # [..., P_sel, b1, b2] or [..., P_sel, b1 = b2]

    def variational_f(self, x: Tensor, output_idx: Optional[LongTensor] = None):
        r"""
        x: [..., b, D_X]
        output_idx: [P_sel], where P_sel is the number of selected outputs
        """
        if output_idx == None:
            output_idx = torch.arange(self.num_outputs, device=x.device)

        mean_func_at_f = self.pick_weighted_mean(
            x=x, x_has_output_dim=False, output_idx=output_idx
        )  # [..., P_sel, b]

        K_uu = self.pick_weighted_cov(
            x1=self.Z.inducing_points, x2=self.Z.inducing_points, x1_has_output_dim=True, x2_has_output_dim=True,
            diag=False, output_idx=output_idx
        )  # [..., P_sel, M, M]

        K_fu = self.pick_weighted_cov(
            x1=x, x2=self.Z.inducing_points, x1_has_output_dim=False, x2_has_output_dim=True,
            diag=False, output_idx=output_idx
        )  # [..., P_sel, M, b]

        K_ff = self.pick_weighted_cov(
            x1=x, x2=x, x1_has_output_dim=False, x2_has_output_dim=False,
            diag=True, output_idx=output_idx
        )  # [..., P_sel, b]

        variational_mean, variational_cov = self.variational_f_base(
            K_uu=K_uu, K_fu=K_fu, K_ff=K_ff, output_idx=output_idx,
            mean_func_at_f=mean_func_at_f
        )  # [..., P_sel, b]

        return variational_mean, variational_cov

    def exp_log_lik(self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor):
        """
        mini-batch approximation for the expected log likelihood term in ELBO.

        x: [..., b, D_X]
        y: [..., b, P], where P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicates missing
        output_idx: [P], the indices of outputs to be selected
        """
        exp_log_lik = 0.

        if torch.all(m.sum(dim=(-1, -2)) == 0):
            warnings.warn("Encounter one empty mini-batch!")
            return exp_log_lik
        else:
            _qf_means, _qf_covs = self.variational_f(x, output_idx)  # [..., P, b], BEFORE masking
            qf_means, qf_covs = _qf_means.mT, _qf_covs.mT  # [..., b, P], BEFORE masking
            pick_qf_mean = qf_means[m].view(*self.batch_shape, -1)  # [..., <b*P]
            pick_qf_cov = qf_covs[m].view(*self.batch_shape, -1)  # [..., <b*P]
            pick_y = y[m].view(*self.batch_shape, -1)  # [..., <b*P]

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

            return exp_log_lik  # [...]

    def elbo(
        self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor,
        coeff_exp_log_lik: float, beta: float = 1.0, average_elbo: bool = False,
    ):
        """
        mini-batch ELBO, b: mini-batch size
        x: [..., b, D_X], i.e. x are shared across outputs
        y: [..., b, P], P is the size of the subset of outputs
        m: [..., b, P], where 0 indicates missing
        output_idx: [P]
        """

        # term 1/2 - exp_log_lik
        exp_log_lik = self.exp_log_lik(x, y, m, output_idx)

        # term 2/2 - KL(q(U)||p(U))
        KL_qU_pU = self.KL_qU_pU.sum(dim=(-1))  # [..., P] -> [...] sum over P

        elbo = (
            coeff_exp_log_lik * exp_log_lik - beta * KL_qU_pU
        ).sum()

        if average_elbo:
            elbo = elbo / coeff_exp_log_lik

        return elbo

    def train_gmogp(
        self, train_dataloader: DataLoader, output_batch_size: int, optimizer: Optimizer, epochs: int,
        beta: float = 1, coeff_exp_log_lik: Optional[float] = None, max_norm: Optional[float] = None, device: str = "cpu", print_epochs: int = 10
    ) -> None:
        """
        """
        self.to(device)
        self.train()
        if coeff_exp_log_lik is None:
            coeff_exp_log_lik = len(train_dataloader.dataset) * self.num_outputs
        output_index_dataloader = None  # cache
        perm = None

        for epoch in range(epochs):
            for batch_X, batch_all_Y, batch_all_m in train_dataloader:  # [b, ..., D_X/P]

                batch_X = batch_X.to(device)
                batch_all_Y = batch_all_Y.to(device)
                batch_all_m = batch_all_m.to(device)

                # re-arrange dims
                if perm is None:
                    ndim = batch_X.ndim
                    perm = list(range(1, ndim - 1)) + [0, ndim - 1]

                batch_X = batch_X.permute(*perm)  # [b, ..., D_X] -> [..., b, D_X]

                if output_index_dataloader is None:
                    output_index_dataset = IndexDataset(num_data=batch_all_Y.size(-1))
                    output_index_dataloader = DataLoader(
                        output_index_dataset,
                        batch_size=output_batch_size,
                        shuffle=True,
                        num_workers=0
                    )

                for output_idx in output_index_dataloader:
                    output_idx = output_idx.to(device)
                    batch_Y = batch_all_Y[..., output_idx]  # [b, ..., P]
                    batch_m = batch_all_m[..., output_idx]  # [b, ..., P]

                    batch_Y, batch_m = batch_Y.permute(*perm), batch_m.permute(*perm)

                    optimizer.zero_grad(set_to_none=True)
                    loss = - self.elbo(batch_X, batch_Y, batch_m, output_idx, coeff_exp_log_lik, beta)
                    loss.backward()

                    if max_norm is not None:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

                        if (epoch + 1) % print_epochs == 0 and total_grad_norm.item() > max_norm:
                            print(
                                f"Gradient norm {total_grad_norm.item():.3f} exceeds the threshold {max_norm:.3f}, clipping applied."
                            )

                    optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    @torch.no_grad()
    def predict(
        self, x_star: Tensor, output_idx: Optional[LongTensor] = None, device="cpu", noiseless: bool = False
    ):
        """
        x_star: [..., n_test, D_X]
        """
        if noiseless:
            assert self.lik_model_type == "Gaussian", "Only Gaussian Likelihood support noiseless prediction."

        self.to(device)
        self.eval()
        x_star = x_star.to(device)

        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device)

        qf_means, qf_covs = self.variational_f(x_star, output_idx)  # [..., P, n_test]
        qf_means, qf_covs = qf_means.mT, qf_covs.mT  # [..., n_test, P]

        if self.lik_model_type == "Gaussian":
            # whether or not pass through likelihood, i.e., noiseless or noisy prediction.
            if noiseless:
                return qf_means, qf_covs  # [..., n_test, P]
            else:
                qy_covs = qf_covs + self.lik_model.sigma.square()
                return qf_means, qy_covs  # [..., n_test, P]

        elif self.lik_model_type == "NegativeBinomial":
            # must pass through likelihood.
            qy_means, qy_vars = self.lik_model.predict(qf_means, qf_covs, output_idx)
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
            name="graphical_mogp_base", input_batch_size=input_batch_size, output_batch_size=output_batch_size, device=device
        )

        return qy_means, qy_vars  # [..., n_test, P]

    @torch.no_grad()
    def predict_gmogp_gaussian(
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