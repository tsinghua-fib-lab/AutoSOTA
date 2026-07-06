from typing import Optional, Union

import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal

from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.utils.transforms import inv_softplus

from models.building_blocks.flow import ArcsinhFlow

__all__ = ["NegativeBinomialLikelihood"]


class NegativeBinomialLikelihood(nn.Module):
    """
    Zero-inflated Negative Binomial distribution.
    TODO: support batch version
    """
    def __init__(self, k_m: float, num_outputs: int, scale_factor: float = 1., alpha_joint: bool = False, alpha_init: float = 1.):
        super().__init__()
        assert k_m >= 0
        assert num_outputs >= 1
        assert alpha_init > 0
        assert scale_factor > 0
        self.k_m = k_m
        self.num_outputs = num_outputs
        self.scale_factor = scale_factor
        self.alpha_joint = alpha_joint
        if self.alpha_joint:
            self.register_parameter(
                "raw_alpha",
                nn.Parameter(
                    inv_softplus(
                        torch.ones(num_outputs, dtype=torch.get_default_dtype()) * alpha_init
                    ), requires_grad=True
                )
            )
        else:
            self.register_buffer(
                'raw_alpha',
                inv_softplus(
                    torch.ones(num_outputs, dtype=torch.get_default_dtype()) * alpha_init
                )
            )

        self.inv_link_func: callable = nn.functional.softplus  # or torch.exp
        # self.inv_link_func = ArcsinhFlow(n_blocks=5, add_init_f0=True)
        self.gh_quad = GaussHermiteQuadrature1D()

    @property
    def alpha(self):
        return nn.functional.softplus(self.raw_alpha) # [P]

    def log_lik_given_f_value(self, f_value: Tensor, y: Tensor, output_idx: Tensor):
        """
        compute the log likelihood, given GP function value f and obs y.
        Used in two cases:
        (1) inside exp_log_lik during training. Because of GaussHermiteQuadrature, f_value: [num_locs..., <b*P], and y, output_idx: [<b*P].
        (2) nll metric computation. f_value, y, output_idx all of shape [..., (s), b, P], where (s) optionally stands for qH samples for dkl-lvmogp.
        """
        assert torch.all(y >= 0)
        assert y.shape == output_idx.shape
        y = y.to(torch.int32)

        # prepare
        pick_alpha = self.alpha[output_idx]
        r_nb = 1 / pick_alpha
        # Use exp or softplus to ensure the mean of the negative binomial is positive.
        m_nb = self.scale_factor * self.inv_link_func(f_value)

        # Add epsilon to m_nb to avoid division by zero
        m_nb_stable = m_nb + 1e-8

        # log prob of NB distribution
        # Using numerically stable implementation:
        # p = r / (r+m) => log(p) = -log1p(m/r)
        # 1-p = m / (r+m) => log(1-p) = -log1p(r/m)
        log_p = -torch.log1p(m_nb_stable / r_nb)
        log_1_minus_p = -torch.log1p(r_nb / m_nb_stable)

        log_prob_nb = (
            torch.lgamma(y + r_nb) - torch.lgamma(r_nb) - torch.lgamma(y + 1)
            + r_nb * log_p + y * log_1_minus_p
        ) # in math, logNB(y | m, alpha)

        if self.k_m == 0:  # No zero-inflation
            return log_prob_nb

        # Zero-inflated part
        # psi = k_m / (k_m + m_nb)
        # 1 - psi = m_nb / (k_m + m_nb)
        # P(y=0) = psi + (1-psi) * P_NB(y=0)
        # P(y>0) = (1-psi) * P_NB(y>0)
        log_k_m = torch.log(torch.tensor(self.k_m, device=f_value.device, dtype=f_value.dtype))
        log_m_nb = torch.log(m_nb_stable)
        log_k_m_plus_m_nb = torch.logaddexp(log_k_m, log_m_nb)

        # log lik for y > 0
        log_1_minus_psi = log_m_nb - log_k_m_plus_m_nb
        log_lik_2 = log_1_minus_psi + log_prob_nb

        # log lik for y = 0
        log_prob_nb_0 = r_nb * log_p
        # log(k_m + m_nb * P_NB(0)) - log(k_m + m_nb)
        log_lik_1 = (
            torch.logaddexp(log_k_m, log_m_nb + log_prob_nb_0)
            - log_k_m_plus_m_nb
        )

        log_lik = torch.where(y == 0, log_lik_1, log_lik_2)

        return log_lik

    def exp_log_lik(self, qf_mean: Tensor, qf_var: Tensor, y: Tensor, output_idx: Tensor, method: str = "gauss_hermite"):
        """
        Compute the expected log likelihood using
            (1) Monte-Carlo approximation.
            (2) Gauss-Hermite Quadrature.
        Note: qf_mean, qf_var, y, output_idx: [..., <b*P]
        """
        if method == "monte_carlo":
            temp_log_lik, n_mc_samples = 0, 1
            for i in range(n_mc_samples): # Monte Carlo with n_mc_samples samples
                f_value = qf_mean + torch.sqrt(qf_var) * torch.randn_like(qf_mean)
                temp_log_lik += self.log_lik_given_f_value(f_value, y, output_idx)
            log_lik = temp_log_lik / n_mc_samples  # [..., <b*P]
            return log_lik   # [..., <b*P]

        elif method == "gauss_hermite":
            qf = Normal(loc = qf_mean, scale = qf_var.sqrt())

            log_lik = self.gh_quad.forward(
                lambda f_value: self.log_lik_given_f_value(f_value, y, output_idx), qf,
            )
            return log_lik  # [..., <b*P]

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'monte_carlo' or 'gauss-hermite'."
            )

    @torch.no_grad()
    def predict(self, qf_means: Tensor, qf_covs: Tensor, output_idx: Optional[Tensor] = None, num_mc: int = 20):
        """
        num_mc: int, number of Monte Carlo samples for the prediction.
        qf_means, qf_covs: [..., (s), n_test, P],
            where optionally s (for dkl_lvmogp) is the number of samples for qH,
        output_idx: [P]
        """
        # Add an MC dimension (size=num_mc) just before n_test
        eps = torch.randn(
            *qf_means.shape[:-2], num_mc, *qf_means.shape[-2:],
            device=qf_means.device, dtype=qf_means.dtype
        )   # [..., (s), num_mc, n_test, P]

        f_value = qf_means.unsqueeze(-3) + torch.sqrt(qf_covs).unsqueeze(-3) * eps # [..., (s), num_mc, n_test, P]
        m_nb = self.scale_factor * self.inv_link_func(f_value)                     # [..., (s), num_mc, n_test, P]
        psi = self.k_m / (self.k_m + m_nb)                                         # [..., (s), num_mc, n_test, P]

        py_means_mc = (1 - psi) * m_nb  # [..., (s), num_mc, n_test, P]
        pick_alpha = self.alpha if output_idx is None else self.alpha[output_idx]
        expanded_alpha = pick_alpha.view(*([1] * (py_means_mc.ndim - 1)), -1)
        py_vars_mc = py_means_mc * (1 + m_nb * (psi + expanded_alpha)) # [..., (s), num_mc, n_test, P]

        # average over the MC dim (-3)
        py_means = py_means_mc.mean(dim=-3)  # [..., (s), n_test, P]
        # the law of total variance
        py_vars = py_vars_mc.mean(dim=-3) + py_means_mc.var(dim=(-3), unbiased=False)

        return py_means, py_vars # [..., (s), n_test, P]

    @torch.no_grad()
    def predict_by_batch(
            self, qf_means: Tensor, qf_covs: Tensor, output_idx: Optional[Tensor] = None, num_mc: int = 20,
            input_batch_size: int = 64, output_batch_size: int = 32
    ):
        """
        For large scale dataset, mini-batch prediction to save memory.
        qf_means, qf_covs: [..., (s), n_test, P]
            where optionally s (for dkl_lvmogp) is the number of samples for qH,
        output_idx: [P] or None
        """
        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=qf_means.device)  # [P]
            assert qf_means.size(-1) == self.num_outputs

        n_test = qf_means.size(-2)
        input_idx = torch.arange(n_test, device=qf_means.device)

        input_chunks = torch.split(input_idx, input_batch_size)
        output_chunks = torch.split(output_idx, output_batch_size)

        list_means, list_vars = [], []  # mini-batch across outputs
        for output_chunk in output_chunks:
            tmp_list_means, tmp_list_vars = [], []  # mini-batch across inputs
            _qf_means = torch.index_select(qf_means, -1, output_chunk)  # [..., (s), n_test, P_chunk]
            _qf_covs = torch.index_select(qf_covs, -1, output_chunk)  # [..., (s), n_test, P_chunk]
            for input_chunk in input_chunks:
                tmp_qf_means = torch.index_select(_qf_means, -2, input_chunk)  # [..., (s), n_input_chunk, P_chunk]
                tmp_qf_covs = torch.index_select(_qf_covs, -2, input_chunk)  # [..., (s), n_input_chunk, P_chunk]
                tmp_py_means, tmp_py_vars = self.predict(tmp_qf_means, tmp_qf_covs, output_chunk, num_mc=num_mc)  # [..., (s), n_input_chunk, P_chunk]
                tmp_list_means.append(tmp_py_means)
                tmp_list_vars.append(tmp_py_vars)
            tmp_means = torch.cat(tmp_list_means, dim=-2)
            tmp_vars = torch.cat(tmp_list_vars, dim=-2)

            list_means.append(tmp_means)
            list_vars.append(tmp_vars)

        qy_means = torch.cat(list_means, dim=-1)
        qy_vars = torch.cat(list_vars, dim=-1)

        return qy_means, qy_vars  # [..., (s), n_test, P]

    @torch.no_grad()
    def predict_log_lik_by_batch(
            self, pred_f: Tensor, y_star: Tensor, output_idx: Tensor, input_batch_size: int = 64, output_batch_size: int = 32
    ):
        """
        compute metric log P(y_star | pred_f) via mini-batch computation.
        pred_f, y_star, output_idx: [..., (s), n_star, P], where P is the number of selected outputs,
            and optionally s (for dkl_lvmogp) is the number of samples for qH.
        """
        assert pred_f.shape == y_star.shape == output_idx.shape

        n_test, P = pred_f.size(-2), pred_f.size(-1)
        tmp_input_idx = torch.arange(n_test, device=pred_f.device)
        tmp_output_idx = torch.arange(P, device=pred_f.device)

        input_chunks = torch.split(tmp_input_idx, input_batch_size)
        output_chunks = torch.split(tmp_output_idx, output_batch_size)

        list_log_lik = []  # mini-batch across outputs
        for output_chunk in output_chunks:
            tmp_list_log_lik = []
            _pred_f = torch.index_select(pred_f, -1, output_chunk)  # [..., (s), n_star, P_chunk]
            _y_star = torch.index_select(y_star, -1, output_chunk)  # [..., (s), n_star, P_chunk]
            _output_idx = torch.index_select(output_idx, -1, output_chunk)  # [..., (s), n_star, P_chunk]
            for input_chunk in input_chunks:
                tmp_pred_f = torch.index_select(_pred_f, -2, input_chunk)  # [..., (s), n_input_chunk, P_chunk]
                tmp_y_star = torch.index_select(_y_star, -2, input_chunk)  # [..., (s), n_input_chunk, P_chunk]
                tmp_output_idx = torch.index_select(_output_idx, -2, input_chunk)  # [..., (s), n_input_chunk, P_chunk]
                log_lik = self.log_lik_given_f_value(tmp_pred_f, tmp_y_star, tmp_output_idx)  # [..., (s), n_input_chunk, P_chunk]
                tmp_list_log_lik.append(log_lik)

            tmp_log_lik = torch.cat(tmp_list_log_lik, dim=-2)  # [..., (s), n_star, P_chunk]
            list_log_lik.append(tmp_log_lik)

        log_lik = torch.cat(list_log_lik, dim=-1)  # [..., (s), n_star, P]

        return log_lik  # [..., (s), n_star, P]

if __name__ == "__main__":
    # test
    torch.set_default_dtype(torch.float64)

    # Setup
    n_data = 10
    n_outputs = 3
    likelihood = NegativeBinomialLikelihood(k_m=0.5, num_outputs=n_outputs, alpha_joint=False)
    f_value = torch.randn(n_data,)
    y = torch.randint(low=0, high=5, size=(n_data,))
    output_idx = torch.randint(low=0, high=n_outputs - 1, size=(n_data,))

    # Test log_lik_given_f_value
    print("Testing log_lik_given_f_value...")
    log_lik = likelihood.log_lik_given_f_value(f_value, y, output_idx)
    print("f_value", f_value)
    print("y", y)
    print("log_lik", log_lik)
    print("-" * 20)

    # Test exp_log_lik
    print("Testing exp_log_lik...")
    qf_mean = torch.randn(n_data)
    qf_var = torch.randn(n_data).square()  # Positive variance

    ## Gauss-Hermite
    exp_log_lik_gh = likelihood.exp_log_lik(qf_mean, qf_var, y, output_idx, method="gauss_hermite")
    print("Gauss-hermite Quadrature:", exp_log_lik_gh)

    ## Monte Carlo
    exp_log_lik_mc = likelihood.exp_log_lik(qf_mean, qf_var, y, output_idx, method="monte_carlo")
    print("Monte-Carlo:", exp_log_lik_mc)
    print("-" * 20)

