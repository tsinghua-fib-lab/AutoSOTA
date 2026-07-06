import copy
import warnings
from typing import Optional

import torch
from torch import Tensor, BoolTensor, LongTensor
from torch import nn
from torch.optim import Optimizer
from torch.distributions import MultivariateNormal, kl_divergence
from torch.utils.data import DataLoader

from linear_operator.utils.cholesky import psd_safe_cholesky

from utils.build_datasets import IndexDataset
from utils.metrics import mc_gaussian_nll
from utils.helpers import wrap_func_by_batch
from models.building_blocks.gp_modules import Prior_H as _Prior_H
from models.building_blocks.gp_modules import Variational_H as _Variational_H
from models.building_blocks.gp_modules import Inducing_points as _Inducing_points
from likelihood.gaussian import GaussianLikelihood
from likelihood.negative_binomial import NegativeBinomialLikelihood


__all__ = [
    "gs_lvmogp_base",
    "gs_Prior_H",
    "gs_Variational_H",
    "gs_Variational_inducing_dist",
    "gs_Inducing_points"
]


class gs_Prior_H(_Prior_H):
    """p(H), fully factorized prior for gs_lvmogp"""
    def __init__(self, Q: int, mean_pH: Tensor, diag_cov_pH: Tensor):
        assert mean_pH.size(-3) == diag_cov_pH.size(-3) == Q
        super(gs_Prior_H, self).__init__(mean_pH, diag_cov_pH)


class gs_Variational_H(_Variational_H):
    """q(H) for gs_lvmogp"""
    def  __init__(self, Q: int, P: int, D_H: int, batch_shape: tuple = (), mean_field: bool = False):
       batch_shape = batch_shape + (Q, )
       super(gs_Variational_H, self).__init__(P, D_H, batch_shape, mean_field)


class gs_Variational_inducing_dist(nn.Module):
    """q(U) for gs_lvmogp, over both H and X"""
    def __init__(self, M_H: int, M_X: int, batch_shape: tuple = ()):
        super(gs_Variational_inducing_dist, self).__init__()
        self.M_H = M_H
        self.M_X = M_X
        self.batch_shape = batch_shape
        mean_qU_shape = batch_shape + (int(M_H * M_X), )
        self.register_parameter(
            "mean_qU",
            nn.Parameter(torch.zeros(mean_qU_shape, dtype=torch.get_default_dtype()), requires_grad=True)
        )
        self.register_parameter(
            "factor_cov_qU_H",
            nn.Parameter(torch.eye(M_H, dtype=torch.get_default_dtype()).repeat(*batch_shape, 1, 1), requires_grad=True)
        )
        self.register_parameter(
            "factor_cov_qU_X",
            nn.Parameter(torch.eye(M_X, dtype=torch.get_default_dtype()).repeat(*batch_shape, 1, 1), requires_grad=True)
        )

    @property
    def cov_qU_H(self):   # [..., M_H, M_H]
        return self.factor_cov_qU_H @ self.factor_cov_qU_H.mT

    @property
    def cov_qU_X(self):   # [..., M_X, M_X]
        return self.factor_cov_qU_X @ self.factor_cov_qU_X.mT

    @property
    def cov_qU(self):
        _cov_qU = torch.einsum('...ij,...kl->...ikjl', self.cov_qU_H, self.cov_qU_X)  # [..., M_H, M_X, M_H, M_X]
        cov_qU = _cov_qU.view(*self.batch_shape, self.M_H * self.M_X, self.M_H * self.M_X)  # [..., M_H * M_X, M_H * M_X]
        return cov_qU   # [..., M_H * M_X, M_H * M_X]


class gs_Inducing_points(_Inducing_points):
    """
    Z_H or Z_X, inducing points/locations for gs_lvmogp
    NOTE: Z_H has extra dim Q at -3, Z_X does not.
    """
    def __init__(self, M: int, num_dims: int, IP_init: Tensor, IP_name: str, IP_joint: bool = True):
        assert IP_init.shape[-2:] == torch.Size([M, num_dims])
        super(gs_Inducing_points, self).__init__(
            M=M, num_dims=num_dims, IP_init=IP_init, IP_name=IP_name, IP_joint=IP_joint
        )


class gs_lvmogp_base(nn.Module):
    """
    GS Latent Variable MOGP.

    Notations:
    Q: number of coregionalization matrices
    D_X: input dims
    D_H: latent variable dims
    M_X: num of inducing variables in input space
    M_H: num of inducing variables in latent variable space
    P: number of outputs
    """
    def __init__(
        self, input_kernels: list, latent_kernels: list, Q: int, pH, qH, qU, zH, zX,
        lik_model: dict = {"type": "Gaussian", "sigma_joint": False, "sigma_init": 0.5},
        # lik_model={"type": "NegativeBinomial", "k_m": 0.1, "scale_factor": 1., "alpha_joint": False, "alpha_init": 1.}
        whitening=True, jitter=1e-6,
    ):
        # check multi_output
        for input_kernel in input_kernels:
            assert not input_kernel.multi_output

        for latent_kernel in latent_kernels:
            assert not latent_kernel.multi_output

        super(gs_lvmogp_base, self).__init__()

        self.Q = Q
        self.pH = pH
        self.qH = qH
        self.qU = qU
        self.zH = zH  # inducing points, latent space
        self.zX = zX  # inducing points, input space
        self.lik_model_type = lik_model["type"]
        self.num_outputs = int(self.pH.mean_pH.size(-2))
        self.M_X, self.M_H = int(self.zX.M), int(self.zH.M)
        self.batch_shape = qU.batch_shape

        # NOTE: kernels should be registered after self.Q
        self.input_kernels = self._check_list_of_kernels(input_kernels)
        self.latent_kernels = self._check_list_of_kernels(latent_kernels)

        self.whitening = whitening
        self.jitter = jitter

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

    def _eval_K_input(self, x: Tensor, y: Tensor, diag: bool = False):
        """x: [..., b1, D_X]; y: [..., b2, D_X]"""
        if diag:
            assert x is y, "diag=True only supports x and y being the same tensor object."

        list_K_input = [
            self.input_kernels[q].forward(x, y, diag=diag) for q in range(self.Q)
        ]

        Q_K_input = torch.stack(list_K_input, dim=-2) if diag else torch.stack(list_K_input, dim=-3)
        return Q_K_input  # [..., Q, b1=b2] or [..., Q, b1, b2]

    def _eval_K_latent(self, x: Tensor, y: Tensor, diag: bool = False):
        """x: [..., Q, b1, D_H]; y:[..., Q, b2, D_H]"""
        if diag:
            assert torch.all(x == y), "diag=True only supports x==y"

        list_K_latent = [
            self.latent_kernels[q].forward(x[..., q, :, :], y[..., q, :, :], diag=diag) for q in range(self.Q)
        ]

        Q_K_latent = torch.stack(list_K_latent, dim=-2) if diag else torch.stack(list_K_latent, dim=-3)
        return Q_K_latent  # [..., Q, b1=b2] or [..., Q, b1, b2]

    def variational_f(self, x: Tensor, H: Tensor):
        r"""
        q(f) = \int p(f|U) q(U) dU
        x: [..., b, D_X];
        H: [... ,Q, P, D_H]
        """
        assert self.whitening

        b, P = x.size(-2), H.size(-2)

        # prepare - K_ff (diagonal)
        Q_K_ff_input = self._eval_K_input(x, x, diag=True)   # [..., Q, b]
        Q_K_ff_latent = self._eval_K_latent(H, H, diag=True)   # [..., Q, P]

        Q_K_ff = torch.einsum('...qi,...qk->...qik', Q_K_ff_latent, Q_K_ff_input)   # [..., Q, P, b]
        Q_K_ff = Q_K_ff.view(*self.batch_shape, self.Q, P * b)   # [..., Q, P * b]
        K_ff = Q_K_ff.sum(dim=-2)   # [..., P * b]

        # unit test
        # test_Q_K_ff_input = self._eval_K_input(x, x, diag=False)   # [..., Q, b, b]
        # test_Q_K_ff_latent = self._eval_K_latent(H, H, diag=False)   # [..., Q, P, P]
        # test_sum_kron_products_1 = 0.
        # for q in range(self.Q):
        #     curr_test_Q_K_ff_input = test_Q_K_ff_input[..., q, :, :].unsqueeze(-3).unsqueeze(-2)   # [..., 1, b, 1, b]
        #     curr_test_Q_K_ff_latent = test_Q_K_ff_latent[..., q, :, :].unsqueeze(-2).unsqueeze(-1)   # [..., P, 1, P, 1]
        #     test_sum_kron_products_1 += curr_test_Q_K_ff_latent * curr_test_Q_K_ff_input   # [..., P, b, P, b]
        # test_sum_kron_products_1 = test_sum_kron_products_1.view(*self.batch_shape, P * b, P * b)    # [..., P * b, P * b]
        # assert torch.allclose(K_ff, test_sum_kron_products_1.diagonal(dim1=-2, dim2=-1))

        # prepare - K_fu
        Q_K_fu_input = self._eval_K_input(x, self.zX.inducing_points, diag=False)   # [..., Q, b, M_X]
        Q_K_fu_latent = self._eval_K_latent(H, self.zH.inducing_points, diag=False)   # [..., Q, P, M_H]
        Q_K_fu = torch.einsum('...qij,...qkl->...qikjl', Q_K_fu_latent, Q_K_fu_input)   # [..., Q, P, b, M_H, M_X]
        Q_K_fu = Q_K_fu.view(
            *self.batch_shape, self.Q, P * b, int(self.M_H * self.M_X)
        )   # [..., Q, P * b, M_H * M_X]
        K_fu = Q_K_fu.sum(dim=-3)   # [..., P * b, M_H * M_X]

        # unit test
        # test_sum_kron_products_2 = 0.
        # for q in range(self.Q):
        #     curr_Q_K_fu_latent = Q_K_fu_latent[..., q, :, :].unsqueeze(-2).unsqueeze(-1)   # [..., P, 1, M_H, 1]
        #     curr_Q_K_fu_input = Q_K_fu_input[..., q, :, :].unsqueeze(-3).unsqueeze(-2)   # [..., 1, b, 1, M_X]
        #     test_sum_kron_products_2 += curr_Q_K_fu_latent * curr_Q_K_fu_input   # [..., P, b, M_H, M_X]
        # test_sum_kron_products_2 = test_sum_kron_products_2.view(*self.batch_shape, P * b, self.M_H * self.M_X)  # [..., P * b, M_H * M_X]
        # assert torch.allclose(K_fu, test_sum_kron_products_2)

        # prepare - K_uu
        Q_K_uu_input = self._eval_K_input(self.zX.inducing_points, self.zX.inducing_points)  # [..., Q, M_X, M_X]
        Q_K_uu_latent = self._eval_K_latent(self.zH.inducing_points, self.zH.inducing_points)  # [..., Q, M_H, M_H]

        ## Cholesky factor of K_uu
        # Case1: Q=1 (take advantage of kronecker product structure)
        if self.Q == 1:
            L_uu_input = psd_safe_cholesky(
                Q_K_uu_input.squeeze(-3) + self.jitter * torch.eye(self.M_X, dtype=torch.get_default_dtype(), device=Q_K_uu_input.device)
            )   # [..., M_X, M_X]

            L_uu_latent = psd_safe_cholesky(
                Q_K_uu_latent.squeeze(-3) + self.jitter * torch.eye(self.M_H, dtype=torch.get_default_dtype(), device=Q_K_uu_latent.device)
            )   # [..., M_H, M_H]

            _L_uu = torch.einsum('...ij,...kl->...ikjl', L_uu_latent, L_uu_input)  # [..., M_H, M_X, M_H, M_X]
            L_uu = _L_uu.view(*self.batch_shape, self.M_H * self.M_X, self.M_H * self.M_X)  # [..., M_H * M_X, M_H * M_X]

        # TODO: for Q=1, have better ways to compute L_uu_inv_K_uf

        # Case2: Q>1
        else:
            Q_K_uu = torch.einsum('...qij,...qkl->...qikjl', Q_K_uu_latent, Q_K_uu_input)
            Q_K_uu = Q_K_uu.view(*self.batch_shape, self.Q, self.M_H * self.M_X, self.M_H * self.M_X)  # [..., Q, M_H * M_X, M_H * M_X]
            K_uu = Q_K_uu.sum(dim=-3)   # [..., M_H * M_X, M_H * M_X]

            # unit test
            # test_sum_kron_products_3 = 0.
            # for q in range(self.Q):
            #     curr_test_Q_K_uu_input = Q_K_uu_input[..., q, :, :].unsqueeze(-3).unsqueeze(-2)    # [..., 1, M_X, 1, M_X]
            #     curr_test_Q_K_uu_latent = Q_K_uu_latent[..., q, :, :].unsqueeze(-2).unsqueeze(-1)  # [..., M_H, 1, M_H, 1]
            #     test_sum_kron_products_3 += curr_test_Q_K_uu_latent * curr_test_Q_K_uu_input  # [..., M_H, M_X, M_H, M_X]
            # test_sum_kron_products_3 = test_sum_kron_products_3.view(*self.batch_shape, self.M_H * self.M_X, self.M_H * self.M_X)
            # assert torch.allclose(K_uu, test_sum_kron_products_3)

            L_uu = psd_safe_cholesky(K_uu + self.jitter * torch.eye(self.M_H * self.M_X, dtype=torch.get_default_dtype(), device=K_uu.device))   # [..., M_H * M_X, M_H * M_X]

        L_uu_inv_K_uf = torch.linalg.solve_triangular(L_uu, K_fu.mT, upper=False)   # [..., M_H * M_X, P * b]

        qf_mean = (L_uu_inv_K_uf.mT @ self.qU.mean_qU.unsqueeze(-1)).squeeze(-1)   # [..., P * b]

        tmp = torch.einsum(
            '...ji,...jk,...ki->...i',
            L_uu_inv_K_uf,   # [..., M_H * M_X, P * b]
            (self.qU.cov_qU - torch.eye(self.qU.cov_qU.size(-1), dtype=torch.get_default_dtype(), device=self.qU.cov_qU.device)),   # [..., M_H * M_X, M_H * M_X]
            L_uu_inv_K_uf,   # [..., M_H * M_X, P * b]
        )

        # unit test
        # tmp2 = L_uu_inv_K_uf.mT @ (self.qU.cov_qU - torch.eye(self.qU.cov_qU.size(-1), dtype=torch.get_default_dtype(), device=self.qU.cov_qU.device)) @ L_uu_inv_K_uf
        # assert torch.allclose(tmp, tmp2.diagonal(dim1=-2, dim2=-1), atol=1e-5)

        qf_cov = (K_ff + tmp + self.jitter)

        return qf_mean, qf_cov   # [..., P * b], [..., P * b]

    @property
    def KL_qU_pU(self):
        """
        KL divergence term between q(U) and p(U)
        """
        assert self.whitening
        # Cholesky of variational cov matrices
        chol_cov_H = psd_safe_cholesky(
            self.qU.cov_qU_H + self.jitter * torch.eye(self.M_H, dtype=torch.get_default_dtype(), device=self.qU.cov_qU_H.device)
        )   # [..., M_H, M_H]

        chol_cov_X = psd_safe_cholesky(
            self.qU.cov_qU_X + self.jitter * torch.eye(self.M_X, dtype=torch.get_default_dtype(), device=self.qU.cov_qU_X.device)
        )   # [..., M_X, M_X]

        trace_H = chol_cov_H.square().sum(dim=(-1, -2))   # [...]
        trace_X = chol_cov_X.square().sum(dim=(-1, -2))   # [...]
        half_log_det_H = torch.diagonal(chol_cov_H, dim1=-1, dim2=-2).log().sum(dim=(-1))   # [...]
        half_log_det_X = torch.diagonal(chol_cov_X, dim1=-1, dim2=-2).log().sum(dim=(-1))   # [...]

        m_T_m = self.qU.mean_qU.square().sum(dim=(-1))   # [...]
        KL = 0.5 * (trace_H * trace_X - self.M_H * self.M_X + m_T_m) - self.M_H * half_log_det_X - self.M_X * half_log_det_H

        # unit test
        # q(U)
        # qU = MultivariateNormal(self.qU.mean_qU, self.qU.cov_qU)

        # p(U)
        # pU = MultivariateNormal(
        #     torch.zeros_like(self.qU.mean_qU, dtype=torch.get_default_dtype(), device=self.qU.mean_qU.device),
        #     torch.eye(
        #         self.M_H * self.M_X, dtype=torch.get_default_dtype(), device=self.qU.cov_qU.device
        #     ).repeat(*self.qU.batch_shape, 1, 1)
        # )
        # KL_2 = kl_divergence(qU, pU)   # [...]

        # Not exactly equal due to jitter addition for my own implementation.
        # print(f"My implemented KL(qU || pU) is: {KL}, PyTorch KL is: {KL_2}.")

        return KL   # [...]

    def KL_qH_pH(self, output_idx: Optional[LongTensor] = None):
        """
        mini-batch approximation for (per-output) KL between q(H) and p(H), p(H) is fully factorized but q(H) might not.
        """
        # select outputs
        if output_idx is not None:
            mean_pH, mean_qH = self.pH.mean_pH[..., output_idx, :], self.qH.mean_qH[..., output_idx, :]
            cov_pH = self.pH.diag_cov_pH[..., output_idx, :]  # [..., Q, P, D_H], P refers to the size of selected outputs
            if self.qH.mean_field:
                cov_qH = self.qH.cov_qH[..., output_idx, :]  # [..., Q, P, D_H]
            else:
                cov_qH = self.qH.cov_qH[..., output_idx, :, :]  # [..., Q, P, D_H, D_H]
        else:
            mean_pH, mean_qH = self.pH.mean_pH, self.qH.mean_qH
            cov_pH, cov_qH = self.pH.diag_cov_pH, self.qH.cov_qH

        # compute KL values
        if self.qH.mean_field:
            term1 = cov_pH.log() - cov_qH.log()
            term2 = (cov_qH + (mean_qH - mean_pH).pow(2)) / cov_pH
            _KLs = 0.5 * (term1 + term2 - 1.)  # [..., Q, P, D_H]
            KLs = _KLs.sum(dim=(-3, -1)).mean(dim=(-1))  # [...], sum over Q, D_H, average over P

            ## unit test
            # qH = MultivariateNormal(mean_qH, torch.diag_embed(cov_qH))
            # pH = MultivariateNormal(mean_pH, torch.diag_embed(cov_pH))
            # KLs_2 = kl_divergence(qH, pH).sum(dim=(-2)).mean(dim=(-1))  # [..., Q, P] -> [...]

            # print(f"My implemented KL(qH || pH) is: {KLs}, PyTorch KL is: {KLs_2}.")

        else:
            D_H = cov_qH.size(-1)
            chol_cov_qH = psd_safe_cholesky(
                cov_qH + self.jitter * torch.eye(D_H, dtype=torch.get_default_dtype(), device=cov_qH.device)
            )  # [..., Q, P, D_H, D_H]
            std_pH = cov_pH.sqrt()  # [..., Q, P, D_H]
            trace = (chol_cov_qH / std_pH.unsqueeze(-1)).square().sum(dim=(-1, -2))  # [..., Q, P]
            mahalanobis = ((mean_pH - mean_qH) / std_pH).square().sum(dim=(-1))  # [..., Q, P]
            log_det_cov_pH = cov_pH.log().sum(dim=(-1))  # [..., Q, P]
            log_det_cov_qH = 2 * torch.diagonal(chol_cov_qH, dim1=-1, dim2=-2).log().sum(dim=(-1))  # [..., Q, P]
            _KLs = 0.5 * (trace - D_H + mahalanobis + log_det_cov_pH - log_det_cov_qH)  # [..., Q, P]
            KLs = _KLs.sum(dim=(-2)).mean(dim=(-1))  # [...], sum over Q, average over P

            ## unit test
            # qH = MultivariateNormal(mean_qH, cov_qH)
            # pH = MultivariateNormal(mean_pH, torch.diag_embed(cov_pH))
            # _KLs_2 = kl_divergence(qH, pH)   # [..., Q, P]
            # KLs_2 = _KLs_2.sum(dim=(-2)).mean(dim=(-1))   # [..., Q, P] -> [...] sum over Q, average over P

            # Not exactly equal due to jitter addition for my own implementation.
            # print(f"My implemented KL(qH || pH) is: {KLs}, PyTorch KL is: {KLs_2}.")

        return KLs   # [...]

    def elbo(
        self, x: Tensor, y: Tensor, m: BoolTensor, output_idx: LongTensor,
        coeff_exp_log_lik: float, beta_u=1., beta_h=1., average_elbo=False,
    ):
        """
        mini-batch elbo, b: mini-batch size
        x: [..., b, D_X] i.e. xs are shared across output
        y: [..., b, P], P is the size of the subset of all outputs
        m: [..., b, P], where 0 indicate missing
        output_idx: [P]
        """
        H = self.qH.sample(output_idx)   # [..., Q, P, D_H]

        qf_mean, qf_cov = self.variational_f(x, H)   # [..., P * b]

        # term 1/3 - exp_log_lik
        # TODO: mask before compute
        y_T, m_T = y.mT, m.mT.bool()   # [..., P, b]
        m_T_flatten = m_T.flatten(start_dim=-2, end_dim=-1)   # [..., P * b]

        if torch.all(m_T_flatten.sum(dim=(-1)) > 0):
            pick_qf_mean = qf_mean[m_T_flatten].view(*self.batch_shape, -1)   # [..., <P*b]
            pick_qf_cov = qf_cov[m_T_flatten].view(*self.batch_shape, -1)     # [..., <P*b]
            pick_y = y_T[m_T].view(*self.batch_shape, -1)   # [..., <P*b]

            if self.lik_model_type == "Gaussian":
                assert isinstance(self.lik_model, GaussianLikelihood)
                _exp_log_lik = self.lik_model.exp_log_lik(
                    qf_mean=pick_qf_mean, qf_var=pick_qf_cov, y=pick_y
                )  # [..., <b*P]

                # _exp_log_lik = (
                #     - ((pick_qf_mean - pick_y).square() + pick_qf_cov) / (2 * self.sigma.pow(2))
                #     - torch.log(self.sigma * math.sqrt(2 * torch.pi))
                # )  # [..., <b*P]

                exp_log_lik = _exp_log_lik.mean(dim=(-1))   # [...], average over <P*b

            elif self.lik_model_type == "NegativeBinomial":
                assert isinstance(self.lik_model, NegativeBinomialLikelihood)
                # pick_output_idx = torch.masked_select(
                #     output_idx.view(*([1] * (m_T.ndim - 1)), -1),  # [P] -> [...,1,P]
                #     m_T.bool()
                # ).view(*self.batch_shape, -1)  # [..., <P*b]

                expanded_idx = output_idx.view(*([1] * (m.ndim - 1)), -1).expand_as(m)  # [..., b, P]
                expanded_idx_T = expanded_idx.mT  # [..., P, b]
                pick_output_idx = expanded_idx_T[m_T].view(*self.batch_shape, -1)  # [..., <P*b]

                _exp_log_lik = self.lik_model.exp_log_lik(
                    qf_mean=pick_qf_mean, qf_var=pick_qf_cov, y=pick_y, output_idx=pick_output_idx
                )  # [..., <b*P]

                exp_log_lik = _exp_log_lik.mean(dim=(-1))  # [...], average over <b*P

            else:
                raise NotImplementedError(f"Likelihood model {self.lik_model_type} is not implemented for exp_log_lik!")

        else:
            warnings.warn("Encounter one empty mini-batch!")
            exp_log_lik = 0.

        # term 2/3 - KL(q(U)||p(U))
        KL_qU_pU = self.KL_qU_pU

        # term 3/3 - KL(q(H)||p(H))
        KL_qH_pH = self.KL_qH_pH(output_idx)

        # sum elbo over (extra) batch dims
        elbo = (
            coeff_exp_log_lik * exp_log_lik
            - beta_u * KL_qU_pU
            - beta_h * self.num_outputs * KL_qH_pH
        ).sum()

        if average_elbo:
            elbo = elbo / coeff_exp_log_lik

        return elbo

    @torch.no_grad()
    def predict(
        self, x_star: Tensor, output_idx: Optional[LongTensor] = None, num_samples: int = 1,
        device="cpu", noiseless: bool = False
    ):
        """
        x_star: [...,  n_test, D_X]
        Get predictive mean and var for output_idx on x_star.
        If output_idx is None, then make predictions for all outputs.
        """
        self.to(device)
        self.eval()
        x_star = x_star.to(device)

        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device)

        n_test, P = x_star.size(-2), len(output_idx)

        qf_means, qf_covs = [], []
        for i in range(num_samples):
            H_samples = self.qH.sample(output_idx)
            qf_mean, qf_cov = self.variational_f(x_star, H_samples)   # [..., P * n_test]
            qf_mean = qf_mean.view(*self.batch_shape, P, n_test)   # [..., P, n_test]
            qf_cov = qf_cov.view(*self.batch_shape, P, n_test)    # [..., P, n_test]
            qf_means.append(qf_mean)
            qf_covs.append(qf_cov)

        qf_means = torch.stack(qf_means, dim=-3).mT   # [..., s, n_test, P]
        qf_covs = torch.stack(qf_covs, dim=-3).mT     # [..., s, n_test, P]

        if noiseless:
            return qf_means, qf_covs

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
    def predict_given_H(
            self, x_star: Tensor, H_values: Tensor, num_samples: int = 1, pH_cov_value: Optional[float] = None,
            device="cpu", noiseless: bool = False
    ):
        """
        Make predictions at (new) outputs with given H values.
        x_star: [..., n_test, D_X]
        H_values: [..., P_test, D_H], P_test is the number of outputs to be predicted
        pH_cov_value: if not None, sample H from N(H_values, pH_cov_value*I) instead of using H_values directly.
        If noiseless is True, then return latent f predictions, otherwise return y predictions (i.e., passed through likelihood).
        """
        assert self.qH.mean_qH.size(-1) == H_values.size(-1), "H_values has unmatched dimensionality with qH!"
        self.to(device)
        self.eval()
        x_star = x_star.to(device)

        n_test, P_test = x_star.size(-2), H_values.size(-2)

        H_values = H_values.unsqueeze(-3).expand(*self.batch_shape, self.Q, P_test, H_values.size(-1))  # [..., Q, P_test, D_H]

        if pH_cov_value is None:
            qf_mean, qf_cov = self.variational_f(x_star, H_values)  # [..., P * n_test]
            qf_mean = qf_mean.view(*self.batch_shape, P_test, n_test).mT  # [..., P, n_test] -> [..., n_test, P]
            qf_cov = qf_cov.view(*self.batch_shape, P_test, n_test).mT   # [..., P, n_test] -> [..., n_test, P]
        else:
            assert pH_cov_value > 0., "pH_cov_value should be positive!"
            qf_means, qf_covs = [], []
            for i in range(num_samples):
                H_samples = H_values + pH_cov_value**0.5 * torch.rand_like(H_values)
                qf_mean, qf_cov = self.variational_f(x_star, H_samples)  # [..., P * n_test]
                qf_mean = qf_mean.view(*self.batch_shape, P_test, n_test)  # [..., P, n_test]
                qf_cov = qf_cov.view(*self.batch_shape, P_test, n_test)  # [..., P, n_test]
                qf_means.append(qf_mean)
                qf_covs.append(qf_cov)

            qf_means = torch.stack(qf_means, dim=-3).mT  # [..., s, n_test, P]
            qf_covs = torch.stack(qf_covs, dim=-3).mT  # [..., s, n_test, P]

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

    @torch.no_grad()
    def predict_by_batch(
            self, x_star: Tensor, output_idx: Optional[LongTensor] = None, num_samples: int = 1, device="cpu", noiseless: bool = False,
            input_batch_size: int = 64, output_batch_size: int = 32,
    ):
        """
        For large scale dataset (with large number of inputs and outputs), split the prediction into mini-batches.
        x_star: [..., n_test, D_X]
        """
        self.eval()
        if output_idx is None:
            output_idx = torch.arange(self.num_outputs, device=device) # [P]

        qy_means, qy_vars = wrap_func_by_batch(
            model=self, func_args={"x_star": x_star, "output_idx": output_idx, "num_samples": num_samples, "noiseless": noiseless},
            name="gs_lvmogp_base", input_batch_size=input_batch_size, output_batch_size=output_batch_size, device=device
        )

        return qy_means, qy_vars  # [..., n_test, P]

    def train_lvmogp(
        self, train_dataloader: DataLoader, output_batch_size: int, optimizer: Optimizer, epochs: int,
        beta_u=1., beta_h=1., coeff_exp_log_lik: Optional[float] = None, max_norm: Optional[float] = None,
        device="cpu", print_epochs=10
    ):
        self.to(device)
        self.train()
        if coeff_exp_log_lik is None:
            # biased if there are missing values
            coeff_exp_log_lik = len(train_dataloader.dataset) * self.num_outputs
        output_index_dataloader = None  # cache
        perm = None   # cache

        for epoch in range(epochs):
            for batch_X, batch_all_Y, batch_all_m in train_dataloader:   # [b, ..., D_X/P]

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
                        num_workers=0,
                    )

                for output_idx in output_index_dataloader:
                    output_idx = output_idx.to(device)
                    batch_Y = batch_all_Y[..., output_idx]   # [b, ..., p] where p<=P is the size of selected outputs
                    batch_m = batch_all_m[..., output_idx]   # [b, ..., p]

                    batch_Y, batch_m = batch_Y.permute(*perm), batch_m.permute(*perm)

                    optimizer.zero_grad(set_to_none=True)
                    loss = - self.elbo(batch_X, batch_Y, batch_m, output_idx, coeff_exp_log_lik, beta_u, beta_h)
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
    def predict_lvmogp_gaussian(
        self, data_dict,  num_samples: int = 10, noiseless: bool = False, num_plot_points: Optional[int] = 2000, device: str = "cpu"
    ):
        # Convention: on device
        assert self.lik_model_type == "Gaussian"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
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

        pick_train_Y = train_Y[train_mask.bool()].view(*self.batch_shape, -1)  # [..., <N*P]
        # TODO: only works when m has same number of True across batch dims
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

        if num_plot_points is None:
            return metric_dict, pred_dict, plot_pred_dict

        if all_X.size(-1) == 1:
            X_min, X_max = all_X.min().item(), all_X.max().item()
            denser_X = torch.linspace(X_min, X_max, num_plot_points, dtype=all_X.dtype, device=device).unsqueeze(-1)  # [n_plot, 1]
            if len(self.batch_shape) > 0.:
                denser_X = denser_X.view(*([1] * len(self.batch_shape)), num_plot_points, 1).expand(*self.batch_shape, num_plot_points, 1)  # [..., n_plot, 1]

            plot_pred_means, plot_pred_vars = self.predict_by_batch(
                denser_X, output_idx=None, num_samples=num_samples, device=device, noiseless=True,
                input_batch_size=128, output_batch_size=128,
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
            self, data_dict,  num_samples: int = 10, device: str = "cpu"
    ):
        assert self.lik_model_type == "NegativeBinomial"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device),
            data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )
        all_Y = train_Y + test_Y  # [..., N, P]

        # predict on all_X for metrics
        pred_means, pred_vars = self.predict_by_batch(
            all_X, output_idx=None, num_samples=num_samples, device=device, noiseless=True,
            input_batch_size=128, output_batch_size=128,
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
        )  # [..., s, N, P]
        all_mc_nb_ll = torch.logsumexp(log_lik, dim=-3) - torch.log(torch.tensor(num_samples))  # [..., N, P], logsumexp trick over qH samples

        # metric
        all_se = (all_Y - py_means).square()  # [..., N, P]

        # metrics: train/test split
        train_se = all_se[train_mask.bool()]  # [..., <N*P]
        test_se = all_se[test_mask.bool()]  # [..., <N*P]
        train_nll = - all_mc_nb_ll[train_mask.bool()]  # [..., <N*P]
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


    def _check_list_of_kernels(self, list_of_kernels):
        if len(list_of_kernels) == self.Q:
            return nn.ModuleList(list_of_kernels)
        else:
            assert len(list_of_kernels) == 1
            kernel = list_of_kernels[0]
            list_of_kernels = nn.ModuleList([copy.deepcopy(kernel) for _ in range(self.Q)])
            # list_of_kernels = nn.ModuleList(kernel.__class__() for _ in range(Q))
            return list_of_kernels
