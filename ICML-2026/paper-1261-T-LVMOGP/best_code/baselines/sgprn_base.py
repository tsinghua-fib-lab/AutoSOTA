import warnings
from typing import Optional, Union

import torch
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor, BoolTensor, LongTensor
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import tensorly as tl

from gpytorch.kernels import Kernel
# from kernels.rbf_kernel import MyRBFKernel

from utils.build_datasets import IndexDataset
from utils.metrics import gaussian_nll

from likelihood.gaussian import GaussianLikelihood
from likelihood.negative_binomial import NegativeBinomialLikelihood


# Set TensorLy backend to PyTorch
tl.set_backend('pytorch')  # Set TensorLy backend to PyTorch


class scalable_gprn_base(nn.Module):
    """
    Scalable Gaussian Process Regression Network (GPRN).

    This class is based on "Scalable Gaussian Process Regression Networks" (IJCAI 2020)
        and code implementation provided in https://github.com/shib0li/Scalable-GPRN.

    Comments:
        (1) We extend the original implementation to mini-batch training (across inputs).
        (2) Currently, we do not support mini-batching across outputs.
        (3) We do not support batched operations yet, i.e., the batch_shape of X_train must be empty.
        (4) We support two types of likelihood models: Gaussian and Negative Binomial. For NB, we use Monte Carlo.
        (5) The model hasn't been highly optimized yet, for instance, mask before compute to reduce computation and carefully reuse intermediate results.

    Notations:
    K: number of latent functions (rank)
    """
    def __init__(
        self, D_X: int, K: int, P: Union[int, list], X_train: Tensor, kernel_F: Kernel, kernel_W: Kernel,
        lik_model: dict = {"type": "Gaussian", "sigma_joint": True, "sigma_init": 0.5},
        # lik_model={"type": "NegativeBinomial", "k_m": 0.1, "scale_factor": 1., "alpha_joint": False, "alpha_init": 1.}
        jitter: float=1e-6,
    ):
        super(scalable_gprn_base, self).__init__()

        self.batch_shape = X_train.shape[:-2]
        assert kernel_F.batch_shape == self.batch_shape
        assert kernel_W.batch_shape == self.batch_shape

        if len(self.batch_shape) > 0:
            raise NotImplementedError(
                "Scalable GPRN does not support batched operations yet. :("
                "This is due to the use of tl.tenalg.multi_mode_dot()"
            )

        if isinstance(P, int):
            self.P_split = False  # no kronecker product structure for output space.
            self.P = P
            self.num_outputs = P  # duplicate here, for compatibility
        elif isinstance(P, list):
            assert len(P) == 2, "Only supports P as a single integer or a list of two integers."
            self.P_split = True
            self.P1, self.P2 = P[0], P[1]
            self.P = int(self.P1 * self.P2)
            self.num_outputs = self.P
        else:
            raise NotImplementedError

        self.D_X = D_X
        self.K = K
        self.X_train = X_train  # [..., N_train, D_X]
        self.N_train = X_train.size(-2)
        self.kernel_F = kernel_F
        self.kernel_W = kernel_W
        self.lik_model_type = lik_model["type"]
        self.jitter = jitter

        # mean parameters
        self.Fmu = nn.Parameter(
            torch.zeros(
                (self.batch_shape + (self.N_train, K)),  dtype=torch.get_default_dtype()
            ),
            requires_grad=True
        )  # [..., N_train, K]

        self.Wmu = nn.Parameter(
            torch.zeros(
                (self.batch_shape + (self.N_train, self.P, K)), dtype=torch.get_default_dtype()
            ),
            requires_grad=True
        )  # [..., N_train, P, K]

        # initialization, follows Li's practice
        nn.init.xavier_normal_(self.Fmu)
        nn.init.xavier_normal_(self.Wmu)

        # covariance parameters, initialized as identity matrices
        self.factor_F_N = nn.Parameter(
            torch.eye(
                self.N_train, dtype=torch.get_default_dtype()
            ).repeat(*self.batch_shape, 1, 1),
            requires_grad=True
        )  # [..., N_train, N_train]

        self.factor_F_K = nn.Parameter(
            torch.eye(
                K, dtype=torch.get_default_dtype()
            ).repeat(*self.batch_shape, 1, 1),
            requires_grad=True
        )  # [..., K, K]

        # ----------------------------------------------------------------------------------------

        self.factor_W_N = nn.Parameter(
            torch.eye(
                self.N_train, dtype=torch.get_default_dtype()
            ).repeat(*self.batch_shape, 1, 1),
            requires_grad=True
        )  # [..., N_train, N_train]

        self.factor_W_K = nn.Parameter(
            torch.eye(
                K, dtype=torch.get_default_dtype()
            ).repeat(*self.batch_shape, 1, 1),
            requires_grad=True
        )  # [..., K, K]

        if not self.P_split:
            self.factor_W_P = nn.Parameter(
                torch.eye(
                    self.P, dtype=torch.get_default_dtype()
                ).repeat(*self.batch_shape, 1, 1),
                requires_grad=True
            )  # [..., P, P]
        elif self.P_split:
            # first sub-space
            self.factor_W_P_1 = nn.Parameter(
                torch.eye(
                    self.P1, dtype=torch.get_default_dtype()
                ).repeat(*self.batch_shape, 1, 1),
                requires_grad=True
            )  # [..., P1, P1]
            # second sub-space
            self.factor_W_P_2 = nn.Parameter(
                torch.eye(
                    self.P2, dtype=torch.get_default_dtype()
                ).repeat(*self.batch_shape, 1, 1),
                requires_grad=True
            )  # [..., P2, P2]

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

    # cov matrices
    @property
    def cov_F_N(self):
        return self.factor_F_N @ self.factor_F_N.mT  # [..., N_train, N_train]

    @property
    def cov_F_K(self):
        return self.factor_F_K @ self.factor_F_K.mT  # [..., K, K]

    @property
    def cov_W_N(self):
        return self.factor_W_N @ self.factor_W_N.mT  # [..., N_train, N_train]

    @property
    def cov_W_K(self):
        return self.factor_W_K @ self.factor_W_K.mT  # [..., K, K]

    @property
    def cov_W_P(self):
        assert not self.P_split
        return self.factor_W_P @ self.factor_W_P.mT  # [..., P, P]

    @property
    def cov_W_P_1(self):
        assert self.P_split
        return self.factor_W_P_1 @ self.factor_W_P_1.mT  # [..., P1, P1]

    @property
    def cov_W_P_2(self):
        assert self.P_split
        return self.factor_W_P_2 @ self.factor_W_P_2.mT  # [..., P2, P2]

    # cholesky factors
    @property
    def chol_F_N(self):
        # cholesky factor of cov_F_N
        N = self.cov_F_N.size(-1)

        chol_cov_F_N = psd_safe_cholesky(
            self.cov_F_N + self.jitter * torch.eye(N, dtype=torch.get_default_dtype(), device=self.cov_F_N.device),
        )  # [..., N_train, N_train]

        return chol_cov_F_N

    @property
    def chol_F_K(self):
        # cholesky factor of cov_F_K
        K = self.cov_F_K.size(-1)

        chol_cov_F_K = psd_safe_cholesky(
            self.cov_F_K + self.jitter * torch.eye(K, dtype=torch.get_default_dtype(), device=self.cov_F_K.device),
        )  # [..., K, K]

        return chol_cov_F_K

    @property
    def chol_W_N(self):
        # cholesky factor of cov_W_N
        N = self.cov_W_N.size(-1)

        chol_cov_W_N = psd_safe_cholesky(
            self.cov_W_N + self.jitter * torch.eye(N, dtype=torch.get_default_dtype(), device=self.cov_W_N.device),
        )  # [..., N_train, N_train]

        return chol_cov_W_N

    @property
    def chol_W_K(self):
        # cholesky factor of cov_W_K
        K = self.cov_W_K.size(-1)

        chol_cov_W_K = psd_safe_cholesky(
            self.cov_W_K + self.jitter * torch.eye(K, dtype=torch.get_default_dtype(), device=self.cov_W_K.device),
        )  # [..., K, K]

        return chol_cov_W_K

    @property
    def chol_W_P(self):
        # cholesky factor of cov_W_P
        assert not self.P_split
        P = self.cov_W_P.size(-1)

        chol_cov_W_P = psd_safe_cholesky(
            self.cov_W_P + self.jitter * torch.eye(P, dtype=torch.get_default_dtype(), device=self.cov_W_P.device),
        )  # [..., P, P]

        return chol_cov_W_P

    @property
    def chol_W_P_1(self):
        # cholesky factor of cov_W_P_1
        assert self.P_split
        P1 = self.cov_W_P_1.size(-1)

        chol_cov_W_P_1 = psd_safe_cholesky(
            self.cov_W_P_1 + self.jitter * torch.eye(P1, dtype=torch.get_default_dtype(), device=self.cov_W_P_1.device),
        )  # [..., P1, P1]

        return chol_cov_W_P_1

    @property
    def chol_W_P_2(self):
        # cholesky factor of cov_W_P_2
        assert self.P_split
        P2 = self.cov_W_P_2.size(-1)

        chol_cov_W_P_2 = psd_safe_cholesky(
            self.cov_W_P_2 + self.jitter * torch.eye(P2, dtype=torch.get_default_dtype(), device=self.cov_W_P_2.device),
        )  # [..., P2, P2]

        return chol_cov_W_P_2

    def KL_qF_pF(self):
        Kf = self.kernel_F.forward(self.X_train, self.X_train)  # [..., N_train, N_train]

        LN = self.chol_F_N  # [..., N_train, N_train]
        LK = self.chol_F_K  # [..., K, K]

        SN = self.cov_F_N  # [..., N_train, N_train]
        SK = self.cov_F_K  # [..., K, K]

        chol_Kf = psd_safe_cholesky(Kf)  # [..., N_train, N_train]
        Kf_inv_SN = torch.cholesky_solve(SN, chol_Kf, upper=False)  # [..., N_train, N_train]
        trace_term = torch.einsum('...ii -> ...', Kf_inv_SN) * torch.einsum('...ii -> ...', SK) # traces, [...]

        Kf_inv_Fmu_Fmu_T = torch.cholesky_solve(self.Fmu @ self.Fmu.mT, chol_Kf, upper=False)  # [..., N_train, N_train]
        quad_term = torch.einsum('...ii -> ...', Kf_inv_Fmu_Fmu_T)  # traces, [...]

        logdet_term2 = self.K * torch.sum(
            torch.log(torch.square(torch.diagonal(chol_Kf, dim1=-2, dim2=-1))),
            dim=(-1),
        ) # [...]

        logdet_term1 = (
            self.K * torch.sum(torch.log(torch.square(torch.diagonal(LN, dim1=-2, dim2=-1))), dim=(-1))
          + self.N_train * torch.sum(torch.log(torch.square(torch.diagonal(LK, dim1=-2, dim2=-1))), dim=(-1))
        )  # [...]

        KLs = 0.5 * (logdet_term2 - logdet_term1 - self.K * self.N_train + trace_term + quad_term)  # [...]

        return KLs

    def KL_qW_pW(self):
        Kw = self.kernel_W.forward(self.X_train, self.X_train)  # [..., N_train, N_train]

        LN = self.chol_W_N  # [..., N_train, N_train]
        LK = self.chol_W_K  # [..., K, K]
        if not self.P_split:
            LP = self.chol_W_P  # [..., P, P]
        elif self.P_split:
            LP1 = self.chol_W_P_1  # [..., P1, P1]
            LP2 = self.chol_W_P_2  # [..., P2, P2]

        SN = self.cov_W_N   # [..., N_train, N_train]
        SK = self.cov_W_K   # [..., K, K]
        if not self.P_split:
            SP = self.cov_W_P   # [..., P, P]
        elif self.P_split:
            SP1 = self.cov_W_P_1  # [..., P1, P1]
            SP2 = self.cov_W_P_2  # [..., P2, P2]

        chol_Kw = psd_safe_cholesky(Kw)  # [..., N_train, N_train]
        Kw_inv_SN = torch.cholesky_solve(SN, chol_Kw, upper=False)  # [..., N_train, N_train]

        # trace terms
        trace_Kw_inv_SN = torch.einsum('...ii -> ...', Kw_inv_SN)  # traces, [...]
        trace_SK = torch.einsum('...ii -> ...', SK) # traces, [...]
        if not self.P_split:
            trace_SP = torch.einsum('...ii -> ...', SP) # traces, [...]
        elif self.P_split:
            trace_SP1 = torch.einsum('...ii -> ...', SP1)  # traces, [...]
            trace_SP2 = torch.einsum('...ii -> ...', SP2)  # traces, [...]
            trace_SP = trace_SP1 * trace_SP2  # [...]
        trace_term = trace_Kw_inv_SN * trace_SK * trace_SP  # [...]

        # quadratic terms
        U = self.Wmu.view(*self.batch_shape, self.N_train, self.P * self.K)  # [..., N_train, P*K]
        Kw_inv_U_U_T = torch.cholesky_solve(U @ U.mT, chol_Kw, upper=False)  # [..., N_train, N_train]

        quad_term = torch.einsum('...ii -> ...', Kw_inv_U_U_T)  # traces, [...]

        # constant term
        D = self.P * self.K * self.N_train

        # log det terms
        chol_Kw = psd_safe_cholesky(Kw)  # [..., N_train, N_train]
        logdet_term2 = (D / self.N_train) * torch.sum(torch.log(torch.square(torch.diagonal(chol_Kw, dim1=-2, dim2=-1))), dim=(-1))  # [...]
        _logdet_term1 = (
            (D / self.N_train) * torch.sum(torch.log(torch.square(torch.diagonal(LN, dim1=-2, dim2=-1))), dim=(-1)) +
            (D / self.K) * torch.sum(torch.log(torch.square(torch.diagonal(LK, dim1=-2, dim2=-1))), dim=(-1))
        )
        if not self.P_split:
            logdet_term1 = _logdet_term1 + (D / self.P) * torch.sum(torch.log(torch.square(torch.diagonal(LP, dim1=-2, dim2=-1))), dim=(-1))
        elif self.P_split:
            _logdet_term1 += (D / self.P1) * torch.sum(torch.log(torch.square(torch.diagonal(LP1, dim1=-2, dim2=-1))), dim=(-1))
            logdet_term1 = _logdet_term1 + (D / self.P2) * torch.sum(torch.log(torch.square(torch.diagonal(LP2, dim1=-2, dim2=-1))), dim=(-1))

        KLs = 0.5 * (logdet_term2 - logdet_term1 - D + trace_term + quad_term)  # [...]

        return KLs

    def exp_log_lik(self, x_idx: Tensor, y: Tensor, m: BoolTensor):
        """
        mini-batch across inputs to compute the expected log likelihood term in ELBO.

        Following Li et al., NO mini-batching across outputs.

        :param x_idx: [b]
        :param y: [..., b, P], where P is the size of ALL outputs
        :param m: [..., b, P], where 0 indicates missing
        """

        if torch.all(m.sum(dim=(-1, -2)) == 0):
            warnings.warn("Encounter one empty mini-batch!")
            return 0.
        else:
            if self.lik_model_type == "Gaussian":
                # Approach 1: exact formula, follows Li's practice
                exp_log_lik_1 = self.exp_log_lik_gaussian_exact(x_idx, y, m)  # [...], average over <b*P

                # Approach 2: Monte Carlo approximation
                # exp_log_lik_2 = self.exp_log_lik_gaussian_mc(x_idx, y, m, n_mc_samples=10)  # [...], average over <b*P

                # BUG: exact and MC are not equal TODO
                # print('Using exact formula for Gaussian likelihood:', exp_log_lik_1)
                # print('Using Monte Carlo approximation for Gaussian likelihood:', exp_log_lik_2)

                return exp_log_lik_1

            elif self.lik_model_type == "NegativeBinomial":
                # NO mini-batching across outputs
                exp_log_lik = self.exp_log_lik_nb_mc(x_idx = x_idx, y = y, m = m, output_idx = None, n_mc_samples=10) # [...], average over <b*P

                return exp_log_lik

            else:
                raise NotImplementedError(f"Likelihood model {self.lik_model_type} is not implemented!")

    def exp_log_lik_gaussian_exact(self, x_idx: Tensor, y: Tensor, m: BoolTensor):
        # x_idx: [b], where b is the size of mini-batch
        # y: [..., b, P], where P is the size of ALL outputs
        # m: [..., b, P], where 0 indicate missing

        # Using the exact formula, following Li's practice

        # Prepare

        ##### W related matrices

        SN = self.cov_W_N.index_select(-2, x_idx).index_select(-1, x_idx)  # [..., b, b]
        SK = self.cov_W_K  # [..., K, K]
        if not self.P_split:
            SP = self.cov_W_P  # [..., P, P]
        elif self.P_split:
            SP1 = self.cov_W_P_1  # [..., P1, P1]
            SP2 = self.cov_W_P_2  # [..., P2, P2]

        ##### F related matrices

        VN = self.cov_F_N.index_select(-2, x_idx).index_select(-1, x_idx)  # [..., b, b]
        VK = self.cov_F_K  # [..., K, K]

        select_y = y * m  # [..., b, P] apply masks
        select_Fmu = self.Fmu.index_select(-2, x_idx)  # [..., b, K], select mini-batch inputs
        select_Wmu = self.Wmu.index_select(-3, x_idx) * m.unsqueeze(-1)  # [..., b, P, K], select mini-batch inputs and apply masks

        # quadratic terms

        Eqhnhn = (torch.einsum('...k, ...ij -> ...kij', torch.diagonal(VN, dim1=-2, dim2=-1), VK) +
                  torch.einsum('...bi, ...bj -> ...bij', select_Fmu, select_Fmu))  # [..., b, K, K]

        EqWnWn_1 = select_Wmu.mT @ select_Wmu  # [..., b, K, K]
        _EqWnWn_2 = torch.einsum('...k, ...ij -> ...kij', torch.diagonal(SN, dim1=-2, dim2=-1), SK)  # [..., b, K, K]

        if not self.P_split:
            diag_SP = torch.diagonal(SP, dim1=-2, dim2=-1)  # [..., P]
            trace_SP_masked = torch.einsum('...bp,...p->...b', m.to(torch.get_default_dtype()), diag_SP)  # [..., b]
            trace_SP_masked = trace_SP_masked.unsqueeze(-1).unsqueeze(-1)  # [..., b, 1, 1]
            EqWnWn_2 = _EqWnWn_2 * trace_SP_masked  # [..., b, K, K]
        elif self.P_split:
            diag_SP1 = torch.diagonal(SP1, dim1=-2, dim2=-1)  # [..., P1]
            diag_SP2 = torch.diagonal(SP2, dim1=-2, dim2=-1)  # [..., P2]
            m_2d = m.view(*m.shape[:-1], self.P1, self.P2).to(torch.get_default_dtype()) # [..., b, P1, P2]
            trace_SP_masked = torch.einsum('...p1,...bp1p2,...p2->...b', diag_SP1, m_2d, diag_SP2)  # [..., b]
            trace_SP_masked = trace_SP_masked.unsqueeze(-1).unsqueeze(-1)  # [..., b, 1, 1]
            EqWnWn_2 = _EqWnWn_2 * trace_SP_masked  # [..., b, K, K]

        EqWnWn = EqWnWn_1 + EqWnWn_2  # [..., b, K, K]

        trace_term = torch.diagonal(
            EqWnWn @ Eqhnhn,
            dim1=-2,
            dim2=-1,
        ).sum(-1).view(*self.batch_shape, -1)  # [..., b]

        Wnhn = (select_Wmu @ select_Fmu.unsqueeze(-1)).squeeze(-1)  # [..., b, P]

        ynWnhn = (select_y * Wnhn).sum(dim=(-1))  # [..., b, P] -> [..., b]

        ynyn = select_y.square().sum(dim=-1)  # [..., b, P] -> [..., b]

        quad_term = -0.5 * (ynyn - 2 * ynWnhn + trace_term) / self.lik_model.sigma.square() # [..., b]

        _exp_log_lik = quad_term - 0.5 * m.sum(dim=(-1)) * torch.log(2 * torch.pi * self.lik_model.sigma.square()) # [..., b]

        exp_log_lik = _exp_log_lik.sum(dim=(-1)) / m.sum(dim=(-1, -2))  # [...]

        return exp_log_lik  # [...]

    def compute_qf_mc(self, x_idx: Tensor, m: BoolTensor, n_mc_samples: int = 10):
        """
        During training, compute the predictive q(f_train) using Monte Carlo sampling.
        This is a helper function for exp_log_lik_gaussian_mc, exp_log_lik_nb_mc.
        """
        # x_idx: [b], where b is the size of mini-batch
        # m: [..., b, P], where 0 indicate missing

        # Prepare, F related tensors
        batch_size = x_idx.size(-1)

        Fmu_select = self.Fmu.index_select(-2, x_idx)  # [..., b, K]
        AN = psd_safe_cholesky(
            self.cov_F_N.index_select(-2, x_idx).index_select(-1, x_idx)  # [..., b, b]
        )
        AK = self.chol_F_K  # [..., K, K]

        # Prepare, W related tensors
        Wmu_select = self.Wmu.index_select(-3, x_idx)  # [..., b, P, K]
        LN = psd_safe_cholesky(
            self.cov_W_N.index_select(-2, x_idx).index_select(-1, x_idx)  # [..., b, b]
        )
        LK = self.chol_W_K  # [..., K, K]
        if not self.P_split:
            LP = self.chol_W_P  # [..., P, P]
        elif self.P_split:
            LP1 = self.chol_W_P_1  # [..., P1, P1]
            LP2 = self.chol_W_P_2  # [..., P2, P2]

        # Gaussian random noise
        eps_f = torch.randn(*self.batch_shape, n_mc_samples, batch_size, self.K, device=x_idx.device)  # [..., n_mc, b, K]

        if self.P_split:
            eps_w = torch.randn(*self.batch_shape, n_mc_samples, batch_size, self.P1, self.P2, self.K, device=x_idx.device)  # [..., n_mc, b, P1, P2, K]
        else:
            eps_w = torch.randn(*self.batch_shape, n_mc_samples, batch_size, self.P, self.K, device=x_idx.device)  # [..., n_mc, b, P, K]

        # Sample F and W
        F_s = tl.tenalg.multi_mode_dot(
            tensor=eps_f,  # [..., n_mc, b, K]
            matrix_or_vec_list=[
                AN,  # [..., b, b]
                AK,  # [..., K, K]
            ],
            modes=[-2, -1]
        ) + Fmu_select.unsqueeze(-3)  # [..., n_mc, b, K]

        if not self.P_split:
            W_s = tl.tenalg.multi_mode_dot(
                tensor=eps_w,  # [..., n_mc, b, P, K]
                matrix_or_vec_list=[
                    LN,  # [..., b, b]
                    LP,  # [..., P, P]
                    LK,  # [..., K, K]
                ],
                modes=[-3, -2, -1]
            ) + Wmu_select.unsqueeze(-4)  # [..., n_mc, b, P, K]
        else:
            _W_s = tl.tenalg.multi_mode_dot(
                tensor=eps_w,  # [..., n_mc, b, P1, P2, K]
                matrix_or_vec_list=[
                    LN,  # [..., b, b]
                    LP1,  # [..., P1, P1]
                    LP2,  # [..., P2, P2]
                    LK,  # [..., K, K]
                ],
                modes=[-4, -3, -2, -1]
            )  # [..., n_mc, b, P1, P2, K]
            W_s = _W_s.view(*_W_s.shape[:-3], self.P, self.K) + Wmu_select.unsqueeze(-4)  # [..., n_mc, b, P, K]

        # predictive f, from product of W and F, before masking
        prod_f = (W_s @ F_s.unsqueeze(-1)).squeeze(-1)  # [..., n_mc, b, P], product of W and F

        qf_mean = prod_f.mean(dim=-3)  # [..., b, P], mean of qF
        qf_var = prod_f.var(dim=-3)  # [..., b, P], variance of qF

        # apply masks
        select_qf_mean, select_qf_var = qf_mean[m].view(*self.batch_shape, -1), qf_var[m].view(*self.batch_shape, -1)  # [..., <b*P], apply masks

        return (select_qf_mean,
                select_qf_var) # [..., <b*P]

    def exp_log_lik_gaussian_mc(self, x_idx: Tensor, y: Tensor, m: BoolTensor, n_mc_samples: int = 10):
        # Using Monte Carlo with n_mc_samples samples for Gaussian likelihood.

        # x_idx: [b], where b is the size of mini-batch
        # y: [..., b, P], where P is the size of ALL outputs
        # m: [..., b, P], where 0 indicate missing
        assert self.lik_model_type == "Gaussian"

        select_qf_mean, select_qf_var = self.compute_qf_mc(x_idx, m, n_mc_samples)
        select_y = y[m].view(*self.batch_shape, -1)  # [..., <b*P]

        # compute exp log likelihood
        _exp_log_lik = self.lik_model.exp_log_lik(select_qf_mean, select_qf_var, select_y)  # [..., <b*P]
        exp_log_lik = _exp_log_lik.mean(dim=-1)  # [...], average over <b*P

        return exp_log_lik

    def exp_log_lik_nb_mc(self, x_idx: Tensor, y: Tensor, m: BoolTensor, output_idx: Tensor = None, n_mc_samples: int = 10):
        """
        Using Monte Carlo with n_mc_samples samples for Negative Binomial likelihood.

        x_idx: [b], where b is the size of mini-batch
        y: [..., b, P], where P is the size of ALL outputs
        m: [..., b, P], where 0 indicates missing
        output_idx: [P], default is None, which means all outputs are selected.
        """
        assert self.lik_model_type == "NegativeBinomial"

        if output_idx is None:
            output_idx = torch.arange(self.P, device=x_idx.device)

        select_qf_mean, select_qf_var = self.compute_qf_mc(x_idx, m, n_mc_samples)  # [..., <b*P]
        select_y = y[m].view(*self.batch_shape, -1)  # [..., <b*P]

        # pick_output_idx = torch.masked_select(
        #     output_idx.view(*([1] * (m.ndim - 1)), -1),  # [P] -> [...,1,P]
        #     m.bool()
        # ).view(*self.batch_shape, -1)  # [..., <b*P]

        expanded_idx = output_idx.view(*([1] * (m.ndim - 1)), -1).expand_as(m)  # [..., b, P]
        pick_output_idx = expanded_idx[m].view(*self.batch_shape, -1)  # [..., <b*P>]

        # compute exp log likelihood
        _exp_log_lik = self.lik_model.exp_log_lik(
            qf_mean=select_qf_mean, qf_var=select_qf_var, y=select_y, output_idx=pick_output_idx, method='gauss_hermite'
        )  # [..., <b*P]

        exp_log_lik = _exp_log_lik.mean(dim=-1)  # [...], average over <b*P

        return exp_log_lik  # [...]

    def elbo(self, x_idx: Tensor, y: Tensor, m: BoolTensor, coeff_exp_log_lik: float, beta_F: float = 1., beta_W: float = 1., average_elbo: bool = False):
        """
        mini-batch ELBO, following Li's practice, NO mini-batching across outputs.
        We extend Li's implementation to support the mini-batching across inputs.
        # TODO: mask before compute

        :param x_idx: [b], the indices of mini-batch inputs relative to X_train.
        :param y: [..., b, P], where P is the size of ALL outputs.
        :param m: [..., b, P], where 0 indicates missing.

        """
        # ELBO = - beta_F * KL(qF || pF) - beta_W * KL(qW || pW) + E_q[log p(Y | X, F, W)]
        kl_qF_pF = self.KL_qF_pF()
        kl_qW_pW = self.KL_qW_pW()
        exp_log_lik = self.exp_log_lik(x_idx, y, m)

        elbo = - beta_F * kl_qF_pF - beta_W * kl_qW_pW + coeff_exp_log_lik * exp_log_lik  # [...], average over <b*P

        if average_elbo:
            elbo = elbo / coeff_exp_log_lik

        return elbo

    def train_sgprn(
        self, train_dataloader: DataLoader, optimizer: Optimizer, epochs: int,
        beta_F: float = 1., beta_W: float = 1., coeff_exp_log_lik: Optional[float] = None,
        max_norm: Optional[float] = None, device: str = "cpu", print_epochs: int = 10
    ):
        # NO mini-batching across outputs, only across inputs.
        assert train_dataloader.dataset.get_idx, "The dataset must ensure get_idx == True to retrieve indices of mini-batch inputs."

        self.to(device)
        self.train()
        if coeff_exp_log_lik is None:
            # biased if there are missing values
            coeff_exp_log_lik = len(train_dataloader.dataset) * self.num_outputs
        perm = None  # cache

        for epoch in range(epochs):
            for batch_X_idx, _, batch_all_Y, batch_all_m in train_dataloader:  # [b], [b, ..., P], P is the size of ALL outputs
                if perm is None:  # re-arrange dims
                    ndim = batch_all_Y.ndim
                    perm = list(range(1, ndim - 1)) + [0, ndim - 1]
                batch_X_idx = batch_X_idx.to(device)  # [b]
                batch_all_Y = batch_all_Y.to(device).permute(*perm)  # [b, ..., P] -> [..., b, P]
                batch_all_m = batch_all_m.to(device).permute(*perm)  # [b, ..., P] -> [..., b, P]

                optimizer.zero_grad(set_to_none=True)
                loss = - self.elbo(batch_X_idx, batch_all_Y, batch_all_m, coeff_exp_log_lik = coeff_exp_log_lik, beta_F = beta_F, beta_W = beta_W)
                loss.backward()

                if max_norm is not None:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
                    if (epoch + 1) % print_epochs == 0 and total_grad_norm > max_norm:
                        print(f"Gradient norm {total_grad_norm:.3f} exceeds the threshold {max_norm:.3f}, clipping applied.")

                optimizer.step()

            if (epoch + 1) % print_epochs == 0:
                print(f'Epoch {epoch + 1} / {epochs}； Loss: {loss.item():.6f}')

    @torch.no_grad()
    def predict(
        self, x_star: Tensor, num_samples: int = 10, device: str = "cpu", noiseless: bool = False
    ):
        """
        x_star: [..., N_test, D_X]
        num_samples: number of draws of W and F at test inputs x_star.
        Get predictive mean and variance of ALL outputs at test inputs x_star.
        """
        if noiseless:
            assert self.lik_model_type == "Gaussian", "Only Gaussian Likelihood support noiseless prediction."

        self.to(device)
        self.eval()
        x_star = x_star.to(device)

        n_test = x_star.size(-2)

        # Step1: Sample F

        ## Step 1.1 Prepare
        Kf11 = self.kernel_F.forward(self.X_train, self.X_train)  # [..., N_train, N_train]
        Kf12 = self.kernel_F.forward(self.X_train, x_star)  # [..., N_train, N_test]
        Kf22 = self.kernel_F.forward(x_star, x_star)  # [..., N_test, N_test]

        Lf11 = psd_safe_cholesky(Kf11)  # [..., N_train, N_train]

        Kf11InvKf12 = torch.cholesky_solve(Kf12, Lf11, upper=False)  # [..., N_train, N_test]
        vf = torch.linalg.solve_triangular(Lf11, Kf12, upper=False)  # [..., N_train, N_test]

        AK = self.chol_F_K  # [..., K, K]

        Fstar_mu = Kf11InvKf12.mT @ self.Fmu  # [..., N_test, K]
        Fstar_v1 = Kf22 - vf.mT @ vf  # [..., N_test, N_test]
        Fstar_v2 = Kf11InvKf12.mT @ self.cov_F_N @ Kf11InvKf12  # [..., N_test, N_test]

        Fstar_std1_diag = torch.diag_embed(
            torch.clamp_min(
                torch.diagonal(Fstar_v1, dim1=-2, dim2=-1), 0
            ).sqrt(),
            dim1=-2, dim2=-1
        )  # [..., N_test, N_test]
        Fstar_std2_diag = torch.diag_embed(
            torch.clamp_min(
                torch.diagonal(Fstar_v2, dim1=-2, dim2=-1), 0
            ).sqrt(),
            dim1=-2, dim2=-1
        )# [..., N_test, N_test]

        ## Step 1.2 Sample eps_f_star
        eps1_f_star = torch.randn(*self.batch_shape, num_samples, n_test, self.K, device=device)
        eps2_f_star = torch.randn(*self.batch_shape, num_samples, n_test, self.K, device=device)

        F_s_term1 = tl.tenalg.multi_mode_dot(
            tensor=eps1_f_star,  # [..., n_mc, N_test, K]
            matrix_or_vec_list=[
                Fstar_std1_diag, # [..., N_test, N_test]
            ],
            modes=[-2]
        )  # [..., n_mc, N_test, K]

        F_s_term2 = tl.tenalg.multi_mode_dot(
            tensor=eps2_f_star,  # [..., n_mc, N_test, K]
            matrix_or_vec_list=[
                Fstar_std2_diag, # [..., N_test, N_test]
                AK  # [..., K, K]
            ],
            modes=[-2, -1]
        )  # [..., n_mc, N_test, K]

        F_s = F_s_term1 + F_s_term2 + Fstar_mu.unsqueeze(-3)  # [..., n_mc, N_test, K]

        # Step2: Sample W

        ## Step 2.1 Prepare
        Kw11 = self.kernel_W.forward(self.X_train, self.X_train)  # [..., N_train, N_train]
        Kw12 = self.kernel_W.forward(self.X_train, x_star)  # [..., N_train, N_test]
        Kw22 = self.kernel_W.forward(x_star, x_star) # [..., N_test, N_test]

        Lw11 = psd_safe_cholesky(Kw11)   # [..., N_train, N_train]

        Kw11InvKw12 = torch.cholesky_solve(Kw12, Lw11, upper=False)  # [..., N_train, N_test]
        vw = torch.linalg.solve_triangular(Lw11, Kw12, upper=False)  # [..., N_train, N_test]

        Wstar_mu = tl.tenalg.multi_mode_dot(
            tensor=self.Wmu,  # [..., N_train, P, K]
            matrix_or_vec_list=[
                Kw11InvKw12.mT, # [..., N_train, N_test] -> [..., N_test, N_train]
            ],
            modes=[-3]
        )  # [..., N_test, P, K]
        Wstar_v1 = Kw22 - vw.mT @ vw  # [..., N_test, N_test]
        Wstar_v2 = Kw11InvKw12.mT @ self.cov_W_N @ Kw11InvKw12  # [..., N_test, N_test]

        Wstar_std1_diag = torch.diag_embed(
            torch.clamp_min(
                torch.diagonal(Wstar_v1, dim1=-2, dim2=-1), 0
            ).sqrt(),
            dim1=-2, dim2=-1
        )  # [..., N_test, N_test]
        Wstar_std2_diag = torch.diag_embed(
            torch.clamp_min(
                torch.diagonal(Wstar_v2, dim1=-2, dim2=-1), 0
            ).sqrt(),
            dim1=-2, dim2=-1
        )  # [..., N_test, N_test]

        # Step 2.2 Sample eps_w_star
        eps1_w_star = torch.randn(*self.batch_shape, num_samples, n_test, self.P, self.K, device=device)

        if self.P_split:
            eps2_w_star_split = torch.randn(*self.batch_shape, num_samples, n_test, self.P1, self.P2, self.K, device=device)
        else:
            eps2_w_star = torch.randn(*self.batch_shape, num_samples, n_test, self.P, self.K, device=device)

        W_s_term1 = tl.tenalg.multi_mode_dot(
            tensor=eps1_w_star,  # [..., n_mc, N_test, P, K]
            matrix_or_vec_list=[
                Wstar_std1_diag,  # [..., N_test, N_test]
            ],
            modes=[-3]
        )  # [..., n_mc, N_test, P, K]

        if self.P_split:
            W_s_term2 = tl.tenalg.multi_mode_dot(
                tensor=eps2_w_star_split,  # [..., n_mc, N_test, P1, P2, K]
                matrix_or_vec_list=[
                    Wstar_std2_diag,  # [..., N_test, N_test]
                    self.chol_W_P_1,  # [..., P1, P1]
                    self.chol_W_P_2,  # [..., P2, P2]
                    self.chol_W_K,    # [..., K, K]
                ],
                modes=[-4, -3, -2, -1]
            ) # [..., n_mc, N_test, P1, P2, K]

            W_s_term2 = W_s_term2.view(*W_s_term2.shape[:-3], self.P, self.K)  # [..., n_mc, N_test, P, K]

        else:
            W_s_term2 = tl.tenalg.multi_mode_dot(
                tensor=eps2_w_star,  # [..., n_mc, N_test, P, K]
                matrix_or_vec_list=[
                    Wstar_std2_diag,  # [..., N_test, N_test]
                    self.chol_W_P,    # [..., P, P]
                    self.chol_W_K,    # [..., K, K]
                ],
                modes=[-3, -2, -1]
            ) # [..., n_mc, N_test, P, K]

        W_s = W_s_term1 + W_s_term2 + Wstar_mu.unsqueeze(-4)  # [..., n_mc, N_test, P, K]

        # Step3: Predictive mean and variance from W and F
        prod_f = (W_s @ F_s.unsqueeze(-1)).squeeze(-1) # [..., n_mc, N_test, P]

        qf_mean = prod_f.mean(dim=-3)  # [..., N_test, P], mean of q(f)
        qf_var = prod_f.var(dim=-3)    # [..., N_test, P], variance of q(f)

        if self.lik_model_type == "Gaussian":
            # whether or not pass through likelihood, i.e. noiseless or noisy prediction.
            if noiseless:
                return qf_mean, qf_var  # [..., N_test, P]
            else:
                qy_var = qf_var + self.lik_model.sigma.square()
                return qf_mean, qy_var  # [..., n_test, P]

        elif self.lik_model_type == "NegativeBinomial":
            # must pass through likelihood
            output_idx = torch.arange(self.num_outputs, device=device)

            qy_means, qy_vars = self.lik_model.predict(
                qf_means=qf_mean, qf_covs=qf_var, output_idx=output_idx, num_mc=20,
            )  # [..., N_test, P]

            return qy_means, qy_vars  # [..., N_test, P]

        else:
            raise NotImplementedError

    @torch.no_grad()
    def predict_sgprn_gaussian(
        self, data_dict, num_samples: int = 10, noiseless: bool = False, num_plot_points: Optional[int] = 2000, device: str = "cpu"
    ):
        # on device
        assert self.lik_model_type == "Gaussian"

        all_X, train_Y, train_mask, test_Y, test_mask = (
            data_dict["all_X"].to(device), data_dict['train_Y'].to(device), data_dict['train_mask'].to(device), data_dict['test_Y'].to(device), data_dict['test_mask'].to(device)
        )

        # predict on all_X for metrics
        pred_means, pred_vars = self.predict(
            x_star=all_X, num_samples=num_samples, device=device, noiseless=noiseless
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

        plot_pred_dict = None

        if num_plot_points is None:
            return metric_dict, pred_dict, plot_pred_dict

        # predict on denser input X for plotting
        if all_X.size(-1) == 1:
            X_min, X_max = all_X.min().item(), all_X.max().item()
            denser_X = torch.linspace(X_min, X_max, num_plot_points, dtype=all_X.dtype, device=device).unsqueeze(-1)  # [n_plot, 1]
            if len(self.batch_shape) > 0:
                denser_X = denser_X.view(*([1] * len(self.batch_shape)), num_plot_points, 1).expand(
                    *self.batch_shape, num_plot_points, 1
                )  # [..., n_plot, 1]

            plot_pred_means, plot_pred_vars = self.predict(
                denser_X, num_samples=num_samples, device=device, noiseless=True
            )  # [..., n_plot, P], we want noiseless for plotting

            plot_pred_dict = {
                "denser_X": denser_X,  # [..., n_plot, D_X]
                "plot_pred_means": plot_pred_means,  # [..., n_plot, P]
                "plot_pred_vars": plot_pred_vars,  # [..., n_plot, P]
            }

        return metric_dict, pred_dict, plot_pred_dict