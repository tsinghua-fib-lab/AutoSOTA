import warnings
from typing import Optional

import torch
from torch import nn, Tensor, LongTensor

# from kernels.rbf_kernel import MyRBFKernel
from gpytorch.kernels import Kernel
from gpytorch.variational.natural_variational_distribution import _NaturalToMuVarSqrt
from gpytorch.variational.tril_natural_variational_distribution import _TrilNaturalToMuVarSqrt
from linear_operator.utils.cholesky import psd_safe_cholesky


__all__ = [
    "Prior_H", "Variational_H", "Variational_inducing_dist", "mo_Variational_inducing_dist",
    "Inducing_points", "GP_with_qU", "svgp_base"
]


class Prior_H(nn.Module):
    """
    p(H), fully factorized prior.
    """
    def __init__(self, mean_pH: Tensor, diag_cov_pH: Tensor):
        # mean_pH, diag_cov_pH: [..., P, D_H]
        super(Prior_H, self).__init__()
        assert mean_pH.shape == diag_cov_pH.shape
        assert torch.all(diag_cov_pH > 0), "Prior H cov must be positive."
        self.register_buffer("mean_pH", mean_pH)
        self.register_buffer("diag_cov_pH", diag_cov_pH)

class Variational_H(nn.Module):
    """q(H)"""
    def __init__(self, P: int, D_H: int, batch_shape: Optional[tuple] = (), mean_field: bool = False):
        super(Variational_H, self).__init__()
        self.mean_field = mean_field
        mean_qH_shape = batch_shape + (P, D_H, )
        self.register_parameter(
            "mean_qH", nn.Parameter(
                torch.zeros(mean_qH_shape, dtype=torch.get_default_dtype()), requires_grad=True
            )
        )

        if mean_field:
            self.register_parameter(
                "raw_diag_cov_qH", nn.Parameter(
                    torch.ones(mean_qH_shape, dtype=torch.get_default_dtype()), requires_grad=True
                )
            )
        else:
            factor_cov_qH_shape = mean_qH_shape[:-1] + (1, 1,)
            self.register_parameter(
                "factor_cov_qH", nn.Parameter(
                    torch.eye(D_H, dtype=torch.get_default_dtype()).repeat(*factor_cov_qH_shape), requires_grad=True
                )
            )

    @property
    def cov_qH(self):
        if self.mean_field:
            return nn.functional.softplus(self.raw_diag_cov_qH)

        cov_qH = self.factor_cov_qH @ self.factor_cov_qH.mT
        return cov_qH

    def sample(self, ids):
        _mean = self.mean_qH[..., ids, :]  # [..., len(ids), D_H]
        _eps = torch.randn_like(_mean, dtype=torch.get_default_dtype(), device=_mean.device)  # [..., len(ids), D_H]
        if self.mean_field:
            _std = torch.sqrt(self.cov_qH[..., ids, :])  # [..., len(ids), D_H]
            return _mean + _eps * _std  # [..., len(ids), D_H]

        _L = psd_safe_cholesky(self.cov_qH[..., ids, :, :])  # [..., len(ids), D_H, D_H]
        return _mean + (_L @ _eps.unsqueeze(-1)).squeeze(-1)  # # [..., len(ids), D_H]

class Delta_H(nn.Module):
    """
    restrict q(H) as a delta distribution, i.e., point estimate of H
    """
    def __init__(
        self, P: int, D_H: int, batch_shape: Optional[tuple] = (), trainable: bool = True, init_as_index: bool = False,
    ):
        super(Delta_H, self).__init__()
        point_qH_shape = batch_shape + (P, D_H, )

        if init_as_index:
            init = torch.arange(P, dtype=torch.get_default_dtype()).view(
                *([1] * len(batch_shape)), P, 1).expand(*batch_shape, P, D_H).clone()
        else:
            init = torch.zeros(point_qH_shape, dtype=torch.get_default_dtype())
            assert trainable, "You initialize Hs as zeros, but they are not trainable, this is very likely a bug."

        if trainable:
            self.register_parameter(
                "mean_qH", nn.Parameter(init, requires_grad=True)
            )  # [..., P, D_H]
        else:
            self.register_buffer(
                "mean_qH", init
            )  # [..., P, D_H]

    def sample(self, ids):
        return self.mean_qH[..., ids, :]  # [..., len(ids), D_H]

class Variational_inducing_dist(nn.Module):
    """
    q(U);
    M: the number of inducing points
    """
    def __init__(self, M: int, batch_shape: Optional[tuple] = (), jitter: float = 1e-6):
        super(Variational_inducing_dist, self).__init__()
        self.M = int(M)
        self.batch_shape = batch_shape
        self.jitter = jitter
        mean_qU_shape = batch_shape + (M, )
        self.register_parameter(
            "mean_qU",
            nn.Parameter(
                torch.zeros(mean_qU_shape, dtype=torch.get_default_dtype()), requires_grad=True
            )
        )

        self.register_parameter(
            "factor_cov_qU",
            nn.Parameter(
                torch.eye(self.M, dtype=torch.get_default_dtype()).repeat(*batch_shape, 1, 1), requires_grad=True
            )
        )

    def forward(self):
        """
        return mean, cholesky factor of cov, cov in one forward pass
        """
        m = self.mean_qU  # [..., M]
        cov_qU = self.factor_cov_qU @ self.factor_cov_qU.mT  # [..., M, M]
        L = psd_safe_cholesky(
            cov_qU + self.jitter * torch.eye(cov_qU.size(-1), dtype=torch.get_default_dtype(), device=cov_qU.device)
        )  # [..., M, M]
        return m, L, cov_qU

    @property
    def cov_qU(self):
        # return self.chol_qU @ self.chol_qU.mT
        return self.factor_cov_qU @ self.factor_cov_qU.mT  # [..., M, M]

    @property
    def chol_qU(self):
        """
        off_diag_term = self.factor_cov_qU.tril(-1)
        diag_term = torch.diag_embed(
            nn.functional.softplus(
                torch.diagonal(self.factor_cov_qU, dim1=-1, dim2=-2)
            ), dim1=-1, dim2=-2
        ) # positive diagonal elements

        return off_diag_term + diag_term
        """
        M = self.cov_qU.size(-1)

        chol_cov_qU = psd_safe_cholesky(
            self.cov_qU + self.jitter * torch.eye(M, dtype=torch.get_default_dtype(), device=self.cov_qU.device)
        )  # [..., M, M]

        return chol_cov_qU

    def pick_mean(self, mean_qU: Tensor, *args, **kwargs):
        # NO dim picking should be done for this class.
        # if one wants to pick mean along the output dim, use mo_Variational_inducing_dist instead.
        return mean_qU

    def pick_cov(self, cov_qU: Tensor, *args, **kwargs):
        # NO dim picking should be done for this class.
        # if one wants to pick cov along the output dim, use mo_Variational_inducing_dist instead.
        return cov_qU

class Natural_Variational_inducing_dist(nn.Module):
    """
    q(U) in natural parameterization and update via natural gradient descent.
    M: the number of inducing points

    Following https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/variational/natural_variational_distribution.py
    """
    def __init__(self, M: int, batch_shape: Optional[tuple] = (), jitter: float = 1e-6):
        super(Natural_Variational_inducing_dist, self).__init__()
        self.M = int(M)
        self.batch_shape = batch_shape
        self.jitter = jitter
        natural_vec_shape = batch_shape + (M, )
        self.register_parameter(
            "natural_vec",
            nn.Parameter(
                torch.zeros(natural_vec_shape, dtype=torch.get_default_dtype(), requires_grad=True)
            )
        )

        self.register_parameter(
            "natural_mat",
            nn.Parameter(
                torch.eye(self.M, dtype=torch.get_default_dtype()).mul(-0.5).repeat(*batch_shape, 1, 1), requires_grad=True
            )
        )

    def forward(self):
        m, L = _NaturalToMuVarSqrt.apply(self.natural_vec, self.natural_mat)  # m: [..., M], L: [..., M, M]
        cov_qU = L @ L.mT  # [..., M, M]
        return m, L, cov_qU

    def pick_mean(self, mean_qU: Tensor, *args, **kwargs):
        # NO dim picking should be done for this class.
        return mean_qU

    def pick_cov(self, cov_qU: Tensor, *args, **kwargs):
        # NO dim picking should be done for this class.
        return cov_qU

class TrilNatural_Variational_inducing_dist(nn.Module):
    """
    q(U) parametrized by natural vector and a triangular decomposition of the natural matrix (which is not the Cholesky)

    Following https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/variational/tril_natural_variational_distribution.py
    """
    def __init__(self, M: int, batch_shape: Optional[tuple] = (), jitter: float = 1e-6):
        super(TrilNatural_Variational_inducing_dist, self).__init__()
        self.M = int(M)
        self.batch_shape = batch_shape
        self.jitter = jitter
        natural_vec_shape = batch_shape + (M,)
        self.register_parameter(
            "natural_vec",
            nn.Parameter(
                torch.zeros(natural_vec_shape, dtype=torch.get_default_dtype(), requires_grad=True)
            )
        )

        self.register_parameter(
            "natural_tril_mat",
            nn.Parameter(
                torch.eye(self.M, dtype=torch.get_default_dtype()).repeat(*batch_shape, 1, 1), requires_grad=True
            )
        )

    def forward(self):
        m, L = _TrilNaturalToMuVarSqrt.apply(self.natural_vec, self.natural_tril_mat)
        cov_qU = L @ L.mT
        return m, L, cov_qU

    def pick_mean(self, mean_qU: Tensor, *args, **kwargs):
        # NO dim picking should be done for this class.
        return mean_qU

    def pick_cov(self, cov_qU: Tensor, *args, **kwargs):
        # NO dim picking should be done for this class.
        return cov_qU

class mo_Variational_inducing_dist(Variational_inducing_dist):
    # multi-output
    def __init__(self, num_outputs: int, M: int, batch_shape: Optional[tuple] = (), jitter: float = 1e-6):
        batch_shape = batch_shape + (num_outputs, )
        super(mo_Variational_inducing_dist, self).__init__(M=M, batch_shape=batch_shape, jitter=jitter)

    # override
    def pick_mean(self, mean_qU: Tensor, output_idx: Optional[LongTensor] = None):
        """
        pick mean along the output dim (-2)
        """
        if output_idx is None:
            return mean_qU
        else:
            return mean_qU.index_select(-2, output_idx)

    # override
    def pick_cov(self, cov_qU, output_idx: Optional[Tensor] = None):
        """
        pick cov along the output dim (-3)
        """
        if output_idx is None:
            return cov_qU
        else:
            return cov_qU.index_select(-3, output_idx)

class mo_Natural_Variational_inducing_dist(Natural_Variational_inducing_dist):
    # multi-output, natural gradient descent
    def __init__(self, num_outputs: int, M: int, batch_shape: Optional[tuple] = (), jitter: float = 1e-6):
        batch_shape = batch_shape + (num_outputs, )
        super(mo_Natural_Variational_inducing_dist, self).__init__(M=M, batch_shape=batch_shape, jitter=jitter)

    # override
    def pick_mean(self, mean_qU: Tensor, output_idx: Optional[LongTensor] = None):
        """
        pick mean along the output dim (-2)
        """
        if output_idx is None:
            return mean_qU
        else:
            return mean_qU.index_select(-2, output_idx)

    # override
    def pick_cov(self, cov_qU, output_idx: Optional[Tensor] = None):
        """
        pick cov along the output dim (-3)
        """
        if output_idx is None:
            return cov_qU
        else:
            return cov_qU.index_select(-3, output_idx)

class mo_TrilNatural_Variational_inducing_dist(TrilNatural_Variational_inducing_dist):
    # multi-output, natural gradient descent
    def __init__(self, num_outputs: int, M: int, batch_shape: Optional[tuple] = (), jitter: float = 1e-6):
        batch_shape = batch_shape + (num_outputs, )
        super(mo_TrilNatural_Variational_inducing_dist, self).__init__(M=M, batch_shape=batch_shape, jitter=jitter)

    # override
    def pick_mean(self, mean_qU: Tensor, output_idx: Optional[LongTensor] = None):
        """
        pick mean along the output dim (-2)
        """
        if output_idx is None:
            return mean_qU
        else:
            return mean_qU.index_select(-2, output_idx)

    # override
    def pick_cov(self, cov_qU, output_idx: Optional[Tensor] = None):
        """
        pick cov along the output dim (-3)
        """
        if output_idx is None:
            return cov_qU
        else:
            return cov_qU.index_select(-3, output_idx)

class Inducing_points(nn.Module):
    """
    this class is Z, inducing points/locations
    IP_init: [..., M, D_T]
    M: the number of inducing points
    """
    def __init__(self, M: int, num_dims: int, IP_init: Tensor, IP_name = "Z", IP_joint = True):
        super(Inducing_points, self).__init__()
        self.IP_name = IP_name
        self.M = M

        assert IP_init.shape[-2:] == torch.Size([M, num_dims])
        self.batch_shape = tuple(IP_init.shape[:-2])   # [...]

        if IP_joint:
            self.register_parameter(IP_name, nn.Parameter(IP_init, requires_grad=True))
        else:
            self.register_buffer(IP_name, IP_init)

    @property
    def inducing_points(self):
        return getattr(self, self.IP_name)


class GP_with_qU(nn.Module):
    """
    Base class for SVGP-style models equipped with Z, qU, except for
        (1) gs_lvmogp (2) sgprn (3) ind_gp_base
    """
    def __init__(
        self, kernel: Kernel, Z: Inducing_points, qU: Variational_inducing_dist,
        whitening: bool = True, jitter: float = 1e-6,
    ):
        super(GP_with_qU, self).__init__()
        self.kernel = kernel
        self.Z = Z
        self.qU = qU
        self.whitening = whitening
        self.jitter = jitter

        assert qU.M == Z.M
        assert Z.batch_shape == qU.batch_shape

        self.cache = {}  # for caching intermediate results
        # We assume KL_qU_pU and variational_f_base are called in pair in one forward pass.
        self.cache["use_cache_for_svgp"] = False  # by default, do not use cache, possibly be altered in child classes
        self.cache["KL_qU_pU_count"], self.cache["variational_f_base_count"] = 0, 0

        if kernel.batch_shape is None or len(kernel.batch_shape) == 0:
            warnings.warn(
                f"Kernel batch_shape is not specified. "
                "In this case, there is no batch shape or share kernel parameters across batch dims. "
            )
        else:
            assert tuple(kernel.batch_shape) == Z.batch_shape, \
                f"Kernel batch_shape {kernel.batch_shape} does not match Z batch_shape {Z.batch_shape}."

    def chol_cov_pU(self, _K_uu: Optional[Tensor] = None):
        """
        cholesky factor of covariance matrix of p(U)
        """
        if _K_uu is None:
            K_uu = self.kernel.forward(self.Z.inducing_points, self.Z.inducing_points)   # [..., (self.num_outputs), M, M], if self.num_outputs dim exist, it should be considered as inside ...
        else:
            K_uu = _K_uu

        M = K_uu.size(-1)
        chol_cov_pU = psd_safe_cholesky(
            K_uu + self.jitter * torch.eye(M, dtype=torch.get_default_dtype(), device=self.Z.inducing_points.device)
        )   # [..., (self.num_outputs), M, M]

        return chol_cov_pU

    # @property
    # def chol_cov_qU(self):
    #     """
    #     cholesky factor of covariance matrix of q(U)
    #     """
    #     chol_cov_qU = self.qU.chol_qU

    #     return chol_cov_qU

    @property
    def KL_qU_pU (self):
        """
        KL divergence between q(U) and p(U)
        """
        assert self.training, "KL divergence can ONLY be computed during training."
        if self.cache["use_cache_for_svgp"]:
            if self.cache["KL_qU_pU_count"] < self.cache["variational_f_base_count"]:
                mean_qU = self.cache["mean_qU"]
                chol_cov_qU = self.cache["chol_cov_qU"]
                cov_qU = self.cache["cov_qU"]
                M = cov_qU.size(-1)
                self.cache["KL_qU_pU_count"] += 1

            elif self.cache["KL_qU_pU_count"] == self.cache["variational_f_base_count"]:
                mean_qU, chol_cov_qU, cov_qU = self.qU.forward()
                M = cov_qU.size(-1)
                self.cache["mean_qU"] = mean_qU
                self.cache["chol_cov_qU"] = chol_cov_qU
                self.cache["cov_qU"] = cov_qU
                self.cache["KL_qU_pU_count"] += 1

            elif self.cache["KL_qU_pU_count"] > self.cache["variational_f_base_count"]:
                raise NotImplementedError("It seems that you call KL_qU_pU in GP_with_qU twice without calling variational_f_base in between. ")
        else:
            mean_qU, chol_cov_qU, cov_qU = self.qU.forward()
            M = cov_qU.size(-1)
        # mean_qU, cov_qU = self.qU.mean_qU, self.qU.cov_qU
        # M = self.qU.cov_qU.size(-1)
        # chol_cov_qU = self.qU.chol_cov_qU

        if self.whitening:   # p(U) = N(0,I)
            trace = chol_cov_qU.square().sum(dim=(-1, -2))  # [...]
            mahalanobis = mean_qU.square().sum(dim=(-1))  # [...]
            half_log_det = torch.diagonal(chol_cov_qU, dim1=-1, dim2=-2).log().sum(dim=(-1))  # [...]
            KL = 0.5 * (trace - M + mahalanobis) - half_log_det  # [...]

            # unit test
            # qU = MultivariateNormal(mean_qU, cov_qU)  # [..., M, M]
            # pU = MultivariateNormal(
            #   torch.zeros_like(mean_qU, dtype=torch.get_default_dtype(), device=mean_qU.device),
            #   torch.eye(M, dtype=torch.get_default_dtype(), device=cov_qU.device).repeat(*cov_qU.shape[:-2], 1, 1)
            # )
            # KL2 = kl_divergence(qU, pU)  # [...]
            # print(f"My implemented KL(qU || pU) is: {KL}, PyTorch KL is: {KL2}.")

        else:
            K_uu = self.kernel.forward(self.Z.inducing_points, self.Z.inducing_points)  # [..., M, M]
            chol_cov_pU = self.chol_cov_pU(_K_uu=K_uu)  # [..., M, M]
            trace = torch.linalg.solve_triangular(chol_cov_pU, chol_cov_qU, upper=False).square().sum(
                dim=(-1, -2))  # [...]
            chol_cov_pU_inv_mu = torch.linalg.solve_triangular(chol_cov_pU, mean_qU.unsqueeze(-1),
                                                               upper=False)  # [..., M, 1]
            mahalanobis = chol_cov_pU_inv_mu.square().sum(dim=(-1, -2))  # [...]
            half_log_det_cov_pU = torch.diagonal(chol_cov_pU, dim1=-1, dim2=-2).log().sum(dim=(-1))  # [...]
            half_log_det_cov_qU = torch.diagonal(chol_cov_qU, dim1=-1, dim2=-2).log().sum(dim=(-1))  # [...]
            KL = 0.5 * (trace - M + mahalanobis) + half_log_det_cov_pU - half_log_det_cov_qU  # [...]

            # unit test
            # qU = MultivariateNormal(mean_qU, cov_qU)  # [..., M, M]
            # pU = MultivariateNormal(torch.zeros_like(mean_qU, dtype=torch.get_default_dtype(), device=mean_qU.device), K_uu)
            # KL2 = kl_divergence(qU, pU)  # [...]
            # print(f"My implemented KL(qU || pU) is: {KL}, PyTorch KL is: {KL2}.")

        return KL   # [...]

    @property
    def m_spherical(self):
        # for SVGP and tighter bound (Gaussian), this is always 1.
        # not 1 ONLY for tighter bound with non-Gaussian likelihoods.
        return 1.

    def variational_f_base(
        self, K_uu: Tensor, K_fu: Tensor, K_ff: Tensor, output_idx: Optional[LongTensor] = None,
        mean_func_at_f: Optional[Tensor] = None,
    ):
        """
        Compute q(f) = N(variational_mean, variational_cov);

        For whitening:
        p(u0) = N(0, I), q(u0) = N(m0, S0) where m0 and S0 are from qU.forward().
        u = \mu(Z) + Lu @ u0 where \mu(Z) is the mean function evaluated at inducing points Z, Lu=cholesky(K_uu).
        Thus, p(u) = N(\mu(Z), K_uu), q(u) = N(\mu(Z) + Lu @ m0, Lu @ S0 @ Lu.T).
        for input x, the variational distribution q(f) is:
            q(f) = \int p(f|u) q(u) du
                mean: K_fu @ K_uu^{-1} @ (\mu(Z) + Lu @ m0 - \mu(Z)) + \mu(x)
                cov: K_ff - K_fu @ K_uu^{-1} @ K_uf + K_fu @ K_uu^{-1} @ Lu @ S0 @ Lu.T @ K_uu^{-1} @ K_uf.

        For non-whitening:
        p(u) = N(\mu(Z), K_uu), q(u) = N(\mu(Z) + m, S) where m and S are from qU.forward().
        for input x, the variational distribution q(f) is:
            q(f) = \int p(f|u) q(u) du
                mean: K_fu @ K_uu^{-1} @ (\mu(Z) + m - \mu(Z)) + \mu(x)
                cov: K_ff - K_fu @ K_uu^{-1} @ K_uf + K_fu @ K_uu^{-1} @ S @ K_uu^{-1} @ K_uf.

        Only support diagonal covariance for q(f).
            K_uu: [..., (P), M, M];
            K_fu: [..., (P), n, M];
            K_ff: [..., (P), n], diagonal cov.
        (Optional) mean_func_at_f: [..., (P), n], evaluated GP's mean function values at data points.
        By default, if mean_func_at_f are None, zero mean function is assumed.
        NOTE:
            (1) for mini-batching across outputs, the output_idx should be specified, which is the indices of outputs to be selected for qU mean and cov.
            K_uu, K_fu and K_ff (optionally mean_func_at_f) should also be computed with the same output_idx.
            (2) For non multi-output case, output_idx should be set to None.
        """
        # check compatibility of output dim
        if output_idx is not None:
            assert K_uu.size(-3) == K_fu.size(-3) == K_ff.size(-2) == len(output_idx)

        if self.training and self.cache["use_cache_for_svgp"]:
            if self.cache["variational_f_base_count"] < self.cache["KL_qU_pU_count"]:
                mean_qU = self.cache["mean_qU"]
                chol_cov_qU = self.cache["chol_cov_qU"]
                cov_qU = self.cache["cov_qU"]
                self.cache["variational_f_base_count"] += 1

            elif self.cache["variational_f_base_count"] == self.cache["KL_qU_pU_count"]:
                mean_qU, chol_cov_qU, cov_qU = self.qU.forward()
                self.cache["mean_qU"] = mean_qU
                self.cache["chol_cov_qU"] = chol_cov_qU
                self.cache["cov_qU"] = cov_qU
                self.cache["variational_f_base_count"] += 1

            elif self.cache["variational_f_base_count"] > self.cache["KL_qU_pU_count"]:
                raise NotImplementedError("It seems that you call variational_f_base in GP_with_qU twice without calling KL_qU_pU in between. ")

        else:
            mean_qU, chol_cov_qU, cov_qU = self.qU.forward()

        _qU_pick_mean = self.qU.pick_mean(mean_qU=mean_qU, output_idx=output_idx)  # [..., (P), M]
        _qU_pick_cov = self.qU.pick_cov(cov_qU=cov_qU, output_idx=output_idx)
        _qU_pick_chol_cov = self.qU.pick_cov(cov_qU=chol_cov_qU, output_idx=output_idx)

        def _whitening_variational_f(K_uu, K_fu, K_ff, mean_func_at_f):
            # M = K_uu.size(-1)  # number of inducing points
            Lu = self.chol_cov_pU(_K_uu=K_uu)  # [..., M, M]
            Lu_inv_Kuf = torch.linalg.solve_triangular(Lu, K_fu.mT, upper=False)  # [..., M, n]

            # TODO: debug
            # change the ordering of the two following lines will result in different training loss
            Lu_inv_Kuf_T = Lu_inv_Kuf.mT
            Lu_inv_Kuf_square = Lu_inv_Kuf.square()

            variational_mean = (Lu_inv_Kuf_T @ _qU_pick_mean.unsqueeze(-1)).squeeze(-1)  # [..., n]
            # variational_mean = (Lu_inv_Kuf.mT @ _qU_pick_mean.unsqueeze(-1)).squeeze(-1)  # [..., n]

            if mean_func_at_f is not None:
                variational_mean = variational_mean + mean_func_at_f  # [..., n]

            Kfu_Kuu_inv_K_uf_diag = Lu_inv_Kuf_square.sum(dim=(-2))
            # Kfu_Kuu_inv_K_uf_diag = Lu_inv_Kuf.square().sum(dim=(-2))  # [..., n]
            D_diag = K_ff - Kfu_Kuu_inv_K_uf_diag  # Bui's notation, [..., n]

            self.cache['D_diag'] = D_diag  # for tighter bound with Gaussian lik.

            # Approach 1:
            # cov_qU = _qU_pick_cov.unsqueeze(-3)  # [..., 1, M, M]
            # Kfu_Lu_T_inv_rearrange = Lu_inv_Kuf.mT.unsqueeze(-2)  # [..., n, 1, M]
            # _variational_term = (
            #     Kfu_Lu_T_inv_rearrange @ cov_qU @ Kfu_Lu_T_inv_rearrange.mT
            # ).squeeze(-1).squeeze(-1)  # [..., n]
            ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### ---

            # Approach 2:
            Ls_T_Lu_inv_K_uf = _qU_pick_chol_cov.mT @ Lu_inv_Kuf  # [..., M, n]
            _variational_term = Ls_T_Lu_inv_K_uf.square().sum(dim=(-2))  # [..., n]
            ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### ---

            # assert torch.allclose(_variational_term, _variational_term_old), \
            #     "Two approaches to compute the variational covariance term do not match!"

            variational_cov = self.m_spherical * D_diag + _variational_term   # [..., n]

            return variational_mean, variational_cov

        def _nonwhitening_variational_f(K_uu, K_fu, K_ff, mean_func_at_f):
            # Approach 1:
            """
            Kuu_inv_Kuf = torch.linalg.solve(K_uu, K_fu.mT)   # [..., M, n]

            variational_mean1 = (Kuu_inv_Kuf.mT @ _qU_pick_mean.unsqueeze(-1)).squeeze(-1)  # [..., n]

            if mean_func_at_f is not None:
                variational_mean1 = variational_mean1 + mean_func_at_f # [..., n]

            Kfu_Kuu_inv_rearrange = Kuu_inv_Kuf.mT.unsqueeze(-2)   # [..., n, 1, M]
            Kfu_Kuu_inv_K_uf_diag = (Kuu_inv_Kuf * K_fu.mT).sum(dim=(-2)) # [..., n]
            D_diag1 = K_ff - Kfu_Kuu_inv_K_uf_diag  # Bui's notation, [..., n]

            self.cache['D_diag'] = D_diag1  # for tighter bound with Gaussian lik.

            cov_qU = _qU_pick_cov.unsqueeze(-3)  # [..., 1, M, M]

            _variational_term1 = (
                Kfu_Kuu_inv_rearrange @ cov_qU @ Kfu_Kuu_inv_rearrange.mT
            ).squeeze(-1).squeeze(-1)   # [..., n]

            variational_cov1 = self.m_spherical * D_diag1 + _variational_term1  # [..., n]
            ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### ---
            ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### ---

            # Approach 2:
            """
            Lu = self.chol_cov_pU(_K_uu=K_uu)  # [..., M, M]
            Lu_inv_Kuf = torch.linalg.solve_triangular(Lu, K_fu.mT, upper=False)  # [..., M, n]

            Lu_inv_mean_qU = torch.linalg.solve_triangular(Lu, _qU_pick_mean.unsqueeze(-1), upper=False).squeeze(-1)  # [..., M]
            variational_mean2 = (Lu_inv_Kuf.mT @ Lu_inv_mean_qU.unsqueeze(-1)).squeeze(-1)  # [..., n]

            if mean_func_at_f is not None:
                variational_mean2 = variational_mean2 + mean_func_at_f # [..., n]

            D_diag2 = K_ff - Lu_inv_Kuf.square().sum(dim=(-2))  # Bui's notation, [..., n]

            self.cache['D_diag'] = D_diag2  # for tighter bound with Gaussian lik.

            Lu_inv_Ls = torch.linalg.solve_triangular(Lu, _qU_pick_chol_cov, upper=False)  # [..., M, M]
            K_uf_Lu_inv_T_Lu_inv_Ls = Lu_inv_Kuf.mT @ Lu_inv_Ls  # [..., n, M]
            _variational_term2 = K_uf_Lu_inv_T_Lu_inv_Ls.square().sum(dim=(-1))  # [..., n]

            variational_cov2 = self.m_spherical * D_diag2 + _variational_term2

            # assert torch.allclose(variational_mean1, variational_mean2), \
            #     "Two approaches to compute the variational mean do not match!"
            # assert torch.allclose(variational_cov1, variational_cov2), \
            #     "Two approaches to compute the variational covariance do not match!"
            
            return variational_mean2, variational_cov2
            # return variational_mean1, variational_cov1

        if self.whitening:
            variational_mean, variational_cov = _whitening_variational_f(K_uu, K_fu, K_ff, mean_func_at_f)

        else:
            variational_mean, variational_cov = _nonwhitening_variational_f(K_uu, K_fu, K_ff, mean_func_at_f)

        return variational_mean, variational_cov  # [..., n]


class svgp_base(GP_with_qU):
    """
    (batched) multi-output SVGP.
    NOTE: We assume each output has independent inducing points, inducing variables and kernel functions.
        Used for LMC, ind_svgp
    """
    def __init__(
        self, num_outputs: int, kernel: Kernel, Z: Inducing_points, qU: mo_Variational_inducing_dist,
        whitening: bool = True, jitter: float = 1e-6,
    ):
        super(svgp_base, self).__init__(
            kernel=kernel, Z=Z, qU=qU, whitening=whitening, jitter=jitter,
        )
        self.num_outputs = num_outputs  # for LMC: num of latent functions

        self.M = self.Z.M   # number of inducing points

        # each output has its own inducing points, inducing variables and kernel functions.
        assert Z.inducing_points.size(-3) == qU.batch_shape[-1] == kernel.batch_shape[-1] == num_outputs

    def variational_f(self, x: Tensor, output_idx: Optional[LongTensor] = None):
        r"""
        q(f) = \int p(f|U) q(U) dU;

        b: mini-batch size
        x: [..., b, D_X]
        output_idx: [select_num_outputs], the indices of outputs to be selected, if None, all outputs are selected.
        return:
            qf_mean, qf_cov: [..., select_num_outputs, b]
        """
        b, batch_shape = x.size(-2), x.shape[:-2]

        # Prepare
        _x = x.unsqueeze(-3)   # [..., 1, b, D_X]
        if output_idx == None:
            output_idx = torch.arange(self.num_outputs, device=x.device)
        x = _x.expand(*batch_shape, len(output_idx), *_x.shape[-2:])   # [..., select_num_outputs, b, D_X]

        K_uu = self.kernel.forward(
            x1=self.Z.inducing_points[..., output_idx, :, :],
            x2=self.Z.inducing_points[..., output_idx, :, :],
            output_idx=output_idx
        )   # [..., select_num_outputs, M, M]
        K_fu = self.kernel.forward(
            x1=x,
            x2=self.Z.inducing_points[..., output_idx, :, :],
            output_idx=output_idx
        )   # [..., select_num_outputs, b, M]
        K_ff = self.kernel.forward(
            x1=x,
            x2=x,
            output_idx=output_idx, diag=True
        )   # [..., select_num_outputs, b]

        qf_means, qf_covs = self.variational_f_base(K_uu, K_fu, K_ff, output_idx)   # [..., select_num_outputs, b]

        return qf_means, qf_covs