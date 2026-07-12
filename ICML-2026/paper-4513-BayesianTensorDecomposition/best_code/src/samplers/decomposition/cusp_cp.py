import torch
import opt_einsum as oe
import torch.distributions as dist
import numpy as np
from torch import nn
from src.utils import patches2image, image2patches

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import PowerTransform


def make_psd(A, eps=1e-6):
    # A: (..., n, n)
    A = 0.5 * (A + A.transpose(-1, -2))
    w, V = torch.linalg.eigh(A)
    w_clamped = w.clamp_min(eps)
    A_psd = (V * w_clamped.unsqueeze(-2)) @ V.transpose(-1, -2)
    A_psd = 0.5 * (A_psd + A_psd.transpose(-1, -2))
    return A_psd


def make_pd_by_cholesky(A, jitter=1e-6, max_tries=10):
    A = 0.5 * (A + A.transpose(-1, -2))
    n = A.size(-1)
    I = torch.eye(n, device=A.device, dtype=A.dtype)

    j = jitter
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(A + j * I)
            return A + j * I, L
        except RuntimeError:
            j *= 10
    A_pd = make_psd(A, eps=j)
    L = torch.linalg.cholesky(A_pd)
    return A_pd, L


# Older torch has no InverseGamma distribution.
class InverseGamma(TransformedDistribution):
    r"""
    Creates an inverse gamma distribution parameterized by :attr:`concentration` and :attr:`rate`
    where::

        X ~ Gamma(concentration, rate)
        Y = 1 / X ~ InverseGamma(concentration, rate)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = InverseGamma(torch.tensor([2.0]), torch.tensor([3.0]))
        >>> m.sample()
        tensor([ 1.2953])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """

    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Gamma

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        neg_one = -base_dist.rate.new_ones(())
        super().__init__(
            base_dist, PowerTransform(neg_one), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self) -> Tensor:
        return self.base_dist.concentration

    @property
    def rate(self) -> Tensor:
        return self.base_dist.rate

    @property
    def mean(self) -> Tensor:
        result = self.rate / (self.concentration - 1)
        return torch.where(self.concentration > 1, result, torch.inf)

    @property
    def mode(self) -> Tensor:
        return self.rate / (self.concentration + 1)

    @property
    def variance(self) -> Tensor:
        result = self.rate.square() / (
            (self.concentration - 1).square() * (self.concentration - 2)
        )
        return torch.where(self.concentration > 2, result, torch.inf)

    def entropy(self):
        return (
            self.concentration
            + self.rate.log()
            + self.concentration.lgamma()
            - (1 + self.concentration) * self.concentration.digamma()
        )


def khatri_rao(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: (m, k)
    B: (n, k)
    returns: (m*n, k)
    """
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError("A and B must be 2D")
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns (k)")

    # (m, 1, k) * (1, n, k) -> (m, n, k) -> reshape to (m*n, k)
    return (A[:, None, :] * B[None, :, :]).reshape(-1, A.shape[1])


class CUSP_CP(nn.Module):
    def __init__(
        self, *,
        shape,
        orig_shape,
        init_rank: int = 10,
        init_method: str = 'randn',
        init_scale: float = 0.1,
        tau_alpha: float = 1e-3,
        tau_beta: float = 1e-3,
        nu_beta: float = 5.0,
        theta_alpha: float = 2.0,
        theta_beta: float = 2.0,
        theta_infty: float = 1e-2,
        use_patch: bool = True,
        patch_size: int = 16,
        stride: int = 8,
        **kwargs
    ):
        super(CUSP_CP, self).__init__()
        if not use_patch:
            self.shape = list(shape)
        else:
            pseudo_x = torch.zeros(list(orig_shape))
            patches = image2patches(
                pseudo_x, patch_size=patch_size, stride=stride)
            self.shape = list(patches.shape)
        self.orig_shape = list(orig_shape)
        self.init_rank = init_rank
        self.init_method = init_method
        self.init_scale = init_scale
        self.use_patch = use_patch
        self.patch_size = patch_size
        self.stride = stride
        self.img_size = orig_shape[1:]

        self.cp_factors = nn.ParameterList([
            nn.Parameter(torch.empty(dim, init_rank), requires_grad=False) for dim in self.shape
        ])
        self.cp_lambda = nn.Parameter(torch.ones(init_rank) / init_rank, requires_grad=False)

        self._init_factors()

        # hyper priors
        self.prior_tau = dist.Gamma(concentration=tau_alpha, rate=tau_beta)
        self.prior_nu = dist.Beta(concentration1=1.0, concentration0=nu_beta)
        self.prior_0 = InverseGamma(concentration=theta_alpha, rate=theta_beta)
        self.theta_infty = theta_infty

        # sample hyper parameters
        tau = self.prior_tau.sample()
        nu = []
        for _ in range(self.init_rank - 1):
            nu_i = self.prior_nu.sample()
            nu.append(nu_i)
        nu.append(torch.tensor(1.0))
        nu = torch.stack(nu)
        omega = self._cummulative_product(nu)
        zeta = self._sample_zeta(omega)
        theta = self._sample_theta(zeta)
        self.tau = nn.Parameter(tau, requires_grad=False)
        self.nu = nn.Parameter(nu, requires_grad=False)
        self.omega = nn.Parameter(omega, requires_grad=False)
        self.zeta = nn.Parameter(zeta, requires_grad=False)
        self.theta = nn.Parameter(theta, requires_grad=False)

        self.post_samples = None
        self.gibbs_n_iter = 0
        self.truncate_b0 = 1.0
        self.truncate_b1 = 0.005

    def _init_factors(self):
        for factor in self.cp_factors:
            if self.init_method == 'randn':
                nn.init.normal_(factor, mean=0.0, std=self.init_scale)
            elif self.init_method == 'uniform':
                nn.init.uniform_(factor, a=-self.init_scale, b=self.init_scale)
            else:
                raise ValueError(f"Unknown initialization method: {self.init_method}")

    def forward(self, reshape=True):
        dim = len(self.shape)
        r = oe.get_symbol(2 * dim)
        in_str = []
        out_str = ''
        for i in range(dim):
            assert oe.get_symbol(i) != r
            in_str.append(f"{oe.get_symbol(i)}{r}")
            out_str += oe.get_symbol(i)
        in_str = ','.join(in_str)
        expr = f"{in_str}, {r} -> {out_str}"
        if reshape:
            if self.use_patch:
                return patches2image(
                    oe.contract(expr, *self.cp_factors, self.cp_lambda),
                    self.img_size, (self.patch_size, self.patch_size), self.stride)
            else:
                return oe.contract(expr, *self.cp_factors, self.cp_lambda).reshape(self.orig_shape)
        else:
            return oe.contract(expr, *self.cp_factors, self.cp_lambda)

    def compute_single_factor(self, rank):
        dim = len(self.shape)
        in_str = []
        factors = []
        out_str = ''
        for i in range(dim):
            in_str.append(oe.get_symbol(i))
            factors.append(self.cp_factors[i][:, rank])
            out_str += oe.get_symbol(i)
        in_str = ','.join(in_str)
        expr = f"{in_str} -> {out_str}"
        return oe.contract(expr, *factors)

    def gibbs(self, data, z_ten, mask, rho, n_iter=200, burnin=100, jump=2):
        device = self.cp_factors[0].device

        post_samples = {
            'tau': [],
            'nu': [],
            'omega': [],
            'zeta': [],
            'theta': [],
            'lambda': [],
            'factors': [],
        }

        # gibbs sampler
        if self.use_patch:
            data = image2patches(data.clone().to(device), patch_size=self.patch_size, stride=self.stride)
            mask = image2patches(mask.clone().to(device), patch_size=self.patch_size, stride=self.stride)
            z_ten = image2patches(z_ten.clone().to(device), patch_size=self.patch_size, stride=self.stride)
        else:
            data = data.clone().to(device).reshape(*self.shape)
            mask = mask.clone().to(device).reshape(*self.shape)
            z_ten = z_ten.clone().to(device).reshape(*self.shape)
        for it in range(n_iter + 1):
            self._post_sample_lambda(rho, data, z_ten, mask)
            self._post_sample_factors(rho, data, z_ten, mask)
            self._post_sample_tau(data, mask)
            self._post_sample_zeta()
            self._post_sample_nu()
            self._post_sample_theta()
            omega = self._cummulative_product(self.nu)
            self.omega.data = omega
            if it < burnin:
                self._adapt_tune()
            if it > burnin and (it - burnin) % jump == 0:
                post_samples['tau'].append(self.tau.cpu())
                post_samples['nu'].append(self.nu.cpu())
                post_samples['omega'].append(self.omega.cpu())
                post_samples['zeta'].append(self.zeta.cpu())
                post_samples['theta'].append(self.theta.cpu())
                post_samples['lambda'].append(self.cp_lambda.cpu())
                post_samples['factors'].append([f.cpu() for f in self.cp_factors])
                # x_hat = self.forward().detach().cpu()
                # x_hat = torch.clamp(x_hat + 0.5, 0.0, 1.0).reshape([256, 256, 3])
            error = ((data - self.forward(reshape=False))[mask == 1] ** 2).mean().sqrt().item()
            print(f"[Iter {it + 1}/{n_iter}, RMSE: {error:.6f}, Rank: {len(self.cp_lambda)}]")
            self.gibbs_n_iter += 1

        self.post_samples = post_samples

    def _cummulative_product(self, nu):
        device = nu[0].device
        omega = torch.cumprod(1 - nu, dim=0)
        omega = torch.cat([torch.tensor([1.0]).to(device), omega[:-1]], dim=0)
        omega = omega * nu
        return omega

    def _cummulative_sum(self, omega):
        return torch.cumsum(omega, dim=0)

    # def _sample_zeta(self, pi):
    #     zeta = torch.rand_like(pi)
    #     zeta[zeta > pi] = 1.0
    #     zeta[zeta <= pi] = 0.0
    #     zeta = 1 - zeta
    #     return zeta

    def _sample_zeta(self, omega):
        # ensure normalization
        prob = omega / omega.sum()
        # prob = torch.cat([omega, 1 - omega.sum()], dim=0)
        zeta = torch.zeros_like(omega)
        for r in range(len(omega)):
            zeta[r] = dist.Categorical(probs=prob).sample()
        return zeta

    def _sample_theta(self, zeta):
        theta = torch.ones_like(zeta) * self.theta_infty
        for r in range(len(theta)):
            if zeta[r] <= r:
                theta[r] = self.prior_0.sample().to(theta.device)
        return theta

    def _post_sample_lambda(self, rho, data, z_ten, mask):
        theta = self.theta
        tau = self.tau
        rank = len(self.cp_lambda)
        lambda_post = torch.zeros(rank, device=self.cp_lambda.device)
        x_hat = self.forward(reshape=False)  # TODO: Should we put it inside the loop?
        for r in range(rank):
            factor_r = self.compute_single_factor(r)  # C^r
            resid_r = data - x_hat + self.cp_lambda[r] * factor_r  # Y - D^r
            resid_r_z = z_ten - x_hat + self.cp_lambda[r] * factor_r  # Z - D^r

            factor_r_mask = factor_r[mask == 1].view(-1)
            resid_r_mask = resid_r[mask == 1].view(-1)

            sigma = 1.0 / (
                theta[r] + tau * (factor_r_mask ** 2).sum() + (factor_r ** 2).sum() / rho ** 2)
            mu = sigma * (
                (tau * factor_r_mask * resid_r_mask).sum() + 
                (factor_r * resid_r_z / rho ** 2).sum()
            )

            lambda_post[r] = dist.Normal(loc=mu, scale=sigma.sqrt()).sample()
        self.cp_lambda.data = lambda_post

    def _post_sample_factors(self, rho, data, z_ten, mask):
        tau = self.tau
        dim = len(self.shape)
        rank = len(self.cp_lambda)
        for i in range(dim):
            factor_i = self.cp_factors[i].clone()  # I x R

            other_factors = [self.cp_factors[j] for j in range(dim) if j != i]
            B = khatri_rao(other_factors[-2], other_factors[-1])
            for j in range(3, dim):
                B = khatri_rao(other_factors[-j], B)
            B = oe.contract('r, ir -> ri', self.cp_lambda, B)
            BBT = oe.contract('ri, Ri -> rR', B, B)

            mask_i = mask.permute(
                i, *[j for j in range(dim) if j != i]
            ).contiguous().view(self.shape[i], -1)
            data_i = data.permute(
                i, *[j for j in range(dim) if j != i]
            ).contiguous().view(self.shape[i], -1)
            Z_i = z_ten.permute(
                i, *[j for j in range(dim) if j != i]
            ).contiguous().view(self.shape[i], -1)

            for idx in range(factor_i.shape[0]):

                Bi = B[:, mask_i[idx] == 1]
                sigma = torch.linalg.inv(
                    torch.eye(rank, device=factor_i.device) +
                    tau * oe.contract('ri, Ri -> rR', Bi, Bi) + BBT / rho ** 2
                )
                yi = data_i[idx, mask_i[idx] == 1]
                mu = sigma @ (
                    tau * oe.contract('i, ri -> r', yi, Bi) +
                    oe.contract('i, ri -> r', Z_i[idx], B) / rho ** 2
                )

                sigma = (sigma + sigma.T) / 2.0
                try:
                    factor_i[idx] = dist.MultivariateNormal(
                        loc=mu, covariance_matrix=sigma).sample()
                except ValueError:  # in case covariance_matrix is not PD
                    # sigma = sigma + 1e-6 * torch.eye(
                    #     sigma.shape[0], device=factor_i.device, dtype=sigma.dtype)
                    _, L = make_pd_by_cholesky(sigma, jitter=1e-6, max_tries=5)
                    factor_i[idx] = dist.MultivariateNormal(
                        loc=mu, scale_tril=L).sample()
            self.cp_factors[i].data = factor_i

    def _post_sample_tau(self, data, mask):
        resid = data - self.forward(reshape=False)
        resid = resid[mask == 1].view(-1)
        alpha_post = self.prior_tau.concentration + 0.5 * resid.numel()
        beta_post = self.prior_tau.rate + 0.5 * (resid ** 2).sum()
        tau = dist.Gamma(concentration=alpha_post, rate=beta_post).sample()
        self.tau.data = tau

    def _post_sample_zeta(self):
        omega = self.omega
        normal_pdf = dist.Normal(
            loc=0.0, scale=self.theta_infty).log_prob(self.cp_lambda).exp()
        student_t_pdf = dist.StudentT(
            df=2.0 * self.prior_0.concentration,
            loc=0.0,
            scale=(self.prior_0.rate / self.prior_0.concentration).sqrt()
        ).log_prob(self.cp_lambda).exp()

        # CODE-01 fix: Enforce monotonic stick-breaking semantics.
        # The CUSP prior requires zeta to be monotonic: once a component is "small"
        # (zeta=1), all subsequent components must also be small. This means we
        # sample a single cutoff index k, not independent zeta values.
        # zeta[r] = 0 (large) for r < k, zeta[r] = 1 (small) for r >= k.
        # k = rank means all components are large.
        rank = len(omega)
        log_prob_k = torch.zeros(rank + 1, device=omega.device)
        for k in range(rank + 1):
            # Components r < k are large (student-t slab)
            # Components r >= k are small (normal spike)
            lp = 0.0
            for r in range(rank):
                if r < k:
                    lp += (student_t_pdf[r] + 1e-10).log()
                else:
                    lp += (normal_pdf[r] + 1e-10).log()
            # Prior: P(k) = omega[k] for k < rank, P(k=rank) = 1 - sum(omega)
            prior_k = omega[k].clamp(min=1e-10).log() if k < rank else (1.0 - omega.sum()).clamp(min=1e-10).log()
            log_prob_k[k] = lp + prior_k
        
        prob_k = torch.softmax(log_prob_k, dim=0)
        k = dist.Categorical(probs=prob_k).sample().item()
        
        zeta = torch.zeros(rank, device=omega.device)
        zeta[k:] = 1.0  # components from k onward are "small"
        self.zeta.data = zeta

    def _post_sample_nu(self):
        zeta = self.zeta
        rank = len(zeta)
        nu_post = torch.ones(rank, device=zeta.device)
        for r in range(rank - 1):
            alpha_post = self.prior_nu.concentration1 + (zeta == r).sum()
            beta_post = self.prior_nu.concentration0 + (zeta > r).sum()
            nu_post[r] = dist.Beta(concentration1=alpha_post, concentration0=beta_post).sample()
        self.nu.data = nu_post

    def _post_sample_theta(self):
        zeta = self.zeta
        theta_post = torch.ones_like(zeta) * self.theta_infty
        for r in range(len(theta_post)):
            if zeta[r] == 0:
                alpha_post = self.prior_0.concentration + 0.5
                beta_post = self.prior_0.rate + 0.5 * (self.cp_lambda[r] ** 2)
                theta_post[r] = InverseGamma(concentration=alpha_post, rate=beta_post).sample()
        self.theta.data = theta_post

    def _adapt_tune(self):
        device = self.cp_lambda.device
        prob = 1. / np.exp(self.truncate_b0 + self.truncate_b1 * self.gibbs_n_iter)
        if np.random.rand() < prob:
            # preserve_idx = (self.zeta.data >= torch.arange(len(self.zeta)).to(device)).nonzero(as_tuple=False).squeeze()
            preserve_idx = (self.cp_lambda.data.abs() >= 1e-4).nonzero(as_tuple=False).squeeze()

            # truncate
            self.cp_lambda.data = self.cp_lambda.data[preserve_idx]
            for d in range(len(self.cp_factors)):
                self.cp_factors[d].data = self.cp_factors[d].data[:, preserve_idx]
            self.nu.data = self.nu.data[preserve_idx]
            self.omega.data = self.omega.data[preserve_idx]
            self.zeta.data = self.zeta.data[preserve_idx]
            self.theta.data = self.theta.data[preserve_idx]

            # add new ones
            for d in range(len(self.cp_factors)):
                dim = self.cp_factors[d].shape[0]
                new_factor = torch.randn(dim, 1).to(device) * self.init_scale
                self.cp_factors[d].data = torch.cat([self.cp_factors[d].data, new_factor], dim=1)
            nu = self.prior_nu.sample().to(device)
            self.nu.data = torch.cat([self.nu.data, nu.unsqueeze(0)], dim=0)
            self.omega.data = self._cummulative_product(self.nu)
            zeta_new = self._sample_zeta(self.omega)
            self.zeta.data = torch.cat([self.zeta.data, zeta_new[-1].unsqueeze(0)], dim=0)
            theta_new = self._sample_theta(self.zeta)
            self.theta.data = torch.cat([self.theta.data, theta_new[-1].unsqueeze(0)], dim=0)
            lambda_new = torch.randn(1).to(device) * theta_new[-1]
            self.cp_lambda.data = torch.cat([self.cp_lambda.data, lambda_new], dim=0)

        else:
            pass
