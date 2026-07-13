import torch
import numpy as np
from scipy.signal import welch
from torch.distributions import Transform, constraints

import torch.distributions as D


class RGBAlpha(D.Distribution):
    arg_constraints = {}
    support = D.constraints.real
    has_rsample = False  # no reparameterization

    def __init__(self, validate_args=None):
        super().__init__(validate_args=validate_args)
        # alpha ~ Beta(1/2, 1/2)  (arcsine law)
        self.alpha_dist = D.Beta(torch.tensor(0.5), torch.tensor(0.5))
        # R,G,B ~ Uniform(0,255)
        self.rgb_dist = D.Uniform(torch.tensor(0.0), torch.tensor(255.0))

    def sample(self, sample_shape=torch.Size()):
        # R,G,B uniform(0,255)
        rgb = self.rgb_dist.sample(sample_shape + (3,))
        # alpha ~ Beta(0.5,0.5), supported in (0,1)
        alpha = self.alpha_dist.sample(sample_shape + (1,))
        return torch.cat([rgb, alpha], dim=-1)

    def log_prob(self, value):
        rgb, alpha = value[..., :3], value[..., 3]
        # log-prob of uniform RGB
        rgb_log_prob = torch.where(
            (rgb >= 0) & (rgb <= 255),
            -torch.log(torch.tensor(255.0)),
            torch.full_like(rgb, float("-inf")),
        )
        rgb_log_prob = rgb_log_prob.sum(-1)
        # log-prob of alpha ~ Beta(0.5,0.5)
        alpha_log_prob = self.alpha_dist.log_prob(alpha)
        return rgb_log_prob + alpha_log_prob


class CosineSquaredAngleDiff(torch.distributions.Distribution):
    def __init__(self, base_dist, device="cpu"):
        super().__init__()
        self.uniform = base_dist
        self.device = device

    def sample(self, sample_shape=torch.Size()):
        # Sample theta_1 and theta_2 in degrees
        R, G, B, theta1, theta2 = self.uniform.sample(sample_shape).T

        # Convert to radians
        theta1_rad = torch.deg2rad(theta1)
        theta2_rad = torch.deg2rad(theta2)

        # Compute cos^2(theta1 - theta2)
        diff = theta1_rad - theta2_rad
        cos_squared = torch.cos(diff) ** 2
        return R, G, B, cos_squared

    def log_prob(self, value):
        # Not a standard distribution, so we can't easily define log_prob
        raise NotImplementedError("log_prob is not implemented for this custom distribution.")


class TriangularTransform(Transform):
    domain = constraints.unit_interval
    codomain = constraints.real
    bijective = False
    sign = +1

    def __init__(self):
        super().__init__()

    def __call__(self, uvs) -> torch.Tensor:
        u, v = uvs[..., 0], uvs[..., 1]
        x = torch.sqrt(u) / 2
        y = v * x
        return torch.stack([x, y], dim=-1)

    def log_abs_det_jacobian(self, uvs, _):
        u = uvs[..., 0]
        return torch.full_like(u, torch.log(torch.tensor(1 / 8.0)))

    def inv(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        u = (2 * x) ** 2
        v = y / x
        return torch.stack([u, v], dim=-1)


class IdentityTransform(Transform):
    """
    Identity transform that does not change the input.
    Useful for cases where no transformation is needed.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inv(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def log_abs_det_jacobian(self, x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), device=x.device)


class LogitBoxTransform(Transform):
    """
    Transforms a vector of independent uniform variables on [a_i, b_i]
    to R^n using a per-dimension logit transform.
    """

    domain = constraints.dependent  # Each dim is bounded between a_i and b_i
    codomain = constraints.real_vector
    bijective = True
    sign = +1

    def __init__(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.a = a
        self.b = b
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Rescale from [a, b] to [0, 1]
        x_unit = (x - self.a) / (self.b - self.a)
        x_unit = x_unit.clamp(self.eps, 1 - self.eps)
        return torch.log(x_unit / (1 - x_unit))

    def inv(self, y: torch.Tensor) -> torch.Tensor:
        # Sigmoid maps R -> [0, 1]
        x_unit = torch.sigmoid(y)
        return self.a + (self.b - self.a) * x_unit

    def log_abs_det_jacobian(self, x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        x_unit = (x - self.a) / (self.b - self.a)
        x_unit = x_unit.clamp(self.eps, 1 - self.eps)
        log_jac = -torch.log(x_unit) - torch.log(1 - x_unit) - torch.log(self.b - self.a)
        return log_jac.sum(-1)  # sum over dimensions


class PowerSpecDens:
    def __init__(self, fs=1.0, nbins=32, logscale=True, device="cpu", dtype=torch.float32):
        """
        Power Spectral Density (PSD) estimator.

        Parameters
        ----------
        fs : float
            Sampling frequency of the time series.
        nbins : int
            Number of bins (frequency resolution).
        logscale : bool
            If True, return log10(PSD).
        device : str
            Torch device ('cpu' or 'cuda').
        dtype : torch.dtype
            Torch dtype (default float32).
        """
        self.fs = fs
        self.nbins = nbins
        self.logscale = logscale
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        """
        Compute the PSD of one or more time series.

        Parameters
        ----------
        x : array_like or torch.Tensor
            Input signal(s), shape (time,) or (batch, time).

        Returns
        -------
        freqs : torch.Tensor, shape (nbins+1,)
            Frequency bins.
        psd : torch.Tensor, shape (batch, nbins+1)
            Power spectral density.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        # Ensure 2D: (batch, time)
        if x.ndim == 1:
            x = x[None, :]

        psd_list = []
        freqs = None

        for sig in x:
            f, p = welch(sig, fs=self.fs, nperseg=self.nbins * 2)
            if self.logscale:
                p = np.log10(p + 1e-12)
            psd_list.append(p)

            if freqs is None:
                freqs = f

        psd = np.stack(psd_list, axis=0)  # (batch, freqs)

        # Convert to torch
        freqs = torch.tensor(freqs, device=self.device, dtype=self.dtype)
        psd = torch.tensor(psd, device=self.device, dtype=self.dtype)

        return freqs, psd
