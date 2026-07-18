import torch
from torch.nn.functional import normalize
from torch import vmap


def batch_trace(matrices):
    """Return the trace of each matrix in a batch."""
    return torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1)


def _interpolated_covariances(variances, t, dim, *, dtype, device):
    """Return covariance matrices for the Gaussian interpolant at time t."""
    identity = torch.eye(dim, dtype=dtype, device=device)
    return (1 - t) * (1 - t) * identity + variances * t * t


class MultivariateGaussianMixture_Analytical:
    def __init__(self, weights, locs, vars, tol=1e-9):
        """Initialize an analytically tractable Gaussian-mixture interpolant."""
        self.weights = normalize(torch.abs(weights), 1, 0)
        self.d = locs.shape[1]
        self.means = locs
        self.vars = vars
        self.tol = tol
        self.prepare_vec_fcts()

    def _single_terms(self, x, t):
        """Return reusable Gaussian terms for a single point and time."""
        covariances = _interpolated_covariances(
            self.vars, t, self.d, dtype=x.dtype, device=x.device
        )
        inv_covariances = torch.linalg.inv(covariances)
        dets = torch.det(covariances)
        norm_coefs = 1 / (torch.sqrt(dets) * (torch.pi * 2) ** (self.d / 2))
        xdisp = x.unsqueeze(0).expand(self.means.shape) - self.means * t
        trans = torch.bmm(inv_covariances, xdisp.unsqueeze(-1)).squeeze(-1)
        mahalanobis = torch.sum(xdisp * trans, -1).squeeze()
        probs = torch.exp(-mahalanobis / 2)
        weights = norm_coefs * self.weights * probs
        coeffs = normalize(weights, 1, 0)
        return covariances, inv_covariances, xdisp, trans, weights, coeffs

    def _single_log_prob(self, x, t):
        """Evaluate the log density of the interpolated mixture at one point."""
        _, _, _, _, weights, _ = self._single_terms(x, t)
        return torch.log(torch.sum(weights) + self.tol)

    def _single_score(self, x, t):
        """Evaluate the score of the interpolated mixture at one point."""
        _, _, _, trans, _, coeffs = self._single_terms(x, t)
        return -torch.sum(coeffs.unsqueeze(-1) * trans, 0)

    def _single_b(self, x, t):
        """Evaluate the stochastic-interpolant drift at one point."""
        _, _, _, trans, _, coeffs = self._single_terms(x, t)
        coeffs = coeffs.unsqueeze(-1)
        genx = x.unsqueeze(0).expand(trans.shape)
        return torch.sum(coeffs * (genx - (1 - t) * trans), 0) / t

    def _single_grad_score(self, x, t):
        """Evaluate the Jacobian of the score at one point."""
        _, inv_covariances, _, trans, _, coeffs = self._single_terms(x, t)
        weighted_trans = coeffs.unsqueeze(-1) * trans
        mscore = torch.sum(weighted_trans, 0)
        qsc = torch.tensordot(mscore, mscore, 0)
        qtr = torch.bmm(trans[:, :, torch.newaxis], trans[:, torch.newaxis, :])
        qtr = qtr - inv_covariances
        return -qsc + torch.sum(coeffs.view(-1, 1, 1) * qtr, 0)

    def _single_grad_div_score(self, x, t):
        """Evaluate the gradient of the score divergence at one point."""
        _, inv_covariances, _, trans, _, coeffs = self._single_terms(x, t)
        coeffs = coeffs.unsqueeze(-1)
        score = -torch.sum(coeffs * trans, 0)
        qsc = torch.tensordot(score, score, 0)
        qtr = torch.bmm(trans[:, :, torch.newaxis], trans[:, torch.newaxis, :])
        qtr = qtr - inv_covariances
        grad_score = -qsc + torch.sum(coeffs.unsqueeze(-1) * qtr, 0)
        tratrans = torch.bmm(inv_covariances, trans[:, :, torch.newaxis]).squeeze(-1)
        secoeff = (torch.sum(trans * trans, -1) - batch_trace(inv_covariances)).unsqueeze(-1)
        return -2 * torch.matmul(grad_score, score) + torch.sum(
            coeffs * (2 * tratrans - secoeff * (trans + score.unsqueeze(0))),
            dim=0,
        )

    def prepare_vec_fcts(self):
        """Create vectorized versions of the single-point analytical functions."""
        self.log_prob = vmap(self._single_log_prob, in_dims=(0, None), out_dims=0)
        self.score = vmap(self._single_score, in_dims=(0, None), out_dims=0)
        self.b = vmap(self._single_b, in_dims=(0, None), out_dims=0)
        self.grad_score = vmap(self._single_grad_score, in_dims=(0, None), out_dims=0)
        self.grad_div_score = vmap(
            self._single_grad_div_score, in_dims=(0, None), out_dims=0
        )


class GMM_itp_rugged:
    def __init__(self, weights, locs, vars, k, r, tol=1e-9):
        """Initialize a Gaussian-mixture interpolant with sinusoidal ruggedness."""
        self.weights = normalize(torch.abs(weights), 1, 0)
        self.d = locs.shape[1]
        self.means = locs
        self.vars = vars
        self.tol = tol
        self.nmix = len(weights)
        self.k = k
        self.r = r
        self.prepare_vec_fcts()

    def _single_terms(self, x, t):
        """Return reusable Gaussian terms for a single point and time."""
        covariances = _interpolated_covariances(
            self.vars, t, self.d, dtype=x.dtype, device=x.device
        )
        inv_covariances = torch.linalg.inv(covariances)
        dets = torch.det(covariances)
        norm_coefs = 1 / (torch.sqrt(dets) * (torch.pi * 2) ** (self.d / 2))
        xdisp = x.unsqueeze(0).expand(self.means.shape) - self.means * t
        trans = torch.bmm(inv_covariances, xdisp.unsqueeze(-1)).squeeze(-1)
        mahalanobis = torch.sum(xdisp * trans, -1).squeeze()
        probs = torch.exp(-mahalanobis / 2)
        weights = norm_coefs * self.weights * probs
        coeffs = normalize(weights, 1, 0)
        return covariances, inv_covariances, xdisp, trans, weights, coeffs

    def _rugged_log_factor(self, x, t):
        """Return the rugged sinusoidal log-density contribution."""
        return self.r * torch.sum(torch.sin(self.k * x)) * t

    def _rugged_score(self, x, t):
        """Return the gradient of the rugged sinusoidal contribution."""
        return self.r * torch.sum(self.k * torch.cos(self.k * x), dim=0) * t

    def _single_dt_terms(self, x, t):
        """Return reusable time-derivative Gaussian terms."""
        covariances, inv_covariances, xdisp, trans, _, coeffs = self._single_terms(x, t)
        identity = torch.eye(self.d, dtype=x.dtype, device=x.device)
        dot_covariances = 2 * self.vars * t - 2 * (1 - t) * identity
        dot_inv_covariances = torch.bmm(
            torch.bmm(inv_covariances, inv_covariances), dot_covariances
        )
        mean_velocity = self.means
        mean_term = torch.sum(mean_velocity * trans, -1).squeeze()
        covariance_term = -torch.sum(
            xdisp * torch.bmm(dot_inv_covariances, xdisp.unsqueeze(-1)).squeeze(-1),
            -1,
        ).squeeze() / 2
        det_term = batch_trace(torch.bmm(covariances, dot_inv_covariances)) / (
            2 * torch.sqrt(torch.det(covariances))
        )
        dt_log_terms = mean_term + covariance_term + det_term
        return inv_covariances, trans, coeffs, mean_velocity, dt_log_terms

    def _single_log_prob(self, x, t):
        """Evaluate the rugged log density at one point."""
        _, _, _, _, weights, _ = self._single_terms(x, t)
        return torch.log(torch.sum(weights) + self.tol) + self._rugged_log_factor(x, t)

    def _single_prob(self, x, t):
        """Evaluate the rugged density at one point."""
        _, _, _, _, weights, _ = self._single_terms(x, t)
        return torch.sum(weights) * torch.exp(self._rugged_log_factor(x, t))

    def _single_score(self, x, t):
        """Evaluate the score of the rugged interpolant at one point."""
        _, _, _, trans, _, coeffs = self._single_terms(x, t)
        return -torch.sum(coeffs.unsqueeze(-1) * trans, 0) + self._rugged_score(x, t)

    def _single_b(self, x, t):
        """Evaluate the stochastic-interpolant drift of the rugged model at one point."""
        _, _, _, trans, _, coeffs = self._single_terms(x, t)
        coeffs = coeffs.unsqueeze(-1)
        genx = x.unsqueeze(0).expand(trans.shape)
        return (
            torch.sum(coeffs * (genx - (1 - t) * trans), 0) / t
            + (1 - t) * self._rugged_score(x, 1.0)
        )

    def _single_grad_score(self, x, t):
        """Evaluate the Jacobian of the rugged score at one point."""
        _, inv_covariances, _, trans, _, coeffs = self._single_terms(x, t)
        weighted_trans = coeffs.unsqueeze(-1) * trans
        mscore = torch.sum(weighted_trans, 0)
        qsc = torch.tensordot(mscore, mscore, 0)
        qtr = torch.bmm(trans[:, :, torch.newaxis], trans[:, torch.newaxis, :])
        qtr = qtr - inv_covariances
        return -qsc + torch.sum(coeffs.view(-1, 1, 1) * qtr, 0)

    def _single_grad_div_score(self, x, t):
        """Evaluate the gradient of the rugged score divergence at one point."""
        _, inv_covariances, _, trans, _, coeffs = self._single_terms(x, t)
        coeffs = coeffs.unsqueeze(-1)
        score = -torch.sum(coeffs * trans, 0)
        qsc = torch.tensordot(score, score, 0)
        qtr = torch.bmm(trans[:, :, torch.newaxis], trans[:, torch.newaxis, :])
        qtr = qtr - inv_covariances
        grad_score = -qsc + torch.sum(coeffs.unsqueeze(-1) * qtr, 0)
        tratrans = torch.bmm(inv_covariances, trans[:, :, torch.newaxis]).squeeze(-1)
        secoeff = (torch.sum(trans * trans, -1) - batch_trace(inv_covariances)).unsqueeze(-1)
        return -2 * torch.matmul(grad_score, score) + torch.sum(
            coeffs * (2 * tratrans - secoeff * (trans + score.unsqueeze(0))),
            dim=0,
        )

    def _single_dt_log_prob(self, x, t):
        """Evaluate the time derivative of the rugged log density at one point."""
        _, _, coeffs, _, dt_log_terms = self._single_dt_terms(x, t)
        return torch.sum(coeffs * dt_log_terms)

    def _single_dt_score(self, x, t):
        """Evaluate the time derivative of the rugged score at one point."""
        inv_covariances, trans, coeffs, mean_velocity, dt_log_terms = (
            self._single_dt_terms(x, t)
        )
        first = -coeffs.view(self.nmix, 1) * dt_log_terms.view(self.nmix, 1) * trans
        second = coeffs.view(self.nmix, 1) * torch.bmm(
            inv_covariances, mean_velocity.unsqueeze(-1)
        ).squeeze(-1)
        third = -self._single_dt_log_prob(x, t) * self._single_score(x, t)
        return torch.sum(first + second, dim=0) + third

    def prepare_vec_fcts(self):
        """Create vectorized versions of the rugged single-point functions."""
        self.prob = vmap(self._single_prob, in_dims=(0, None), out_dims=0)
        self.log_prob = vmap(self._single_log_prob, in_dims=(0, None), out_dims=0)
        self.score = vmap(self._single_score, in_dims=(0, None), out_dims=0)
        self.b = vmap(self._single_b, in_dims=(0, None), out_dims=0)
        self.grad_score = vmap(self._single_grad_score, in_dims=(0, None), out_dims=0)
        self.grad_div_score = vmap(
            self._single_grad_div_score, in_dims=(0, None), out_dims=0
        )
        self.dt_log_prob = vmap(self._single_dt_log_prob, in_dims=(0, None), out_dims=0)
        self.dt_score = vmap(self._single_dt_score, in_dims=(0, None), out_dims=0)
