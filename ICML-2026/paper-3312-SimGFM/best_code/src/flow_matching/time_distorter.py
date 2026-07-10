import torch
import numpy as np
from scipy.stats import norm
from scipy.special import beta as beta_func, betaln
from scipy.stats import beta as sp_beta
from scipy.interpolate import interp1d


def beta_pdf(x, alpha, beta):
    """Beta distribution PDF."""
    # coeff = np.exp(betaln(alpha, beta))
    # return x ** (alpha - 1) * (1 - x) ** (beta - 1) / coeff
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / beta_func(alpha, beta)


def objective_function(alpha, beta, y, t):
    """Objective function to minimize (mean squared error)."""
    y_pred = beta_pdf(t, alpha, beta)
    regularization = (alpha + beta) + (1 / alpha + 1 / beta)
    error = np.mean((y - y_pred) ** 2)
    error = error + 0.0001 * regularization
    return error


class TimeDistorter:

    def __init__(
        self,
        train_distortion,
        sample_distortion,
    ):
        self.train_distortion = train_distortion  # used for sample_ft
        self.sample_distortion = sample_distortion  # used for get_ft
        print(
            f"TimeDistorter: train_distortion={train_distortion}, sample_distortion={sample_distortion}"
        )

    def _parse_name_and_params(self, spec: str):
        """Parse scheduler/distortion spec like 'name', 'name-k', or 'name(p1,p2,...)'.

        Returns (name, [params as floats]).
        """
        if '(' in spec and spec.endswith(')'):
            name = spec[: spec.find('(')]
            inside = spec[spec.find('(') + 1 : -1]
            parts = [p.strip() for p in inside.split(',') if p.strip() != '']
            try:
                params = [float(p) for p in parts]
            except ValueError:
                raise ValueError(f"Parameters in {spec} must be numeric.")
            return name, params
        # keep backward compatibility with name-k style
        if '-' in spec:
            name, k = spec.split('-', 1)
            try:
                return name, [float(k)]
            except ValueError:
                return name, []
        return spec, []

    def train_ft(self, batch_size, device):
        t_uniform = torch.rand((batch_size, 1), device=device)
        t_distort = self.apply_distortion(t_uniform, self.train_distortion)

        return t_distort, t_uniform

    def sample_kappa(self, t):
        """Map original t in [0,1] to kappa(t) using sampling distortion and return derivative d kappa / dt.

        This is used only during sampling; no ft is needed when using fixed steps.
        """
        return self.compute_kappa_and_grad(t, self.sample_distortion)

    def fit(self, difficulty, t_array, learning_rate=0.01, iterations=1000):
        """Fit a beta distribution to data using the method of moments."""
        alpha, beta = self.alpha, self.beta
        t_array = t_array + 1e-6  # Avoid division by zero

        for _ in range(iterations):
            y_pred = beta_pdf(t_array, alpha, beta)

            # Numerical approximation of the gradients
            epsilon = 1e-5
            grad_alpha = (
                objective_function(alpha + epsilon, beta, difficulty, t_array)
                - objective_function(alpha - epsilon, beta, difficulty, t_array)
            ) / (2 * epsilon)
            grad_beta = (
                objective_function(alpha, beta + epsilon, difficulty, t_array)
                - objective_function(alpha, beta - epsilon, difficulty, t_array)
            ) / (2 * epsilon)

            # # Add regularization gradient components
            # grad_alpha += learning_rate * (1 - 1 / alpha**2)
            # grad_beta += learning_rate * (1 + 1 / beta**2)

            # Update parameters
            alpha -= learning_rate * grad_alpha
            beta -= learning_rate * grad_beta

            alpha = min(max(0.3, alpha), 3)
            beta = min(max(0.3, beta), 3)

        y_pred = beta_pdf(t_array, alpha, beta)
        self.approximate_f_inverse(alpha, beta)

        return y_pred, alpha, beta

    def approximate_f_inverse(self, alpha, beta):
        # Generate data points
        t_values = np.linspace(0, 1, 100000)
        f_values = sp_beta.cdf(t_values, alpha, beta)

        # Sort and remove duplicates
        sorted_indices = np.argsort(f_values)
        f_values_sorted = f_values[sorted_indices]
        t_values_sorted = t_values[sorted_indices]

        # Remove duplicates
        _, unique_indices = np.unique(f_values_sorted, return_index=True)
        f_values_unique = f_values_sorted[unique_indices]
        t_values_unique = t_values_sorted[unique_indices]

        # Create the interpolation function for the inverse
        f_inv = interp1d(
            f_values_unique,
            t_values_unique,
            bounds_error=False,
            fill_value="extrapolate",
        )

        self.f_inv = f_inv

    def apply_distortion(self, t, distortion_type, is_sample=False):
        """Apply the training-time distortion to get f(t).

        Only returns ft. Sampling should use sample_kappa to compute kappa and derivative.
        """
        assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"
        name, params = self._parse_name_and_params(distortion_type)
        if name in ("identity", "linear"):
            ft = t
        elif name in ("cos", "cosine"):
            ft = (1 - torch.cos(t * torch.pi)) / 2
        elif name == "revcos":
            ft = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        elif name == "polyinc":
            if len(params) == 1:
                p = params[0]
                ft = torch.pow(t, p)
            else:
                ft = t ** 2
        elif name == "polydec":
            # default quadratic decreasing
            if len(params) == 1:
                n = params[0]
                ft = 1 - (1 - t) ** n
            else:
                ft = 2 * t - t ** 2
        elif name in ("polynomial", "poly"):
            # κ(t) = -2 t^3 + 3 t^2 + a (t^3 - 2 t^2 + t) + b (t^3 - t^2)
            # For training distortion, we simply return κ(t) as f(t)
            a = params[0] if len(params) >= 1 else 0.0
            b = params[1] if len(params) >= 2 else 0.0
            t2 = t * t
            t3 = t2 * t
            ft = -2 * t3 + 3 * t2 + a * (t3 - 2 * t2 + t) + b * (t3 - t2)
        elif name == "lognorm":
            # Sample from LogNormal(mu, sigma) truncated to [0, 1]. If out of range, resample.
            mu = params[0] if len(params) >= 1 else 0.0
            sigma = params[1] if len(params) >= 2 else 1.0
            sigma = max(sigma, 1e-6)

            ft = torch.empty_like(t)
            remaining_mask = torch.ones_like(t, dtype=torch.bool)
            max_resamples = 100
            attempts = 0

            # Rejection sampling loop
            while torch.any(remaining_mask) and attempts < max_resamples:
                num_remaining = int(remaining_mask.sum().item())
                z = torch.randn(num_remaining, device=t.device, dtype=t.dtype)
                samples = torch.exp(mu + sigma * z)
                valid = (samples >= 0.0) & (samples <= 1.0)

                flat_ft = ft.view(-1)
                flat_mask = remaining_mask.view(-1)
                indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)

                if valid.any():
                    flat_ft[indices[valid]] = samples[valid]

                # Update mask for positions still needing valid samples
                new_flat_mask = torch.zeros_like(flat_mask)
                if (~valid).any():
                    new_flat_mask[indices[~valid]] = True
                remaining_mask = new_flat_mask.view_as(remaining_mask)
                attempts += 1

            # Fallback for any remaining positions after many attempts: uniform in [0,1]
            if torch.any(remaining_mask):
                fallback = torch.rand(int(remaining_mask.sum().item()), device=t.device, dtype=t.dtype)
                flat_ft = ft.view(-1)
                flat_mask = remaining_mask.view(-1)
                indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
                flat_ft[indices] = fallback

            ft = ft.clamp(0.0, 1.0)
        elif name == "beta":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        elif name == "logitnormal":
            raise ValueError(f"Unsupported for now: {distortion_type}")
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        return ft

    def compute_kappa_and_grad(self, t, distortion_type):
        """Compute kappa(t) and d kappa / dt from original t using common Flow Matching schedules.

        Supported types (sampling): identity/linear, kappa (2t-t^2), cos/cosine, revcos,
        polydec (or polydec-N), polyinc (or polyinc-N).
        """
        assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"
        name, params = self._parse_name_and_params(distortion_type)
        eps = 1e-8

        if name in ("identity", "linear"):
            kappa_t = t
            d_kappa_t = torch.ones_like(t)
        elif name == "kappa":
            kappa_t = 2 * t - t ** 2
            d_kappa_t = 2 - 2 * t
        elif name in ("cos", "cosine"):
            kappa_t = (1 - torch.cos(t * torch.pi)) / 2
            d_kappa_t = (torch.pi / 2) * torch.sin(t * torch.pi)
        elif name == "revcos":
            kappa_t = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
            d_kappa_t = 2 - (torch.pi / 2) * torch.sin(t * torch.pi)
        elif name == "polydec":
            if len(params) == 1:
                n = params[0]
                kappa_t = 1 - (1 - t) ** n
                d_kappa_t = n * (1 - t) ** (n - 1)
            else:
                kappa_t = 2 * t - t ** 2
                d_kappa_t = 2 - 2 * t
        elif name == "polyinc":
            if len(params) == 1:
                p = params[0]
                kappa_t = torch.pow(t, p)
                d_kappa_t = p * torch.pow(torch.clamp(t, min=eps), p - 1)
            else:
                kappa_t = t ** 2
                d_kappa_t = 2 * t
        elif name in ("polynomial", "poly"):
            # κ(t) = -2 t^3 + 3 t^2 + a (t^3 - 2 t^2 + t) + b (t^3 - t^2)
            # κ'(t) = -6 t^2 + 6 t + a (3 t^2 - 4 t + 1) + b (3 t^2 - 2 t)
            a = params[0] if len(params) >= 1 else 0.0
            b = params[1] if len(params) >= 2 else 0.0
            t2 = t * t
            t3 = t2 * t
            kappa_t = -2 * t3 + 3 * t2 + a * (t3 - 2 * t2 + t) + b * (t3 - t2)
            d_kappa_t = (
                -6 * t2 + 6 * t + a * (3 * t2 - 4 * t + 1) + b * (3 * t2 - 2 * t)
            )
        elif name.startswith('polydec') and '-' in distortion_type:
            # backward-compat for 'polydec-N'
            try:
                n = float(distortion_type.split('-')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid distortion_type format: {distortion_type}")
            kappa_t = 1 - (1 - t) ** n
            d_kappa_t = n * (1 - t) ** (n - 1)
        elif name.startswith('polyinc') and '-' in distortion_type:
            try:
                p = float(distortion_type.split('-')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid distortion_type format: {distortion_type}")
            kappa_t = torch.pow(t, p)
            d_kappa_t = p * torch.pow(torch.clamp(t, min=eps), p - 1)
        else:
            raise NotImplementedError(
                f"Unsupported sampling distortion for kappa: {distortion_type}"
            )

        return kappa_t, d_kappa_t
