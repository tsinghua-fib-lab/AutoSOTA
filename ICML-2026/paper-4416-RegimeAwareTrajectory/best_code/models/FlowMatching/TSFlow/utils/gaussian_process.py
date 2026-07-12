import torch
from torch.distributions import MultivariateNormal as MV
from torchtyping import TensorType

import sys
from models.FlowMatching.TSFlow.utils.variables import Prior


def isotropic_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return gamma * torch.eye(t.size(0), device=t.device, dtype=t.dtype)


def radial_basis_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * (t - u) ** 2)


def ornstein_uhlenbeck(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * torch.abs(t - u))


def periodic_kernel(
    gamma: TensorType[float], t: TensorType[float, "length"], u: TensorType[float, "length"]
) -> TensorType[float, "length", "length"]:
    return torch.exp(-gamma * torch.sin(t - u) ** 2)


kernel_function_map = {
    Prior.ISO: isotropic_kernel,
    Prior.SE: radial_basis_kernel,
    Prior.OU: ornstein_uhlenbeck,
    Prior.PE: periodic_kernel,
}


def get_gp(kernel: Prior, gamma: TensorType[float], length: int, freq: int) -> TensorType[float, "length", "length"]:
    """
    Generate a Gaussian process covariance matrix using the specified kernel.

    Args:
        kernel (Prior): The type of kernel to use.
        gamma (TensorType[float]): Kernel parameter.
        length (int): The number of time points (size of the covariance matrix).
        freq (int): Frequency parameter used to determine the time scale.
    """
    t = torch.arange(length, device=gamma.device, dtype=gamma.dtype) * (torch.pi / freq)
    cov = kernel_function_map[kernel](gamma, t, t.unsqueeze(1))
    return cov


class Q0Dist(torch.nn.Module):
    def __init__(
        self,
        kernel: Prior,
        context_freqs: int,
        prediction_length: int,
        freq: int = 24,
        gamma: float = 1.0,
        iso: float = 1e-4,
    ):
        """
        Initialize the GP distribution.

        Args:
            kernel (Prior): The kernel type (e.g. Prior.SE, Prior.OU, Prior.PE, etc.).
            context_freqs (int): The number of frequency groups in the context.
            prediction_length (int): The prediction horizon.
            freq (int, optional): Frequency parameter for time computations. Defaults to 24.
            gamma (float, optional): The kernel hyperparameter. Defaults to 1.0.
            iso (float, optional): The isotropic noise level. Defaults to 1e-4.
        """
        super().__init__()
        context_length = context_freqs # modified this, due to ours dataset are not multiply with prediction length
        # context_length = context_freqs * prediction_length
        window_length = context_length + prediction_length

        # Convert gamma to a tensor for consistency.
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.iso = torch.tensor(iso, dtype=torch.float64)
        self.kernel = kernel
        self.freq = freq

        mean = torch.zeros(window_length)
        cov = get_gp(kernel, self.gamma, window_length, freq) + self.iso * torch.eye(window_length)

        # GP reg
        t_obs = torch.cat([torch.ones(context_length), torch.zeros(prediction_length)]) == 1
        t_new = ~t_obs

        # Extract the relevant submatrices from the covariance matrix.
        K = cov[t_obs][:, t_obs]
        K_star = cov[t_obs][:, t_new]
        K_star_star = cov[t_new][:, t_new]

        # Add a small amount of noise to the diagonal to ensure invertibility.
        K += 1e-4 * torch.eye(K.size(0))
        K_inv = torch.linalg.inv(K)
        K_inv_x_K_star = K_inv @ K_star
        cov_reg = K_star_star - K_star.T @ K_inv_x_K_star

        # Register buffers
        self.register_buffer("cov_inv", torch.linalg.inv(cov).float(), persistent=False)
        self.register_buffer("K_inv_x_K_star", K_inv_x_K_star.float(), persistent=False)
        self.register_buffer("cov_reg", cov_reg.float(), persistent=False)

        self.dist = MV(loc=mean.float(), covariance_matrix=cov.float())

    def _apply(self, fn):
        """
        Override _apply so that when the module is moved to a new device,
        we update our precomputed distribution (self.dist) accordingly.
        """
        super()._apply(fn)
        self.dist = MV(
            loc=self.dist.loc.to(self.cov_inv.device),
            scale_tril=self.dist.scale_tril.to(self.cov_inv.device),
        )
        return self

    def log_likelihood(self, x: TensorType[float, "batch_size", "length"]) -> TensorType[float]:
        """
        Compute the log likelihood of the data given the GP distribution.
        """
        x = x.to(self.cov_inv.device)
        return -self.dist.log_prob(x)

    def forward(self, num_samples: int) -> TensorType[float, "num_samples", "length"]:
        samples = self.dist.sample(torch.Size([num_samples]))
        return samples

    def gp_regression(self, x: TensorType[float], prediction_length: int) -> MV:
        """
        Perform GP regression on input data.

        Args:
            x (torch.Tensor): Input data tensor of shape [batch_size, window_length].
            prediction_length (int): Number of prediction points.

        Returns:
            MultivariateNormal: GP regression posterior.
        """
        batch_size = x.size(0)
        # Reshape x to [batch_size, num_groups, freq] for per-frequency computations.
        x_reshaped = x.reshape(batch_size, -1, self.freq)
        loc_freq = x_reshaped.mean(1, keepdims=False)

        # For periodic kernels, set location bias to zero.
        if self.kernel == Prior.PE:
            loc_freq = torch.zeros_like(loc_freq)

        # Remove the location bias.
        repeat_factor = self.K_inv_x_K_star.shape[0] // self.freq
        x_centered = x - loc_freq.repeat(1, repeat_factor)

        # Compute the regression mean.
        mean = x_centered @ self.K_inv_x_K_star
        mean = mean + loc_freq.repeat(1, prediction_length // self.freq)

        # Scale the regression covariance.
        cov = self.cov_reg
        return MV(mean, cov)


class Q0Linear(torch.nn.Module):
    def __init__(
        self,
        context_freqs,
        prediction_length,
        freq=24,
        **kwargs,
    ):
        super().__init__()
        context_length = context_freqs * prediction_length
        self.context_length = context_length
        self.freq = freq
        self.mean_linear = torch.nn.Sequential(
            torch.nn.Linear(context_length + 2 * freq, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
        )
        self.cov_linear = torch.nn.Sequential(
            torch.nn.Linear(context_length + 2 * freq, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.ReLU(),
            torch.nn.Linear(prediction_length, prediction_length),
            torch.nn.Softplus(),
        )

    def regression(self, x, prediction_length):
        print(x.shape)
        loc_freq = x.reshape(x.shape[0], -1, self.freq).mean(1, keepdims=False)
        x = x - loc_freq.repeat(1, self.context_length // self.freq)
        std_freq = (x.reshape(x.shape[0], -1, self.freq).std(1, keepdims=False)) + 1e-4
        x = x / std_freq.repeat(1, self.context_length // self.freq)
        mean = self.mean_linear(torch.cat([x, loc_freq, std_freq], 1)) + loc_freq.repeat(
            1, prediction_length // self.freq
        )
        cov = self.cov_linear(torch.cat([x, loc_freq, std_freq], 1))

        scalar = std_freq.repeat(1, prediction_length // self.freq)
        cov = cov[:, None] * torch.eye(prediction_length, device=x.device) * scalar.unsqueeze(-1)  # + 1e-3 * torch.eye(
        return mean, cov


if __name__ == "__main__":
    a = 3
    q0 = Q0Dist(kernel=Prior.SE, context_freqs=5, prediction_length=5, gamma=1.0)
    breakpoint()
    a = 5
