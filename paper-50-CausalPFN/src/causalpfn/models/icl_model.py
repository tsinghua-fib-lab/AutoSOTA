import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loading import load_pretrained_tabdpt_model
from .model import TabDPTLongContextModel
from .utils import pad_x


class InContextModel(nn.Module):

    def __init__(
        self,
        model: TabDPTLongContextModel,
        model_config: dict,  # the config containing the constructor arguments for the model
        sigma: float = 0.5,
        vmin: float = -10.0,
        vmax: float = 10.0,
    ):
        super().__init__()
        self.model: nn.Module = model
        self.model_config = model_config

        # self.prepare_input is to be called for each of the models to do any model-specific preprocessing
        self.prepare_input = lambda x, y: (pad_x(x, model.num_features), y)
        self.nbins = model_config["model"]["nbins"]
        model_config["model_type"] = "tabdpt"
        model_config["sigma"] = sigma
        self.sigma = sigma

        # NOTE: These variables are stored to avoid re-initializing them for each forward pass
        self.vmin = vmin
        self.vmax = vmax

        bin_edges = torch.linspace(self.vmin, self.vmax, self.nbins + 1)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + 0.5 * bin_width  # shape: (nbins,)

        self.register_buffer("bin_edges", bin_edges)  # (nbins+1,)
        self.register_buffer("bin_width", bin_width)  # () – 0-D tensor
        self.register_buffer("bin_centers", bin_centers)  # (nbins,)

    def _predict_mean(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.bin_centers, dim=-1)

    def _sample_from_logits(self, logits: torch.Tensor, n_samples: int) -> torch.Tensor:
        # Flatten trailing dim to (..., nbins)
        orig_shape = logits.shape[:-1]

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)  # shape: (..., nbins)

        # Reshape for sampling: (batch_size, nbins)
        logits_reshaped = probs.reshape(-1, self.nbins)

        # Sample indices
        sampled_indices = torch.multinomial(
            logits_reshaped, num_samples=n_samples, replacement=True
        )  # (batch_size, n_samples)

        # Convert indices to values using bin_centers
        samples = self.bin_centers[sampled_indices]  # (batch_size, n_samples)

        # Reshape back to (..., n_samples)
        samples = samples.view(*orig_shape, n_samples)

        return samples

    def _hl_gaussian_cross_entropy_loss(
        self,
        logits: torch.Tensor,
        y_target: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """
        Calculate the cross-entropy loss between the predicted distribution and the target tensor.
        Args:
        logits: Tensor of shape (..., nbins) containing the predicted distribution
        y_target: Tensor of shape (...) containing target distribution
        vmin: Minimum value of the distribution
        vmax: Maximum value of the distribution
        sigma: Standard deviation for the Gaussian smoothing

        Note: This function construct a soft target distribution (one per element in y_target)
        by integrating a Gaussian centered at y_i over each bin. (Histogram Loss Gaussian or HL-Gauss)
        """
        assert sigma > 0, "Sigma must be positive."
        assert (
            y_target.shape == logits.shape[:-1]
        ), "y_target must have the same shape as logits except for the last dimension."
        # We'll compute a distribution by integrating the Gaussian cdf in each bin
        # so for bin k, the probability is cdf(upper_edge) - cdf(lower_edge).
        #   cdf(x) = 0.5 * [1 + erf((x - mu) / (sqrt(2)*sigma))]

        # Expand y_target so we can broadcast:
        # y_target: (...) => (..., 1) so we can compare with each bin center
        y_target_expanded = y_target.unsqueeze(-1)  # => (..., 1)

        # We'll use the normal CDF in a piecewise manner:
        def normal_cdf(x, mean, std):
            # cdf = 0.5 * [1 + erf((x - mean)/(sqrt(2)*std))]
            return 0.5 * (1.0 + torch.erf((x - mean) / (math.sqrt(2) * std)))

        # lower and upper edges for each bin, shape (nbins,)
        lower_edges = self.bin_centers - 0.5 * self.bin_width
        upper_edges = self.bin_centers + 0.5 * self.bin_width

        # Now we want to do cdf(upper_edges) - cdf(lower_edges) for each data point
        # We'll broadcast them so that shape => (..., nbins)
        cdf_upper = normal_cdf(upper_edges, y_target_expanded, sigma)
        cdf_lower = normal_cdf(lower_edges, y_target_expanded, sigma)

        # Probability in each bin is difference of cdfs
        p = cdf_upper - cdf_lower  # shape: (..., nbins)

        # Because of numerical issues, it might not sum exactly to 1, so we can renormalize:
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)  # shape: (..., nbins)

        # Now we can compute the cross-entropy loss with the *soft* target:  CE = - sum_k [ p_k * log(q_k) ]
        log_probs = F.log_softmax(logits, dim=-1)  # shape: (..., nbins)
        ce_loss = -torch.sum(p * log_probs, dim=-1).mean(dim=-1)
        return ce_loss

    def _get_y0_y1_shift_scale(
        self,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
    ):
        # compute the mean and std of the treatment and control groups in the observational data in the context
        treated_mask = (t_context == 1).long()
        control_mask = (t_context == 0).long()
        treated_counts = treated_mask.sum(dim=1, keepdim=True)
        control_counts = control_mask.sum(dim=1, keepdim=True)

        y1_mean = (y_context * treated_mask).sum(dim=1, keepdim=True) / treated_counts
        y0_mean = (y_context * control_mask).sum(dim=1, keepdim=True) / control_counts
        y1_std = (
            torch.sqrt(((y_context - y1_mean) ** 2 * treated_mask).sum(dim=1, keepdim=True) / treated_counts) + 1e-20
        )
        y0_std = (
            torch.sqrt(((y_context - y0_mean) ** 2 * control_mask).sum(dim=1, keepdim=True) / control_counts) + 1e-20
        )

        # compute the shift and scale for the treatment and control groups
        # if no treatment or control samples are present, we set the shift and scale according to the observational data
        y0_scale = torch.where(control_counts > 0, y0_std, y1_std)
        y1_scale = torch.where(treated_counts > 0, y1_std, y0_std)
        y0_shift = torch.where(control_counts > 0, y0_mean, y1_mean)
        y1_shift = torch.where(treated_counts > 0, y1_mean, y0_mean)

        return y0_shift, y0_scale, y1_shift, y1_scale

    def cepo_losses(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        E_y0_query: torch.Tensor,
        E_y1_query: torch.Tensor,
        sigma: float | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        sigma = sigma or self.sigma
        y0_shift, y0_scale, y1_shift, y1_scale = self._get_y0_y1_shift_scale(t_context, y_context)

        # z-standardize the outcomes
        y_standardized = torch.where(
            t_context == 1, (y_context - y1_shift) / y1_scale, (y_context - y0_shift) / y0_scale
        )
        E_y1_standardized = (E_y1_query - y1_shift) / y1_scale
        E_y0_standardized = (E_y0_query - y0_shift) / y0_scale

        # get access to both the factual and counterfactual predictions by evaluating each of the query points at
        # both t=1 and t=0
        random_treatments = torch.randint(0, 2, E_y0_query.shape, device=E_y0_query.device)

        x_and_t_context = torch.cat(
            [
                t_context.unsqueeze(-1),
                X_context,
            ],
            dim=2,
        )  # shape: (batch_size,  context_len , num_features + 1)

        x_and_t_query = torch.cat(
            [
                random_treatments.unsqueeze(-1),
                X_query,
            ],
            dim=2,
        )  # shape: (batch_size,  query_len , num_features + 1)

        Ey_target = torch.where(
            random_treatments == 0, E_y0_standardized, E_y1_standardized
        )  # shape: (batch_size, query_len)

        x_and_t = torch.cat(
            [x_and_t_context, x_and_t_query], dim=1
        )  # shape: (batch_size, context_len + query_len, num_features + 1)
        x_src, y_src = self.prepare_input(x_and_t, y_standardized)

        logits = self.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(
            0, 1
        )  # shape: (batch_size, query_len, nbins)
        logits = logits[:, :, -self.model.nbins :]  # only keep the last nbins, which are the predictions

        logits /= temperature  # Apply temperature scaling
        return self._hl_gaussian_cross_entropy_loss(
            logits=logits,
            y_target=Ey_target,
            sigma=sigma,
        )

    def forward(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        E_y0_query: torch.Tensor,
        E_y1_query: torch.Tensor,
        sigma: float | None = None,
        temperature: float = 1.0,
    ):
        """
        The forward method will simply call the cepo_losses method and returns the loss for training.
        This is done to support multi-GPU training.
        """
        return self.cepo_losses(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=X_query,
            E_y0_query=E_y0_query,
            E_y1_query=E_y1_query,
            sigma=sigma,
            temperature=temperature,
        )

    def predict_cepo(
        self,
        X_context: torch.Tensor,
        t_context: torch.Tensor,
        y_context: torch.Tensor,
        X_query: torch.Tensor,
        t_query: torch.Tensor,
        temperature: torch.Tensor, # shape: (num_temperatures, ),
        n_samples: int | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        y0_shift, y0_scale, y1_shift, y1_scale = self._get_y0_y1_shift_scale(t_context, y_context)
        y_standardized = torch.where(
            t_context == 1, (y_context - y1_shift) / y1_scale, (y_context - y0_shift) / y0_scale
        )

        x_and_t_context = torch.cat(
            [
                t_context.unsqueeze(-1),
                X_context,
            ],
            dim=2,
        )  # shape: (batch_size,  context_len , num_features + 1)

        x_and_t_query = torch.cat(
            [
                t_query.unsqueeze(-1),
                X_query,
            ],
            dim=2,
        )  # shape: (batch_size,  query_len , num_features + 1)

        x_and_t = torch.cat(
            [x_and_t_context, x_and_t_query], dim=1
        )  # shape: (batch_size, context_len + query_len, num_features + 1)

        x_src, y_src = self.prepare_input(x_and_t, y_standardized)

        logits = self.model(x_src.transpose(0, 1), y_src.transpose(0, 1)).transpose(
            0, 1
        )  # shape: (batch_size, query_len, nbins)
        logits = logits[:, :, -self.model.nbins :]  # only keep the last nbins, which are the predictions

        logits = logits.unsqueeze(1)  # shape: (batch_size, 1, query_len, nbins)
        y1_scale, y1_shift = y1_scale.unsqueeze(1), y1_shift.unsqueeze(1)  # shape: (batch_size, 1, query_len)
        y0_scale, y0_shift = y0_scale.unsqueeze(1), y0_shift.unsqueeze(1)  # shape: (batch_size, 1, query_len)
        t_query = t_query.unsqueeze(1)  # shape: (batch_size, 1, query_len)

        temperature = temperature[None, :, None, None]  # shape: (1, num_temperatures, 1, 1)

        logits = logits / temperature  # Apply temperature scaling

        mean = self._predict_mean(logits)  # shape: (batch_size, num_temperatures, query_len)
        mean_shift_scaled = torch.where(t_query == 1, mean * y1_scale + y1_shift, mean * y0_scale + y0_shift)

        if n_samples is None:
            return mean_shift_scaled

        # shape: (batch_size, num_temperatures, query_len, n_samples)
        samples = self._sample_from_logits(logits, n_samples)
        samples_shift_scaled = torch.where(
            t_query.unsqueeze(-1) == 1, samples * y1_scale + y1_shift, samples * y0_scale + y0_shift
        )
        return mean_shift_scaled, samples_shift_scaled

    @classmethod
    def load(cls, model_state: dict, model_config: dict) -> "InContextModel":
        inner_model_state = {k.replace("model.", ""): v for k, v in model_state.items() if k.startswith("model.")}
        if model_config["model_type"] == "tabdpt":
            ckpt_loaded = {"cfg": model_config, "model": inner_model_state}
            base_model = load_pretrained_tabdpt_model(ckpt_path=None, ckpt=ckpt_loaded)
        else:
            raise ValueError(f"Unknown model. Supported model is 'tabdpt'.")

        sigma = model_config.get("sigma", 0.5)
        return InContextModel(model=base_model, model_config=model_config, sigma=sigma)
