# Taken from https://github.com/acerbilab/amortized-conditioning-engine/blob/main/src/model/target_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.distributions.utils import broadcast_all
from typing import Optional, Dict, Any, List, Tuple
from .utils import AttrDict, initialize_head


class GaussianHead(nn.Module):
    def __init__(
        self,
        dim_y: int,
        d_model: int,
        dim_feedforward: int,
        name: Optional[str] = None,
        bound_std: bool = True,
        std_min: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bound_std = bound_std
        self.std_min = std_min
        self.name = name
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y * 2),
        )

    def forward(
        self,
        batch: Any,
        z_target: torch.Tensor,
        reduce_ll: bool = True,
        predict: bool = False,
        num_samples: int = 0,
    ) -> AttrDict:
        out = self.predictor(z_target)
        mean, raw_std = torch.chunk(out, 2, dim=-1)
        std = (
            self.std_min +  torch.exp(raw_std) if self.bound_std else torch.exp(raw_std) # F.softplus(raw_std) if self.bound_std else torch.exp(raw_std)
        )

        pred_tar = Normal(mean, std)
        outs = AttrDict()
        if predict:
            outs.mean = mean
            outs.scale = std
            outs.samples = pred_tar.sample((num_samples,)).movedim(0, 2)
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
            outs.losses = -outs.tar_ll
        else:
            outs.tar_ll = (
                pred_tar.log_prob(batch.yt).sum(-1).mean()
                if reduce_ll
                else pred_tar.log_prob(batch.yt).sum(-1)
            )
            outs.loss = -outs.tar_ll
        return outs


class MixtureGaussian(nn.Module):
    def __init__(
        self,
        dim_y: int,
        d_model: int,
        dim_feedforward: int,
        num_components: int,
        name: Optional[str] = None,
        trange: Optional[Tuple[float, float]] = None,
        single_head: bool = False,
        std_min: float = 1e-3,
        loss_latent_weight: float = 1.0,
        loss_data_weight: float = 1.0,
        discrete_index: Optional[List[int]] = None,
        bias_init: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dim_y = dim_y
        self.single_head = single_head
        self.bias_init = bias_init
        self.discrete_index = discrete_index
        if discrete_index:
            self.class_head = nn.Sequential(
                nn.Linear(self.d_model, self.dim_feedforward),
                nn.ReLU(),
                nn.Linear(self.dim_feedforward, len(discrete_index) * self.dim_y),
            )

        self.heads = initialize_head(
            d_model, dim_feedforward, dim_y, single_head, num_components
        )
        self.num_components = num_components

        if trange is None:
            max_range = 1.0
            min_range = -1.0
        else:
            min_range, max_range = tuple(trange)
        if bias_init:
            self.mean_global_bias = nn.Parameter(
                torch.linspace(min_range, max_range, num_components)
            )
            delta = (
                0.5 * (max_range - min_range) / (num_components - 1)
            )  # 1/2 *(max(range) - min(range)) / (num_components - 1)

            self.std_global_bias = nn.Parameter(
                torch.ones_like(self.mean_global_bias)
                * inverse_softplus(torch.tensor(delta))
            )
            self.weights_global_bias = nn.Parameter(torch.zeros(num_components))

        self.std_min = std_min
        self.loss_data_weight = loss_data_weight
        self.loss_latent_weight = loss_latent_weight

    def forward(
        self,
        batch: AttrDict,
        z_target: torch.Tensor,
        reduce_ll: bool = True,
        predict: bool = False,
        num_samples: int = 1000,
    ) -> AttrDict:
        # Iterate over each head to get their outputs
        if self.single_head:
            output = self.heads(z_target)
            if self.num_components == 1:
                raw_mean, raw_std = torch.chunk(output, 2, dim=-1)
                raw_weights = torch.ones_like(raw_mean)
            else:
                raw_mean, raw_std, raw_weights = torch.chunk(output, 3, dim=-1)
        else:
            outputs = [
                head(z_target) for head in self.heads
            ]  # list of [B, T, dim_y * 3] * components
            raw_mean, raw_std, raw_weights = self._map_raw_output(outputs)

        # Adding bias terms
        mean, std, weights = self.add_bias(raw_mean, raw_std, raw_weights)
        tar_ll = self.compute_ll(batch.yt, mean, std, weights)  # log-likelihoods [B, T]
        log_probs = tar_ll  # save it for calculating log probs in SBI

        if self.discrete_index:
            output_d = self.class_head(z_target)
            continuous_mask, discrete_mask = self.build_mask(
                batch
            )  # Mask [batch, num_target]
            # Compute discrete loss inputs
            output_permuted, discrete_labels, filtered_output_d = self._discrete(
                batch, output_d, discrete_mask
            )

            cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
            discrete_loss = cross_entropy_loss(output_permuted, discrete_labels)

            tar_ll = torch.where(
                discrete_mask, -discrete_loss, tar_ll
            )  # update log-likelihoods

        # Weighted loss
        latent_loss_mask = batch.xt[:, :, 0] > 1  # latents marker > 1
        data_loss_mask = ~latent_loss_mask
        loss = torch.zeros_like(tar_ll)
        loss[data_loss_mask] = -tar_ll[data_loss_mask] * self.loss_data_weight
        loss[latent_loss_mask] = -tar_ll[latent_loss_mask] * self.loss_latent_weight

        if reduce_ll:
            loss = loss.mean()
            tar_ll = tar_ll.mean()

        outs = AttrDict()

        if predict:
            samples, median, q1, q3 = self.predict_out(
                mean, std, weights, batch, num_samples
            )
            outs.samples = samples
            # [BxT, num_samples] -> [B, T, 1]
            outs.mean = samples.mean(dim=2, keepdim=True)  # sample mean
            outs.median = median
            outs.q1 = q1
            outs.q3 = q3

            # also outputs params
            outs.mixture_means = mean
            outs.mixture_stds = std
            outs.mixture_weights = weights

            if self.discrete_index:
                outs.logits = filtered_output_d
                outs.class_pred = torch.argmax(
                    filtered_output_d
                    * (discrete_mask.unsqueeze(-1).expand(filtered_output_d.shape)),
                    dim=-1,
                )
                outs.discrete_mask = discrete_mask
                outs.continuous_mask = continuous_mask

            outs.losses = -log_probs

        outs.loss = loss  # weighted negative log-likelihood
        outs.tar_ll = tar_ll  # log-likelihood
        return outs

    def add_bias(
        self, raw_mean: torch.Tensor, raw_std: torch.Tensor, raw_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.bias_init:
            mean = raw_mean + self.mean_global_bias
            std = torch.exp(raw_std + self.std_global_bias) + self.std_min  # F.softplus(raw_std + self.std_global_bias) + self.std_min
            # std = torch.maximum(std, torch.tensor(0.1)) # needed for image
            weights = F.softmax(raw_weights + self.weights_global_bias, dim=-1)
        else:
            mean = raw_mean
            std = torch.exp(raw_std) + self.std_min  #  F.softplus(raw_std) + self.std_min
            weights = F.softmax(raw_weights, dim=-1)
        return mean, std, weights

    @staticmethod
    def compute_ll(
        value: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes log-likelihood loss for a Gaussian mixture model.

        Parameters:
        - value (Tensor): Observed values. Shape: [B, Target, 1]
        - means (Tensor): Gaussian means. Shape: [B, Target, num_components]
        - stds (Tensor): Gaussian standard deviations. Shape: [Target, num_components]
        - weights (Tensor): Gaussian mixing weights. Shape: [Target, num_components]

        Returns:
        - Tensor: Computed loss. Shape: [B, Target]
        """
        components = Normal(means, stds, validate_args=False)
        log_probs = components.log_prob(value)
        weighted_log_probs = log_probs + torch.log(weights)
        return torch.logsumexp(weighted_log_probs, dim=-1)

    def predict_out(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        weights: torch.Tensor,
        batch: Any,
        num_samples: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        samples_flattened = sample(
            mean.view(-1, self.num_components),
            std.view(-1, self.num_components),
            weights.view(-1, self.num_components),
            num_sample=num_samples,
        )  # [BxT, num_samples]

        samples = samples_flattened.view(
            batch.yt.shape[0], batch.yt.shape[1], num_samples
        )  # [BxT, num_samples] -> [B, T, num_samples]

        # Calculate the quantiles
        q1 = torch.quantile(samples, 0.25, dim=-1)
        median = torch.quantile(samples, 0.50, dim=-1)
        q3 = torch.quantile(samples, 0.75, dim=-1)

        return samples, median, q1, q3

    @staticmethod
    def _map_raw_output(
        outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concatenated = (
            torch.stack(outputs).movedim(0, -1).flatten(-2, -1)
        )  # [B, T, dim_y * 3 * components]
        raw_mean, raw_std, raw_weights = torch.chunk(
            concatenated, 3, dim=-1
        )  # 3 x [B, T, components]
        return raw_mean, raw_std, raw_weights

    @staticmethod
    def _discrete(
        batch: Any, output_d: torch.Tensor, discrete_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies a discrete mask to the model's output tensors and modifies their dimensions to prepare
        for cross-entropy loss computation.
        """
        filtered_output_d = output_d * discrete_mask.unsqueeze(-1).expand(
            output_d.shape
        )
        discrete_labels = (
            batch.yt * discrete_mask.unsqueeze(-1).expand(batch.yt.shape)
        ).long()

        output_permuted = filtered_output_d.permute(
            0, 2, 1
        )  # [B, num_classes, T] for CrossEntropyLoss

        return output_permuted, discrete_labels.squeeze(-1), filtered_output_d

    def build_mask(self, batch: AttrDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates binary masks for separating target data into discrete and continuous components.

        Parameters:
        - batch (attrdict): Data batch containing target values `yt` to be categorized.

        Returns [B, T]:
        - torch.Tensor: Mask for continuous data (binary tensor; `True` for continuous components).
        - torch.Tensor: Mask for discrete data (binary tensor; `True` for discrete components).

        Note:
        Assumes `batch.yt` contains values comparable to integers for discreteness detection.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # List of discrete values
        discrete_values = torch.tensor(self.discrete_index, device=device)

        # Create the mask
        mask_discrete = torch.isin(batch.yt, torch.tensor(discrete_values))
        continuous_mask = ~mask_discrete

        # continuous_mask = continuous_mask.expand(batch.xte.shape[0], batch.xte.shape[1], components)
        return continuous_mask.squeeze(-1), mask_discrete.squeeze(-1)


def sample(
    means: torch.Tensor,
    stds: torch.Tensor,
    weights: torch.Tensor,
    num_sample: int = 1000,
) -> torch.Tensor:
    # TODO: swap out with pytorch inbuilt torch.distributions.mixture_same_family.MixtureSameFamily
    # Sample component indices for each batch

    mixture_dists = Categorical(weights, validate_args=False)
    component_indices = mixture_dists.sample((num_sample,))  # Shape [num_samples, TxB]

    # Means and stds are of shape [BxT, num_components]
    # selects the means and stds for each sample based on mixture indices [num_samples, BxT, 1]
    selected_means = torch.gather(
        means.unsqueeze(0).expand(num_sample, -1, -1),
        2,
        component_indices.unsqueeze(-1),
    )
    selected_stds = torch.gather(
        stds.unsqueeze(0).expand(num_sample, -1, -1), 2, component_indices.unsqueeze(-1)
    )

    # Flatten to match the distribution requirements
    selected_means = selected_means.squeeze(-1)  # [num_samples, BxT]
    selected_stds = selected_stds.squeeze(-1)

    normal_dists = Normal(selected_means, selected_stds, validate_args=False)
    samples = normal_dists.sample()  # [num_samples, BxT]

    return samples.t()


def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(y)  # torch.log(torch.special.expm1(y))

