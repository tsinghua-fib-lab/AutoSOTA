'''
* @author: EmpyreanMoon
*
* @create: 2024-12-22 16:45
*
* @description:
'''
import torch
from einops import rearrange
from sympy.stats.rv import sampling_E

from probts import Forecaster
from probts.model.nn.prob.k2VAE.k2vae import k2VAE
import torch.nn as nn
import torch.nn.functional as F


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class k2VAEModel(Forecaster):
    def __init__(
            self,
            d_model,
            d_ff,
            e_layers,
            dropout,
            activation,
            n_heads,
            factor,
            patch_len,
            multistep,
            dynamic_dim,
            hidden_layers,
            hidden_dim,
            weight_beta=0.001,
            sample_schedule=5,
            init_kalman='identity',
            init_koopman='both',
            **kwargs
    ):
        """
        Initialize the model with parameters.
        """
        super().__init__(**kwargs)
        # Initialize model parameters here
        config = ConvertedParams(kwargs)
        config.d_model = d_model
        config.d_ff = d_ff
        config.hidden_layers = hidden_layers
        config.dropout = dropout
        config.activation = activation
        config.e_layers = e_layers
        config.n_heads = n_heads
        config.factor = factor
        config.patch_len = patch_len
        config.multistep = multistep
        config.dynamic_dim = dynamic_dim
        config.hidden_dim = hidden_dim
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        self.weight_beta = weight_beta

        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman

        self.model = k2VAE(config)

    def forward(self, input):
        """
        Forward pass for the model.

        Parameters:
        inputs [Tensor]: Input tensor for the model.

        Returns:
        Tensor: Output tensor.
        """
        # Perform the forward pass of the model
        rec, prior_dist, post_dist = self.model(input)
        return rec, prior_dist, post_dist

    def kld_loss(self, dist):
        # Extract the mean and covariance matrix from the distribution
        mu_q = dist.loc  # Shape: (B, P, hidden)
        cov_q = dist.covariance_matrix  # Shape: (B, P, hidden, hidden)
        # Compute the log determinant of the covariance matrix
        log_det_q = torch.linalg.slogdet(cov_q)[1]  # Shape: (B, P)

        # Compute the trace of the covariance matrix (sum of diagonal elements)
        trace_q = torch.einsum('btii->bt', cov_q)  # Shape: (B, P)

        # Compute the squared norm of the mean for each time step
        mu_term = torch.sum(mu_q ** 2, dim=-1)  # Shape: (B, P)

        # Get the latent space dimension (256 in this case)
        latent_dim = mu_q.size(-1)

        # Compute the KL divergence for each time step
        kld = 0.5 * (trace_q + mu_term - latent_dim - log_det_q)  # Shape: (B, P)

        # Take the mean over the time steps, resulting in one KL value per batch sample
        kld = kld.mean(dim=-1)  # Shape: (B,)

        # Take the mean over the batch to get the final average KL divergence
        kld = kld.mean()  # Scalar

        return kld

    def loss(self, batch_data, threshold=1e2):
        """
        Compute the loss for the given batch data.

        Parameters:
        batch_data [dict]: Dictionary containing input data and possibly target data.

        Returns:
        Tensor: Computed loss.
        """
        # Extract inputs and targets from batch_data
        self.model.train()
        input = batch_data.past_target_cdf[:, -self.context_length:, :]
        target = batch_data.future_target_cdf
        # Forward pass
        rec, prior_dist, post_dist = self.forward(input)
        rec_loss = F.mse_loss(rec, input) + F.mse_loss(post_dist.loc, target)
        post_loss = -post_dist.log_prob(target).mean()
        kld_loss = self.kld_loss(prior_dist)
        weight_alpha = 1 if post_loss < threshold else 0
        # stabilize training process
        loss = rec_loss + weight_alpha * post_loss + self.weight_beta * kld_loss
        return loss

    def sample_from_distribution(self, input, num_samples):
        samples = self.model.sample(input, num_samples)
        return rearrange(samples, 'n b l c -> b n l c')

    def forecast(self, batch_data, num_samples=None):
        """
        Generate forecasts for the given batch data.

        Parameters:
        batch_data [dict]: Dictionary containing input data.
        num_samples [int, optional]: Number of samples per distribution during evaluation. Defaults to None.

        Returns:
        Tensor: Forecasted outputs.
        """
        # Perform the forward pass to get the outputs
        self.model.eval()
        input = batch_data.past_target_cdf[:, -self.context_length:, :]
        with torch.no_grad():
            if num_samples is not None:
                # If num_samples is specified, use it to sample from the distribution
                outputs = self.sample_from_distribution(input, num_samples)
            else:
                outputs = None
        return outputs  # [batch_size, num_samples, prediction_length, var_num]
