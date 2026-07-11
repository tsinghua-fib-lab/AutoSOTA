import torch
import torch.nn as nn
from typing import Optional
from torch.nn.utils.parametrizations import spectral_norm


class Decoder(nn.Module):
    """
    Decoder for Grammar VAE with configurable modes: 'positional' (simpler) or 'autoregressive'.
    """

    def __init__(
        self,
        latent_rep_size: int,
        hidden_size: int,
        output_size: int,
        max_length: int,
        rnn_type: str = "gru",
        num_layers: int = 3,
        dropout: float = 0.0,
        mode: str = "positional",
    ):
        super().__init__()
        self.mode = mode
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(latent_rep_size, hidden_size, bias=False)

        self.elu = nn.ELU()

        # RNN layer (GRU or RNN)
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif rnn_type.lower() == "rnn":
            self.rnn = nn.RNN(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'gru' or 'rnn'.")

        
        self.time_distributed_dense = nn.Linear(hidden_size, output_size, bias=False)

        # Configure mode-specific components
        if self.mode == "autoregressive":
            self.embedding = nn.Embedding(output_size, hidden_size)
         
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        Args:
            z: shape (batch_size, num_samples, latent_dim)
        Returns:
            logits: shape (batch_size, num_samples, max_length, output_size)
        """

        batch_size, num_samples, latent_dim = z.shape

        # Reshape z to process all samples
        z_flat = z.view(-1, latent_dim)  # (batch_size * num_samples, latent_dim)

        if self.mode == "positional":
            # h = self.relu(self.linear_in(z_flat))
            h = self.elu(self.linear_in(z_flat))
            h = h.unsqueeze(1).repeat(1, self.max_length, 1)
            h_out, _ = self.rnn(h)
            # h_out = self.relu(h_out)
            h_out = self.elu(h_out)
            logits = self.time_distributed_dense(
                h_out
            )  # (batch_size * num_samples, max_length, output_size)

            # Reshape back to include sample dimension
            logits = logits.view(batch_size, num_samples, self.max_length, -1)

        return logits
