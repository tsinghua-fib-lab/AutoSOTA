import torch
import torch.nn as nn
import math


class SinusoidalPositionalEmbedding1D(nn.Module):
    def __init__(self, dim: int, max_len: int = 3000):
        """
        Args:
            dim: embedding dimension (must be even ideally)
            max_len: maximum sequence length for which to precompute embeddings
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        # Create constant positional encodings
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )  # [dim/2]

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with the model (CPU/GPU) but isn’t trainable
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: Tensor of position indices, shape (batch_size, seq_len)
                            or (seq_len,)

        Returns:
            Positional embeddings, shape (batch_size, seq_len, embedding_dim)
            or (seq_len, embedding_dim)
        """
        seq_len = position_ids.max() + 1
        if seq_len > self.max_len:
            # Dynamically expand if needed
            self._extend_pe(seq_len)

        return self.pe[position_ids]

    def _extend_pe(self, new_max_len: int | torch.Tensor):
        """Extend the positional embedding table dynamically."""
        position = torch.arange(new_max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(new_max_len, self.dim, device=self.pe.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe
        self.max_len = new_max_len
