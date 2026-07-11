
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.nn.utils.parametrizations import spectral_norm


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        z_dim: int,
        max_length: int,
        conv_sizes: List[int],
        kernel_sizes: List[int],
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(conv_sizes) != len(kernel_sizes):
            raise ValueError("conv_sizes and kernel_sizes must have the same length")

        # Compute sequence length after each valid (no-pad) conv
        self.layer_lengths = []
        current_length = max_length
        for k in kernel_sizes:
            current_length = current_length - (k - 1)
            if current_length <= 0:
                raise ValueError(
                    f"Kernel sizes shrink length ≤ 0 (got {k}); "
                    "reduce kernel size(s) or increase max_length."
                )
            self.layer_lengths.append(current_length)

        # Conv stack: Conv1d (SN, bias=False) -> LayerNorm([C,L]) -> ELU -> optional Dropout
        self.convs = nn.ModuleList()
        for i, (c_out, k) in enumerate(zip(conv_sizes, kernel_sizes)):

            
            c_in = input_dim if i == 0 else conv_sizes[i - 1]
            layer_length = self.layer_lengths[i]

            conv = nn.Conv1d(c_in, c_out, k, bias=False)
            norm = nn.LayerNorm([c_out, layer_length]) if use_batch_norm else nn.Identity()
            act  = nn.ELU()
            drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

            self.convs.append(nn.Sequential(conv, norm, act, drop))

        # Flatten -> hidden (SN) -> ELU -> optional Dropout
        self.output_size = self.layer_lengths[-1] * conv_sizes[-1]
        self.linear = nn.Sequential(
            # spectral_norm(nn.Linear(self.output_size, hidden_dim, bias=False)),
            nn.Linear(self.output_size, hidden_dim, bias=False),
            nn.ELU(),
            nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity(),
        )

        # Latent heads (SN), no bias
        # self.mu     = spectral_norm(nn.Linear(hidden_dim, z_dim, bias=False))
        # self.logvar = spectral_norm(nn.Linear(hidden_dim, z_dim, bias=False))
        self.mu     = nn.Linear(hidden_dim, z_dim, bias=False)
        self.logvar = nn.Linear(hidden_dim, z_dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C_in, L]  -> conv stack -> flatten -> hidden -> (mu, logvar)
        returns: (mu, logvar) each [B, z_dim]
        """
        for block in self.convs:
            x = block(x)  # keeps shape [B, C_i, L_i] consistent with LayerNorm

        x = x.view(x.size(0), -1)  # flatten to [B, C_last * L_last]
        h = self.linear(x)
        return self.mu(h), self.logvar(h)
