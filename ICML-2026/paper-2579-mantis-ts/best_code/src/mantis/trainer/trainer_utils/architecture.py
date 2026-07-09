import torch

from torch import nn


class FineTuningNetwork(nn.Module):
    """
    A nn.Module wrapper to combine adapter, encoder and prediction head.

    Parameters
    ----------
    encoder: nn.Module
        The encoder (foundation model) that projects from ``(n_samples, n_channels, seq_len)`` to
        ``(n_samples, hidden_dim)``. If None, it is assumed that the input matrix represents already the embeddings, so
        the input is directly passed through ``head``.
    head: nn.Module, default=None
        Head is a part of the network that follows the foundation model and projects from the embedding space of shape
        ``(n_samples, hidden_dim)`` to the probability matrix of shape ``(n_samples, n_classes)``. The way this class
        is implemented, ``head`` cannot be None.
    adapter: nn.Module, default=None
        Adapter is a part of the network that precedes the foundation model and reduces the original data matrix
        of shape ``(n_samples, n_channels, seq_len)`` to ``(n_samples, new_n_channels, seq_len)``.
        By default, adapter is None, i.e., not used.
    """
    def __init__(self, encoder, head, adapter=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.adapter = adapter

    def forward(self, x):
        if self.encoder is None:
            return self.head(x)
        elif self.adapter is None:
            if x.shape[1] > 1:
                Warning(
                    "The data is multi-variate! Applying encoder to all channels independently")
                return self.head(torch.cat([self.encoder(x[:, [i], :]) for i in range(x.shape[1])], dim=-1))
            else:
                return self.head(self.encoder(x))
        else:
            adapter_output = self.adapter(x)
            return self.head(torch.cat([
                self.encoder(adapter_output[:, [i], :]) for i in range(adapter_output.shape[1])
            ], dim=-1))
