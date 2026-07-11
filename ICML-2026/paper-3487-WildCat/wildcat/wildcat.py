import torch
import torch.nn as nn
from math import sqrt

from wildcat.compress_kv import compress_kv
from wildcat.weighted_attention import weighted_attention

class WildCat(nn.Module):
    """Implementation of WildCat module."""

    def __init__(
        self,
        scale: float | None = None,
        r: int | None = None,
        num_bins: int = 1,
        subsample_ratio: float = 0.25,
        precompile: bool = True,
        **kwargs: dict,
    ):
        """Initialize the WildCat module.

        Args:
            scale (float): scale for dot-product attention.
              If `None`, scale is chosen as 1/sqrt(keys.shape[-1]) in forward.
            r (int): number of key-value pairs to select, a nonnegative integer
            num_bins (int): number of bins into which the sequence should be divided;
              compression is performed independently on each bin.
            subsample_ratio (float): proportion of key-value pairs to keep when no r is provided.
            compile (bool): whether to compile the forward pass with torch.compile for fast inference.
        """
        super().__init__()

        self.scale = scale
        self.r = r
        self.num_bins = num_bins
        self.subsample_ratio = subsample_ratio

        if precompile:
            self.forward = torch.compile(self.forward, mode="max-autotune")

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        
        """Forward pass of the WildCat module."""

        # Make input tensors have three dimensions (batch_size*num_heads, sequence_length, model_dimension)
        queries, keys, values, queries_shape = align_shapes(queries, keys, values)

        B, N, E = keys.shape
        D = values.shape[-1]

        # Number of chunks of sequence
        C = self.num_bins

        # Determine coreset size per bin
        # If no coreset size is provided, compress to self.subsample_ratio
        r = self.r or max(1, int(N* self.subsample_ratio)) # int 
        remainder = N % C
        if C > 1:
            r = r + (-r) % C  # pad to next multiple of C
            bin_r = r // C # bin_r == r == self.r when C == 1
        else:
            bin_r = r

        # Scale parameter of self-attention softmax
        scale = scale or self.scale or 1 / sqrt(E)

        # The attention output takes values in a convex polytope bounded by max_val and min_val
        max_val = values.amax(dim = -2, keepdim=True)
        min_val = values.amin(dim = -2, keepdim=True)

        # Recenter keys.
        kbar = keys.mean(dim = -2, keepdim=True)
        keys = keys - kbar

        if remainder > 0:
            keys, remainder_keys = keys.split([N - remainder, remainder], dim=1)
            values, remainder_values = values.split([N - remainder, remainder], dim=1)
        else:
            remainder_keys = remainder_values = None

        keys = keys.reshape(B * C, (N-remainder) // C, E)
        values = values.reshape(B * C, (N-remainder) // C, D)

        # Core compression routine
        q_scale = queries.square().sum(dim = -1).sqrt().amax(dim = -1)

        cmpd_keys, cmpd_values, w = compress_kv(keys, values, bin_r, scale, q_scale)
        cmpd_keys = cmpd_keys.reshape(B, r, E)
        cmpd_values = cmpd_values.reshape(B, r, D)
        w = w.reshape(B, r)

        # Remainder bin
        # Handle case when sequence can't be chunked into num_bins
        if remainder_keys is not None:
            cmpd_remainder_keys, cmpd_remainder_values, remainder_w = compress_kv(
                    remainder_keys, remainder_values, bin_r, scale, q_scale
            )
            cmpd_keys = torch.cat((cmpd_keys, cmpd_remainder_keys), dim = 1)
            cmpd_values = torch.cat((cmpd_values, cmpd_remainder_values), dim = 1)
            w = torch.cat((w, remainder_w), dim =1)

        # # Reconstruction of attention output.
        out = weighted_attention(queries = queries,
                                 core_keys = cmpd_keys,
                                 core_values = cmpd_values,
                                 core_one = w,
                                 scale = scale,
                                 min_val = min_val,
                                 max_val = max_val
                                 )
        
        out = out.view(*queries_shape[:-1], D)

        return out
    

def align_shapes(queries, keys, values):
    """
    Assertions and possible reshapes
    The compression algorithms in compressed attention assume input tensors with shape (b, n, d).

    This function checks the shapes of the input tensors and reshapes them if necessary.

    Args:
        queries: (B, (H), T, E) The tensor containing the queries
        keys: (B, (H), S, E) The tensor containing the keys
        values: (B, (H), S, D) The tensor containing the values

    Returns:
        queries: (B', T, E) The reshaped queries tensor
        keys: (B', S, E) The reshaped keys tensor
        values: (B', S, D) The reshaped values tensor
        queries_shape: tuple containing the original shape of queries
    """

    queries_shape = queries.shape

    if len(keys.shape) > 3:      
        # Reshape inputs to (n_batch*n_heads, n_seq, dim)
        queries = queries.reshape(-1, queries.shape[-2], queries.shape[-1])
        keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])
        values = values.reshape(-1, values.shape[-2], values.shape[-1])

    return queries, keys, values, queries_shape



