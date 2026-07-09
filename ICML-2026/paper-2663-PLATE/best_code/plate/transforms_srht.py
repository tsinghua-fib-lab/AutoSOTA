"""
Fast Structured Randomized Hadamard Transform (SRHT) for PLATE.
"""

import torch


def _next_pow2(n: int) -> int:
    """Next power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def fwht_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT) in-place.
    
    Args:
        x: [B, N] contiguous tensor where N is a power of 2
        
    Returns:
        Normalized transform (divided by sqrt(N))
    """
    B, N = x.shape
    h = 1
    while h < N:
        # Butterfly pattern: split into pairs and compute (a+b, a-b)
        x = x.view(B, -1, 2, h)
        # Compute in-place without cloning to save memory
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        # Store results in temporary variables to avoid aliasing issues
        sum_ab = a + b
        diff_ab = a - b
        x[:, :, 0, :] = sum_ab
        x[:, :, 1, :] = diff_ab
        x = x.view(B, -1)
        h <<= 1
    
    # Normalize by 1/sqrt(N) to make it orthogonal
    x.mul_(N ** -0.5)
    return x


class SRHT:
    """
    Structured Randomized Hadamard Transform (SRHT).
    
    Represents an orthogonal transform H = R @ P @ H_N @ D @ P^T where:
    - P is a fixed permutation
    - D is a diagonal sign matrix (Rademacher)
    - H_N is the normalized Hadamard matrix (via FWHT)
    - R restricts to the first d coordinates
    
    Application cost: O(d log d), no dense matrices.
    """
    
    def __init__(self, d: int, seed: int, device="cuda", dtype=torch.float32):
        """
        Initialize SRHT transform.
        
        Args:
            d: Dimension
            seed: Random seed for deterministic permutation and signs
            device: Device to store state on
            dtype: Data type for sign vector
        """
        self.d = d
        self.N = _next_pow2(d)
        
        # Generate deterministic permutation and signs using local generator
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        
        # Permutation
        self.perm = torch.randperm(d, generator=g, device=device)
        self.invperm = torch.empty_like(self.perm)
        self.invperm[self.perm] = torch.arange(d, device=device)
        
        # Rademacher signs (+1 or -1)
        self.sign = torch.where(
            torch.rand(d, generator=g, device=device) < 0.5,
            torch.tensor(-1.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )
    
    def to(self, device=None, dtype=None):
        """Move SRHT state to device/dtype."""
        if device is not None:
            self.perm = self.perm.to(device)
            self.invperm = self.invperm.to(device)
            self.sign = self.sign.to(device)
        if dtype is not None:
            self.sign = self.sign.to(dtype)
        return self
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply SRHT: H @ X where X is [..., d] or [..., d, *].
        
        Args:
            X: Input tensor with last dimension (or second-to-last) = d
            
        Returns:
            Transformed tensor of same shape
        """
        # X: [..., d] or [B, d]
        # 1. Permute and apply signs
        X = X[..., self.perm] * self.sign
        
        # 2. Zero-pad to power of 2 if needed
        if self.N != self.d:
            pad_shape = list(X.shape)
            pad_shape[-1] = self.N - self.d
            pad = torch.zeros(pad_shape, device=X.device, dtype=X.dtype)
            X = torch.cat([X, pad], dim=-1)
        
        # 3. Apply FWHT (requires contiguous [B, N])
        orig_shape = X.shape
        X = X.reshape(-1, self.N)  # Flatten batch dims
        X = fwht_inplace(X.contiguous())
        X = X.reshape(orig_shape)
        
        # 4. Remove padding
        if self.N != self.d:
            X = X[..., :self.d]
        
        return X
    
    def adjoint(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply SRHT adjoint: H^T @ X where X is [..., d].
        
        Args:
            X: Input tensor with last dimension = d
            
        Returns:
            Transformed tensor of same shape
        """
        # 1. Zero-pad to power of 2 if needed
        if self.N != self.d:
            pad_shape = list(X.shape)
            pad_shape[-1] = self.N - self.d
            pad = torch.zeros(pad_shape, device=X.device, dtype=X.dtype)
            X = torch.cat([X, pad], dim=-1)
        
        # 2. Apply FWHT (Hadamard is self-adjoint)
        orig_shape = X.shape
        X = X.reshape(-1, self.N)
        X = fwht_inplace(X.contiguous())
        X = X.reshape(orig_shape)
        
        # 3. Remove padding
        if self.N != self.d:
            X = X[..., :self.d]
        
        # 4. Apply signs and inverse permutation
        X = X * self.sign
        X = X[..., self.invperm]
        
        return X
    
    def forward_sparse(self, Z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Sparse-aware SRHT: Compute H @ E_S @ Z^T efficiently.
        
        **OPTIMIZED**: Batch-compute H @ E_S by applying H to a k-column identity matrix.
        While this still computes the full columns, it's vectorized and much faster
        than a Python loop.
        
        Args:
            Z: Coefficients [batch, k]
            indices: Selected indices [k] corresponding to columns of E_S
            
        Returns:
            Y: Result [batch, d]
        """
        k = indices.size(0)
        device = Z.device
        dtype = Z.dtype
        
        # Create sparse identity matrix for selected columns: [d, k]
        E_S = torch.zeros(self.d, k, device=device, dtype=dtype)
        E_S[indices, torch.arange(k, device=device)] = 1.0
        
        # Apply H to all columns at once: H @ E_S = [d, k]
        # Transpose so we can use batch dimension: [k, d]
        H_E_S = self.forward(E_S.T)  # [k, d]
        
        # Contract with Z: Z @ H_E_S^T = [batch, k] @ [k, d] = [batch, d]
        Y = Z @ H_E_S
        
        return Y


@torch.no_grad()
def batched_hutch_scores(W: torch.Tensor, H: SRHT, q: int, seed: int, side: str = 'input') -> torch.Tensor:
    """
    Estimate diagonal of H^T @ Gram @ H using batched Hutch++.
    
    Where Gram = W^T @ W (input side) or W @ W^T (output side).
    All q probes processed in one pass with fp32 precision for stability.
    
    Args:
        W: Weight matrix [d_out, d_in]
        H: SRHT transform (d_in for input side, d_out for output side)
        q: Number of Rademacher probes
        seed: Seed for probe generation (local generator, no global state)
        side: 'input' for W^T @ W or 'output' for W @ W^T
        
    Returns:
        scores: [d] estimated diagonal energies
    """
    
    # Ensure contiguous matrices for efficient GEMMs
    Wc = W.contiguous()
    Wtc = W.t().contiguous()
    
    # Generate Rademacher probes in W's dtype using local generator
    g = torch.Generator(device=W.device)
    g.manual_seed(seed)
    Z = (torch.rand(H.d, q, generator=g, device=W.device) < 0.5).to(Wc.dtype).mul_(2).sub_(1)  # [d, q]
    
    # Batched Hutch++
    U = H.forward(Z.T).T  # [d, q]
    
    if side == 'input':
        # Gram = W^T @ W, so apply: W @ U, then W^T @ result
        V = Wc @ U  # [d_out, q]
        Y = Wtc @ V  # [d_in, q]
    else:  # output
        # Gram = W @ W^T, so apply: W^T @ U, then W @ result
        V = Wtc @ U  # [d_in, q]
        Y = Wc @ V  # [d_out, q]
    
    T = H.adjoint(Y.T).T  # [d, q]
    
    # Diagonal estimate: mean(Z ⊙ T)
    scores = (Z * T).mean(dim=1)  # [d]
    
    return scores.abs()  


@torch.no_grad()
def srht_columns(H: SRHT, idx: torch.Tensor) -> torch.Tensor:
    """
    Compute H @ E_idx, i.e., selected columns of H.
    
    Args:
        H: SRHT transform
        idx: [m] indices of columns to select
        
    Returns:
        H_cols: [d, m] selected columns
    """
    d = H.d
    m = idx.numel()
    
    # Memory optimization: Process columns in chunks to avoid large dense matrices
    # For m=16384, d=27648: dense E would be ~1.8GB. We chunk it.
    chunk_size = min(2048, m)  # Process 2048 columns at a time (~200MB per chunk)
    
    HC_chunks = []
    for i in range(0, m, chunk_size):
        end = min(i + chunk_size, m)
        chunk_m = end - i
        idx_chunk = idx[i:end]
        
        # Create small selection matrix for this chunk (use H's dtype)
        E_chunk = torch.zeros(chunk_m, d, device=H.perm.device, dtype=H.sign.dtype)
        E_chunk[torch.arange(chunk_m, device=E_chunk.device), idx_chunk] = 1.0
        
        # Apply H to this chunk
        HC_chunk = H.forward(E_chunk)  # [chunk_m, d]
        HC_chunks.append(HC_chunk)
        
        # Free memory immediately
        del E_chunk
    
    # Concatenate results
    HC = torch.cat(HC_chunks, dim=0)  # [m, d]
    
    return HC.t().contiguous()  # [d, m]


