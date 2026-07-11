import torch

def rp_nystrom(
    keys: torch.Tensor,
    sqd_knorm: torch.Tensor,
    r: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implements the randomly pivoted Cholesky algorithm optimized for torch.compile.

    Args:   
        keys (Tensor): torch.Tensor of shape (..., n, E) where n is the number of keys
        sqd_knorm (Tensor): squared norms of keys, shape (..., n)
        r (int): rank of the Nystrom approximation

    Returns:
        coreset (LongTensor): indices of the chosen landmark points; shape (..., r)
        weights (Tensor): Nystrom weights of shape (..., r, n)
    """
    keys_dtype, device = keys.dtype, keys.device
    dtype = torch.float32 if keys_dtype in [torch.bfloat16, torch.float16] else keys_dtype

    keys = keys.to(dtype)
    sqd_knorm = sqd_knorm.to(dtype)
    hsqd_knorm = sqd_knorm / 2.

    n = keys.shape[-2]
    batch_shape = keys.shape[:-2]

    # Pre-allocate all tensors
    kernel_core = torch.zeros((*batch_shape, r, n), dtype=dtype, device=device)
    kernel_core_dim = kernel_core.shape[0]
    kernel_inv = torch.zeros((*batch_shape, r, r), dtype=dtype, device=device)
    res_diagonal = torch.ones((*batch_shape, n), dtype=dtype, device=device)

    coreset_list = [None] * r 

    uniform = torch.empty((*batch_shape, n), dtype=dtype, device=device)
    g = torch.full((*batch_shape, r), -1., dtype=dtype, device=device)

    # Main loop:
    for i in range(r):
        # Sample with Gumbel-max trick (more compile-friendly)
        uniform.uniform_()
        scores = torch.log(res_diagonal) + sqd_knorm - torch.log(-torch.log(uniform))
        ids = torch.argmax(scores, dim=-1, keepdim=True)
        
        # Update coreset
        coreset_list[i] = ids

        if i > 0:
            # Gather kernel values for previously selected points
            a = torch.gather(kernel_core[:, :i, :], -1, ids[..., None].expand(kernel_core_dim, i, 1)).squeeze(2)
            
            # Compute Cholesky factor of kernel inverse
            # bmm faster than einsum
            g[..., :i] = torch.bmm(kernel_inv[..., :i, :i], a.unsqueeze(-1)).squeeze(-1)
            g[..., :i+1] *= torch.rsqrt(res_diagonal.gather(-1, ids))
            
        # Update kernel inverse in-place
        kernel_inv[..., :i+1, :i+1] += g[..., :i+1].unsqueeze(-1) * g[...,:i+1].unsqueeze(-2)
        
        # Compute kernel row corresponding to selected point
        kernel_row = gsn_kernel(keys, ids, hsqd_knorm).clamp(max = 1.)
        kernel_core[..., i, :] = kernel_row.squeeze(-2)

        if i < r-1:
            # Update residual diagonal
            y = torch.einsum(
                "...si, ...s -> ...i", kernel_core[..., :i+1, :], g[..., :i+1])
            
            res_diagonal -= y.square()
            # Set diagonal entries for selected points to zero
            res_diagonal.scatter_(-1, ids, 0.0)
            # Enforce nonnegativity
            res_diagonal.clamp_(min=0.0)

    # Concatenate indices
    coreset = torch.cat(coreset_list, dim=-1)
    
    return coreset, kernel_inv.to(keys_dtype), kernel_core.to(keys_dtype)


def gsn_kernel(
        keys: torch.Tensor,
        ids: torch.LongTensor,
        halfsqdkeynorms: torch.Tensor,
    ) -> torch.Tensor:
        """Returns tensor of Gaussian kernel matrices
        kernel_mat
            = exp(keys[...,ids,:] @ keys[...,:,:].T 
                - halfsqdkeynorms[...,ids] - halfsqdkeynorms.T)

        Note: Assumes key has already been scaled appropriately by
        sqrt(softmax_temp)

        Args:
            key: tensor of shape [..., n, E]
            ids: tensor of shape [..., r]
            halfsqdkeynorms: tensor of shape [..., n]]]

        Returns tensor of shape [..., r, n]
        """
        E = keys.shape[-1]
        key_term = torch.einsum(
            '...re, ...ne -> ...rn', keys.gather(-2, ids.unsqueeze(-1).expand(*ids.shape, E)), keys)
        ###TODO: check if inplace exp_ is faster
        return torch.exp(key_term - halfsqdkeynorms.gather(-1, ids).unsqueeze(-1)
                        - halfsqdkeynorms.unsqueeze(-2))

