import torch
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import spsolve


@torch.no_grad()
def get_final_states(
    architecture: list[torch.nn.Linear], x0_batch: torch.Tensor, xL_batch: torch.Tensor
):
    """
    Solve for optimal internal states x^{(1)},...,x^{(L-1)} in a linear bottom-up predictive coding model
    for a batch of inputs, using a sparse block tridiagonal system.

    Args:
        W (list of torch.Tensor): Weight matrices [W^{(1)}, ..., W^{(L)}],
                                  where W[l] has shape (d_{l+1}, d_l).
        x0_batch (torch.Tensor): Batch of input states x^{(0)} of shape (B, d0).
        xL_batch (torch.Tensor): Batch of top states x^{(L)} of shape (B, dL).

    Returns:
        List[torch.Tensor]: List of optimal states [x^{(1)}, ..., x^{(L-1)}],
                            each of shape (B, d_l).
    """
    W = [layer.weight for layer in architecture]

    L = len(W)
    batch_size = x0_batch.shape[0]
    dims = [W[l - 1].shape[0] for l in range(1, L)]

    # No need for sparse matrix in low dimensionality
    if len(dims) == 0:
        return []
    if len(dims) == 1:
        # directly solve (I + W^T W) x = b
        W0 = W[0]
        W1 = W[1]
        A = torch.eye(W0.shape[0]) + W1.T @ W1
        b = (W0 @ x0_batch.mT + W1.T @ xL_batch.mT).mT
        x1 = torch.linalg.solve(A, b.T).T
        return [x1]


    # Build sparse block matrix A (shared for all batch elements)
    blocks = []
    for l_idx, dim in enumerate(dims):
        l = l_idx + 1
        I = torch.eye(dim)
        W_up = W[l]
        diag_block = I + W_up.T @ W_up

        row_blocks = []
        for m_idx in range(len(dims)):
            if m_idx == l_idx:
                row_blocks.append(csc_matrix(diag_block.numpy()))
            elif m_idx == l_idx - 1:
                W_l = W[l_idx]
                row_blocks.append(csc_matrix(-W_l.numpy()))
            elif m_idx == l_idx + 1:
                row_blocks.append(csc_matrix(-W_up.T.numpy()))
            else:
                row_blocks.append(None)
        blocks.append(row_blocks)

    A = bmat(blocks).tocsc()

    # Build batch of right-hand sides b (shape (B, total_d))
    total_d = sum(dims)
    b_batch = torch.zeros(batch_size, total_d)

    offset = 0
    for l_idx, dim in enumerate(dims):
        l = l_idx + 1
        if l_idx == 0:
            b_block = (W[0] @ x0_batch.mT).mT  # (B, d1)
        elif l_idx == len(dims) - 1:
            b_block = (W[L - 1].T @ xL_batch.mT).mT  # (B, d_{L-1})
        else:
            b_block = torch.zeros(batch_size, dim)
        b_batch[:, offset : offset + dim] = b_block
        offset += dim

    # Solve for each batch element
    X_batch = torch.zeros_like(b_batch)
    for b in range(batch_size):
        X_np = spsolve(A, b_batch[b].numpy())
        X_batch[b] = torch.from_numpy(X_np)

    # Split into list of tensors [x^1, ..., x^{L-1}] each (B, d_l)
    x_states = []
    offset = 0
    for dim in dims:
        x_part = X_batch[:, offset : offset + dim]
        x_states.append(x_part)
        offset += dim

    return x_states
