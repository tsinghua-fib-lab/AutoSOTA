import torch

def weighted_attention(
        queries,
        core_keys,
        core_values,
        core_one,
        scale,
        min_val,
        max_val,
):
    
    # Compute stabilised reduced attention matrix
    QK = scale*torch.einsum("...te, ...re -> ...tr", queries, core_keys)
    QK -= QK.amax(-1, keepdim=True)
    QK = QK.exp()

    # Compute associated normalisation vector
    QK1 = torch.einsum("...tr, ...r -> ...t", QK, core_one).unsqueeze(-1)

    # Multiply by Nystrom-weighted values
    # TODO: Determine reasonable cut-off threshold
    eps = 1e-20
    out = torch.where(QK1 > eps, torch.einsum("...tr, ...rd -> ...td", QK, core_values) / QK1, 0.)

    # # Add in impact of value centers
    # out = out + vbar 

    # Exact attention output should always lie in the range of the original
    # values, so enforce this constraint
    out = out.clamp(min = min_val, max = max_val)
    return out