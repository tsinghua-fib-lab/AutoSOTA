import torch
from contextlib import contextmanager
from functools import partial
from torch.cuda.amp import autocast

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


@contextmanager
def null_context():
    yield


def linear_attention_normalization(q, k, causal=False):
    if not causal:
        return torch.einsum('...nm,...m->...n', q, k.sum(dim=-2))
    else:
        return torch.einsum('...nm,...nm->...n', q, k.cumsum(dim=-2))


# efficient causal linear attention, created by EPFL
def causal_linear_attention(q, k, v, need_weights=False):
    from fast_transformers.causal_product import causal_dot_product
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)
    causal_dot_product_fn = amp.float_function(causal_dot_product) if is_half else causal_dot_product
    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))
        q_k_v = causal_dot_product_fn(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        if need_weights:
            attn = torch.einsum('...im,...jm', q, k)
            causal_mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool,
                                                device=k.device), diagonal=1)
            attn.masked_fill_(causal_mask, 0.0)
        else:
            attn = None
    return q_k_v, attn


# non-causal linear attention
def linear_attention(q, k, v, need_weights=False):
    k_v = torch.einsum('...nm,...nd->...md', k, v)
    q_k_v = torch.einsum('...nm,...md->...nd', q, k_v)
    attn = None if not need_weights else torch.einsum('...im,...jm->...ij', q, k)
    return q_k_v, attn
