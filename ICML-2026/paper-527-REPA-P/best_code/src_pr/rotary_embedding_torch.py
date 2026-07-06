import torch


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding:
    def __init__(self, dim):
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq

    def _get_cos_sin(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('n,d->nd', positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    def rotate_queries_or_keys(self, x):
        rotary_dim = min(self.dim, x.shape[-1])
        if rotary_dim <= 0:
            return x
        rotary_dim = rotary_dim - (rotary_dim % 2)
        if rotary_dim == 0:
            return x

        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        cos, sin = self._get_cos_sin(x.shape[-2], x.device, x.dtype)
        cos = cos[..., :rotary_dim]
        sin = sin[..., :rotary_dim]
        x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
        return torch.cat((x_rot, x_pass), dim=-1)
