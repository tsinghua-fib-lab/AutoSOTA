import math
import scipy
import torch
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, reduce
from scipy.fftpack import next_fast_len


def exists(x):
    """
    Check if the input is not None.

    Args:
        x: The input to check.

    Returns:
        bool: True if the input is not None, False otherwise.
    """
    return x is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return the default value.

    Args:
        val: The value to check.
        d: The default value or a callable that returns the default value.

    Returns:
        The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    """
    Return the input tensor unchanged.

    Args:
        t: The input tensor.
        *args: Additional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        The input tensor unchanged.
    """
    return t

def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at indices specified by tensor `t` and reshapes the result.
    Args:
        a (torch.Tensor): The input tensor from which values are extracted.
        t (torch.Tensor): The tensor containing indices to extract from `a`.
        x_shape (tuple): The shape of the tensor `x` which determines the final shape of the output.
    Returns:
        torch.Tensor: A tensor containing the extracted values, reshaped to match the shape of `x` except for the first dimension.
    """

    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cond_fn(x, t, classifier=None, y=None, classifier_scale=1.):
    """
    Compute the gradient of the classifier's log probabilities with respect to the input.

    Args:
        classifier (nn.Module): The classifier model used to compute logits.
        x (torch.Tensor): The input tensor for which gradients are computed.
        t (torch.Tensor): The time step tensor.
        y (torch.Tensor, optional): The target labels tensor. Must not be None.
        classifier_scale (float, optional): Scaling factor for the gradients. Default is 1.

    Returns:
        torch.Tensor: The gradient of the selected log probabilities with respect to the input tensor, scaled by classifier_scale.
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

# normalization functions

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding module.

    This module generates sinusoidal positional embeddings for input tensors.
    The embeddings are computed using sine and cosine functions with different frequencies.

    Attributes:
        dim (int): The dimension of the positional embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# learnable positional embeds

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding module.

    This module generates learnable positional embeddings for input tensors.
    The embeddings are learned during training and can adapt to the specific task.

    Attributes:
        d_model (int): The dimension of the positional embeddings.
        dropout (float): The dropout rate applied to the embeddings.
        max_len (int): The maximum length of the input sequences.
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    """Standard positional encoding using sine and cosine functions."""

    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and transpose
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (not a parameter but should be saved)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
    

class Transformer_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=1, padding=0),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd,  kernel_size=3, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x)
    

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x
    

class AdaInsNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x


class ResBlock(nn.Module):
    def __init__(self, ch: int, width: int = 32, dilation: int = 1, groups: int = 4):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(ch, width, 3, padding=pad, dilation=dilation)
        self.gn1   = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, width, 3, padding=pad, dilation=dilation)
        self.gn2   = nn.GroupNorm(groups, ch)

    def forward(self, x):
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.silu(x + h)


class RelDistGate(nn.Module):
    """
    G(t,i,j) = sum_k w_k(t) * phi_k(|i-j|/L); symmetric & length-agnostic.
    """
    def __init__(self, heads: int, K: int = 6, t_dim: int = 256, basis: str = "fourier"):
        super().__init__()
        self.heads, self.K, self.basis = heads, K, basis
        self.w_mlp = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(t_dim, heads * K))

    @staticmethod
    def _phi(r: torch.Tensor, K: int, kind: str):
        # r: (L,L) in [0,1]
        if kind == "poly":
            Phi = [torch.ones_like(r)]
            Phi += [r**k for k in range(1, K)]
        else:  # fourier
            # [1, cos(pi r), sin(pi r), cos(2pi r), sin(2pi r), ...]
            Phi = [torch.ones_like(r)]
            m = torch.arange(1, (K//2)+1, device=r.device, dtype=r.dtype)
            Phi += [torch.cos(m_i*torch.pi*r) for m_i in m]
            Phi += [torch.sin(m_i*torch.pi*r) for m_i in m]
            Phi = Phi[:K]
        return torch.stack(Phi, dim=0)  # (K,L,L)

    def forward(self, t_emb: torch.Tensor, L: int) -> torch.Tensor:
        # t_emb: (N, t_dim)
        N, device, dtype = t_emb.size(0), t_emb.device, t_emb.dtype
        idx = torch.arange(L, device=device, dtype=dtype)
        r = (idx[:, None] - idx[None, :]).abs() / max(L - 1, 1)  # (L,L)
        Phi = self._phi(r, self.K, self.basis)                    # (K,L,L)
        W = self.w_mlp(t_emb).view(N, self.heads, self.K)  # (N,H,K)
        G = torch.sigmoid(torch.einsum('nhk,klm->nhlm', W, Phi))                 # (N,H,L,L)

        """
        import matplotlib.pyplot as plt
        plt.imshow(Phi[0].cpu().detach().numpy())

        plt.colorbar()
        plt.show()

        plt.imshow(W[0].cpu().detach().numpy())

        plt.colorbar()
        plt.show()
        """
        return G



class LowRankGate(nn.Module):
    def __init__(self, t_emb_dim : int, heads: int, L: int, R: int = 4, P: int = 4):
        super().__init__()
        self.heads, self.R, self.P, self.L = heads, R, P, L
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, heads*R*P))
        # 1D basis ψ_p(x) = {1, x, x^2, cos(πx), sin(πx), ...} length-agnostic
        self.kind = "poly"

    def _psi(self, x, P):
        if self.kind == "poly":
            return torch.stack([x**p for p in range(P)], dim=0)  # (P,L)
        else:  # fourier
            m = torch.arange(1, P//2+1, device=x.device, dtype=x.dtype)
            Phi = [torch.ones_like(x)]
            Phi += [torch.cos(mk*torch.pi*x) for mk in m] + [torch.sin(mk*torch.pi*x) for mk in m]
            return torch.stack(Phi[:P], dim=0)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        N = t_emb.size(0)
        L = self.L
        x = torch.linspace(0, 1, L, device=t_emb.device, dtype=t_emb.dtype)
        Psi = self._psi(x, self.P)                          # (P,L)

        coef = self.t_mlp(t_emb).view(N, self.heads, self.R, self.P)  # (N,H,R,P)
        A = torch.einsum('nhrp,pl->nhrl', coef, Psi)        # (N,H,R,L)
        # G = sum_r a_r(i) a_r(j)
        G = torch.einsum('nhrl,nhrm->nhlm', A, A)           # (N,H,L,L)
        # normalize if needed
        return G


class AtlasGate(nn.Module):
    def __init__(self, t_emb_dim : int, heads: int, K: int):
        super().__init__()
        self.heads = heads
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, heads*K))
        # 1D basis ψ_p(x) = {1, x, x^2, cos(πx), sin(πx), ...} length-agnostic

    def forward(self, U: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        N, L, K = U.shape
        H = self.heads
        Phi = torch.einsum('nlk,nlm->nklm', U, U)  # (N,K,L,L), Phi_k = u_k u_k^T
        W = torch.sigmoid(self.t_mlp(t_emb)).view(N, H, K)  # weights by t
        G = torch.einsum('nhk,nklm->nhlm', W, Phi)  # (N,H,L,L)
        return G