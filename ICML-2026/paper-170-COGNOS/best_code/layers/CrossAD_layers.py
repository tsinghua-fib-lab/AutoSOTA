import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, x_attn_weights, cross_attn_weights = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, x_attn_weights, cross_attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1,
                 activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_, x_attn_weights = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
        )
        x = x + self.dropout(x_)
        x = self.norm1(x)

        x_, cross_attn_weights = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
        )
        x = x + self.dropout(x_)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), x_attn_weights, cross_attn_weights


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        x_, attn_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask,
        )
        x = x + self.dropout(x_)
        x = self.norm1(x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn_weights


class ContextNet(nn.Module):
    def __init__(self, router, querys, extractor):
        super().__init__()
        self.router = router
        self.querys = querys
        self.extractor = extractor

    def forward(self, x_enc, local_repr, mask=None):
        q_indices = self.router(x_enc)
        q = torch.einsum('bn,nqd->bqd', q_indices, self.querys)  # q: [bs x query_len x d_model]

        query_latent_distances, context = self.extractor(q, local_repr, mask)
        return query_latent_distances, context


class Router(nn.Module):
    def __init__(self, seq_len, n_vars, n_query, topk=5):
        super().__init__()
        self.k = topk
        self.fc = nn.Sequential(nn.Flatten(-2), nn.Linear(seq_len * n_vars, n_query))

    def forward(self, x):
        bs, t, c = x.shape
        # fft
        x_freq = torch.fft.rfft(x, dim=1, n=t)
        # topk
        _, indices = torch.topk(x_freq.abs(), self.k, dim=1)  # indices: [bs x k x c]
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing="ij")
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        mask = torch.zeros_like(x_freq, dtype=torch.bool)  # mask: [bs x f x c]
        mask[index_tuple] = True
        x_freq[~mask] = torch.tensor(0.0 + 0j, device=x_freq.device)
        # ifft
        x = torch.fft.irfft(x_freq, dim=1, n=t)
        # mlp
        logits = self.fc(x)  # logits: [bs x n_query]
        # gumbel softmax
        q_indices = F.gumbel_softmax(logits, tau=1, hard=True)  # q_indices: [bs x n_query]

        return q_indices


class Extractor(nn.Module):
    def __init__(self, layers,
                 context_size=64, query_len=5, d_model=128, decay=0.99, epsilon=1e-5
                 ):
        super().__init__()
        # context
        self.context_size = context_size
        self.query_len = query_len
        self.d_model = d_model
        self.register_buffer("context",
                             torch.randn(context_size, query_len, d_model))  # context: [N x query_len x d_model]
        self.register_buffer("ema_count", torch.ones(context_size))
        self.register_buffer("ema_dw", torch.zeros(context_size, query_len, d_model))
        self.decay = decay
        self.epsilon = epsilon
        # extractor
        self.extractor = nn.ModuleList(layers)

    def update_context(self, q):
        # q: [bs x query_len x d_model]

        _, q_len, d = q.shape
        q_flat = q.reshape(-1, q_len * d)  # [bs x query_len*d_model]
        g_flat = self.context.reshape(-1, q_len * d)  # [N x query_len*d_model]
        N, D = g_flat.shape

        distances = (
                torch.sum(q_flat ** 2, dim=1, keepdim=True) +
                torch.sum(g_flat ** 2, dim=1) -
                2 * torch.matmul(q_flat, g_flat.t())
        )  # [bs x N] soft
        # distances = torch.sum((q_flat.unsqueeze(1)-g_flat.unsqueeze(0))**2, dim=-1)
        indices = torch.argmin(distances.float(), dim=-1)  # [bs]
        encodings = F.one_hot(indices, N).float()  # [bs x N] hard
        q_context = torch.einsum("bn,nqd->bqd", [encodings, self.context])  # [bs x query_len x d_model]
        q_hat = torch.einsum("bn,bqd->nqd", [encodings, q])  # [N x query_len x d_model]

        # query_latent_distances
        query_latent_distances = torch.mean(F.mse_loss(q_context.detach(), q, reduction="none"), dim=(1, 2))  # [bs]

        if self.training:
            with torch.no_grad():
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)  # [N]
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + self.epsilon) / (n + D * self.epsilon) * n

                dw = torch.einsum("bn,bqd->nqd", [encodings, q])  # [N x query_len x d_model]
                self.ema_dw = self.decay * self.ema_dw + (1 - self.decay) * dw  # [N x query_len x d_model]
                self.context = self.ema_dw / self.ema_count.unsqueeze(-1).unsqueeze(-1)
        return query_latent_distances, q_hat

    def concat_context(self, context):
        return context.view(-1, self.d_model)  # [N*query_len x d_model]

    def forward(self, q, local_repr, mask=None):
        # q: [bs, query_len, d_model]
        # local_repr: [bs, ms_t, d_model]
        for layer in self.extractor:
            q = layer(q, local_repr, mask)  # [bs x query_len x d_model]

        query_latent_distances, q_hat = self.update_context(q)
        context = self.concat_context(
            q_hat + self.context.detach() - q_hat.detach())  # context: [N*query_len x d_model]
        return query_latent_distances, context  # N,query_len,d - bs,q_query_len,d_model


class ExtractorLayer(nn.Module):
    def __init__(self, cross_attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super().__init__()

        d_ff = d_ff or 4 * d_model
        # attention
        self.cross_attention = cross_attention
        # ffn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q, local_repr, mask=None):
        # q: [bs, query_len, d_model]
        # local_repr: [bs, ms_t, d_model]

        # cross_attention
        q = q + self.dropout(self.cross_attention(
            q, local_repr, local_repr,
            attn_mask=mask
        )[0])
        q = self.norm1(q)  # q: [bs x query_len x d_model]

        # ffn
        y = self.dropout(self.activation(self.conv1(q.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))  # y: [bs x query_len x d_model]

        return self.norm2(q + y)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None):
        bs, n_heads, q_len, d_k = q.size()
        scale = 1. / math.sqrt(d_k)
        attn_scores = torch.matmul(q, k) * scale  # attn_scores: [bs x n_heads x q_len x k_len]
        if attn_mask is not None:  # attn_mask: [q_len x k_len]
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights: [bs x n_heads x q_len x k_len]
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x q_len x d_v]

        return output.contiguous(), attn_weights


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_k=None, d_v=None, proj_dropout=0., qkv_bias=True):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.attn = attention

        self.proj = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K, V, attn_mask):
        bs, _, _ = Q.size()
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # q_s: [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)  # k_s: [bs x n_heads x d_k x k_len]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).permute(0, 2, 1, 3)  # v_s: [bs x n_heads x k_len x d_v]
        out, attn_weights = self.attn(q_s, k_s, v_s,
                                      attn_mask=attn_mask)  # out: [bs x n_heads x q_len x d_v], attn_weights: [bs x n_heads x q_len x k_len]
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, -1,
                                                        self.n_heads * self.d_v)  # out: [bs x q_len x n_heads * d_v]
        out = self.proj(out)  # out: [bs x q_len x d_model]
        attn_weights = attn_weights.mean(dim=1)  # attn_weights: [bs x q_len x k_len]
        return out, attn_weights