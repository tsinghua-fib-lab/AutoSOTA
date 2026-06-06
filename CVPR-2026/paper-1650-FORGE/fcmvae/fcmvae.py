import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetrize(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.transpose(-1, -2))


def normalize_adj_per_sample(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = symmetrize(a)
    a_min = a.amin(dim=(1, 2), keepdim=True)
    a_max = a.amax(dim=(1, 2), keepdim=True)
    a_norm = (a - a_min) / (a_max - a_min + eps)
    eye = torch.eye(a.shape[-1], device=a.device).unsqueeze(0).expand_as(a_norm)
    return a_norm * (1.0 - eye) + eye


def normalized_laplacian(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b, n, _ = a.shape
    deg = a.sum(dim=-1) + eps
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[~torch.isfinite(d_inv_sqrt)] = 0.0
    d_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    eye = torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0).expand(b, n, n)
    lap = eye - d_inv_sqrt @ a @ d_inv_sqrt
    return symmetrize(lap)


def laplacian_positional_encoding(a: torch.Tensor, k_pe: int = 16) -> torch.Tensor:
    _, n, _ = a.shape
    lap = normalized_laplacian(a)
    _, evecs = torch.linalg.eigh(lap)
    k = min(k_pe, n - 1)
    p = evecs[:, :, 1 : 1 + k]
    col_sum = p.sum(dim=1, keepdim=True)
    sign = torch.sign(torch.where(col_sum.abs() < 1e-9, torch.ones_like(col_sum), col_sum))
    return p * sign


class MultiHeadSelfAttentionWithBias(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_pe: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gamma_adj = nn.Parameter(torch.zeros(n_heads))
        self.gamma_pe = nn.Parameter(torch.zeros(n_heads))
        self.pe_weight = nn.Parameter(torch.empty(n_heads, k_pe, self.d_head))
        nn.init.xavier_uniform_(self.pe_weight)

    def forward(self, x: torch.Tensor, a_bias: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h, dh = self.n_heads, self.d_head
        q = self.q_proj(x).view(b, n, h, dh).transpose(1, 2)
        k = self.k_proj(x).view(b, n, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
        scores = scores + self.gamma_adj.view(1, h, 1, 1) * a_bias.unsqueeze(1).expand(b, h, n, n)
        k_eff = p.shape[-1]
        pe_weight = self.pe_weight[:, :k_eff, :]
        p_proj = torch.einsum("bnk,hkd->bhnd", p, pe_weight)
        pe_bias = torch.matmul(p_proj, p_proj.transpose(-2, -1)) / math.sqrt(dh)
        scores = scores + self.gamma_pe.view(1, h, 1, 1) * pe_bias
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, h * dh)
        return self.o_proj(out)


class TransformerFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.net(x))


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_pe: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithBias(d_model, n_heads, k_pe, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TransformerFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, a_bias: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x), a_bias, p)
        return h + self.ffn(self.ln2(h))


class GraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_node: int = 116,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        k_pe: int = 16,
        d_ff: int = 512,
        dropout: float = 0.1,
        out_dim: int = 256,
    ):
        super().__init__()
        self.k_pe = k_pe
        self.input_proj = nn.Linear(n_node + k_pe + 1, d_model)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d_model, n_heads, k_pe=k_pe, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.readout = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim), nn.ELU())

    def forward(self, x_roi_rows: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        _, n, _ = x_roi_rows.shape
        a_norm = normalize_adj_per_sample(a)
        eye = torch.eye(n, device=a.device, dtype=a.dtype).unsqueeze(0)
        a_bias = torch.clamp(a_norm + eye, 0.0, 1.0)
        p = laplacian_positional_encoding(a_norm, k_pe=self.k_pe)
        if p.shape[-1] < self.k_pe:
            p = F.pad(p, (0, self.k_pe - p.shape[-1]))
        deg = a_norm.sum(dim=-1, keepdim=True)
        x = self.input_proj(torch.cat([x_roi_rows, p, deg], dim=-1))
        for layer in self.layers:
            x = layer(x, a_bias, p)
        return self.readout(x.mean(dim=1))


class VAEGraphGate(nn.Module):
    def __init__(
        self,
        n_node: int = 116,
        latent_dim: int = 116,
        k_rank: int = 16,
        enc_d_model: int = 256,
        enc_heads: int = 4,
        enc_layers: int = 2,
        enc_k_pe: int = 16,
        enc_d_ff: int = 512,
        enc_dropout: float = 0.1,
        enc_out_dim: int = 256,
        num_classes: int = 2,
        gate_eps: float = 1e-2,
        gate_scale_init: float = 8.0,
    ):
        super().__init__()
        self.n_node = n_node
        self.latent_dim = latent_dim
        self.k_rank = k_rank
        self.num_classes = num_classes
        self.encoder = GraphTransformerEncoder(
            n_node=n_node,
            d_model=enc_d_model,
            n_heads=enc_heads,
            n_layers=enc_layers,
            k_pe=enc_k_pe,
            d_ff=enc_d_ff,
            dropout=enc_dropout,
            out_dim=enc_out_dim,
        )
        self.fc21 = nn.Linear(enc_out_dim, latent_dim)
        self.fc22 = nn.Linear(enc_out_dim, latent_dim)
        self.fcS = nn.Linear(latent_dim, n_node * k_rank)
        self.Wk = nn.Parameter(0.05 * torch.randn(k_rank))
        self.edge_bias = nn.Parameter(torch.zeros(n_node, n_node))
        self.fcy1 = nn.Linear(latent_dim, 1)
        self.log_sigma_y = nn.Parameter(torch.tensor(math.log(0.25), dtype=torch.float32))
        self.gate_scale = nn.Parameter(torch.tensor(float(gate_scale_init), dtype=torch.float32))
        self.register_buffer("gate_logit_eps", torch.tensor(float(math.log(gate_eps / (1.0 - gate_eps))), dtype=torch.float32))
        self.register_buffer("label_values", torch.arange(num_classes, dtype=torch.float32))
        self.register_buffer("label_probs", torch.full((num_classes,), 1.0 / max(1, num_classes), dtype=torch.float32))
        self.register_buffer("cond_adj", torch.stack([torch.eye(n_node) for _ in range(num_classes)], dim=0))
        self.register_buffer("global_adj", torch.eye(n_node))
        self.register_buffer("fisher_mu", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("fisher_sigma", torch.tensor(1.0, dtype=torch.float32))

    def set_label_values(self, label_values: torch.Tensor):
        label_values = torch.as_tensor(label_values, dtype=torch.float32).view(-1)
        if label_values.numel() != self.num_classes:
            raise ValueError(f"num label values {label_values.numel()} != num_classes {self.num_classes}")
        self.label_values = label_values.detach().cpu()

    def labels_to_indices(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1).to(self.label_values.device, dtype=self.label_values.dtype)
        dist = torch.abs(labels.view(-1, 1) - self.label_values.view(1, -1))
        return dist.argmin(dim=1).long()

    @torch.no_grad()
    def fit_fisher_statistics(self, x_vec: torch.Tensor):
        fc = torch.as_tensor(x_vec, dtype=torch.float32).view(-1, self.n_node, self.n_node)
        fc = torch.clamp(fc, min=-1.0 + 1e-4, max=1.0 - 1e-4)
        z = torch.atanh(fc)
        off_diag = ~torch.eye(self.n_node, dtype=torch.bool, device=z.device)
        z_edges = z[:, off_diag]
        mu = z_edges.mean()
        sigma = z_edges.std(unbiased=False).clamp_min(1e-6)
        self.fisher_mu.copy_(mu.to(self.fisher_mu.device))
        self.fisher_sigma.copy_(sigma.to(self.fisher_sigma.device))
        return {"fisher_mu": float(mu.item()), "fisher_sigma": float(sigma.item())}

    def log_rate_to_fc(self, log_rate: torch.Tensor) -> torch.Tensor:
        z = log_rate * torch.clamp(self.fisher_sigma, min=1e-6) + self.fisher_mu
        fc = symmetrize(torch.tanh(z))
        eye = torch.eye(self.n_node, dtype=fc.dtype, device=fc.device).unsqueeze(0)
        return fc * (1.0 - eye)

    def encode(self, x_vec: torch.Tensor, a_prior: torch.Tensor):
        b = x_vec.shape[0]
        h = self.encoder(x_vec.view(b, self.n_node, self.n_node), a_prior)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def prior_params(self, labels: torch.Tensor):
        device = next(self.parameters()).device
        labels = labels.view(-1)
        idx = self.labels_to_indices(labels).to(device)
        mu_p = torch.zeros(labels.shape[0], self.latent_dim, device=device)
        return mu_p, torch.zeros_like(mu_p), idx

    def decode(self, z: torch.Tensor, a_prior: torch.Tensor):
        b = z.shape[0]
        s = F.normalize(self.fcS(z).view(b, self.n_node, self.k_rank), dim=-1)
        interaction = torch.einsum("bnk,k,bmk->bnm", s, self.Wk, s)
        log_rate = symmetrize(self.edge_bias.unsqueeze(0) + interaction)
        a_norm = normalize_adj_per_sample(torch.clamp(a_prior, min=0.0))
        gate = torch.sigmoid(self.gate_scale * a_norm + self.gate_logit_eps)
        log_rate = symmetrize(log_rate + torch.log(torch.clamp(gate, min=1e-8)))
        return torch.clamp(log_rate, min=-10.0, max=10.0), s

    def forward(self, x_vec: torch.Tensor, a_prior: torch.Tensor, labels: torch.Tensor):
        mu_q, logvar_q = self.encode(x_vec, a_prior)
        z = self.reparameterize(mu_q, logvar_q)
        log_rate, s = self.decode(z, a_prior)
        y_hat = self.fcy1(mu_q).view(-1, 1)
        mu_p, logvar_p, label_idx = self.prior_params(labels)
        return log_rate, mu_q, logvar_q, y_hat, s, z, mu_p, logvar_p, label_idx

    @torch.no_grad()
    def fit_conditional_statistics(self, y: torch.Tensor, a_prior: torch.Tensor):
        y = torch.as_tensor(y, dtype=torch.float32).view(-1)
        if self.label_values.numel() != self.num_classes:
            uniq = torch.unique(y)
            uniq, _ = torch.sort(uniq)
            if uniq.numel() == self.num_classes:
                self.set_label_values(uniq)
        idx = self.labels_to_indices(y)
        a_prior = symmetrize(torch.as_tensor(a_prior, dtype=torch.float32))
        a_prior = torch.clamp(a_prior, min=0.0)
        cond_adj, counts = [], []
        for c in range(self.num_classes):
            mask = idx == c
            count = int(mask.sum().item())
            counts.append(count)
            cond_adj.append(symmetrize(a_prior[mask].mean(dim=0)) if count > 0 else torch.eye(self.n_node, dtype=torch.float32))
        self.cond_adj = torch.stack(cond_adj, dim=0)
        self.global_adj = symmetrize(a_prior.mean(dim=0)) if a_prior.shape[0] > 0 else torch.eye(self.n_node)
        probs = torch.tensor(counts, dtype=torch.float32)
        self.label_probs = probs / probs.sum().clamp_min(1.0)
        return {"labels": self.label_values.detach().cpu(), "counts": torch.tensor(counts, dtype=torch.long)}

    def _device(self, device=None):
        return next(self.parameters()).device if device is None else torch.device(device)

    def _expand_prior(self, base_adj: torch.Tensor, num_samples: int, device: torch.device):
        base_adj = base_adj.to(device)
        if base_adj.dim() == 2:
            return base_adj.unsqueeze(0).expand(num_samples, -1, -1)
        return base_adj

    @torch.no_grad()
    def sample(self, num_samples: int, a_prior: torch.Tensor = None, temperature: float = 1.0, device=None, return_latent: bool = False):
        device = self._device(device)
        z = torch.randn(num_samples, self.latent_dim, device=device) * float(temperature)
        if a_prior is None:
            a_prior = self.global_adj.to(device).unsqueeze(0).expand(num_samples, -1, -1)
        else:
            a_prior = self._expand_prior(a_prior, num_samples, device)
        fc_mats = self.log_rate_to_fc(self.decode(z, a_prior)[0])
        if return_latent:
            return fc_mats, z
        return fc_mats

    @torch.no_grad()
    def conditional_sample(self, labels, a_prior: torch.Tensor = None, temperature: float = 1.0, device=None, return_latent: bool = False):
        device = self._device(device)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=self.label_values.dtype)
        labels = labels.to(device).view(-1)
        _, _, idx = self.prior_params(labels)
        z = self._sample_latent_given_label(labels, temperature=temperature)
        if a_prior is None:
            a_prior = self.cond_adj.to(device)[idx]
        else:
            a_prior = self._expand_prior(a_prior, labels.shape[0], device)
        fc_mats = self.log_rate_to_fc(self.decode(z, a_prior)[0])
        if return_latent:
            return fc_mats, z
        return fc_mats

    @torch.no_grad()
    def _sample_latent_given_label(self, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        device = next(self.parameters()).device
        labels = labels.to(device=device, dtype=torch.float32).view(-1)
        w = self.fcy1.weight.view(-1)
        b = self.fcy1.bias.view(()).to(device)
        sigma2 = torch.exp(2.0 * torch.clamp(self.log_sigma_y, min=math.log(1e-4), max=math.log(10.0)))
        eye = torch.eye(self.latent_dim, dtype=w.dtype, device=device)
        precision = eye + torch.outer(w, w) / sigma2
        cov = torch.linalg.inv(precision)
        mean_direction = cov @ (w / sigma2)
        mean = (labels - b).unsqueeze(1) * mean_direction.unsqueeze(0)
        l = torch.linalg.cholesky(symmetrize(cov) + 1e-6 * eye)
        eps = torch.randn(labels.shape[0], self.latent_dim, dtype=w.dtype, device=device)
        return mean + float(temperature) * (eps @ l.T)

    @torch.no_grad()
    def export_npz(
        self,
        out_path: str,
        num_samples: int,
        labels=None,
        temperature: float = 1.0,
        conditional: bool = True,
        adj_threshold: float = 0.4,
        seed: int = None,
    ):
        if seed is not None:
            cpu_state = torch.random.get_rng_state()
            torch.manual_seed(seed)
            cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            if labels is None:
                if conditional:
                    idx = torch.multinomial(self.label_probs.cpu(), num_samples=num_samples, replacement=True)
                    labels = self.label_values.cpu()[idx]
                else:
                    labels = torch.zeros(num_samples, dtype=torch.float32)
            elif not torch.is_tensor(labels):
                labels = torch.tensor(labels)
            labels = labels.view(-1)
            if labels.numel() != num_samples:
                raise ValueError(f"labels length {labels.numel()} != num_samples {num_samples}")
            if conditional:
                fc_mats, z = self.conditional_sample(labels=labels, temperature=temperature, return_latent=True)
            else:
                fc_mats, z = self.sample(num_samples=num_samples, temperature=temperature, return_latent=True)
            fc_mats = fc_mats.detach().cpu().numpy().astype(np.float32)
            z = z.detach().cpu().numpy().astype(np.float32)
            labels_np = labels.detach().cpu().numpy().astype(np.float32)
            adjs = (fc_mats >= float(adj_threshold)).astype(np.float32)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            np.savez(out_path, fc_mats=fc_mats, rois=fc_mats, labels=labels_np, adjs=adjs, z=z)
            uniq, counts = np.unique(labels_np, return_counts=True)
            return {"path": out_path, "num_samples": int(num_samples), "label_hist": {str(float(k)): int(v) for k, v in zip(uniq, counts)}}
        finally:
            if seed is not None:
                torch.random.set_rng_state(cpu_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
