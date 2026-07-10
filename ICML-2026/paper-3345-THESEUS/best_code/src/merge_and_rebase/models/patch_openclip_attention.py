# src/merge_and_rebase/models/patch_openclip_attention.py
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

KernelFn = Callable[[torch.Tensor], torch.Tensor]


def _elu_plus_one_kernel(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def _relu_plus_eps_kernel(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    return F.relu(x) + eps


def _exp_kernel(x: torch.Tensor) -> torch.Tensor:
    # Clamp avoids inf values in mixed precision.
    return torch.exp(torch.clamp(x, min=-10.0, max=10.0))


def build_linear_kernel(name: str, *, eps: float) -> tuple[str, KernelFn]:
    key = name.strip().lower()
    if key in ("elu", "elu_plus_one"):
        return "elu_plus_one", _elu_plus_one_kernel
    if key in ("relu", "relu_plus_eps"):
        return "relu_plus_eps", lambda x: _relu_plus_eps_kernel(x, eps=eps)
    if key == "exp":
        return "exp", _exp_kernel
    raise ValueError(f"Unknown linear-attention kernel '{name}'. Choose from: elu_plus_one, relu_plus_eps, exp")


def normalize_linear_rule(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"kernel", "delta"}:
        return key
    raise ValueError(f"Unknown linear_rule '{name}'. Choose from: kernel, delta")


def _module_device_dtype(module: nn.Module) -> tuple[torch.device | None, torch.dtype | None]:
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    if tensor is None:
        return None, None
    return tensor.device, tensor.dtype


class DeltaMemory(nn.Module):
    """
    Learnable initial fast-weight memory W0 per head.
    Full mode: parameter W0[h] in R^{D x D}
    Low-rank mode: W0[h] = A[h] @ B[h], rank r.
    """

    def __init__(self, num_heads: int, head_dim: int, *, rank: int = 0) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = max(0, int(rank))
        if self.rank == 0:
            self.w0 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.head_dim))
            self.a = None
            self.b = None
        else:
            self.w0 = None
            self.a = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.rank))
            self.b = nn.Parameter(torch.zeros(self.num_heads, self.rank, self.head_dim))

    def matrix(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.rank == 0:
            assert self.w0 is not None
            return self.w0.to(device=device, dtype=dtype)
        assert self.a is not None and self.b is not None
        return torch.einsum(
            "hir,hrj->hij",
            self.a.to(device=device, dtype=dtype),
            self.b.to(device=device, dtype=dtype),
        )


class LoRAableMHA(nn.Module):
    """
    Replacement for torch.nn.MultiheadAttention that exposes q/k/v/out as nn.Linear.
    Supports both batch_first=True ([N,L,C]) and False ([L,N,C]).
    Self-attention only (q=k=v=x).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        batch_first: bool,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = bool(batch_first)
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = float(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

    @staticmethod
    def from_torch_mha(mha: nn.MultiheadAttention, proj_dropout: float = 0.0) -> LoRAableMHA:
        m = LoRAableMHA(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            batch_first=getattr(mha, "batch_first", False),
            attn_dropout=getattr(mha, "dropout", 0.0),
            proj_dropout=proj_dropout,
            bias=mha.in_proj_bias is not None,
        )

        W = mha.in_proj_weight.detach()
        C = mha.embed_dim
        m.q_proj.weight.data.copy_(W[0:C, :])
        m.k_proj.weight.data.copy_(W[C : 2 * C, :])
        m.v_proj.weight.data.copy_(W[2 * C : 3 * C, :])

        if mha.in_proj_bias is not None:
            b = mha.in_proj_bias.detach()
            m.q_proj.bias.data.copy_(b[0:C])
            m.k_proj.bias.data.copy_(b[C : 2 * C])
            m.v_proj.bias.data.copy_(b[2 * C : 3 * C])

        m.out_proj.weight.data.copy_(mha.out_proj.weight.detach())
        if mha.out_proj.bias is not None and m.out_proj.bias is not None:
            m.out_proj.bias.data.copy_(mha.out_proj.bias.detach())

        return m

    def _to_batch_first(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        # Normalize layout to batch-first [N, L, C]
        if self.batch_first:
            q_b = query
            k_b = key
            v_b = value
            n_batch, lq, c = q_b.shape
            _, lk, _ = k_b.shape
            return q_b, k_b, v_b, n_batch, lq, lk

        # [L, N, C] -> [N, L, C]
        lq, n_batch, c = query.shape
        lk, _, _ = key.shape
        return query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), n_batch, lq, lk

    def _from_batch_first(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x
        return x.transpose(0, 1)

    def _build_sdpa_mask(
        self,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        attn_mask_sdpa = None

        if key_padding_mask is not None:
            # key_padding_mask: [N, Lk], True = masked
            kpm = key_padding_mask[:, None, None, :].to(torch.bool)  # [N,1,1,Lk]
            attn_mask_sdpa = kpm

        if attn_mask is not None:
            am = attn_mask.to(torch.bool)
            # Accept [Lq, Lk] or broadcastable variants
            if am.dim() == 2:
                am = am[None, None, :, :]  # [1,1,Lq,Lk]
            elif am.dim() == 3:
                am = am[:, None, :, :]  # [N,1,Lq,Lk] if provided that way
            attn_mask_sdpa = am if attn_mask_sdpa is None else (attn_mask_sdpa | am)
        return attn_mask_sdpa

    def _project_qkv(self, q_b: torch.Tensor, k_b: torch.Tensor, v_b: torch.Tensor) -> tuple[torch.Tensor, ...]:
        q = self.q_proj(q_b)
        k = self.k_proj(k_b)
        v = self.v_proj(v_b)
        return q, k, v

    def _reshape_heads(self, x: torch.Tensor, n_batch: int, seq: int) -> torch.Tensor:
        return x.reshape(n_batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        q_b, k_b, v_b, n_batch, lq, lk = self._to_batch_first(query, key, value)
        q, k, v = self._project_qkv(q_b, k_b, v_b)
        q = self._reshape_heads(q, n_batch, lq)
        k = self._reshape_heads(k, n_batch, lk)
        v = self._reshape_heads(v, n_batch, lk)
        attn_mask_sdpa = self._build_sdpa_mask(key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_sdpa,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # [N, heads, Lq, D]

        # Back to [N, Lq, C]
        y = y.transpose(1, 2).reshape(n_batch, lq, self.embed_dim)
        y = self.out_proj(y)
        y = self.proj_dropout(y)

        y = self._from_batch_first(y)

        # OpenCLIP uses [0] only, weights not needed
        return y, None

    def to_torch_mha(self) -> nn.MultiheadAttention:
        """
        Convert this module back to a fused nn.MultiheadAttention.
        """
        fused = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            bias=self.q_proj.bias is not None,
            batch_first=self.batch_first,
        )

        ref_w = self.q_proj.weight
        fused = fused.to(device=ref_w.device, dtype=ref_w.dtype)

        with torch.no_grad():
            fused.in_proj_weight.copy_(
                torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
            )

            if fused.in_proj_bias is not None:
                q_bias = self.q_proj.bias if self.q_proj.bias is not None else torch.zeros_like(self.q_proj.weight[:, 0])
                k_bias = self.k_proj.bias if self.k_proj.bias is not None else torch.zeros_like(self.k_proj.weight[:, 0])
                v_bias = self.v_proj.bias if self.v_proj.bias is not None else torch.zeros_like(self.v_proj.weight[:, 0])
                fused.in_proj_bias.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))

            fused.out_proj.weight.copy_(self.out_proj.weight)
            if fused.out_proj.bias is not None and self.out_proj.bias is not None:
                fused.out_proj.bias.copy_(self.out_proj.bias)

        fused.train(self.training)
        return fused


class LoRAableLinearMHA(LoRAableMHA):
    """
    LoRA-compatible normalized linear attention:
      KV = sum_l phi(k_l) v_l^T
      S  = sum_l phi(k_l)
      y  = (phi(q)^T KV) / (phi(q)^T S + eps)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        batch_first: bool,
        kernel: str = "elu_plus_one",
        eps: float = 1e-6,
        ramp_steps: int = 0,
        linear_rule: str = "kernel",
        delta_eta: float = 1.0,
        delta_exclude_cls_from_store: bool = True,
        delta_cls_only_readout: bool = False,
        delta_learn_w0: bool = False,
        delta_w0_rank: int = 0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=batch_first,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            bias=bias,
        )
        self.kernel_name, self.kernel_fn = build_linear_kernel(kernel, eps=float(eps))
        self.eps = float(eps)
        self.scale = self.head_dim**-0.5
        self.ramp_steps = max(0, int(ramp_steps))
        self.blend_lambda = 1.0 if self.ramp_steps == 0 else 0.0
        self.linear_rule = normalize_linear_rule(linear_rule)
        self.delta_eta = float(delta_eta)
        self.delta_exclude_cls_from_store = bool(delta_exclude_cls_from_store)
        self.delta_cls_only_readout = bool(delta_cls_only_readout)
        self.delta_learn_w0 = bool(delta_learn_w0)
        self.delta_w0_rank = max(0, int(delta_w0_rank))
        self.delta_mem = (
            DeltaMemory(self.num_heads, self.head_dim, rank=self.delta_w0_rank) if self.delta_learn_w0 else None
        )

    @staticmethod
    def from_loraable_mha(
        mha: LoRAableMHA,
        *,
        kernel: str = "elu_plus_one",
        eps: float = 1e-6,
        ramp_steps: int = 0,
        linear_rule: str = "kernel",
        delta_eta: float = 1.0,
        delta_exclude_cls_from_store: bool = True,
        delta_cls_only_readout: bool = False,
        delta_learn_w0: bool = False,
        delta_w0_rank: int = 0,
    ) -> LoRAableLinearMHA:
        m = LoRAableLinearMHA(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            batch_first=mha.batch_first,
            kernel=kernel,
            eps=eps,
            ramp_steps=ramp_steps,
            linear_rule=linear_rule,
            delta_eta=delta_eta,
            delta_exclude_cls_from_store=delta_exclude_cls_from_store,
            delta_cls_only_readout=delta_cls_only_readout,
            delta_learn_w0=delta_learn_w0,
            delta_w0_rank=delta_w0_rank,
            attn_dropout=mha.attn_dropout,
            proj_dropout=0.0,
            bias=mha.q_proj.bias is not None,
        )
        m.proj_dropout = mha.proj_dropout
        miss, unexp = m.load_state_dict(mha.state_dict(), strict=False)
        if unexp:
            raise RuntimeError(f"Unexpected keys when initializing LoRAableLinearMHA: {unexp[:20]}")
        if miss:
            allowed_missing = {"delta_mem.w0", "delta_mem.a", "delta_mem.b"}
            if any(k not in allowed_missing for k in miss):
                raise RuntimeError(f"Missing keys when initializing LoRAableLinearMHA: {miss[:20]}")
        return m

    def set_kernel(self, *, kernel: str, eps: float) -> bool:
        kernel_name, kernel_fn = build_linear_kernel(kernel, eps=float(eps))
        changed = kernel_name != self.kernel_name or float(eps) != self.eps
        self.kernel_name = kernel_name
        self.kernel_fn = kernel_fn
        self.eps = float(eps)
        return changed

    def set_ramp(self, *, ramp_steps: int) -> bool:
        next_steps = max(0, int(ramp_steps))
        changed = next_steps != self.ramp_steps
        self.ramp_steps = next_steps
        if self.ramp_steps == 0:
            self.blend_lambda = 1.0
        return changed

    def set_linear_rule(
        self,
        *,
        linear_rule: str,
        delta_eta: float,
        delta_exclude_cls_from_store: bool,
        delta_cls_only_readout: bool,
        delta_learn_w0: bool,
        delta_w0_rank: int,
    ) -> bool:
        next_rule = normalize_linear_rule(linear_rule)
        next_eta = float(delta_eta)
        next_exclude = bool(delta_exclude_cls_from_store)
        next_cls_only = bool(delta_cls_only_readout)
        next_learn_w0 = bool(delta_learn_w0)
        next_w0_rank = max(0, int(delta_w0_rank))
        if next_learn_w0 != self.delta_learn_w0 or next_w0_rank != self.delta_w0_rank:
            raise ValueError(
                "Changing delta_learn_w0/delta_w0_rank on an existing LoRAableLinearMHA is not supported. "
                "Rebuild from base attention with patch_openclip_vit_attn on a fresh model."
            )
        changed = (
            next_rule != self.linear_rule
            or next_eta != self.delta_eta
            or next_exclude != self.delta_exclude_cls_from_store
            or next_cls_only != self.delta_cls_only_readout
        )
        self.linear_rule = next_rule
        self.delta_eta = next_eta
        self.delta_exclude_cls_from_store = next_exclude
        self.delta_cls_only_readout = next_cls_only
        return changed

    def set_ramp_step(self, step: int) -> float:
        if self.ramp_steps <= 0:
            self.blend_lambda = 1.0
            return self.blend_lambda
        frac = float(step) / float(self.ramp_steps)
        self.blend_lambda = float(min(1.0, max(0.0, frac)))
        return self.blend_lambda

    def to_softmax_mha(self) -> LoRAableMHA:
        m = LoRAableMHA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=self.batch_first,
            attn_dropout=self.attn_dropout,
            proj_dropout=0.0,
            bias=self.q_proj.bias is not None,
        )
        m.proj_dropout = self.proj_dropout
        m.load_state_dict(self.state_dict(), strict=True)
        return m

    def _linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q * self.scale
        compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        q_phi = self.kernel_fn(q.to(compute_dtype))
        k_phi = self.kernel_fn(k.to(compute_dtype))
        v_comp = v.to(compute_dtype)

        kv = torch.einsum("bhld,bhle->bhde", k_phi, v_comp)
        s = k_phi.sum(dim=2)
        numer = torch.einsum("bhld,bhde->bhle", q_phi, kv)
        denom = torch.einsum("bhld,bhd->bhl", q_phi, s)
        y = numer / (denom[..., None] + self.eps)
        return y.to(q.dtype)

    def _delta_rule_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        eta: float = 1.0,
        exclude_cls_from_store: bool = True,
        cls_only_readout: bool = False,
    ) -> torch.Tensor:
        # q,k,v: [B,H,L,D]
        compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        q_phi = self.kernel_fn(q.to(compute_dtype))  # [B,H,L,D]
        k_phi = self.kernel_fn(k.to(compute_dtype))  # [B,H,L,D]
        v_comp = v.to(compute_dtype)  # [B,H,L,D]

        B, H, L, D = q_phi.shape

        # Fast weights W per sample+head are reset each forward but can start from learnable W0.
        if self.delta_mem is None:
            W = torch.zeros((B, H, D, D), device=q.device, dtype=compute_dtype)
        else:
            W0 = self.delta_mem.matrix(device=q.device, dtype=compute_dtype)  # [H,D,D]
            W = W0.unsqueeze(0).expand(B, -1, -1, -1).clone()  # [B,H,D,D]

        # Choose which tokens write into memory (often: store patches, query with CLS)
        t0 = 1 if exclude_cls_from_store else 0

        # Online delta-rule updates
        for t in range(t0, L):
            x = k_phi[:, :, t, :]  # [B,H,D]
            y = v_comp[:, :, t, :]  # [B,H,D]

            y_hat = torch.einsum("bhde,bhe->bhd", W, x)  # W @ x
            err = y - y_hat  # [B,H,D]

            x_norm2 = (x * x).sum(dim=-1, keepdim=True) + self.eps  # [B,H,1]
            gain = (eta / x_norm2) * err  # [B,H,D]

            W = W + torch.einsum("bhd,bhe->bhde", gain, x)  # outer product

        if cls_only_readout:
            # Only output token 0 (CLS), zero elsewhere
            y0 = torch.einsum("bhde,bhe->bhd", W, q_phi[:, :, 0, :])  # [B,H,D]
            out = torch.zeros((B, H, L, D), device=q.device, dtype=compute_dtype)
            out[:, :, 0, :] = y0
            return out.to(q.dtype)

        out = torch.einsum("bhde,bhld->bhle", W, q_phi)  # [B,H,L,D]
        return out.to(q.dtype)

    def _run_linear_rule(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.linear_rule == "kernel":
            return self._linear_attention(q=q, k=k, v=v)
        return self._delta_rule_attention(
            q=q,
            k=k,
            v=v,
            eta=self.delta_eta,
            exclude_cls_from_store=self.delta_exclude_cls_from_store,
            cls_only_readout=self.delta_cls_only_readout,
        )

    def _softmax_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        # Keep compatibility for masked/casual cases via SDPA softmax path.
        if key_padding_mask is not None or attn_mask is not None or is_causal:
            return super().forward(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        q_b, k_b, v_b, n_batch, lq, lk = self._to_batch_first(query, key, value)
        q, k, v = self._project_qkv(q_b, k_b, v_b)
        q = self._reshape_heads(q, n_batch, lq)
        k = self._reshape_heads(k, n_batch, lk)
        v = self._reshape_heads(v, n_batch, lk)
        blend = float(self.blend_lambda)
        if blend <= 0.0:
            y = self._softmax_attention(q=q, k=k, v=v)
        elif blend >= 1.0:
            y = self._run_linear_rule(q=q, k=k, v=v)
        else:
            y_softmax = self._softmax_attention(q=q, k=k, v=v)
            y_linear = self._run_linear_rule(q=q, k=k, v=v)
            y = (1.0 - blend) * y_softmax + blend * y_linear
        y = y.transpose(1, 2).reshape(n_batch, lq, self.embed_dim)
        y = self.out_proj(y)
        y = self.proj_dropout(y)
        y = self._from_batch_first(y)
        return y, None


def split_openclip_vit_attn(
    visual: nn.Module,
    proj_dropout: float = 0.0,
    *,
    attn_impl: str = "softmax",
    kernel: str = "elu_plus_one",
    eps: float = 1e-6,
    ramp_steps: int = 0,
    linear_rule: str = "kernel",
    delta_eta: float = 1.0,
    delta_exclude_cls_from_store: bool = True,
    delta_cls_only_readout: bool = False,
    delta_learn_w0: bool = False,
    delta_w0_rank: int = 0,
) -> int:
    """
    Replaces nn.MultiheadAttention under visual.transformer.resblocks.*.attn
    with LoRA-compatible attention modules.
    """
    attn_impl_key = str(attn_impl).strip().lower()
    if attn_impl_key not in {"softmax", "linear"}:
        raise ValueError(f"Unknown attn_impl '{attn_impl}'. Choose from: softmax, linear")
    linear_rule_key = normalize_linear_rule(linear_rule)

    n_patched = 0
    transformer = getattr(visual, "transformer", None)
    if transformer is None:
        raise ValueError("visual has no .transformer; are you patching the right module?")
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        raise ValueError("visual.transformer has no .resblocks; unexpected OpenCLIP structure")

    for blk in resblocks:
        attn = getattr(blk, "attn", None)
        replacement: nn.Module | None = None

        if isinstance(attn, nn.MultiheadAttention):
            base = LoRAableMHA.from_torch_mha(attn, proj_dropout=proj_dropout)
            replacement = (
                base
                if attn_impl_key == "softmax"
                else LoRAableLinearMHA.from_loraable_mha(
                    base,
                    kernel=kernel,
                    eps=eps,
                    ramp_steps=ramp_steps,
                    linear_rule=linear_rule_key,
                    delta_eta=delta_eta,
                    delta_exclude_cls_from_store=delta_exclude_cls_from_store,
                    delta_cls_only_readout=delta_cls_only_readout,
                    delta_learn_w0=delta_learn_w0,
                    delta_w0_rank=delta_w0_rank,
                )
            )
        elif isinstance(attn, LoRAableMHA) and not isinstance(attn, LoRAableLinearMHA):
            if attn_impl_key == "linear":
                replacement = LoRAableLinearMHA.from_loraable_mha(
                    attn,
                    kernel=kernel,
                    eps=eps,
                    ramp_steps=ramp_steps,
                    linear_rule=linear_rule_key,
                    delta_eta=delta_eta,
                    delta_exclude_cls_from_store=delta_exclude_cls_from_store,
                    delta_cls_only_readout=delta_cls_only_readout,
                    delta_learn_w0=delta_learn_w0,
                    delta_w0_rank=delta_w0_rank,
                )
        elif isinstance(attn, LoRAableLinearMHA):
            if attn_impl_key == "softmax":
                replacement = attn.to_softmax_mha()
            else:
                changed = attn.set_kernel(kernel=kernel, eps=eps)
                changed = attn.set_ramp(ramp_steps=ramp_steps) or changed
                changed = attn.set_linear_rule(
                    linear_rule=linear_rule_key,
                    delta_eta=delta_eta,
                    delta_exclude_cls_from_store=delta_exclude_cls_from_store,
                    delta_cls_only_readout=delta_cls_only_readout,
                    delta_learn_w0=delta_learn_w0,
                    delta_w0_rank=delta_w0_rank,
                ) or changed
                if changed:
                    n_patched += 1
        elif attn is not None:
            raise TypeError(
                f"Unsupported attention module type {type(attn)}. "
                "Expected nn.MultiheadAttention, LoRAableMHA, or LoRAableLinearMHA."
            )

        if replacement is not None:
            device, dtype = _module_device_dtype(attn)
            if device is not None or dtype is not None:
                replacement = replacement.to(device=device, dtype=dtype)
            replacement.train(attn.training)
            blk.attn = replacement
            n_patched += 1

    return n_patched


patch_openclip_vit_attn = split_openclip_vit_attn


def merge_openclip_vit_attn(visual: nn.Module) -> int:
    """
    Recompose LoRAable attention modules under visual.transformer.resblocks.*.attn
    back into fused nn.MultiheadAttention modules.
    """
    n_unpatched = 0
    transformer = getattr(visual, "transformer", None)
    if transformer is None:
        raise ValueError("visual has no .transformer; are you unpatching the right module?")
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        raise ValueError("visual.transformer has no .resblocks; unexpected OpenCLIP structure")

    for blk in resblocks:
        attn = getattr(blk, "attn", None)
        replacement: nn.Module | None = None

        if isinstance(attn, LoRAableLinearMHA):
            replacement = attn.to_softmax_mha().to_torch_mha()
        elif isinstance(attn, LoRAableMHA):
            replacement = attn.to_torch_mha()
        elif isinstance(attn, nn.MultiheadAttention):
            replacement = None
        elif attn is not None:
            raise TypeError(
                f"Unsupported attention module type {type(attn)}. "
                "Expected nn.MultiheadAttention, LoRAableMHA, or LoRAableLinearMHA."
            )

        if replacement is not None:
            blk.attn = replacement
            n_unpatched += 1

    return n_unpatched


def set_linear_attention_ramp_step(module: nn.Module, *, step: int) -> int:
    n_updated = 0
    for sub in module.modules():
        if isinstance(sub, LoRAableLinearMHA):
            sub.set_ramp_step(step)
            n_updated += 1
    return n_updated
