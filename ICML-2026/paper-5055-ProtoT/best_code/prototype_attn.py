# Prototype (low-rank) Broadcast Mixer
# ------------------------------------------------------------
# This module implements a global, low-rank token mixer via r "prototype hubs".
# It includes:
# (1) Value path (V/U projections): aggregate Pi^T * (V x) and project back via U.
# (2) Multiple low-rank terms as "heads": H heads, each with r hubs, summed in a ReZero residual.
# (5) Balance & sparsity controls: entropy and hub-balance regularizers; optional top-k or sparsemax routing.
#
# Notes on this specific variant:
# - The forward() below uses a prefix-mean aggregation path (cumulative cumsum) for P
#   That is, it computes per-timestep prototype means P_n up to each position and broadcasts using the current Pi.
#
# Shapes (conventions used below):
#   B: batch size
#   S: sequence length
#   C: feature dim of tokens
#   H: number of heads
#   r: hubs per head
#   C_v: value projection dim (per head)
#
# Public entry points:
#   - MixerConfig: configuration dataclass.
#   - ProtoBroadcastMixerUpgraded: nn.Module with forward() and step().
#
# Diagnostics & regularizers:
#   forward(..., return_aux=True) returns (y, aux) where aux may contain:
#     - 'entropy_loss': add to the training loss if w_entropy != 0
#     - 'balance_loss': add to the training loss if w_balance != 0
#     - 'alpha_per_head', 'tau_per_head', 'avg_hub_mass': quick monitoring stats.
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SwiGLU

# ---- Custom RMSNorm for PyTorch < 2.4 compatibility ----
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x = (x.float() / rms).to(x.dtype)
        if self.weight is not None:
            x = x * self.weight
        return x

# ---- Utilities ----

def _prefix_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}

def topk_softmax(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Softmax restricted to the top-k entries along a dimension (others masked to -inf)."""
    if k >= logits.size(dim):
        return F.softmax(logits, dim=dim)
    topk = torch.topk(logits, k=k, dim=dim)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(dim, topk.indices, topk.values)
    return F.softmax(mask, dim=dim)

def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016): produces probabilities with exact zeros."""
    # Move target dim to last for simpler manipulation
    z = logits.transpose(dim, -1)
    original_size = z.size()
    z = z.reshape(-1, z.size(-1))

    # Sort in descending order
    zs = torch.sort(z, descending=True, dim=-1).values

    # Compute cumulative thresholds
    cumsum_range = torch.arange(1, zs.size(-1) + 1, device=z.device, dtype=z.dtype).unsqueeze(0)
    cssv = torch.cumsum(zs, dim=-1)

    # Find k: largest k with 1 + k * zs_k > cssv_k
    cond = 1 + cumsum_range * zs > cssv
    k = cond.sum(dim=-1).clamp(min=1)  # at least 1

    # Compute tau for the active set
    idx = (k - 1).unsqueeze(-1)
    zs_k = torch.gather(zs, 1, idx).squeeze(1)
    cssv_k = torch.gather(cssv, 1, idx).squeeze(1)
    tau = (cssv_k - 1) / k

    # Final probabilities (clipped at 0)
    p = torch.clamp(z - tau.unsqueeze(-1), min=0)
    p = p.reshape(original_size).transpose(dim, -1)
    return p

def entropy_loss(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Average entropy over the given dimension (use +weight to encourage *lower* entropy; negative to increase it)."""
    eps = 1e-12
    pl = torch.clamp(p, min=eps)
    ent = -(pl * torch.log(pl)).sum(dim=dim)
    return ent.mean()

def hub_balance_loss(pi: torch.Tensor) -> torch.Tensor:
    """Encourage uniform hub utilization. Works for shapes [B,S,r] or [B,S,H,r]."""
    if pi.dim() == 4:
        B, S, H, r = pi.shape
        mass = pi.sum(dim=(0,1))                    # [H, r] total mass per head/hub
        mass = mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)
        uni = torch.full_like(mass, 1.0 / r)
        return ((mass - uni) ** 2).mean()
    else:
        B, S, r = pi.shape
        mass = pi.sum(dim=(0,1))                    # [r]
        mass = mass / (mass.sum() + 1e-8)
        uni = torch.full_like(mass, 1.0 / r)
        return ((mass - uni) ** 2).mean()
    

# ---- RoPE helper (single place) --------------------------------------------
class Rotary(nn.Module):
    """RoPE on a configurable fraction/sub-dimension of the last channel.
    Only the first `rot_dim` channels are rotated; the rest pass through.
    Args:   dim: total feature dimension of x
            base: RoPE base
            fraction: fraction of channels to rotate (0..1], rounded down to an even count
            rot_dim: alternatively, an explicit number of channels to rotate (takes precedence over `fraction`)
    """
    def __init__(self, dim: int, base: float = 1e6, fraction: float = 1.0, rot_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        if rot_dim is None:
            # even number of channels to rotate
            rot = int((dim * max(0.0, min(1.0, fraction))) // 2) * 2
        else:
            rot = int(rot_dim)
            rot = (rot // 2) * 2
        self.rot_dim = max(0, min(rot, dim))  # clamp for safety
        # Frequencies defined over the *rotated subspace* size
        if self.rot_dim > 0:
            inv = 1.0 / (base ** (torch.arange(0, self.rot_dim, 2, dtype=torch.float32) / self.rot_dim))
            self.register_buffer("inv_freq", inv, persistent=False)
        else:
            self.register_buffer("inv_freq", torch.empty(0, dtype=torch.float32), persistent=False)

    def _cos_sin(self, positions: torch.Tensor, dtype, device):
        if self.rot_dim == 0:
            return None, None
        inv = self.inv_freq  # [rot_dim/2]
        freqs = torch.einsum("s,d->sd", positions.to(inv.dtype), inv)  # [S, rot_dim/2]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def _rotate(x, cos, sin, rot_dim: int):
        # x: [B,S,C], rotate only first rot_dim channels (even), leave the tail as-is
        B, S, C = x.shape
        if rot_dim == 0:
            return x
        x_rot, x_tail = x[..., :rot_dim], x[..., rot_dim:]
        x_rot = x_rot.view(B, S, rot_dim // 2, 2)
        x0, x1 = x_rot[..., 0], x_rot[..., 1]  # [B,S,rot_dim/2]
        cos = cos.unsqueeze(0)                 # [1,S,rot_dim/2]
        sin = sin.unsqueeze(0)
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        y_rot = torch.stack((y0, y1), dim=-1).reshape(B, S, rot_dim)
        return torch.cat((y_rot, x_tail), dim=-1)

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: [B,S,C], positions: [S] (0..S-1)"""
        cos, sin = self._cos_sin(positions, x.dtype, x.device)
        if cos is None:
            return x
        return self._rotate(x, cos, sin, self.rot_dim)

    def apply_step(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """x: [B,1,C], t: int position"""
        if self.rot_dim == 0:
            return x
        pos = torch.tensor([t], device=x.device)
        cos, sin = self._cos_sin(pos, x.dtype, x.device)  # [1, rot_dim/2]
        return self._rotate(x, cos, sin, self.rot_dim)


# ---- Core module ----

@dataclass
class MixerConfig:
    """Configuration for the prototype mixer.
    Args:
        C: Token feature dimension.
        r: Hubs per head.
        num_heads: Number of independent low-rank terms to sum.
        C_v: Value projection dim; defaults to max(8, C//2).
        use_value_low_rank: If False, disable low-rank value bottleneck and use full token width in value stream.
        shared_routing: If False, use separate write and read routing.
        use_mass_norm: Divide each prototype summary by its routing mass to stabilize scale.
        sparsity: 'none' | 'topk' | 'sparsemax' for routing Π and optional read routing.
        k: Top-k value for 'topk' sparsity.
        w_entropy: Weight for entropy regularizer term returned in aux.
        w_balance: Weight for hub utilization balance regularizer returned in aux.
        tied_memory: Optional external memory for prototypes; shape [H*r, C] or [H, r, C].
    """
    C: int
    r: int
    num_heads: int = 1
    C_v: Optional[int] = None
    use_value_low_rank: bool = True
    shared_routing: bool = True
    use_mass_norm: bool = True
    sparsity: Literal["none", "topk", "sparsemax"] = "none"
    k: Optional[int] = None
    w_entropy: float = 0.0
    w_balance: float = 0.0
    tied_memory: Optional[torch.Tensor] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    # --- discounted prefix (EMA) options ---
    use_discounted_prefix: bool = True
    per_hub_beta: bool = True
    beta_init: float = 0.9   # start with long memory
    # --- RoPE positional options ---
    use_rope_q: bool = False  # rope on sequence positions only (and only in queries and keys)
    use_rope_v: bool = False  # rotate values on write; unrotate memories on read
    rope_q_fraction: float = 1.0  # fraction of channels (h_dim) to rotate for rope_q
    rope_v_fraction: float = 1.0  # fraction of channels (h_dim) to rotate for rope_v
    rope_base: float = 1e6  # 1e6 is more gentle than 1e4 at use_rope_q=True
    hub_dropout: float = 0.0
    local_conv_kernel_size: int = 5
    alpha_init: float = 1.0


class CausalDWConv1d(nn.Module):
    """Strictly causal depthwise conv on the sequence axis.
       Expects/returns [B, S, C]."""
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dw = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,   # depthwise
            padding=0,
            bias=bias,
        )

    def forward(self, x):                 # x: [B, S, C]
        pad = (self.kernel_size - 1) * self.dilation
        x = x.transpose(1, 2)             # [B, C, S]
        x = F.pad(x, (pad, 0))            # left pad only → causal
        y = self.dw(x)                    # [B, C, S]
        return y.transpose(1, 2)          # [B, S, C]


class ProtoBroadcastMixerUpgraded(nn.Module):
    """Prototype (low-rank) route→aggregate→broadcast mixer with upgrades.

    Forward path (this variant):
        1) Routing Π via dot(q, prototypes / τ).
        2) Value path Vx.
        3) Prefix aggregation: P_prefix[t] = sum_{i<=t} Π[i] ⊗ (Vx[i]) (optionally mass-normalized).
        4) Per-hub MLP on the prefix means P_n[t].
        5) Broadcast: y_v[t] = Π[t] · P_n[t].
        6) Project back with U and ReZero-gate per head, then residual-add to x.
    """
    def __init__(self, cfg: MixerConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        C = cfg.C
        H = cfg.num_heads
        r = cfg.r
        self.C_v_total = cfg.C_v if cfg.C_v is not None else max(8, C // 2)
        self.C_v = max(8, self.C_v_total // H)        # per-head bottleneck
        self.layer_id = layer_id

        # Per-head preprojections for queries/keys; keys are used for separate read routing.
        self.pre_q = nn.ModuleList([nn.Identity() for _ in range(H)]) if cfg.tied_memory is None else nn.ModuleList([nn.Identity() for _ in range(H)])
        if not cfg.shared_routing:
            self.pre_r  = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(H)])  # for read routing

        # Value and output projections per head.
        self.V = nn.ModuleList([nn.Linear(C, self.C_v, bias=False) for _ in range(H)])
        self.U = nn.ModuleList([nn.Linear(self.C_v, C, bias=False) for _ in range(H)])

        # Prototypes: tie to external memory if provided; otherwise learn per-head tables.
        self.use_memory = cfg.tied_memory is not None
        if self.use_memory:
            mem = cfg.tied_memory
            if mem.dim() == 2:
                # Memory provided as [H*r, C]; split evenly across heads.
                total_r = mem.size(0)
                assert total_r % H == 0, "tied_memory first dim must be divisible by num_heads"
                r_per_head = total_r // H
                self.r_per_head = r_per_head
                self.register_buffer("M", mem)
            elif mem.dim() == 3:
                # Memory provided as [H, r, C]; flatten head/hub for a uniform accessor.
                assert mem.size(0) == H and mem.size(2) == C
                self.r_per_head = mem.size(1)
                self.register_buffer("M", mem.reshape(H * mem.size(1), C))
            else:
                raise ValueError("tied_memory must be [total_r, C] or [H, r, C]")
        else:
            assert cfg.r % H == 0, "Set cfg.r as TOTAL hubs; divisible by num_heads"
            self.r_per_head = cfg.r // H
            self.P_tables = nn.ParameterList([nn.Parameter(torch.randn(self.r_per_head, C) / math.sqrt(C)) for _ in range(H)])
            #self.P_tables_read = nn.ParameterList([nn.Parameter(torch.randn(self.r_per_head, C) / math.sqrt(C)) for _ in range(H)]) if not cfg.shared_routing else None
            with torch.no_grad():  # Normalize each prototype vector at initialization.
                for p in self.P_tables:
                    p.copy_(p / p.norm(dim=1, keepdim=True).clamp_min(1e-6))

        # --- per-head(/per-hub) EMA decay for discounted prefix means ---
        if cfg.use_discounted_prefix:
            shape = (H, self.r_per_head) if cfg.per_hub_beta else (H, 1)
            init = math.log(cfg.beta_init) - math.log(1.0 - cfg.beta_init)  # logit(beta_init)
            self.logit_beta = nn.Parameter(torch.full(shape, init))
        else:
            self.logit_beta = None

        # --- RoPE options ---
        if cfg.use_rope_q:
            self.rope_q   = Rotary(C, base=cfg.rope_base, fraction=cfg.rope_q_fraction)  
        else:
            self.rope_q = None                          # query RoPE disabled
        if cfg.use_rope_v:
            self.rope_v   = Rotary(self.C_v, base=cfg.rope_base, fraction=cfg.rope_v_fraction)  # rotate values on write (and unrotate memories on read)
        else:
            self.rope_v = None

        # ReZero gate alpha and routing temperature tau per head.
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(cfg.alpha_init)) for _ in range(H)])
        self.tau = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(H)])  # learned routing temperature
        if not cfg.shared_routing:
            self.tau_read = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(H)])  # for read routing

        self.local_dw = CausalDWConv1d(
            channels=self.C_v,
            kernel_size=cfg.local_conv_kernel_size,
            dilation=1,
            bias=False,
        )  # local context path
        # Intervention masks for rooting (optional; set via public methods below). Shape [num_heads, hubs_per_head], broadcast over batch/seq.
        self.write_mask: Optional[torch.Tensor] = None
        self.read_mask: Optional[torch.Tensor] = None

    # --- helpers ---

    def _get_head_protos(self, h: int) -> torch.Tensor:
        """Return the r×C prototype matrix for head h (tied memory or learnable table)."""
        if self.use_memory:
            start = h * self.r_per_head
            end = (h + 1) * self.r_per_head
            return self.M[start:end]                 # [r, C]
        else:
            return self.P_tables[h]                  # [r, C]

    def _route(self, q: torch.Tensor, protos: torch.Tensor, tau: torch.Tensor,
               sparsity: str, k: Optional[int], return_scores: bool = False,
               head_idx: Optional[int] = None, is_read: bool = False) -> torch.Tensor:
        """Compute routing Π from queries and prototypes.
        Args:
            q: [B, S, C] token queries (optionally pre-projected)
            protos: [r, C] head-specific prototypes
            tau: scalar temperature (per head)
            sparsity: 'none' | 'topk' | 'sparsemax'
            k: top-k value if sparsity='topk'
            head_idx: head index for optional intervention masks
            is_read: whether to apply the read mask rather than the write mask
        Returns:
            Π: [B, S, r] routing probabilities per token over hubs.
        """
        #protos = F.normalize(protos, p=2, dim=1, eps=1e-6)  # optional row-normalized prototypes for stability
        #q = F.normalize(q, p=2, dim=-1, eps=1e-6)          # norm queries too; perf is similar, but extra compute
        scores = torch.einsum("bsc,rc->bsr", q, protos) / (tau.abs() + 1e-6)
        mask_to_use = self.read_mask if is_read else self.write_mask
        if mask_to_use is not None and head_idx is not None:
            head_mask = mask_to_use[head_idx].to(scores.device, dtype=scores.dtype)
            score_mask = torch.where(
                head_mask <= 0,
                torch.full_like(head_mask, -float("inf")),
                torch.zeros_like(head_mask),
            )
            scores = scores + score_mask.view(1, 1, -1)
        if sparsity == "topk" and k is not None:
            probs = topk_softmax(scores, k=k, dim=-1)
        elif sparsity == "sparsemax":
            probs = sparsemax(scores, dim=-1)
        else:
            probs = torch.softmax(scores, dim=-1)
        return (probs, scores) if return_scores else probs

    def _normalize_mask(self, mask: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=self.alpha[0].dtype)
        if mask.dim() == 1:
            if self.cfg.num_heads != 1:
                raise ValueError(f"1D {name} mask is only supported when num_heads == 1")
            mask = mask.unsqueeze(0)
        if mask.dim() != 2:
            raise ValueError(f"{name} mask must have shape [num_heads, hubs_per_head]")
        if mask.size(0) not in (1, self.cfg.num_heads):
            raise ValueError(f"{name} mask head dimension mismatch")
        if mask.size(1) != self.r_per_head:
            raise ValueError(f"{name} mask prototype dimension mismatch")
        mask = mask.to(device=self.alpha[0].device, dtype=self.alpha[0].dtype)
        if mask.size(0) == 1 and self.cfg.num_heads > 1:
            mask = mask.expand(self.cfg.num_heads, -1)
        return mask

    def set_write_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Set a per-head write mask where positive values keep a hub and non-positive values block it."""
        self.write_mask = self._normalize_mask(mask, "write")

    def clear_write_mask(self) -> None:
        self.write_mask = None

    def set_read_mask(self, mask: Optional[torch.Tensor]) -> None:
        """Set a per-head read mask where positive values keep a hub and non-positive values block it."""
        self.read_mask = self._normalize_mask(mask, "read")

    def clear_read_mask(self) -> None:
        self.read_mask = None

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        pad_mask: Optional[torch.Tensor] = None,
        layer_id: Optional[int] = None,
        *,
        force_write: Optional[torch.Tensor] = None,
        force_read: Optional[torch.Tensor] = None,
    ):
        """Prefix-mean prototype mixing over a full sequence.
        Args:
            x: [B, S, C] token features
            return_aux: if True, return (y, aux) including regularizer terms and diagnostics
        Returns:
            y or (y, aux): residual-mixed output (same shape as x)
        """
        cfg = self.cfg
        B, S, C = x.shape
        H, r = cfg.num_heads, self.r_per_head

        # Precompute positions once for both stages when using RoPE
        pos = torch.arange(S, device=x.device) if (cfg.use_rope_q or cfg.use_rope_v) else None

        all_y = []
        ent_terms = []
        pi_collect = []  # keep Π per head for balance loss / diagnostics

        m = None
        if pad_mask is not None:
            # True = pad ⇒ 0 contribution/mass
            m = (~pad_mask).to(x.dtype).unsqueeze(-1)   # [B,S,1]

        for h in range(H):
            # Prototypes & queries for routing
            protos = self._get_head_protos(h)        # [r, C]
            q_in = self.pre_q[h](x)                  # [B, S, C] (Identity if tied memory)
            # --- RoPE on TOKEN side (≈ Key=X) ---
            if self.rope_q is not None:
                q_in = self.rope_q.apply(q_in, pos)    # rotate after pre_q: correct rotary space
            tau = self.tau[h].abs() + 1e-6           # positive scalar

            # Routing: Π[b, s, k] over hubs k
            route_out = self._route(q_in, protos, tau, cfg.sparsity, cfg.k, return_scores=return_aux, head_idx=h)   # [B, S, r]
            if return_aux:
                Pi, scores = route_out
            else:
                Pi = route_out 

            # --- Hub Dropout (if enabled) ---
            if self.training and cfg.hub_dropout > 0.0:
                hub_mask = torch.bernoulli(torch.full((B, S, r), 1.0 - cfg.hub_dropout, device=Pi.device)).to(Pi.dtype)
                Pi = Pi * hub_mask
                Pi = Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)  # keep normalization differentiable

            if force_write is not None:
                fw = torch.as_tensor(force_write, device=Pi.device, dtype=Pi.dtype)
                if fw.dim() == 1:
                    fw = fw.view(1, 1, -1).expand(B, 1, -1)
                elif fw.dim() == 2:
                    fw = fw.unsqueeze(1)
                if fw.shape != (B, 1, r):
                    raise ValueError(
                        f"force_write must have shape [{r}] or [{B}, {r}], "
                        f"got {tuple(fw.shape)}"
                    )
                Pi = Pi.clone()
                Pi[:, -1, :] = torch.softmax(fw[:, 0, :], dim=-1)

            # Mask pads so they do not contribute sums or mass
            Pi_eff = Pi if m is None else Pi * m                      # [B,S,r]
            if return_aux:  # Capture diagnostics w/o affecting training
                with torch.no_grad():
                    # 1) Router confidence & sparsity
                    tk = torch.topk(Pi_eff, k=min(2, r), dim=-1).values                   # [B,S,<=2]
                    top1 = tk[..., 0]                                                     # [B,S]
                    top2 = tk[..., 1] if r >= 2 else torch.zeros_like(top1)
                    margin = (top1 - top2).mean()
                    # : exact sparsity if sparsemax/topk, otherwise thresholded proxy
                    active_exact = (Pi_eff > 0).float().sum(dim=-1).float().mean()        # works esp. for sparsemax
                    active_t01   = (Pi_eff > 0.01).float().sum(dim=-1).float().mean()
                    active_t001  = (Pi_eff > 0.001).float().sum(dim=-1).float().mean()
                    # 2) Load concentration & dead hubs
                    hub_mass = Pi_eff.sum(dim=(0, 1))                                     # [r]
                    share = hub_mass / (hub_mass.sum() + 1e-12)
                    max_share = share.max()
                    # Gini (robust, differentiability not needed)
                    sorted_share, _ = torch.sort(share)
                    r_idx = torch.arange(1, r + 1, device=share.device, dtype=share.dtype)
                    gini = 2 * (r_idx * sorted_share).sum() / (r * sorted_share.sum().clamp_min(1e-12)) - (r + 1) / r
                    dead_frac = (hub_mass < 1e-6).float().mean()
                    # 3) Logit scale (if requested)
                    logit_mean = scores.mean() if return_aux else torch.tensor(0., device=Pi.device)
                    logit_std  = scores.std()  if return_aux else torch.tensor(0., device=Pi.device)
            pi_collect.append(Pi_eff)   # use masked Π for balance stats/aux
            # --- Value path: project tokens to C_v, and rotate if using RoPE ---
            Vx = self.V[h](x)                        # [B, S, C_v]
            if self.rope_v is not None:                      # apply configured Rotary
                Vx = self.rope_v.apply(Vx, pos)              # rotate values by pos
            # Note: per-head RMSNorm on Vx improves performance slightly, but makes training slower

            # Causal convolution for L0 and L1 only
            # Captures local context and substantially reduces alpha collapse at L0, with a small slowdown.
            if self.layer_id < 2:
                Vx_local = self.local_dw(Vx)
                Vx = 0.5*Vx + 0.5*Vx_local

            if cfg.use_discounted_prefix:
                # --- discounted (EMA) prefix means (in fp32 for numeric stability) ---
                with torch.autocast(device_type='cuda' if x.is_cuda else 'cpu', enabled=False):
                    beta = torch.sigmoid(self.logit_beta[h])                # [r] or [1]
                    if beta.numel() == 1:
                        beta = beta.expand(r)                               # [r]
                    b = beta.view(1, 1, r, 1).float()                       # [1,1,r,1] (fp32)
                    s_idx = torch.arange(S, device=x.device, dtype=torch.float32).view(1, S, 1, 1)
                    b_pow = torch.pow(b, s_idx)                             # [1,S,r,1]

                    # contributions over time: X = Π ⊗ (Vx)
                    contrib = Pi_eff.float().unsqueeze(-1) * Vx.float().unsqueeze(2)   # [B,S,r,C_v]
                    # sum_state via scaled cumsum: y_s = b^s * cumsum( (1-b) * x_s / b^s )
                    tilde = (1.0 - b) * contrib / (b_pow + 1e-12)           # [B,S,r,C_v]
                    sum_state_all = torch.cumsum(tilde, dim=1) * b_pow      # [B,S,r,C_v]

                    if cfg.use_mass_norm:
                        b_pow_m = b_pow.squeeze(-1)                         # [1,S,r]
                        mass_tilde = (1.0 - beta.view(1,1,r).float()) * Pi_eff.float() / (b_pow_m + 1e-12)  # [B,S,r]
                        mass_all = torch.cumsum(mass_tilde, dim=1) * b_pow_m            # [B,S,r]
                        # Shift right by one position so the memory path reads from the past only.
                        # This removes same-position write-then-read behavior.
                        sum_prev  = torch.roll(sum_state_all, shifts=1, dims=1)
                        sum_prev[:, 0].zero_()
                        mass_prev = torch.roll(mass_all,      shifts=1, dims=1)
                        mass_prev[:, 0].zero_()
                        Pn = sum_prev / (mass_prev.unsqueeze(-1).clamp_min(1e-6))            # [B,S,r,C_v]
                        if return_aux:  # Capture diagnostics w/o affecting training
                            with torch.no_grad():
                                # Prefix mass health (avoid divide-by-small)
                                mp = mass_prev[:, 1:]  # [B,S-1,r] : ignore t=0 which is always zero
                                mp_flat = mp.reshape(-1)
                                # Robust summary without incurring a quantile CUDA kernel.
                                prefix_mass_min = mp_flat.min()
                                prefix_mass_med = mp_flat.median()
                                # tiny-mass fraction (could indicate instability)
                                prefix_mass_lt1e3 = (mp_flat < 1e-3).float().mean()
                                prefix_mass_lt1e4 = (mp_flat < 1e-4).float().mean()
                    else:
                        Pn = sum_state_all  
                        Pn = torch.roll(Pn, shifts=1, dims=1); Pn[:,0].zero_()  # shift right by 1 along S, as above

                # back to model dtype (bf16/half/etc.)
                Pn = Pn.to(x.dtype)

            else:
                # Original prefix-mean path (cumsum)
                contrib   = Pi_eff.unsqueeze(-1) * Vx.unsqueeze(2)          # [B, S, r, C_v]
                P_prefix  = torch.cumsum(contrib, dim=1)                # [B, S, r, C_v]
                if cfg.use_mass_norm:
                    mass_prefix = torch.cumsum(Pi_eff, dim=1).unsqueeze(-1) + 1e-6
                    Pn = P_prefix / mass_prefix
                else:
                    Pn = P_prefix
                Pn = torch.roll(Pn, shifts=1, dims=1); Pn[:, 0].zero_()  # shift right by 1 along S, pad zero at s=0

            if return_aux:
                with torch.no_grad():
                    # Scale of signals
                    Vx_mean_norm = Vx.pow(2).sum(dim=-1).sqrt().mean()
                    Pn_mean_norm = Pn.pow(2).sum(dim=-1).sqrt().mean()

            # --- Memory: unrotate (RoPE) on READ (per hub; do not flatten over r) ---
            if self.rope_v is not None:
                # reshape to rotate within each hub independently
                Pn = Pn.permute(0, 2, 1, 3).reshape(B * r, S, self.C_v)   # [B*r, S, C_v]
                Pn = self.rope_v.apply(Pn, -pos)                          # inverse rotation
                Pn = Pn.view(B, r, S, self.C_v).permute(0, 2, 1, 3)       # back to [B,S,r,C_v]

            # This read-routing is basically helping aggregate Pn across hubs
            if cfg.shared_routing:
                yv = torch.einsum("bsr,bsrc->bsc", Pi_eff, Pn)              # [B, S, C_v]
            else:
                # Read gate R (independent of write gate)
                q_r   = self.pre_r[h](x)                                  # [B,S,C]
                # --- RoPE on TOKEN side (Query=XW) ---
                if self.rope_q is not None:
                    q_r = self.rope_q.apply(q_r, pos)  # rotate after pre_r: same rotary space as scores
                tau_r = self.tau_read[h].abs() + 1e-6
                Pi_read = self._route(q_r, protos, tau_r, sparsity='none', k=cfg.k, head_idx=h, is_read=True)  # [B,S,r]  # sparsity only on write Π
                if force_read is not None:
                    fr = torch.as_tensor(force_read, device=Pi_read.device, dtype=Pi_read.dtype)
                    if fr.dim() == 1:
                        fr = fr.view(1, 1, -1).expand(B, 1, -1)
                    elif fr.dim() == 2:
                        fr = fr.unsqueeze(1)
                    if fr.shape != (B, 1, r):
                        raise ValueError(
                            f"force_read must have shape [{r}] or [{B}, {r}], "
                            f"got {tuple(fr.shape)}"
                        )
                    Pi_read = Pi_read.clone()
                    Pi_read[:, -1, :] = torch.softmax(fr[:, 0, :], dim=-1)
                Pi_read_eff = Pi_read if m is None else Pi_read * m
                yv = torch.einsum("bsr,bsrc->bsc", Pi_read_eff, Pn)              # [B, S, C_v]

            # Project back to C and gate
            y_head = self.U[h](yv)                                     # [B, S, C]
            y_head = self.alpha[h] * y_head                            # ReZero gate per head
            all_y.append(y_head)

            # Optional routing entropy term (for regularization)
            if cfg.w_entropy != 0.0:
                if pad_mask is None:
                    ent_terms.append(entropy_loss(Pi, dim=-1))
                else:  # account for pads in entropy
                    eps = 1e-12
                    H_ent = -(Pi.clamp_min(eps) * Pi.clamp_min(eps).log()).sum(dim=-1)  # [B,S]
                    w = (~pad_mask).float()
                    ent_terms.append((H_ent * w).sum() / (w.sum().clamp_min(1.0)))

        # Sum contributions from all heads
        y = torch.stack(all_y, dim=0).sum(dim=0) / math.sqrt(cfg.num_heads)      # [B, S, C]

        if return_aux:  # Capture diagnostics w/o affecting training
            # Build aux dict with regularizers and quick stats
            aux: Dict[str, Any] = {}
            if cfg.w_entropy != 0.0:
                aux["entropy_loss"] = cfg.w_entropy * torch.stack(ent_terms).mean()
            if cfg.w_balance != 0.0:
                if len(pi_collect) > 1:
                    Pi_all = torch.stack(pi_collect, dim=2)                # [B, S, H, r]
                else:
                    Pi_all = pi_collect[0]                                 # [B, S, r]
                aux["balance_loss"] = cfg.w_balance * hub_balance_loss(Pi_all)
            # Add actual routing matrix Pi to aux for analysis
            if len(pi_collect) > 1:
                aux["routing_Pi"] = torch.stack(pi_collect, dim=2)  # [B, S, H, r]
            else:
                aux["routing_Pi"] = pi_collect[0].unsqueeze(2)      # [B, S, 1, r] → consistent shape
            with torch.no_grad():
                aux["alpha_per_head"] = torch.stack([a.detach() for a in self.alpha])     # [H]
                aux["tau_per_head"]   = torch.stack([t.detach() for t in self.tau])       # [H]
                if not cfg.shared_routing and hasattr(self, "tau_read"):
                    aux["tau_read_per_head"] = torch.stack([t.detach() for t in self.tau_read])  # [H]

                if len(pi_collect) > 0:
                    if len(pi_collect) == 1:
                        mass = pi_collect[0].mean(dim=(0, 1))                              # [r]
                    else:
                        mass = torch.stack([pi.mean(dim=(0, 1)) for pi in pi_collect])     # [H, r]
                    aux["avg_hub_mass"] = mass.detach()
                else:
                    aux["avg_hub_mass"] = torch.empty(0, device=x.device, dtype=x.dtype)

            if not cfg.shared_routing:
                aux["Pi_read"] = Pi_read.detach()  # read routing matrix
                aux["read_routing_Pi"] = Pi_read.detach()

            with torch.no_grad():
                # Router confidence & sparsity
                aux["router_top1_mean"] = top1.mean().detach()
                aux["router_margin_mean"] = margin.detach()
                aux["active_hubs_per_token_exact"] = active_exact.detach()
                aux["active_hubs_per_token@1%"] = active_t01.detach()
                aux["active_hubs_per_token@0.1%"] = active_t001.detach()
                # Load concentration
                aux["hub_max_share"] = max_share.detach()
                aux["hub_gini"] = gini.detach()
                aux["hub_dead_frac"] = dead_frac.detach()
                # Logit scale
                aux["router_logit_mean"] = logit_mean.detach()
                aux["router_logit_std"]  = logit_std.detach()
                # Prefix mass (only present if discounted prefix path executed)
                if cfg.use_discounted_prefix:
                    aux["prefix_mass_min"] = prefix_mass_min.detach()
                    aux["prefix_mass_median"] = prefix_mass_med.detach()
                    aux["prefix_mass_lt1e3_frac"] = prefix_mass_lt1e3.detach()
                    aux["prefix_mass_lt1e4_frac"] = prefix_mass_lt1e4.detach()
                # Signal scales (optional but handy)
                aux["Vx_mean_norm"] = Vx_mean_norm.detach()
                aux["Pn_mean_norm"] = Pn_mean_norm.detach()
                # Memory half-life from β
                if self.logit_beta is not None:
                    beta = torch.sigmoid(self.logit_beta[h])
                    # effective window and half-life
                    eff_window = 1.0 / (1.0 - beta.clamp(max=0.999999))
                    half_life = torch.log(torch.tensor(0.5, device=beta.device, dtype=beta.dtype)) / torch.log(beta.clamp(min=1e-6))
                    aux["beta_mean"] = beta.mean().detach()
                    aux["memory_eff_window_mean"] = eff_window.mean().detach()
                    aux["memory_half_life_mean"] = half_life.mean().detach()

            if not cfg.shared_routing:
                # Compare write vs read gates
                with torch.no_grad():
                    # cosine similarity per token
                    num = (Pi_read * Pi_eff).sum(-1)
                    den = (Pi_read.norm(dim=-1) * Pi_eff.norm(dim=-1)).clamp_min(1e-9)
                    cos = (num / den).mean()
                    # JS divergence (symmetrized KL on probs)
                    eps = 1e-12
                    M = 0.5 * (Pi_read.clamp_min(eps) + Pi_eff.clamp_min(eps))
                    kl_a = (Pi_eff * (Pi_eff.clamp_min(eps).log() - M.log())).sum(-1).mean()
                    kl_b = (Pi_read * (Pi_read.clamp_min(eps).log() - M.log())).sum(-1).mean()
                    aux["read_write_cosine"] = cos.detach()
                    aux["read_write_JS"] = (0.5 * (kl_a + kl_b)).detach()

            if layer_id is not None:  # add prefix if layer_ids is provided
                aux = _prefix_dict(aux, f"L{layer_id}.")
            return (y, aux)
        else:
            return y


class FFN(nn.Module):
    """
    Proper SwiGLU FFN (Shazeer, 2020): y = W_o( SiLU(W_g x) ⊙ W_v x )
    Using mult≈2.67 gives ~8·C^2 FLOPs (similar to LLaMA's gated FFN), vs ~16·C^2 before.
    """
    def __init__(self, dim, ffn_inner_size: int, dropout=0.0):
        super().__init__()
        self.swiglu = SwiGLU(dim, ffn_inner_size)
        self.proj = nn.Linear(ffn_inner_size, dim,  bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.swiglu(x)
        return self.proj(self.drop(x))


class MixerBlock(nn.Module):
    def __init__(self, dim, cfg_mixer: MixerConfig, ffn_inner_size: int, dropout=0.0, use_rmsnorm=True, layer_id: Optional[int] = None):
        super().__init__()
        self.layer_id = layer_id
        self.norm1 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        self.mixer = ProtoBroadcastMixerUpgraded(cfg_mixer, layer_id=layer_id)
        self.norm2 = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)
        self.ffn   = FFN(dim, ffn_inner_size=ffn_inner_size, dropout=dropout)
        self.resid_drop1 = nn.Dropout(dropout)
        self.resid_drop2 = nn.Dropout(dropout)

    def set_write_mask(self, mask: Optional[torch.Tensor]) -> None:
        self.mixer.set_write_mask(mask)

    def clear_write_mask(self) -> None:
        self.mixer.clear_write_mask()

    def set_read_mask(self, mask: Optional[torch.Tensor]) -> None:
        self.mixer.set_read_mask(mask)

    def clear_read_mask(self) -> None:
        self.mixer.clear_read_mask()

    def forward(
        self,
        x,
        pad_mask=None,
        return_aux: bool = False,
        *,
        force_write: Optional[torch.Tensor] = None,
        force_read: Optional[torch.Tensor] = None,
    ):
        if return_aux:
            mixer_out, aux = self.mixer(
                self.norm1(x),
                pad_mask=pad_mask,
                return_aux=True,
                layer_id=self.layer_id,
                force_write=force_write,
                force_read=force_read,
            )
            x = x + self.resid_drop1(mixer_out)
            x = x + self.resid_drop2(self.ffn(self.norm2(x)))
            return x, aux
        else:
            x = x + self.resid_drop1(self.mixer(
                self.norm1(x),
                pad_mask=pad_mask,
                force_write=force_write,
                force_read=force_read,
            ))
            x = x + self.resid_drop2(self.ffn(self.norm2(x)))
            return x


class MixerStack(nn.Module):
    def __init__(self, dim, r, depth, ffn_inner_size: int, dropout=0.0, use_rmsnorm=True,
                 w_entropy: float = 0.0, w_balance: float = 0.0, local_conv_kernel_size: int = 5,
                 alpha_init: float = 1.0, use_mass_norm: bool = True, use_value_low_rank: bool = True,
                 hub_dropout: float = 0.0,
                 tau_init: float = 1.0):
        super().__init__()
        # Only difference across this ablation: value width (C/2 vs C).
        C_v = int(dim / 2) if use_value_low_rank else dim

        cfg = MixerConfig(
            C=dim, r=r, num_heads=1, C_v=C_v,
            shared_routing=False, use_mass_norm=use_mass_norm,  # use_mass_norm helps with perf (initial result)
            use_value_low_rank=use_value_low_rank,
            sparsity="none", k=4,  # none, topk, or sparsemax sparsity in write-routing (cross-attn #1)
            use_discounted_prefix=True,  # this is a type of positional information ~ pos embedding
            use_rope_q=False,  # RoPE on sequence positions in queries and keys
            use_rope_v=False,  # rope on values (write) and memories (read)
            w_entropy=w_entropy,  # set via outer config
            w_balance=w_balance,  # set via outer config
            hub_dropout=hub_dropout,
            local_conv_kernel_size=local_conv_kernel_size,
            alpha_init=alpha_init,
        )
        import copy  # Do copy.deepcopy(cfg) to ensure each block has its own config instance for targeted edits
        self.blocks = nn.ModuleList([
            MixerBlock(dim, copy.deepcopy(cfg), ffn_inner_size=ffn_inner_size, dropout=dropout, use_rmsnorm=use_rmsnorm, layer_id=idx) for idx in range(depth)])
        self.blocks[0].mixer.cfg.shared_routing = True  # only for L0, to prevent alpha-collapse
        self.blocks[0].mixer.tau[0].data.fill_(3.0)  # sharper initial routing at L0: reduces alpha-collapse at L0, without hurting ppl
        #self.blocks[1].mixer.cfg.shared_routing = True  # similar ppl, but faster; disabled due to tuning for more than 6 layers
        #self.blocks[1].mixer.tau[0].data.fill_(3.0)  # sharper initial routing at L1: helps with ppl (slightly)?
        #self.blocks[2].mixer.cfg.shared_routing = True  # similar ppl, but faster; disabled due to tuning for more than 6 layers
        #self.blocks[2].mixer.tau[0].data.fill_(3.0)  # sharper initial routing at L2: helps with ppl (slightly)?
        # Shared routing on block >=3 harms ppl
        # Per-layer tau initialization (configurable)
        if tau_init != 1.0:
            for idx in range(1, depth):
                self.blocks[idx].mixer.tau[0].data.fill_(tau_init)
                if hasattr(self.blocks[idx].mixer, "tau_read"):
                    for h in range(len(self.blocks[idx].mixer.tau_read)):
                        self.blocks[idx].mixer.tau_read[h].data.fill_(tau_init)
        self.norm_out = RMSNorm(dim)

    def set_write_mask(self, layer_idx: int, mask: Optional[torch.Tensor]) -> None:
        self.blocks[layer_idx].set_write_mask(mask)

    def clear_all_write_masks(self) -> None:
        for blk in self.blocks:
            blk.clear_write_mask()

    def set_read_mask(self, layer_idx: int, mask: Optional[torch.Tensor]) -> None:
        self.blocks[layer_idx].set_read_mask(mask)

    def clear_all_read_masks(self) -> None:
        for blk in self.blocks:
            blk.clear_read_mask()

    def forward(
        self,
        x,
        pad_mask=None,
        return_aux: bool = False,
        *,
        force_write: Optional[torch.Tensor] = None,
        force_read: Optional[torch.Tensor] = None,
    ):
        if return_aux:
            aux_collector = []
            for blk in self.blocks:
                x, aux = blk(
                    x,
                    pad_mask=pad_mask,
                    return_aux=True,
                    force_write=force_write,
                    force_read=force_read,
                )
                aux_collector.append(aux)
            x = self.norm_out(x)
            # Expose per-layer stats without "L{i}." prefixes; keep last-layer keys at top-level
            def _strip_prefix(d, prefix: str):
                out = {}
                for k, v in d.items():
                    if isinstance(k, str) and k.startswith(prefix):
                        out[k[len(prefix):]] = v
                    else:
                        out[k] = v
                return out

            if aux_collector:
                per_layer = []
                for i, d in enumerate(aux_collector):
                    per_layer.append(_strip_prefix(d, f"L{i}."))
                merged_aux = {"per_layer": per_layer}
                def _agg(key, reduce="sum"):
                    vals = [d[key] for d in per_layer if key in d]
                    if not vals: 
                        return None
                    t = torch.stack(vals)
                    return t.sum() if reduce == "sum" else t.mean()
                ent = _agg("entropy_loss", reduce="mean")      # or "sum", depending on desired depth scaling
                bal = _agg("balance_loss", reduce="mean")      # --||--
                if ent is not None: merged_aux["entropy_loss"] = ent
                if bal is not None: merged_aux["balance_loss"] = bal
            else:
                merged_aux = {}
            return x, merged_aux
        else:
            for blk in self.blocks:
                x = blk(
                    x,
                    pad_mask=pad_mask,
                    force_write=force_write,
                    force_read=force_read,
                )
            return self.norm_out(x)


# ---- Embedding + Output Projection (LM head) wrapper ----

class ProtoBroadcastLM(nn.Module):
    """
    Lightweight autoregressive LM wrapper:
      - token embedding (+ learned absolute position embedding)
      - MixerStack backbone (causal)
      - output projection with optional weight tying to token embedding
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        r: int,
        max_seq_len: int,
        ffn_inner_size: int,
        dropout: float = 0.0,
        pad_id: int = 0,
        tie_weights: bool = True,
        w_entropy: float = 0.0,
        w_balance: float = 0.0,
        local_conv_kernel_size: int = 5,
        alpha_init: float = 1.0,
        use_mass_norm: bool = True,
        use_value_low_rank: bool = True,
        hub_dropout: float = 0.0,
        tau_init: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.tie_weights = tie_weights

        # embeddings
        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)  # padding_idx=pad_id makes no difference on perf or speed
        self.drop = nn.Dropout(dropout)

        # mixer backbone (causal)
        self.backbone = MixerStack(
            dim=dim, r=r, depth=depth, ffn_inner_size=ffn_inner_size, dropout=dropout,
            w_entropy=w_entropy, w_balance=w_balance, local_conv_kernel_size=local_conv_kernel_size,
            alpha_init=alpha_init, use_mass_norm=use_mass_norm, use_value_low_rank=use_value_low_rank,
            hub_dropout=hub_dropout,
            tau_init=tau_init)

        # LM head (optionally tied to tok_emb)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # init
        self.reset_parameters()

    def set_write_mask(self, layer_idx: int, mask: Optional[torch.Tensor]) -> None:
        self.backbone.set_write_mask(layer_idx, mask)

    def clear_all_write_masks(self) -> None:
        self.backbone.clear_all_write_masks()

    def set_read_mask(self, layer_idx: int, mask: Optional[torch.Tensor]) -> None:
        self.backbone.set_read_mask(layer_idx, mask)

    def clear_all_read_masks(self) -> None:
        self.backbone.clear_all_read_masks()

    def reset_parameters(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        if not self.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ---- full-sequence forward ----
    def forward(
        self,
        input_ids: torch.LongTensor,
        pad_mask=None,
        return_logits: bool = True,
        return_aux: bool = False,
        *,
        force_write: Optional[torch.Tensor] = None,
        force_read: Optional[torch.Tensor] = None,
    ):
        """
        input_ids: [B, S]
        pad_mask: [B, S] (optional)
        return_logits: if False, return last hidden state
        return_aux: if True, return (output, aux_dict) with routing info
        """
        x = self.tok_emb(input_ids)                                     # [B, S, C]
        x = self.drop(x)
    
        if return_aux:
            # Forward through backbone with aux collection
            x, aux_outputs = self.backbone(
                x,
                pad_mask=pad_mask,
                return_aux=True,
                force_write=force_write,
                force_read=force_read,
            )
            logits = self.lm_head(x)                                    # [B, S, V]
            if return_logits:
                return logits, aux_outputs
            else:
                return x, aux_outputs
        else:
            # Normal forward
            x = self.backbone(
                x,
                pad_mask=pad_mask,
                force_write=force_write,
                force_read=force_read,
            )                                                           # [B, S, C]
            logits = self.lm_head(x)                                    # [B, S, V]
            return logits if return_logits else x

    @torch.no_grad()
    def get_last_routing(self, text, tokenizer=None, device=None):
        """Return the final layer's write/read routing vectors at the last token."""
        if not callable(tokenizer):
            raise TypeError("tokenizer must be a callable that maps text to token IDs")
        dev = device or next(self.parameters()).device
        ids = torch.tensor([tokenizer(text)], device=dev)

        _, aux = self.forward(ids, return_aux=True)
        per_layer = aux.get("per_layer", [])
        if not per_layer:
            raise RuntimeError("routing diagnostics were not returned by the backbone")
        last = per_layer[-1]

        pi_write = last.get("routing_Pi")
        if pi_write is None:
            raise RuntimeError("write routing diagnostics are unavailable")
        write_vec = pi_write[0, -1, 0, :].detach().cpu()

        pi_read = last.get("Pi_read")
        if pi_read is None:
            read_vec = write_vec
        else:
            read_vec = pi_read[0, -1, :].detach().cpu()

        return {"write_weights": write_vec, "read_weights": read_vec}

    @torch.no_grad()
    def next_token_probs_forced(
        self,
        text,
        tokenizer=None,
        device=None,
        force_write=None,
        force_read=None,
    ):
        """Return next-token probabilities with last-token routing clamped."""
        if not callable(tokenizer):
            raise TypeError("tokenizer must be a callable that maps text to token IDs")
        dev = device or next(self.parameters()).device
        ids = torch.tensor([tokenizer(text)], device=dev)

        if force_write is not None:
            force_write = torch.as_tensor(force_write, device=dev, dtype=torch.float32)
        if force_read is not None:
            force_read = torch.as_tensor(force_read, device=dev, dtype=torch.float32)

        logits = self.forward(ids, force_write=force_write, force_read=force_read)
        return F.softmax(logits[0, -1], dim=-1)
