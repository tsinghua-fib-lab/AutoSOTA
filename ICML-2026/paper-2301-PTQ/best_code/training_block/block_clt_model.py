import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Shape suffixes:
# B: Batch Size
# L: Total number of LM layers
# D: CLT Latent Dim (d_hidden)
# H: Embedding Dimension of LM (d_model)
# Note the batch size B is really the same as B * T in block_clt_module.py,
# but we don't need to know the sequence length T here, so we just use B.
# ──────────────────────────────────────────────────────────────────────────────

class BlockCrossLayerTranscoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_hidden: int,
        k: int = 16,
        auxk: int = 32,
        batch_size: int = 32,
        dead_steps_threshold: int = 10000,
        block_size: int = 6,
    ):
        """
        Block Cross-Layer Transcoder (Block CLT).

        Identical to CrossLayerTranscoder, except that for target layers
        tgt >= block_size (0-indexed), the decoder sum uses only a sliding
        window of the most recent `block_size` source layers instead of all
        previous layers.

        Specifically, for target layer tgt (0-indexed):
            src runs over range(max(0, tgt - block_size + 1), tgt + 1)

        Example with num_layers=12, block_size=6 (1-indexed):
            Layer  1-6  (tgt 0-5):  full accumulation  1..tgt
            Layer  7    (tgt 6):    src layers 2..7
            Layer  8    (tgt 7):    src layers 3..8
            ...
            Layer 12    (tgt 11):   src layers 7..12

        Args:
            num_layers:           Number of transformer layers in the backbone.
            d_model:              Embedding dimension of the backbone.
            d_hidden:             Hidden dimension of the CLT.
            k:                    Top-k value for sparse activation.
            auxk:                 Top-k value for auxiliary loss.
            batch_size:           Used for dead-neuron threshold scaling.
            dead_steps_threshold: Steps before a neuron is considered dead.
            block_size:           Window size / number of "normal" prefix layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.auxk = auxk
        self.batch_size = batch_size
        self.dead_steps_threshold = dead_steps_threshold / batch_size
        self.block_size = block_size
        self.skip_ln = (num_layers != 6)

        # --- Encoders ---
        # If you're training new models, you should initialzie nn.Linear with bias=False no matter how many layers. We add this if loop to preserve backwards compatibility.
        self.encoders = nn.ModuleList([nn.Linear(d_model, d_hidden) for _ in range(num_layers)] if num_layers <= 12 else nn.ModuleList([nn.Linear(d_model, d_hidden, bias=False) for _ in range(num_layers)]))
        self.b_enc = nn.ParameterList([nn.Parameter(torch.zeros(d_hidden)) for _ in range(num_layers)])
        self.b_pre = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(num_layers)])

        # --- Decoders (windowed triangular structure) ---
        # For tgt < block_size:  src in [0, tgt]           (same as full CLT)
        # For tgt >= block_size: src in [tgt-block_size+1, tgt]  (sliding window)
        self.decoders = nn.ParameterDict()
        total_decoders = 0
        for tgt in range(num_layers):
            src_start = max(0, tgt - block_size + 1)
            for src in range(src_start, tgt + 1):
                w_DH = torch.empty(d_hidden, d_model)
                nn.init.kaiming_uniform_(self.encoders[src].weight, a=math.sqrt(5))
                w_DH.data = self.encoders[src].weight.data.clone()
                w_DH.data /= w_DH.data.norm(dim=0)
                self.decoders[f"{src}_{tgt}"] = nn.Parameter(w_DH)
                total_decoders += 1

        # Sanity check: count expected decoders
        expected = 0
        for tgt in range(num_layers):
            expected += min(tgt + 1, block_size)
        assert len(self.decoders) == expected, \
            f"Decoder count mismatch: got {len(self.decoders)}, expected {expected}"

        self.register_buffer("stats_last_nonzero", torch.zeros((num_layers, d_hidden), dtype=torch.long))

    def _src_range(self, tgt: int):
        """Returns the range of valid source layers for a given target layer."""
        src_start = max(0, tgt - self.block_size + 1)
        return range(src_start, tgt + 1)

    def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Applies Top-K activation: keeps top k values, sets others to zero."""
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def LN(self, x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Manual LayerNorm to return mean and std for reconstruction."""
        if self.skip_ln:
            mu = torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)
            std = torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)
            return x, mu, std
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def forward(self, x_stack: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x_stack: Tensor of shape (B, L, H) containing residual stream states.
        Returns:
            recons_stack: The reconstructed updates.
            auxk_stack:   Auxiliary outputs for dead neuron training.
            dead_mask_stack: Boolean mask of dead neurons.
        """
        B, L, H = x_stack.shape

        # 1. Pre-process and Encode all layers
        latents_list_L = []
        pre_acts_list_L = []
        mu_list_L, std_list_L = [], []
        stats_updates_L = []

        for l in range(L):
            x_layer_BH = x_stack[:, l, :]
            x_layernorm_BH, mu, std = self.LN(x_layer_BH)
            mu_list_L.append(mu)
            std_list_L.append(std)
            x_layernorm_BH = x_layernorm_BH - self.b_pre[l]

            pre_acts_BD = self.encoders[l](x_layernorm_BH) + self.b_enc[l]
            pre_acts_list_L.append(pre_acts_BD)

            latents_BD = self.topK_activation(pre_acts_BD, k=self.k)
            latents_list_L.append(latents_BD)

            is_dead_D = (latents_BD == 0).all(dim=0).long()
            stats_updates_L.append(is_dead_D)

        with torch.no_grad():
            stack_is_dead_LD = torch.stack(stats_updates_L)
            self.stats_last_nonzero *= stack_is_dead_LD
            self.stats_last_nonzero += 1

        # 2. Decode (Windowed Cross-Layer)
        recons_list_L = []

        for tgt in range(L):
            recon_accum_BH = torch.zeros(B, H, device=x_stack.device)

            # Only accumulate from the windowed source range
            for src in self._src_range(tgt):
                w_dec_DH = self.decoders[f"{src}_{tgt}"]
                recon_accum_BH = recon_accum_BH + (latents_list_L[src] @ w_dec_DH)

            recon_accum_BH = recon_accum_BH + self.b_pre[tgt]
            recon_accum_BH = recon_accum_BH * std_list_L[tgt] + mu_list_L[tgt]
            recons_list_L.append(recon_accum_BH)

        recons_stack_BLH = torch.stack(recons_list_L, dim=1)

        # 3. AuxK (Dead Neuron Recovery)
        auxk_list_L = []
        dead_mask_stack_LD = self.stats_last_nonzero > self.dead_steps_threshold
        total_dead = dead_mask_stack_LD.sum().item()

        if total_dead > 0:
            for l in range(L):
                num_dead_l = dead_mask_stack_LD[l].sum().item()
                if num_dead_l > 0:
                    k_aux = min(H // 2, num_dead_l)
                    auxk_latents_BD = torch.where(dead_mask_stack_LD[l][None, :], pre_acts_list_L[l], -torch.inf)
                    auxk_acts_BD = self.topK_activation(auxk_latents_BD, k=k_aux)

                    # Decode using self-layer decoder only (always exists on diagonal)
                    w_dec_self_DH = self.decoders[f"{l}_{l}"]
                    aux_out_BH = (auxk_acts_BD @ w_dec_self_DH) + self.b_pre[l]
                    aux_out_BH = aux_out_BH * std_list_L[l] + mu_list_L[l]
                    auxk_list_L.append(aux_out_BH)
                else:
                    auxk_list_L.append(torch.zeros_like(recons_list_L[l]))

            auxk_stack_BLH = torch.stack(auxk_list_L, dim=1)
        else:
            auxk_stack_BLH = None

        return recons_stack_BLH, auxk_stack_BLH, dead_mask_stack_LD

    @torch.no_grad()
    def norm_weights(self):
        """Normalizes decoder weights to unit norm."""
        for param in self.decoders.values():
            param.data /= param.data.norm(dim=0)

    @torch.no_grad()
    def norm_grad(self):
        """Projects gradients to keep weights on the unit sphere."""
        for param in self.decoders.values():
            if param.grad is not None:
                dot_products = torch.sum(param.data * param.grad, dim=0)
                param.grad.sub_(param.data * dot_products.unsqueeze(0))
