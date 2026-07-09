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
# ──────────────────────────────────────────────────────────────────────────────

class PerLayerTranscoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_hidden: int,
        k: int = 16,
        auxk: int = 32,
        batch_size: int = 32,
        dead_steps_threshold: int = 10000,
    ):
        """
        Initializes the Per-Layer Transcoder (PLT).
        Each layer is INDEPENDENT. Layer L predicts Layer L+1.
        """
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.auxk = auxk
        self.batch_size = batch_size
        self.dead_steps_threshold = dead_steps_threshold / batch_size
        self.skip_ln = (num_layers != 6)

        # --- Encoders ---
        # If you're training new replacement models, you should initialzie nn.Linear with bias=False no matter how many layers. We add this if loop to preserve backwards compatibility.
        self.encoders = nn.ModuleList([nn.Linear(d_model, d_hidden) for _ in range(num_layers)] if num_layers <= 12 else nn.ModuleList([nn.Linear(d_model, d_hidden, bias=False) for _ in range(num_layers)]))
        
        # --- Decoders ---
        self.decoders = nn.ParameterList([
            nn.Parameter(torch.empty(d_hidden, d_model)) for _ in range(num_layers)
        ])
        
        self.b_enc = nn.ParameterList([nn.Parameter(torch.zeros(d_hidden)) for _ in range(num_layers)])
        self.b_pre = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(num_layers)])
        
        # Initialize Decoders
        for i in range(num_layers):
            nn.init.kaiming_uniform_(self.encoders[i].weight, a=math.sqrt(5))
            
            self.decoders[i].data = self.encoders[i].weight.data.clone()
            
            self.decoders[i].data /= self.decoders[i].data.norm(dim=0, keepdim=True)

        # Buffer for tracking dead neurons
        self.register_buffer("stats_last_nonzero", torch.zeros((num_layers, d_hidden), dtype=torch.long))

    def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def LN(self, x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # if self.skip_ln:
        #     # Return identity
        #     mu = torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)
        #     std = torch.ones((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)
        #     return x, mu, std
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def forward(self, x_stack: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, L, H = x_stack.shape
        
        recons_list_L = [] # each entry is (B, H)
        auxk_list_L = []
        stats_updates_L = []
        
        for l in range(L):
            # 1. Normalize & Center
            xl_BH, mu, std = self.LN(x_stack[:, l, :])
            xl_BH = xl_BH - self.b_pre[l]
            
            # 2. Encode
            # (B, H) @ (H, D) -> (B, D)
            pre_acts_BD = self.encoders[l](xl_BH) + self.b_enc[l]
            
            # 3. Activate
            latents_BD = self.topK_activation(pre_acts_BD, k=self.k)

            # 4. Decode
            w_dec_DH = self.decoders[l]
            recon_BH = (latents_BD @ w_dec_DH) + self.b_pre[l]
            
            # 5. Denormalize
            recon_BH = recon_BH * std + mu
            recons_list_L.append(recon_BH)

            # --- Stats & AuxK ---
            is_dead_D = (latents_BD == 0).all(dim=0).long() 
            stats_updates_L.append(is_dead_D)

            if self.stats_last_nonzero[l].sum() > self.dead_steps_threshold:
                dead_mask = self.stats_last_nonzero[l] > self.dead_steps_threshold
                num_dead = dead_mask.sum().item()
                if num_dead > 0:
                    k_aux = min(H // 2, num_dead)
                    aux_latents = torch.where(dead_mask[None, :], pre_acts_BD, -torch.inf)
                    aux_acts = self.topK_activation(aux_latents, k=k_aux)

                    aux_out_BH = (aux_acts @ w_dec_DH) + self.b_pre[l]
                    aux_out_BH = aux_out_BH * std + mu
                    auxk_list_L.append(aux_out_BH)
                else:
                    auxk_list_L.append(torch.zeros_like(recon_BH))
            else:
                auxk_list_L.append(torch.zeros_like(recon_BH))

        # Update stats
        with torch.no_grad():
            stack_is_dead_LD = torch.stack(stats_updates_L)
            self.stats_last_nonzero *= stack_is_dead_LD
            self.stats_last_nonzero += 1

        recons_stack_BLH = torch.stack(recons_list_L, dim=1)
        auxk_stack_BLH = torch.stack(auxk_list_L, dim=1) if len(auxk_list_L) > 0 else None
        dead_mask_stack_LD = self.stats_last_nonzero > self.dead_steps_threshold
        return recons_stack_BLH, auxk_stack_BLH, dead_mask_stack_LD

    @torch.no_grad()
    def norm_weights(self):
        """Normalizes decoder weights to unit norm."""
        for i in range(len(self.decoders)):
            # MATCH CLT: dim=0 (Hidden dimension)
            self.decoders[i].data /= self.decoders[i].data.norm(dim=0, keepdim=True)

    @torch.no_grad()
    def norm_grad(self):
        """Projects gradients."""
        for i in range(len(self.decoders)):
            param = self.decoders[i]
            if param.grad is not None:
                dot_products = torch.sum(param.data * param.grad, dim=0, keepdim=True)
                param.grad.sub_(param.data * dot_products)