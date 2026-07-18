from torch import nn
from types import SimpleNamespace
import torch
import math
import torch.nn.functional as F
from .label_encoder import MultiLabelEncoder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(device=t.device)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class SiTBlock(nn.Module):
    """
    A DiT/SiT block with Adaptive Layer Norm (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # adaLN modulation: predicts 6 parameters (shift/scale for norm1, norm2, and gating for attn/mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def _forward_ad_safe_self_attention(self, x):
        """
        Manual self-attention path that avoids aten::_native_multi_head_attention,
        which currently does not support forward-mode AD used by torch.func.jvp.
        """
        batch_size, seq_len, hidden_size = x.shape
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.attn.out_proj(attn_out)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Self-Attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self._forward_ad_safe_self_attention(x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. MLP with modulation
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of SiT: AdaLN -> Linear -> Unpatchify (Reshape)
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ClassConditionalSiT(nn.Module):
    """
    Scalable Interpolant Transformer (SiT) / DiT implementation replacing the UNet.
    """
    def __init__(
        self,
        input_size=32,               # H/W of input image
        sample_size=None,            # Not used, for compatibility
        patch_size=2,
        in_channels=4,               # Latent channels (usually 4 for VAE latents)
        hidden_size=1152,            # DiT-XL size, reduce to 384 or 768 for smaller models
        depth=28,
        num_heads=16,
        num_class_per_label=(10,),   # Tuple of classes per label header
        interaction='sum',           # 'cat' or 'sum'
        learn_sigma=False            # If True, output doubled channels
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 1. Patch Embedding
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # 2. Positional Embedding (Learnable)
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)

        # 3. Conditioning (Time + Label)
        # We need the output of both to match hidden_size so we can sum them
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = MultiLabelEncoder(
            num_class_per_label=num_class_per_label,
            d_latent=hidden_size,
            interaction=interaction
        )

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 5. Final Layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        # The trainer looks for model.config.sample_size and model.config.in_channels
        self.config = SimpleNamespace()
        self.config.sample_size = self.input_size  # Map input_size to sample_size
        self.config.in_channels = self.in_channels
        self.config.input_size = self.input_size
        self.config.patch_size = self.patch_size
        # Add any other config params your trainer might check:
        self.config.out_channels = self.out_channels

    def initialize_weights(self):
        # Initialize transformer-like weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers for "Identity" initialization
        # This allows the model to start training as a standard ViT
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
        # Pos embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Patch embedder
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W) tensor of spatial inputs (images or latents)
        t: (N,) tensor of diffusion timesteps
        y: (N, L) tensor of class labels
        """
        # 1. Create Conditioning Vector c
        # In SiT/DiT, c = t_emb + y_emb
        t_emb = self.t_embedder(t) # (N, D)
        y_emb = self.y_embedder(y) # (N, D)
        
        # Combine conditionings
        c = t_emb + y_emb 
        
        # 2. Patchify and add Positional Embedding
        # (N, C, H, W) -> (N, D, H/p, W/p) -> (N, D, NumPatches) -> (N, NumPatches, D)
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)

        # 4. Final Layer & Unpatchify
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x
