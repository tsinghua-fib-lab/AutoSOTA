"""
GAT-FM Model: Graph Attention Network with Flow Matching for Protein Prediction.

?adaLN-Zero  GAT-FM ?DiTiffusion Transformer?
?- TimestepEmbedder: ?- DatasetEmbedder: ?
- RNAProjector: ?RNA ?- GATBlock: ?adaLN-Zero  GAT ?- DiTBlock: ?adaLN-Zero  DiT ?- GATFM: ATBlock?- DITFM: iTBlock?"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
try:
    from timm.models.vision_transformer import Attention, Mlp
except ImportError:
    raise ImportError("timm is required for DITFM model. Install it with: pip install timm")


# =============================================================================
# Utility Functions
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    ?    x: ?(B, N, D) ?(B, D)
    shift: 
    scale: 
    : x * (1 + scale) + shift
    """
    if x.dim() == 3:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale) + shift


# =============================================================================
# Embedding Layers
# =============================================================================

class TimestepEmbedder(nn.Module):
    """
    ?    ?DiT:  -> MLP(Linear + SiLU + Linear)
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        hidden_size: 
        frequency_embedding_size: ?        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        ?        t: (N,) ,  batch ?index
        dim: 
        max_period: 
        : (N, D) 
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,)  [0, 1]
        : (B, hidden_size) ?        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DatasetEmbedder(nn.Module):
    """
    ?Panel ID ?     batch ?    """
    
    def __init__(self, num_datasets: int, hidden_size: int, dropout_prob: float = 0.0):
        """
        num_datasets: ?        hidden_size: 
        dropout_prob: CFG?        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_datasets + use_cfg_embedding, hidden_size)
        self.num_datasets = num_datasets
        self.dropout_prob = dropout_prob

    def token_drop(self, dataset_ids: torch.Tensor, force_drop_ids: torch.Tensor = None) -> torch.Tensor:
        """
        D classifier-free guidance?        """
        if force_drop_ids is None:
            drop_ids = torch.rand(dataset_ids.shape[0], device=dataset_ids.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        dataset_ids = torch.where(drop_ids, self.num_datasets, dataset_ids)
        return dataset_ids

    def forward(self, dataset_ids: torch.Tensor, train: bool = True, force_drop_ids: torch.Tensor = None) -> torch.Tensor:
        """
        dataset_ids: (B,) ?        train: 
        force_drop_ids: IDFG?        : (B, hidden_size) 
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            dataset_ids = self.token_drop(dataset_ids, force_drop_ids)
        embeddings = self.embedding_table(dataset_ids)
        return embeddings


class RNAProjector(nn.Module):
    """
    RNA? RNA(cGPT)?    """
    
    def __init__(self, input_size: int = 512, hidden_size: int = 256):
        """
        input_size: RNA (?12)
        hidden_size: ?        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, rna_embed: torch.Tensor) -> torch.Tensor:
        """
        rna_embed: (B, 512) RNA
        : (B, hidden_size) RNA
        """
        return self.proj(rna_embed)


# =============================================================================
# GAT Block with adaLN-Zero
# =============================================================================

class GATBlock(nn.Module):
    """
    GAT ?adaLN-Zero ?     GATv2Conv  DiT (modulate)?     c = time_embed + rna_embed + dataset_embed?    
    ?    x -> LayerNorm -> modulate -> GATv2Conv -> gate -> 
    x -> LayerNorm -> modulate -> MLP -> gate -> 
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        edge_dim: int = None,
    ):
        """Initialize a GAT block with adaLN-Zero conditioning."""
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.gat = GATv2Conv(
            in_channels=hidden_size,
            out_channels=hidden_size // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=False,
        )
        
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
        
        # adaLN-Zero6?        # (shift_gat, scale_gat, gate_gat, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x:  (num_nodes, hidden_size)
        c:  (num_nodes, hidden_size)batch
        edge_index: ?2, num_edges)
        edge_attr: 
        batch: 
        : (num_nodes, hidden_size) ?        """
        # modulation
        shift_gat, scale_gat, gate_gat, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # GATadaLN-Zero?        x_normed = self.norm1(x)
        x_modulated = modulate(x_normed, shift_gat, scale_gat)
        x_gat = self.gat(x_modulated, edge_index, edge_attr=edge_attr)
        x = x + gate_gat * x_gat
        
        # MLPadaLN-Zero?        x_normed = self.norm2(x)
        x_modulated = modulate(x_normed, shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_modulated)
        x = x + gate_mlp * x_mlp
        
        return x


# =============================================================================
# DiT Block with adaLN-Zero
# =============================================================================

class DiTBlock(nn.Module):
    """
    DiT ?adaLN-Zero ?    ttentionGATv2Conv?     c = time_embed + rna_embed + dataset_embed?    
    ?    x -> LayerNorm -> modulate -> Attention -> gate -> 
    x -> LayerNorm -> modulate -> MLP -> gate -> 
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **block_kwargs,
    ):
        """Initialize a DiT block with adaLN-Zero conditioning."""
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=dropout,
        )
        
        # adaLN-Zero6?        # (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x:  (B, N, hidden_size)
        c:  (B, hidden_size)
        edge_index: DiTBlock?        edge_attr: DiTBlock?        batch: iTBlock?        : (B, N, hidden_size) ?        """
        # 
        if x.dim() != 3:
            raise ValueError(f"DiTBlockx?B, N, D){x.shape}")
        if c.dim() != 2:
            raise ValueError(f"DiTBlockc?B, D){c.shape}")
        
        B, N, _ = x.shape
        
        # modulationatch
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # daLN-Zero?        # modulatensqueeze?B, hidden_size)hiftcale
        x_normed = self.norm1(x)
        x_modulated = modulate(x_normed, shift_attn, scale_attn)
        x_attn = self.attn(x_modulated)
        x = x + gate_attn.unsqueeze(1) * x_attn
        
        # MLPadaLN-Zero?        x_normed = self.norm2(x)
        x_modulated = modulate(x_normed, shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_modulated)
        x = x + gate_mlp.unsqueeze(1) * x_mlp
        
        return x


class FinalLayer(nn.Module):
    """
    GAT-FM?    adaLN??    """
    
    def __init__(self, hidden_size: int, out_channels: int = 1):
        """
        hidden_size: 
        out_channels: 1
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x:  (num_nodes, hidden_size)
        c:  (num_nodes, hidden_size)
        : (num_nodes, out_channels)out_channels=1?num_nodes,)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x


# =============================================================================
# Main GAT-FM Model
# =============================================================================

class GATFM(nn.Module):
    """
    Flow Matching?    ?v(x_t, t)?x_t ?    ?      1. _t->protein_proj, t->time_embed, rna->rna_proj, dataset_id->dataset_embed
      2. : c = time_embed + rna_embed + dataset_embed
      3. GATBlockadaLN-Zero
      4. 
    """
    
    def __init__(
        self,
        protein_dim: int,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_datasets: int = 1,
        rna_embed_dim: int = 512,
        dataset_dropout_prob: float = 0.0,
        dropout: float = 0.0,
    ):
        """
        protein_dim: (P_union)
        hidden_size: ?        depth: GAT?        num_heads: ?        mlp_ratio: MLP?        num_datasets: ataset
        rna_embed_dim: RNA
        dataset_dropout_prob: ropoutFG
        dropout: dropout
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        
        self.protein_proj = nn.Linear(1, hidden_size, bias=True)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.rna_proj = RNAProjector(rna_embed_dim, hidden_size)
        self.dataset_embedder = DatasetEmbedder(num_datasets, hidden_size, dataset_dropout_prob)
        
        # GAT
        self.blocks = nn.ModuleList([
            GATBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # 1
        self.final_layer = FinalLayer(hidden_size, out_channels=1)
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights following DiT-style defaults."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # 
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
        cond_rna: torch.Tensor,
        cond_dataset: torch.Tensor,
        mask: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        GAT-FM 
        x_t:  (B, P)
        t: ?(B,)
        edge_index: ?2, E)?nodes 0~P-1)
        cond_rna: RNA(B, 512)
        cond_dataset: ?(B,) LongTensor
        mask: mask (B, P) (1)
        edge_attr: ?(E, edge_dim) ?
        : ?(B, P)
        """
        B, P = x_t.shape[0], x_t.shape[1]
        device = x_t.device
        
        # (B, P)->(B*P, 1)(B*P, hidden_size)
        # 
        x_flat = x_t.view(-1, 1)  # (B*P, 1)
        x = self.protein_proj(x_flat)  # (B*P, hidden_size)
        
        # 
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        rna_emb = self.rna_proj(cond_rna)  # (B, hidden_size)
        dataset_emb = self.dataset_embedder(cond_dataset, self.training)  # (B, hidden_size)
        
        # 
        c_batch = t_emb + rna_emb + dataset_emb  # (B, hidden_size)
        
        # ?(B, hidden_size)->(B*P, hidden_size)
        c = c_batch.unsqueeze(1).expand(B, P, self.hidden_size).contiguous().view(B * P, self.hidden_size)
        
        # atchbase graphndex
        # base edge_index0~P-10~B*P-1
        batched_edge_index = []
        for b in range(B):
            offset = b * P
            batched_edges = edge_index + offset
            batched_edge_index.append(batched_edges)
        
        if len(batched_edge_index) > 0:
            batched_edge_index = torch.cat(batched_edge_index, dim=1).to(device)
        else:
            batched_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        if edge_attr is not None:
            batched_edge_attr = edge_attr.repeat(B, 1)  # (B*E, edge_dim)
        else:
            batched_edge_attr = None
        
        # GAT
        for block in self.blocks:
            x = block(x, c, batched_edge_index, batched_edge_attr)
        
        # ?B*P, hidden_size)->(B*P,)
        v_flat = self.final_layer(x, c)  # (B*P,)
        
        # ?(B, P)
        v = v_flat.view(B, P)
        
        return v

    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
        cond_rna: torch.Tensor,
        cond_dataset: torch.Tensor,
        cfg_scale: float = 2.0,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        FGclassifier-free guidance?        cfg_scale:  (1.0CFG)
        : 
        """
        # 
        v_cond = self.forward(x_t, t, edge_index, cond_rna, cond_dataset, mask)
        print(f"CFG scale: {cfg_scale}")
        if cfg_scale == 1.0:
            return v_cond

        if self.dataset_embedder.dropout_prob > 0:
            cond_dataset_uncond = torch.full_like(cond_dataset, self.dataset_embedder.num_datasets)
        else:
            import warnings
            warnings.warn("CFG requested but dataset_dropout_prob=0. Returning conditional prediction.")
            return v_cond
        v_uncond = self.forward(x_t, t, edge_index, cond_rna, cond_dataset_uncond, mask)
        
        # CFG
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        return v


# =============================================================================
# DITFM Model (DiT-based Flow Matching)
# =============================================================================

class DITFM(nn.Module):
    """
    Flow MatchingiT?    DiTBlockGATBlock?    ?v(x_t, t)?x_t ?    ?      1. _t->protein_proj, t->time_embed, rna->rna_proj, dataset_id->dataset_embed
      2. : c = time_embed + rna_embed + dataset_embed
      3. DiTBlockadaLN-Zero
      4. 
    """
    
    def __init__(
        self,
        protein_dim: int,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_datasets: int = 1,
        rna_embed_dim: int = 512,
        dataset_dropout_prob: float = 0.0,
        dropout: float = 0.0,
    ):
        """
        protein_dim: (P_union)
        hidden_size: ?        depth: DiT?        num_heads: ?        mlp_ratio: MLP?        num_datasets: ataset
        rna_embed_dim: RNA
        dataset_dropout_prob: ropoutFG
        dropout: dropout
        """
        super().__init__()
        
        self.protein_dim = protein_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        
        self.protein_proj = nn.Linear(1, hidden_size, bias=True)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        if rna_embed_dim == hidden_size:
            self.rna_proj = nn.Identity()
        else:
            self.rna_proj = RNAProjector(rna_embed_dim, hidden_size)
        self.dataset_embedder = DatasetEmbedder(num_datasets, hidden_size, dataset_dropout_prob)
        
        # DiT
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # ATBlock
        self.gat_block = GATBlock(hidden_size, num_heads, mlp_ratio, dropout)

        # 1
        self.final_layer = FinalLayer(hidden_size, out_channels=1)
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights following DiT-style defaults."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        nn.init.normal_(self.dataset_embedder.embedding_table.weight, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # GATBlockdaLN
        nn.init.constant_(self.gat_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.gat_block.adaLN_modulation[-1].bias, 0)
        
        # 
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
        cond_rna: torch.Tensor,
        cond_dataset: torch.Tensor,
        mask: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass for DITFM with optional classifier-free guidance.

        Args:
            x_t: Noisy protein expression of shape (B, P).
            t: Diffusion time steps of shape (B,).
            edge_index: Base graph edges.
            cond_rna: RNA embeddings of shape (B, 512).
            cond_dataset: Dataset ids of shape (B,).
            mask: Optional protein observation mask of shape (B, P).
            edge_attr: Optional edge features.
            cfg_scale: Guidance scale. 1.0 disables CFG.

        Returns:
            Predicted velocity field of shape (B, P).
        """
        B, P = x_t.shape[0], x_t.shape[1]
        device = x_t.device

        def _compute(cond_dataset_local: torch.Tensor) -> torch.Tensor:
            # (B, P)->(B, P, 1)->(B, P, hidden_size)
            x = x_t.unsqueeze(-1)  # (B, P, 1)
            x = self.protein_proj(x)  # (B, P, hidden_size)

            # 
            t_emb = self.t_embedder(t)  # (B, hidden_size)
            rna_emb = self.rna_proj(cond_rna)  # (B, hidden_size)
            dataset_emb = self.dataset_embedder(cond_dataset_local, self.training)  # (B, hidden_size)

            # 
            c_batch = t_emb + rna_emb + dataset_emb  # (B, hidden_size)

            # DiT
            for block in self.blocks:
                x = block(x, c_batch)
            
            # ATBlock
            # ?B, P, hidden_size)?B*P, hidden_size)GATBlock
            x_flat = x.reshape(B * P, self.hidden_size)  # (B*P, hidden_size)
            
            # ?(B, hidden_size)->(B*P, hidden_size)
            c_flat = c_batch.unsqueeze(1).expand(B, P, self.hidden_size).contiguous().view(B * P, self.hidden_size)
            
            # atchbase graphndex
            batched_edge_index = []
            for b in range(B):
                offset = b * P
                batched_edges = edge_index + offset
                batched_edge_index.append(batched_edges)
            
            if len(batched_edge_index) > 0:
                batched_edge_index = torch.cat(batched_edge_index, dim=1).to(device)
            else:
                batched_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            
            if edge_attr is not None:
                batched_edge_attr = edge_attr.repeat(B, 1)  # (B*E, edge_dim)
            else:
                batched_edge_attr = None
            
            # GATBlock
            x_flat = self.gat_block(x_flat, c_flat, batched_edge_index, batched_edge_attr)
            
            # ?B, P, hidden_size)
            x = x_flat.reshape(B, P, self.hidden_size)

            # ?B, P, hidden_size)->(B, P)
            c_tokens = c_batch.unsqueeze(1).expand(B, P, self.hidden_size)  # (B, P, hidden_size)
            v_flat = self.final_layer(x.reshape(B * P, self.hidden_size), c_tokens.reshape(B * P, self.hidden_size))
            v = v_flat.reshape(B, P)  # (B, P)
            return v

        # 
        v_cond = _compute(cond_dataset)

        if cfg_scale == 1.0:
            return v_cond

        if self.dataset_embedder.dropout_prob == 0:
            import warnings
            warnings.warn(
                f"CFG requested (cfg_scale={cfg_scale}) but dataset_dropout_prob=0. "
                "Returning conditional prediction. Set dataset_dropout_prob > 0 to enable CFG."
            )
            return v_cond
        
        # num_datasetsnull token
        cond_dataset_uncond = torch.full_like(cond_dataset, self.dataset_embedder.num_datasets)
        v_uncond = _compute(cond_dataset_uncond)

        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        return v
