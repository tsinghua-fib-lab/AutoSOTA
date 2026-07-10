import torch
import torch.nn as nn

from models.KFCore import KFCore
from hflayers.activation import HopfieldCore
from models.EinsteinCore import EinsteinCore


class SingleInstanceClassifier(nn.Module):
    def __init__(self, mode, input_dim, hidden_dim, num_classes, beta=None, num_states=1, num_memories=64):
        super().__init__()
        self.mode = mode
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        # Karcher Flow Models
        if mode == "kf_attention":
            self.core = KFCore(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
        elif mode == "kf_pooling":
            self.core = KFCore(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
            self.static_query = nn.Parameter(torch.randn(1, num_states, hidden_dim) * 0.02)
        elif mode == "kf_layer":
            self.core = KFCore(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
            self.static_key = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)
            self.static_value = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)
            
        # HNIAYN Models
        elif mode == "hf_attention":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1)
        elif mode == "hf_pooling":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1, query_as_static=True)
            self.static_query = nn.Parameter(torch.randn(num_states, 1, hidden_dim) * 0.02)
        elif mode == "hf_layer":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1, key_as_static=True, value_as_static=True)
            self.static_key = nn.Parameter(torch.randn(num_memories, 1, hidden_dim) * 0.02)
            self.static_value = nn.Parameter(torch.randn(num_memories, 1, hidden_dim) * 0.02)

        # Hyperbolic Attention Models
        elif mode == "ein_attention":
            self.core = EinsteinCore(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
        elif mode == "ein_pooling":
            self.core = EinsteinCore(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
            self.static_query = nn.Parameter(torch.randn(1, num_states, hidden_dim) * 0.02)
        elif mode == "ein_layer":
            self.core = EinsteinCore(state_dim=hidden_dim, memory_dim=hidden_dim, hopfield_dim=hidden_dim, out_dim=hidden_dim, beta=beta)
            self.static_key = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)
            self.static_value = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = x.view(B, -1)
        embeds = self.embedder(x)

        hniayn_embeds = embeds.unsqueeze(0) # Sequence-First (seq_len, B, hidden_dim)
        embeds = embeds.unsqueeze(1) # Batch-First (B, seq_len, hidden_dim)

        if self.mode == "kf_attention":
            z = self.core(embeds, embeds, embeds)
            z = z.squeeze(1)
        elif self.mode == "kf_pooling":
            q = self.static_query.expand(B, -1, -1)
            z = self.core(q, embeds, embeds)
            z = z.mean(dim=1)
        elif self.mode == "kf_layer":
            k = self.static_key.expand(B, -1, -1)
            v = self.static_value.expand(B, -1, -1)
            z = self.core(embeds, k, v)
            z = z.mean(dim=1)

        elif self.mode == "hf_attention":
            z, *_ = self.core(hniayn_embeds, hniayn_embeds, hniayn_embeds)
            z = z.squeeze(0)
        elif self.mode == "hf_pooling":
            q = self.static_query.expand(-1, B, -1)
            z, *_ = self.core(q, hniayn_embeds, hniayn_embeds)
            z = z.mean(dim=0)
        elif self.mode == "hf_layer":
            k = self.static_key.expand(-1, B, -1)
            v = self.static_value.expand(-1, B, -1)
            z, *_ = self.core(hniayn_embeds, k, v)
            z = z.mean(dim=0)

        elif self.mode == "ein_attention":
            z = self.core(embeds, embeds, embeds)
            z = z.squeeze(1)
        elif self.mode == "ein_pooling":
            q = self.static_query.expand(B, -1, -1)
            z = self.core(q, embeds, embeds)
            z = z.mean(dim=1)
        elif self.mode == "ein_layer":
            k = self.static_key.expand(B, -1, -1)
            v = self.static_value.expand(B, -1, -1)
            z = self.core(embeds, k, v)
            z = z.mean(dim=1)
            
        return self.classifier(z)


class MILClassifier(nn.Module):
    def __init__(self, mode, input_dim, hidden_dim, num_classes, beta=None, num_states=1, num_memories=64, bag_dropout=0.75):
        super().__init__()
        self.mode = mode
        self.bag_dropout = bag_dropout

        self.embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

        # Karcher Flow Models
        if mode == "kf_attention":
            self.core = KFCore(hidden_dim, hidden_dim, hidden_dim, hidden_dim, beta)
        elif mode == "kf_pooling":
            self.core = KFCore(hidden_dim, hidden_dim, hidden_dim, hidden_dim, beta)
            self.static_query = nn.Parameter(torch.randn(1, num_states, hidden_dim) * 0.02)
        elif mode == "kf_layer":
            self.core = KFCore(hidden_dim, hidden_dim, hidden_dim, hidden_dim, beta)
            self.static_key = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)
            self.static_value = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)

        # Hopfield (HNIAYN) Models
        elif mode == "hf_attention":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1)
        elif mode == "hf_pooling":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1, query_as_static=True)
            self.static_query = nn.Parameter(torch.randn(num_states, 1, hidden_dim) * 0.02)
        elif mode == "hf_layer":
            self.core = HopfieldCore(embed_dim=hidden_dim, num_heads=1,
                                     key_as_static=True, value_as_static=True)
            self.static_key = nn.Parameter(torch.randn(num_memories, 1, hidden_dim) * 0.02)
            self.static_value = nn.Parameter(torch.randn(num_memories, 1, hidden_dim) * 0.02)

        # Hyperbolic Attention Models
        elif mode == "ein_attention":
            self.core = EinsteinCore(hidden_dim, hidden_dim, hidden_dim, hidden_dim, beta)
        elif mode == "ein_pooling":
            self.core = EinsteinCore(hidden_dim, hidden_dim, hidden_dim, hidden_dim, beta)
            self.static_query = nn.Parameter(torch.randn(1, num_states, hidden_dim) * 0.02)
        elif mode == "ein_layer":
            self.core = EinsteinCore(hidden_dim, hidden_dim, hidden_dim, hidden_dim, beta)
            self.static_key = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)
            self.static_value = nn.Parameter(torch.randn(1, num_memories, hidden_dim) * 0.02)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, bag_size, input_dim = x.shape

        x = x.view(B * bag_size, input_dim)
        embeds = self.embedder(x)
        embeds = embeds.view(B, bag_size, -1) # Batch-First (B, bag_size, hidden_dim)

        # Apply bag dropout
        if self.training and self.bag_dropout > 0.0:
            mask = torch.rand(B, bag_size, 1, device=embeds.device) > self.bag_dropout
            embeds = embeds * mask.float()

        hniayn_embeds = embeds.transpose(0, 1) # Sequence-First (bag_size, B, hidden_dim)

        if self.mode == "kf_attention":
            z = self.core(embeds, embeds, embeds).mean(dim=1)
        elif self.mode == "kf_pooling":
            q = self.static_query.expand(B, -1, -1)
            z = self.core(q, embeds, embeds).mean(dim=1)
        elif self.mode == "kf_layer":
            k = self.static_key.expand(B, -1, -1)
            v = self.static_value.expand(B, -1, -1)
            z = self.core(embeds, k, v).mean(dim=1)

        elif self.mode == "hf_attention":
            z, *_ = self.core(hniayn_embeds, hniayn_embeds, hniayn_embeds)
            z = z.mean(dim=0)
        elif self.mode == "hf_pooling":
            q = self.static_query.expand(-1, B, -1)
            z, *_ = self.core(q, hniayn_embeds, hniayn_embeds)
            z = z.mean(dim=0)
        elif self.mode == "hf_layer":
            k = self.static_key.expand(-1, B, -1)
            v = self.static_value.expand(-1, B, -1)
            z, *_ = self.core(hniayn_embeds, k, v)
            z = z.mean(dim=0)
        elif self.mode == "ein_attention":
            z = self.core(embeds, embeds, embeds).mean(dim=1)
        elif self.mode == "ein_pooling":
            q = self.static_query.expand(B, -1, -1)
            z = self.core(q, embeds, embeds).mean(dim=1)
        elif self.mode == "ein_layer":
            k = self.static_key.expand(B, -1, -1)
            v = self.static_value.expand(B, -1, -1)
            z = self.core(embeds, k, v).mean(dim=1)

        return self.classifier(z).squeeze(-1)

