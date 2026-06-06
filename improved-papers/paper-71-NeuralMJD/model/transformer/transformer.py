from copy import deepcopy

import torch
import torch.nn as nn

from ..utils import PositionalEmbedding, mask_adjs, mask_nodes, FlexIdentity
from einops import rearrange


class GraphTopoNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph based on graph topology.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.node_type_encoder = nn.Sequential(PositionalEmbedding(num_channels=hidden_dim),
                                               nn.Linear(hidden_dim, hidden_dim))
        self.in_degree_encoder = nn.Sequential(PositionalEmbedding(num_channels=hidden_dim),
                                               nn.Linear(hidden_dim, hidden_dim))
        self.out_degree_encoder = nn.Sequential(PositionalEmbedding(num_channels=hidden_dim),
                                               nn.Linear(hidden_dim, hidden_dim))


    def forward(self, batched_data):
        node_type, in_degree, out_degree = (
            batched_data["node_type"],              # [B, N, 1]
            batched_data["in_degree"],              # [B, N]
            batched_data["out_degree"],             # [B, N]
        )

        node_feature = self.node_type_encoder(node_type).squeeze(dim=-2)    # [B, N, D]

        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        node_feature = mask_nodes(node_feature, batched_data["node_mask"])  # [B, N, D]

        return node_feature


class GraphTopoAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, num_heads, hidden_dim, n_layers):
        super().__init__()
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.out_dims = num_heads * n_layers  # untie attention bias across layers

        self.spatial_pos_encoder = nn.Sequential(
            PositionalEmbedding(num_channels=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dims),
        )

        self.edge_type_encoder = nn.Sequential(
            PositionalEmbedding(num_channels=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dims),
        )

        self.edge_attr_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dims),
        )

    def forward(self, batched_data):
        spatial_pos = batched_data["spatial_pos"]
        edge_attr = batched_data["edge_attr"]
        adj_matrix = batched_data["adj_matrix"]
        node_mask = batched_data["node_mask"]

        # spatial pos
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)    # [B, N, N, H*D] -> [B, H*D, N, N]

        # edge attribute
        edge_attr_bias = self.edge_attr_encoder(edge_attr).permute(0, 3, 1, 2)          # [B, N, N, H*D] -> [B, H*D, N, N]

        # edge feature
        edge_type_bias = self.edge_type_encoder(adj_matrix).permute(0, 3, 1, 2)         # [B, N, N, H*D] -> [B, H*D, N, N]

        graph_attn_bias = spatial_pos_bias + edge_attr_bias + edge_type_bias

        graph_attn_bias = mask_adjs(graph_attn_bias, node_mask)

        return graph_attn_bias


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        activation_fn: str = "relu",
        pre_layernorm: bool = False,
        time_emb_dim: int = 320,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.nhead = num_attention_heads
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.pre_layernorm = pre_layernorm
        self.time_emb_dim = time_emb_dim

        # Initialize blocks
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim, bias=True)

        # Implementation of Feedforward model
        norm_layer = nn.LayerNorm(embedding_dim, bias=True)
        self.norm1 = deepcopy(norm_layer)
        self.norm2 = deepcopy(norm_layer)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embedding_dim, ffn_embedding_dim, bias=True)
        self.linear2 = nn.Linear(ffn_embedding_dim, embedding_dim, bias=True)

        if activation_fn == "gelu":
            self.activation_fn = nn.GELU()
        elif activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "silu":
            self.activation_fn = nn.SiLU()
        else:
            raise ValueError(f"Activation function {activation_fn} not supported")

    def forward(self, x, attn_bias = None, attn_mask = None, node_mask = None):
        """
        @param x: [B, N, D]
        @param attn_bias: [B, H, N, N] or None
        @param attn_mask: [B, N, N] or None
        @param node_mask: [B, N] or None
        """
        # Init
        B, N, D = x.shape
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, "attn_mask must be bool tensor"
        x = mask_nodes(x, node_mask)

        ### self-attention layer
        residual = x
        if self.pre_layernorm:
            x = self.norm1(x)

        q_vec = self.q_proj(x)        
        k_vec = self.k_proj(x)
        v_vec = self.v_proj(x)

        # reshape for multi-head attention
        q_vec = q_vec.view(B, N, self.nhead, D // self.nhead).transpose(1, 2)  # [B, nhead, N, D//nhead], sequence length is the second-from-last dimension
        k_vec = k_vec.view(B, N, self.nhead, D // self.nhead).transpose(1, 2)  # [B, nhead, N, D//nhead]
        v_vec = v_vec.view(B, N, self.nhead, D // self.nhead).transpose(1, 2)  # [B, nhead, N, D//nhead]
        
        # scaled dot-product attention
        attn_additive = attn_bias if attn_bias is not None else torch.zeros_like(attn_mask).unsqueeze(1).repeat(1, self.nhead, 1, 1).float()
        attn_additive = attn_additive.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))
        attn_additive = attn_additive.masked_fill(~node_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.nhead, -1, N), 0.0)  # for rows full of -inf, set to 0.0

        tgt = torch.nn.functional.scaled_dot_product_attention(q_vec, k_vec, v_vec,
                                                               attn_mask=attn_additive,
                                                               dropout_p=self.dropout if self.training else 0.0)  # [B, nhead, N, D//nhead]
        tgt = tgt.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        tgt = self.o_proj(tgt)

        # output layer
        x = residual + self.dropout1(tgt)               # residual connection
        if not self.pre_layernorm:
            x = self.norm1(x)                           # [B, N, D]

        ### feedforward layer
        if self.pre_layernorm:
            x = self.norm2(x)

        tgt2 = self.linear2(self.dropout2(self.activation_fn(self.linear1(x))))
        x = x + self.dropout3(tgt2)

        if not self.pre_layernorm:
            x = self.norm2(x)

        x = mask_nodes(x, node_mask)
        return x


class ContextEncodingLayer(nn.Module):
    def __init__(self, in_seq_dim, in_seq_length, num_static_features, 
                 embedding_dim, ffn_embedding_dim, num_attention_heads, 
                 dropout, activation_fn, pre_layernorm, time_emb_dim, 
                 light_mode):
        super().__init__()

        # node-independent past dynamic data encoding
        self.mlp_node_past_dyn_data = nn.Sequential(nn.Linear(in_seq_dim * in_seq_length, ffn_embedding_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(ffn_embedding_dim, embedding_dim))


        # node-independent past static data encoding
        self.mlp_node_past_static_data = nn.Sequential(nn.Linear(num_static_features, ffn_embedding_dim),
                                                       nn.ReLU(),
                                                       nn.Linear(ffn_embedding_dim, embedding_dim))
        
        # fusion layer for node-independent dynamic and static features
        self.mlp_node_past_feat_fusion = nn.Sequential(nn.LayerNorm(embedding_dim * 2),
                                                       nn.Linear(embedding_dim * 2, ffn_embedding_dim),
                                                       nn.ReLU(),
                                                       nn.Linear(ffn_embedding_dim, embedding_dim))
        
        # inter-node feature fusion using transformer encoder
        self.light_mode = light_mode
        if not self.light_mode:
            self.tf_node_feat_encoder = GraphTransformerEncoderLayer(embedding_dim=embedding_dim, ffn_embedding_dim=ffn_embedding_dim, 
                                                                     num_attention_heads=num_attention_heads, dropout=dropout, 
                                                                     activation_fn=activation_fn, pre_layernorm=pre_layernorm, 
                                                                     time_emb_dim=None)
        else:
            self.tf_node_feat_encoder = FlexIdentity()

        # fusion context embedding + topology embedding
        concat_in_dim = 2 * embedding_dim
        self.mlp_cat_ctx_emb = nn.Sequential(nn.LayerNorm(concat_in_dim),
                                             nn.Linear(concat_in_dim, ffn_embedding_dim),
                                             nn.ReLU(),
                                             nn.Linear(ffn_embedding_dim, embedding_dim))

    def forward(self, node_past_dyn_data, node_past_static_data, node_topo_emb, attn_mask, node_mask):
        """
        @param node_past_dyn_data: [B, N, *, Past]
        @param node_past_static_data: [B, N, *]
        @param node_topo_emb: [B, N, D]
        @param attn_mask: [B, N, N]
        @param node_mask: [B, N]
        """
        B, N = node_past_dyn_data.shape[:2]

        # node-independent past dynamic data encoding
        node_past_dyn_emb = self.mlp_node_past_dyn_data(node_past_dyn_data.view(B, N, -1))      # [B, N, D]
        node_past_dyn_emb = mask_nodes(node_past_dyn_emb, node_mask)

        # node-independent past static data encoding
        node_past_static_emb = self.mlp_node_past_static_data(node_past_static_data.float())    # [B, N, D]
        node_past_static_emb = mask_nodes(node_past_static_emb, node_mask)

        node_past_emb = torch.cat([node_past_dyn_emb, node_past_static_emb], dim=-1)    # [B, N, 2D]
        node_past_emb = self.mlp_node_past_feat_fusion(node_past_emb)                   # [B, N, D]
        node_past_emb = mask_nodes(node_past_emb, node_mask)

        # inter-node feature fusion using transformer encoder
        if not self.light_mode:
            node_ctx_emb = self.tf_node_feat_encoder(node_past_emb, attn_bias=None, attn_mask=attn_mask, node_mask=node_mask)  # [B, N, D]
        else:
            node_ctx_emb = node_past_emb

        # fusion context embedding + topology embedding
        node_emb = torch.cat([node_ctx_emb, node_topo_emb], dim=-1)  # [B, N, 2D]
        node_emb = self.mlp_cat_ctx_emb(node_emb)  # [B, N, D]
        node_emb = mask_nodes(node_emb, node_mask)

        return node_emb
        

class MJDTransformer(nn.Module):
    def __init__(
        self,
        # input/output dimensions
        in_seq_length,
        in_seq_dim,
        out_seq_length,
        out_seq_dim,
        num_static_features,          
        # transformer parameters
        num_encoder_layers,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        pre_layernorm,
        activation_fn,
        dropout,
        light_mode,
    ) -> None:
        # initialize the model
        super().__init__()

        self.in_seq_length = in_seq_length
        self.in_seq_dim = in_seq_dim
        self.out_seq_length = out_seq_length
        self.out_seq_dim = out_seq_dim
        self.num_static_features = num_static_features

        self.num_encoder_layers = num_encoder_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.pre_layernorm = pre_layernorm
        self.activation_fn = activation_fn

        time_emb_dim = embedding_dim

        # topology-specific graph encoding
        self.graph_topo_node_feat = GraphTopoNodeFeature(hidden_dim=embedding_dim)

        self.graph_topo_attn_bias = GraphTopoAttnBias(num_heads=num_attention_heads, hidden_dim=embedding_dim, n_layers=num_encoder_layers)

        # context encoding layer
        self.context_encoding_layer = ContextEncodingLayer(in_seq_dim, in_seq_length, num_static_features, 
                                                           embedding_dim, ffn_embedding_dim, num_attention_heads, 
                                                           dropout, activation_fn, pre_layernorm, time_emb_dim, 
                                                           light_mode)

        # transformer encoder layers
        self.encoder_layers = nn.ModuleList([GraphTransformerEncoderLayer(
                                            embedding_dim=embedding_dim,
                                            ffn_embedding_dim=ffn_embedding_dim,
                                            num_attention_heads=num_attention_heads,
                                            dropout=dropout,
                                            activation_fn=activation_fn,
                                            pre_layernorm=pre_layernorm,
                                            time_emb_dim=time_emb_dim)
                                            for _ in range(num_encoder_layers)
                                            ])

        # readout layer for node-level prediction
        self.readout_layer = nn.Sequential(
            nn.Linear(embedding_dim, ffn_embedding_dim),
            nn.ReLU(),
            nn.Linear(ffn_embedding_dim, out_seq_dim * out_seq_length)
        )

    def forward(self, batched_data):
        
        """init function"""
        # compute padding mask. This is needed for multi-head attention
        node_mask = batched_data['node_mask']
        dev = node_mask.device
        n_graph, n_node = node_mask.shape

        attn_mask = torch.ones(n_graph, n_node, n_node).to(dev).bool()
        attn_mask = attn_mask.masked_fill(~node_mask[:, None, :], False)
        attn_mask = attn_mask.masked_fill(~node_mask[:, :, None], False)
        batched_data['attn_mask'] = attn_mask

        """get topology-specific features"""
        # init graphormer-introduced structural node features; this is irrelevant to the input context data
        topo_node_emb = self.graph_topo_node_feat(batched_data)      # [B, N, D]

        attn_bias = self.graph_topo_attn_bias(batched_data)         # [B, L * H, N, N]
        attn_bias = rearrange(attn_bias, 'b (l h) m n -> b l h m n', h=self.num_attention_heads)  # [B, L, H, N, N]
        assert attn_bias.size(1) == self.num_encoder_layers

        """init context embedding"""
        node_emb = self.context_encoding_layer(batched_data['node_past_dyn_data'], batched_data['node_past_static_data'], 
                                               topo_node_emb, attn_mask, node_mask)

        """transformer layers"""
        x = node_emb
        for i_layer, layer in enumerate(self.encoder_layers):
            x = layer(x, attn_bias[:, i_layer].contiguous(), attn_mask, node_mask)

        """readout layer"""
        output = self.readout_layer(x)              # [B, N, Future * D]
        output = mask_nodes(output, node_mask)
        output = output.view(n_graph, n_node, self.out_seq_length, -1)

        return output