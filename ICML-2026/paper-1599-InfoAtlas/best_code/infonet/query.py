import torch
import torch.nn as nn
from typing import Optional
from .attention_block import CrossAttentionBlock, SelfAttentionBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Query_Gen_transformer(nn.Module):
    def __init__(self, input_dim_x, input_dim_y, dim, hidden_dim=512, dropout=float(0.0)):
        super(Query_Gen_transformer, self).__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y

        self.mlp_x = nn.Sequential(
            nn.Linear(self.input_dim_x, hidden_dim//2),
            #nn.Linear(1, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.mlp_y = nn.Sequential(
            nn.Linear(self.input_dim_y, hidden_dim//2),
            #nn.Linear(1, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.attention1 = MultiHeadAttention_kqv(
            q_dim=self.dim,
            kv_dim=self.hidden_dim,
            heads=8,
            dropout=dropout
        )

        self.attention2 = MultiHeadAttention_kqv(
            q_dim=self.dim,
            kv_dim=self.hidden_dim,
            heads=8,
            dropout=dropout
        )

        self.query = nn.Parameter(torch.randn(1, dim, dim))
        self.norm_q = nn.LayerNorm(self.dim)
        self.norm_k = nn.LayerNorm(self.hidden_dim)
        self.norm_v = nn.LayerNorm(self.hidden_dim)

        self.self_attn_block = nn.ModuleList([
            SelfAttentionBlock(
                q_dim=dim,
                num_heads=8,
                widening_factor=1,
                dropout=dropout
            ) for _ in range(8)
        ])
        self.mlp = MLP(dim, widening_factor=1, dropout=0.0)
        
        
    def forward(self, input):
        
        batch_size = input.shape[0]
        query = self.query.repeat(batch_size, 1, 1)
        X = input[:, :, 0:self.input_dim_x]
        Y = input[:, :, self.input_dim_x:]

        X_long = self.mlp_x(X).unsqueeze(2)
        Y_long = self.mlp_y(Y).unsqueeze(2)

        Q = self.norm_q(query)
        K = self.norm_k(X_long)
        V = self.norm_v(Y_long)
        #print(X.shape, X_long.shape, Q.shape, K.shape, V.shape)
        attention = self.attention1(Q, K, V)
        attention += self.attention2(Q, V, K)
        x = query + attention
        x = x + self.mlp(x)

        for self_attn_layer in self.self_attn_block:
                x = self_attn_layer(x)
        return x
        
class MultiHeadAttention_kqv(nn.Module):
    
    def __init__(
        self,
        q_dim:      int,
        kv_dim:     int,
        qk_out_dim: Optional[int] = None,
        v_out_dim:  Optional[int] = None,
        output_dim: Optional[int] = None,
        heads:      int = 1,
        dropout:    float = 0.0
        ):
        
        super().__init__()

        if qk_out_dim is None:
            qk_out_dim = q_dim
        if v_out_dim is None:
            v_out_dim  = qk_out_dim
        if output_dim is None:
            output_dim = v_out_dim

        self.heads       = heads
        self.qk_head_dim = qk_out_dim // heads
        self.v_head_dim  = v_out_dim // heads

        self.qeury = nn.Linear(q_dim, qk_out_dim)
        self.key   = nn.Linear(kv_dim, qk_out_dim)
        self.value = nn.Linear(kv_dim, v_out_dim)

        self.projection = nn.Linear(v_out_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        x_v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
        ):
        
        batch = x_q.shape[0]
        query_len, key_len, value_len = x_q.shape[1], x_k.shape[1], x_v.shape[1]

        queries = self.qeury(x_q)
        keys    = self.key(x_k)
        values  = self.value(x_v)
        

        # [N, len, embed_size] --> [N, len, heads, head_dim]
        queries = queries.reshape(batch, query_len, self.heads, self.qk_head_dim)
        keys    = keys.reshape(batch, key_len, self.heads, self.qk_head_dim)
        values  = values.reshape(batch, value_len, self.heads, self.v_head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        if attention_mask is not None:
            energy = energy.masked_fill(attention_mask == 0, float("-1e20"))
        attention = torch.softmax(energy / (self.qk_head_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch, query_len, self.heads * self.v_head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)

        out = self.projection(out)
        # (N, query_len, embed_size)
        return out

class MLP(nn.Module):

    def __init__(
        self,
        hidden_dim:int,
        widening_factor:int,
        dropout: float = 0.0
        ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, widening_factor * hidden_dim),
            nn.GELU(),
            nn.Linear(widening_factor * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)
'''
input_dim = 4
dim = 4000
batch_size = 1000
model = Query_Gen_transformer_PE(input_dim, batch_size, dim)

input_data = torch.rand(1, batch_size, input_dim)

output = model(input_data)
print(output.shape)  
'''
