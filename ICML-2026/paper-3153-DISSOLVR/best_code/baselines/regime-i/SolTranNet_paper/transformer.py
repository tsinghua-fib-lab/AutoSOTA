"""
Code based on:
Maziarka et al "Molecule Attention Transformer" -> https://github.com/ardigen/MAT
"""

import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


### Model definition -- the default values are the architecture for SolTranNet
def make_model(d_atom=28, N=8, d_model=8, h=2, dropout=0.1, lambda_attention=0.5,
               N_dense=1, leaky_relu_slope=0.0, aggregation_type='mean', 
               dense_output_nonlinearity='relu',n_output=1,
               scale_norm=False, init_type='uniform', use_adapter=False, n_generator_layers=1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout, lambda_attention)
    ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    model = GraphTransformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, scale_norm, use_adapter), N, scale_norm),
        Embeddings(d_model, d_atom, dropout),
        Generator(d_model, aggregation_type, n_output, n_generator_layers, leaky_relu_slope, dropout, scale_norm))

    return model


class GraphTransformer(nn.Module):
    def __init__(self, encoder, src_embed, generator):
        super(GraphTransformer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator
        
    def forward(self, src, src_mask, adj_matrix, edges_att):
        "Take in and process masked src and target sequences."
        return self.predict(self.encode(src, src_mask, adj_matrix, edges_att), src_mask)
    
    def encode(self, src, src_mask, adj_matrix, edges_att):
        return self.encoder(self.src_embed(src), src_mask, adj_matrix, edges_att)
    
    def predict(self, out, out_mask):
        return self.generator(out, out_mask)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, aggregation_type='mean', n_output=1, n_layers=1, 
                 leaky_relu_slope=0.01, dropout=0.0, scale_norm=False):
        super(Generator, self).__init__()
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            self.proj = []
            for i in range(n_layers-1):
                self.proj.append(nn.Linear(d_model, d_model))
                self.proj.append(nn.LeakyReLU(leaky_relu_slope))
                self.proj.append(ScaleNorm(d_model) if scale_norm else LayerNorm(d_model))
                self.proj.append(nn.Dropout(dropout))
            self.proj.append(nn.Linear(d_model, n_output))
            self.proj = torch.nn.Sequential(*self.proj)
        self.aggregation_type = aggregation_type

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = x * mask
        if self.aggregation_type == 'mean':
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=(1))
            out_avg_pooling = out_sum / mask_sum
        elif self.aggregation_type == 'sum':
            out_sum = out_masked.sum(dim=1)
            out_avg_pooling = out_sum
        elif self.aggregation_type == 'dummy_node':
            out_avg_pooling = out_masked[:,0]
        projected = self.proj(out_avg_pooling)
        return projected
    
    
class PositionGenerator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model):
        super(PositionGenerator, self).__init__()
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 3)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        out_masked = self.norm(x) * mask
        projected = self.proj(out_masked)
        return projected
    

### Encoder

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, scale_norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = ScaleNorm(layer.size) if scale_norm else LayerNorm(layer.size)
        
    def forward(self, x, mask, adj_matrix, edges_att):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, edges_att)
        return self.norm(x)

    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class ScaleNorm(nn.Module):
    """ScaleNorm"""
    "All g’s in SCALE NORM are initialized to sqrt(d)"
    def __init__(self, scale, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps
        
    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm

    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, scale_norm, use_adapter):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.use_adapter = use_adapter
        self.adapter = Adapter(size, 8) if use_adapter else None

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.use_adapter:
            return x + self.dropout(self.adapter(sublayer(self.norm(x))))
        return x + self.dropout(sublayer(self.norm(x)))

    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm, use_adapter):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, scale_norm, use_adapter), 2)
        self.size = size

    def forward(self, x, mask, adj_matrix, edges_att):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, edges_att, mask))
        return self.sublayer[1](x, self.feed_forward)

    
### Attention
def attention(query, key, value, adj_matrix, edges_att,
              mask=None, dropout=None, lambdas=(0.5, 0.5),
              eps=1e-6, inf=1e12):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0, -inf)
    p_attn = F.softmax(scores, dim = -1)

    # Prepare adjacency matrix
    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix
    
    lambda_attention, lambda_adjacency = lambdas
    p_weighted = lambda_attention * p_attn + lambda_adjacency * p_adj
    
    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_features = torch.matmul(p_weighted, value)     
    return atoms_features, p_weighted, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, lambda_attention=0.5):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        lambda_adjacency = 1. - lambda_attention
        self.lambdas = (lambda_attention, lambda_adjacency)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, adj_matrix, edges_att, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn, self.self_attn = attention(query, key, value, adj_matrix, edges_att,
                                                 mask=mask, dropout=self.dropout, lambdas=self.lambdas)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


### Conv 1x1 aka Positionwise feed forward

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == 'relu':
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == 'tanh':
            self.tanh = torch.nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == 'none':
            self.dense_output_nonlinearity = lambda x: x
            

    def forward(self, x):
        if self.N_dense == 0:
            return x
        
        for i in range(len(self.linears)-1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
            
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))

    
## Embeddings

class Embeddings(nn.Module):
    def __init__(self, d_model, d_atom, dropout):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))
