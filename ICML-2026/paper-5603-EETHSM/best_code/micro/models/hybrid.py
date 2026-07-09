import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import SimplePyTorchTFLayer, SimpleHandmadeTFLayer
from .ssm import SimpleSSMLayer
from .mlp import SimpleMLPLayer


# Sine Positional Encodings, if desired
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)].to(x.device)

        
class HybridModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Token Embedding
        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)

        # Choice of positional encodings. For these small models, learned seems better
        self.positional_encoding = args.positional_encoding
        if args.positional_encoding == "sine":
            self.pos_encoder = PositionalEncoding(args.embed_dim, args.sequence_len)
        if args.positional_encoding == "learned":
            p = torch.zeros((args.sequence_len, args.embed_dim))
            torch.nn.init.xavier_uniform_(p)
            self.pos_encoder = nn.Parameter(p)
        
        self.layers = []
        for layer in args.layers:
            # Transformer Layers
            if layer == "TF":
                if "do_norm" not in vars(args).keys(): args.do_norm = True
                if args.pytorch_transformer:
                    self.layers.append(SimplePyTorchTFLayer(args.embed_dim, args.num_heads, 
                                                            causal=True))
                else:
                    self.layers.append(SimpleHandmadeTFLayer(args.embed_dim, args.num_heads, 
                                                             causal=True, do_norm=args.do_norm))

            # Transformer (non-causal) Layers
            if layer == "TF-nC":
                if "do_norm" not in vars(args).keys(): args.do_norm = True
                if args.pytorch_transformer:
                    self.layers.append(SimplePyTorchTFLayer(args.embed_dim, args.num_heads, 
                                                            causal=False))
                else:
                    self.layers.append(SimpleHandmadeTFLayer(args.embed_dim, args.num_heads, 
                                                             causal=False, do_norm=args.do_norm))

            # MLP layers (already included in transformer layers)
            if layer == "MLP":
                self.layers.append(SimpleMLPLayer(args.embed_dim, args.embed_dim, args.embed_dim))

            # Mamba layers
            if layer == "SSM":
                if "d_conv" not in vars(args).keys(): args.d_conv = 4 
                if "expand" not in vars(args).keys(): args.expand = 2
                if not args.d_conv: args.d_conv = 4
                if not args.expand: args.expand = 2
                self.layers.append(SimpleSSMLayer(args.embed_dim, args.state_dim, 
                                                  args.d_conv, args.expand))
        
        self.layers = nn.ModuleList(self.layers)
        self.decoder = nn.Linear(args.embed_dim, args.vocab_size)

    
    def forward(self, x, mask):
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        
        if self.positional_encoding == "sine":
            x = x + self.pos_encoder(x) # For the sinusoidal positional encoding class
        if self.positional_encoding == "learned":
            x = x + self.pos_encoder # For the learned positional encodings

        for layer in self.layers:
            # x = x + layer(x, mask)
            x = layer(x, mask)

        return self.decoder(x)