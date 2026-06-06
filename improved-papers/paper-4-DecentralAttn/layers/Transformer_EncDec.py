import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x= encoder_layer(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.norm1 = nn.LayerNorm(configs.d_model)
        # self.attn = Attention(configs.d_model, configs.n_heads) # # for ablation
        self.attn = CoTAR(configs.d_model, configs.d_model//2)

        self.norm2 = nn.LayerNorm(configs.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(configs.d_model, int(2*configs.d_model)),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(int(2*configs.d_model), configs.d_model),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        
        x_att = self.attn(x)
        x = self.norm1(x + x_att)
        
        x_ln = self.mlp(x)
        x = self.norm2(x + x_ln)
        
        return x
    
    
class CoTAR(nn.Module):
    def __init__(self, d_model, d_core=64):
        super(CoTAR, self).__init__()

        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_core)
        self.lin3 = nn.Linear(d_model + d_core, d_model)
        self.lin4 = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape

        # MLP
        core = F.gelu(self.lin1(x))
        core = self.lin2(core)

        weight = F.softmax(core, dim=1)
        core = torch.sum(core * weight, dim=1, keepdim=True).repeat(1, N, 1)

        # MLP
        core_cat = torch.cat([x, core], -1)
        core_cat = F.gelu(self.lin3(core_cat))
        core_cat = self.lin4(core_cat)
        out = core_cat

        return out

    
class Attention(nn.Module):
    def __init__(self, dim, n_heads, init=0):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = (dim // n_heads) ** -0.5 
        if init:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, n, dim = x.shape
        n_heads = self.n_heads
        head_dim = dim // n_heads

        qkv = self.qkv(x).reshape(b, n, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = F.softmax(attn, dim=-1)  

        out = attn @ v  
        out = out.transpose(1, 2).reshape(b, n, dim)  

        return self.out_proj(out)
