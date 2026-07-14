import torch
from torch import nn
import math

# -----------------------------------------
# Noise level prediction
# -----------------------------------------
class TimeRegressor(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        return self.net(z)  # shape (B,1)

# -----------------------------------------
# Sinusoidal time embedding
# -----------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )  # (half,)
        args = t @ freqs.unsqueeze(0)  # (batch, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (batch, dim)
        return emb
    
# -----------------------------------------
# Residual block and ResidualMLP
# -----------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(self.fc2(h))
        if x.shape[-1] != h.shape[-1]:
            x = x[..., :h.shape[-1]]
        return self.norm(x + h)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, time_embed_dim=128, dropout=0.1, flow=True):
        super().__init__()
        self.flow = flow
        # time embedding only if flow=True
        if self.flow:
            self.time_embed = SinusoidalPosEmb(time_embed_dim)
            adjust_dim = input_dim + time_embed_dim
        else:
            adjust_dim = input_dim

        self.stem = nn.LayerNorm(adjust_dim)
        self.blocks = nn.ModuleList([
            ResBlock(adjust_dim, 2 * input_dim, input_dim, dropout),
            ResBlock(input_dim,  4 * input_dim, input_dim, dropout),
            ResBlock(input_dim,   2 * input_dim, input_dim, dropout),
        ])

    def forward(self, x, t=None):
        # x: (batch, input_dim), t: (batch,1) or None
        if self.flow:
            if t is None:
                raise ValueError("ResidualMLP(flow=True) requires a time tensor 't'")
            te = self.time_embed(t)
            h = torch.cat([x, te], dim=-1)
        else:
            h = x
        h = self.stem(h)
        for block in self.blocks:
            h = block(h)
        return h
    
# -----------------------------------------
# MLPs
# -----------------------------------------

class MLP_Small(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        super(MLP_Small, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = int(1.5 * self.input_dim)
        self.middle_dim = int(2 * self.input_dim)  # Middle layer
        self.dropout_rate = dropout_rate
        self.flow = flow
        
        adjust_dim = input_dim + 1 if flow else input_dim

        self.network = nn.Sequential(
            nn.Linear(adjust_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.hidden_dim1, self.middle_dim),
            nn.BatchNorm1d(self.middle_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.middle_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.hidden_dim1, self.input_dim)
        )
    
    def forward(self, x_input, t=None):
        if self.flow:
            x_input = torch.cat([x_input, t.float()], dim=1)

        return self.network(x_input.float())

class MLP_Medium(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        super(MLP_Medium, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = int(2.0 * input_dim)
        self.hidden_dim2 = int(3.0 * input_dim)
        self.hidden_dim3 = int(3.0 * input_dim)
        self.dropout_rate = dropout_rate
        self.flow = flow

        adjust_dim = input_dim + 1 if flow else input_dim

        # First block
        self.fc1 = nn.Linear(adjust_dim, self.hidden_dim1)
        self.norm1 = nn.LayerNorm(self.hidden_dim1)
        
        # Second block
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.norm2 = nn.LayerNorm(self.hidden_dim2)
        
        # Third block (with optional residual connection from block2 output)
        self.fc3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.norm3 = nn.LayerNorm(self.hidden_dim3)
        
        # Fourth block
        self.fc4 = nn.Linear(self.hidden_dim3, self.hidden_dim2)
        self.norm4 = nn.LayerNorm(self.hidden_dim2)
        
        # Fifth block
        self.fc5 = nn.Linear(self.hidden_dim2, self.hidden_dim1)
        self.norm5 = nn.LayerNorm(self.hidden_dim1)
        
        # Output
        self.fc_out = nn.Linear(self.hidden_dim1, input_dim)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x_input, t=None):
        if self.flow:
            x_input = torch.cat([x_input, t.float()], dim=1)

        # 1st block
        x = self.fc1(x_input)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 2nd block
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 3rd block (residual connection from x before this block is optional)
        residual = x
        x = self.fc3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = x + residual  # residual addition
        x = self.dropout(x)

        # 4th block
        x = self.fc4(x)
        x = self.norm4(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 5th block
        x = self.fc5(x)
        x = self.norm5(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Output
        out = self.fc_out(x)
        return out


class MLP_Large(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        super(MLP_Large, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = int(3.0 * input_dim)
        self.hidden_dim2 = int(4.0 * input_dim)
        self.hidden_dim3 = int(4.0 * input_dim)
        self.hidden_dim4 = int(4.0 * input_dim)
        self.hidden_dim5 = int(3.0 * input_dim)
        self.dropout_rate = dropout_rate
        self.flow = flow

        adjust_dim = input_dim + 1 if flow else input_dim

        self.fc1 = nn.Linear(adjust_dim, self.hidden_dim1)
        self.norm1 = nn.LayerNorm(self.hidden_dim1)

        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.norm2 = nn.LayerNorm(self.hidden_dim2)

        self.fc3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.norm3 = nn.LayerNorm(self.hidden_dim3)

        self.fc4 = nn.Linear(self.hidden_dim3, self.hidden_dim4)
        self.norm4 = nn.LayerNorm(self.hidden_dim4)

        self.fc5 = nn.Linear(self.hidden_dim4, self.hidden_dim3)
        self.norm5 = nn.LayerNorm(self.hidden_dim3)

        self.fc6 = nn.Linear(self.hidden_dim3, self.hidden_dim5)
        self.norm6 = nn.LayerNorm(self.hidden_dim5)
        
        self.fc_out = nn.Linear(self.hidden_dim5, input_dim)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x_input, t=None):
        if self.flow:
            x_input = torch.cat([x_input, t.float()], dim=1)

        # 1
        x = self.fc1(x_input)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 2
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 3 (residual from previous block optional)
        residual = x
        x = self.fc3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = x + residual
        x = self.dropout(x)

        # 4
        x = self.fc4(x)
        x = self.norm4(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 5 (another residual)
        residual = x
        x = self.fc5(x)
        x = self.norm5(x)
        x = self.activation(x)
        x = x + residual
        x = self.dropout(x)

        # 6
        x = self.fc6(x)
        x = self.norm6(x)
        x = self.activation(x)
        x = self.dropout(x)

        # output
        out = self.fc_out(x)
        return out
    
import torch
from torch import nn

##############################################
# Fully-connected U-Net alternative (Small)
##############################################
class UNet_Small(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        """
        This U-Net alternative mirrors MLP_Small.
          - If flow=True, a scalar t is concatenated to the input.
          - The encoder downsamples to a latent (middle) dimension
            and the decoder upsamples while concatenating skip connections.
        """
        super(UNet_Small, self).__init__()
        self.input_dim = input_dim
        self.flow = flow
        in_features = input_dim + 1 if flow else input_dim

        # Choose dimensions similar to MLP_Small:
        # hidden_dim1 = int(1.5 * input_dim) and middle_dim = int(2 * input_dim)
        d1 = int(1.5 * input_dim)   # encoder level 1
        d2 = int(2.0 * input_dim)   # encoder level 2 and bottleneck

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Linear(in_features, d1),
            nn.BatchNorm1d(d1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Linear(d1, d2),
            nn.BatchNorm1d(d2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Linear(d2, d2),
            nn.BatchNorm1d(d2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # --- Decoder ---
        # First up‐block: use skip connection from encoder block 2 (e2)
        self.dec2 = nn.Sequential(
            nn.Linear(d2 + d2, d1),
            nn.BatchNorm1d(d1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Final output layer: combine with skip from encoder block 1 (e1)
        self.dec1 = nn.Sequential(
            nn.Linear(d1 + d1, input_dim)
        )
    
    def forward(self, x, t=None):
        # When flow=True, t should be a (batch, 1) tensor.
        if self.flow:
            x = torch.cat([x, t.float()], dim=1)
        
        # Encoder
        e1 = self.enc1(x)         # shape: (batch, d1)
        e2 = self.enc2(e1)        # shape: (batch, d2)
        
        # Bottleneck
        b = self.bottleneck(e2)   # shape: (batch, d2)
        
        # Decoder: first combine bottleneck with corresponding skip (e2)
        d2 = self.dec2(torch.cat([b, e2], dim=1))  # shape: (batch, d1)
        # Then combine with the first encoder output (e1)
        out = self.dec1(torch.cat([d2, e1], dim=1))  # shape: (batch, input_dim)
        return out

##############################################
# Fully-connected U-Net alternative (Medium)
##############################################
class UNet_Medium(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        """
        This network mirrors MLP_Medium by using three encoding layers.
        We use LayerNorm (as in your MLP_Medium) and include skip connections.
        """
        super(UNet_Medium, self).__init__()
        self.input_dim = input_dim
        self.flow = flow
        in_features = input_dim + 1 if flow else input_dim

        # Dimensions similar to MLP_Medium:
        d1 = int(2.0 * input_dim)  # hidden_dim1
        d2 = int(3.0 * input_dim)  # hidden_dim2
        d3 = int(3.0 * input_dim)  # hidden_dim3

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Linear(in_features, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Linear(d1, d2),
            nn.LayerNorm(d2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.enc3 = nn.Sequential(
            nn.Linear(d2, d3),
            nn.LayerNorm(d3),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Linear(d3, d3),
            nn.LayerNorm(d3),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # --- Decoder ---
        # First decoder block: combine bottleneck and encoder level 3
        self.dec3 = nn.Sequential(
            nn.Linear(d3 + d3, d2),
            nn.LayerNorm(d2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Second decoder block: combine with encoder level 2
        self.dec2 = nn.Sequential(
            nn.Linear(d2 + d2, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Final output: combine with encoder level 1
        self.dec1 = nn.Sequential(
            nn.Linear(d1 + d1, input_dim)
        )
    
    def forward(self, x, t=None):
        if self.flow:
            x = torch.cat([x, t.float()], dim=1)
        
        # Encoder
        e1 = self.enc1(x)   # shape: (batch, d1)
        e2 = self.enc2(e1)  # shape: (batch, d2)
        e3 = self.enc3(e2)  # shape: (batch, d3)
        
        # Bottleneck
        b = self.bottleneck(e3)  # shape: (batch, d3)
        
        # Decoder: first up from bottleneck (skip from e3)
        d3 = self.dec3(torch.cat([b, e3], dim=1))  # shape: (batch, d2)
        # Then combine with encoder layer 2
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # shape: (batch, d1)
        # Final output from skip with encoder layer 1
        out = self.dec1(torch.cat([d2, e1], dim=1))  # shape: (batch, input_dim)
        return out

##############################################
# Fully-connected U-Net alternative (Large)
##############################################
class UNet_Large(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.25, flow=True):
        """
        This network mirrors MLP_Large by using four encoding blocks.
        We use LayerNorm and include multiple skip connections.
        """
        super(UNet_Large, self).__init__()
        self.input_dim = input_dim
        self.flow = flow
        in_features = input_dim + 1 if flow else input_dim

        # Dimensions based on MLP_Large:
        d1 = int(3.0 * input_dim)  # hidden_dim1
        d2 = int(4.0 * input_dim)  # hidden_dim2
        d3 = int(4.0 * input_dim)  # hidden_dim3
        d4 = int(4.0 * input_dim)  # hidden_dim4

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Linear(in_features, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.enc2 = nn.Sequential(
            nn.Linear(d1, d2),
            nn.LayerNorm(d2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.enc3 = nn.Sequential(
            nn.Linear(d2, d3),
            nn.LayerNorm(d3),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.enc4 = nn.Sequential(
            nn.Linear(d3, d4),
            nn.LayerNorm(d4),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Linear(d4, d4),
            nn.LayerNorm(d4),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # --- Decoder ---
        # First decoder block: combine bottleneck with encoder level 4
        self.dec4 = nn.Sequential(
            nn.Linear(d4 + d4, d3),
            nn.LayerNorm(d3),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Second decoder block: combine with encoder level 3
        self.dec3 = nn.Sequential(
            nn.Linear(d3 + d3, d2),
            nn.LayerNorm(d2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Third decoder block: combine with encoder level 2
        self.dec2 = nn.Sequential(
            nn.Linear(d2 + d2, d1),
            nn.LayerNorm(d1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        # Final output: combine with encoder level 1
        self.dec1 = nn.Sequential(
            nn.Linear(d1 + d1, input_dim)
        )
    
    def forward(self, x, t=None):
        if self.flow:
            x = torch.cat([x, t.float()], dim=1)
        
        # Encoder
        e1 = self.enc1(x)   # shape: (batch, d1)
        e2 = self.enc2(e1)  # shape: (batch, d2)
        e3 = self.enc3(e2)  # shape: (batch, d3)
        e4 = self.enc4(e3)  # shape: (batch, d4)
        
        # Bottleneck
        b = self.bottleneck(e4)  # shape: (batch, d4)
        
        # Decoder: upsample and combine skip connections in reverse order
        d4 = self.dec4(torch.cat([b, e4], dim=1))  # shape: (batch, d3)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # shape: (batch, d2)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # shape: (batch, d1)
        out = self.dec1(torch.cat([d2, e1], dim=1))  # shape: (batch, input_dim)
        return out

