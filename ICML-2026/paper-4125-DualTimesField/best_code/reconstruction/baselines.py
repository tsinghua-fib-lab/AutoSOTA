import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        hidden_dim=64,
        num_layers=3,
        omega_0=30,
        omega_hidden=30
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        layers = []
        layers.append(SineLayer(1, hidden_dim, is_first=True, omega_0=omega_0))
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_hidden))
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_variables)
        
        with torch.no_grad():
            self.output.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_hidden,
                np.sqrt(6 / hidden_dim) / omega_hidden
            )
    
    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(x.shape[0], -1)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        h = self.net(t)
        output = self.output(h)
        return output
    
    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        return {'total': rec_loss, 'reconstruction': rec_loss}
    

class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, 
                 omega_0=10.0, sigma_0=10.0):
        super().__init__()
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mu = nn.Parameter(torch.zeros(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features) * sigma_0)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def forward(self, x):
        lin = self.linear(x)
        freq = self.omega_0 * lin
        gauss = torch.exp(-0.5 * ((lin - self.mu) / (self.gamma + 1e-8)) ** 2)
        return torch.sin(freq) * gauss


class WIRE(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        hidden_dim=64,
        num_layers=3,
        omega_0=10.0,
        sigma_0=10.0
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        self.first_layer = GaborLayer(
            1, hidden_dim, is_first=True, omega_0=omega_0, sigma_0=sigma_0
        )
        
        self.hidden_layers = nn.ModuleList([
            GaborLayer(hidden_dim, hidden_dim, omega_0=omega_0, sigma_0=sigma_0)
            for _ in range(num_layers - 1)
        ])
        
        self.output = nn.Linear(hidden_dim, num_variables)
    
    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(x.shape[0], -1)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        h = self.first_layer(t)
        for layer in self.hidden_layers:
            h = layer(h)
        
        output = self.output(h)
        return output
    
    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        return {'total': rec_loss, 'reconstruction': rec_loss}
    

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_dim, num_layers):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        
        layers = [nn.Linear(input_size, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.fc = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_dim, theta_size)
    
    def forward(self, x):
        return self.theta(self.fc(x))


class TrendBlock(NBeatsBlock):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, degree=3):
        super().__init__(input_size, degree + 1, hidden_dim, num_layers)
        self.output_size = output_size
        self.degree = degree
        t = torch.linspace(0, 1, output_size)
        powers = torch.stack([t ** i for i in range(degree + 1)], dim=0)
        self.register_buffer('T', powers)
    
    def forward(self, x):
        theta = super().forward(x)
        return torch.einsum('bd,dt->bt', theta, self.T)


class SeasonalityBlock(NBeatsBlock):
    def __init__(self, input_size, output_size, hidden_dim, num_layers, num_harmonics=8):
        super().__init__(input_size, 2 * num_harmonics, hidden_dim, num_layers)
        self.output_size = output_size
        self.num_harmonics = num_harmonics
        t = torch.linspace(0, 2 * np.pi, output_size)
        harmonics = []
        for k in range(1, num_harmonics + 1):
            harmonics.append(torch.cos(k * t))
            harmonics.append(torch.sin(k * t))
        self.register_buffer('S', torch.stack(harmonics, dim=0))
    
    def forward(self, x):
        theta = super().forward(x)
        return torch.einsum('bd,dt->bt', theta, self.S)


class GenericBlock(NBeatsBlock):
    def __init__(self, input_size, output_size, hidden_dim, num_layers):
        super().__init__(input_size, output_size, hidden_dim, num_layers)
        self.output_size = output_size
    
    def forward(self, x):
        return super().forward(x)


class NBeatsBaseline(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        hidden_dim=64,
        num_blocks=4,
        num_layers=3,
        stack_type='generic'
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(num_variables, hidden_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if stack_type == 'interpretable':
                if i < num_blocks // 2:
                    block = TrendBlock(seq_length * hidden_dim // 4, seq_length, hidden_dim, num_layers)
                else:
                    block = SeasonalityBlock(seq_length * hidden_dim // 4, seq_length, hidden_dim, num_layers)
            else:
                block = GenericBlock(seq_length * hidden_dim // 4, seq_length, hidden_dim, num_layers)
            self.blocks.append(block)
        
        self.pool = nn.AdaptiveAvgPool1d(seq_length // 4)
        self.output_proj = nn.Linear(hidden_dim, num_variables)
    
    def forward(self, x, t=None):
        B, T, D = x.shape
        h = self.input_proj(x)
        h_pooled = self.pool(h.transpose(1, 2)).transpose(1, 2)
        h_flat = h_pooled.reshape(B, -1)
        
        forecast = torch.zeros(B, T, device=x.device)
        for block in self.blocks:
            block_out = block(h_flat)
            forecast = forecast + block_out
        
        forecast = forecast.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        output = self.output_proj(forecast)
        return output
    
    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        return {'total': rec_loss, 'reconstruction': rec_loss}
    

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.num_kernels = num_kernels
        kernels = []
        for i in range(num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, padding=i))
        self.kernels = nn.ModuleList(kernels)
    
    def forward(self, x):
        res = []
        for kernel in self.kernels:
            res.append(kernel(x))
        res = torch.stack(res, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    def __init__(self, seq_length, hidden_dim, num_kernels=6, top_k=3):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        
        self.conv = nn.Sequential(
            Inception_Block_V1(hidden_dim, hidden_dim, num_kernels),
            nn.GELU(),
            Inception_Block_V1(hidden_dim, hidden_dim, num_kernels)
        )
    
    def forward(self, x):
        B, T, D = x.shape
        
        x_fft = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_fft).mean(dim=-1)
        amplitude[:, 0] = 0
        
        _, top_indices = torch.topk(amplitude, self.top_k, dim=1)
        top_indices = top_indices.detach()
        
        period_list = []
        for i in range(self.top_k):
            freq = top_indices[:, i].float() + 1
            period = (T / freq).clamp(min=2, max=T)
            period_list.append(period)
        
        res = torch.zeros_like(x)
        for i in range(self.top_k):
            period = int(period_list[i].mean().item())
            if period < 2:
                period = 2
            
            if T % period != 0:
                pad_len = period - (T % period)
                x_pad = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            else:
                x_pad = x
                pad_len = 0
            
            T_pad = x_pad.shape[1]
            x_2d = x_pad.reshape(B, T_pad // period, period, D)
            x_2d = x_2d.permute(0, 3, 1, 2)
            
            x_2d = self.conv(x_2d)
            
            x_2d = x_2d.permute(0, 2, 3, 1)
            x_out = x_2d.reshape(B, T_pad, D)
            
            if pad_len > 0:
                x_out = x_out[:, :T, :]
            
            res = res + x_out
        
        return res / self.top_k


class TimesNetBaseline(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        hidden_dim=64,
        num_layers=2,
        num_kernels=6,
        top_k=3
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(num_variables, hidden_dim)
        
        self.blocks = nn.ModuleList([
            TimesBlock(seq_length, hidden_dim, num_kernels, top_k)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_variables)
    
    def forward(self, x, t=None):
        h = self.input_proj(x)
        
        for block in self.blocks:
            h = h + block(h)
            h = self.layer_norm(h)
        
        output = self.output_proj(h)
        return output
    
    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        return {'total': rec_loss, 'reconstruction': rec_loss}
    

class PatchTSTBaseline(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        patch_len=16,
        stride=8,
        d_model=64,
        n_heads=8,
        num_layers=3,
        d_ff=256,
        dropout=0.1
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.patch_len = max(2, min(patch_len, seq_length))
        self.stride = max(1, min(stride, self.patch_len))

        self.patch_num = max(1, (seq_length - self.patch_len) // self.stride + 1)

        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_num, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * self.patch_num, seq_length)

    def forward(self, x, t=None):
        # x: [B, T, D]
        B, T, D = x.shape
        z = x.permute(0, 2, 1)  # [B, D, T]

        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B, D, patch_num, patch_len]

        tokens = self.patch_embed(patches)
        tokens = tokens + self.pos_embedding.unsqueeze(1)
        tokens = tokens.reshape(B * D, self.patch_num, -1)

        encoded = self.encoder(tokens)
        encoded = self.dropout(encoded)

        flat = encoded.reshape(B * D, -1)
        out = self.head(flat)
        out = out.reshape(B, D, T).permute(0, 2, 1)
        return out

    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        return {'total': rec_loss, 'reconstruction': rec_loss}


class iTransformerBaseline(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        d_model=64,
        n_heads=8,
        num_layers=3,
        d_ff=256,
        dropout=0.1,
        use_norm=True
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.use_norm = use_norm

        self.variate_embedding = nn.Linear(seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projector = nn.Linear(d_model, seq_length)

    def forward(self, x, t=None):
        # x: [B, T, D]
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = (x - means) / std
        else:
            x_norm = x

        z = x_norm.permute(0, 2, 1)  # [B, D, T], variates as tokens
        tokens = self.variate_embedding(z)  # [B, D, d_model]
        encoded = self.encoder(tokens)      # [B, D, d_model]

        out = self.projector(encoded).permute(0, 2, 1)  # [B, T, D]

        if self.use_norm:
            out = out * std + means

        return out

    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        return {'total': rec_loss, 'reconstruction': rec_loss}


class PCABaseline(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        n_components=16
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.n_components = n_components
        self.input_dim = seq_length * num_variables
        
        self.encoder = nn.Linear(self.input_dim, n_components, bias=False)
        self.decoder = nn.Linear(n_components, self.input_dim, bias=False)
        
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, x, t=None):
        B, T, D = x.shape
        x_flat = x.reshape(B, -1)
        z = self.encoder(x_flat)
        x_rec = self.decoder(z)
        return x_rec.reshape(B, T, D)
    
    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        
        W = self.encoder.weight
        ortho_loss = torch.norm(W @ W.T - torch.eye(self.n_components, device=W.device))
        
        return {'total': rec_loss + 0.01 * ortho_loss, 'reconstruction': rec_loss}
    

class FourierBaseline(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length,
        n_components=16
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.n_components = n_components
        
        self.coeffs_real = nn.Parameter(torch.randn(n_components, num_variables) * 0.01)
        self.coeffs_imag = nn.Parameter(torch.randn(n_components, num_variables) * 0.01)
        self.freqs = nn.Parameter(torch.linspace(1, n_components, n_components))
        self.phases = nn.Parameter(torch.zeros(n_components, num_variables))
    
    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(x.shape[0], -1)
        
        B, T = t.shape
        t_expanded = t.unsqueeze(-1)
        freqs = self.freqs.view(1, 1, -1)
        phases = self.phases.unsqueeze(0).unsqueeze(0)
        
        angles = 2 * 3.14159 * freqs * t_expanded
        
        cos_terms = torch.cos(angles)
        sin_terms = torch.sin(angles)
        
        output = torch.einsum('btk,kd->btd', cos_terms, self.coeffs_real)
        output = output + torch.einsum('btk,kd->btd', sin_terms, self.coeffs_imag)
        
        return output
    
    def compute_loss(self, x, t):
        output = self.forward(x, t)
        rec_loss = F.mse_loss(output, x)
        return {'total': rec_loss, 'reconstruction': rec_loss}
    