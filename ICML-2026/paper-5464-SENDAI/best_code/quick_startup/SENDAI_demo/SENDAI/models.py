"""
Neural network architectures for SENDAI.

Models:
- SHRED: Base LSTM encoder-decoder
- DASHRED: SHRED with latent transform for domain adaptation
- LatentGAN: GAN for latent space alignment
- HF_SHRED_INR: High-frequency pathway with coordinate-based INR
- FullDASHRED: Complete hierarchical model (LF + HF pathways)
"""

import numpy as np
import copy
import torch
import torch.nn as nn


class SHRED(nn.Module):
    """
    SHallow REcurrent Decoder for spatiotemporal reconstruction.
    
    Uses LSTM encoder with Takens' time-delay embedding followed by
    MLP decoder to reconstruct full spatial state from sparse sensors.
    """
    
    def __init__(self, n_sensors, lags, hidden_size, state_dim,
                 num_layers=2, decoder_layers=[256, 256], dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(hidden_size)
        
        layers = []
        prev = hidden_size
        for size in decoder_layers:
            layers.extend([
                nn.Linear(prev, size),
                nn.LayerNorm(size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = size
        layers.append(nn.Linear(prev, state_dim))
        self.decoder = nn.Sequential(*layers)
    
    def encode(self, x):
        """Encode sensor sequence to latent."""
        _, (h_n, _) = self.lstm(x)
        return self.norm(h_n[-1])
    
    def forward(self, x):
        """Forward pass returning (reconstruction, latent)."""
        z = self.encode(x)
        return self.decoder(z), z


class DASHRED(nn.Module):
    """
    Domain-Adapted SHRED with latent transform for sim-to-real gap closure.
    """
    
    def __init__(self, base_shred, freeze_decoder=False):
        super().__init__()
        self.lstm = copy.deepcopy(base_shred.lstm)
        self.norm = copy.deepcopy(base_shred.norm)
        self.decoder = copy.deepcopy(base_shred.decoder)
        self.hidden_size = base_shred.hidden_size
        
        self.transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        for m in self.transform.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False
    
    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.norm(h_n[-1])
    
    def forward(self, x, apply_transform=True):
        z = self.encode(x)
        z_t = z + self.scale * self.transform(z) if apply_transform else z
        return self.decoder(z_t), z, z_t


class LatentGAN(nn.Module):
    """GAN for latent space alignment between simulation and real domains."""
    
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        with torch.no_grad():
            self.generator[-1].weight.mul_(0.1)
            self.generator[-1].bias.zero_()
        
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, z):
        return z + self.generator(z)


class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for coordinate-based networks.
    Maps (x, y) to high-dimensional features via sinusoids.
    """
    
    def __init__(self, input_dim=2, num_frequencies=16, max_frequency=8.0,
                 include_input=True, log_sampling=True):
        super().__init__()
        self.include_input = include_input
        
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0, np.log2(max_frequency), num_frequencies)
        else:
            freq_bands = torch.linspace(1.0, max_frequency, num_frequencies)
        
        self.register_buffer('freq_bands', freq_bands)
        
        self.output_dim = 2 * num_frequencies * input_dim
        if include_input:
            self.output_dim += input_dim
    
    def forward(self, coords):
        scaled = coords.unsqueeze(-1) * self.freq_bands * 2 * np.pi
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = encoded.view(*coords.shape[:-1], -1)
        
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded


class SirenLayer(nn.Module):
    """SIREN layer with sinusoidal activation: y = sin(omega * (Wx + b))"""
    
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class HF_SHRED_INR(nn.Module):
    """
    High-frequency pathway with coordinate-based Implicit Neural Representation.
    
    Produces smooth spatial outputs by querying a learned continuous function
    at all coordinates, conditioned on encoded sensor residuals.
    
    Architecture:
        sensor_residuals -> Encoder -> latent z
        For each (x, y): PE(x,y) + z -> Decoder -> u_hf(x,y)
    """
    
    def __init__(self, n_sensors, H, W, sensor_indices,
                 latent_dim=64,
                 encoder_hidden=[128, 128],
                 decoder_hidden=[256, 256, 128],
                 pe_num_frequencies=16,
                 pe_max_frequency=8.0,
                 pe_include_input=True,
                 activation='relu',
                 omega_0=30.0,
                 dropout=0.1):
        super().__init__()
        
        self.H, self.W = H, W
        self.state_dim = H * W
        self.latent_dim = latent_dim
        
        self.register_buffer('sensor_indices', torch.tensor(sensor_indices, dtype=torch.long))
        
        # Coordinate grid (normalized to [0, 1])
        y_coords = torch.linspace(0, 1, H)
        x_coords = torch.linspace(0, 1, W)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('coord_grid', torch.stack([yy, xx], dim=-1).view(-1, 2))
        
        # Positional encoding
        self.pe = FourierPositionalEncoding(
            input_dim=2,
            num_frequencies=pe_num_frequencies,
            max_frequency=pe_max_frequency,
            include_input=pe_include_input
        )
        
        # Encoder: sensor residuals -> latent
        enc_layers = []
        prev = n_sensors
        for h in encoder_hidden:
            enc_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder: (PE, z) -> value
        dec_input_dim = self.pe.output_dim + latent_dim
        
        if activation == 'siren':
            dec_layers = []
            prev = dec_input_dim
            for i, h in enumerate(decoder_hidden):
                dec_layers.append(SirenLayer(prev, h, omega_0=omega_0, is_first=(i == 0)))
                prev = h
            dec_layers.append(nn.Linear(prev, 1))
            self.decoder = nn.Sequential(*dec_layers)
        else:
            dec_layers = []
            prev = dec_input_dim
            for h in decoder_hidden:
                dec_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h)])
                dec_layers.append(nn.SiLU() if activation == 'swish' else nn.ReLU())
                dec_layers.append(nn.Dropout(dropout))
                prev = h
            dec_layers.append(nn.Linear(prev, 1))
            self.decoder = nn.Sequential(*dec_layers)
        
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        # Small output initialization
        with torch.no_grad():
            for layer in reversed(list(self.decoder.modules())):
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(0.1)
                    if layer.bias is not None:
                        layer.bias.zero_()
                    break
    
    def encode(self, sensor_residual):
        return self.encoder(sensor_residual)
    
    def decode_at_coords(self, z, coords):
        """Decode at coordinates given latent code."""
        batch_size, N = z.shape[0], coords.shape[0]
        
        pe = self.pe(coords)
        pe_exp = pe.unsqueeze(0).expand(batch_size, -1, -1)
        z_exp = z.unsqueeze(1).expand(-1, N, -1)
        
        dec_input = torch.cat([pe_exp, z_exp], dim=-1)
        dec_flat = dec_input.reshape(-1, dec_input.shape[-1])
        values = self.decoder(dec_flat).reshape(batch_size, N)
        
        return values
    
    def forward(self, sensor_residual):
        """Full forward: sensor residuals -> full spatial field."""
        z = self.encode(sensor_residual)
        u_hf = self.decode_at_coords(z, self.coord_grid)
        return self.scale * u_hf
    
    def smoothness_loss(self, u_hf, loss_type='laplacian'):
        """Compute spatial smoothness regularization."""
        u_2d = u_hf.view(-1, self.H, self.W)
        
        if loss_type == 'gradient':
            dx = u_2d[:, :, 1:] - u_2d[:, :, :-1]
            dy = u_2d[:, 1:, :] - u_2d[:, :-1, :]
            return torch.mean(dx**2) + torch.mean(dy**2)
        else:  # laplacian
            lap = (u_2d[:, 2:, 1:-1] + u_2d[:, :-2, 1:-1] +
                   u_2d[:, 1:-1, 2:] + u_2d[:, 1:-1, :-2] -
                   4 * u_2d[:, 1:-1, 1:-1])
            return torch.mean(lap**2)


class HF_SHRED_MLP(nn.Module):
    """Simple MLP-based HF layer (fallback when INR is disabled)."""
    
    def __init__(self, n_sensors, state_dim, hidden_layers=[128, 256, 256], dropout=0.1):
        super().__init__()
        
        layers = []
        prev = n_sensors
        for size in hidden_layers:
            layers.extend([
                nn.Linear(prev, size),
                nn.LayerNorm(size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = size
        layers.append(nn.Linear(prev, state_dim))
        self.mlp = nn.Sequential(*layers)
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, sensor_residual):
        return self.scale * self.mlp(sensor_residual)


class FullDASHRED(nn.Module):
    """
    Complete SENDAI model with hierarchical LF + HF pathways.
    
    Pipeline:
        1. LSTM encode -> latent
        2. DA transform -> aligned latent  
        3. Decode -> LF prediction (u_da)
        4. For each HF layer: compute sensor residual -> INR -> u_hf
        5. Output = u_da + sum(u_hf)
    """
    
    def __init__(self, dashred, n_sensors, state_dim, sensor_indices, H, W,
                 n_hf_layers=2, use_inr=True, inr_config=None):
        super().__init__()
        
        # LF pathway (from DA-SHRED)
        self.lstm = dashred.lstm
        self.norm = dashred.norm
        self.decoder = dashred.decoder
        self.transform = dashred.transform
        self.da_scale = dashred.scale
        self.hidden_size = dashred.hidden_size
        
        self.use_inr = use_inr
        self.n_hf_layers = n_hf_layers
        self.H, self.W = H, W
        
        self.register_buffer('sensor_indices', torch.tensor(sensor_indices, dtype=torch.long))
        
        # Default INR config
        if inr_config is None:
            inr_config = {
                'latent_dim': 64,
                'encoder_hidden': [128, 128],
                'decoder_hidden': [256, 256, 128],
                'pe_num_frequencies': 16,
                'pe_max_frequency': 8.0,
                'pe_include_input': True,
                'activation': 'relu',
                'omega_0': 30.0,
            }
        
        # HF layers
        if use_inr:
            self.hf_layers = nn.ModuleList([
                HF_SHRED_INR(
                    n_sensors=n_sensors, H=H, W=W, sensor_indices=sensor_indices,
                    latent_dim=inr_config.get('latent_dim', 64),
                    encoder_hidden=inr_config.get('encoder_hidden', [128, 128]),
                    decoder_hidden=inr_config.get('decoder_hidden', [256, 256, 128]),
                    pe_num_frequencies=inr_config.get('pe_num_frequencies', 16),
                    pe_max_frequency=inr_config.get('pe_max_frequency', 8.0),
                    pe_include_input=inr_config.get('pe_include_input', True),
                    activation=inr_config.get('activation', 'relu'),
                    omega_0=inr_config.get('omega_0', 30.0),
                )
                for _ in range(n_hf_layers)
            ])
        else:
            self.hf_layers = nn.ModuleList([
                HF_SHRED_MLP(n_sensors, state_dim)
                for _ in range(n_hf_layers)
            ])
        
        # Track discovered frequencies for exclusion
        self.discovered_freqs = [[] for _ in range(n_hf_layers)]
    
    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.norm(h_n[-1])
    
    def forward_da_only(self, x):
        """LF pathway only."""
        z = self.encode(x)
        z_t = z + self.da_scale * self.transform(z)
        return self.decoder(z_t), z, z_t
    
    def forward(self, sensor_history, current_sensors=None, use_hf=True):
        """
        Full forward pass.
        
        Args:
            sensor_history: (batch, lags, n_sensors)
            current_sensors: (batch, n_sensors) in state scale, required if use_hf=True
            use_hf: whether to apply HF correction
        
        Returns:
            u_total, u_da, u_hf_combined, z, z_t
        """
        u_da, z, z_t = self.forward_da_only(sensor_history)
        
        if not use_hf or self.n_hf_layers == 0:
            return u_da, u_da, torch.zeros_like(u_da), z, z_t
        
        if current_sensors is None:
            raise ValueError("current_sensors required when use_hf=True")
        
        u_cumulative = u_da.clone()
        u_hf_combined = torch.zeros_like(u_da)
        
        for hf_layer in self.hf_layers:
            sensors_pred = u_cumulative[:, self.sensor_indices].detach()
            residual = current_sensors - sensors_pred
            u_hf = hf_layer(residual)
            u_hf_combined = u_hf_combined + u_hf
            u_cumulative = u_cumulative + u_hf.detach()
        
        return u_da + u_hf_combined, u_da, u_hf_combined, z, z_t
    
    def set_discovered_freqs(self, layer_idx, freqs):
        """Store discovered frequencies for a layer."""
        if layer_idx < self.n_hf_layers:
            self.discovered_freqs[layer_idx] = freqs
    
    def get_excluded_freqs(self, layer_idx):
        """Get frequencies to exclude (from all previous layers)."""
        excluded = []
        for i in range(layer_idx):
            excluded.extend(self.discovered_freqs[i])
        return excluded
