import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies=16, freq_cutoff=8.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_cutoff = freq_cutoff
        self.B = nn.Parameter(torch.linspace(0.5, 2.0, num_frequencies))
        
    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        freqs = self._constrained_frequencies()
        angles = 2 * np.pi * t * freqs.view(1, 1, -1)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    
    def _constrained_frequencies(self):
        return torch.sigmoid(self.B) * self.freq_cutoff
    
    def get_frequencies(self):
        return self._constrained_frequencies()
    
    def frequency_constraint_loss(self):
        freqs = torch.sigmoid(self.B) * self.freq_cutoff
        return F.relu(freqs - self.freq_cutoff).mean()


class ContinuousTimeField(nn.Module):
    def __init__(
        self,
        num_variables,
        num_frequencies=16,
        hidden_dim=64,
        num_layers=3,
        freq_cutoff=8.0
    ):
        super().__init__()
        self.num_variables = num_variables
        self.num_frequencies = num_frequencies
        self.hidden_dim = hidden_dim
        
        self.fourier_features = FourierFeatures(num_frequencies, freq_cutoff)
        
        self.data_encoder = nn.Sequential(
            nn.Linear(num_variables, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_dim),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_variables)
        )
        
    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(x.shape[0], -1)
        if t.dim() == 3:
            t = t.squeeze(-1)
        
        x_low = self._lowpass_filter(x)
        
        gamma_t = self.fourier_features(t)
        h_t = self.time_encoder(gamma_t)
        h_x = self.data_encoder(x_low)
        
        h = torch.cat([h_t, h_x], dim=-1)
        return self.fusion(h)
    
    def _lowpass_filter(self, x):
        B, T, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1)
        n_freqs = x_fft.shape[1]
        cutoff = max(1, n_freqs // 8)
        mask = torch.zeros(n_freqs, device=x.device)
        mask[:cutoff] = 1.0
        taper = torch.linspace(1, 0, min(cutoff, n_freqs - cutoff), device=x.device)
        mask[cutoff:cutoff + len(taper)] = taper
        x_fft_filtered = x_fft * mask.view(1, -1, 1)
        return torch.fft.irfft(x_fft_filtered, n=T, dim=1)
    
    def compute_smoothness_loss(self, x, t):
        output = self.forward(x, t)
        grad = output[:, 1:] - output[:, :-1]
        return (grad ** 2).mean()
    
    def frequency_constraint_loss(self):
        return self.fourier_features.frequency_constraint_loss()
    
    def get_frequencies(self):
        return self.fourier_features.get_frequencies()


class GaborAtom(nn.Module):
    def __init__(self, num_atoms, num_variables, hidden_dim=64):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        
        self.tau = nn.Parameter(torch.linspace(0.05, 0.95, num_atoms))
        self.log_sigma = nn.Parameter(torch.zeros(num_atoms))
        self.omega = nn.Parameter(torch.linspace(10, 100, num_atoms))
        self.phi = nn.Parameter(torch.zeros(num_atoms))
        
        self.encoder = nn.Sequential(
            nn.Linear(num_variables, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        self.amplitude_head = nn.Linear(hidden_dim, num_atoms * num_variables)
        self.gate_head = nn.Linear(hidden_dim, num_atoms)
        
        self.residual_net = nn.Sequential(
            nn.Linear(num_variables, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_variables)
        )
        
        self._last_amplitude = None
        self._last_gate = None
        
    def forward(self, x, t, sigma_addition=0.0):
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(x.shape[0], -1)
        if t.dim() == 3:
            t = t.squeeze(-1)
        B, T = t.shape
        D = x.shape[-1]
        
        h = self.encoder(x)
        h_pooled = h.mean(dim=1)
        amplitude_raw = self.amplitude_head(h_pooled).view(B, self.num_atoms, D)
        gate_logits = self.gate_head(h_pooled)
        gate = torch.sigmoid(gate_logits * 5).unsqueeze(-1)
        amplitude = amplitude_raw * gate
        self._last_amplitude = amplitude.detach()
        self._last_gate = gate.squeeze(-1).detach()
        
        t_expanded = t.unsqueeze(-1)
        tau = self.tau.view(1, 1, -1)
        sigma = F.softplus(self.log_sigma).view(1, 1, -1) + sigma_addition + 0.02
        omega = F.softplus(self.omega).view(1, 1, -1)
        phi = self.phi.view(1, 1, -1)
        
        envelope = torch.exp(-((t_expanded - tau) ** 2) / (2 * sigma ** 2))
        oscillation = torch.cos(omega * (t_expanded - tau) + phi)
        basis = envelope * oscillation
        
        gabor_out = torch.einsum('btk,bkd->btd', basis, amplitude)
        residual_out = self.residual_net(x)
        
        return gabor_out + residual_out
    
    def get_params(self):
        return {
            'tau': self.tau.detach(),
            'sigma': F.softplus(self.log_sigma).detach(),
            'omega': F.softplus(self.omega).detach(),
            'phi': self.phi.detach()
        }


class DiscreteGeometricField(nn.Module):
    def __init__(
        self,
        num_variables,
        num_atoms=16,
        hidden_dim=64,
        sigma_base=0.05,
        sparsity_lambda=0.001
    ):
        super().__init__()
        self.num_variables = num_variables
        self.num_atoms = num_atoms
        self.sigma_base = sigma_base
        self.sparsity_lambda = sparsity_lambda
        
        self.atoms = GaborAtom(num_atoms, num_variables, hidden_dim)
        
    def forward(self, x, t, sigma_addition=0.0):
        return self.atoms(x, t, sigma_addition)
    
    def compute_sparsity_loss(self):
        if self.atoms._last_gate is None:
            return torch.tensor(0.0)
        gate = self.atoms._last_gate
        l1_gate = gate.mean()
        target_sparsity = 0.3
        sparsity_loss = F.relu(l1_gate - target_sparsity)
        return self.sparsity_lambda * (l1_gate + 10 * sparsity_loss)
    
    def get_active_atoms(self, threshold=0.5):
        if self.atoms._last_gate is None:
            return self.num_atoms
        gate = self.atoms._last_gate
        gate_mean = gate.mean(dim=0)
        active = (gate_mean > threshold).sum().item()
        return max(1, active)
    
    def initialize_from_residual(self, residual, t):
        with torch.no_grad():
            B, T, D = residual.shape
            
            fft_result = torch.fft.rfft(residual, dim=1)
            magnitudes = torch.abs(fft_result).mean(dim=(0, 2))
            top_k = min(self.num_atoms, len(magnitudes))
            _, top_indices = torch.topk(magnitudes, top_k)
            freqs = top_indices.float() / T * 2 * np.pi
            self.atoms.omega.data[:top_k] = freqs.clamp(min=1, max=100)
            
            energy = (residual ** 2).mean(dim=2)
            for k in range(self.num_atoms):
                idx = torch.argmax(energy.mean(dim=0))
                self.atoms.tau.data[k] = idx.float() / T
                start = max(0, idx - T // (2 * self.num_atoms))
                end = min(T, idx + T // (2 * self.num_atoms))
                energy[:, start:end] *= 0.1
    
    def get_atom_params(self):
        return self.atoms.get_params()


class ScaleAnnealingScheduler:
    def __init__(self, sigma_base=0.1, total_epochs=1000, warmup_ratio=0.3):
        self.sigma_base = sigma_base
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        
    def get_eta(self, epoch):
        if epoch < self.warmup_epochs:
            return 1.0
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs + 1e-8)
        return 0.5 * (1 + np.cos(np.pi * progress))


class DualTimesField(nn.Module):
    def __init__(
        self,
        num_variables,
        seq_length=512,
        num_frequencies=16,
        hidden_dim=64,
        num_layers=3,
        freq_cutoff=10.0,
        num_atoms=16,
        sigma_base=0.05,
        sparsity_lambda=0.001,
        smoothness_lambda=0.001
    ):
        super().__init__()
        self.num_variables = num_variables
        self.seq_length = seq_length
        self.num_frequencies = num_frequencies
        self.num_atoms = num_atoms
        self.smoothness_lambda = smoothness_lambda
        
        self.ctf = ContinuousTimeField(
            num_variables=num_variables,
            num_frequencies=num_frequencies,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            freq_cutoff=freq_cutoff
        )
        
        self.dgf = DiscreteGeometricField(
            num_variables=num_variables,
            num_atoms=num_atoms,
            hidden_dim=hidden_dim,
            sigma_base=sigma_base,
            sparsity_lambda=sparsity_lambda
        )
        
        self.scale_scheduler = ScaleAnnealingScheduler(
            sigma_base=sigma_base,
            total_epochs=1000,
            warmup_ratio=0.3
        )
        
        self.current_epoch = 0
        
    def forward(self, x, t):
        ctf_output = self.ctf(x, t)
        
        residual = x - ctf_output
        
        eta = self.scale_scheduler.get_eta(self.current_epoch)
        sigma_addition = eta * self.scale_scheduler.sigma_base
        dgf_output = self.dgf(residual, t, sigma_addition)
        
        output = ctf_output + dgf_output
        return output, ctf_output, dgf_output
    
    def compute_loss(self, x, t):
        output, ctf_output, dgf_output = self.forward(x, t)
        
        rec_loss = F.smooth_l1_loss(output, x, beta=0.01)
        
        residual = x - ctf_output
        dgf_loss = F.smooth_l1_loss(dgf_output, residual, beta=0.01)
        
        sparsity_loss = self.dgf.compute_sparsity_loss()
        smoothness_loss = self.ctf.compute_smoothness_loss(x, t)
        freq_constraint_loss = self.ctf.frequency_constraint_loss()
        
        total_loss = rec_loss + 0.3 * dgf_loss + sparsity_loss + self.smoothness_lambda * smoothness_loss
        
        return {
            'total': total_loss,
            'reconstruction': rec_loss,
            'dgf': dgf_loss,
            'sparsity': sparsity_loss,
            'smoothness': smoothness_loss,
            'freq_constraint': freq_constraint_loss
        }
    
    def initialize_atoms(self, x, t):
        with torch.no_grad():
            ctf_output = self.ctf(x, t)
            residual = x - ctf_output
            self.dgf.initialize_from_residual(residual, t)
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_tokens(self):
        ctf_token = {'frequencies': self.ctf.get_frequencies().detach()}
        
        atom_params = self.dgf.get_atom_params()
        dgf_tokens = []
        for k in range(self.num_atoms):
            dgf_tokens.append({
                'tau': atom_params['tau'][k],
                'sigma': atom_params['sigma'][k],
                'omega': atom_params['omega'][k],
                'phi': atom_params['phi'][k]
            })
        
        return ctf_token, dgf_tokens
