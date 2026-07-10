import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.ffn_layers import SwiGLU
from prime.IdentityDropout import IdentityDropout

class FullFrequencyLeadLagFilters(nn.Module):
    """
    Preserves ALL frequency-specific lead-lag information without any averaging
    """
    
    def __init__(self, configs, bands_to_keep):
        super().__init__()
        
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.learnable_diagonal = configs.learnable_diagonal
        self.bands_to_keep = bands_to_keep
        self.seq_len = configs.seq_len
        self.idrop = configs.idrop if hasattr(configs, 'idrop') else 0.0
        
        # Network that processes ALL frequency bins directly
        # Input: (phase_diff, log_magnitude) for EACH frequency bin
        # Output: d_model features for each (i,j) pair
        input_dim = bands_to_keep * 2  # 2 features per frequency bin
        
        self.frequency_filter_network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
    
    def forward(self, X_fft):
        """
        Transform frequency-specific lead-lag info directly to filters
        
        Args:
            X_fft: (B, N, freq_bins) - complex FFT coefficients
            
        Returns:
            filters: (B, N, N, d_model) - frequency-aware lead-lag filters
        """
        window = torch.hann_window(self.seq_len, device=X_fft.device, dtype=X_fft.dtype)
        X_fft = X_fft * window.unsqueeze(0).unsqueeze(0)
        X_fft = torch.fft.rfft(X_fft, norm='ortho', dim=-1)
        X_fft = X_fft[:, :, 1:self.bands_to_keep+1]
        
        B, N, freq_bins = X_fft.shape
        
        # Handle frequency bin dimension
        if freq_bins > self.bands_to_keep:
            # Sample key frequencies
            freq_indices = torch.linspace(0, freq_bins-1, self.freq_bins, 
                                        device=X_fft.device, dtype=torch.long)
            X_fft = X_fft[:, :, freq_indices]
            freq_bins = self.bands_to_keep
        
        # Compute cross-spectral info for all pairs and all frequencies
        X_i = X_fft.unsqueeze(2)  # (B, N, 1, freq_bins)
        X_j = X_fft.unsqueeze(1)  # (B, 1, N, freq_bins)
        cross_spec = X_i * torch.conj(X_j)  # (B, N, N, freq_bins)
        
        # Extract phase and magnitude for ALL frequencies
        phase_all_freqs = torch.angle(cross_spec)  # (B, N, N, freq_bins)
        mag_all_freqs = torch.abs(cross_spec)      # (B, N, N, freq_bins)
        
        # Log transform magnitudes
        # log_mag_all_freqs = torch.log(torch.clamp(mag_all_freqs, 1e-10, 1e10))
        log_mag_all_freqs = torch.log1p(mag_all_freqs)
        
        # Concatenate phase and magnitude for all frequencies
        # This preserves ALL frequency-specific information
        freq_features = torch.cat([
            phase_all_freqs,     # (B, N, N, freq_bins)
            log_mag_all_freqs    # (B, N, N, freq_bins)
        ], dim=-1)  # (B, N, N, 2 * freq_bins)
        
        # filters = freq_features.mean(dim=0, keepdim=True)
        filters = self.frequency_filter_network(freq_features)  # (B, N, N, d_model)
        
        if not self.learnable_diagonal:
            # Handle diagonal elements (self-relationships)
            diagonal_mask = torch.eye(N, device=X_fft.device, dtype=torch.bool)
            diagonal_mask = diagonal_mask.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
            filters = filters.masked_fill(diagonal_mask.expand_as(filters), 1.0)
        
        # diagonal_values = self._compute_autocorr_diagonal(X_fft)
        # diagonal_indices = torch.arange(N, device=X_fft.device)
        # filters[:, diagonal_indices, diagonal_indices, :] = diagonal_values
        
        return IdentityDropout(p=self.idrop)(filters)
        # return filters
    
    def _compute_autocorr_diagonal(self, X_fft):
        """
        Compute diagonal values based on signal predictability and strength
        """
        B, N, freq_bins = X_fft.shape
        
        # 1. Autocorrelation (Predictability) - Most important for self-attention
        autocorr_lag1 = torch.real(torch.sum(X_fft[:, :, :-1] * torch.conj(X_fft[:, :, 1:]), dim=-1))
        signal_energy = torch.sum(torch.abs(X_fft) ** 2, dim=-1) + 1e-10
        predictability = torch.clamp(autocorr_lag1 / signal_energy, -1, 1)  # (B, N)
        
        # 2. Signal Strength (Normalized)
        strength = torch.log1p(signal_energy)  # (B, N)
        strength = strength / (torch.max(strength, dim=1, keepdim=True)[0] + 1e-10)  # Normalize
        
        # Combine: predictable + strong signals get higher self-attention weights
        robustness_score = 0.7 * torch.clamp(predictability, 0, 1) + 0.3 * strength  # (B, N)
        
        # Expand to d_model and scale
        diagonal_values = robustness_score.unsqueeze(-1).expand(-1, -1, self.d_model)
        diagonal_values = 0.7 + 0.6 * diagonal_values  # Range [0.7, 1.3]
        
        return diagonal_values

class PrimeFilters(nn.Module):
    def __init__(self, configs, bands_to_keep):
        super().__init__()
        
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.idrop = configs.idrop
        self.filter_ablation = configs.filter_ablation
        
        total_features = 15
        if self.filter_ablation == 1:
            total_features = 15
        elif self.filter_ablation == 2:
            total_features = 12
        elif self.filter_ablation == 3:
            total_features = 3
        elif self.filter_ablation == 4:
            total_features = 15
        
        self.filter_network = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Linear(total_features, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        
    def forward(self, x):
        B, N, L = x.shape
        # lead_lag_raw = self.compute_time_domain_lead_lag_memory_efficient(x)  # (B, N, N, 3)
        lead_lag_raw = self.compute_spectral_features(x) # (B, N, N, 12)
        instantaneous_raw = self.compute_instantaneous_coupling(x)      # (B, N, N, 3)
        
        # lead_lag_scaled = self.selective_transform_spectral(lead_lag_raw)
        # instantaneous_scaled = self.selective_transform_instantaneous(instantaneous_raw)

        # all_features = torch.cat([lead_lag_scaled, instantaneous_scaled], dim=-1)
        # all_features = torch.cat([lead_lag_raw, instantaneous_raw], dim=-1)

        if self.filter_ablation == 1:
            def xavier_uniform_around_one(tensor):
                fan_in = tensor.size(-1)
                fan_out = tensor.size(-2) if len(tensor.shape) > 1 else tensor.size(-1)
                bound = math.sqrt(6.0 / (fan_in + fan_out))  # Xavier bound
                
                # Uniform distribution around 1.0
                nn.init.uniform_(tensor, 1.0 - bound, 1.0 + bound)
                return tensor

            tensor = torch.empty(B, N, N, 15)
            init_tensor = xavier_uniform_around_one(tensor).to(x.device)
            all_features = init_tensor
        
        elif self.filter_ablation == 2:
            all_features = lead_lag_raw
        
        elif self.filter_ablation == 3:
            all_features = instantaneous_raw
        
        else:
            all_features = torch.cat([lead_lag_raw, instantaneous_raw], dim=-1)
        
        # all_features = all_features.mean(dim=0, keepdim=True)
        
        filters = self.filter_network(all_features)
        
        return IdentityDropout(self.idrop)(filters)
    
    def compute_spectral_features(self, x):
        """
        Adopt the frequency-domain analysis from GeneralPALFilters
        Returns comprehensive spectral features across multiple frequency bands
        """
        B, N, L = x.shape
        
        # Apply window and FFT (same as GeneralPALFilters)
        window = torch.hann_window(L, device=x.device, dtype=x.dtype)
        x_windowed = x * window.unsqueeze(0).unsqueeze(0)
        X_fft = torch.fft.rfft(x_windowed, dim=-1)
        
        freq_bins = X_fft.shape[-1]
        
        # Define multiple frequency bands for comprehensive analysis
        ultra_low_end = max(1, int(freq_bins * 0.15))    
        low_end = max(ultra_low_end, int(freq_bins * 0.2)) 
        mid_end = max(low_end, int(freq_bins * 0.6)) 
        high_end = max(mid_end, int(freq_bins * 0.8)) 
        
        # Extract frequency bands
        X_ultra_low = X_fft[:, :, 0:ultra_low_end]      # Most persistent patterns
        X_low = X_fft[:, :, ultra_low_end:low_end]       # Medium-term patterns
        X_mid = X_fft[:, :, low_end:mid_end]       # Medium-term patterns
        X_high = X_fft[:, :, mid_end:high_end]       # Medium-term patterns
        
        # Compute band-specific features using coherence-weighted method
        ultra_low_features = self._compute_band_features(X_ultra_low)  # (B, N, N, 3)
        low_features = self._compute_band_features(X_low)             # (B, N, N, 3)
        mid_features = self._compute_band_features(X_mid)             # (B, N, N, 3)
        high_features = self._compute_band_features(X_high)             # (B, N, N, 3)  
        
        # Concatenate all spectral features
        return torch.cat([ultra_low_features, low_features, mid_features, high_features], dim=-1)  # (B, N, N, 12)
    
    def _compute_band_features(self, X_band):
        """
        Coherence-weighted feature extraction (from GeneralPALFilters)
        """
        B, N, freq_bins = X_band.shape
        
        if freq_bins == 0:
            return torch.zeros(B, N, N, 3, device=X_band.device)
        
        # Vectorized cross-spectral computation
        X_i = X_band.unsqueeze(2)  # (B, N, 1, freq_bins)
        X_j = X_band.unsqueeze(1)  # (B, 1, N, freq_bins)
        
        cross_spec = X_i * torch.conj(X_j)  # (B, N, N, freq_bins)
        auto_spec_i = torch.abs(X_i) ** 2   # (B, N, 1, freq_bins)
        auto_spec_j = torch.abs(X_j) ** 2   # (B, 1, N, freq_bins)
        
        # Phase differences, magnitudes, and coherence
        phase_per_freq = torch.angle(cross_spec)  # (B, N, N, freq_bins)
        mag_per_freq = torch.abs(cross_spec)      # (B, N, N, freq_bins)
        coherence_per_freq = (mag_per_freq ** 2) / (auto_spec_i * auto_spec_j + 1e-8)
        coherence_per_freq = torch.clamp(coherence_per_freq, 0, 1)
        
        # KEY: Coherence-weighted averages (noise-resistant!)
        coherence_sum = torch.sum(coherence_per_freq, dim=-1, keepdim=True) + 1e-8
        weights = coherence_per_freq / coherence_sum
        
        # Weighted averages - emphasizes frequencies with high coherence
        avg_phase = torch.sum(phase_per_freq * weights, dim=-1)     # (B, N, N)
        avg_magnitude = torch.sum(mag_per_freq * weights, dim=-1)   # (B, N, N)
        avg_coherence = torch.mean(coherence_per_freq, dim=-1)      # (B, N, N)
        
        return torch.stack([avg_phase, torch.log1p(avg_magnitude), avg_coherence], dim=-1)
    
    def compute_instantaneous_coupling(self, x):
        """
        Capture same-time relationships that lead-lag analysis misses
        """
        B, N, L = x.shape
        
        # 1. Instantaneous correlation (lag-0 relationship strength)
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        instant_corr = torch.bmm(x_norm, x_norm.transpose(-2, -1)) / L  # (B, N, N)
        
        # 2. Synchronized volatility (both variables volatile at same time)
        # Compute rolling volatility
        window_size = min(10, L // 4)
        x_unfolded = x.unfold(-1, window_size, 1)  # (B, N, L-window_size+1, window_size)
        rolling_vol = torch.std(x_unfolded, dim=-1)  # (B, N, L-window_size+1)
        
        # Correlation of volatilities (volatility clustering)
        vol_norm = (rolling_vol - rolling_vol.mean(dim=-1, keepdim=True)) / (rolling_vol.std(dim=-1, keepdim=True) + 1e-8)
        vol_corr = torch.bmm(vol_norm, vol_norm.transpose(-2, -1)) / rolling_vol.shape[-1]  # (B, N, N)
        
        # 3. Rank correlation (monotonic nonlinear same-time relationships)
        x_ranks = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
        rank_mean = x_ranks.mean(dim=-1, keepdim=True)
        x_ranks_centered = x_ranks - rank_mean
        rank_corr = torch.bmm(x_ranks_centered, x_ranks_centered.transpose(-2, -1)) / (L - 1)
        
        # Normalize rank correlation
        rank_std = torch.sqrt(torch.diagonal(rank_corr, dim1=-2, dim2=-1) + 1e-8).unsqueeze(-1)
        rank_corr = rank_corr / (rank_std * rank_std.transpose(-2, -1) + 1e-8)
        rank_corr = torch.nan_to_num(rank_corr, nan=0.0)
        
        return torch.stack([instant_corr, vol_corr, rank_corr], dim=-1)  # (B, N, N, 3)
    
    def selective_transform_spectral(self, spectral_features):
        """
        Only include features that are well-scaled or add unique information
        """
        phases = spectral_features[:, :, :, [0, 3, 6]]    # [-π, π] range
        mags = spectral_features[:, :, :, [1, 4, 7]]      # Wide range (log-transformed)
        cohs = spectral_features[:, :, :, [2, 5, 8]]      # [0, 1] range
        
        # Only include well-scaled originals + meaningful transforms
        return torch.cat([
            # Keep well-scaled originals
            phases,                         # Original phases (good scale)
            cohs,                          # Original coherences (good scale)
            
            # Transform poorly-scaled features
            torch.sigmoid(mags),           # Transform magnitudes (bad scale)
            
            # Add meaningful nonlinear versions
            torch.tanh(phases),            # Bounded phases for thresholding
            cohs ** 2,                     # Emphasized coherences
            torch.sigmoid(mags * 2)        # Sharpened magnitude thresholds
        ], dim=-1)  # Total: 18 features (3+3+3+3+3+3)

    def selective_transform_instantaneous(self, instantaneous_raw):
        """
        Transform based on what makes sense for each feature type
        Fixed to maintain proper tensor dimensions
        """
        instant_corr = instantaneous_raw[:, :, :, 0]  # (B, N, N)
        vol_corr = instantaneous_raw[:, :, :, 1]      # (B, N, N)  
        rank_corr = instantaneous_raw[:, :, :, 2]     # (B, N, N)
        
        # Stack features to restore the last dimension
        return torch.stack([
            # Keep originals (all well-scaled)
            instant_corr,
            vol_corr,
            rank_corr,
            
            # Add meaningful transforms only
            instant_corr ** 2,              # Strength emphasis
            vol_corr ** 2,                  # Volatility strength
            instant_corr * rank_corr        # Linear-nonlinear interaction
        ], dim=-1)  # (B, N, N, 6)

class SimplePALFilters(nn.Module):
    def __init__(self, configs, bands_to_keep):
        super().__init__()
        
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.idrop = configs.idrop
        
        # Streamlined feature set
        lead_lag_features = 3      # Your core temporal features
        instantaneous_features = 3 # Complementary same-time features  
        reliability_features = 2 # Relationship strength indicators
        independence_features = 2  # "No relationship" indicators
        
        total_features = lead_lag_features + instantaneous_features + reliability_features + independence_features
        
        self.filter_network = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Linear(total_features, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        
        # Learnable feature importance weights
        self.feature_importance = nn.Parameter(torch.ones(5))  # 5 feature groups
        
    def forward(self, x):
        # Core lead-lag information (keep this)
        lead_lag_features = self.compute_time_domain_lead_lag_memory_efficient(x)  # (B, N, N, 3)
        
        # Complementary features
        instantaneous_features = self.compute_instantaneous_coupling(x)      # (B, N, N, 3)
        reliability_features = self.compute_relationship_reliability(x)      # (B, N, N, 2)
        independence_features = self.compute_independence_indicators(x)      # (B, N, N, 2)
        
        # Apply learnable importance weighting
        weights = F.softmax(self.feature_importance, dim=0)
        
        weighted_features = torch.cat([
            lead_lag_features * weights[0],
            instantaneous_features * weights[1], 
            reliability_features * weights[2],
            independence_features * weights[3]
        ], dim=-1)
        # weighted_features = torch.cat([
        #     lead_lag_features,
        #     instantaneous_features,
        #     reliability_features,
        #     independence_features,
        # ], dim=-1)
        
        filters = self.filter_network(weighted_features)
        
        return IdentityDropout(self.idrop)(filters)
    
    def compute_time_domain_lead_lag_memory_efficient(self, x, max_lag=10):
        """
        Memory-efficient version using correlation theorem (FFT-based)
        Much faster for large max_lag values
        """
        B, N, L = x.shape
        
        window = torch.hann_window(L, device=x.device, dtype=x.dtype)
        x_windowed = x * window.unsqueeze(0).unsqueeze(0)
        
        # Normalize
        # x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        x_centered = x_windowed - x_windowed.mean(dim=-1, keepdim=True)
        x_norm = x_centered / (x_centered.std(dim=-1, keepdim=True) + 1e-8)
        
        # Zero-pad for circular correlation
        x_padded = F.pad(x_norm, (0, L))  # (B, N, 2*L)
        
        # FFT-based cross-correlation (much faster for large lags)
        X_fft = torch.fft.fft(x_padded, dim=-1)  # (B, N, 2*L)
        
        # Compute all pairwise cross-correlations using FFT
        X_i = X_fft.unsqueeze(1)  # (B, 1, N, 2*L)
        X_j_conj = torch.conj(X_fft).unsqueeze(2)  # (B, N, 1, 2*L)
        
        cross_corr_fft = X_i * X_j_conj  # (B, N, N, 2*L)
        cross_corr = torch.fft.ifft(cross_corr_fft, dim=-1).real  # (B, N, N, 2*L)
        
        # Extract relevant lags [0, max_lag]
        cross_corr = cross_corr[:, :, :, :max_lag+1] / L  # Normalize by length
        
        # Find maximum and extract features (same as before)
        abs_correlations = torch.abs(cross_corr)
        max_corr, max_lag_idx = torch.max(abs_correlations, dim=-1)
        
        lead_lag_direction = cross_corr.gather(-1, max_lag_idx.unsqueeze(-1)).squeeze(-1)
        lead_lag_direction = torch.sign(lead_lag_direction)
        
        normalized_lag = max_lag_idx.float() / max(max_lag, 1)
        
        return torch.stack([max_corr, normalized_lag, lead_lag_direction], dim=-1)
    
    def compute_instantaneous_coupling(self, x):
        """
        Capture same-time relationships that lead-lag analysis misses
        """
        B, N, L = x.shape
        
        # 1. Instantaneous correlation (lag-0 relationship strength)
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        instant_corr = torch.bmm(x_norm, x_norm.transpose(-2, -1)) / L  # (B, N, N)
        
        # 2. Synchronized volatility (both variables volatile at same time)
        # Compute rolling volatility
        window_size = min(10, L // 4)
        x_unfolded = x.unfold(-1, window_size, 1)  # (B, N, L-window_size+1, window_size)
        rolling_vol = torch.std(x_unfolded, dim=-1)  # (B, N, L-window_size+1)
        
        # Correlation of volatilities (volatility clustering)
        vol_norm = (rolling_vol - rolling_vol.mean(dim=-1, keepdim=True)) / (rolling_vol.std(dim=-1, keepdim=True) + 1e-8)
        vol_corr = torch.bmm(vol_norm, vol_norm.transpose(-2, -1)) / rolling_vol.shape[-1]  # (B, N, N)
        
        # 3. Rank correlation (monotonic nonlinear same-time relationships)
        x_ranks = torch.argsort(torch.argsort(x, dim=-1), dim=-1).float()
        rank_mean = x_ranks.mean(dim=-1, keepdim=True)
        x_ranks_centered = x_ranks - rank_mean
        rank_corr = torch.bmm(x_ranks_centered, x_ranks_centered.transpose(-2, -1)) / (L - 1)
        
        # Normalize rank correlation
        rank_std = torch.sqrt(torch.diagonal(rank_corr, dim1=-2, dim2=-1) + 1e-8).unsqueeze(-1)
        rank_corr = rank_corr / (rank_std * rank_std.transpose(-2, -1) + 1e-8)
        rank_corr = torch.nan_to_num(rank_corr, nan=0.0)
        
        return torch.stack([instant_corr, vol_corr, rank_corr], dim=-1)  # (B, N, N, 3)
    
    def compute_relationship_reliability(self, x):
        """
        Assess how reliable/stable the relationship is between channel pairs
        """
        B, N, L = x.shape
        
        # 1. Relationship stability over time
        # Split sequence into thirds and compare correlations
        third = L // 3
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        
        # Correlations in different time periods
        corr_early = torch.bmm(x_norm[:, :, :third], x_norm[:, :, :third].transpose(-2, -1)) / third
        corr_late = torch.bmm(x_norm[:, :, -third:], x_norm[:, :, -third:].transpose(-2, -1)) / third
        
        # Stability = negative absolute difference (stable relationships have consistent correlations)
        stability = -torch.abs(corr_early - corr_late)  # (B, N, N)
        
        # 2. Signal-to-noise ratio of the relationship
        # Higher SNR = more reliable relationship
        signal_power = torch.var(x, dim=-1)  # (B, N)
        noise_proxy = torch.var(torch.diff(x, dim=-1), dim=-1)  # High-frequency noise
        snr = signal_power / (noise_proxy + 1e-8)  # (B, N)
        
        # Pairwise SNR geometric mean (both channels should have good SNR for reliable relationship)
        snr_i = snr.unsqueeze(2)  # (B, N, 1)
        snr_j = snr.unsqueeze(1)  # (B, 1, N)
        pairwise_snr = torch.sqrt(snr_i * snr_j)  # (B, N, N)
        pairwise_snr = torch.log1p(pairwise_snr)  # Log transform for stability
        
        return torch.stack([stability, pairwise_snr], dim=-1)  # (B, N, N, 2)
    
    def compute_independence_indicators(self, x):
        """
        Detect when two channels have little to no meaningful relationship
        These features should guide the network to dampen attention between unrelated channels
        """
        B, N, L = x.shape
        
        # 1. Statistical independence test (correlation close to zero across multiple lags)
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        
        # Test correlations at multiple lags
        max_test_lag = min(5, L // 4)
        lag_correlations = []
        
        for lag in range(max_test_lag + 1):
            if lag == 0:
                corr = torch.bmm(x_norm, x_norm.transpose(-2, -1)) / L
            else:
                x_current = x_norm[:, :, lag:]
                x_lagged = x_norm[:, :, :-lag]
                corr = torch.bmm(x_current, x_lagged.transpose(-2, -1)) / (L - lag)
            
            lag_correlations.append(torch.abs(corr))
        
        # Average absolute correlation across lags (low = likely independent)
        avg_abs_corr = torch.stack(lag_correlations, dim=-1).mean(dim=-1)  # (B, N, N)
        independence_score = 1.0 - avg_abs_corr  # High score = likely independent
        
        # 2. Distributional dissimilarity (different distributions = likely unrelated)
        # Compare empirical distributions using simple moments
        x_skewness = torch.mean(((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)) ** 3, dim=-1)  # (B, N)
        x_kurtosis = torch.mean(((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)) ** 4, dim=-1)  # (B, N)
        
        # Pairwise distributional dissimilarity
        skew_diff = torch.abs(x_skewness.unsqueeze(2) - x_skewness.unsqueeze(1))  # (B, N, N)
        kurt_diff = torch.abs(x_kurtosis.unsqueeze(2) - x_kurtosis.unsqueeze(1))  # (B, N, N)
        
        # Combine skewness and kurtosis differences (high = different distributions = likely unrelated)
        distributional_dissimilarity = (skew_diff + kurt_diff) / 2  # (B, N, N)
        
        return torch.stack([independence_score, distributional_dissimilarity], dim=-1)  # (B, N, N, 2)