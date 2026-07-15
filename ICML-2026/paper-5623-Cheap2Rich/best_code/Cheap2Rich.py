import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy
import os
import time
from skimage.metrics import structural_similarity as ssim


seed = int(os.environ.get("RUN_SEED", "42"))
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Using seed: {seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else device)  # For Apple Silicon Macs
print(f"Using device: {device}")


# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, U, sensor_indices, lags, scaler=None, fit_scaler=False):
        self.U = U
        self.sensor_indices = sensor_indices
        self.lags = lags
        self.S = U[:, sensor_indices]

        if scaler is None:
            self.scaler_U = MinMaxScaler()
            self.scaler_S = MinMaxScaler()
        else:
            self.scaler_U, self.scaler_S = scaler

        if fit_scaler:
            self.U_scaled = self.scaler_U.fit_transform(self.U)
            self.S_scaled = self.scaler_S.fit_transform(self.S)
        else:
            self.U_scaled = self.scaler_U.transform(self.U)
            self.S_scaled = self.scaler_S.transform(self.S)

        self.valid_indices = np.arange(lags, len(U))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        sensor_history = self.S_scaled[t - self.lags:t]
        full_state = self.U_scaled[t]
        return (torch.tensor(sensor_history, dtype=torch.float32),
                torch.tensor(full_state, dtype=torch.float32))

    def get_scalers(self):
        return (self.scaler_U, self.scaler_S)


# Frequency Sparsity Losses
def frequency_sparsity_bandlimited(signal, max_freq):
    """
    Sparsity ONLY within targeted frequency band.
    Heavily penalizes energy outside the band (untargeted by sensors).
    """
    fft_coeffs = torch.fft.rfft(signal, dim=-1)
    magnitudes = torch.abs(fft_coeffs)

    # Split into targeted and untargeted bands
    targeted = magnitudes[:, :max_freq + 1]
    untargeted = magnitudes[:, max_freq + 1:] if magnitudes.shape[-1] > max_freq + 1 else None

    # Sparsity on targeted band only (normalized L1)
    l1_res = torch.sum(targeted, dim=-1)
    l2_res = torch.sqrt(torch.sum(targeted ** 2, dim=-1) + 1e-8)
    sparsity_loss = l1_res / (l2_res + 1e-8)

    # HEAVY penalty for out-of-band energy
    if untargeted is not None and untargeted.shape[-1] > 0:
        out_of_band_energy = torch.sum(untargeted ** 2, dim=-1)
        total_energy = torch.sum(magnitudes ** 2, dim=-1) + 1e-8
        out_of_band_ratio = out_of_band_energy / total_energy
        return torch.mean(sparsity_loss) + 100.0 * torch.mean(out_of_band_ratio)
    else:
        return torch.mean(sparsity_loss)


def ssim1d(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    """Differentiable 1D SSIM."""
    import torch.nn.functional as F
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    mu_x = F.avg_pool1d(x, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool1d(y, window_size, stride=1, padding=window_size//2)
    sigma_x_sq = F.avg_pool1d(x**2, window_size, stride=1, padding=window_size//2) - mu_x**2
    sigma_y_sq = F.avg_pool1d(y**2, window_size, stride=1, padding=window_size//2) - mu_y**2
    sigma_xy = F.avg_pool1d(x*y, window_size, stride=1, padding=window_size//2) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.squeeze(1).mean(dim=1)


# Models
class SHRED(nn.Module):
    """Standard SHRED encoder-decoder"""

    def __init__(self, num_sensors, lags, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_sensors, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def encode(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.norm(h_n[-1])

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


class HF_SHRED(nn.Module):
    """
    HF-SHRED for high-frequency residual learning.

    Features:
    1. Time-delay embedding: Includes temporal derivatives (velocity info)
    2. Attention over lags: Learns which temporal lags are most informative
    3. Spatial deformation: Learns position-dependent warping to correct velocity
    """

    def __init__(self, num_sensors, lags, hidden_size, output_size,
                 velocity_correction='deformation', use_time_derivatives=True, use_lag_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_sensors = num_sensors
        self.lags = lags
        self.velocity_correction = velocity_correction
        self.use_time_derivatives = use_time_derivatives
        self.use_lag_attention = use_lag_attention

        # Input size depends on whether we use time derivatives
        if use_time_derivatives:
            input_size = num_sensors * 3  # value, d/dt, d²/dt²
        else:
            input_size = num_sensors

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_size)

        # Attention over lags
        if use_lag_attention:
            self.lag_attention = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.Tanh(),
                nn.Linear(64, lags),
                nn.Softmax(dim=-1)
            )
            self.lstm_all_steps = nn.LSTM(input_size, hidden_size, num_layers=1,
                                          batch_first=True)

        # Main decoder for spatial pattern
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        # Velocity/deformation correction options
        if velocity_correction == 'frequency_phase':
            n_freqs = output_size // 2 + 1
            self.phase_predictor = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.Tanh(),
                nn.Linear(64, n_freqs)
            )
        elif velocity_correction == 'deformation':
            # Predicts a spatially-varying shift field
            # This can handle non-uniform velocity across space
            self.deformation_net = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, output_size)  # One shift per spatial point
            )
            # Also predict an amplitude modulation (to handle amplitude variations)
            self.amplitude_net = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, output_size),
                nn.Softplus()  # Positive multiplier
            )

        self.scale = nn.Parameter(torch.tensor(0.5))

    def compute_time_derivatives(self, x):
        """Compute temporal derivatives from sensor history."""
        batch_size, lags, n_sensors = x.shape

        # First derivative (velocity)
        dx_dt = torch.zeros_like(x)
        dx_dt[:, 1:-1, :] = (x[:, 2:, :] - x[:, :-2, :]) / 2.0
        dx_dt[:, 0, :] = x[:, 1, :] - x[:, 0, :]
        dx_dt[:, -1, :] = x[:, -1, :] - x[:, -2, :]

        # Second derivative (acceleration)
        d2x_dt2 = torch.zeros_like(x)
        d2x_dt2[:, 1:-1, :] = x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :]
        d2x_dt2[:, 0, :] = d2x_dt2[:, 1, :]
        d2x_dt2[:, -1, :] = d2x_dt2[:, -2, :]

        return torch.cat([x, dx_dt, d2x_dt2], dim=-1)

    def encode(self, x):
        """Encode with optional time derivatives and lag attention."""
        if self.use_time_derivatives:
            x_embedded = self.compute_time_derivatives(x)
        else:
            x_embedded = x

        if self.use_lag_attention:
            all_hidden, _ = self.lstm_all_steps(x_embedded)
            _, (h_final, _) = self.lstm(x_embedded)
            h_final = self.norm(h_final[-1])

            attn_weights = self.lag_attention(h_final)
            z = torch.sum(attn_weights.unsqueeze(-1) * all_hidden, dim=1)
            self.last_attention_weights = attn_weights.detach()
        else:
            _, (h_n, _) = self.lstm(x_embedded)
            z = self.norm(h_n[-1])

        return z

    def apply_frequency_phase(self, signal, phase_shifts):
        """Apply frequency-dependent phase shifts."""
        fft = torch.fft.rfft(signal, dim=-1)
        phase_correction = torch.exp(1j * phase_shifts)
        corrected_fft = fft * phase_correction
        return torch.fft.irfft(corrected_fft, n=signal.shape[-1], dim=-1)

    def apply_deformation(self, signal, shift_field, amplitude_field=None):
        """
        Apply spatially-varying deformation (warping).

        shift_field: (batch, N) - how much to shift at each position
        amplitude_field: (batch, N) - amplitude modulation at each position

        This implements: output[x] = amplitude[x] * signal[x + shift[x]]
        """
        batch_size, N = signal.shape
        device = signal.device

        # Create base coordinates
        x_base = torch.linspace(0, N - 1, N, device=device).unsqueeze(0).expand(batch_size, -1)

        # Apply shift (with periodic boundary)
        x_warped = (x_base + shift_field) % N

        # Bilinear interpolation for smooth warping
        x_floor = x_warped.long() % N
        x_ceil = (x_floor + 1) % N
        w_ceil = (x_warped - x_warped.floor())
        w_floor = 1 - w_ceil

        # Gather and interpolate
        signal_floor = torch.gather(signal, 1, x_floor)
        signal_ceil = torch.gather(signal, 1, x_ceil)
        warped_signal = w_floor * signal_floor + w_ceil * signal_ceil

        # Apply amplitude modulation if provided
        if amplitude_field is not None:
            warped_signal = warped_signal * amplitude_field

        return warped_signal

    def forward(self, x):
        z = self.encode(x)
        base_output = self.scale * self.decoder(z)

        # Apply velocity/deformation correction
        if self.velocity_correction == 'frequency_phase':
            phase_shifts = self.phase_predictor(z) * 0.5
            output = self.apply_frequency_phase(base_output, phase_shifts)
        elif self.velocity_correction == 'deformation':
            # Predict spatially-varying shift and amplitude
            shift_field = self.deformation_net(z)
            shift_field = torch.tanh(shift_field) * 10  # Max ±10 grid points shift

            amplitude_field = self.amplitude_net(z)
            amplitude_field = amplitude_field * 0.5 + 0.5  # Range [0.5, 1.5] approximately

            output = self.apply_deformation(base_output, shift_field, amplitude_field)
        else:
            output = base_output

        return output, z

    def get_attention_weights(self):
        """Return last computed attention weights for visualization."""
        if hasattr(self, 'last_attention_weights'):
            return self.last_attention_weights
        return None


class LatentGAN(nn.Module):
    """GAN for latent alignment"""

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


class SparseFreqDASHRED(nn.Module):
    """DA-SHRED with sparse-frequency HF learning"""

    def __init__(self, lf_shred, num_sensors, lags, hidden_size, output_size, sensor_indices):
        super().__init__()

        # LF pathway
        self.lf_lstm = copy.deepcopy(lf_shred.lstm)
        self.lf_norm = copy.deepcopy(lf_shred.norm)
        self.lf_decoder = copy.deepcopy(lf_shred.decoder)
        self.gan = LatentGAN(lf_shred.hidden_size)

        # HF pathway with enhanced temporal processing + spatial deformation
        # The HF patterns have spatially and temporally varying velocity
        self.hf_shred = HF_SHRED(num_sensors, lags, hidden_size, output_size,
                                 velocity_correction='deformation',  # Spatially-varying warping
                                 use_time_derivatives=True,
                                 use_lag_attention=True)

        self.register_buffer('sensor_indices', torch.tensor(sensor_indices, dtype=torch.long))
        self.lags = lags
        self.num_sensors = num_sensors
        self.output_size = output_size

    def encode_lf(self, x):
        _, (h_n, _) = self.lf_lstm(x)
        return self.lf_norm(h_n[-1])

    def decode_lf(self, z):
        return self.lf_decoder(z)

    def forward(self, sensor_history, use_gan=True):
        batch_size = sensor_history.shape[0]

        # LF pathway
        z_lf = self.encode_lf(sensor_history)
        if use_gan:
            z_lf = self.gan(z_lf)
        u_lf = self.decode_lf(z_lf)

        # Compute residual history
        residual_history = torch.zeros_like(sensor_history)
        sensors_lf_pred = u_lf[:, self.sensor_indices]

        for lag in range(self.lags):
            residual_history[:, lag, :] = sensor_history[:, lag, :] - sensors_lf_pred.detach()

        # HF pathway (with its own velocity/phase correction)
        u_hf, z_hf = self.hf_shred(residual_history)

        u_total = u_lf + u_hf

        return u_total, u_lf, u_hf, z_lf, z_hf


# Training Functions
def train_lf_shred(model, train_loader, epochs=200, lr=1e-3):
    print("\n=== Stage 1: Train LF-SHRED on Simulation ===")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sensors, targets in train_loader:
            sensors, targets = sensors.to(device), targets.to(device)
            optimizer.zero_grad()
            pred, _ = model(sensors)
            mse_loss = F.mse_loss(pred, targets)
            ssim_loss_val = (1 - ssim1d(pred, targets)).mean()
            loss = mse_loss + 0.01 * ssim_loss_val
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")
    return model


def train_gan(dashred, z_sim, z_real, epochs=300, lr=1e-4):
    print("\n=== Stage 2: Train GAN for LF Alignment ===")

    z_sim = torch.tensor(z_sim, dtype=torch.float32).to(device)
    z_real = torch.tensor(z_real, dtype=torch.float32).to(device)

    opt_g = optim.Adam(dashred.gan.generator.parameters(), lr=lr)
    opt_d = optim.Adam(dashred.gan.discriminator.parameters(), lr=lr)

    batch_size = 32
    n_batches = min(len(z_sim), len(z_real)) // batch_size

    for epoch in range(epochs):
        perm_sim = torch.randperm(len(z_sim))
        perm_real = torch.randperm(len(z_real))

        for i in range(n_batches):
            z_s = z_sim[perm_sim[i * batch_size:(i + 1) * batch_size]]
            z_r = z_real[perm_real[i * batch_size:(i + 1) * batch_size]]

            # Discriminator
            opt_d.zero_grad()
            z_fake = dashred.gan(z_s)
            d_loss = F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_r), torch.ones(len(z_r), 1, device=device)
            ) + F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_fake.detach()), torch.zeros(len(z_s), 1, device=device)
            )
            d_loss.backward()
            opt_d.step()

            # Generator
            opt_g.zero_grad()
            z_fake = dashred.gan(z_s)
            g_loss = F.binary_cross_entropy_with_logits(
                dashred.gan.discriminator(z_fake), torch.ones(len(z_s), 1, device=device)
            )
            g_loss.backward()
            opt_g.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")


def train_hf_sparse(dashred, train_loader_real, sensor_indices, epochs=500, lr=1e-3,
                    lambda_sparse=0.1, sparsity_type='bandlimited', stage_name="Stage 3"):
    """Train HF-SHRED with sensor-only supervision + frequency sparsity."""

    max_targeted_freq = len(sensor_indices) // 2
    print(f"\n=== {stage_name}: Train HF-SHRED (Sensor-Only + {sparsity_type} Sparsity) ===")
    print(f"    lambda_sparse = {lambda_sparse}")
    print(f"    Max target frequency: k <= {max_targeted_freq}")

    params = list(dashred.hf_shred.parameters())
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

    sensor_idx = torch.tensor(sensor_indices, dtype=torch.long).to(device)

    history = {'sensor_loss': [], 'sparsity_loss': [], 'total_loss': []}

    # Estimate target residual scale from first batch
    with torch.no_grad():
        sample_sensors, _ = next(iter(train_loader_real))
        sample_sensors = sample_sensors.to(device)
        _, u_lf, _, _, _ = dashred(sample_sensors, use_gan=True)
        sensors_current = sample_sensors[:, -1, :]
        sensors_lf = u_lf[:, sensor_idx]
        residual_scale = (sensors_current - sensors_lf).abs().mean().item()
    print(f"    Estimated residual scale: {residual_scale:.4f}")
    max_hf_magnitude = residual_scale * 3

    warmup_epochs = 100

    for epoch in range(epochs):
        dashred.train()
        epoch_sensor_loss = 0
        epoch_sparsity_loss = 0
        epoch_mag_loss = 0

        if epoch < warmup_epochs:
            current_lambda = 0.0
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            current_lambda = lambda_sparse * min(1.0, progress * 2)

        for sensors, targets in train_loader_real:
            sensors = sensors.to(device)

            optimizer.zero_grad()

            u_total, u_lf, u_hf, _, _ = dashred(sensors, use_gan=True)

            # Loss 1: Sensor residual matching
            sensors_current = sensors[:, -1, :]
            sensors_lf = u_lf[:, sensor_idx].detach()
            sensors_residual_true = sensors_current - sensors_lf
            sensors_hf = u_hf[:, sensor_idx]

            sensor_loss = F.mse_loss(sensors_hf, sensors_residual_true)
            ssim_hf_loss = (1 - ssim1d(sensors_hf, sensors_residual_true)).mean()

            # Loss 2: Frequency sparsity (bandlimited)
            sparsity_loss = frequency_sparsity_bandlimited(u_hf, max_targeted_freq)

            # Loss 3: Magnitude constraint
            hf_magnitude = u_hf.abs().mean()
            magnitude_penalty = F.relu(hf_magnitude - max_hf_magnitude) ** 2

            loss = sensor_loss + 0.005 * ssim_hf_loss + current_lambda * sparsity_loss + 1.0 * magnitude_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_sensor_loss += sensor_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            epoch_mag_loss += magnitude_penalty.item()

        n_batches = len(train_loader_real)
        epoch_sensor_loss /= n_batches
        epoch_sparsity_loss /= n_batches
        epoch_mag_loss /= n_batches

        history['sensor_loss'].append(epoch_sensor_loss)
        history['sparsity_loss'].append(epoch_sparsity_loss)
        history['total_loss'].append(epoch_sensor_loss + current_lambda * epoch_sparsity_loss)

        scheduler.step(epoch_sensor_loss)

        if (epoch + 1) % 50 == 0:
            dashred.eval()
            with torch.no_grad():
                sample_sensors = next(iter(train_loader_real))[0][:1].to(device)
                _, _, u_hf_sample, _, _ = dashred(sample_sensors, use_gan=True)
                fft_mag = torch.abs(torch.fft.rfft(u_hf_sample, dim=-1)).squeeze().cpu().numpy()
                top_freqs = np.argsort(fft_mag)[-5:][::-1]
                hf_max = u_hf_sample.abs().max().item()

            print(f"  Epoch {epoch + 1}/{epochs}, Sensor: {epoch_sensor_loss:.6f}, "
                  f"Sparsity: {epoch_sparsity_loss:.4f}, Mag_pen: {epoch_mag_loss:.4f}, "
                  f"HF_max: {hf_max:.4f}, Top freqs: {top_freqs}")

    return dashred, history



def inspect_model_predictions(ground_truth, all_preds, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Plot predictions
    im1 = axes[0].imshow(all_preds.T, aspect='auto', cmap='RdBu_r', origin='lower', vmin=-0.04288087, vmax=1.0402466)
    # extent=(0, all_preds.shape[0], 0, all_preds.shape[1]))
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('X Index')
    axes[0].set_title(f"{title_prefix} Predictions on High-Fidelity Data", fontdict={'fontsize': 16})
    plt.colorbar(im1, ax=axes[0], label='Value')

    # Plot error
    error = np.abs(all_preds - ground_truth)
    im2 = axes[1].imshow(error.T, aspect='auto', cmap='inferno', origin='lower',
                         vmin=0, vmax=0.75)
    axes[1].set_xlabel('Time Index')
    axes[1].set_ylabel('X Index')
    axes[1].set_title(f"{title_prefix} Absolute Error on High-Fidelity Data", fontdict={'fontsize': 16})
    plt.colorbar(im2, ax=axes[1], label='Value')

    plt.tight_layout()
    plt.show()


# Main
if __name__ == "__main__":
    # Check for data files
    data_dir = "."
    sim_file = os.path.join(data_dir, "Koch_model_processed.npy")
    real_file = os.path.join(data_dir, "high_fidelity_sim_processed.npy")

    if not os.path.exists(sim_file) or not os.path.exists(real_file):
        print(f"ERROR: Data files not found!")
        print(f"  Expected: {sim_file}")
        print(f"  Expected: {real_file}")
        print("\nPlease ensure the RDE data files are in the DA_data/ directory.")
        exit(1)

    # Load RDE data
    print("[1] Loading RDE data...")
    U_sim = np.load(sim_file, allow_pickle=True)
    U_real = np.load(real_file, allow_pickle=True)

    # Optional: Apply Gaussian smoothing to real data (as in original)
    from scipy.ndimage import gaussian_filter1d


    def smooth_data(data, sigma_temporal=0.15, sigma_spatial=0.7):
        smoothed = gaussian_filter1d(data, sigma=sigma_temporal, axis=0)
        smoothed = gaussian_filter1d(smoothed, sigma=sigma_spatial, axis=1)
        return smoothed


    #U_real = smooth_data(U_real)

    print(f"    U_sim shape: {U_sim.shape}, range: [{U_sim.min():.4f}, {U_sim.max():.4f}]")
    print(f"    U_real shape: {U_real.shape}, range: [{U_real.min():.4f}, {U_real.max():.4f}]")

    # === DIAGNOSTIC: Plot raw data to verify what we're working with ===
    print("\n[1.5] DIAGNOSTIC: Visualizing raw data...")
    print(f"    U_sim shape: {U_sim.shape}")
    print(f"    U_real shape: {U_real.shape}")

    fig_raw, axes_raw = plt.subplots(1, 3, figsize=(15, 5))

    # Use consistent color range
    vmin_raw = min(U_sim.min(), U_real.min())
    vmax_raw = max(U_sim.max(), U_real.max())

    # Simulation
    im0 = axes_raw[0].imshow(U_sim.T, aspect='auto', cmap='RdBu_r', origin='lower',
                             vmin=vmin_raw, vmax=vmax_raw)
    axes_raw[0].set_title(f'Simulation (Koch Model)\nShape: {U_sim.shape}', fontsize=12)
    axes_raw[0].set_xlabel('Time')
    axes_raw[0].set_ylabel('Space (x)')
    plt.colorbar(im0, ax=axes_raw[0])

    # Real Physics (after smoothing)
    im1 = axes_raw[1].imshow(U_real.T, aspect='auto', cmap='RdBu_r', origin='lower',
                             vmin=vmin_raw, vmax=vmax_raw)
    axes_raw[1].set_title(f'Real Physics (High Fidelity, smoothed)\nShape: {U_real.shape}', fontsize=12)
    axes_raw[1].set_xlabel('Time')
    axes_raw[1].set_ylabel('Space (x)')
    plt.colorbar(im1, ax=axes_raw[1])

    # Difference
    min_t = min(U_sim.shape[0], U_real.shape[0])
    diff = U_real[:min_t] - U_sim[:min_t]
    im2 = axes_raw[2].imshow(diff.T, aspect='auto', cmap='RdBu_r', origin='lower')
    axes_raw[2].set_title('Difference (Real - Sim)', fontsize=12)
    axes_raw[2].set_xlabel('Time')
    axes_raw[2].set_ylabel('Space (x)')
    plt.colorbar(im2, ax=axes_raw[2])

    plt.tight_layout()
    plt.savefig('RDE_raw_data_diagnostic.png', dpi=150, bbox_inches='tight')
    print("    Saved: RDE_raw_data_diagnostic.png")

    # Parameters
    N = U_sim.shape[1]  # Spatial points
    num_sensors = 25
    lags = 25
    hidden_size = 32

    # Sparsity settings
    lambda_sparse = 0.1
    sparsity_type = 'bandlimited'

    # Create datasets
    print("\n[2] Creating datasets...")
    sensor_indices = np.linspace(0, N - 1, num_sensors, dtype=int)
    print(f"    Sensors: {num_sensors} at indices {sensor_indices[:5]}...{sensor_indices[-5:]}")
    print(f"    Max frequency: k <= {num_sensors // 2}")

    n_train = int(0.8 * len(U_sim))

    # IMPORTANT: Fit scaler on combined data to handle different scales
    U_combined = np.vstack([U_sim, U_real])
    temp_dataset = TimeSeriesDataset(U_combined, sensor_indices, lags, fit_scaler=True)
    combined_scaler = temp_dataset.get_scalers()

    train_sim = TimeSeriesDataset(U_sim[:n_train], sensor_indices, lags, scaler=combined_scaler, fit_scaler=False)
    valid_sim = TimeSeriesDataset(U_sim[n_train:], sensor_indices, lags, scaler=combined_scaler)
    train_real = TimeSeriesDataset(U_real[:n_train], sensor_indices, lags, scaler=combined_scaler)
    valid_real = TimeSeriesDataset(U_real[n_train:], sensor_indices, lags, scaler=combined_scaler)

    scaler_U, _ = combined_scaler

    train_loader_sim = DataLoader(train_sim, batch_size=32, shuffle=True)
    train_loader_real = DataLoader(train_real, batch_size=32, shuffle=True)
    valid_loader_real = DataLoader(valid_real, batch_size=32)

    print(f"    Train samples (sim): {len(train_sim)}")
    print(f"    Train samples (real): {len(train_real)}")
    print(f"    Valid samples (real): {len(valid_real)}")

    # Stage 1: Train LF-SHRED on simulation
    print("\n[3] Stage 1: Train LF-SHRED on simulation...")
    
    # start timing here
    start_time = time.time()

    lf_shred = SHRED(num_sensors, lags, hidden_size, N).to(device)
    lf_shred = train_lf_shred(lf_shred, train_loader_sim, epochs=300)

    # params of lf_shred
    total_params_lf = sum(p.numel() for p in lf_shred.parameters())
    trainable_params_lf = sum(p.numel() for p in lf_shred.parameters() if p.requires_grad)
    print(f"\n[Summary of params] Total parameters in LF-SHRED: {total_params_lf}, Trainable parameters: {trainable_params_lf}")

    # Evaluate LF-SHRED baseline
    lf_shred.eval()
    with torch.no_grad():
        preds, targets_list = [], []
        for sensors, targets in valid_loader_real:
            pred, _ = lf_shred(sensors.to(device))
            preds.append(pred.cpu())
            targets_list.append(targets)
        preds = scaler_U.inverse_transform(torch.cat(preds).numpy())
        targets_np = scaler_U.inverse_transform(torch.cat(targets_list).numpy())
        baseline_rmse = np.sqrt(np.mean((preds - targets_np) ** 2))
    print(f"    LF-SHRED baseline RMSE on real: {baseline_rmse:.6f}")

    time_to_train_lf = time.time() - start_time
    print(f"    Time to train LF-SHRED: {time_to_train_lf:.2f} seconds ({time_to_train_lf/60:.2f} minutes)")
    # Create DA-SHRED
    print("\n[4] Creating SparseFreq DA-SHRED...")
    dashred = SparseFreqDASHRED(lf_shred, num_sensors, lags, hidden_size, N, sensor_indices).to(device)

    # Stage 2: Extract latents and train GAN
    print("\n[5] Extracting latents for GAN...")
    dashred.eval()
    Z_sim_list, Z_real_list = [], []
    with torch.no_grad():
        for sensors, _ in train_loader_sim:
            Z_sim_list.append(dashred.encode_lf(sensors.to(device)).cpu())
        for sensors, _ in train_loader_real:
            Z_real_list.append(dashred.encode_lf(sensors.to(device)).cpu())
    Z_sim = torch.cat(Z_sim_list).numpy()
    Z_real = torch.cat(Z_real_list).numpy()
    print(f"    Z_sim: {Z_sim.shape}, Z_real: {Z_real.shape}")

    # Diagnostic: Check latent distribution difference
    print(f"    Z_sim mean: {Z_sim.mean():.4f}, std: {Z_sim.std():.4f}")
    print(f"    Z_real mean: {Z_real.mean():.4f}, std: {Z_real.std():.4f}")

    print("\n[6] Stage 2: Train GAN...")
    train_gan(dashred, Z_sim, Z_real, epochs=400)
    time_to_train_gan = time.time() - start_time - time_to_train_lf
    print(f"    Time to train GAN: {time_to_train_gan:.2f} seconds ({time_to_train_gan/60:.2f} minutes)")
    

    # Diagnostic: Check sensor residual
    print("\n[6.5] DIAGNOSTIC: Checking sensor residual...")
    dashred.eval()
    sensor_idx = torch.tensor(sensor_indices, dtype=torch.long).to(device)
    with torch.no_grad():
        sample_sensors, sample_targets = next(iter(train_loader_real))
        sample_sensors = sample_sensors.to(device)
        sample_targets = sample_targets.to(device)

        z_lf = dashred.encode_lf(sample_sensors)
        z_lf_aligned = dashred.gan(z_lf)
        u_lf = dashred.decode_lf(z_lf_aligned)

        sensors_current = sample_sensors[:, -1, :]
        sensors_lf_pred = u_lf[:, sensor_idx]
        sensors_residual = sensors_current - sensors_lf_pred

        full_residual = sample_targets - u_lf
        hf_at_sensors = full_residual[:, sensor_idx]

        print(f"    Sensors_current range: [{sensors_current.min():.4f}, {sensors_current.max():.4f}]")
        print(f"    Sensors_lf_pred range: [{sensors_lf_pred.min():.4f}, {sensors_lf_pred.max():.4f}]")
        print(f"    Sensor residual range: [{sensors_residual.min():.4f}, {sensors_residual.max():.4f}]")
        print(f"    Sensor residual mean abs: {sensors_residual.abs().mean():.6f}")
        print(f"    Full residual mean abs: {full_residual.abs().mean():.6f}")

    elapsed_time = time.time() - start_time
    print(f"\n[Timing] before HF training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Stage 3: Train HF with sparsity
    print(f"\n[7] Stage 3: Train HF-SHRED with {sparsity_type} sparsity...")
    dashred, history = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=500, lr=3e-4, lambda_sparse=lambda_sparse, sparsity_type=sparsity_type,
        stage_name="Stage 3"
    )

    # Stage 4: Fine-tune with reduced sparsity
    print("\n[7.5] Stage 4: Fine-tuning with reduced sparsity...")
    dashred, history2 = train_hf_sparse(
        dashred, train_loader_real, sensor_indices,
        epochs=200, lr=3e-4, lambda_sparse=lambda_sparse * 0.1, sparsity_type=sparsity_type,
        stage_name="Stage 4 (Fine-tune)"
    )
    
    # print a summary of the number of parameters
    total_params = sum(p.numel() for p in dashred.parameters())
    trainable_params = sum(p.numel() for p in dashred.parameters() if p.requires_grad)
    print(f"\n[Summary] Total parameters in DA-SHRED: {total_params}, Trainable parameters: {trainable_params}")
    time_to_train_hf = time.time() - start_time - time_to_train_lf - time_to_train_gan
    print(f"    Time to train HF-SHRED: {time_to_train_hf:.2f} seconds ({time_to_train_hf/60:.2f} minutes)")

    # Evaluate
    print("\n[8] Evaluating...")
    dashred.eval()

    results = {'lf_only': [], 'lf_hf': [], 'targets': []}

    with torch.no_grad():
        for sensors, targets in valid_loader_real:
            sensors = sensors.to(device)

            z_lf = dashred.encode_lf(sensors)
            z_lf_aligned = dashred.gan(z_lf)
            u_lf = dashred.decode_lf(z_lf_aligned)

            u_total, _, u_hf, _, _ = dashred(sensors, use_gan=True)

            results['lf_only'].append(u_lf.cpu())
            results['lf_hf'].append(u_total.cpu())
            results['targets'].append(targets)

    for k in results:
        results[k] = scaler_U.inverse_transform(torch.cat(results[k]).numpy())

    # The validation dataset starts at index n_train in the SCALED data
    # But the TimeSeriesDataset only returns samples starting from index `lags`
    # So valid_real contains samples from indices [n_train + lags, len(U_real)]
    # The number of valid samples = len(U_real) - n_train - lags

    # For the ORIGINAL unscaled data, we need the same time indices
    # valid_real dataset covers time indices: n_train + lags to len(U_real) - 1
    # But wait - valid_real is created from U_real[n_train:], so within that slice,
    # the valid indices are [lags, len(U_real) - n_train]
    # In absolute terms: [n_train + lags, len(U_real)]

    valid_start_idx = n_train + lags  # First valid timestep in absolute terms
    n_valid_samples = len(valid_real)  # Number of samples in validation set
    valid_end_idx = valid_start_idx + n_valid_samples


    # Get original unscaled data for the SAME time window
    U_real_valid = U_real[valid_start_idx:valid_end_idx]
    U_sim_valid = U_sim[valid_start_idx:valid_end_idx]

    # Make sure shapes match
    min_len = min(len(results['lf_hf']), len(U_real_valid))

    results['lf_hf'] = results['lf_hf'][:min_len]
    results['lf_only'] = results['lf_only'][:min_len]
    results['targets'] = results['targets'][:min_len]
    U_real_valid = U_real_valid[:min_len]
    U_sim_valid = U_sim_valid[:min_len]

    # Compute errors using ORIGINAL unscaled real data
    mse_baseline = np.mean((preds[:min_len] - U_real_valid) ** 2) if len(preds) >= min_len else np.mean(
        (preds - targets_np) ** 2)
    mse_lf = np.mean((results['lf_only'] - U_real_valid) ** 2)
    mse_total = np.mean((results['lf_hf'] - U_real_valid) ** 2)

    print(f"\n=== Results (using original unscaled data) ===")
    print(f"  LF-SHRED baseline RMSE: {np.sqrt(mse_baseline):.6f}")
    print(f"  LF-only (with GAN) RMSE: {np.sqrt(mse_lf):.6f}")
    print(f"  LF+HF RMSE: {np.sqrt(mse_total):.6f}")
    print(f"  Improvement over baseline: {(mse_baseline - mse_total) / mse_baseline * 100:.1f}%")

    # Analyze discovered frequencies
    print("\n[9] Analyzing discovered frequencies...")
    dashred.eval()
    with torch.no_grad():
        sample = next(iter(valid_loader_real))[0][:1].to(device)
        _, _, u_hf, _, _ = dashred(sample, use_gan=True)
        u_hf_np = u_hf.cpu().numpy().squeeze()

        fft_mag = np.abs(np.fft.rfft(u_hf_np))
        freqs = np.arange(len(fft_mag))

        top_5 = np.argsort(fft_mag)[-5:][::-1]
        print(f"  Top 5 frequencies in HF output: {top_5}")
        print(f"  Their magnitudes: {fft_mag[top_5]}")

    # Plot
    print("\n[10] Plotting...")

    # === FOUR-PANEL COMPARISON ON VALIDATION SET ===
    print(f"    Plotting four-panel comparison (VALIDATION set)...")
    print(f"    U_sim_valid shape: {U_sim_valid.shape}")
    print(f"    U_real_valid shape: {U_real_valid.shape}")

    dt = 0.1  # Approximate timestep

    fig_valid, axes_valid = plt.subplots(2, 2, figsize=(14, 11))

    # Color range for validation
    all_data_valid = [U_sim_valid, U_real_valid, results['lf_only'], results['lf_hf']]
    vmin_valid = min(d.min() for d in all_data_valid)
    vmax_valid = max(d.max() for d in all_data_valid)

    print(vmin_valid, vmax_valid)

    # Time axis for validation (starts after training)
    t_valid_start = (n_train + lags) * dt
    t_valid_end = t_valid_start + len(U_real_valid) * dt
    extent_valid = [t_valid_start, t_valid_end * 10, 0, N]

    # Panel 1: Simulation (Koch model)
    im1_valid = axes_valid[0, 0].imshow(U_sim_valid.T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin_valid, vmax=vmax_valid, extent=extent_valid)
    axes_valid[0, 0].set_title('(a) Simulation', fontsize=20)
    axes_valid[0, 0].set_xlabel('Time (t)', fontsize=18)
    axes_valid[0, 0].set_ylabel('Space (x)', fontsize=18)
    axes_valid[0, 0].tick_params(axis='both', labelsize=15)

    # Panel 2: Real Physics (High-fidelity)
    im2_valid = axes_valid[0, 1].imshow(U_real_valid.T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin_valid, vmax=vmax_valid, extent=extent_valid)
    axes_valid[0, 1].set_title('(b) Real Physics', fontsize=20)
    axes_valid[0, 1].set_xlabel('Time (t)', fontsize=18)
    axes_valid[0, 1].set_ylabel('Space (x)', fontsize=18)
    axes_valid[0, 1].tick_params(axis='both', labelsize=15)

    # Panel 3: LF Reconstruction (base model)
    im3_valid = axes_valid[1, 0].imshow(results['lf_only'].T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin_valid, vmax=vmax_valid, extent=extent_valid)
    axes_valid[1, 0].set_title('(c) LF Reconstruction', fontsize=20)
    axes_valid[1, 0].set_xlabel('Time (t)', fontsize=18)
    axes_valid[1, 0].set_ylabel('Space (x)', fontsize=18)
    axes_valid[1, 0].tick_params(axis='both', labelsize=15)

    # Panel 4: LF + HF Reconstruction
    im4_valid = axes_valid[1, 1].imshow(results['lf_hf'].T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin_valid, vmax=vmax_valid, extent=extent_valid)
    axes_valid[1, 1].set_title('(d) LF + HF Reconstruction', fontsize=20)
    axes_valid[1, 1].set_xlabel('Time (t)', fontsize=18)
    axes_valid[1, 1].set_ylabel('Space (x)', fontsize=18)
    axes_valid[1, 1].tick_params(axis='both', labelsize=15)

    data_range_valid = U_real_valid.max() - U_real_valid.min()

    ssim_val1_valid = ssim(U_real_valid, U_sim_valid, data_range=data_range_valid)
    print(f"SSIM between U_sim_valid predictions and high-fidelity data: {ssim_val1_valid:.6f}")
    ssim_val2_valid = ssim(U_real_valid, results['lf_only'], data_range=data_range_valid)
    print(f"SSIM between lf_only predictions and high-fidelity data: {ssim_val2_valid:.6f}")
    ssim_val3_valid = ssim(U_real_valid, results['lf_hf'], data_range=data_range_valid)
    print(f"SSIM between lf_hf predictions and high-fidelity data: {ssim_val3_valid:.6f}")

    # Adjust layout first, then add colorbar
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    # Add colorbar on the right side (outside the plots)
    cbar_ax_valid = fig_valid.add_axes([0.94, 0.12, 0.02, 0.76])
    cbar_valid = fig_valid.colorbar(im4_valid, cax=cbar_ax_valid)
    cbar_valid.ax.tick_params(labelsize=15)

    # Compute RMSE on validation data
    rmse_lf_valid = np.sqrt(mse_lf)
    rmse_total_valid = np.sqrt(mse_total)

    plt.savefig('RDE_four_panel_validation.png', dpi=150, bbox_inches='tight')
    print("    Saved: RDE_four_panel_validation.png")
    plt.close(fig_valid)

    # === DETAILED RESULTS PLOTS (VALIDATION set) ===
    print(f"    Plotting detailed results (VALIDATION)...")
    fig_valid_detail, axes_vd = plt.subplots(2, 2, figsize=(14, 11))

    idx_valid = len(U_real_valid) // 2  # Middle timestep of validation
    x_valid = np.arange(N)

    # Compute consistent y-axis limits for first three panels
    hf_true_valid = U_real_valid[idx_valid] - results['lf_only'][idx_valid]
    hf_pred_valid = results['lf_hf'][idx_valid] - results['lf_only'][idx_valid]

    # ymin/ymax for panels 1 & 2 (reconstruction plots)
    recon_data_valid = [U_real_valid[idx_valid], results['lf_only'][idx_valid], results['lf_hf'][idx_valid]]
    ymin_recon_valid = min(d.min() for d in recon_data_valid) - 0.05
    ymax_recon_valid = max(d.max() for d in recon_data_valid) + 0.05

    # ymin/ymax for panel 3 (HF component)
    hf_data_valid = [hf_true_valid, hf_pred_valid]
    ymin_hf_valid = min(d.min() for d in hf_data_valid) - 0.05
    ymax_hf_valid = max(d.max() for d in hf_data_valid) + 0.05

    # Panel 1: LF Reconstruction
    axes_vd[0, 0].plot(x_valid, U_real_valid[idx_valid], 'b-', lw=2, label='Ground Truth')
    axes_vd[0, 0].plot(x_valid, results['lf_only'][idx_valid], 'r--', lw=1.5, label='LF only')
    axes_vd[0, 0].set_title('LF Reconstruction', fontsize=20)
    axes_vd[0, 0].legend(fontsize=15)
    axes_vd[0, 0].grid(alpha=0.3)
    axes_vd[0, 0].tick_params(axis='both', labelsize=15)
    axes_vd[0, 0].set_ylim(ymin_recon_valid, ymax_recon_valid)

    # Panel 2: Full Reconstruction (LF + HF)
    axes_vd[0, 1].plot(x_valid, U_real_valid[idx_valid], 'b-', lw=2, label='Ground Truth')
    axes_vd[0, 1].plot(x_valid, results['lf_hf'][idx_valid], 'g--', lw=1.5, label='LF + HF')
    axes_vd[0, 1].set_title('Full Reconstruction (LF + HF)', fontsize=20)
    axes_vd[0, 1].legend(fontsize=15)
    axes_vd[0, 1].grid(alpha=0.3)
    axes_vd[0, 1].tick_params(axis='both', labelsize=15)
    axes_vd[0, 1].set_ylim(ymin_recon_valid, ymax_recon_valid)

    # Panel 3: HF comparison
    axes_vd[1, 0].plot(x_valid, hf_true_valid, 'b-', lw=2, label='True Residual')
    axes_vd[1, 0].plot(x_valid, hf_pred_valid, 'r--', lw=1.5, label='Predicted HF')
    axes_vd[1, 0].set_title('High Frequency Component', fontsize=20)
    axes_vd[1, 0].legend(fontsize=15)
    axes_vd[1, 0].grid(alpha=0.3)
    axes_vd[1, 0].tick_params(axis='both', labelsize=15)
    axes_vd[1, 0].set_ylim(ymin_hf_valid, ymax_hf_valid)

    # Panel 4: Frequency spectrum of HF
    hf_pred_fft_valid = np.abs(np.fft.rfft(hf_pred_valid))
    freqs_valid = np.arange(len(hf_pred_fft_valid))
    axes_vd[1, 1].stem(freqs_valid[1:30], hf_pred_fft_valid[1:30], basefmt=' ')
    axes_vd[1, 1].set_xlabel('Frequency (k)', fontsize=18)
    axes_vd[1, 1].set_ylabel('Magnitude', fontsize=18)
    axes_vd[1, 1].set_title('HF Frequency Spectrum', fontsize=20)
    axes_vd[1, 1].grid(alpha=0.3)
    axes_vd[1, 1].tick_params(axis='both', labelsize=15)

    plt.tight_layout()
    plt.savefig('sparse_freq_RDE_results_validation.png', dpi=150, bbox_inches='tight')
    print("    Saved: sparse_freq_RDE_results_validation.png")
    plt.close(fig_valid_detail)

    # === FOUR-PANEL COMPARISON ON FULL DATASET ===
    # Run model on full real data for better visualization
    print(f"    Running model on full dataset for visualization...")

    full_real_dataset = TimeSeriesDataset(U_real, sensor_indices, lags, scaler=combined_scaler)
    full_real_loader = DataLoader(full_real_dataset, batch_size=64, shuffle=False)

    full_results = {'lf_only': [], 'lf_hf': []}
    dashred.eval()
    with torch.no_grad():
        for sensors, targets in full_real_loader:
            sensors = sensors.to(device)
            z_lf = dashred.encode_lf(sensors)
            z_lf_aligned = dashred.gan(z_lf)
            u_lf = dashred.decode_lf(z_lf_aligned)
            u_total, _, _, _, _ = dashred(sensors, use_gan=True)
            full_results['lf_only'].append(u_lf.cpu())
            full_results['lf_hf'].append(u_total.cpu())

    for k in full_results:
        full_results[k] = scaler_U.inverse_transform(torch.cat(full_results[k]).numpy())

    # The full dataset starts from index `lags` (first valid index)
    U_real_full = U_real[lags:]  # Original unscaled, same length as full_results
    U_sim_full = U_sim[lags:]

    # Make sure shapes match
    n_full = min(len(full_results['lf_hf']), len(U_real_full))
    full_results['lf_hf'] = full_results['lf_hf'][:n_full]
    full_results['lf_only'] = full_results['lf_only'][:n_full]
    U_real_full = U_real_full[:n_full]
    U_sim_full = U_sim_full[:n_full]

    print(f"    Full visualization shapes: U_real_full={U_real_full.shape}, lf_hf={full_results['lf_hf'].shape}")

    # Create time axis for full data
    t_start_full = lags * dt
    time_axis = np.linspace(t_start_full, t_start_full + n_full * dt, n_full)

    print(f"    Plotting four-panel comparison...")
    fig_4panel, axes_4 = plt.subplots(2, 2, figsize=(14, 11))

    # Use consistent color range across all panels
    all_data = [U_sim_full, U_real_full, full_results['lf_only'], full_results['lf_hf']]
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)
    #print(vmin, vmax)

    extent = [time_axis[0], time_axis[-1]*10, 0, N]  # [t_min, t_max, x_min, x_max]
    #extent = [0, 250, 0, N]

    # Panel 1: Simulation (Koch model)
    im1 = axes_4[0, 0].imshow(U_sim_full.T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    axes_4[0, 0].set_title('(a) Simulation', fontsize=20)
    axes_4[0, 0].set_xlabel('Time (t)', fontsize=18)
    axes_4[0, 0].set_ylabel('Space (x)', fontsize=18)
    axes_4[0, 0].tick_params(axis='both', labelsize=15)

    # Panel 2: Real Physics (High-fidelity)
    im2 = axes_4[0, 1].imshow(U_real_full.T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    axes_4[0, 1].set_title('(b) Real Physics', fontsize=20)
    axes_4[0, 1].set_xlabel('Time (t)', fontsize=18)
    axes_4[0, 1].set_ylabel('Space (x)', fontsize=18)
    axes_4[0, 1].tick_params(axis='both', labelsize=15)

    # Panel 3: LF Reconstruction (base model)
    im3 = axes_4[1, 0].imshow(full_results['lf_only'].T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    axes_4[1, 0].set_title('(c) LF Reconstruction', fontsize=20)
    axes_4[1, 0].set_xlabel('Time (t)', fontsize=18)
    axes_4[1, 0].set_ylabel('Space (x)', fontsize=18)
    axes_4[1, 0].tick_params(axis='both', labelsize=15)

    # Panel 4: LF + HF Reconstruction
    im4 = axes_4[1, 1].imshow(full_results['lf_hf'].T, aspect='auto', cmap='RdBu_r',
                              origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    axes_4[1, 1].set_title('(d) LF + HF Reconstruction', fontsize=20)
    axes_4[1, 1].set_xlabel('Time (t)', fontsize=18)
    axes_4[1, 1].set_ylabel('Space (x)', fontsize=18)
    axes_4[1, 1].tick_params(axis='both', labelsize=15)

    data_range = U_real_full.max() - U_real_full.min()

    ssim_val1 = ssim(U_real_full, U_sim_full, data_range=data_range)
    print(f"SSIM between U_sim_full predictions and high-fidelity data: {ssim_val1:.6f}")
    ssim_val2 = ssim(U_real_full, full_results['lf_only'], data_range=data_range)
    print(f"SSIM between lf_only predictions and high-fidelity data: {ssim_val2:.6f}")
    ssim_val3 = ssim(U_real_full, full_results['lf_hf'], data_range=data_range)
    print(f"SSIM between lf_hf predictions and high-fidelity data: {ssim_val3:.6f}")

    #inspect_model_predictions(U_real_full, full_results['lf_hf'], title_prefix="")

    # Adjust layout first, then add colorbar
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    # Add colorbar on the right side (outside the plots)
    cbar_ax = fig_4panel.add_axes([0.94, 0.12, 0.02, 0.76])
    cbar = fig_4panel.colorbar(im4, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=15)

    # Compute RMSE on full data
    rmse_lf_full = np.sqrt(np.mean((full_results['lf_only'] - U_real_full) ** 2))
    rmse_total_full = np.sqrt(np.mean((full_results['lf_hf'] - U_real_full) ** 2))
    print(f"  Full dataset LF-only RMSE: {rmse_lf_full:.6f}")
    print(f"  Full dataset LF+HF RMSE: {rmse_total_full:.6f}")

    plt.savefig('RDE_four_panel_comparison.png', dpi=150, bbox_inches='tight')
    print("    Saved: RDE_four_panel_comparison.png")
    plt.close(fig_4panel)

    # === DETAILED RESULTS PLOTS (using FULL dataset) ===
    print(f"    Plotting detailed results...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    idx = n_full // 2  # Middle timestep
    x = np.arange(N)

    # Compute consistent y-axis limits for first three panels
    hf_true = U_real_full[idx] - full_results['lf_only'][idx]
    hf_pred = full_results['lf_hf'][idx] - full_results['lf_only'][idx]

    # ymin/ymax for panels 1 & 2 (reconstruction plots)
    recon_data = [U_real_full[idx], full_results['lf_only'][idx], full_results['lf_hf'][idx]]
    ymin_recon = min(d.min() for d in recon_data) - 0.05
    ymax_recon = max(d.max() for d in recon_data) + 0.05

    # ymin/ymax for panel 3 (HF component)
    hf_data = [hf_true, hf_pred]
    ymin_hf = min(d.min() for d in hf_data) - 0.05
    ymax_hf = max(d.max() for d in hf_data) + 0.05

    # Panel 1: LF Reconstruction
    axes[0, 0].plot(x, U_real_full[idx], 'b-', lw=2, label='Ground Truth')
    axes[0, 0].plot(x, full_results['lf_only'][idx], 'r--', lw=1.5, label='LF only')
    axes[0, 0].set_title('LF Reconstruction', fontsize=20)
    axes[0, 0].legend(fontsize=15)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=15)
    axes[0, 0].set_ylim(ymin_recon, ymax_recon)

    # Panel 2: Full Reconstruction (LF + HF)
    axes[0, 1].plot(x, U_real_full[idx], 'b-', lw=2, label='Ground Truth')
    axes[0, 1].plot(x, full_results['lf_hf'][idx], 'g--', lw=1.5, label='LF + HF')
    axes[0, 1].set_title('Full Reconstruction (LF + HF)', fontsize=20)
    axes[0, 1].legend(fontsize=15)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=15)
    axes[0, 1].set_ylim(ymin_recon, ymax_recon)

    # Panel 3: HF comparison
    axes[1, 0].plot(x, hf_true, 'b-', lw=2, label='True Residual')
    axes[1, 0].plot(x, hf_pred, 'r--', lw=1.5, label='Predicted HF')
    axes[1, 0].set_title('High Frequency Component', fontsize=20)
    axes[1, 0].legend(fontsize=15)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=15)
    axes[1, 0].set_ylim(ymin_hf, ymax_hf)

    # Panel 4: Frequency spectrum of HF
    axes[1, 1].stem(freqs[1:30], fft_mag[1:30], basefmt=' ')
    axes[1, 1].set_xlabel('Frequency (k)', fontsize=18)
    axes[1, 1].set_ylabel('Magnitude', fontsize=18)
    axes[1, 1].set_title('HF Frequency Spectrum', fontsize=20)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].tick_params(axis='both', labelsize=15)

    #plt.suptitle(f'Sparse-Frequency DA-SHRED on RDE Data\n'
    #             f'LF RMSE: {rmse_lf_full:.4f} → LF+HF RMSE: {rmse_total_full:.4f} '
    #             f'({(rmse_lf_full - rmse_total_full) / rmse_lf_full * 100:.1f}% improvement)', fontsize=18)
    plt.tight_layout()
    plt.savefig('sparse_freq_RDE_results.png', dpi=150, bbox_inches='tight')
    print("    Saved: sparse_freq_RDE_results.png")

    # # === ATTENTION WEIGHTS VISUALIZATION ===
    # if dashred.hf_shred.use_lag_attention:
    #     print("\n[11] Visualizing attention weights...")
    #     attn_weights = dashred.hf_shred.get_attention_weights()
    #     if attn_weights is not None:
    #         fig_attn, ax_attn = plt.subplots(1, 1, figsize=(10, 4))
    #
    #         # Average attention across batch
    #         avg_attn = attn_weights.mean(dim=0).cpu().numpy()
    #
    #         ax_attn.bar(range(len(avg_attn)), avg_attn, color='steelblue', alpha=0.7)
    #         ax_attn.set_xlabel('Lag Index (0 = oldest, {} = most recent)'.format(lags - 1), fontsize=12)
    #         ax_attn.set_ylabel('Attention Weight', fontsize=12)
    #         ax_attn.set_title('HF-SHRED Attention over Temporal Lags\n(Which timesteps matter most for HF prediction?)',
    #                           fontsize=14)
    #         ax_attn.grid(alpha=0.3)
    #
    #         # Mark which lags have highest attention
    #         top_lags = np.argsort(avg_attn)[-3:][::-1]
    #         for lag in top_lags:
    #             ax_attn.annotate(f'lag {lag}', xy=(lag, avg_attn[lag]),
    #                              xytext=(lag, avg_attn[lag] + 0.02),
    #                              ha='center', fontsize=10, color='red')
    #
    #         plt.tight_layout()
    #         plt.savefig('RDE_attention_weights.png', dpi=150, bbox_inches='tight')
    #         print("    Saved: RDE_attention_weights.png")
    #         print(f"    Top 3 attended lags: {top_lags} (0=oldest, {lags - 1}=newest)")
    #
    # plt.show()

    print("\nDone!")