"""
Training functions for SENDAI.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .models import LatentGAN
from .utils import get_device


device = get_device()


# =============================================================================
# FREQUENCY SPARSITY LOSSES
# =============================================================================

def freq_sparsity_bandlimited(signal, H, W, max_freq=16):
    """
    Bandlimited frequency sparsity loss.
    Encourages sparse in-band frequencies, heavily penalizes out-of-band.
    """
    signal_2d = signal.view(-1, H, W)
    fft = torch.fft.rfft2(signal_2d)
    mag = torch.abs(fft)
    
    ky = torch.fft.fftfreq(H, device=signal.device) * H
    kx = torch.fft.rfftfreq(W, device=signal.device) * W
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
    freq_radius = torch.sqrt(ky_grid**2 + kx_grid**2)
    
    in_band = (freq_radius <= max_freq).float()
    out_band = (freq_radius > max_freq).float()
    
    # In-band: L1/L2 sparsity
    in_mag = (mag * in_band).view(signal_2d.shape[0], -1)
    l1 = torch.sum(in_mag, dim=-1)
    l2 = torch.sqrt(torch.sum(in_mag**2, dim=-1) + 1e-8)
    sparsity = torch.mean(l1 / (l2 + 1e-8))
    
    # Out-of-band: heavy penalty
    out_mag = (mag * out_band).view(signal_2d.shape[0], -1)
    out_energy = torch.sum(out_mag**2, dim=-1)
    total_energy = torch.sum(mag.view(signal_2d.shape[0], -1)**2, dim=-1) + 1e-8
    out_ratio = out_energy / total_energy
    
    return sparsity + 100.0 * torch.mean(out_ratio)


def freq_sparsity_topk(signal, H, W, k=4):
    """Top-k sparsity: penalize energy outside top-k frequencies."""
    signal_2d = signal.view(-1, H, W)
    fft = torch.fft.rfft2(signal_2d)
    mag = torch.abs(fft).view(signal_2d.shape[0], -1)
    
    topk_vals, _ = torch.topk(mag, k, dim=-1)
    topk_energy = torch.sum(topk_vals**2, dim=-1)
    total_energy = torch.sum(mag**2, dim=-1) + 1e-8
    
    return torch.mean(1 - topk_energy / total_energy)


def freq_sparsity_combined(signal, H, W, max_freq=16, target_k=3):
    """Combined bandlimited + top-k sparsity."""
    return freq_sparsity_bandlimited(signal, H, W, max_freq) + 10.0 * freq_sparsity_topk(signal, H, W, target_k)


def freq_sparsity_with_exclusion(signal, H, W, max_freq=16, excluded_freqs=None, exclusion_radius=2.0):
    """
    Bandlimited sparsity with frequency exclusion for hierarchical peeling.
    Excludes frequencies discovered by previous layers.
    """
    signal_2d = signal.view(-1, H, W)
    fft = torch.fft.rfft2(signal_2d)
    mag = torch.abs(fft)
    
    ky = torch.fft.fftfreq(H, device=signal.device) * H
    kx = torch.fft.rfftfreq(W, device=signal.device) * W
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
    freq_radius = torch.sqrt(ky_grid**2 + kx_grid**2)
    
    out_band = (freq_radius > max_freq).float()
    
    # Exclusion mask
    exclusion_mask = torch.zeros_like(freq_radius)
    if excluded_freqs:
        for (ky_exc, kx_exc) in excluded_freqs:
            dist = torch.sqrt((ky_grid - ky_exc)**2 + (kx_grid - kx_exc)**2)
            exclusion_mask = torch.maximum(exclusion_mask, (dist < exclusion_radius).float())
    
    in_band = (1 - out_band) * (1 - exclusion_mask)
    
    # In-band sparsity
    in_mag = (mag * in_band).view(signal_2d.shape[0], -1)
    l1 = torch.sum(in_mag, dim=-1)
    l2 = torch.sqrt(torch.sum(in_mag**2, dim=-1) + 1e-8)
    sparsity = torch.mean(l1 / (l2 + 1e-8))
    
    # Penalties
    total_energy = torch.sum(mag.view(signal_2d.shape[0], -1)**2, dim=-1) + 1e-8
    
    out_mag = (mag * out_band).view(signal_2d.shape[0], -1)
    out_ratio = torch.sum(out_mag**2, dim=-1) / total_energy
    
    exc_mag = (mag * exclusion_mask).view(signal_2d.shape[0], -1)
    exc_ratio = torch.sum(exc_mag**2, dim=-1) / total_energy
    
    return sparsity + 100.0 * torch.mean(out_ratio) + 100.0 * torch.mean(exc_ratio)


def get_top_frequencies(fft_mag, H, W, top_k=6):
    """
    Extract top-k unique frequencies (removing conjugate duplicates).
    
    Returns list of dicts with ky, kx, magnitude, radius.
    """
    fft_mag = fft_mag.copy()
    fft_mag[0, 0] = 0  # Exclude DC
    
    flat_indices = np.argsort(fft_mag.ravel())[::-1]
    
    unique_freqs = []
    seen = set()
    
    for idx in flat_indices:
        ky_raw, kx_raw = np.unravel_index(idx, fft_mag.shape)
        
        ky = int(ky_raw) if ky_raw <= H // 2 else int(ky_raw) - H
        kx = int(kx_raw) if kx_raw <= W // 2 else int(kx_raw) - W
        
        # Canonical form (handle conjugate symmetry)
        if ky < 0 or (ky == 0 and kx < 0):
            canonical = (-ky, -kx)
        else:
            canonical = (ky, kx)
        
        if canonical not in seen:
            seen.add(canonical)
            unique_freqs.append({
                'ky': ky, 'kx': kx,
                'magnitude': fft_mag[ky_raw, kx_raw],
                'radius': np.sqrt(ky**2 + kx**2)
            })
        
        if len(unique_freqs) >= top_k:
            break
    
    return unique_freqs


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_shred(model, train_loader, valid_loader, epochs=300, lr=1e-4, patience=30):
    """
    Stage 1: Train SHRED on simulation with full state supervision.
    """
    print("\n=== Stage 1: Train SHRED on Simulation ===")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.to(device)
    best_loss, best_state, wait = float('inf'), None, 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sensors, state, _ in train_loader:
            sensors, state = sensors.to(device), state.to(device)
            optimizer.zero_grad()
            pred, _ = model(sensors)
            loss = F.mse_loss(pred, state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * sensors.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for sensors, state, _ in valid_loader:
                sensors, state = sensors.to(device), state.to(device)
                pred, _ = model(sensors)
                valid_loss += F.mse_loss(pred, state).item() * sensors.size(0)
        valid_loss /= len(valid_loader.dataset)
        
        scheduler.step()
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train: {train_loss:.6f}, Valid: {valid_loss:.6f}")
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_dashred(model, train_loader_sim, train_loader_real, sensor_indices,
                  epochs=300, lr=1e-4, patience=25, gan_epochs=200):
    """
    Stage 2: Train DA-SHRED with GAN alignment + full state supervision on simulation.
    """
    print("\n=== Stage 2: Train DA-SHRED (GAN + Full State on Simulation) ===")
    
    model.to(device)
    sensor_idx = torch.tensor(sensor_indices, dtype=torch.long).to(device)
    
    # 2a: Train GAN for latent alignment
    print("\n  [2a] Training GAN for latent alignment...")
    
    model.eval()
    Z_sim, Z_real = [], []
    with torch.no_grad():
        for sensors, _, _ in train_loader_sim:
            Z_sim.append(model.encode(sensors.to(device)).cpu())
        for sensors, _, _ in train_loader_real:
            Z_real.append(model.encode(sensors.to(device)).cpu())
    
    Z_sim = torch.cat(Z_sim).to(device)
    Z_real = torch.cat(Z_real).to(device)
    
    gan = LatentGAN(model.hidden_size).to(device)
    opt_g = optim.Adam(gan.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    min_samples = min(len(Z_sim), len(Z_real))
    batch_size = max(4, min(32, min_samples // 2))
    n_batches = max(1, min_samples // batch_size)
    
    g_loss_val, d_loss_val = 0.0, 0.0
    
    for epoch in range(gan_epochs):
        perm_sim = torch.randperm(len(Z_sim))
        perm_real = torch.randperm(len(Z_real))
        
        for i in range(n_batches):
            z_s = Z_sim[perm_sim[i*batch_size:(i+1)*batch_size]]
            z_r = Z_real[perm_real[i*batch_size:(i+1)*batch_size]]
            
            # Discriminator
            opt_d.zero_grad()
            z_fake = gan(z_s)
            d_loss = F.binary_cross_entropy_with_logits(
                gan.discriminator(z_r), torch.ones(len(z_r), 1, device=device)
            ) + F.binary_cross_entropy_with_logits(
                gan.discriminator(z_fake.detach()), torch.zeros(len(z_s), 1, device=device)
            )
            d_loss.backward()
            opt_d.step()
            
            # Generator
            opt_g.zero_grad()
            z_fake = gan(z_s)
            g_loss = F.binary_cross_entropy_with_logits(
                gan.discriminator(z_fake), torch.ones(len(z_s), 1, device=device)
            )
            g_loss.backward()
            opt_g.step()
            
            g_loss_val, d_loss_val = g_loss.item(), d_loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{gan_epochs}, G: {g_loss_val:.4f}, D: {d_loss_val:.4f}")
    
    # 2b: Fine-tune DA-SHRED with full state supervision on simulation
    print("\n  [2b] Fine-tuning DA-SHRED with full state supervision...")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr/2, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30)
    
    best_loss, best_state, wait = float('inf'), None, 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for sensors, full_state, _ in train_loader_sim:
            sensors, full_state = sensors.to(device), full_state.to(device)
            optimizer.zero_grad()
            
            pred, z, z_t = model(sensors, apply_transform=True)
            state_loss = F.mse_loss(pred, full_state)
            reg_loss = torch.mean((z_t - z) ** 2)
            loss = state_loss + 0.01 * reg_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += state_loss.item() * sensors.size(0)
        
        train_loss /= len(train_loader_sim.dataset)
        scheduler.step(train_loss)
        
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {train_loss:.6f}, scale: {model.scale.item():.4f}")
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_hf_layer(full_model, train_loader_real, H, W, layer_idx,
                   epochs=500, warmup=100, lr=5e-4,
                   lambda_sparse=0.05, lambda_smooth=0.1,
                   max_freq=16, target_k=None,
                   excluded_freqs=None, exclusion_radius=2.0,
                   layer_name="HF"):
    """
    Train a single HF peeling layer.
    
    Args:
        full_model: FullDASHRED model
        train_loader_real: DataLoader for real data
        H, W: spatial dimensions
        layer_idx: which HF layer to train (0-indexed)
        epochs: training epochs
        warmup: epochs before applying sparsity
        lr: learning rate
        lambda_sparse: sparsity weight
        lambda_smooth: smoothness weight
        max_freq: maximum frequency radius
        target_k: number of modes to discover (None = bandlimited only)
        excluded_freqs: list of (ky, kx) to exclude
        exclusion_radius: exclusion zone radius
        layer_name: name for logging
    
    Returns:
        model, history, discovered_freqs
    """
    use_inr = full_model.use_inr
    print(f"\n=== Training {layer_name} (Layer {layer_idx}) {'[INR]' if use_inr else ''} ===")
    
    # Select sparsity function
    if target_k is not None:
        if excluded_freqs:
            sparsity_fn = lambda x: (freq_sparsity_with_exclusion(x, H, W, max_freq, excluded_freqs, exclusion_radius) +
                                     10.0 * freq_sparsity_topk(x, H, W, target_k))
        else:
            sparsity_fn = lambda x: freq_sparsity_combined(x, H, W, max_freq, target_k)
    else:
        if excluded_freqs:
            sparsity_fn = lambda x: freq_sparsity_with_exclusion(x, H, W, max_freq, excluded_freqs, exclusion_radius)
        else:
            sparsity_fn = lambda x: freq_sparsity_bandlimited(x, H, W, max_freq)
    
    # Only train current HF layer
    params = list(full_model.hf_layers[layer_idx].parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)
    
    sensor_idx = full_model.sensor_indices
    
    # Estimate residual scale
    full_model.eval()
    with torch.no_grad():
        sample = next(iter(train_loader_real))
        sensors, current = sample[0].to(device), sample[2].to(device)
        u_da, _, _ = full_model.forward_da_only(sensors)
        residual = current - u_da[:, sensor_idx]
        residual_scale = residual.abs().mean().item()
    
    max_hf_mag = residual_scale * 5
    history = {'sensor_loss': [], 'sparsity_loss': [], 'smooth_loss': []}
    
    full_model.to(device)
    
    # Freeze all except current HF layer
    for p in full_model.lstm.parameters():
        p.requires_grad = False
    for p in full_model.decoder.parameters():
        p.requires_grad = False
    for p in full_model.transform.parameters():
        p.requires_grad = False
    for i, hf in enumerate(full_model.hf_layers):
        for p in hf.parameters():
            p.requires_grad = (i == layer_idx)
    
    for epoch in range(epochs):
        full_model.train()
        
        # Gradual sparsity warmup
        if epoch < warmup:
            curr_lambda = 0.0
            curr_smooth = lambda_smooth * 0.5
        else:
            progress = (epoch - warmup) / (epochs - warmup)
            curr_lambda = lambda_sparse * min(1.0, progress * 2)
            curr_smooth = lambda_smooth
        
        epoch_loss = {'sensor': 0, 'sparsity': 0, 'smooth': 0}
        
        for sensors, _, current in train_loader_real:
            sensors, current = sensors.to(device), current.to(device)
            optimizer.zero_grad()
            
            # Get cumulative prediction through previous layers
            with torch.no_grad():
                u_cumulative, _, _ = full_model.forward_da_only(sensors)
                for i in range(layer_idx):
                    res = current - u_cumulative[:, sensor_idx]
                    u_cumulative = u_cumulative + full_model.hf_layers[i](res)
            
            # Compute residual for this layer
            residual = current - u_cumulative[:, sensor_idx].detach()
            
            # Apply HF layer
            hf_layer = full_model.hf_layers[layer_idx]
            u_hf = hf_layer(residual)
            
            # Losses
            sensor_loss = F.mse_loss(u_hf[:, sensor_idx], residual)
            sparsity_loss = sparsity_fn(u_hf)
            mag_penalty = F.relu(u_hf.abs().mean() - max_hf_mag)**2
            
            if use_inr and hasattr(hf_layer, 'smoothness_loss'):
                smooth_loss = hf_layer.smoothness_loss(u_hf)
            else:
                u_2d = u_hf.view(-1, H, W)
                dx = u_2d[:, :, 1:] - u_2d[:, :, :-1]
                dy = u_2d[:, 1:, :] - u_2d[:, :-1, :]
                smooth_loss = torch.mean(dx**2) + torch.mean(dy**2)
            
            loss = sensor_loss + curr_lambda * sparsity_loss + 0.5 * mag_penalty + curr_smooth * smooth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss['sensor'] += sensor_loss.item()
            epoch_loss['sparsity'] += sparsity_loss.item()
            epoch_loss['smooth'] += smooth_loss.item()
        
        n = len(train_loader_real)
        for k in epoch_loss:
            epoch_loss[k] /= n
            history[f'{k}_loss'].append(epoch_loss[k])
        
        scheduler.step(epoch_loss['sensor'])
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Sensor: {epoch_loss['sensor']:.6f}, "
                  f"Sparsity: {epoch_loss['sparsity']:.4f}, scale: {hf_layer.scale.item():.4f}")
    
    # Extract discovered frequencies
    full_model.eval()
    with torch.no_grad():
        sample = next(iter(train_loader_real))
        sensors, current = sample[0][:1].to(device), sample[2][:1].to(device)
        
        u_cumulative, _, _ = full_model.forward_da_only(sensors)
        for i in range(layer_idx):
            res = current - u_cumulative[:, sensor_idx]
            u_cumulative = u_cumulative + full_model.hf_layers[i](res)
        
        residual = current - u_cumulative[:, sensor_idx]
        u_hf = full_model.hf_layers[layer_idx](residual)
        
        fft_mag = np.abs(np.fft.fft2(u_hf.view(H, W).cpu().numpy()))
        num_modes = target_k if target_k else 4
        top_freqs = get_top_frequencies(fft_mag, H, W, num_modes)
        
        discovered = [(f['ky'], f['kx'], f['magnitude'], f['radius']) for f in top_freqs]
        
        print(f"\n  {layer_name} discovered {len(discovered)} modes:")
        for i, f in enumerate(top_freqs):
            in_band = "OK" if f['radius'] <= max_freq else "OUT"
            print(f"    {i+1}. (ky={f['ky']:3d}, kx={f['kx']:3d}), mag={f['magnitude']:.2f}, r={f['radius']:.1f} [{in_band}]")
    
    return full_model, history, discovered


def train_hf_finetune(full_model, train_loader_real, H, W, layer_idx,
                      epochs=200, lr=1e-4,
                      lambda_sparse=0.005, lambda_smooth=0.05,
                      max_freq=16, target_k=None,
                      excluded_freqs=None, exclusion_radius=2.0,
                      layer_name="HF"):
    """Fine-tune HF layer with reduced sparsity."""
    use_inr = full_model.use_inr
    print(f"\n  Fine-tuning {layer_name} (reduced sparsity) {'[INR]' if use_inr else ''}")
    
    if target_k is not None:
        if excluded_freqs:
            sparsity_fn = lambda x: (freq_sparsity_with_exclusion(x, H, W, max_freq, excluded_freqs, exclusion_radius) +
                                     10.0 * freq_sparsity_topk(x, H, W, target_k))
        else:
            sparsity_fn = lambda x: freq_sparsity_combined(x, H, W, max_freq, target_k)
    else:
        if excluded_freqs:
            sparsity_fn = lambda x: freq_sparsity_with_exclusion(x, H, W, max_freq, excluded_freqs, exclusion_radius)
        else:
            sparsity_fn = lambda x: freq_sparsity_bandlimited(x, H, W, max_freq)
    
    params = list(full_model.hf_layers[layer_idx].parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30)
    
    sensor_idx = full_model.sensor_indices
    
    # Estimate max magnitude
    full_model.eval()
    with torch.no_grad():
        sample = next(iter(train_loader_real))
        sensors, current = sample[0].to(device), sample[2].to(device)
        u_da, _, _ = full_model.forward_da_only(sensors)
        residual = current - u_da[:, sensor_idx]
        max_hf_mag = residual.abs().mean().item() * 5
    
    history = {'sensor_loss': [], 'sparsity_loss': [], 'smooth_loss': []}
    
    # Freeze all except current HF layer
    for p in full_model.lstm.parameters():
        p.requires_grad = False
    for p in full_model.decoder.parameters():
        p.requires_grad = False
    for p in full_model.transform.parameters():
        p.requires_grad = False
    for i, hf in enumerate(full_model.hf_layers):
        for p in hf.parameters():
            p.requires_grad = (i == layer_idx)
    
    for epoch in range(epochs):
        full_model.train()
        epoch_loss = {'sensor': 0, 'sparsity': 0, 'smooth': 0}
        
        for sensors, _, current in train_loader_real:
            sensors, current = sensors.to(device), current.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                u_cumulative, _, _ = full_model.forward_da_only(sensors)
                for i in range(layer_idx):
                    res = current - u_cumulative[:, sensor_idx]
                    u_cumulative = u_cumulative + full_model.hf_layers[i](res)
            
            residual = current - u_cumulative[:, sensor_idx].detach()
            
            hf_layer = full_model.hf_layers[layer_idx]
            u_hf = hf_layer(residual)
            
            sensor_loss = F.mse_loss(u_hf[:, sensor_idx], residual)
            sparsity_loss = sparsity_fn(u_hf)
            mag_penalty = F.relu(u_hf.abs().mean() - max_hf_mag)**2
            
            if use_inr and hasattr(hf_layer, 'smoothness_loss'):
                smooth_loss = hf_layer.smoothness_loss(u_hf)
            else:
                u_2d = u_hf.view(-1, H, W)
                dx = u_2d[:, :, 1:] - u_2d[:, :, :-1]
                dy = u_2d[:, 1:, :] - u_2d[:, :-1, :]
                smooth_loss = torch.mean(dx**2) + torch.mean(dy**2)
            
            loss = sensor_loss + lambda_sparse * sparsity_loss + 0.5 * mag_penalty + lambda_smooth * smooth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss['sensor'] += sensor_loss.item()
            epoch_loss['sparsity'] += sparsity_loss.item()
            epoch_loss['smooth'] += smooth_loss.item()
        
        n = len(train_loader_real)
        for k in epoch_loss:
            epoch_loss[k] /= n
            history[f'{k}_loss'].append(epoch_loss[k])
        
        scheduler.step(epoch_loss['sensor'])
        
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Sensor: {epoch_loss['sensor']:.6f}")
    
    # Final frequency analysis
    full_model.eval()
    with torch.no_grad():
        sample = next(iter(train_loader_real))
        sensors, current = sample[0][:1].to(device), sample[2][:1].to(device)
        
        u_cumulative, _, _ = full_model.forward_da_only(sensors)
        for i in range(layer_idx):
            res = current - u_cumulative[:, sensor_idx]
            u_cumulative = u_cumulative + full_model.hf_layers[i](res)
        
        residual = current - u_cumulative[:, sensor_idx]
        u_hf = full_model.hf_layers[layer_idx](residual)
        
        fft_mag = np.abs(np.fft.fft2(u_hf.view(H, W).cpu().numpy()))
        num_modes = target_k if target_k else 4
        top_freqs = get_top_frequencies(fft_mag, H, W, num_modes)
        
        final_freqs = [(f['ky'], f['kx'], f['magnitude'], f['radius']) for f in top_freqs]
    
    return full_model, history, final_freqs


def train_hierarchical_hf(full_model, train_loader_real, H, W, peel_config,
                          max_freq=16, exclusion_radius=2.0,
                          lambda_smooth=0.1, smooth_type='laplacian',
                          warmup=100, base_lr=1e-4):
    """
    Train hierarchical HF peeling layers sequentially.
    
    Args:
        full_model: FullDASHRED model
        train_loader_real: DataLoader for real data
        H, W: spatial dimensions
        peel_config: list of dicts with layer configs
        max_freq: maximum frequency radius
        exclusion_radius: exclusion zone radius
        lambda_smooth: smoothness weight
        smooth_type: 'gradient' or 'laplacian'
        warmup: warmup epochs
        base_lr: base learning rate
    
    Returns:
        model, all_histories, all_discovered_freqs
    """
    use_inr = full_model.use_inr
    
    print("\n" + "="*70)
    print(f"HIERARCHICAL HF PEELING {'[INR]' if use_inr else ''}")
    print("="*70)
    
    all_histories = {}
    all_discovered_freqs = {}
    accumulated_excluded = []
    
    for layer_idx, config in enumerate(peel_config):
        layer_name = config.get('name', f'HF_Peel{layer_idx + 1}')
        target_k = config.get('target_k', None)
        epochs = config.get('epochs', 500)
        lambda_sparse = config.get('lambda_sparse', 0.05)
        finetune_epochs = config.get('finetune_epochs', 200)
        finetune_lambda = config.get('finetune_lambda', 0.005)
        
        print(f"\n{'='*70}")
        print(f"LAYER {layer_idx + 1}: {layer_name}")
        print(f"{'='*70}")
        
        # Main training
        full_model, history, discovered = train_hf_layer(
            full_model, train_loader_real, H, W, layer_idx,
            epochs=epochs, warmup=warmup, lr=base_lr*5,
            lambda_sparse=lambda_sparse, lambda_smooth=lambda_smooth,
            max_freq=max_freq, target_k=target_k,
            excluded_freqs=accumulated_excluded if layer_idx > 0 else None,
            exclusion_radius=exclusion_radius,
            layer_name=layer_name
        )
        
        # Fine-tuning
        full_model, history2, final_freqs = train_hf_finetune(
            full_model, train_loader_real, H, W, layer_idx,
            epochs=finetune_epochs, lr=base_lr,
            lambda_sparse=finetune_lambda, lambda_smooth=lambda_smooth * 0.5,
            max_freq=max_freq, target_k=target_k,
            excluded_freqs=accumulated_excluded if layer_idx > 0 else None,
            exclusion_radius=exclusion_radius,
            layer_name=layer_name
        )
        
        # Merge histories
        for key in history:
            if key in history2:
                history[key].extend(history2[key])
        
        all_histories[layer_name] = history
        all_discovered_freqs[layer_name] = final_freqs
        
        # Store and accumulate
        freq_tuples = [(f[0], f[1]) for f in final_freqs]
        full_model.set_discovered_freqs(layer_idx, freq_tuples)
        accumulated_excluded.extend(freq_tuples)
    
    # Summary
    print("\n" + "="*70)
    print("PEELING SUMMARY")
    print("="*70)
    for layer_name, freqs in all_discovered_freqs.items():
        print(f"\n  {layer_name}:")
        for i, (ky, kx, mag, radius) in enumerate(freqs):
            status = "in-band" if radius <= max_freq else "OUT"
            print(f"    Mode {i+1}: (ky={ky:3d}, kx={kx:3d}), mag={mag:.2f}, r={radius:.1f} [{status}]")
    
    return full_model, all_histories, all_discovered_freqs


def evaluate(model, dataset, scaler_state, use_hf=True):
    """
    Evaluate model on dataset.
    
    Returns:
        results: dict with 'targets', 'da', 'total', 'hf' in original scale
        rmse_da, rmse_total
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=64)
    
    results = {'da': [], 'total': [], 'hf': [], 'targets': []}
    
    with torch.no_grad():
        for sensors, state, current in loader:
            sensors, current = sensors.to(device), current.to(device)
            
            if use_hf:
                u_total, u_da, u_hf, _, _ = model(sensors, current, use_hf=True)
            else:
                u_da, _, _ = model.forward_da_only(sensors)
                u_total = u_da
                u_hf = torch.zeros_like(u_da)
            
            results['da'].append(u_da.cpu().numpy())
            results['total'].append(u_total.cpu().numpy())
            results['hf'].append(u_hf.cpu().numpy())
            results['targets'].append(state.numpy())
    
    for k in results:
        results[k] = np.vstack(results[k])
    
    # Inverse transform
    results['targets'] = scaler_state.inverse_transform(results['targets'])
    results['da'] = scaler_state.inverse_transform(results['da'])
    results['total'] = scaler_state.inverse_transform(results['total'])
    results['hf'] = results['hf'] * scaler_state.scale_
    
    rmse_da = np.sqrt(np.mean((results['da'] - results['targets'])**2))
    rmse_total = np.sqrt(np.mean((results['total'] - results['targets'])**2))
    
    return results, rmse_da, rmse_total
