"""
Training functions for Flux Field Prediction

All baseline models predict vectors (B, N, 3), which are then converted
to flux (B, E) using TorchFluxMapper for loss computation.

Spectral models (HSD_base, HSD) predict flux coefficients directly.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import EarlyStopping, LpLoss


# ==========================================
# Baseline Training Functions
# ==========================================

def train_gno(model, data_mgr, torch_mapper, X_train, Y_train, X_val, Y_val,
              epochs, cfg, logger):
    """
    Train GNO with flux loss.
    
    Args:
        model: GNO model
        data_mgr: DataManager instance
        torch_mapper: TorchFluxMapper for vector->flux conversion
        X_train: (B, N) input scalar field
        Y_train: (B, N, 3) target vector field
        X_val, Y_val: validation data
        epochs: number of epochs
        cfg: config object
        logger: logger instance
    """
    logger.log("\n" + "-"*50)
    logger.log("Training GNO (Graph Neural Operator)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    # Precompute target flux
    with torch.no_grad():
        Y_train_flux = torch_mapper(Y_train)
        Y_val_flux = torch_mapper(Y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, Y_train_flux)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_vec_batch, y_flux_batch in train_loader:
            optimizer.zero_grad()
            gno_in, _ = data_mgr.prepare_gno_batch(x_batch, None)
            pred_vec = model(gno_in, data_mgr.pts)  # (B, 3, N)
            pred_vec = pred_vec.permute(0, 2, 1)    # (B, N, 3)
            
            # Convert to flux for loss
            pred_flux = torch_mapper(pred_vec)
            loss = criterion(pred_flux, y_flux_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_in, _ = data_mgr.prepare_gno_batch(X_val, None)
            val_pred_vec = model(val_in, data_mgr.pts).permute(0, 2, 1)
            val_pred_flux = torch_mapper(val_pred_vec)
            val_loss = criterion(val_pred_flux, Y_val_flux).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if torch.cuda.is_available():
            total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            fetch_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

        if early_stop(val_loss):
            logger.log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_time = time.time() - t0
    avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

    logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")

    return model, train_losses, val_losses, avg_vram, elapsed_time


def train_fno(model, data_mgr, torch_mapper, X_train, Y_train, X_val, Y_val,
              epochs, cfg, logger):
    """Train FNO with flux loss."""
    logger.log("\n" + "-"*50)
    logger.log("Training FNO (Fourier Neural Operator)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    batch_size_fno = min(cfg.BATCH_SIZE, 32)

    # Precompute target flux
    with torch.no_grad():
        Y_train_flux = torch_mapper(Y_train)
        Y_val_flux = torch_mapper(Y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, Y_train_flux)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_fno, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_vec_batch, y_flux_batch in train_loader:
            optimizer.zero_grad()
            fno_in, _ = data_mgr.prepare_fno_batch(x_batch, None)
            pred_grid = model(fno_in)
            pred_vec = data_mgr.decode_fno_output(pred_grid)  # (B, N, 3)
            
            pred_flux = torch_mapper(pred_vec)
            loss = criterion(pred_flux, y_flux_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, X_val.shape[0], batch_size_fno):
                x_b = X_val[i:i+batch_size_fno]
                fno_in, _ = data_mgr.prepare_fno_batch(x_b, None)
                pred_grid = model(fno_in)
                pred_vec = data_mgr.decode_fno_output(pred_grid)
                pred_flux = torch_mapper(pred_vec)
                val_preds.append(pred_flux)
            val_pred_flux = torch.cat(val_preds, dim=0)
            val_loss = criterion(val_pred_flux, Y_val_flux).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if torch.cuda.is_available():
            total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            fetch_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

        if early_stop(val_loss):
            logger.log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_time = time.time() - t0
    avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

    logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")

    return model, train_losses, val_losses, avg_vram, elapsed_time


def train_mgn(model, data_mgr, torch_mapper, X_train, Y_train, X_val, Y_val,
              epochs, cfg, device, logger):
    """Train MGN with flux loss."""
    logger.log("\n" + "-"*50)
    logger.log("Training MGN (MeshGraphNets)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    # Precompute target flux
    with torch.no_grad():
        Y_train_flux = torch_mapper(Y_train)
        Y_val_flux = torch_mapper(Y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, Y_train_flux)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0
    
    edge_index = data_mgr.edge_index
    edge_attr = data_mgr.edge_attr

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_vec_batch, y_flux_batch in train_loader:
            optimizer.zero_grad()
            
            B = x_batch.shape[0]
            batch_preds = []
            
            for b in range(B):
                node_feat = torch.cat([x_batch[b:b+1].T, data_mgr.pts], dim=1)
                pred = model(node_feat, edge_index, edge_attr)  # (N, 3)
                batch_preds.append(pred)
            
            pred_vec = torch.stack(batch_preds)  # (B, N, 3)
            pred_flux = torch_mapper(pred_vec)
            
            loss = criterion(pred_flux, y_flux_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_preds = []
            for b in range(X_val.shape[0]):
                node_feat = torch.cat([X_val[b:b+1].T, data_mgr.pts], dim=1)
                pred = model(node_feat, edge_index, edge_attr)
                val_preds.append(pred)
            val_pred_vec = torch.stack(val_preds)
            val_pred_flux = torch_mapper(val_pred_vec)
            val_loss = criterion(val_pred_flux, Y_val_flux).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if torch.cuda.is_available():
            total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            fetch_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

        if early_stop(val_loss):
            logger.log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_time = time.time() - t0
    avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

    logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")

    return model, train_losses, val_losses, avg_vram, elapsed_time


def train_deeponet(model, data_mgr, torch_mapper, X_train, Y_train, X_val, Y_val,
                   epochs, cfg, logger):
    """Train DeepONet with flux loss."""
    logger.log("\n" + "-"*50)
    logger.log("Training DeepONet")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    # Precompute target flux
    with torch.no_grad():
        Y_train_flux = torch_mapper(Y_train)
        Y_val_flux = torch_mapper(Y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, Y_train_flux)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0
    
    trunk_input = data_mgr.pts

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_vec_batch, y_flux_batch in train_loader:
            optimizer.zero_grad()
            
            pred_vec = model(x_batch, trunk_input)  # (B, N, 3)
            pred_flux = torch_mapper(pred_vec)
            
            loss = criterion(pred_flux, y_flux_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred_vec = model(X_val, trunk_input)
            val_pred_flux = torch_mapper(val_pred_vec)
            val_loss = criterion(val_pred_flux, Y_val_flux).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if torch.cuda.is_available():
            total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            fetch_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

        if early_stop(val_loss):
            logger.log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_time = time.time() - t0
    avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

    logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")

    return model, train_losses, val_losses, avg_vram, elapsed_time


def train_geofno(model, data_mgr, torch_mapper, X_train, Y_train, X_val, Y_val,
                 epochs, cfg, logger):
    """Train GeoFNO with flux loss."""
    logger.log("\n" + "-"*50)
    logger.log("Training GeoFNO (Geometry-Adaptive FNO)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    batch_size = min(cfg.BATCH_SIZE, 32)

    # Precompute target flux
    with torch.no_grad():
        Y_train_flux = torch_mapper(Y_train)
        Y_val_flux = torch_mapper(Y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, Y_train_flux)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_vec_batch, y_flux_batch in train_loader:
            optimizer.zero_grad()
            
            x_coords, features = data_mgr.prepare_geofno_batch(x_batch)
            pred_vec = model(x_coords, features)  # (B, N, 3)
            pred_flux = torch_mapper(pred_vec)
            
            loss = criterion(pred_flux, y_flux_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, X_val.shape[0], batch_size):
                x_b = X_val[i:i+batch_size]
                x_coords, features = data_mgr.prepare_geofno_batch(x_b)
                pred_vec = model(x_coords, features)
                pred_flux = torch_mapper(pred_vec)
                val_preds.append(pred_flux)
            val_pred_flux = torch.cat(val_preds, dim=0)
            val_loss = criterion(val_pred_flux, Y_val_flux).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        if torch.cuda.is_available():
            total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            fetch_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

        if early_stop(val_loss):
            logger.log(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_time = time.time() - t0
    avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

    logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")

    return model, train_losses, val_losses, avg_vram, elapsed_time


# # ==========================================
# # Spectral Model Training Functions
# # ==========================================

# def train_HSD_base(model, train_loader, test_loader, Phi1_torch, epochs, cfg, device, logger):
#     """
#     Training for pure spectral HSD_base model.
    
#     Model predicts c1 coefficients, which are reconstructed to flux via Phi1.
#     """
    
#     logger.log("\n" + "-"*50)
#     logger.log("Training HSD_base (Spectral Flux Operator)")
#     logger.log("-" * 50)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5
#     )
    
#     criterion_rel = LpLoss(size_average=True)

#     train_losses, val_losses = [], []
#     best_val_loss = float("inf")
#     best_state = None

#     t0 = time.time()
#     total_vram_mb = 0.0
#     fetch_count = 0

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         n_batches = 0

#         for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in train_loader:
#             c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
#             gt_flux = gt_flux.to(device)

#             optimizer.zero_grad()

#             # Model returns c1 coefficients
#             pred_c1 = model(c0, c1, c2)
#             pred_flux = torch.matmul(pred_c1, Phi1_torch.t())

#             loss = criterion_rel(pred_flux, gt_flux)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             total_loss += loss.item()
#             n_batches += 1

#         avg_train_loss = total_loss / n_batches
#         train_losses.append(avg_train_loss)

#         model.eval()
#         total_val_loss = 0.0
#         n_val_batches = 0
        
#         with torch.no_grad():
#             for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in test_loader:
#                 c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
#                 gt_flux = gt_flux.to(device)

#                 pred_c1 = model(c0, c1, c2)
#                 pred_flux = torch.matmul(pred_c1, Phi1_torch.t())

#                 loss = criterion_rel(pred_flux, gt_flux)
#                 total_val_loss += loss.item()
#                 n_val_batches += 1

#         avg_val_loss = total_val_loss / n_val_batches
#         val_losses.append(avg_val_loss)

#         scheduler.step(avg_val_loss)

#         if torch.cuda.is_available():
#             total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
#             fetch_count += 1

#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

#     if best_state is not None:
#         model.load_state_dict(best_state)
    
#     elapsed_time = time.time() - t0
#     avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

#     logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")

#     return model, train_losses, val_losses, avg_vram, elapsed_time


# def train_HSD(model, train_loader, test_loader, epochs, cfg, device, logger):
#     """
#     Training for HSD spectral + FNO hybrid model.
    
#     Model outputs flux directly (combines spectral base + FNO residual).
#     """
    
#     logger.log("\n" + "-"*50)
#     logger.log("Training HSD Model (Spectral + Residual FNO)")
#     logger.log("-" * 50)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
#     criterion_rel = LpLoss(size_average=True)
#     criterion_l1 = nn.L1Loss(reduction='mean')
    
#     LAMBDA_L1 = 1e-2  # Regularization weight for residual sparsity

#     train_losses, val_losses = [], []
#     best_val_loss = float("inf")
#     best_state = None

#     t0 = time.time()
#     total_vram_mb = 0.0
#     fetch_count = 0

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         n_batches = 0

#         for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in train_loader:
#             c0 = c0.to(device)
#             c1 = c1.to(device)
#             c2 = c2.to(device)
#             gt_flux = gt_flux.to(device)
#             x_spatial = x_spatial.to(device)

#             optimizer.zero_grad()

#             flux_total, flux_base, flux_res = model(c0, c1, c2, x_spatial)

#             loss_main = criterion_rel(flux_total, gt_flux)
#             # L1 regularization on residual to encourage spectral dominance
#             loss_sparsity = criterion_l1(flux_res, torch.zeros_like(flux_res))
            
#             loss = loss_main + LAMBDA_L1 * loss_sparsity

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             total_loss += loss.item()
#             n_batches += 1

#         avg_train_loss = total_loss / n_batches
#         train_losses.append(avg_train_loss)

#         model.eval()
#         total_val_loss = 0.0
#         n_val_batches = 0
        
#         with torch.no_grad():
#             for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in train_loader:
#                 c0 = c0.to(device)
#                 c1 = c1.to(device)
#                 c2 = c2.to(device)
#                 gt_flux = gt_flux.to(device)
#                 x_spatial = x_spatial.to(device)

#                 flux_total, _, _ = model(c0, c1, c2, x_spatial)

#                 loss = criterion_rel(flux_total, gt_flux)
#                 total_val_loss += loss.item()
#                 n_val_batches += 1

#         avg_val_loss = total_val_loss / n_val_batches
#         val_losses.append(avg_val_loss)

#         scheduler.step()

#         if torch.cuda.is_available():
#             total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
#             fetch_count += 1

#         if (epoch + 1) % 10 == 0 or epoch == 0:
#             res_scale = model.res_scale.item()
#             logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | ResScale: {res_scale:.4f}")

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

#     if best_state is not None:
#         model.load_state_dict(best_state)

#     elapsed_time = time.time() - t0
#     avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

#     logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")
#     logger.log(f"  Final Residual Scale: {model.res_scale.item():.4f}")

#     return model, train_losses, val_losses, avg_vram, elapsed_time


# ==========================================
# Spectral Model Training Functions
# ==========================================

def train_HSD_base(model, train_loader, test_loader, Phi1_torch, epochs, cfg, device, logger, host_ops):
    """
    Training for pure spectral HSD_base model.
    
    Model predicts c1 coefficients, which are reconstructed to flux via Phi1.
    Includes divergence supervision via c0_tgt (computed in real-time).
    
    Args:
        host_ops: HighOrderSpectralOperators instance (for B1 and Phi0)
    """
    
    logger.log("\n" + "-"*50)
    logger.log("Training HSD_base (Spectral Flux Operator)")
    logger.log("-" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    criterion_rel = LpLoss(size_average=True)

    # === 准备计算 c0_tgt 需要的算子 (在循环外准备好，节省时间) ===
    # B1 形状是 (Edges, Nodes)，我们需要 B1.T (Nodes, Edges) 来计算散度
    B1_T_scipy = host_ops.B1.T.tocoo()
    indices = torch.from_numpy(np.vstack((B1_T_scipy.row, B1_T_scipy.col))).long()
    values = torch.from_numpy(B1_T_scipy.data.astype(np.float32))
    shape = B1_T_scipy.shape
    B1_T_torch = torch.sparse_coo_tensor(indices, values, shape, device=device)
    
    # Phi0 用于投影到 spectral space
    Phi0_torch = torch.from_numpy(host_ops.Phi0[:, :cfg.K_EIGENS].astype(np.float32)).to(device)
    
    # 散度损失权重 (可调整，通常 0.1 ~ 1.0)
    LAMBDA_DIV = 0.5

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in train_loader:
            c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
            gt_flux = gt_flux.to(device)

            optimizer.zero_grad()

            # Model returns c1 coefficients and pred_alpha
            pred_c1, pred_alpha = model(c0, c1, c2)
            pred_flux = torch.matmul(pred_c1, Phi1_torch.t())

            # 主 Loss: Flux 重建
            loss_flux = criterion_rel(pred_flux, gt_flux)
            
            # === 实时计算 c0_tgt ===
            with torch.no_grad():
                # 1. 计算物理空间的散度: div = B1.T @ flux
                # (Nodes, Edges) @ (Edges, Batch) -> (Nodes, Batch) -> transpose -> (Batch, Nodes)
                div_physical = torch.sparse.mm(B1_T_torch, gt_flux.T).T  # (Batch, Nodes)
                
                # 2. 投影到 Spectral 空间: c0_tgt = div @ Phi0
                # (Batch, Nodes) @ (Nodes, k0) -> (Batch, k0)
                c0_tgt = torch.matmul(div_physical, Phi0_torch)
            
            # 散度 Loss: 监督 pred_alpha (gradient potential)
            loss_div = criterion_rel(pred_alpha, c0_tgt)
            
            # 总 Loss
            loss = loss_flux + LAMBDA_DIV * loss_div
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in test_loader:
                c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
                gt_flux = gt_flux.to(device)

                pred_c1, _ = model(c0, c1, c2)
                pred_flux = torch.matmul(pred_c1, Phi1_torch.t())

                loss = criterion_rel(pred_flux, gt_flux)
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if torch.cuda.is_available():
            total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            fetch_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    
    elapsed_time = time.time() - t0
    avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

    logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")

    return model, train_losses, val_losses, avg_vram, elapsed_time


def train_HSD(model, train_loader, test_loader, epochs, cfg, device, logger, host_ops):
    """
    Training for HSD spectral + FNO hybrid model.
    
    Model outputs flux directly (combines spectral base + FNO residual).
    Includes divergence supervision via c0_tgt (computed in real-time).
    
    Args:
        host_ops: HighOrderSpectralOperators instance (for B1 and Phi0)
    """
    
    logger.log("\n" + "-"*50)
    logger.log("Training HSD Model (Spectral + Residual FNO)")
    logger.log("-" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion_rel = LpLoss(size_average=True)
    criterion_l1 = nn.L1Loss(reduction='mean')
    
    LAMBDA_L1 = 1e-2   # Regularization weight for residual sparsity
    LAMBDA_DIV = 0.5   # 散度损失权重

    # === 准备计算 c0_tgt 需要的算子 (在循环外准备好，节省时间) ===
    B1_T_scipy = host_ops.B1.T.tocoo()
    indices = torch.from_numpy(np.vstack((B1_T_scipy.row, B1_T_scipy.col))).long()
    values = torch.from_numpy(B1_T_scipy.data.astype(np.float32))
    shape = B1_T_scipy.shape
    B1_T_torch = torch.sparse_coo_tensor(indices, values, shape, device=device)
    
    Phi0_torch = torch.from_numpy(host_ops.Phi0[:, :cfg.K_EIGENS].astype(np.float32)).to(device)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in train_loader:
            c0 = c0.to(device)
            c1 = c1.to(device)
            c2 = c2.to(device)
            gt_flux = gt_flux.to(device)
            x_spatial = x_spatial.to(device)

            optimizer.zero_grad()

            # Model returns flux and pred_alpha
            flux_total, flux_base, flux_res, pred_alpha = model(c0, c1, c2, x_spatial)

            # 主 Loss: Flux 重建
            loss_main = criterion_rel(flux_total, gt_flux)
            
            # L1 regularization on residual to encourage spectral dominance
            loss_sparsity = criterion_l1(flux_res, torch.zeros_like(flux_res))
            
            # === 实时计算 c0_tgt ===
            with torch.no_grad():
                div_physical = torch.sparse.mm(B1_T_torch, gt_flux.T).T  # (Batch, Nodes)
                c0_tgt = torch.matmul(div_physical, Phi0_torch)  # (Batch, k0)
            
            # 散度 Loss: 监督 pred_alpha
            loss_div = criterion_rel(pred_alpha, c0_tgt)
            
            # 总 Loss
            loss = loss_main + LAMBDA_L1 * loss_sparsity + LAMBDA_DIV * loss_div

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation (注意：原代码误用 train_loader，这里改为 test_loader)
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for c0, c1, c2, c1_tgt, gt_flux, x_spatial, y_vec in test_loader:
                c0 = c0.to(device)
                c1 = c1.to(device)
                c2 = c2.to(device)
                gt_flux = gt_flux.to(device)
                x_spatial = x_spatial.to(device)

                flux_total, _, _, _ = model(c0, c1, c2, x_spatial)

                loss = criterion_rel(flux_total, gt_flux)
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        scheduler.step()

        if torch.cuda.is_available():
            total_vram_mb += torch.cuda.memory_allocated() / (1024 ** 2)
            fetch_count += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            res_scale = model.res_scale.item()
            logger.log(f"  Epoch {epoch+1:3d}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | ResScale: {res_scale:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_time = time.time() - t0
    avg_vram = (total_vram_mb / fetch_count) if fetch_count > 0 else 0.0

    logger.log(f"  Training time: {elapsed_time:.2f}s | Avg VRAM: {avg_vram:.1f} MB")
    logger.log(f"  Final Residual Scale: {model.res_scale.item():.4f}")

    return model, train_losses, val_losses, avg_vram, elapsed_time