"""
Training functions for all models
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import EarlyStopping, LpLoss


def train_gno(model, data_mgr, X_train, Y_train, X_val, Y_val, epochs, cfg, logger):
    logger.log("\n" + "-"*50)
    logger.log("Training GNO (Graph Neural Operator)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # criterion = nn.MSELoss()
    criterion = LpLoss(size_average=True)
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            gno_in, gno_tgt = data_mgr.prepare_gno_batch(x_batch, y_batch)
            pred = model(gno_in, data_mgr.pts)
            loss = criterion(pred, gno_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_in, val_tgt = data_mgr.prepare_gno_batch(X_val, Y_val)
            val_pred = model(val_in, data_mgr.pts)
            val_loss = criterion(val_pred, val_tgt).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        # Track VRAM usage
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



def train_fno(model, data_mgr, X_train, Y_train, X_val, Y_val, epochs, cfg, logger):
    logger.log("\n" + "-"*50)
    logger.log("Training FNO (Fourier Neural Operator)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = LpLoss(size_average=True)
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    batch_size_fno = min(cfg.BATCH_SIZE, 32)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_fno, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 准备 FNO 输入（只需要 x，不需要 y 的 grid 版本）
            fno_in, _ = data_mgr.prepare_fno_batch(x_batch, None)
            
            # 前向传播
            pred_grid = model(fno_in)
            
            # 解码回 mesh 空间
            pred_mesh = data_mgr.decode_fno_output(pred_grid).squeeze(-1)  # (B, N)
            
            # 在 mesh 空间计算 loss
            loss = criterion(pred_mesh, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        with torch.no_grad():
            val_in, _ = data_mgr.prepare_fno_batch(X_val, None)
            val_pred_grid = model(val_in)
            val_pred_mesh = data_mgr.decode_fno_output(val_pred_grid).squeeze(-1)
            val_loss = criterion(val_pred_mesh, Y_val).item()
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

def train_mgn(model, data_mgr, X_train, Y_train, X_val, Y_val, epochs, cfg, device, logger):
    logger.log("\n" + "-"*50)
    logger.log("Training MGN (MeshGraphNets)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # criterion = nn.MSELoss()
    criterion = LpLoss(size_average=True)
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0
    
    edge_index = data_mgr.edge_index
    edge_attr = data_mgr.edge_attr

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            B = x_batch.shape[0]
            batch_preds = []
            
            for b in range(B):
                node_feat = torch.cat([
                    x_batch[b:b+1].T,
                    data_mgr.pts
                ], dim=1)
                
                pred = model(node_feat, edge_index, edge_attr)
                batch_preds.append(pred.squeeze(-1))
            
            pred_batch = torch.stack(batch_preds)
            
            loss = criterion(pred_batch, y_batch)
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
                node_feat = torch.cat([
                    X_val[b:b+1].T,
                    data_mgr.pts
                ], dim=1)
                pred = model(node_feat, edge_index, edge_attr)
                val_preds.append(pred.squeeze(-1))
            val_pred = torch.stack(val_preds)
            val_loss = criterion(val_pred, Y_val).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        # Track VRAM usage
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


def train_deeponet(model, data_mgr, X_train, Y_train, X_val, Y_val, epochs, cfg, logger):
    logger.log("\n" + "-"*50)
    logger.log("Training DeepONet")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # criterion = nn.MSELoss()
    criterion = LpLoss(size_average=True)
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0
    
    trunk_input = data_mgr.pts

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            pred = model(x_batch, trunk_input).squeeze(-1)
            
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val, trunk_input).squeeze(-1)
            val_loss = criterion(val_pred, Y_val).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        # Track VRAM usage
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


def train_geofno(model, data_mgr, X_train, Y_train, X_val, Y_val, epochs, cfg, logger):
    logger.log("\n" + "-"*50)
    logger.log("Training GeoFNO (Geometry-Adaptive FNO)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # criterion = nn.MSELoss()
    criterion = LpLoss(size_average=True)
    early_stop = EarlyStopping(patience=cfg.PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None

    batch_size = min(cfg.BATCH_SIZE, 32)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            x_coords, features = data_mgr.prepare_geofno_batch(x_batch)
            pred = model(x_coords, features).squeeze(-1)
            
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            x_coords_val, features_val = data_mgr.prepare_geofno_batch(X_val)
            val_pred = model(x_coords_val, features_val).squeeze(-1)
            val_loss = criterion(val_pred, Y_val).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        # Track VRAM usage
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


def train_spectral(model, train_loader, test_loader, Phi0_torch, epochs, cfg, device, logger):
    """Training for pure spectral HSD_base model."""
    
    logger.log("\n" + "-"*50)
    logger.log("Training Spectral Physics-Aware Operator")
    logger.log("-" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    criterion_rel = LpLoss(size_average=True)
    criterion_l1 = nn.L1Loss(reduction='mean')

    LAMBDA_L1 = 1e-2

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

        for c0, c1, c2, c0_tgt, y_tgt, x_spatial in train_loader:
            c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
            c0_tgt = c0_tgt.to(device)
            y_tgt = y_tgt.to(device)

            optimizer.zero_grad()

            pred_c0 = model(c0, c1, c2)
            pred_u = torch.matmul(pred_c0, Phi0_torch.t())

            loss_main = criterion_rel(pred_u, y_tgt)
            loss_sparsity = criterion_l1(pred_c0, torch.zeros_like(pred_c0))
            
            loss = loss_main + LAMBDA_L1 * loss_sparsity

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for c0, c1, c2, c0_tgt, y_tgt, x_spatial in test_loader:
                c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
                y_tgt = y_tgt.to(device)

                pred_c0 = model(c0, c1, c2)
                pred_u = torch.matmul(pred_c0, Phi0_torch.t())

                loss = criterion_rel(pred_u, y_tgt)
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / n_val_batches
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Track VRAM usage
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



def train_HSD(model, train_loader, test_loader, epochs, cfg, device, logger):
    """Training for HSD spectral + FNO model."""
    
    logger.log("\n" + "-"*50)
    logger.log("Training HSD Model (Spectral + Residual FNO)")
    logger.log("-" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion_rel = LpLoss(size_average=True)
    criterion_l1 = nn.L1Loss(reduction='mean')
    
    LAMBDA_L1 = 1e-2

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

        for c0, c1, c2, c0_tgt, y_tgt, x_spatial in train_loader:
            c0 = c0.to(device)
            c1 = c1.to(device)
            c2 = c2.to(device)
            y_tgt = y_tgt.to(device)
            x_spatial = x_spatial.to(device)

            optimizer.zero_grad()

            u_total, u_hodge, u_fno = model(c0, c1, c2, x_spatial)

            loss_main = criterion_rel(u_total, y_tgt)
            
      
            loss_sparsity = criterion_l1(u_fno, torch.zeros_like(u_fno))
            
            loss = loss_main + LAMBDA_L1 * loss_sparsity

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for c0, c1, c2, c0_tgt, y_tgt, x_spatial in test_loader:
                c0 = c0.to(device)
                c1 = c1.to(device)
                c2 = c2.to(device)
                y_tgt = y_tgt.to(device)
                x_spatial = x_spatial.to(device)

                u_total, _, _ = model(c0, c1, c2, x_spatial)

                loss = criterion_rel(u_total, y_tgt)
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

