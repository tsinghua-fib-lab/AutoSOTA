import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import EarlyStopping, LpLoss


def train_gno(model, data_mgr, torch_mapper, X_train, Y_train, X_val, Y_val,
              epochs, cfg, logger):
    logger.log("\n" + "-"*50)
    logger.log("Training GNO (Graph Neural Operator)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
            gno_in, _ = data_mgr.prepare_gno_batch(x_batch, None)
            pred = model(gno_in, data_mgr.pts)
            pred = pred.squeeze(-1)

            loss = criterion(pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_in, _ = data_mgr.prepare_gno_batch(X_val, None)
            val_pred = model(val_in, data_mgr.pts).squeeze(-1)

            val_loss = criterion(val_pred, Y_val).item()
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
            fno_in, _ = data_mgr.prepare_fno_batch(x_batch, None)
            pred_grid = model(fno_in)
            pred = data_mgr.decode_fno_output(pred_grid)

            loss = criterion(pred, y_batch)

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
                pred = data_mgr.decode_fno_output(pred_grid)
                val_preds.append(pred)

            val_pred = torch.cat(val_preds, dim=0)

            val_loss = criterion(val_pred, Y_val).item()
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
    logger.log("\n" + "-"*50)
    logger.log("Training MGN (MeshGraphNets)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
                node_feat = torch.cat([x_batch[b:b+1].T, data_mgr.pts], dim=1)
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
                node_feat = torch.cat([X_val[b:b+1].T, data_mgr.pts], dim=1)
                pred = model(node_feat, edge_index, edge_attr)
                val_preds.append(pred.squeeze(-1))
            val_pred = torch.stack(val_preds)

            val_loss = criterion(val_pred, Y_val).item()
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
    logger.log("\n" + "-"*50)
    logger.log("Training DeepONet")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
    logger.log("\n" + "-"*50)
    logger.log("Training GeoFNO (Geometry-Adaptive FNO)")
    logger.log("-"*50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
            val_preds = []
            for i in range(0, X_val.shape[0], batch_size):
                x_b = X_val[i:i+batch_size]
                x_coords, features = data_mgr.prepare_geofno_batch(x_b)
                pred = model(x_coords, features).squeeze(-1)
                val_preds.append(pred)

            val_pred = torch.cat(val_preds, dim=0)

            val_loss = criterion(val_pred, Y_val).item()
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


def train_HSD_base(model, train_loader, test_loader, Phi0_torch, epochs, cfg, device, logger, host_ops):
    logger.log("\n" + "-" * 50)
    logger.log("Training HSD_base - Scalar Field")
    logger.log("-" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    criterion_rel = LpLoss(size_average=True)

    LAMBDA_MAIN = 1.0
    LAMBDA_SMOOTH = 0.1

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

        for c0, c1, c2, c0_tgt, gt_field, x_spatial, y_target in train_loader:
            c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()

            coeffs = model(c0, c1, c2)
            pred = torch.matmul(coeffs, Phi0_torch.t())

            loss_main = criterion_rel(pred, y_target)

            freq_weights = torch.arange(1, model.k0 + 1, device=device).float()
            freq_weights = freq_weights / freq_weights.sum()
            weighted_coeffs = coeffs * freq_weights.unsqueeze(0)
            loss_smooth = torch.mean(weighted_coeffs ** 2)

            loss = LAMBDA_MAIN * loss_main + LAMBDA_SMOOTH * loss_smooth

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
            for c0, c1, c2, c0_tgt, gt_field, x_spatial, y_target in test_loader:
                c0, c1, c2 = c0.to(device), c1.to(device), c2.to(device)
                y_target = y_target.to(device)

                coeffs = model(c0, c1, c2)
                pred = torch.matmul(coeffs, Phi0_torch.t())

                loss = criterion_rel(pred, y_target)
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
    logger.log("\n" + "-" * 50)
    logger.log("Training HSD - Scalar Field")
    logger.log("-" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_OURS, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_rel = LpLoss(size_average=True)
    criterion_l1 = nn.L1Loss(reduction='mean')

    LAMBDA_MAIN = 1.0
    LAMBDA_BASE = 0.3
    LAMBDA_SPARSE = 1e-2
    LAMBDA_SMOOTH = 0.05

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()
    total_vram_mb = 0.0
    fetch_count = 0

    for epoch in range(epochs):
        model.train()
        total_weighted_loss = 0.0
        total_main_loss = 0.0
        n_batches = 0

        for c0, c1, c2, c0_tgt, gt_field, x_spatial, y_target in train_loader:
            c0 = c0.to(device)
            c1 = c1.to(device)
            c2 = c2.to(device)
            x_spatial = x_spatial.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()

            pred_total, pred_base, pred_residual = model(c0, c1, c2, x_spatial)

            loss_main = criterion_rel(pred_total, y_target)
            loss_base = criterion_rel(pred_base, y_target)
            loss_sparse = criterion_l1(pred_residual, torch.zeros_like(pred_residual))

            if pred_total.shape[1] > 1:
                diff = pred_total[:, 1:] - pred_total[:, :-1]
                loss_smooth = torch.mean(diff ** 2)
            else:
                loss_smooth = torch.tensor(0.0, device=device)

            loss = (LAMBDA_MAIN * loss_main
                    + LAMBDA_BASE * loss_base
                    + LAMBDA_SPARSE * loss_sparse
                    + LAMBDA_SMOOTH * loss_smooth)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_weighted_loss += loss.item()
            total_main_loss += loss_main.item()
            n_batches += 1

        avg_train_loss = total_main_loss / n_batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for c0, c1, c2, c0_tgt, gt_field, x_spatial, y_target in test_loader:
                c0 = c0.to(device)
                c1 = c1.to(device)
                c2 = c2.to(device)
                x_spatial = x_spatial.to(device)
                y_target = y_target.to(device)

                pred_total, _, _ = model(c0, c1, c2, x_spatial)

                loss = criterion_rel(pred_total, y_target)
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