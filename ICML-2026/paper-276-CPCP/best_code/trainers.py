import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from losses import *
from utils import to_numpy

def train_mean(model, x_tr, y_tr, epochs=100, warmup_epochs=10, lr_max=2e-3, lr_min=1e-4):
    opt = optim.Adam(model.parameters(), lr=lr_max)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=1024, shuffle=True)
    # Linear warmup + cosine annealing schedule
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs * len(loader))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=(epochs - warmup_epochs) * len(loader), eta_min=lr_min)
    model.train()
    step = 0
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss = nn.HuberLoss(delta=1.0)(model(bx), by)
            loss.backward()
            opt.step()
            if step < warmup_epochs * len(loader):
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            step += 1
    return model

def train_cqr_nd(model, x_tr, y_tr, taus, mode='pinball', epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=1024, shuffle=True)
    D = y_tr.shape[1]
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            preds = model(bx)
            if mode == 'ald': 
                loss = ald_loss_cqr_nd(preds, by, taus)
            else:
                q_lo = preds[:, :D]; q_hi = preds[:, D:]
                loss = (pinball_loss(q_lo, by, taus[0]) + pinball_loss(q_hi, by, taus[1])).mean()
            loss.backward()
            opt.step()
    return model

def train_rcp_score(model, x_tr, s_tr, tau, mode='pinball', epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            if mode == 'ald': loss = ald_loss_rcp(preds, bs, tau)
            elif mode == 'pinball': loss = pinball_loss(preds, bs, tau).mean()
            loss.backward()
            opt.step()
    return model

def train_plcp_model(model, x_tr, s_tr, target_tau, loss_type='pinball', epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            h_probs, q_params = model(bx)
            s_exp = bs.repeat(1, q_params.shape[1])
            q_exp = q_params.repeat(bx.shape[0], 1)
            diff = s_exp - q_exp
            elem_loss = torch.max(target_tau * diff, (target_tau - 1) * diff)
            loss = torch.mean(torch.sum(h_probs * elem_loss, dim=1))
            loss.backward()
            opt.step()
    return model

def train_multivariate_gaussian(model, x_tr, y_tr, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, y_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            mu, L = model(bx)
            loss = multivariate_nll_loss(mu, L, by)
            loss.backward()
            opt.step()
    return model


def train_multivariate_gaussian_robust(model, x_tr, y_tr, epochs=100):
    # lower learning rate
    opt = optim.Adam(model.parameters(), lr=2e-4)     
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr), 
        batch_size=1024, 
        shuffle=True
    )    
    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            mu, L = model(bx)            
            loss = multivariate_nll_loss_robust(mu, L, by)            
            # NAN detect
            if torch.isnan(loss) or torch.isinf(loss):
                # print(f"Warning: NaN loss in epoch {epoch}, skipping batch.")
                opt.zero_grad() # Clear gradients
                continue            
            loss.backward()            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)            
            opt.step()
    return model


def train_rcp_multi_head(model, x_tr, s_tr, taus_list, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            loss = 0
            for i, t in enumerate(taus_list):
                loss += pinball_loss(preds[:, i:i+1], bs, t).mean()
            loss.backward()
            opt.step()
    return model

def train_three_head_base(model, x_tr, s_tr, taus_list, epochs=100):
    opt = optim.Adam(model.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    model.train()
    tau_lo, tau_main, tau_hi = taus_list
    for _ in range(epochs):
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            loss_elements = pinball_loss(preds[:, 0:1], bs, tau_lo) + \
                            pinball_loss(preds[:, 1:2], bs, tau_main) + \
                            pinball_loss(preds[:, 2:3], bs, tau_hi)
            loss = loss_elements.mean()
            loss.backward()
            opt.step()
    return model

def finetune_main_head_improved(model, x_tr, s_tr, target_tau, epsilon, epochs=100, 
                                mode='vanilla', clip_max=5.0, mix_ratio=0.5, 
                                save_weights_path=None):
    """
    Implements the core 'Colorful Pinball' optimization: 
    Finetunes the main quantile head using density-weighted pinball loss.
    """
    # Freeze shared and auxiliary heads
    for param in model.shared.parameters(): param.requires_grad = False
    for param in model.head_lo_gap.parameters(): param.requires_grad = False
    for param in model.head_hi_gap.parameters(): param.requires_grad = False
    
    tau_lo = max(0.01, target_tau - epsilon)
    tau_hi = min(0.99, target_tau + epsilon)
    
    # Optional: Debug weight statistics
    if save_weights_path is not None:
        model.eval()
        with torch.no_grad():
            preds_all = model(x_tr)
            q_lo, q_hi = preds_all[:, 0:1], preds_all[:, 2:3]
            q_diff = (q_hi - q_lo) + 1e-6
            raw_weights = (tau_hi - tau_lo) / q_diff
            norm_weights = raw_weights / raw_weights.mean()
            
            df_w = pd.DataFrame({
                'q_diff': to_numpy(q_diff).flatten(),
                'raw_weight': to_numpy(raw_weights).flatten(),
                'norm_weight': to_numpy(norm_weights).flatten()
            })
            os.makedirs(os.path.dirname(save_weights_path), exist_ok=True)
            df_w.to_csv(save_weights_path, index=False)
    
    # Training Loop
    opt = optim.Adam(model.head_main.parameters(), lr=2e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_tr, s_tr), batch_size=1024, shuffle=True)
    weight_update_freq = 20  # Recompute global density weights every K epochs
    
    # Pre-compute global density weights on full dataset
    model.eval()
    with torch.no_grad():
        preds_all = model(x_tr)
        q_lo_all, q_hi_all = preds_all[:, 0:1], preds_all[:, 2:3]
        q_diff_global = (q_hi_all - q_lo_all) + torch.clamp(1e-4 - (q_hi_all - q_lo_all), min=0)
        density_global = (tau_hi - tau_lo) / q_diff_global
        density_global = density_global / density_global.mean()
        if mode == 'clip':
            density_global = torch.clamp(density_global, max=clip_max)
    
    model.train() 
    for epoch in range(epochs):
        # Recompute global density weights periodically
        if epoch > 0 and epoch % weight_update_freq == 0:
            model.eval()
            with torch.no_grad():
                preds_all = model(x_tr)
                q_lo_all, q_hi_all = preds_all[:, 0:1], preds_all[:, 2:3]
                q_diff_global = (q_hi_all - q_lo_all) + torch.clamp(1e-4 - (q_hi_all - q_lo_all), min=0)
                density_global = (tau_hi - tau_lo) / q_diff_global
                density_global = density_global / density_global.mean()
                if mode == 'clip':
                    density_global = torch.clamp(density_global, max=clip_max)
            model.train()
        
        for bx, bs in loader:
            opt.zero_grad()
            preds = model(bx)
            q_main = preds[:, 1:2]
            
            # Use global density weights (indexed to match batch)
            batch_indices = list(range(bx.shape[0]))
            # Use per-batch density for current batch samples from global weights
            # Since we shuffle the loader, just use per-batch weights from current preds
            with torch.no_grad():
                q_lo, q_hi = preds[:, 0:1], preds[:, 2:3]
                q_diff = (q_hi - q_lo) + torch.clamp(1e-4 - (q_hi - q_lo), min=0)
                density_weight = (tau_hi - tau_lo) / q_diff
                density_weight = density_weight / density_weight.mean()
                if mode == 'clip':
                    density_weight = torch.clamp(density_weight, max=clip_max)
            
            loss_pinball = pinball_loss(q_main, bs, target_tau)
            loss_weighted = (loss_pinball * density_weight).mean()
            loss_original = loss_pinball.mean()
            
            # Loss Mixing: Alpha * Weighted + (1-Alpha) * Original
            final_loss = mix_ratio * loss_weighted + (1.0 - mix_ratio) * loss_original
            
            final_loss.backward()
            opt.step()
            
    # Unfreeze for future use
    for param in model.parameters(): param.requires_grad = True
    return model