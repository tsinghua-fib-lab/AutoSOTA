# src/validation_utils.py

import torch
import numpy as np
import gpytorch
import json
import time
import copy
import os
from src.utils import optimize_t_for_x_batch_torch

class ValidationLogger:
    """A simple JSON Lines (.jsonl) logger."""
    def __init__(self, filepath):
        self.filepath = filepath
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, 'w')
        print(f"ValidationLogger initialized. Logging to {filepath}")

    def log(self, data_dict):
        json.dump(data_dict, self.file)
        self.file.write('\n')
    
    def flush(self):
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

def _get_mu_sigma_on_grid(model, x_batch_torch, t_grid_I_torch):
    """
    Evaluate mu and sigma on a custom t-grid T_I corresponding to each x.
    
    x_batch_torch: (N, D_x)
    t_grid_I_torch: (N, G)
    """
    N, D_x = x_batch_torch.shape
    G = t_grid_I_torch.shape[1]
    device = model.device
    dtype = model.dtype
    
    # Create super batch of size (N*G, D_x+1)
    x_rep = x_batch_torch.unsqueeze(1).expand(-1, G, -1).reshape(N*G, D_x)
    t_rep = t_grid_I_torch.reshape(N*G, 1)
    super_batch = torch.cat([x_rep, t_rep], dim=1)
    
    mean_list = []
    var_list = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # pred_batch_size used in utils.py is 4096
        for sb in torch.split(super_batch, 4096, dim=0):
            # pred = model.likelihood(model(sb))
            pred = model(sb) # Calls __call__
            mean_list.append(pred.mean)
            var_list.append(pred.variance)
            
    mean_flat = torch.cat(mean_list, dim=0)
    var_flat = torch.cat(var_list, dim=0)
    
    mean_grid = mean_flat.view(N, G)
    std_grid = var_flat.view(N, G).clamp_min(1e-9).sqrt()
    
    return mean_grid, std_grid

def _get_f_on_grid(dataset, x_batch_np, t_grid_I_torch):
    """Evaluate ground truth f on the T_I grid."""
    N, D_x = x_batch_np.shape
    G = t_grid_I_torch.shape[1]
    
    x_rep_np = np.repeat(x_batch_np, G, axis=0)
    t_rep_np = t_grid_I_torch.cpu().numpy().flatten()
    
    f_flat_np = dataset.get_f(x_rep_np, t_rep_np)
    f_grid_torch = torch.tensor(f_flat_np, 
                                device=t_grid_I_torch.device, 
                                dtype=t_grid_I_torch.dtype).view(N, G)
    return f_grid_torch


def validate_assumptions_on_batch(
    model, 
    x_batch_np: np.ndarray, 
    dataset, 
    logger: ValidationLogger, 
    round_info: dict, 
    beta: float, 
    t_grid_size: int = 101,
    t_mu_batch_np: np.ndarray = None,
    t_ucb_batch_np: np.ndarray = None,
    T_GRID_I_SIZE: int = 51 # Grid size for validation inside I
):
    """
    Perform full theoretical validation on a batch of x candidates.
    """
    device = model.device
    dtype = model.dtype
    x_batch_torch = torch.tensor(x_batch_np, dtype=dtype, device=device)
    N_eval = x_batch_torch.shape[0]

    try:
        # --- Step 1: Determine I = [a, b] and width w ---
        # (t_mu and t_ucb passed from sampler to avoid redundant computation)
        # (t_mu_batch_np corresponds to t_lcb_batch_np)
        t_mu_batch = torch.tensor(t_mu_batch_np, device=device, dtype=dtype)
        t_ucb_batch = torch.tensor(t_ucb_batch_np, device=device, dtype=dtype)
        
        a_batch = torch.min(t_mu_batch, t_ucb_batch)
        b_batch = torch.max(t_mu_batch, t_ucb_batch)
        w_batch = (b_batch - a_batch).clamp_min(1e-6) # width w

        # --- Step 2: Create T_I grid on interval I ---
        # t_grid_I (N, G_I)
        t_grid_I = torch.linspace(0.0, 1.0, T_GRID_I_SIZE, device=device, dtype=dtype).unsqueeze(0)
        t_grid_I = a_batch.unsqueeze(1) + w_batch.unsqueeze(1) * t_grid_I

        # --- Step 3: Evaluate mu, sigma, f on T_I ---
        mean_grid, std_grid = _get_mu_sigma_on_grid(model, x_batch_torch, t_grid_I)
        f_grid = _get_f_on_grid(dataset, x_batch_np, t_grid_I)

        # --- Step 4: Validate A2 (Confidence Band) ---
        # Assumption A2: |f(t)-mu(t)| <= beta*sigma(t)
        coverage_mask = torch.abs(f_grid - mean_grid) <= (beta * std_grid)
        A2_coverage_freq = coverage_mask.float().mean().item()

        # --- Step 5: Validate A1mu (Curvature of mu) - using finite differences ---
        # Assumption A1mu: -L_I <= mu''(t) <= -m_I < 0
        dt = (w_batch / (T_GRID_I_SIZE - 1)).unsqueeze(1) # (N, 1)
        # mu' ~ (N, G-1)
        d_mu = torch.diff(mean_grid, n=1, dim=1) / dt
        
        # --- (*** Correction ***) ---
        # mu'' ~ (N, G-2)
        d2_mu = torch.diff(d_mu, n=1, dim=1) / dt
        # --- (*** End of correction ***) ---
        
        # m_I = inf(-mu''), L_I = sup(-mu'')
        m_I_batch = -torch.max(d2_mu, dim=1)[0]
        L_I_batch = -torch.min(d2_mu, dim=1)[0]
        
        A1mu_mI_mean = m_I_batch.mean().item()
        A1mu_L_I_mean = L_I_batch.mean().item()
        A1mu_strong_concave_freq = (m_I_batch > 0).float().mean().item()

        # --- Step 6: Validate A4 (Gradient of sigma) - using finite differences ---
        # Assumption A4: sup|sigma'(t)| <= (C_k/ell) * sup(sigma(t))
        # sigma' ~ (N, G-1)
        d_sigma = torch.diff(std_grid, n=1, dim=1) / dt
        
        sup_abs_d_sigma = torch.max(torch.abs(d_sigma), dim=1)[0]
        sup_sigma = torch.max(std_grid, dim=1)[0]
        
        A4_ratio = sup_abs_d_sigma / (sup_sigma + 1e-8)
        A4_ratio_mean = A4_ratio.mean().item()

        # --- Step 7: Validate Theorem 1'' (Midpoint sampling and upper bound) ---
        # Perform this only on the "most uncertain" sample (max w) to save time
        
        idx_max_w = torch.argmax(w_batch)
        
        x_i = x_batch_torch[idx_max_w:idx_max_w+1] # (1, D_x)
        t_grid_i = t_grid_I[idx_max_w:idx_max_w+1] # (1, G_I)
        w_i = w_batch[idx_max_w]
        
        std_grid_i = std_grid[idx_max_w]         # (G_I,)
        
        S_I = torch.max(std_grid_i)
        s_I = torch.min(std_grid_i)
        kappa_I = s_I / S_I # Variance flatness
        
        # --- (*** Correction: passing kappa_I ***) ---
        thm1_log = _validate_theorem_1_prime(
            model, x_i, t_grid_i, s_I, S_I, w_i, T_GRID_I_SIZE,
            kappa_I=kappa_I
        )
        # --- (*** End of correction ***) ---

        # --- Step 8: Log everything ---
        log_data = {
            **round_info,
            "timestamp": time.time(),
            "w_mean": w_batch.mean().item(),
            "w_max": w_i.item(),
            "A2_coverage_freq": A2_coverage_freq,
            "A1mu_mI_mean": A1mu_mI_mean,
            "A1mu_L_I_mean": A1mu_L_I_mean,
            "A1mu_strong_concave_freq": A1mu_strong_concave_freq,
            "A4_ratio_mean": A4_ratio_mean,
            "Thm1_Target_x_idx": idx_max_w.item(),
            "Thm1_kappa_I": kappa_I.item(),
            **thm1_log # Merge logs from Thm1 validation
        }
        logger.log(log_data)

    except Exception as e:
        print(f"\n!!! Validation Failed (PID {os.getpid()}) !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 
        logger.log({**round_info, "error": str(e)})


def _validate_theorem_1_prime(model, x_i, t_grid_i, s_I, S_I, w_i, T_GRID_I_SIZE,
                              kappa_I): # (*** Correction: receiving kappa_I ***)
    """
    Validate Theorem 1' and 1'' on a single sample x_i.
    Computes rho, lambda, simulates sampling, and compares actual contraction with bounds.
    """
    
    device = model.device
    dtype = model.dtype
    
    # --- 1. Compute rho_D(u, t) and lambda_I(t) ---
    # x_i (1, D_x), t_grid_i (1, G_I)
    
    # (G_I, D_x+1)
    xt_inputs = torch.cat([
        x_i.expand(T_GRID_I_SIZE, -1),
        t_grid_i.view(-1, 1)
    ], dim=1)
    
    # Get joint distribution of this x_i over T_I
    dist = model.model(xt_inputs)
    # cov_matrix = k_D(u, t) for u,t in T_I
    cov_matrix = dist.covariance_matrix # (G_I, G_I)
    
    sigma_vec = torch.sqrt(torch.diag(cov_matrix)) # (G_I,)
    # rho_matrix[j, k] = rho_D(t_j, t_k)
    rho_matrix = cov_matrix / (sigma_vec.unsqueeze(1) @ sigma_vec.unsqueeze(0))
    
    # lambda_I(t_j) = inf_k rho_D(t_j, t_k)
    lambda_I_t = torch.min(torch.abs(rho_matrix), dim=1)[0] # (G_I,)
    
    # --- 2. Find minimax optimal sampling point t_R* (Theorem 1') ---
    # Phi_I(t) = (lambda_I(t)^2 * s_I^2 * sigma(t)^2) / (sigma(t)^2 + sigma_eps^2)
    sigma_eps_sq = model.likelihood.noise.item()
    sigma_t_sq = sigma_vec.pow(2)
    
    phi_I_t = (lambda_I_t.pow(2) * s_I.pow(2) * sigma_t_sq) / (sigma_t_sq + sigma_eps_sq)
    
    idx_t_R_star = torch.argmax(phi_I_t)
    t_R_star = t_grid_i.flatten()[idx_t_R_star]
    
    # --- 3. Find midpoint t_R (Theorem 1'') ---
    idx_t_R_midpoint = T_GRID_I_SIZE // 2
    t_R_midpoint = t_grid_i.flatten()[idx_t_R_midpoint]

    # --- 4. Simulate sampling at midpoint t_R ---
    # Using "mean-field" update: assume y_obs = mu(x, t_R)
    
    model_plus = copy.deepcopy(model)
    
    # Get value at t_R
    xt_midpoint = torch.cat([x_i, t_R_midpoint.view(1, 1)], dim=1) # (1, D_x+1)
    pred_midpoint = model.likelihood(model(xt_midpoint))
    y_obs_pseudo = pred_midpoint.mean.cpu().detach().numpy()
    x_t_obs_np = xt_midpoint.cpu().numpy() # (1, D_x+1)
    
    # Update using numpy input
    model_plus.update(x_t_obs_np, y_obs_pseudo)
    
    # --- 5. Compute post-sampling sup(sigma^+) (Actual contraction) ---
    _, std_grid_plus = _get_mu_sigma_on_grid(model_plus, x_i, t_grid_i)
    actual_sup_sigma_plus = torch.max(std_grid_plus)
    
    del model_plus # Release memory

    # --- 6. Compute Theorem 1' bound (using t_R_midpoint) ---
    # Bound^2 = S_I^2 - Phi_I(t_R)
    sigma_t_R = sigma_vec[idx_t_R_midpoint]
    lambda_I_t_R = lambda_I_t[idx_t_R_midpoint]
    
    reduction_term = (lambda_I_t_R.pow(2) * s_I.pow(2) * sigma_t_R.pow(2)) / (sigma_t_R.pow(2) + sigma_eps_sq)
    bound_sup_sigma_plus = torch.sqrt((S_I.pow(2) - reduction_term).clamp_min(0.0))

    # --- 7. (Optional) Compute Theorem 1'' closed-form bound (looser) ---
    # Bound_Closed^2 = S_I^2 - (kappa_I^4 * phi(w/2)^2 * S_I^4) / (S_I^2 + sigma_eps^2)
    # Using lambda_I(t_R) as lower bound for phi(w/2)
    # kappa_I = s_I / S_I (Now passed as parameter)
    reduction_term_closed = (kappa_I.pow(4) * lambda_I_t_R.pow(2) * S_I.pow(4)) / (S_I.pow(2) + sigma_eps_sq)
    bound_closed_form = torch.sqrt((S_I.pow(2) - reduction_term_closed).clamp_min(0.0))
    
    return {
        "S_I": S_I.item(),
        "s_I": s_I.item(),
        "w": w_i.item(),
        "lambda_at_midpoint": lambda_I_t_R.item(),
        "t_R_star_vs_midpoint": (t_R_star - t_R_midpoint).item(),
        "Actual_sup_sigma_plus": actual_sup_sigma_plus.item(),
        "Bound_Thm1_Prime": bound_sup_sigma_plus.item(),
        "Bound_Thm1_DoublePrime": bound_closed_form.item(),
        "Gap_Actual_vs_Bound1P": (bound_sup_sigma_plus - actual_sup_sigma_plus).item(),
        "Gap_Actual_vs_Bound1PP": (bound_closed_form - actual_sup_sigma_plus).item(),
    }