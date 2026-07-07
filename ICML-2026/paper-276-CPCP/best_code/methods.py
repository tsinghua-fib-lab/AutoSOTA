import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import *
from trainers import *
from metrics import *
from utils import to_tensor, to_numpy, DEVICE

# Split conformal prediction
def run_split(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha):
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        scores = np.max(np.abs(Y_cal - mu_cal), axis=1)
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        y_lo, y_hi = mu_te - q, mu_te + q
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)

# CQR
def run_cqr(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, mode='pinball'):
    D = Y_tr.shape[1]
    taus = [alpha/2, 1-alpha/2]
    
    if mode == 'ald': 
        model = ALDNet(X_tr.shape[1], 3*D).to(DEVICE)
    else: 
        model = Net(X_tr.shape[1], 2*D).to(DEVICE)
    
    train_cqr_nd(model, to_tensor(X_tr), to_tensor(Y_tr), taus, mode)
    model.eval()
    
    # Calibration
    with torch.no_grad():
        cal_out = model(to_tensor(X_cal))
        q_lo, q_hi = cal_out[:, :D], cal_out[:, D:2*D]
        
        cal_qs = torch.stack([q_lo, q_hi], dim=2)
        ql = torch.min(cal_qs, dim=2).values
        qh = torch.max(cal_qs, dim=2).values
        Y_cal_ts = to_tensor(Y_cal)
        
        # Conformity Score: Signed distance to interval
        resid_vec = torch.maximum(ql - Y_cal_ts, Y_cal_ts - qh)
        scores = resid_vec.max(dim=1).values.cpu().numpy()
        
        n = len(scores)
        q = np.quantile(scores, np.ceil((1-alpha)*(n+1))/n)
        
    # Inference
    with torch.no_grad():
        te_out = model(to_tensor(X_te))
        q_lo_te, q_hi_te = te_out[:, :D], te_out[:, D:2*D]
        te_qs = torch.stack([q_lo_te, q_hi_te], dim=2)
        ql_te = torch.min(te_qs, dim=2).values
        qh_te = torch.max(te_qs, dim=2).values
        
        y_lo = to_numpy(ql_te - q)
        y_hi = to_numpy(qh_te + q)
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)

# Gaussian-scoring
def run_gaussian_scoring(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha):
    D = Y_tr.shape[1]
    model = MultivariateGaussianNet(X_tr.shape[1], D).to(DEVICE)
    train_multivariate_gaussian(model, to_tensor(X_tr), to_tensor(Y_tr), epochs=200)
    model.eval()
    
    # Calibration
    with torch.no_grad():
        mu_cal, L_cal = model(to_tensor(X_cal))
        diff_cal = to_tensor(Y_cal) - mu_cal
        z_cal = torch.linalg.solve_triangular(L_cal, diff_cal.unsqueeze(2), upper=False).squeeze(2)
        scores = torch.norm(z_cal, p=2, dim=1).cpu().numpy()
        n = len(scores)
        q = np.quantile(scores, np.ceil((1-alpha)*(n+1))/n)
        
    # Inference
    with torch.no_grad():
        mu_te, L_te = model(to_tensor(X_te))
        diff_te = to_tensor(Y_te) - mu_te
        z_te = torch.linalg.solve_triangular(L_te, diff_te.unsqueeze(2), upper=False).squeeze(2)
        test_scores = torch.norm(z_te, p=2, dim=1).cpu().numpy()
        covered = (test_scores <= q)
        
        # Size Calculation
        diag_L = torch.diagonal(L_te, dim1=1, dim2=2)
        log_det_sigma = 2 * torch.sum(torch.log(diag_L + 1e-8), dim=1).cpu().numpy()
        log_vol_radius = 0.5 * log_det_sigma + D * np.log(q + 1e-8)
        log_vol_diameter = log_vol_radius + D * np.log(2) 
        size_val = np.mean(log_vol_diameter) / D
        
    # Use unified metrics function with pre-calculated coverage and size
    return get_metrics_nd(Y_te, None, None, X_te, alpha, covered=covered, size_metric=size_val)


# Gaussian scoring with better numerical stability, on WEC dataset (high-dimensional and colinearity) 
def run_gaussian_scoring_robust(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha):
    """
    Robust Gaussian Scoring with Softplus diagonal correction and safe log-volume calculation.
    """
    D = Y_tr.shape[1]
    scaler = StandardScaler()
    Y_tr_s = scaler.fit_transform(Y_tr)
    Y_cal_s = scaler.transform(Y_cal)
    Y_te_s = scaler.transform(Y_te)

    # 1. Pre-calculate scaler log-sum safely to prevent -inf if variance is near zero
    scale_safe = scaler.scale_.copy()
    scale_safe[scale_safe < 1e-6] = 1.0 
    log_scale_sum = np.sum(np.log(scale_safe))

    model = MultivariateGaussianNet(X_tr.shape[1], D).to(DEVICE)
    train_multivariate_gaussian(model, to_tensor(X_tr), to_tensor(Y_tr_s), epochs=200)
    model.eval()

    diag_idx = torch.arange(D).to(DEVICE)

    # Helper to enforce positive definite L matrix
    def ensure_pos_diag(L_tensor):
        diags = L_tensor[:, diag_idx, diag_idx]
        # Softplus guarantees positivity (unlike += 1e-3 which fails for negative outputs)
        L_tensor[:, diag_idx, diag_idx] = F.softplus(diags) + 1e-6
        return L_tensor

    # --- Calibration ---
    with torch.no_grad():
        mu_cal, L_cal = model(to_tensor(X_cal))        
        L_cal = ensure_pos_diag(L_cal) # Apply stability fix

        diff_cal = to_tensor(Y_cal_s) - mu_cal
        
        # Solve L*z = y - mu
        # Using try-except is recommended for high-dim matrices (optional but safer)
        try:
            z_cal = torch.linalg.solve_triangular(L_cal, diff_cal.unsqueeze(2), upper=False).squeeze(2)
        except RuntimeError:
            # Fallback for singular matrix
            return {'Cov': 0, 'Size': np.nan, 'WSC': 0, 'MSCE_30': 0, 'MSCE_10': 0, 'L1-ERT': 0, 'L2-ERT': 0}

        scores = torch.norm(z_cal, p=2, dim=1).cpu().numpy()        
        scores = scores[np.isfinite(scores)]
        
        if len(scores) == 0: 
            return {'Cov': 0, 'Size': np.nan, 'WSC': 0, 'MSCE_30': 0, 'MSCE_10': 0, 'L1-ERT': 0, 'L2-ERT': 0}        
        
        n = len(scores)
        q = np.quantile(scores, np.ceil((1-alpha)*(n+1))/n)
        
    # --- Inference ---
    with torch.no_grad():
        mu_te, L_te = model(to_tensor(X_te))        
        L_te = ensure_pos_diag(L_te) # Apply stability fix

        diff_te = to_tensor(Y_te_s) - mu_te
        z_te = torch.linalg.solve_triangular(L_te, diff_te.unsqueeze(2), upper=False).squeeze(2)
        test_scores = torch.norm(z_te, p=2, dim=1).cpu().numpy()        
        test_scores = np.nan_to_num(test_scores, nan=np.inf)        
        
        covered = (test_scores <= q)
            
        # --- Robust Size Calculation ---
        # 1. Determinant part: 2 * sum(log(diag(L)))
        # diag_L is now guaranteed positive via ensure_pos_diag, so no NaNs here
        diag_L = torch.diagonal(L_te, dim1=1, dim2=2)
        log_det_sigma = 2 * torch.sum(torch.log(diag_L), dim=1).cpu().numpy()
        
        # 2. Radius part: D * log(q)
        # Handle case where q is extremely small (overfitting)
        safe_q = max(q, 1e-8)
        
        log_vol_radius = 0.5 * log_det_sigma + D * np.log(safe_q)
        log_vol_diameter = log_vol_radius + D * np.log(2)        
        
        # 3. Add scaler contribution
        log_vol_diameter += log_scale_sum
        
        size_val = np.mean(log_vol_diameter) / D        
        
    # Use the existing interface as requested
    return get_metrics_nd(Y_te, None, None, X_te, alpha, covered=covered, size_metric=size_val)


# RCP 
def run_rcp(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, mode='pinball'):
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        
    X_est, X_conf, S_est, S_conf = train_test_split(X_cal, S_cal, test_size=0.5, random_state=42)
    r_net = ALDNet(X_tr.shape[1], 2).to(DEVICE) if mode=='ald' else Net(X_tr.shape[1], 1).to(DEVICE)
    train_rcp_score(r_net, to_tensor(X_est), to_tensor(S_est.reshape(-1,1)), 1-alpha, mode)
    r_net.eval()
    
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_conf)))[:, 0].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 0].flatten()
        scores = S_conf - tau_conf
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        width = (tau_te + q)[:, None]
        y_lo, y_hi = mu_te - width, mu_te + width
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)

# Only multi-task part for ablation study
def run_rcp_multi_head(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha):
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        
    X_est, X_conf, S_est, S_conf = train_test_split(X_cal, S_cal, test_size=0.5, random_state=42)
    r_net = Net(X_tr.shape[1], 3).to(DEVICE)
    train_rcp_multi_head(r_net, to_tensor(X_est), to_tensor(S_est.reshape(-1,1)), [1-alpha-0.05, 1-alpha, 1-alpha+0.05])
    r_net.eval()
    
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_conf)))[:, 1].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 1].flatten()
        scores = S_conf - tau_conf
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        width = (tau_te + q)[:, None]
        y_lo, y_hi = mu_te - width, mu_te + width
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)

# Partition learning conformal prediction
def run_plcp(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, n_groups, score_type):
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
        
    X_ptr, X_pcal, S_ptr, S_pcal = train_test_split(X_cal, S_cal, test_size=0.5, random_state=42)
    plcp = PLCPNet(X_tr.shape[1], n_groups).to(DEVICE)
    train_plcp_model(plcp, to_tensor(X_ptr), to_tensor(S_ptr.reshape(-1,1)), 1-alpha, score_type)
    plcp.eval()
    
    with torch.no_grad():
        probs, _ = plcp(to_tensor(X_pcal))
        grps = torch.argmax(probs, dim=1).cpu().numpy()
        g_q = np.quantile(S_pcal, np.ceil((1-alpha)*(len(S_pcal)+1))/len(S_pcal))
        
        q_dict = {
            k: np.quantile(S_pcal[grps==k], np.ceil((1-alpha)*(len(S_pcal[grps==k])+1))/len(S_pcal[grps==k])) 
            if np.sum(grps==k) > 10 else g_q 
            for k in range(n_groups)
        }
        
        probs_te, _ = plcp(to_tensor(X_te))
        grps_te = torch.argmax(probs_te, dim=1).cpu().numpy()
        q_te = np.array([q_dict.get(g, g_q) for g in grps_te])[:, None]
        y_lo, y_hi = mu_te - q_te, mu_te + q_te
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)


# CPCP with better stability, loss mixing + weight clip
def run_rcp_density_improved(X_tr, Y_tr, X_cal, Y_cal, X_te, Y_te, alpha, 
                             epsilon=0.02, mode='vanilla', clip_max=5.0, mix_ratio=1.0,
                             dataset_name="unknown", seed=0):
    """
    Main Runner for Colorful Pinball Conformal Prediction (CPCP).
    """
    D = Y_tr.shape[1]
    mu_net = Net(X_tr.shape[1], D).to(DEVICE)
    train_mean(mu_net, to_tensor(X_tr), to_tensor(Y_tr))
    mu_net.eval()
    
    with torch.no_grad():
        mu_cal = to_numpy(mu_net(to_tensor(X_cal)))
        S_cal = np.max(np.abs(Y_cal - mu_cal), axis=1)
        mu_te = to_numpy(mu_net(to_tensor(X_te)))
    
    # Split calibration data for three stages: 1. Aux Quantiles, 2. Weight Est & Finetune, 3. Conformalization
    n = len(X_cal)
    idx1 = int(0.4 * n)
    idx2 = int(0.8 * n)
    rng = np.random.default_rng(42 + seed)
    perm = rng.permutation(n)
    X_cal, S_cal = X_cal[perm], S_cal[perm]
    
    X_est1, S_est1 = X_cal[:idx1], S_cal[:idx1]
    X_est2, S_est2 = X_cal[idx1:idx2], S_cal[idx1:idx2]
    X_score, S_score = X_cal[idx2:], S_cal[idx2:]
    
    target_q = 1 - alpha
    taus_list = [max(0.01, target_q - epsilon), target_q, min(0.99, target_q + epsilon)]
    
    # 1. Train auxiliary quantiles
    r_net = MonotonicThreeHeadNet(X_tr.shape[1]).to(DEVICE)
    train_three_head_base(r_net, to_tensor(X_est1), to_tensor(S_est1.reshape(-1,1)), taus_list, epochs=200)
    
    # 2. Fine-tune main head with density weights (CPCP core)
    finetune_main_head_improved(r_net, to_tensor(X_est2), to_tensor(S_est2.reshape(-1,1)), 
                                target_tau=target_q, epsilon=epsilon, epochs=200,
                                mode=mode, clip_max=clip_max, mix_ratio=mix_ratio,
                                save_weights_path=None)
    
    r_net.eval()
    with torch.no_grad():
        tau_conf = to_numpy(r_net(to_tensor(X_score)))[:, 1].flatten()
        tau_te = to_numpy(r_net(to_tensor(X_te)))[:, 1].flatten()
        tau_conf, tau_te = np.maximum(tau_conf, 1e-4), np.maximum(tau_te, 1e-4)
        
        scores = S_score - tau_conf
        q = np.quantile(scores, np.ceil((1-alpha)*(len(scores)+1))/len(scores))
        
        width = (tau_te + q)[:, None]
        y_lo, y_hi = mu_te - width, mu_te + width
        
    return get_metrics_nd(Y_te, y_lo, y_hi, X_te, alpha)