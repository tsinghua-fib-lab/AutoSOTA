import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from generate_dist import generate_simulation_data_torch_true
from scipy.stats import norm
import numpy as np
import warnings

# Suppress cvxpy DPP compilation warning
warnings.filterwarnings('ignore', message='.*too many parameters for efficient DPP compilation.*')

# Remove grem and compute_grem_results_cvx functions

def finalize_grem_results(qf, residuals, totVa, n, p, y, x, optns, xOut=None, yM=None, wc=None, xMean=None, model=None):
    """
    Finalize GREM results by computing statistics and optionally predictions.
    """
    # qfSupp <- 1:M / M
    M = qf.shape[1]
    qfSupp = torch.arange(1, M + 1, dtype=torch.float32) / M
    
    # resVa <- sum(residuals^2)
    resVa = torch.sum(residuals**2)
    
    # RSquare <- 1 - resVa / totVa
    RSquare = 1 - resVa / totVa
    
    # adjRSquare <- RSquare - (1 - RSquare) * p / (n - p - 1)
    adjRSquare = RSquare - (1 - RSquare) * p / (n - p - 1)
    
    # Initialize result dictionary
    res = {
        'qf': qf,
        'qfSupp': qfSupp,
        'RSquare': RSquare,
        'adjRSquare': adjRSquare,
        'residuals': residuals,
        'y': y,
        'x': x,
        'optns': optns
    }
    return res

def generate_xin_yin():
    """
    Equivalent to R:
    xin = seq(0,1,0.05)
    yin = lapply(xin, function(x) { rnorm(100, rnorm(1,x,0.005), 0.05) })
    Returns:
        xin: torch tensor
        yin: list of torch tensors
    """
    xin = torch.arange(0, 1.001, 0.05, dtype=torch.float32)
    torch.manual_seed(42)  # For reproducibility
    yin = []
    for x in xin:
        mean_val = torch.normal(x, 0.005)
        y_values = torch.normal(mean_val, 0.05, (100,))
        yin.append(y_values.tolist())
    return xin, yin

def plot_normal_quantile_and_grem(qf=None, qfSupp=None, yin=None, xin=None, theta=None, mu=None, sigma=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    plt.figure(figsize=(12, 8))
    n_plot = min(10, len(yin))
    idxs = np.random.choice(len(yin), n_plot, replace=False)
    # Project xin to scalar if theta is provided
    if theta is not None and xin is not None and xin.ndim > 1:
        xin_proj = xin @ theta
    else:
        xin_proj = xin
    for i, idx in enumerate(idxs):
        # True mu and sigma for this observation
        true_mu = float(mu[idx]) if mu is not None else (float(xin_proj[idx]) if xin_proj.ndim == 1 else float(xin_proj[idx] @ theta))
        true_sigma = float(sigma[idx]) if sigma is not None else (np.exp(true_mu) / (1 + np.exp(true_mu)))
        # Quantile levels
        quantiles = qfSupp.cpu().numpy() if qfSupp is not None else np.linspace(0, 1, 100)
        # True quantile function (inverse CDF)
        true_qf = norm.ppf(quantiles, loc=true_mu, scale=true_sigma)
        # Plot true quantile function
        plt.plot(quantiles, true_qf, 'g-', alpha=0.7, label='True Quantile' if i == 0 else None)
        # Plot GREM quantile function
        plt.plot(quantiles, qf[idx].cpu().numpy(), 'b--', alpha=0.7, label='GREM Quantile' if i == 0 else None)
    plt.xlabel('Quantile Level')
    plt.ylabel('Value')
    plt.title('True Quantile Function vs GREM Quantile Function')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_normal_quantile_and_grem_xproj(qf=None, qfSupp=None, xin_proj=None, theta=None, mu=None, sigma=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    # Choose a few quantile levels to plot (e.g., 5 evenly spaced)
    n_quantiles = 5
    quantile_idxs = np.linspace(0, qf.shape[1] - 1, n_quantiles, dtype=int)
    quantile_levels = qfSupp.cpu().numpy()[quantile_idxs]

    # Sort by projected x for a smooth plot
    sort_idx = np.argsort(xin_proj.cpu().numpy())
    xin_proj_sorted = xin_proj.cpu().numpy()[sort_idx]
    mu_sorted = mu.cpu().numpy()[sort_idx]
    sigma_sorted = sigma.cpu().numpy()[sort_idx]
    qf_sorted = qf.cpu().numpy()[sort_idx, :]
    
    plt.figure(figsize=(12, 8))
    for i, q_idx in enumerate(quantile_idxs):
        # True quantile values for all observations at this quantile level
        true_q = norm.ppf(quantile_levels[i], loc=mu_sorted, scale=sigma_sorted)
        # GREM quantile values for all observations at this quantile level
        pred_q = qf_sorted[:, q_idx]
        plt.plot(xin_proj_sorted, true_q, 'g-', alpha=0.7, label=f'True QF q={quantile_levels[i]:.2f}' if i == 0 else None)
        plt.plot(xin_proj_sorted, pred_q, 'b--', alpha=0.7, label=f'GREM QF q={quantile_levels[i]:.2f}' if i == 0 else None)
    plt.xlabel('Projected x (x @ theta)')
    plt.ylabel('Quantile Value')
    plt.title('Quantile Function vs Projected x (True vs GREM)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_normal_quantile_and_grem_pairs(qf=None, qfSupp=None, xin_proj=None, theta=None, mu=None, sigma=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    plt.figure(figsize=(10, 6))
    n_plot = min(10, qf.shape[0])
    idxs = np.random.choice(qf.shape[0], n_plot, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n_plot))
    quantiles = qfSupp.cpu().numpy() if qfSupp is not None else np.linspace(0, 1, qf.shape[1])

    for i, idx in enumerate(idxs):
        # True quantile function for this observation
        true_mu = float(mu[idx])
        true_sigma = float(sigma[idx])
        true_qf = norm.ppf(quantiles, loc=true_mu, scale=true_sigma)
        # GREM quantile function for this observation
        pred_qf = qf[idx].cpu().numpy()
        # Plot both with the same color
        plt.plot(quantiles, true_qf, color=colors[i], linestyle='-', label=f'Obs {idx+1} True' if i == 0 else None)
        plt.plot(quantiles, pred_qf, color=colors[i], linestyle='--', label=f'Obs {idx+1} GREM' if i == 0 else None)

    plt.xlabel('Quantile Level')
    plt.ylabel('Value')
    plt.title('True vs GREM Quantile Functions (10 Random Observations)')
    # Only show one legend entry for true and one for GREM
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

def plot_true_vs_grem_quantiles(qf, qfSupp, mu, sigma, n_plot=5):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    plt.figure(figsize=(8, 5))
    quantile_levels = qfSupp.cpu().numpy() if qfSupp is not None else np.linspace(0, 1, qf.shape[1])
    idxs = np.random.choice(qf.shape[0], n_plot, replace=False)
    for i, idx in enumerate(idxs):
        # Empirical (predicted) quantile function from GREM
        plt.plot(quantile_levels, qf[idx].cpu().numpy(), label=f'Obs {idx+1} GREM', linestyle='--')
        # True quantile function (from normal)
        true_qf = norm.ppf(quantile_levels, loc=mu[idx].item(), scale=sigma[idx].item())
        plt.plot(quantile_levels, true_qf, label=f'Obs {idx+1} True', linestyle='-')
    plt.xlabel('Quantile Level')
    plt.ylabel('Value')
    plt.title('Empirical (GREM) vs True Quantile Functions (5 Random Observations)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_middle_50_curves(qf, qfSupp, mu, sigma, n_curves=50):
    """
    Plot the middle 50 curves from GREM results.
    This shows quantile functions for observations in the middle range of the data.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    
    n_total = qf.shape[0]
    start_idx = max(0, (n_total - n_curves) // 2)
    end_idx = min(n_total, start_idx + n_curves)
    
    # Get middle 50 observations
    middle_indices = np.arange(start_idx, end_idx)
    
    plt.figure(figsize=(12, 8))
    quantile_levels = qfSupp.cpu().numpy() if qfSupp is not None else np.linspace(0, 1, qf.shape[1])
    
    # Plot GREM quantile functions
    for i, idx in enumerate(middle_indices):
        # GREM quantile function
        grem_qf = qf[idx].cpu().numpy()
        plt.plot(quantile_levels, grem_qf, 'b-', alpha=0.3, linewidth=0.8)
        
        # True quantile function
        true_mu = float(mu[idx])
        true_sigma = float(sigma[idx])
        true_qf = norm.ppf(quantile_levels, loc=true_mu, scale=true_sigma)
        plt.plot(quantile_levels, true_qf, 'r-', alpha=0.3, linewidth=0.8)
    
    # Add mean curves for comparison
    grem_mean = torch.mean(qf[middle_indices], dim=0).cpu().numpy()
    true_mean = np.mean([norm.ppf(quantile_levels, loc=float(mu[idx]), scale=float(sigma[idx])) 
                        for idx in middle_indices], axis=0)
    
    plt.plot(quantile_levels, grem_mean, 'b-', linewidth=3, label='GREM Mean (Middle 50)')
    plt.plot(quantile_levels, true_mean, 'r-', linewidth=3, label='True Mean (Middle 50)')
    
    plt.xlabel('Quantile Level')
    plt.ylabel('Value')
    plt.title(f'Middle {n_curves} Curves: GREM vs True Quantile Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Middle {n_curves} curves summary:")
    print(f"Observations {start_idx} to {end_idx-1} (out of {n_total})")
    print(f"GREM mean curve range: [{grem_mean.min():.3f}, {grem_mean.max():.3f}]")
    print(f"True mean curve range: [{true_mean.min():.3f}, {true_mean.max():.3f}]")
    print(f"Mean absolute difference: {np.mean(np.abs(grem_mean - true_mean)):.4f}")

def DeSI_distribution(y=None, x=None, xOut=None, h=None, optns=None):
    """
    DeSI distribution regression function.
    Local version of GREM with bandwidth parameter h.
    Now supports per-sample (vector) bandwidths for h.
    """
    if y is None or x is None or h is None:
        raise ValueError("y, x, and h (bandwidth) must be provided")
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    n = len(y)
    p = x.shape[1] if x.dim() > 1 else 1
    if x.dim() == 1:
        x = x.reshape(-1, 1)
    if x.dim() == 1:
        x = x.reshape(-1, 1)
    if optns is None:
        optns = {}
    M = optns.get('M', 100)
    def lcm(a, b):
        return abs(a * b) // torch.gcd(torch.tensor(a), torch.tensor(b)).item()
    def plcm(arr):
        result = arr[0]
        for num in arr[1:]:
            result = lcm(result, num)
        return result
    sample_sizes = [len(yi) for yi in y]
    lcm_size = plcm(sample_sizes)
    max_size = 10000
    if lcm_size > max_size:
        print(f"Warning: LCM size {lcm_size} is too large, limiting to {max_size}")
        lcm_size = max_size
    yM = torch.zeros((n, lcm_size), dtype=torch.float32)
    for i in range(n):
        yi = y[i]
        m = len(yi)
        if not isinstance(yi, torch.Tensor):
            yi = torch.tensor(yi, dtype=torch.float32)
        repeats = lcm_size // m
        if repeats == 0:
            repeats = 1
        yM[i, :] = torch.tile(yi, (repeats,))[:lcm_size]
    qf, residuals, totVa = compute_lrem_results(x, yM, M, n, p, h, optns, xOut)
    result = finalize_grem_results(qf, residuals, totVa, n, p, y, x, optns, xOut, yM, None, None)
    return result


def compute_lrem_results(xin, yM, M, n, p, h, optns=None, xout=None):
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    device = yM.device
    dtype = yM.dtype
    # Ensure xin and xout are 1D tensors (lists of scalars)
    xin = xin.squeeze()
    if xin.dim() > 1:
        xin = xin.view(-1)
    if xout is not None:
        x_pred = xout.squeeze()
        if x_pred.dim() > 1:
            x_pred = x_pred.view(-1)
        n_pred = x_pred.shape[0]
    else:
        x_pred = xin
        n_pred = xin.shape[0]
    qf_list = []
    for j in range(n_pred):
        z = x_pred[j]  # scalar
        # Support per-sample bandwidths: h can be a tensor or scalar
        h_j = h[j] if isinstance(h, torch.Tensor) and h.numel() > 1 else h
        def kernel(u):
            return torch.exp(-0.5 * (u / h_j) ** 2) / (h_j * (2 * torch.pi) ** 0.5 + 1e-8)
        diff1d = xin - z  # shape (n,)
        Kh = kernel(diff1d)
        u0 = torch.mean(Kh)
        u1 = torch.mean(Kh * diff1d)
        u2 = torch.mean(Kh * diff1d ** 2)
        sigma0_sq = u0 * u2 - u1 ** 2 + 1e-8
        sL = Kh * (u2 - u1 * diff1d) / (sigma0_sq + 1e-8)
        qNew = torch.sum(yM * sL[:, None], dim=0) / n
        qf_list.append(qNew)
    qf = torch.stack(qf_list, dim=0)
    # --- Monotonic projection using cvxpylayers (make qf monotonic) ---
    x_var = cp.Variable(M)
    q_param = cp.Parameter(M)
    A_param = cp.Parameter((M-1, M))
    l_param = cp.Parameter(M-1)
    objective = cp.Minimize(0.5 * cp.sum_squares(x_var) - q_param @ x_var)
    constraints = [A_param @ x_var >= l_param]
    problem = cp.Problem(objective, constraints)
    layer = CvxpyLayer(problem, parameters=[q_param, A_param, l_param], variables=[x_var])
    A = torch.zeros((M-1, M), dtype=dtype, device=device)
    for j in range(M-1):
        A[j, j] = -1
        A[j, j+1] = 1
    A_t = A
    l_t = torch.zeros(M-1, dtype=dtype, device=device)
    qf_proj_list = []
    for i in range(n_pred):
        q_t = qf[i, :]
        x_star, = layer(q_t, A_t, l_t, solver_args={'eps_abs': 1e-8, 'eps_rel': 1e-8, 'max_iters': 50000})
        qf_proj_list.append(x_star)
    qf = torch.stack(qf_proj_list, dim=0)
    # --- End monotonic projection ---
    if xout is None:
        residuals = torch.sqrt(torch.mean((yM - qf) ** 2, dim=1))
    else:
        residuals = torch.zeros(n_pred, dtype=dtype, device=device)
    totVa = torch.sum(yM ** 2)
    return qf, residuals, totVa

if __name__ == "__main__":
    print("Testing Local REM (LREM) implementation with PyTorch...")
    X, Y, theta, mu, sigma = generate_simulation_data_torch_true(n=1600, qf_size=100, p=4, link="quadratic", seed=7)
    n = X.shape[0]
    # Example: use a vector of bandwidths for demonstration
    h = 0.5 # bandwidths from 1 to 10 for each xout sample
    # Split data: 80% for xin/yin, 20% for xout/yout
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    idx_in = idx[:n_train]
    idx_out = idx[n_train:]
    xin = X[idx_in]
    yin = [Y[i].tolist() for i in idx_in]
    xout = X[idx_out]
    yout = [Y[i].tolist() for i in idx_out]
    mu_out = mu[idx_out]
    sigma_out = sigma[idx_out]
    # Project normalized xin and xout to scalar using theta
    xin_proj = xin @ theta
    xout_proj = xout @ theta
    # Normalize xin_proj and xout_proj using xin_proj's statistics
    proj_mean = xin_proj.mean()
    proj_std = xin_proj.std() + 1e-8
    xin_proj_norm = (xin_proj - proj_mean) / proj_std
    xout_proj_norm = (xout_proj - proj_mean) / proj_std
    # Run DeSI distribution regression with normalized projected xin/yin and xout
    desi_result_xout = DeSI_distribution(x=xin_proj_norm, y=yin, h=h, xOut=xout_proj_norm)
    qf_pred_xout = desi_result_xout['qf']
    qfSupp_xout = desi_result_xout['qfSupp']
    # Plot predicted and true quantiles for xout
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm
    plt.figure(figsize=(10, 6))
    n_plot = min(5, xout.shape[0])
    idxs = np.linspace(0, xout.shape[0] - 1, n_plot, dtype=int)
    for i, idx in enumerate(idxs):
        # True quantile function for this xout
        true_qf = norm.ppf(qfSupp_xout.cpu().numpy(), loc=mu_out[idx].item(), scale=sigma_out[idx].item())
        plt.plot(qfSupp_xout.cpu().numpy(), true_qf, 'g-', alpha=0.7, label='True Quantile' if i == 0 else None)
        plt.plot(qfSupp_xout.cpu().numpy(), qf_pred_xout[idx].cpu().numpy(), 'b--', alpha=0.7, label='LREM Predicted' if i == 0 else None)
    plt.xlabel('Quantile Level')
    plt.ylabel('Value')
    plt.title('True vs LREM Predicted Quantile Functions (xOut, 20% holdout)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("LREM xOut (holdout) test completed!")
