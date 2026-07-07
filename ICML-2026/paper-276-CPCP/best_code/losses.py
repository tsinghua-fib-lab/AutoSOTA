import torch
import torch.nn as nn

# Naive pinball loss
def pinball_loss(y_pred, y_true, tau):
    """Standard quantile regression loss."""
    diff = y_true - y_pred
    return torch.max(tau * diff, (tau - 1) * diff)

# Asymmetric laplace distribution negative log likelihood for CQR
def ald_loss_cqr_nd(preds, y_true, taus):
    """Loss for CQR (N-dimensional) with Aleatoric uncertainty estimation."""
    D = y_true.shape[1]
    q_lo = preds[:, :D]
    q_hi = preds[:, D:2*D]
    sigma = nn.Softplus()(preds[:, 2*D:]) + 1e-4
    
    loss = pinball_loss(q_lo, y_true, taus[0]) + pinball_loss(q_hi, y_true, taus[1])
    nll = (loss / sigma) + 2 * torch.log(sigma)
    return nll.mean()

# Asymmetric laplace distribution negative log likelihood for RCP
def ald_loss_rcp(preds, s_true, tau):
    """Loss for Rectified Conformal Prediction score regression."""
    q = preds[:, 0:1]
    sigma = nn.Softplus()(preds[:, 1:2]) + 1e-4
    return (pinball_loss(q, s_true, tau) / sigma + torch.log(sigma)).mean()

# NLL loss of Gaussian, robust version for better numerical stability
def multivariate_nll_loss_robust(mu, L, y_true):
    diag_elements = torch.diagonal(L, dim1=1, dim2=2)
    log_det = 2 * torch.sum(torch.log(diag_elements + 1e-6), dim=1) 
    diff = (y_true - mu).unsqueeze(2) # (N, D, 1)    
    try:
        z = torch.linalg.solve_triangular(L, diff, upper=False)
    except RuntimeError:
        return torch.tensor(1e6, device=y_true.device, requires_grad=True)
            
    mahalanobis_sq = torch.sum(z.squeeze(2)**2, dim=1)
    loss = 0.5 * (log_det + mahalanobis_sq).mean()
    return loss

# NLL loss of Gaussian
def multivariate_nll_loss(mu, L, y_true):
    """
    Negative Log Likelihood for Multivariate Gaussian.
    L is the Cholesky factor of Sigma (Sigma = L @ L.T).
    """
    diag_elements = torch.diagonal(L, dim1=1, dim2=2)
    log_det = 2 * torch.sum(torch.log(diag_elements), dim=1)
    
    # Mahalanobis Term: (y - mu)^T * Sigma^-1 * (y - mu)
    # Solved via: || L^-1 * (y - mu) ||^2
    diff = (y_true - mu).unsqueeze(2)
    z = torch.linalg.solve_triangular(L, diff, upper=False)
    mahalanobis_sq = torch.sum(z.squeeze(2)**2, dim=1)
    
    loss = 0.5 * (log_det + mahalanobis_sq)
    return loss.mean()