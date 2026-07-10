import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def _median_heuristic(x: torch.Tensor) -> float:
    """
    Compute sigma^2 for RBF kernel using median heuristic on pairwise squared Euclidean distances.
    """
    # Detach to prevent gradient flow through the bandwidth selection itself
    x = x.detach()
    d2 = torch.cdist(x, x, p=2.0) ** 2
    mask = d2 > 0
    if mask.any():
        med = torch.median(d2[mask])
    else:
        med = torch.tensor(1.0, device=x.device)
    return float(med.item())

def _rbf_kernel_matrix(x: torch.Tensor, y: torch.Tensor, sigma2: float) -> torch.Tensor:
    """
    Compute RBF kernel matrix: k(x, y) = exp(- ||x-y||^2 / sigma2)
    """
    d2 = torch.cdist(x, y, p=2.0) ** 2
    # Avoid division by zero
    sigma2 = max(sigma2, 1e-12)
    return torch.exp(-0.5 * d2 / sigma2)

def compute_neurips2023_kce_loss(
    y_true: torch.Tensor,       # [B] Ground truth labels (indices)
    probs: torch.Tensor,        # [B, C] Predicted probabilities
    features: torch.Tensor,     # [B, D] Conditioning features (Logits or Dist Params)
    sigma2_z: Optional[float] = None
) -> torch.Tensor:
    """
    Computes the Kernel Calibration Error (MMD) Loss as defined in NeurIPS 2023.
    Metric: MMD^2 [ (Y, Z), (\hat{Y}, Z) ]
    Kernel: Product Kernel k((y,z), (y',z')) = k_Y(y,y') * k_Z(z,z')
    """
    B, C = probs.shape
    
    # 1. Prepare Features Kernel (K_ZZ)
    if sigma2_z is None:
        sigma2_z = _median_heuristic(features)
    K_zz = _rbf_kernel_matrix(features, features, sigma2_z)

    # 2. Prepare Label Kernel (K_YY)
    # We use RBF on the One-Hot vs Probs representation.
    y_onehot = F.one_hot(y_true.long(), num_classes=C).float()
    
    # Calculate a common sigma for the label space using median heuristic on the union
    with torch.no_grad():
        y_all = torch.cat([y_onehot, probs], dim=0)
        sigma2_y = _median_heuristic(y_all)
        
    def k_y_func(a, b):
        return _rbf_kernel_matrix(a, b, sigma2_y)

    # Term 1: Real-Real (OneHot vs OneHot)
    K_yy_target = k_y_func(y_onehot, y_onehot)
    
    # Term 2: Pred-Pred (Probs vs Probs)
    K_yy_pred = k_y_func(probs, probs)
    
    # Term 3: Cross (OneHot vs Probs)
    K_yy_cross = k_y_func(y_onehot, probs)

    # 3. Compute MMD^2 (V-statistic for stability)
    term_target = (K_yy_target * K_zz).mean()
    term_pred = (K_yy_pred * K_zz).mean()
    term_cross = (K_yy_cross * K_zz).mean()
    
    loss_kce = term_target + term_pred - 2 * term_cross
    
    # Ensure non-negative output
    return F.relu(loss_kce)


# ================================================================
# 1. Base Softmax MLP classifier
# ================================================================
class Softmax_Classifier(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
        self.layer2 = nn.Linear(n_hidden, n_hidden // 2, bias=True)
        self.layer3 = nn.Linear(n_hidden // 2, output_dim, bias=True)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)

        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()

    def forward(self, x, return_probs=False, t=0.0): # t kept for compatibility
        residual = self.projection(x)
        x = torch.relu(self.layer1(x))
        x = x + residual
        if self.is_dropout: x = self.dropout(x)

        x = torch.relu(self.layer2(x))
        if self.is_dropout: x = self.dropout(x)

        logits = self.layer3(x)

        if return_probs:
            return F.softmax(logits, dim=1)
        else:
            return logits

    def loss_softmax_classifier(self, x, y, t=0.0):
        logits = self.forward(x, return_probs=False)
        loss = F.cross_entropy(logits, y)
        return loss


class SoftmaxMMD_Classifier(Softmax_Classifier):
    def loss_with_mmd(self, x, y, lambda_kce: float = 0.1, t: float = 0.0, sigma2_z: float = None):
        # 1. Get Logits (Features)
        logits = self.forward(x, return_probs=False)
        probs = F.softmax(logits, dim=1)
        
        # 2. NLL Loss
        loss_nll = F.cross_entropy(logits, y)
        
        # 3. KCE (MMD) Loss
        # USE LOGITS AS FEATURES Z
        kce = compute_neurips2023_kce_loss(
            y_true=y, probs=probs, features=logits, sigma2_z=sigma2_z
        )
        
        loss = loss_nll + lambda_kce * kce
        return loss, {"nll": loss_nll.detach(), "kce": kce.detach()}


class ICSoftmax_Classifier(nn.Module):
    """
    IC version of Softmax Classifier with Gradient Detachment.
    """
    def __init__(self, input_dim, n_hidden, output_dim, q_dim=2, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)
        
        self.layer1_q = nn.Linear(1, 2 * q_dim)
        self.layer2_q = nn.Linear(2 * q_dim, q_dim)
        self.q_dim = q_dim

        self.layer2 = nn.Linear(n_hidden + q_dim, n_hidden // 2, bias=True)
        self.layer3 = nn.Linear(n_hidden // 2 + q_dim, output_dim, bias=True)

        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()

    def extract_features(self, x):
        residual = self.projection(x)
        x_feat = torch.relu(self.layer1(x)) + residual
        if self.is_dropout: x_feat = self.dropout(x_feat)
        return x_feat

    def predict_from_features(self, x_feat, q, return_probs=False):
        q_feat = torch.exp(self.layer2_q(torch.relu(self.layer1_q(q))))
        x_cat1 = torch.cat([x_feat, q_feat], dim=1)
        x_hidden = torch.relu(self.layer2(x_cat1))
        if self.is_dropout: x_hidden = self.dropout(x_hidden)
        x_cat2 = torch.cat([x_hidden, q_feat], dim=1)
        logits = self.layer3(x_cat2)
        return F.softmax(logits, dim=1) if return_probs else logits

    def forward(self, x, q=None, return_probs=False, t=0.0):
        if q is None: q = torch.ones(x.shape[0], 1, device=x.device)
        x_feat = self.extract_features(x)
        return self.predict_from_features(x_feat, q, return_probs)

    def loss_icsoftmax_classifier(self, x, y, q_unused, lambda_reg=0.9, t=0.0):
        # 1. Classification (Gradient Flowing)
        q_clean = torch.ones(x.shape[0], 1, device=x.device)
        probs_clean = self.forward(x, q=q_clean, return_probs=True, t=t)
        loss_nll = F.nll_loss((probs_clean+1e-8).log(), y)

        # 2. Calibration (Gradient Detached from Backbone)
        K = 5
        x_aug = x.repeat_interleave(K, dim=0)
        q_aug = torch.rand(x_aug.shape[0], 1, device=x.device)

        with torch.no_grad():
            x_feat_aug = self.extract_features(x_aug)
        x_feat_aug = x_feat_aug.detach() # Explicit detach

        probs_aug = self.predict_from_features(x_feat_aug, q_aug, return_probs=True)
        confidences_aug, _ = torch.max(probs_aug, dim=1)
        loss_cal = torch.mean(torch.abs(confidences_aug - q_aug.view(-1)))

        return lambda_reg * loss_nll + (1 - lambda_reg) * loss_cal


# ================================================================
# 2. ALD-based models
# ================================================================
class ALD_Classifier(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
        self.layer2_theta = nn.Linear(n_hidden, n_hidden // 2, bias=True)
        self.layer2_sigma = nn.Linear(n_hidden, n_hidden // 2, bias=True)
        self.layer2_kappa = nn.Linear(n_hidden, n_hidden // 2, bias=True)
        self.layer3_theta = nn.Linear(n_hidden // 2, output_dim, bias=True)
        self.layer3_sigma = nn.Linear(n_hidden // 2, output_dim, bias=True)
        self.layer3_kappa = nn.Linear(n_hidden // 2, output_dim, bias=True)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)

        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()

    def forward(self, x, return_probs=False, t=0.0):
        residual = self.projection(x)
        x = torch.relu(self.layer1(x)) + residual
        if self.is_dropout: x = self.dropout(x)

        theta = torch.relu(self.layer2_theta(x))
        if self.is_dropout: theta = self.dropout(theta)
        theta = self.layer3_theta(theta)

        sigma = torch.relu(self.layer2_sigma(x))
        if self.is_dropout: sigma = self.dropout(sigma)
        sigma = torch.exp(self.layer3_sigma(sigma).clamp(min=-9.0, max=10.0))

        kappa = torch.relu(self.layer2_kappa(x))
        if self.is_dropout: kappa = self.dropout(kappa)
        kappa = torch.exp(self.layer3_kappa(kappa).clamp(min=-9.0, max=10.0))

        if return_probs:
            return self.ald_probs(theta, sigma, kappa, t=t)
        else:
            return theta, sigma, kappa

    def ald_probs(self, theta, sigma, kappa, t=0.0):
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=theta.device))
        if isinstance(t, (int, float)): t = torch.tensor(t, device=theta.device)
        if t.ndim == 0: t = t.view(1, 1).expand_as(theta)
        cond = (t > theta)
        z1 = -sqrt2 * kappa * (t - theta) / sigma
        z2 = -sqrt2 * (theta - t) / (sigma * kappa)
        z1, z2 = z1.clamp(-50, 50), z2.clamp(-50, 50)
        exp1, exp2 = torch.exp(z1), torch.exp(z2)
        term1 = 1 - 1 / (1 + kappa ** 2) * exp1
        term2 = kappa ** 2 / (1 + kappa ** 2) * exp2
        ald_cdf = torch.where(cond, term1, term2).clamp(1e-6, 1-1e-6)
        return ald_cdf / (ald_cdf.sum(dim=1, keepdim=True) + 1e-8)

    def loss_ald_classifier(self, x, y, t=0.0):
        probs = self.forward(x, return_probs=True, t=t)
        return F.nll_loss((probs+1e-8).log(), y)


class ALDMMD_Classifier(ALD_Classifier):
    def loss_with_mmd(self, x, y, lambda_kce: float = 0.1, t: float = 0.0, sigma2_z: float = None):
        # 1. Get raw Parameters
        theta, sigma, kappa = self.forward(x, return_probs=False)
        # 2. Get Probs
        probs = self.ald_probs(theta, sigma, kappa, t=t)
        
        loss_nll = F.nll_loss((probs + 1e-8).log(), y)
        
        # 3. KCE (MMD)
        # USE RAW PARAMETERS AS FEATURES Z (More informative than probs)
        features = torch.cat([theta, sigma, kappa], dim=1)
        kce = compute_neurips2023_kce_loss(y, probs, features, sigma2_z)
        
        loss = loss_nll + lambda_kce * kce
        return loss, {"nll": loss_nll.detach(), "kce": kce.detach()}


class ICALD_Classifier(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, q_dim=1, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)

        self.layer1_q = nn.Linear(1, 2 * q_dim)
        self.layer2_q = nn.Linear(2 * q_dim, q_dim)
        self.q_dim = q_dim

        self.layer2_theta = nn.Linear(n_hidden + q_dim, n_hidden // 2)
        self.bn2_theta = nn.BatchNorm1d(n_hidden // 2)
        self.layer2_sigma = nn.Linear(n_hidden + q_dim, n_hidden // 2)
        self.bn2_sigma = nn.BatchNorm1d(n_hidden // 2)
        self.layer2_kappa = nn.Linear(n_hidden + q_dim, n_hidden // 2)
        self.bn2_kappa = nn.BatchNorm1d(n_hidden // 2)
        self.layer3_theta = nn.Linear(n_hidden // 2 + q_dim, output_dim)
        self.layer3_sigma = nn.Linear(n_hidden // 2 + q_dim, output_dim)
        self.layer3_kappa = nn.Linear(n_hidden // 2 + q_dim, output_dim)

        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()

    def extract_features(self, x):
        res = self.projection(x)
        x = self.layer1(x)
        x = self.bn1(x)
        return torch.relu(x) + res

    def predict_from_features(self, features, q, return_probs=False, t=0.0):
        q_emb = torch.exp(self.layer2_q(torch.relu(self.layer1_q(q))))
        x_cat = torch.cat([features, q_emb], dim=1)

        def branch(l2, bn, l3):
            h = l2(x_cat)
            h = bn(h)
            h = torch.relu(h)
            if self.is_dropout: h = self.dropout(h)
            return l3(torch.cat([h, q_emb], dim=1))

        theta = branch(self.layer2_theta, self.bn2_theta, self.layer3_theta)
        sigma = torch.exp(branch(self.layer2_sigma, self.bn2_sigma, self.layer3_sigma).clamp(-9, 10))
        kappa = torch.exp(branch(self.layer2_kappa, self.bn2_kappa, self.layer3_kappa).clamp(-9, 10))

        if return_probs:
            return self.ald_probs(theta, sigma, kappa, t)
        else:
            return theta, sigma, kappa

    def forward(self, x, q=None, return_probs=False, t=0.0):
        if q is None: q = torch.ones(x.shape[0], 1, device=x.device)
        feat = self.extract_features(x)
        return self.predict_from_features(feat, q, return_probs, t)

    def ald_probs(self, theta, sigma, kappa, t=0.0):
        # Re-use the same logic as ALD_Classifier but inline to avoid inheritance complexity
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=theta.device))
        if isinstance(t, (int, float)): t = torch.tensor(t, device=theta.device)
        if t.ndim == 0: t = t.view(1, 1).expand_as(theta)
        cond = (t > theta)
        z1 = (-sqrt2 * kappa * (t - theta) / sigma).clamp(-50, 50)
        z2 = (-sqrt2 * (theta - t) / (sigma * kappa)).clamp(-50, 50)
        exp1, exp2 = torch.exp(z1), torch.exp(z2)
        cdf = torch.where(cond, 1 - 1/(1+kappa**2)*exp1, kappa**2/(1+kappa**2)*exp2).clamp(1e-6, 1-1e-6)
        return cdf / (cdf.sum(1, keepdim=True) + 1e-8)

    def loss_icald_classifier(self, x, y, q_unused=None, lambda_reg=0.9, t=0.0):
        q_clean = torch.ones(x.shape[0], 1, device=x.device)
        probs_clean = self.forward(x, q=q_clean, return_probs=True, t=t)
        loss_nll = F.nll_loss((probs_clean+1e-8).log(), y)

        K = 5
        x_aug = x.repeat_interleave(K, dim=0)
        q_aug = torch.rand(x_aug.shape[0], 1, device=x.device)
        
        with torch.no_grad():
            feat_aug = self.extract_features(x_aug)
        feat_aug = feat_aug.detach() # Gradient Stop

        probs_aug = self.predict_from_features(feat_aug, q_aug, return_probs=True, t=t)
        conf, _ = probs_aug.max(1)
        loss_cal = (conf - q_aug.view(-1)).abs().mean()

        return lambda_reg * loss_nll + (1 - lambda_reg) * loss_cal


# ================================================================
# 5. Norm Models
# ================================================================
class Norm_Classifier(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()
        self.layer2_mu = nn.Linear(n_hidden, n_hidden//2)
        self.layer2_sigma = nn.Linear(n_hidden, n_hidden//2)
        self.layer3_mu = nn.Linear(n_hidden//2, output_dim)
        self.layer3_sigma = nn.Linear(n_hidden//2, output_dim)

    def forward(self, x, return_probs=False, t=0.0):
        res = self.projection(x)
        x = torch.relu(self.layer1(x)) + res
        if self.is_dropout: x = self.dropout(x)
        
        mu = self.layer3_mu(torch.relu(self.layer2_mu(x)))
        sigma = torch.exp(self.layer3_sigma(torch.relu(self.layer2_sigma(x))).clamp(-9, 10))
        
        if return_probs: return self.norm_probs(mu, sigma, t)
        return mu, sigma

    def norm_probs(self, mu, sigma, t=0.0):
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=mu.device))
        if isinstance(t, (int, float)): t = torch.tensor(t, device=mu.device)
        if t.ndim == 0: t = t.view(1, 1).expand_as(mu)
        z = (t - mu) / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / sqrt2)).clamp(1e-6, 1-1e-6)
        return cdf / (cdf.sum(1, keepdim=True) + 1e-8)

    def loss_norm_classifier(self, x, y, t=0.0):
        probs = self.forward(x, return_probs=True, t=t)
        return F.nll_loss((probs+1e-8).log(), y)

class NormMMD_Classifier(Norm_Classifier):
    def loss_with_mmd(self, x, y, lambda_kce=0.1, t=0.0, sigma2_z=None):
        mu, sigma = self.forward(x, return_probs=False)
        probs = self.norm_probs(mu, sigma, t)
        loss_nll = F.nll_loss((probs+1e-8).log(), y)
        features = torch.cat([mu, sigma], dim=1)
        kce = compute_neurips2023_kce_loss(y, probs, features, sigma2_z)
        return loss_nll + lambda_kce * kce, {"nll": loss_nll.detach(), "kce": kce.detach()}


# ================================================================
# 6. Exp Models
# ================================================================
class Exp_Classifier(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)
        self.layer2_lam = nn.Linear(n_hidden, n_hidden//2)
        self.layer3_lam = nn.Linear(n_hidden//2, output_dim)
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()

    def forward(self, x, return_probs=False, t=0.0):
        res = self.projection(x)
        x = torch.relu(self.layer1(x)) + res
        if self.is_dropout: x = self.dropout(x)
        lam = torch.exp(self.layer3_lam(torch.relu(self.layer2_lam(x))).clamp(-9, 10))
        if return_probs: return self.exp_probs(lam, t)
        return lam

    def exp_probs(self, lam, t):
        if isinstance(t, (int, float)): t = torch.tensor(t, device=lam.device)
        if t.ndim == 0: t = t.view(1, 1).expand_as(lam)
        cdf = (1.0 - torch.exp(-lam * t.clamp(min=0.0))).clamp(1e-6, 1-1e-6)
        return cdf / (cdf.sum(1, keepdim=True) + 1e-8)

    def loss_exp_classifier(self, x, y, t=0.0):
        return F.nll_loss((self.forward(x, return_probs=True, t=t)+1e-8).log(), y)


# ================================================================
# 7. Weibull Models
# ================================================================
class Weibull_Classifier(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)
        self.layer2_k = nn.Linear(n_hidden, n_hidden//2)
        self.layer2_lam = nn.Linear(n_hidden, n_hidden//2)
        self.layer3_k = nn.Linear(n_hidden//2, output_dim)
        self.layer3_lam = nn.Linear(n_hidden//2, output_dim)
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()

    def forward(self, x, return_probs=False, t=0.0):
        res = self.projection(x)
        x = torch.relu(self.layer1(x)) + res
        if self.is_dropout: x = self.dropout(x)
        k = F.softplus(self.layer3_k(torch.relu(self.layer2_k(x)))).clamp(1e-3, 10)
        lam = F.softplus(self.layer3_lam(torch.relu(self.layer2_lam(x)))).clamp(1e-3, 10)
        if return_probs: return self.weibull_probs(k, lam, t)
        return k, lam

    def weibull_probs(self, k, lam, t):
        if isinstance(t, (int, float)): t = torch.tensor(t, device=k.device)
        if t.ndim == 0: t = t.view(1, 1).expand_as(k)
        t_pos = t.clamp(min=0.0)
        ratio = (t_pos/lam).clamp(1e-6, 1e3)
        cdf = (1.0 - torch.exp(-torch.pow(ratio, k).clamp(0, 50))).clamp(1e-6, 1-1e-6)
        return cdf / (cdf.sum(1, keepdim=True) + 1e-8)

    def loss_weibull_classifier(self, x, y, t=0.0):
        return F.nll_loss((self.forward(x, return_probs=True, t=t)+1e-8).log(), y)



class LogNorm_Classifier(nn.Module):
    """
    Predict per-class LogNormal(mu, sigma), where Y>0 and log(Y) ~ N(mu, sigma^2).
    """
    def __init__(self, input_dim, n_hidden, output_dim, is_dropout=True, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
        self.layer2_mu = nn.Linear(n_hidden, n_hidden // 2, bias=True)
        self.layer2_sigma = nn.Linear(n_hidden, n_hidden // 2, bias=True)
        self.layer3_mu = nn.Linear(n_hidden // 2, output_dim, bias=True)
        self.layer3_sigma = nn.Linear(n_hidden // 2, output_dim, bias=True)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)

        self.is_dropout = is_dropout
        self.dropout_rate = dropout_rate
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_probs=False, t=0.0):
        residual = self.projection(x)
        x = torch.relu(self.layer1(x))
        x = x + residual

        if self.is_dropout:
            x = self.dropout(x)

        mu = torch.relu(self.layer2_mu(x))
        if self.is_dropout:
            mu = self.dropout(mu)
        mu = self.layer3_mu(mu)

        sigma = torch.relu(self.layer2_sigma(x))
        if self.is_dropout:
            sigma = self.dropout(sigma)
        sigma = torch.exp(self.layer3_sigma(sigma).clamp(min=-9.0, max=10.0))

        if return_probs:
            probs = self.lognorm_probs(mu, sigma, t=t)
            return probs
        else:
            return mu, sigma

    def lognorm_cdf(self, t, mu, sigma):
        eps = 1e-8
        t_pos = t.clamp(min=eps)
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=mu.dtype, device=mu.device))
        z = (torch.log(t_pos) - mu) / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / sqrt2))
        cdf = torch.where(t <= 0, torch.zeros_like(cdf), cdf)
        return cdf

    def lognorm_probs(self, mu, sigma, t=0.0):
        if isinstance(t, (int, float)):
            t = torch.tensor(t, dtype=mu.dtype, device=mu.device)
        if t.ndim == 0:
            t = t.view(1, 1).expand_as(mu)

        cdf_t = self.lognorm_cdf(t, mu, sigma)
        cdf_t = cdf_t.clamp(min=1e-6, max=1.0 - 1e-6)

        probs = cdf_t / (cdf_t.sum(dim=1, keepdim=True) + 1e-8)
        return probs

    def loss_lognorm_classifier(self, x, y, t=0.0):
        probs = self.forward(x, return_probs=True, t=t)
        logits = torch.log(probs + 1e-8)
        loss = F.nll_loss(logits, y)
        return loss


# ================================================================
# 8. Logistic Models
# ================================================================
class Logistic_Classifier(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim, is_dropout=False, dropout_rate=None):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, n_hidden)
        self.projection = nn.Linear(input_dim, n_hidden, bias=False)
        self.layer2_mu = nn.Linear(n_hidden, n_hidden//2)
        self.layer2_s = nn.Linear(n_hidden, n_hidden//2)
        self.layer3_mu = nn.Linear(n_hidden//2, output_dim)
        self.layer3_s = nn.Linear(n_hidden//2, output_dim)
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(dropout_rate) if is_dropout else nn.Identity()

    def forward(self, x, return_probs=False, t=0.0):
        res = self.projection(x)
        x = torch.relu(self.layer1(x)) + res
        if self.is_dropout: x = self.dropout(x)
        mu = self.layer3_mu(torch.relu(self.layer2_mu(x)))
        s = torch.exp(self.layer3_s(torch.relu(self.layer2_s(x))).clamp(-9, 10))
        if return_probs: return self.logistic_probs(mu, s, t)
        return mu, s

    def logistic_probs(self, mu, s, t):
        if isinstance(t, (int, float)): t = torch.tensor(t, device=mu.device)
        if t.ndim == 0: t = t.view(1, 1).expand_as(mu)
        cdf = torch.sigmoid((t - mu) / s).clamp(1e-6, 1-1e-6)
        return cdf / (cdf.sum(1, keepdim=True) + 1e-8)

    def loss_logistic_classifier(self, x, y, t=0.0):
        return F.nll_loss((self.forward(x, return_probs=True, t=t)+1e-8).log(), y)

class LogisticMMD_Classifier(Logistic_Classifier):
    def loss_with_mmd(self, x, y, lambda_kce=0.1, t=0.0, sigma2_z=None):
        mu, s = self.forward(x, return_probs=False)
        probs = self.logistic_probs(mu, s, t)
        loss_nll = F.nll_loss((probs+1e-8).log(), y)
        features = torch.cat([mu, s], dim=1)
        kce = compute_neurips2023_kce_loss(y, probs, features, sigma2_z)
        return loss_nll + lambda_kce * kce, {"nll": loss_nll.detach(), "kce": kce.detach()}

