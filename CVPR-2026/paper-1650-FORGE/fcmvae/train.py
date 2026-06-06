import math
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    log_sigma = torch.clamp(log_sigma, min=math.log(1e-4), max=math.log(10.0))
    sigma2 = torch.exp(2.0 * log_sigma)
    return 0.5 * (((x - mean) ** 2) / sigma2 + 2.0 * log_sigma + math.log(2.0 * math.pi)).mean()


def fisher_intensity(fc: torch.Tensor, fisher_mu: torch.Tensor, fisher_sigma: torch.Tensor, max_log_intensity: float = 8.0) -> torch.Tensor:
    z = torch.atanh(torch.clamp(fc, min=-1.0 + 1e-4, max=1.0 - 1e-4))
    log_intensity = (z - fisher_mu) / torch.clamp(fisher_sigma, min=1e-6)
    return torch.exp(torch.clamp(log_intensity, min=-max_log_intensity, max=max_log_intensity))


def poisson_intensity_nll(log_rate: torch.Tensor, fc_true: torch.Tensor, fisher_mu: torch.Tensor, fisher_sigma: torch.Tensor) -> torch.Tensor:
    target = fisher_intensity(fc_true, fisher_mu, fisher_sigma)
    loss = F.poisson_nll_loss(log_rate, target, log_input=True, full=False, reduction="none")
    n = loss.shape[-1]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=loss.device)
    return loss[:, off_diag].mean()


def kl_diag_gaussians(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kl.sum(dim=1).mean()


def supervised_elbo_loss(
    log_rate: torch.Tensor,
    fc_true: torch.Tensor,
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
    fisher_mu: torch.Tensor,
    fisher_sigma: torch.Tensor,
    log_sigma_y: torch.Tensor,
    beta_kl: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rec = poisson_intensity_nll(log_rate, fc_true, fisher_mu, fisher_sigma)
    y_nll = gaussian_nll(y_true.view(-1, 1), y_hat, log_sigma_y)
    kld = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    loss = rec + y_nll + float(beta_kl) * kld
    return loss, {"rec_poisson": float(rec.item()), "y_nll": float(y_nll.item()), "kld": float(kld.item())}


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device, beta_kl: float = 2.0) -> float:
    model.train()
    total = 0.0
    n_samples = 0
    for x_vec, trait, a in loader:
        x_vec = x_vec.to(device)
        trait = trait.to(device)
        a_prior = a.to(device)
        y_cls = trait.view(-1)
        fc_true = x_vec.view(-1, model.n_node, model.n_node)
        optimizer.zero_grad()
        log_rate, mu_q, logvar_q, y_hat, _, _, mu_p, logvar_p, _ = model(x_vec, a_prior, labels=y_cls)
        loss, _ = supervised_elbo_loss(
            log_rate,
            fc_true,
            y_hat,
            trait,
            mu_q,
            logvar_q,
            mu_p,
            logvar_p,
            model.fisher_mu,
            model.fisher_sigma,
            model.log_sigma_y,
            beta_kl=beta_kl,
        )
        loss.backward()
        optimizer.step()
        total += loss.item() * x_vec.shape[0]
        n_samples += x_vec.shape[0]
    return total / max(1, n_samples)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, beta_kl: float = 2.0) -> Tuple[float, float]:
    model.eval()
    total = 0.0
    n_samples = 0
    pred_trait, true_trait = [], []
    for x_vec, trait, a in loader:
        x_vec = x_vec.to(device)
        trait = trait.to(device)
        a_prior = a.to(device)
        y_cls = trait.view(-1)
        fc_true = x_vec.view(-1, model.n_node, model.n_node)
        log_rate, mu_q, logvar_q, y_hat, _, _, mu_p, logvar_p, _ = model(x_vec, a_prior, labels=y_cls)
        loss, _ = supervised_elbo_loss(
            log_rate,
            fc_true,
            y_hat,
            trait,
            mu_q,
            logvar_q,
            mu_p,
            logvar_p,
            model.fisher_mu,
            model.fisher_sigma,
            model.log_sigma_y,
            beta_kl=beta_kl,
        )
        total += loss.item() * x_vec.shape[0]
        n_samples += x_vec.shape[0]
        pred_trait.append(y_hat.view(-1).cpu())
        true_trait.append(trait.view(-1).cpu())
    total_loss = total / max(1, n_samples)
    if len(pred_trait) == 0:
        return total_loss, 0.0
    return total_loss, F.mse_loss(torch.cat(pred_trait), torch.cat(true_trait)).item()
