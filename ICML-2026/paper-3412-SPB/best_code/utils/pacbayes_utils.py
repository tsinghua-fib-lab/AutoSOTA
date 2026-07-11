# Utilities for Monte-Carlo Gibbs-loss estimation, and McAllester PAC-Bayes bound.

import torch
import numpy as np
import math

from tqdm import tqdm

from utils.params import set_flat_params

def kl_diag_gaussians(mu_q, sigma_q, mu_p, sigma_p):
    """
    KL(Q||P), Q = N(mu_q, sigma_q^2 I), P = N(mu_p, sigma_p^2 I)
    """
    D = mu_q.numel()
    var_q = float(sigma_q**2)
    var_p = float(sigma_p**2)
    term1 = D * math.log(sigma_p / sigma_q)
    term2 = (D * var_q + float(((mu_q - mu_p).float()**2).sum())) / (2.0 * var_p)
    term3 = -0.5 * D
    return float(term1 + term2 + term3)

def mcallester_bound(hat_Rn_Q, kl, n, delta=0.05):
    """
    McAllester PAC-Bayes bound (as used in the project):
      E_w[R(w)] <= hat_Rn_Q + sqrt((KL + log(n/delta)) / (2*(n-1)))
    Returns (bound, complexity_term).
    """
    term = (kl + math.log(n / delta)) / (2.0 * max(1, (n - 1)))
    complexity = math.sqrt(max(0.0, term))
    return hat_Rn_Q + complexity, complexity

def predict_with_weights(model, flat_w, inputs, device='cpu'):
    # Load flat weights into model, run a forward pass and return predicted classes (CPU tensor).
    set_flat_params(model, flat_w)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.to(device))
        preds = outputs.argmax(dim=1)
    return preds.cpu()

def estimate_gibbs_loss(model, mu_flat, sigma_q, data_loader, device='cpu', S=200, fast=False):
    """
    Monte-Carlo estimate of empirical Gibbs 0-1 loss.
    Returns: (hat_Rn_Q, stderr, losses_per_sample)
    - S: number of posterior weight samples
    - hat_Rn_Q: mean empirical Gibbs loss over S samples
    - stderr: standard error over the S estimates
    """
    seed = 0

    torch.manual_seed(seed)
    model = model.to(device)
    mu_flat = mu_flat.to(device)
    losses = []

    n = len(data_loader.dataset)

    for s in tqdm(range(S), desc=f"Estimating Gibb's loss", unit="sample"):
        # sample weights from Q = N(mu, sigma_q^2 I)
        eps = torch.randn_like(mu_flat) * sigma_q
        w_sample = mu_flat + eps
        total = 0
        incorrect = 0
        # iterate over dataset to compute 0-1 loss for this sampled weight
        for i, batch in enumerate(data_loader):
            imgs = batch["x"].to(device)
            labels = batch["y"].to(device)
            preds = predict_with_weights(model, w_sample, imgs, device=device).to(device)
            incorrect += (preds != labels).sum().item()
            total += labels.size(0)
            if fast and i > 4:
                break
        losses.append(incorrect / total)

    losses = np.array(losses)
    hat_Rn_Q = float(losses.mean())
    stderr = float(losses.std(ddof=1) / math.sqrt(len(losses)))
    return hat_Rn_Q, stderr, losses

# Optional helper: sweep sigma_q and record bound, kl, gibbs loss
def sweep_sigma_q_and_compute_curve(model, mu_flat, mu_p, sigma_p, train_loader, device='cpu',
                                    sigmas=None, S=200, n=None, delta=0.05):
    """
     For each sigma_q in `sigmas`:
       - compute KL(Q||P) (diagonal Gaussian)
       - estimate empirical Gibbs loss via Monte-Carlo
       - compute McAllester bound
     Returns list of result dicts for plotting/analysis.
     """
    if sigmas is None:
        sigmas = np.logspace(-6, -1, 12)
    if n is None:
        # infer n from train_loader if not provided
        try:
            n = len(train_loader.dataset)
        except:
            raise ValueError("Provide n or a train_loader with dataset length")
    results = []
    for sigma_q in sigmas:
        sigma_p = sigma_q
        kl = kl_diag_gaussians(mu_flat, sigma_q, mu_p, sigma_p)
        hat_Rn_Q, stderr, losses = estimate_gibbs_loss(model, mu_flat, sigma_q, train_loader, device=device, S=S)
        bound, complexity = mcallester_bound(hat_Rn_Q, kl, n, delta=delta)
        results.append({
            'sigma_q': float(sigma_q),
            'kl': kl,
            'hat_Rn_Q': hat_Rn_Q,
            'stderr': stderr,
            'bound': bound,
            'complexity': complexity
        })
        # short progress print for experiments
        print(f"sigma_q={sigma_q:.1e}  kl={kl:.3e}  hat_Rn_Q={hat_Rn_Q:.4f}  bound={bound:.4f}")
    return results
