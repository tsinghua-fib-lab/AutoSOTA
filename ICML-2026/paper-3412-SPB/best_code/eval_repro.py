"""Reproduction evaluation script for "Symmetries in PAC-Bayesian Learning"

Reproduces KL divergence, McAllester PAC-Bayes bound, and test risk for
Rotated MNIST with SO(2) symmetry and C8 equivariant CNN.

Usage: python eval_repro.py
"""
import sys, os, torch, numpy as np, math
from torch.utils.data import DataLoader

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.params import get_flat_params, set_flat_params
from utils.prior_posterior import GaussianPrior, GaussianPosterior
from utils.pacbayes_utils import kl_diag_gaussians, mcallester_bound
from create_data.dataset import TransformedDataset
from models.models_mnist import BaselineCNN, EquivariantCNN

# --- Configuration ---
DATA_DIR = "create_data/rot_mnist"
DEVICE = "cpu"
S = 200       # Monte Carlo samples
DELTA = 0.05   # confidence parameter
PRIOR_SIGMA = 0.05
POST_SIGMA = 0.05


def estimate_gibbs_risk(model, mu_flat, sigma_q, loader, device, S):
    """Estimate Gibbs 0-1 risk by Monte Carlo sampling from posterior."""
    torch.manual_seed(0)
    model.eval()
    mu_flat = mu_flat.to(device)
    losses = []
    for s in range(S):
        eps = torch.randn_like(mu_flat) * sigma_q
        w_sample = mu_flat + eps
        incorrect, total = 0, 0
        for batch in loader:
            imgs, labels = batch["x"].to(device), batch["y"].to(device)
            set_flat_params(model, w_sample)
            with torch.no_grad():
                preds = model(imgs).argmax(dim=1)
            incorrect += (preds != labels).sum().item()
            total += labels.size(0)
        losses.append(incorrect / total)
    losses = np.array(losses)
    return float(losses.mean()), float(losses.std(ddof=1) / math.sqrt(len(losses)))


def evaluate_model(model_cls, checkpoint_path, prior_path, label):
    """Run full PAC-Bayes evaluation for one model."""
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    train_loader = DataLoader(TransformedDataset(f"{DATA_DIR}/train.pt"), batch_size=256)
    test_loader = DataLoader(TransformedDataset(f"{DATA_DIR}/test.pt"), batch_size=256)

    prior_data = torch.load(prior_path, map_location=DEVICE)
    prior = GaussianPrior(prior_data["mu"], PRIOR_SIGMA)
    posterior = GaussianPosterior(model, get_flat_params(model), sigma=POST_SIGMA)

    # KL divergence (deterministic)
    kl = kl_diag_gaussians(posterior.mu, posterior.sigma, prior.mu, prior.sigma)

    # Empirical Gibbs risk (on training set) — used for bound
    train_risk, _ = estimate_gibbs_risk(model, posterior.mu, posterior.sigma,
                                        train_loader, DEVICE, S)

    # True Gibbs risk (on test set)
    test_risk, test_stderr = estimate_gibbs_risk(model, posterior.mu, posterior.sigma,
                                                  test_loader, DEVICE, S)

    # McAllester bound
    n_train = len(train_loader.dataset)
    bound, complexity = mcallester_bound(train_risk, kl, n_train, DELTA)

    print(f"\n{'='*60}")
    print(f"Results for {label}:")
    print(f"  KL Divergence:      {kl:.4f}")
    print(f"  McAllester Bound:   {bound:.4f}")
    print(f"  Complexity Term:    {complexity:.4f}")
    print(f"  Test Risk (Gibbs):  {test_risk:.4f} +/- {test_stderr:.4f}")

    return {"kl": kl, "bound": bound, "complexity": complexity,
            "test_risk": test_risk, "test_stderr": test_stderr}


if __name__ == "__main__":
    print("=" * 60)
    print("Symmetries in PAC-Bayesian Learning — Reproduction Eval")
    print(f"Dataset: Rotated MNIST (SO(2), C8 approx)")
    print(f"Settings: S={S}, delta={DELTA}, prior_sigma={PRIOR_SIGMA}")
    print("=" * 60)

    res_base = evaluate_model(
        BaselineCNN,
        f"{DATA_DIR}/baseline.pt",
        f"{DATA_DIR}/prior_mu_baseline.pt",
        "Baseline CNN"
    )

    res_eq = evaluate_model(
        EquivariantCNN,
        f"{DATA_DIR}/equivariant.pt",
        f"{DATA_DIR}/prior_mu_equivariant.pt",
        "Equivariant CNN (C8)"
    )

    print(f"\n{'='*60}")
    print("Comparison with Paper (Table 1, Rotated MNIST SO(2)):")
    print(f"  KL:          Our={res_eq['kl']:.1f}  Paper=7804.3")
    print(f"  Bound:       Our={res_eq['bound']:.4f}  Paper=0.505")
    print(f"  Test Risk:   Our={res_eq['test_risk']:.4f}  Paper=0.219")
    print(f"{'='*60}")
