"""Evaluation for Iteration 1: Weight Decay priors.

Usage: python eval_iter1_wd.py
"""
import sys, os, torch, numpy as np, math
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.params import get_flat_params, set_flat_params
from utils.prior_posterior import GaussianPrior, GaussianPosterior
from utils.pacbayes_utils import kl_diag_gaussians, mcallester_bound
from create_data.dataset import TransformedDataset
from models.models_mnist import BaselineCNN, EquivariantCNN

DATA_DIR = "create_data/rot_mnist"
DEVICE = "cpu"
S = 200
DELTA = 0.05
PRIOR_SIGMA = 0.05
POST_SIGMA = 0.05


def estimate_gibbs_risk(model, mu_flat, sigma_q, loader, device, S):
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
    model = model_cls().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    train_loader = DataLoader(TransformedDataset(f"{DATA_DIR}/train.pt"), batch_size=256)
    test_loader = DataLoader(TransformedDataset(f"{DATA_DIR}/test.pt"), batch_size=256)

    prior_data = torch.load(prior_path, map_location=DEVICE)
    prior = GaussianPrior(prior_data["mu"], PRIOR_SIGMA)
    posterior = GaussianPosterior(model, get_flat_params(model), sigma=POST_SIGMA)

    kl = kl_diag_gaussians(posterior.mu, posterior.sigma, prior.mu, prior.sigma)
    train_risk, _ = estimate_gibbs_risk(model, posterior.mu, posterior.sigma,
                                        train_loader, DEVICE, S)
    test_risk, test_stderr = estimate_gibbs_risk(model, posterior.mu, posterior.sigma,
                                                  test_loader, DEVICE, S)

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
    print("Iteration 1: Weight Decay in Prior — Evaluation")
    print(f"Settings: S={S}, delta={DELTA}, sigma={PRIOR_SIGMA}, wd=1e-4 in prior")
    print("=" * 60)

    res_base = evaluate_model(
        BaselineCNN,
        f"{DATA_DIR}/baseline_wd.pt",
        f"{DATA_DIR}/prior_mu_baseline_wd.pt",
        "Baseline CNN (wd=1e-4)"
    )

    res_eq = evaluate_model(
        EquivariantCNN,
        f"{DATA_DIR}/equivariant_wd.pt",
        f"{DATA_DIR}/prior_mu_equivariant_wd.pt",
        "Equivariant CNN C8 (wd=1e-4)"
    )

    print(f"\n{'='*60}")
    print("Comparison with Baseline:")
    print(f"  Baseline KL:     7368.9  ->  WD KL:     {res_eq['kl']:.1f}")
    print(f"  Baseline Bound:  0.5154 ->  WD Bound:   {res_eq['bound']:.4f}")
    print(f"  Baseline Risk:   0.2148 ->  WD Risk:    {res_eq['test_risk']:.4f}")
    print(f"{'='*60}")
