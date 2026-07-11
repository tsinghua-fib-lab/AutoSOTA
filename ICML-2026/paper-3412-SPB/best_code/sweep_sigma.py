"""Sigma sweep for PAC-Bayesian bound optimization.

Sweeps sigma (sigma_P = sigma_Q) to find the value that minimizes
the McAllester bound, using the existing trained checkpoints.
"""
import sys, os, torch, numpy as np, math
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.params import get_flat_params, set_flat_params
from utils.pacbayes_utils import kl_diag_gaussians, mcallester_bound
from create_data.dataset import TransformedDataset
from models.models_mnist import EquivariantCNN

DATA_DIR = "create_data/rot_mnist"
DEVICE = "cpu"
DELTA = 0.05


def estimate_risk(model, mu_flat, sigma_q, loader, device, S):
    """Estimate Gibbs 0-1 risk by Monte Carlo sampling from posterior."""
    torch.manual_seed(0)
    model.eval()
    mu_flat = mu_flat.to(device)
    losses = []
    for s in range(S):
        eps = torch.randn_like(mu_flat)
        w_sample = mu_flat + sigma_q * eps
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


def main():
    print("Loading Equivariant CNN checkpoint...", flush=True)
    model = EquivariantCNN().to(DEVICE)
    model.load_state_dict(torch.load(f"{DATA_DIR}/equivariant.pt", map_location=DEVICE))
    posterior_mu = get_flat_params(model)

    prior_data = torch.load(f"{DATA_DIR}/prior_mu_equivariant.pt", map_location=DEVICE)
    prior_mu = prior_data["mu"]

    train_loader = DataLoader(TransformedDataset(f"{DATA_DIR}/train.pt"), batch_size=256)
    test_loader = DataLoader(TransformedDataset(f"{DATA_DIR}/test.pt"), batch_size=256)
    n_train = len(train_loader.dataset)

    sigmas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    S = 50

    results = []
    best_bound = float("inf")
    best_sigma = None

    print(f"\nSigma Sweep: S={S}, {len(sigmas)} sigma values")
    print(f"{'sigma':>8s}  {'KL':>10s}  {'train_risk':>10s}  {'complexity':>10s}  {'bound':>10s}  {'test_risk':>10s}")
    print("-" * 70)

    for sigma in sigmas:
        kl = kl_diag_gaussians(posterior_mu, sigma, prior_mu, sigma)
        train_risk, train_se = estimate_risk(
            model, posterior_mu, sigma, train_loader, DEVICE, S
        )
        bound, complexity = mcallester_bound(train_risk, kl, n_train, DELTA)
        test_risk, test_se = estimate_risk(
            model, posterior_mu, sigma, test_loader, DEVICE, S
        )

        results.append({
            "sigma": sigma, "kl": kl, "train_risk": train_risk,
            "complexity": complexity, "bound": bound, "test_risk": test_risk,
        })

        marker = ""
        if bound < best_bound:
            best_bound = bound
            best_sigma = sigma
            marker = " <-- BEST"

        print(f"{sigma:8.4f}  {kl:10.1f}  {train_risk:10.4f}  {complexity:10.4f}  {bound:10.4f}  {test_risk:10.4f}{marker}")

    print(f"\nBest sigma: {best_sigma} -> bound={best_bound:.4f}")

    sorted_results = sorted(results, key=lambda r: r["bound"])
    print("Top 3 sigmas for fine sweep:")
    for r in sorted_results[:3]:
        print(f"  sigma={r['sigma']:.4f} bound={r['bound']:.4f} kl={r['kl']:.1f} train_risk={r['train_risk']:.4f} test_risk={r['test_risk']:.4f}")

    torch.save(results, f"{DATA_DIR}/sigma_sweep_results.pt")
    print(f"\nResults saved to {DATA_DIR}/sigma_sweep_results.pt")


if __name__ == "__main__":
    main()
