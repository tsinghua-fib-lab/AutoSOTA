import numpy as np
import matplotlib.pyplot as plt
from utils.pacbayes_utils import kl_diag_gaussians, mcallester_bound, estimate_gibbs_loss


def evaluate_pacbayes(posterior, prior, train_loader, test_loader, device="cpu", S=200, delta=0.05):

    model = posterior.model

    kl = kl_diag_gaussians(posterior.mu, posterior.sigma, prior.mu, prior.sigma)

    #print("Estimate expected empirical risk")
    train_mean, _, _ = estimate_gibbs_loss(model, posterior.mu, posterior.sigma, train_loader, device=device, S=S, fast=True)

    n = len(train_loader.dataset)

    bound, complexity = mcallester_bound(train_mean, kl, n, delta)

    #print("Estimate expected true risk")
    test_mean, test_stderr, samples = estimate_gibbs_loss(model, posterior.mu, posterior.sigma, test_loader, device=device, S=S, fast=True)

    return {
        "kl": kl,
        "bound": bound,
        "complexity": complexity,
        "test_mean": test_mean,
        "test_stderr": test_stderr,
        "samples": samples,
    }


def plot_posteriors(result1, result2, labels, bins=60, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    c1 = 'tab:blue'
    c2 = 'tab:orange'

    bins = np.linspace(0, 1, bins + 1)

    ax.hist(result1["samples"], bins=bins, alpha=0.5, label=labels[0], color=c1)
    ax.hist(result2["samples"], bins=bins, alpha=0.5, label=labels[1], color=c2)

    ax.axvline(result1["test_mean"], linestyle='--', label="baseline", color=c1)
    ax.axvline(result1["bound"], linestyle='-.', color=c1)

    ax.axvline(result2["test_mean"], linestyle='--', label="equivariant", color=c2)
    ax.axvline(result2["bound"], linestyle='-.', color=c2)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Risk")
    ax.set_ylabel("Count")
    ax.legend()

    if save_path:
        plt.savefig(save_path)

    return fig