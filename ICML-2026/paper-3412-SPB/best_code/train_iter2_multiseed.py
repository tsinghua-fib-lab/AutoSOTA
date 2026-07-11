"""Iteration 2: Multi-Seed Prior Averaging (IDEA-07).

Trains equivariant prior K=5 times with different seeds, averages the
mu vectors, then trains posterior from the averaged prior.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_model import train_model
from training.train_prior import train_prior
from models.models_mnist import EquivariantCNN

DATA_DIR = "create_data/rot_mnist"
K = 5  # number of seeds


def main():
    print("=" * 60)
    print("Iteration 2: Multi-Seed Prior Averaging (K=5)")
    print("=" * 60)

    # Train prior K times with different seeds
    mu_list = []
    for seed in range(K):
        print(f"\n[{seed+1}/{K}] Training Equivariant CNN prior (seed={seed})...")
        train_prior(
            model_cls=EquivariantCNN,
            data_dir=DATA_DIR,
            save_path=f"{DATA_DIR}/prior_mu_eq_seed{seed}.pt",
            epochs=5,
            seed=seed,
        )
        data = torch.load(f"{DATA_DIR}/prior_mu_eq_seed{seed}.pt", map_location="cpu")
        mu_list.append(data["mu"])
        print(f"  mu norm: {data['mu'].norm():.4f}")

    # Average prior mu vectors
    avg_mu = torch.stack(mu_list).mean(dim=0)
    print(f"\nAveraged prior mu norm: {avg_mu.norm():.4f}")

    # Also compute pairwise distances for diagnostics
    print("Pairwise ||mu_i - mu_j||^2:")
    for i in range(K):
        for j in range(i+1, K):
            dist2 = float(((mu_list[i] - mu_list[j])**2).sum())
            print(f"  seed {i} vs seed {j}: {dist2:.4f}")

    # Save averaged prior
    torch.save({"mu": avg_mu, "sigma": 5e-2},
               f"{DATA_DIR}/prior_mu_equivariant_avg5.pt")
    print(f"\nAveraged prior saved to {DATA_DIR}/prior_mu_equivariant_avg5.pt")

    # Train posterior from averaged prior
    print("\nTraining Equivariant CNN posterior from averaged prior...")
    train_model(
        model_cls=EquivariantCNN,
        data_dir=DATA_DIR,
        prior_path=f"{DATA_DIR}/prior_mu_equivariant_avg5.pt",
        save_path=f"{DATA_DIR}/equivariant_avg5.pt",
        epochs=1,
        device="cpu",
    )

    print("\nIteration 2 training complete!")


if __name__ == "__main__":
    main()
