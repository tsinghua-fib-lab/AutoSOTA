"""Iteration 1: Weight Decay in Prior Training (IDEA-08).

Trains priors with weight_decay=1e-4 (vs baseline 0.0), then trains
posteriors from the regularized priors. Saves to *_wd.pt files.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_model import train_model
from training.train_prior import train_prior
from models.models_mnist import BaselineCNN, EquivariantCNN

DATA_DIR = "create_data/rot_mnist"
WD = 1e-4


def main():
    print("=" * 60)
    print("Iteration 1: Weight Decay in Prior Training (wd=1e-4)")
    print("=" * 60)

    # Train Baseline CNN prior with weight decay
    print("\n[1/4] Training Baseline CNN prior (wd=1e-4)...")
    train_prior(
        model_cls=BaselineCNN,
        data_dir=DATA_DIR,
        save_path=f"{DATA_DIR}/prior_mu_baseline_wd.pt",
        epochs=5,
        weight_decay=WD,
    )

    # Train Baseline posterior from regularized prior
    print("\n[2/4] Training Baseline CNN posterior...")
    train_model(
        model_cls=BaselineCNN,
        data_dir=DATA_DIR,
        prior_path=f"{DATA_DIR}/prior_mu_baseline_wd.pt",
        save_path=f"{DATA_DIR}/baseline_wd.pt",
        epochs=1,
        device="cpu",
    )

    # Train Equivariant CNN prior with weight decay
    print("\n[3/4] Training Equivariant CNN prior (wd=1e-4)...")
    train_prior(
        model_cls=EquivariantCNN,
        data_dir=DATA_DIR,
        save_path=f"{DATA_DIR}/prior_mu_equivariant_wd.pt",
        epochs=5,
        weight_decay=WD,
    )

    # Train Equivariant posterior from regularized prior
    print("\n[4/4] Training Equivariant CNN posterior...")
    train_model(
        model_cls=EquivariantCNN,
        data_dir=DATA_DIR,
        prior_path=f"{DATA_DIR}/prior_mu_equivariant_wd.pt",
        save_path=f"{DATA_DIR}/equivariant_wd.pt",
        epochs=1,
        device="cpu",
    )

    print("\nIteration 1 training complete!")


if __name__ == "__main__":
    main()
