"""Iteration 3: Label Smoothing in Posterior Training (IDEA-09).

Uses label_smoothing=0.1 in CrossEntropyLoss during posterior training.
Keeps baseline prior (single seed, no weight decay).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_model import train_model
from models.models_mnist import EquivariantCNN

DATA_DIR = "create_data/rot_mnist"


def main():
    print("=" * 60)
    print("Iteration 3: Label Smoothing (alpha=0.1)")
    print("=" * 60)

    # Use BASELINE prior (best so far)
    print("\nTraining Equivariant CNN posterior with label_smoothing=0.1...")
    train_model(
        model_cls=EquivariantCNN,
        data_dir=DATA_DIR,
        prior_path=f"{DATA_DIR}/prior_mu_equivariant.pt",
        save_path=f"{DATA_DIR}/equivariant_ls01.pt",
        epochs=1,
        device="cpu",
        label_smoothing=0.1,
    )

    print("\nIteration 3 training complete!")


if __name__ == "__main__":
    main()
