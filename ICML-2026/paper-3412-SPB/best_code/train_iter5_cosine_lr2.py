"""Iteration 5: Cosine Annealing with higher LR (2e-3)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_model import train_model
from models.models_mnist import EquivariantCNN

DATA_DIR = "create_data/rot_mnist"


def main():
    print("=" * 60)
    print("Iteration 5: Cosine Annealing LR=2e-3")
    print("=" * 60)

    print("\nTraining Equivariant CNN posterior (cosine, lr=2e-3)...")
    train_model(
        model_cls=EquivariantCNN,
        data_dir=DATA_DIR,
        prior_path=f"{DATA_DIR}/prior_mu_equivariant.pt",
        save_path=f"{DATA_DIR}/equivariant_cosine_lr2.pt",
        epochs=1,
        device="cpu",
        use_cosine=True,
        lr=2e-3,
    )

    print("\nIteration 5 training complete!")


if __name__ == "__main__":
    main()
