"""Iteration 4: Cosine Annealing LR Schedule (IDEA-10)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_model import train_model
from models.models_mnist import EquivariantCNN

DATA_DIR = "create_data/rot_mnist"


def main():
    print("=" * 60)
    print("Iteration 4: Cosine Annealing LR Schedule")
    print("=" * 60)

    print("\nTraining Equivariant CNN posterior with cosine annealing...")
    train_model(
        model_cls=EquivariantCNN,
        data_dir=DATA_DIR,
        prior_path=f"{DATA_DIR}/prior_mu_equivariant.pt",
        save_path=f"{DATA_DIR}/equivariant_cosine.pt",
        epochs=1,
        device="cpu",
        use_cosine=True,
    )

    print("\nIteration 4 training complete!")


if __name__ == "__main__":
    main()
