from training.train_model import train_model
from training.train_prior import train_prior

from create_data.dataset import TransformedDataset, ModelNet10PointCloud

from models.models_mnist import BaselineCNN, EquivariantCNN
try:
    from models.models_cifar import BaselineCNN as CIFARBaselineCNN
    from models.models_cifar import EquivariantCNN as CIFAREquivariantCNN
    from models.models_cifar100 import BaselineCNN as CIFAR100BaselineCNN
    from models.models_cifar100 import EquivariantCNN as CIFAR100EquivariantCNN
except ImportError:
    pass
try:
    from models.models_modelnet import BaselinePointNet as ModelNetBaseline
    from models.models_modelnet import EquivariantModelNet as ModelNetRotEquivariant
    from models.models_modelnet import ScaleInvariantModel as ModelNetScaleEquivariant
except ImportError:
    pass
try:
    from models.models_top_tagging import LorentzBaseline as TopTaggingBaseline
    from models.models_top_tagging import LorentzInvariantModel as TopTaggingEquivariant
except ImportError:
    pass


def run_train(base_model, equiv_model, dir, epochs_posterior=50, epochs_prior=1):

    data_dir = "create_data/" + dir

    # baseline
    train_prior(
        model_cls=base_model,
        data_dir=data_dir,
        save_path=data_dir+"/prior_mu_baseline.pt",
        epochs=epochs_prior
    )
    train_model(
        model_cls=base_model,
        data_dir=data_dir,
        prior_path=data_dir+"/prior_mu_baseline.pt",
        save_path=data_dir+"/baseline.pt",
        epochs=epochs_posterior,
        device="cpu"
    )

    # equivariant
    train_prior(
        model_cls=equiv_model,
        data_dir=data_dir,
        save_path=data_dir+"/prior_mu_equivariant.pt",
        epochs=epochs_prior
    )

    train_model(
        model_cls=equiv_model,
        data_dir=data_dir,
        prior_path=data_dir+"/prior_mu_equivariant.pt",
        save_path=data_dir+"/equivariant.pt",
        epochs=epochs_posterior,
        device="cpu"
    )

def run_all():
    print("\nTraining the models")

    # MNIST rotated
    print("Rotated MNIST:")
    run_train(BaselineCNN, EquivariantCNN, "rot_mnist", epochs_posterior=1, epochs_prior=5)

    # MNIST affine
    print("\nRotated and translated MNIST:")
    run_train(BaselineCNN, EquivariantCNN, "affine_mnist", epochs_posterior=1, epochs_prior=10)

    # CIFAR rotated
    print("\nRotated CIFAR10:")
    run_train(CIFARBaselineCNN, CIFAREquivariantCNN, "rot_cifar", epochs_posterior=1, epochs_prior=10)

    # CIFAR100 affine
    print("\nRotated and translated CIFAR100:")
    run_train(CIFAR100BaselineCNN, CIFAR100EquivariantCNN, "affine_cifar100", epochs_posterior=2, epochs_prior=10)

    # ModelNet rotated
    print("\nRotated ModelNet:")
    run_train(ModelNetBaseline, ModelNetRotEquivariant, "modelnet10_so3", epochs_posterior=1, epochs_prior=4)

    # ModelNet scaled
    print("\nScaled ModelNet:")
    run_train(ModelNetBaseline, ModelNetScaleEquivariant, "modelnet10_scaling", epochs_posterior=1, epochs_prior=15)

    # TopTagging Lorentz Group
    print("\nTopTagging Lorentz Group:")
    run_train(TopTaggingBaseline, TopTaggingEquivariant, "top_tagging_lorentz", epochs_posterior=1, epochs_prior=16)

if __name__ == "__main__":

    run_all()