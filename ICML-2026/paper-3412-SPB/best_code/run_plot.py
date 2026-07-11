import torch
from torch.utils.data import DataLoader
import numpy as np

from create_data.dataset import TransformedDataset
from utils.params import get_flat_params
from utils.evaluation import (evaluate_pacbayes, plot_posteriors)
from utils.prior_posterior import (GaussianPrior, GaussianPosterior)
from utils.pacbayes_utils import sweep_sigma_q_and_compute_curve
from training.evaluate import evaluate
from models.models_mnist import BaselineCNN as MNIST_Baseline, EquivariantCNN as MNIST_Equi
try:
    from models.models_cifar import BaselineCNN as CIFAR_Baseline, EquivariantCNN as CIFAR_Equi
    from models.models_cifar100 import BaselineCNN as CIFAR100_Baseline, EquivariantCNN as CIFAR100_Equi
except ImportError:
    pass
try:
    from models.models_modelnet import BaselinePointNet as ModelNet_Baseline, EquivariantModelNet as ModelNet_Rot_Equiv, ScaleInvariantModel as ModelNet_Scale_Equiv
except ImportError:
    pass
try:
    from models.models_top_tagging import LorentzBaseline as TopTagging_Baseline, LorentzInvariantModel as TopTagging_Equiv
except ImportError:
    pass

def run_experiment(model_base_class, model_eq_class, data_dir, device="cpu", S=200):

    model_base = model_base_class().to(device)
    model_base.load_state_dict(torch.load(data_dir + "/baseline.pt", map_location=device))
    model_eq = model_eq_class().to(device)
    model_eq.load_state_dict(torch.load(data_dir + "/equivariant.pt", map_location=device))

    train_loader = DataLoader(
        TransformedDataset(f"{data_dir}/train.pt"),
        batch_size=256
    )

    test_loader = DataLoader(
        TransformedDataset(f"{data_dir}/test.pt"),
        batch_size=256
    )

    prior_base_data = torch.load(f"{data_dir}/prior_mu_baseline.pt")
    prior_eq_data   = torch.load(f"{data_dir}/prior_mu_equivariant.pt")

    prior_base = GaussianPrior(prior_base_data["mu"], 5e-2)
    prior_eq   = GaussianPrior(prior_eq_data["mu"], 5e-2)
    #prior_eq_data["sigma"], prior_base_data["sigma"]

    post_base = GaussianPosterior(model_base, get_flat_params(model_base), sigma=5e-2)
    post_eq = GaussianPosterior(model_eq, get_flat_params(model_eq), sigma=5e-2)

    #sweep_sigma_q_and_compute_curve(model_eq, post_eq.mu, prior_eq.mu, prior_eq.sigma, train_loader, S=10, sigmas=np.logspace(-2, 2, 12))

    res_base = evaluate_pacbayes(post_base, prior_base, train_loader, test_loader, S=S)
    res_eq   = evaluate_pacbayes(post_eq, prior_eq, train_loader, test_loader, S=S)

    plot_posteriors(
        res_base,
        res_eq,
        labels=("Baseline", "Equivariant"),
        save_path=f"{data_dir}/histogram.png"
    )
    print("Results for Baseline:")
    print(f"KL: {res_base['kl']:.4f}",
          f"McAllester Bound: {res_base['bound']:.4f}",
          f"Complexity Term: {res_base['complexity']:.4f}",
          f"Test Risk: {res_base['test_mean']:.4f}",
          f"Test Risk std.: {res_base['test_stderr']:.4f}")
    print("Results for Equivariant:")
    print(f"KL: {res_eq['kl']:.4f}",
          f"McAllester Bound: {res_eq['bound']:.4f}",
          f"Complexity Term: {res_eq['complexity']:.4f}",
          f"Test Risk: {res_eq['test_mean']:.4f}",
          f"Test Risk std.: {res_eq['test_stderr']:.4f}")

    return res_base, res_eq

def run_all():
    print("\nplotting and evaluating the results")

    # rotated MNIST
    print("Rotated MNIST")
    run_experiment(MNIST_Baseline, MNIST_Equi, "create_data/rot_mnist", S=20)

    # affine MNIST
    print("\nAffine MNIST")
    run_experiment(MNIST_Baseline, MNIST_Equi, "create_data/affine_mnist", S=20)

    # rotated CIFAR
    print("\nRotated CIFAR")
    run_experiment(CIFAR_Baseline, CIFAR_Equi, "create_data/rot_cifar", S=20)

    # affine CIFAR100
    print("\nAffine CIFAR100")
    run_experiment(CIFAR100_Baseline, CIFAR100_Equi, "create_data/affine_cifar100", S=20)

    # rotated ModelNet
    print("\nRotated ModelNet")
    run_experiment(ModelNet_Baseline, ModelNet_Rot_Equiv, "create_data/modelnet10_so3", S=20)

    # scaled ModelNet
    print("\nScaled ModelNet")
    run_experiment(ModelNet_Baseline, ModelNet_Scale_Equiv, "create_data/modelnet10_scaling", S=20)

    print("\nTopTagging Lorentz Group")
    run_experiment(TopTagging_Baseline, TopTagging_Equiv, "create_data/top_tagging_lorentz", S=20)


if __name__ == "__main__":
    run_all()