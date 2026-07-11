from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import functional as TF

from create_data.generators import generate_transformed_dataset
from create_data.transforms import RandomRotation, RandomAffine
try:
    from create_data.transforms import RandomTurntableSO3, RandomScaling, RandomLorentz
except ImportError:
    pass
from create_data.visualize import show_grid
try:
    from create_data.dataset import ModelNet10PointCloud
except ImportError:
    pass
try:
    from create_data.dataset import TopTaggingDataset
except ImportError:
    pass


def run_mnist_rotation():
    generate_transformed_dataset(
        base_dataset=(
            MNIST(root="./data", train=True, download=True),
            MNIST(root="./data", train=False, download=True),
        ),
        transform_fn=RandomRotation(),
        out_dir="create_data/rot_mnist",
        train_split=(40000, 10000, 10000),
        seed=0,
        to_tensor_fn=TF.to_tensor
    )

    show_grid("create_data/rot_mnist/train.pt", "create_data/rot_mnist")


def run_cifar_rotation():
    generate_transformed_dataset(
        base_dataset=(
            CIFAR10(root="./data", train=True, download=True),
            CIFAR10(root="./data", train=False, download=True),
        ),
        transform_fn=RandomRotation(fill=(0, 0, 0)),
        out_dir="create_data/rot_cifar",
        train_split=(30000, 10000, 10000),
        seed=0,
        to_tensor_fn=TF.to_tensor
    )

    show_grid("create_data/rot_cifar/train.pt", "create_data/rot_cifar")


def run_mnist_affine():
    generate_transformed_dataset(
        base_dataset=(
            MNIST(root="./data", train=True, download=True),
            MNIST(root="./data", train=False, download=True),
        ),
        transform_fn=RandomAffine(),
        out_dir="create_data/affine_mnist",
        train_split=(40000, 10000, 10000),
        seed=0,
        to_tensor_fn=TF.to_tensor
    )

    show_grid("create_data/affine_mnist/train.pt", "create_data/affine_mnist", title_fn=lambda y, m: f"{y}, {m['angle']:.1f}°")

def run_cifar100_affine():
    generate_transformed_dataset(
        base_dataset=(
            CIFAR100(root="./data", train=True, download=True),
            CIFAR100(root="./data", train=False, download=True),
        ),
        transform_fn=RandomAffine(),
        out_dir="create_data/affine_cifar100",
        train_split=(30000, 10000, 10000),
        seed=0,
        to_tensor_fn=TF.to_tensor
    )

    show_grid("create_data/affine_cifar100/train.pt", "create_data/affine_cifar100",
              title_fn=lambda y, m: f"{y}, {m['angle']:.1f}°")


def run_modelnet_so3():
    generate_transformed_dataset(
        base_dataset=(
            ModelNet10PointCloud(root="./data/ModelNet10", split="train"),
            ModelNet10PointCloud(root="./data/ModelNet10", split="test"),
        ),
        transform_fn=RandomTurntableSO3(),
        out_dir="create_data/modelnet10_so3",
        train_split=(2395, 798, 798),
        seed=0,
        to_tensor_fn=lambda x: x,
    )


def run_modelnet_scaling():
    generate_transformed_dataset(
        base_dataset=(
            ModelNet10PointCloud(root="./data/ModelNet10", split="train"),
            ModelNet10PointCloud(root="./data/ModelNet10", split="test"),
        ),
        transform_fn=RandomScaling(),
        out_dir="create_data/modelnet10_scaling",
        train_split=(2395, 798, 798),
        seed=0,
        to_tensor_fn=lambda x: x,
    )

def run_top_tagging_lorentz():
    generate_transformed_dataset(
        base_dataset=(
            TopTaggingDataset(split="train", num_particles=30),
            TopTaggingDataset(split="test", num_particles=30),
        ),
        transform_fn=RandomLorentz(),
        out_dir="create_data/top_tagging_lorentz",
        train_split=(60000, 15000, 15000),
        copies_per_sample=4,
        seed=0,
        to_tensor_fn=lambda x: x,
    )

def run_all():
    print("\nGenerate the transformed data")

    print("Rotated MNIST:")
    run_mnist_rotation()

    print("\nRotated and translated MNIST")
    run_mnist_affine()

    print("\nRotated CIFAR10")
    run_cifar_rotation()

    print("\nRotated and translated CIFAR100")
    run_cifar100_affine()

    print("\nModelNet10 with SO(3) action")
    run_modelnet_so3()

    print("\nModelNet10 with scaling action")
    run_modelnet_scaling()

    print("\nTopTagging with Lorentz group")
    run_top_tagging_lorentz()

if __name__ == "__main__":

    run_all()