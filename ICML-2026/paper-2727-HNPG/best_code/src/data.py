import os
os.environ["GEOMSTATS_BACKEND"] = "numpy"

import numpy as np

import torchvision
import torchvision.transforms as transforms





def load_synthetic(num_points, dim, R, rng):
    u = rng.normal(size=(num_points, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rad = R * (rng.random(num_points) ** (1.0 / dim))
    return (u * rad[:, None]).astype(np.float64)

def load_images(
    dataset: str,
    num_images: int,
    rng: np.random.Generator,
    dim=None,
    R=3.0,
):
    if rng is None:
        rng = np.random.default_rng()

    dataset_lower = dataset.lower()

    if dataset_lower == "synthetic":
        if dim is None:
            raise ValueError("dim required for synthetic dataset")
        return load_synthetic(num_images, dim, R, rng)

    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_lower == "mnist":
        data = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_lower == "cifar10":
        data = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist', 'cifar10', or 'synthetic'")

    if num_images > len(data):
        raise ValueError(
            f"Requested {num_images} images but {dataset} train set has only {len(data)}."
        )

    indices = rng.choice(len(data), size=num_images, replace=False)
    images = []
    for idx in indices:
        img, _ = data[int(idx)]
        images.append(img.numpy().flatten())

    return np.array(images, dtype=np.float64)

def load_images_in_class(
    dataset: str,
    class_id: int,
    num_images: int,
    rng: np.random.Generator,
):
    if rng is None:
        rng = np.random.default_rng()

    dataset_lower = dataset.lower()
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_lower == "mnist":
        data = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_lower == "cifar10":
        data = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist' or 'cifar10'")

    targets = np.asarray(data.targets)
    class_indices = np.flatnonzero(targets == class_id).astype(int)
    if len(class_indices) == 0:
        raise ValueError(f"No training images for class_id={class_id} in {dataset}")
    if num_images > len(class_indices):
        raise ValueError(
            f"Need {num_images} images but class {class_id} has only {len(class_indices)}."
        )

    indices = rng.choice(class_indices, size=num_images, replace=False)
    images = []
    for idx in indices:
        img, _ = data[int(idx)]
        images.append(img.numpy().flatten())

    return np.array(images, dtype=np.float64)