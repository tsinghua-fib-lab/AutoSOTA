import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import patch_cifar10  # noqa: E402 - monkey-patch torchvision CIFAR-10 MD5 check

import random
from dataclasses import dataclass
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import timm

from mocoea.attacks import PGDAttack
from mocoea.bezier import BezierAdversarialUnconstrained
from mocoea.evolutionary_attack import EvolutionaryAttack
from mocoea.normalization import get_cifar10_classes, normalize_cifar10, normalize_imagenet


@dataclass
class DatasetConfig:
    name: str
    data_root: str
    checkpoint: str
    output_dir: str
    device: torch.device
    normalize: object
    fixed_classes: dict
    class_names: object
    epsilons: dict
    max_per_class: int
    seed: int
    valset: object = None
    cat_id: int = None
    dog_id: int = None


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_batch(x):
    return x.unsqueeze(0) if x.dim() == 3 else x


def build_dataset_config(args):
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset == "cifar10":
        return DatasetConfig(
            name="cifar10",
            data_root=args.data_root or "./data",
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device=device,
            normalize=normalize_cifar10,
            fixed_classes={
                "setting_A": 3,
                "setting_B": 3,
                "setting_C": (3, 5),
            },
            class_names=get_cifar10_classes(),
            epsilons={
                "linf": 8 / 255,
                "l2": 0.5,
                "l1": 10.0,
            },
            max_per_class=300,
            seed=args.seed,
        )

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    data_root = args.data_root or "./dataset/imagenet/val"
    valset = ImageFolder(root=data_root, transform=transform_val)
    cat_id = valset.class_to_idx["n02124075"]
    dog_id = valset.class_to_idx["n02099712"]

    return DatasetConfig(
        name="imagenet",
        data_root=data_root,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=device,
        normalize=normalize_imagenet,
        fixed_classes={
            "setting_A": cat_id,
            "setting_B": cat_id,
            "setting_C": (cat_id, dog_id),
        },
        class_names={
            cat_id: "egyptian cat",
            dog_id: "labrador retriever",
        },
        epsilons={
            "linf": 4 / 255,
            "l2": 2.0,
            "l1": 75.0,
        },
        max_per_class=50,
        seed=args.seed,
        valset=valset,
        cat_id=cat_id,
        dog_id=dog_id,
    )


def load_model(config):
    if config.name == "cifar10":
        model = resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        model.fc = torch.nn.Linear(512, 10)

        checkpoint = torch.load(config.checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model with accuracy: {checkpoint['acc']:.2f}%")
        return model.to(config.device).eval()

    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.eval()
    return model.to(config.device)


def make_eval_loader(config, batch_size=None, shuffle=False):
    if config.name == "cifar10":
        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=False, download=False, transform=transform_test
        )
        return torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size or 1,
            shuffle=shuffle,
            num_workers=0,  # NFS workaround
        )

    needed_classes = [config.cat_id, config.dog_id]
    indices = [i for i, y in enumerate(config.valset.targets) if y in needed_classes]
    val_subset = torch.utils.data.Subset(config.valset, indices)
    return torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size or 64,
        shuffle=shuffle,
        num_workers=0,  # NFS workaround
        pin_memory=True,
    )


def import_attack_modules(config):
    return PGDAttack, BezierAdversarialUnconstrained


def import_evolutionary_attack(config):
    return EvolutionaryAttack
