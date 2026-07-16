import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
import numpy as np
import medmnist
import timm
from rich.console import Console
from rich.table import Table

from dp_kfac.trainer import set_seed
from dp_kfac.results import save_results_csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

console = Console()

SCENARIOS = {
    "1_DistMatch": {
        "name": "Ideal Transfer",
        "private": "fashionmnist",
        "public": "mnist",
        "img_size": 28,
        "classes": 10,
        "description": "MNIST -> FashionMNIST: High A alignment, High G alignment",
        "prediction": "Public should win (Public ~ Oracle)",
    },
    "2_DistDisjoint": {
        "name": "Texture Mismatch",
        "private": "pathmnist",
        "public": "mnist",
        "img_size": 28,
        "classes": 9,
        "description": "MNIST -> PathMNIST: Orthogonal A (Object vs Texture), Medium G",
        "prediction": "Noise should win (Public A actively hurts)",
    },
    "3_TaskHarder": {
        "name": "Task Shift",
        "private": "cifar100",
        "public": "cifar10",
        "img_size": 32,
        "classes": 100,
        "description": "CIFAR-10 -> CIFAR-100: High A alignment, Low G alignment",
        "prediction": "Hybrid should win (A_pub good, G_pub biased)",
    },
    "4_TotalMismatch": {
        "name": "Total Mismatch",
        "private": "cifar100",
        "public": "mnist",
        "img_size": 32,
        "classes": 100,
        "description": "MNIST -> CIFAR-100: Partial A (spatial prior), Very Low G",
        "prediction": "Hybrid should win (A_pub spatial help, G_pub hurts)",
    },
    "5_MedGlobal": {
        "name": "MedMNIST-Global (Checkmate)",
        "private": "medmnist_global",
        "public": "mnist",
        "img_size": 28,
        "classes": 47,
        "description": "MNIST -> MedMNIST-Global: HARD test - Mixed Texture+Structure, No Center Bias",
        "prediction": "Pink Noise should win (translation invariant + spatial correlation)",
    },
}

ALL_PRECOND_TYPES = {
    "Oracle (Private)": "oracle",
    "Identity (No Precond)": "identity",
    "A_pub G_pub_pub": "A_pub__G_pub_pub",
    "A_pub G_pub_noise": "A_pub__G_pub_noise",
    "A_pub G_pink_noise": "A_pub__G_pink_noise",
    "A_pink G_pub_pub": "A_pink__G_pub_pub",
    "A_pink G_pub_noise": "A_pink__G_pub_noise",
    "A_pink G_pink_noise": "A_pink__G_pink_noise",
    "A_syn G_syn (Clustered Pink)": "A_syn__G_syn_clustered",
}

SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-2
EPSILON = 2.0
DELTA = 1e-5
PRECOND_STEPS = 10
CLIP_NORM = 1.0
NUM_WORKERS = 16

PRECOND_UPDATE_EVERY = 1

MODEL_TYPE = "convnext"
CROSSVIT_IMG_SIZE = 240
CONVNEXT_IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STANDARD_MEAN = [0.5, 0.5, 0.5]
STANDARD_STD = [0.5, 0.5, 0.5]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, img_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        fc_size = img_size // 4
        self.fc1 = nn.Linear(32 * fc_size * fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CrossViTClassifier(nn.Module):
    def __init__(self, num_classes=100, img_size=240, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            "crossvit_tiny_240", pretrained=True, num_classes=0
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=100, img_size=224):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_model(num_classes, img_size, private_dataset):
    if MODEL_TYPE == "simple_cnn":
        model_choice = "simple_cnn"
    elif MODEL_TYPE == "crossvit":
        model_choice = "crossvit"
    elif MODEL_TYPE == "convnext":
        model_choice = "convnext"
    elif MODEL_TYPE == "auto":
        if private_dataset in ["cifar100", "medmnist_global"]:
            model_choice = "convnext"
        else:
            model_choice = "simple_cnn"
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    if model_choice == "crossvit":
        model = CrossViTClassifier(num_classes=num_classes, img_size=CROSSVIT_IMG_SIZE)
        actual_img_size = CROSSVIT_IMG_SIZE
        model_type_str = "crossvit"
    elif model_choice == "convnext":
        model = ConvNeXtClassifier(num_classes=num_classes, img_size=CONVNEXT_IMG_SIZE)
        actual_img_size = CONVNEXT_IMG_SIZE
        model_type_str = "convnext"
    else:
        model = SimpleCNN(num_classes=num_classes, img_size=img_size)
        actual_img_size = img_size
        model_type_str = "simple_cnn"

    return model, actual_img_size, model_type_str


def get_mnist_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=NUM_WORKERS, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_fashionmnist_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=NUM_WORKERS, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_pathmnist_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    num_workers = 4 if use_imagenet_norm else 2
    info = medmnist.INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = DataClass(split="train", transform=transform, download=True, root="./data")
    test_ds = DataClass(split="test", transform=transform, download=True, root="./data")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_cifar10_loaders(batch_size, img_size=32, use_imagenet_norm=False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    num_workers = 4 if use_imagenet_norm else 2
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_cifar100_loaders(batch_size, img_size=32, use_imagenet_norm=False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    num_workers = 4 if use_imagenet_norm else 2
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


class MedMNISTSubset(torch.utils.data.Dataset):
    def __init__(self, medmnist_ds, label_offset, transform, max_samples=None):
        self.ds = medmnist_ds
        self.label_offset = label_offset
        self.transform = transform
        self.max_samples = min(max_samples, len(medmnist_ds)) if max_samples else len(medmnist_ds)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        if idx >= self.max_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.max_samples}")
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
        label = int(label.item() if hasattr(label, "item") else label) + self.label_offset
        return img, label


def get_medmnist_global_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    mean = IMAGENET_MEAN if use_imagenet_norm else STANDARD_MEAN
    std = IMAGENET_STD if use_imagenet_norm else STANDARD_STD
    num_workers = 4 if use_imagenet_norm else 2

    flags = [
        "pathmnist",
        "bloodmnist",
        "dermamnist",
        "octmnist",
        "tissuemnist",
        "organamnist",
    ]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_datasets = []
    test_datasets = []
    current_label_offset = 0

    console.print(f"[dim]Constructing MedMNIST-Global from: {flags}[/dim]")

    for flag in flags:
        info = medmnist.INFO[flag]
        n_classes = len(info["label"])
        DataClass = getattr(medmnist, info["python_class"])

        train_ds = DataClass(split="train", transform=None, download=True, root="./data")
        test_ds = DataClass(split="test", transform=None, download=True, root="./data")

        train_max = len(train_ds) // 2
        test_max = len(test_ds) // 2

        train_subset = MedMNISTSubset(train_ds, current_label_offset, transform, max_samples=train_max)
        test_subset = MedMNISTSubset(test_ds, current_label_offset, transform, max_samples=test_max)
        train_datasets.append(train_subset)
        test_datasets.append(test_subset)

        console.print(
            f"  [dim]+ {flag}: {n_classes} classes "
            f"(IDs {current_label_offset}-{current_label_offset + n_classes - 1}), "
            f"train={len(train_subset)}/{len(train_ds)}, "
            f"test={len(test_subset)}/{len(test_ds)}[/dim]"
        )

        current_label_offset += n_classes

    total_classes = current_label_offset
    console.print(f"[green]Total MedMNIST-Global: {total_classes} classes[/green]")

    full_train_ds = ConcatDataset(train_datasets)
    full_test_ds = ConcatDataset(test_datasets)

    console.print(f"[green]Total samples: train={len(full_train_ds)}, test={len(full_test_ds)}[/green]")

    train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(full_test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)

    return train_loader, test_loader, len(full_train_ds), total_classes


def get_dataset_loaders(dataset_name, batch_size, img_size, use_imagenet_norm=False):
    if dataset_name == "medmnist_global":
        return get_medmnist_global_loaders(batch_size, img_size, use_imagenet_norm)

    loaders = {
        "mnist": get_mnist_loaders,
        "fashionmnist": get_fashionmnist_loaders,
        "pathmnist": get_pathmnist_loaders,
        "cifar10": get_cifar10_loaders,
        "cifar100": get_cifar100_loaders,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return loaders[dataset_name](batch_size, img_size, use_imagenet_norm)


def get_noise_batch(batch_size, img_size, device):
    return torch.randn(batch_size, 3, img_size, img_size, device=device)


def get_pink_noise_batch(batch_size, img_size, device, alpha=1.0):
    white_noise_freq = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.cfloat, device=device)

    freqs = torch.fft.fftfreq(img_size, device=device)
    fx, fy = torch.meshgrid(freqs, freqs, indexing="ij")

    f = torch.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0

    scale = 1.0 / (f**alpha)
    scale[0, 0] = 0.0
    scale = scale.view(1, 1, img_size, img_size)

    pink_noise_freq = white_noise_freq * scale
    pink_noise = torch.fft.ifft2(pink_noise_freq).real

    std = pink_noise.view(batch_size, -1).std(dim=1, keepdim=True)
    pink_noise = pink_noise / (std.view(batch_size, 1, 1, 1) + 1e-8) * 0.5

    return pink_noise


class SyntheticPinkClusterTask:
    def __init__(self, num_classes, img_size, device, jitter_scale=0.5):
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device
        self.jitter_scale = jitter_scale

        console.print(f"  [dim]Generating {num_classes} Pink Noise Centroids...[/dim]")
        self.centroids = get_pink_noise_batch(num_classes, img_size, device, alpha=1.0)
        self.centroids = self.centroids / self.centroids.std()

    def get_batch(self, batch_size):
        labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        base_signals = self.centroids[labels]
        jitter = get_pink_noise_batch(batch_size, self.img_size, self.device, alpha=1.0)
        inputs = base_signals + (jitter * self.jitter_scale)
        inputs = (inputs - inputs.mean(dim=(1, 2, 3), keepdim=True)) / (inputs.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        return inputs, labels


class KFACRecorder:
    def __init__(self, model, only_trainable=False):
        self.activations = {}
        self.backprops = {}
        self.handles = []
        self.enabled = False

        target = model._module if hasattr(model, "_module") else model
        for name, module in target.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if only_trainable:
                    has_trainable = any(p.requires_grad for p in module.parameters())
                    if not has_trainable:
                        continue

                self.handles.append(
                    module.register_forward_hook(self._save_act(name))
                )
                self.handles.append(
                    module.register_full_backward_hook(self._save_grad(name))
                )

    def _save_act(self, name):
        def hook(mod, inp, out):
            if self.enabled:
                self.activations[name] = inp[0].detach()
        return hook

    def _save_grad(self, name):
        def hook(mod, grad_in, grad_out):
            if self.enabled:
                self.backprops[name] = grad_out[0].detach()
        return hook

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def clear(self):
        self.activations = {}
        self.backprops = {}


def compute_kfac_covs(model, recorder, data, labels, device, num_classes):
    recorder.enable()
    model.zero_grad()

    data = data.to(device)
    labels = labels.to(device)

    out = model(data)
    loss = F.cross_entropy(out, labels)
    loss.backward()

    cov_A, cov_G = {}, {}
    eps = 1e-4

    target = model._module if hasattr(model, "_module") else model
    name_to_mod = dict(target.named_modules())

    for name in recorder.backprops:
        if name not in name_to_mod:
            continue
        mod = name_to_mod[name]

        if isinstance(mod, nn.Linear):
            A = recorder.activations[name]
            if A.dim() > 2:
                A = A.reshape(-1, A.shape[-1])
            cov_A[name] = (A.T @ A) / A.size(0) + eps * torch.eye(A.size(1), device=device)

            G = recorder.backprops[name]
            if G.dim() > 2:
                G = G.reshape(-1, G.shape[-1])
            cov_G[name] = (G.T @ G) / G.size(0) + eps * torch.eye(G.size(1), device=device)

        elif isinstance(mod, nn.Conv2d):
            X = recorder.activations[name]
            X_unfold = F.unfold(X, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            X_unfold = X_unfold.transpose(1, 2).reshape(-1, X_unfold.size(1))
            cov_A[name] = (X_unfold.T @ X_unfold) / X_unfold.size(0) + eps * torch.eye(X_unfold.size(1), device=device)

            GY = recorder.backprops[name]
            GY = GY.permute(0, 2, 3, 1).reshape(-1, GY.size(1))
            cov_G[name] = (GY.T @ GY) / GY.size(0) + eps * torch.eye(GY.size(1), device=device)

    recorder.disable()
    recorder.clear()
    model.zero_grad()

    return cov_A, cov_G


def accumulate_covs(cov_list_A, cov_list_G):
    if not cov_list_A:
        return {}, {}

    avg_A, avg_G = {}, {}
    for name in cov_list_A[0]:
        avg_A[name] = sum(c[name] for c in cov_list_A) / len(cov_list_A)
    for name in cov_list_G[0]:
        avg_G[name] = sum(c[name] for c in cov_list_G) / len(cov_list_G)

    return avg_A, avg_G


def compute_inv_sqrt(cov_A, cov_G):
    inv_A, inv_G = {}, {}
    eps = 1e-3

    for name in cov_A:
        A = cov_A[name]
        eva, evc = torch.linalg.eigh(A + eps * torch.eye(A.size(0), device=A.device))
        inv_A[name] = evc @ torch.diag(eva.clamp(min=1e-6).rsqrt()) @ evc.T

        G = cov_G[name]
        evg, evcg = torch.linalg.eigh(G + eps * torch.eye(G.size(0), device=G.device))
        inv_G[name] = evcg @ torch.diag(evg.clamp(min=1e-6).rsqrt()) @ evcg.T

    return inv_A, inv_G


def precondition_grads(model, inv_A, inv_G):
    target = model._module if hasattr(model, "_module") else model

    for name, mod in target.named_modules():
        if name not in inv_A or name not in inv_G:
            continue
        if not hasattr(mod.weight, "grad_sample") or mod.weight.grad_sample is None:
            continue

        g_w = mod.weight.grad_sample

        if isinstance(mod, nn.Conv2d):
            bs, out_ch = g_w.shape[:2]
            g_w = g_w.view(bs, out_ch, -1)

        temp = torch.einsum("oj,bjk->bok", inv_G[name], g_w)
        new_w = torch.einsum("bok,ki->boi", temp, inv_A[name])

        if isinstance(mod, nn.Conv2d):
            new_w = new_w.view_as(mod.weight.grad_sample)

        mod.weight.grad_sample = new_w

        if mod.bias is not None and hasattr(mod.bias, "grad_sample") and mod.bias.grad_sample is not None:
            g_b = mod.bias.grad_sample
            new_b = torch.einsum("oj,bj->bo", inv_G[name], g_b)
            mod.bias.grad_sample = new_b


def apply_dp(model, noise_mult, clip_norm, batch_size):
    params = [p for p in model.parameters() if hasattr(p, "grad_sample") and p.grad_sample is not None]
    if not params:
        return

    device = params[0].device

    norms_sq = torch.zeros(batch_size, device=device)
    for p in params:
        norms_sq += p.grad_sample.reshape(batch_size, -1).norm(2, dim=1) ** 2
    norms = norms_sq.sqrt()

    clip_factor = (clip_norm / (norms + 1e-6)).clamp(max=1.0)

    for p in params:
        flat = p.grad_sample.reshape(batch_size, -1)
        clipped = flat * clip_factor.unsqueeze(1)
        summed = clipped.sum(dim=0)
        noise = torch.randn_like(summed) * (noise_mult * clip_norm)
        p.grad = ((summed + noise) / batch_size).view_as(p)
        p.grad_sample = None


def compute_cov_A(model, recorder, img_source, steps, device, img_size, num_classes, public_iter=None):
    if hasattr(model, "disable_hooks"):
        model.disable_hooks()

    cov_A_list = []

    for _ in range(steps):
        if img_source == "pink":
            data = get_pink_noise_batch(BATCH_SIZE, img_size, device, alpha=1.0)
            labels = torch.randint(0, num_classes, (BATCH_SIZE,), device=device)
        elif img_source == "pub":
            data, labels = next(public_iter)
            data = data.to(device)
            labels = labels.to(device).squeeze().long().clamp(max=num_classes - 1)
        else:
            raise ValueError(f"Unknown img_source for A: {img_source}")

        cov_A, _ = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_A_list.append(cov_A)

    if hasattr(model, "enable_hooks"):
        model.enable_hooks()

    avg_A = {}
    if cov_A_list:
        for name in cov_A_list[0]:
            avg_A[name] = sum(c[name] for c in cov_A_list) / len(cov_A_list)

    return avg_A


def compute_cov_G(model, recorder, img_source, label_source, steps, device, img_size, num_classes, public_iter=None):
    if img_source == "pink" and label_source == "pub":
        raise ValueError("G_pink_pub is invalid: pink noise images have no corresponding public labels!")

    if hasattr(model, "disable_hooks"):
        model.disable_hooks()

    cov_G_list = []

    for _ in range(steps):
        if img_source == "pink":
            data = get_pink_noise_batch(BATCH_SIZE, img_size, device, alpha=1.0)
            labels = torch.randint(0, num_classes, (BATCH_SIZE,), device=device)
        elif img_source == "pub":
            data, pub_labels = next(public_iter)
            data = data.to(device)
            pub_labels = pub_labels.to(device).squeeze().long().clamp(max=num_classes - 1)

            if label_source == "pub":
                labels = pub_labels
            elif label_source == "noise":
                labels = torch.randint(0, num_classes, (data.size(0),), device=device)
            else:
                raise ValueError(f"Unknown label_source for G: {label_source}")
        else:
            raise ValueError(f"Unknown img_source for G: {img_source}")

        _, cov_G = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_G_list.append(cov_G)

    if hasattr(model, "enable_hooks"):
        model.enable_hooks()

    avg_G = {}
    if cov_G_list:
        for name in cov_G_list[0]:
            avg_G[name] = sum(c[name] for c in cov_G_list) / len(cov_G_list)

    return avg_G


def compute_preconditioner_oracle(model, recorder, private_iter, steps, device, img_size, num_classes):
    if hasattr(model, "disable_hooks"):
        model.disable_hooks()

    cov_A_list, cov_G_list = [], []

    for _ in range(steps):
        data, labels = next(private_iter)
        data = data.to(device)
        labels = labels.to(device).squeeze().long()

        cov_A, cov_G = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_A_list.append(cov_A)
        cov_G_list.append(cov_G)

    if hasattr(model, "enable_hooks"):
        model.enable_hooks()

    return accumulate_covs(cov_A_list, cov_G_list)


def compute_cov_synthetic_clustered(model, recorder, steps, device, img_size, num_classes, jitter_scale=0.5):
    if hasattr(model, "disable_hooks"):
        model.disable_hooks()

    synthetic_task = SyntheticPinkClusterTask(
        num_classes=num_classes,
        img_size=img_size,
        device=device,
        jitter_scale=jitter_scale,
    )

    cov_A_list, cov_G_list = [], []

    for _ in range(steps):
        data, labels = synthetic_task.get_batch(BATCH_SIZE)
        cov_A, cov_G = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_A_list.append(cov_A)
        cov_G_list.append(cov_G)

    if hasattr(model, "enable_hooks"):
        model.enable_hooks()

    return accumulate_covs(cov_A_list, cov_G_list)


def make_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def train_epoch(model, train_loader, optimizer, inv_A, inv_G, noise_mult, clip_norm, device, use_sum_reduction=False):
    model.train()
    total_loss = 0
    n_batches = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device).squeeze().long()
        bs = data.size(0)

        optimizer.zero_grad()
        out = model(data)

        if use_sum_reduction:
            loss = nn.CrossEntropyLoss(reduction="sum")(out, target)
            loss.backward()
            total_loss += loss.item() / bs
        else:
            loss = nn.CrossEntropyLoss(reduction="mean")(out, target)
            loss.backward()
            total_loss += loss.item()

        if inv_A and inv_G:
            precondition_grads(model, inv_A, inv_G)

        apply_dp(model, noise_mult, clip_norm, bs)
        optimizer.step()

        n_batches += 1

    return total_loss / n_batches


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).squeeze().long()
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


def compute_preconditioner_for_type(model, recorder, precond_type, img_size, num_classes,
                                     private_iter, public_loader):
    public_iter = make_iterator(public_loader)
    cov_A, cov_G = {}, {}

    if precond_type == "oracle":
        cov_A, cov_G = compute_preconditioner_oracle(model, recorder, private_iter, PRECOND_STEPS, DEVICE, img_size, num_classes)

    elif precond_type == "identity":
        cov_A, cov_G = {}, {}

    elif precond_type == "A_pub__G_pub_pub":
        cov_A = compute_cov_A(model, recorder, "pub", PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)
        public_iter = make_iterator(public_loader)
        cov_G = compute_cov_G(model, recorder, "pub", "pub", PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == "A_pub__G_pub_noise":
        cov_A = compute_cov_A(model, recorder, "pub", PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)
        public_iter = make_iterator(public_loader)
        cov_G = compute_cov_G(model, recorder, "pub", "noise", PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == "A_pub__G_pink_noise":
        cov_A = compute_cov_A(model, recorder, "pub", PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)
        cov_G = compute_cov_G(model, recorder, "pink", "noise", PRECOND_STEPS, DEVICE, img_size, num_classes)

    elif precond_type == "A_pink__G_pub_pub":
        cov_A = compute_cov_A(model, recorder, "pink", PRECOND_STEPS, DEVICE, img_size, num_classes)
        cov_G = compute_cov_G(model, recorder, "pub", "pub", PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == "A_pink__G_pub_noise":
        cov_A = compute_cov_A(model, recorder, "pink", PRECOND_STEPS, DEVICE, img_size, num_classes)
        cov_G = compute_cov_G(model, recorder, "pub", "noise", PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == "A_pink__G_pink_noise":
        cov_A = compute_cov_A(model, recorder, "pink", PRECOND_STEPS, DEVICE, img_size, num_classes)
        cov_G = compute_cov_G(model, recorder, "pink", "noise", PRECOND_STEPS, DEVICE, img_size, num_classes)

    elif precond_type == "A_syn__G_syn_clustered":
        cov_A, cov_G = compute_cov_synthetic_clustered(
            model, recorder, PRECOND_STEPS, DEVICE, img_size, num_classes, jitter_scale=0.5
        )

    else:
        raise ValueError(f"Unknown preconditioner type: {precond_type}")

    return cov_A, cov_G


def run_single_experiment(scenario_config, precond_name, precond_type, cached_loaders, seed):
    set_seed(seed)

    img_size = scenario_config["img_size"]
    num_classes = scenario_config["classes"]
    private_dataset = scenario_config["private"]

    train_loader = cached_loaders["train_loader"]
    test_loader = cached_loaders["test_loader"]
    train_len = cached_loaders["train_len"]
    public_loader = cached_loaders["public_loader"]
    model_type = cached_loaders.get("model_type", "simple_cnn")

    use_pretrained = model_type in ["crossvit", "convnext"]

    model, _, _ = get_model(num_classes, img_size, private_dataset)
    model = model.to(DEVICE)

    if use_pretrained:
        model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
        target = model._module

        for param in target.parameters():
            param.requires_grad = False

        for param in target.classifier.parameters():
            param.requires_grad = True

        if model_type == "crossvit":
            if hasattr(target.backbone, "norm"):
                for param in target.backbone.norm.parameters():
                    param.requires_grad = True
            if hasattr(target.backbone, "blocks") and len(target.backbone.blocks) > 0:
                last_block = target.backbone.blocks[-1]
                for param in last_block.parameters():
                    param.requires_grad = True

        elif model_type == "convnext":
            if hasattr(target.backbone, "norm_pre"):
                for param in target.backbone.norm_pre.parameters():
                    param.requires_grad = True
            if hasattr(target.backbone, "stages") and len(target.backbone.stages) > 0:
                last_stage = target.backbone.stages[-1]
                for param in last_stage.parameters():
                    param.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        unfrozen_names = [name for name, param in target.named_parameters() if param.requires_grad]

        model_name = "CrossViT" if model_type == "crossvit" else "ConvNeXt"
        console.print(
            f"  [dim]+ {model_name} trainable params: "
            f"{len(trainable_params)} / {len(list(model.parameters()))}[/dim]"
        )
        console.print(
            f"  [dim]+ Unfrozen layers: "
            f"{unfrozen_names[:3]} ... {unfrozen_names[-3:] if len(unfrozen_names) > 3 else unfrozen_names}[/dim]"
        )

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
        )
        use_sum_reduction = True
    else:
        model = GradSampleModule(model)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        use_sum_reduction = False

    sample_rate = BATCH_SIZE / train_len
    noise_mult = get_noise_multiplier(
        target_epsilon=EPSILON, target_delta=DELTA,
        sample_rate=sample_rate, epochs=EPOCHS, accountant="rdp",
    )

    recorder = KFACRecorder(model, only_trainable=use_pretrained)

    private_iter = make_iterator(train_loader)

    cov_A, cov_G = compute_preconditioner_for_type(
        model, recorder, precond_type, img_size, num_classes, private_iter, public_loader,
    )

    if cov_A and cov_G:
        inv_A, inv_G = compute_inv_sqrt(cov_A, cov_G)
    else:
        inv_A, inv_G = {}, {}

    results = []
    for epoch in range(EPOCHS):
        if PRECOND_UPDATE_EVERY and PRECOND_UPDATE_EVERY > 0 and epoch > 0 and epoch % PRECOND_UPDATE_EVERY == 0:
            if precond_type != "identity":
                console.print(f"      [dim]Updating preconditioner at epoch {epoch + 1}[/dim]")
                private_iter = make_iterator(train_loader)
                cov_A, cov_G = compute_preconditioner_for_type(
                    model, recorder, precond_type, img_size, num_classes, private_iter, public_loader,
                )
                if cov_A and cov_G:
                    inv_A, inv_G = compute_inv_sqrt(cov_A, cov_G)

        loss = train_epoch(model, train_loader, optimizer, inv_A, inv_G, noise_mult, clip_norm=CLIP_NORM, device=DEVICE, use_sum_reduction=use_sum_reduction)
        acc = evaluate(model, test_loader, DEVICE)
        results.append({"epoch": epoch, "loss": loss, "accuracy": acc})
        console.print(f"    Epoch {epoch + 1}/{EPOCHS}: Loss={loss:.4f}, Acc={acc:.2f}%")

    return results


def run_scenario(scenario_key, scenario_config, precond_types, seeds):
    scenario_config = scenario_config.copy()
    img_size = scenario_config["img_size"]
    private_dataset = scenario_config["private"]
    public_dataset = scenario_config["public"]
    num_classes = scenario_config["classes"]

    _, actual_img_size, model_type_str = get_model(num_classes, img_size, private_dataset)

    use_pretrained = model_type_str in ["crossvit", "convnext"]

    if use_pretrained:
        img_size = actual_img_size
        scenario_config["img_size"] = img_size
        console.print(f"\n[yellow]Using {model_type_str.upper()} - resizing images to {img_size}x{img_size}[/yellow]")

    loader_result = get_dataset_loaders(private_dataset, BATCH_SIZE, img_size, use_imagenet_norm=use_pretrained)
    if private_dataset == "medmnist_global":
        train_loader, test_loader, train_len, actual_classes = loader_result
        scenario_config["classes"] = actual_classes
    else:
        train_loader, test_loader, train_len = loader_result

    public_loader, _, _ = get_dataset_loaders(public_dataset, BATCH_SIZE, img_size, use_imagenet_norm=use_pretrained)[:3]

    model_display_names = {
        "simple_cnn": "SimpleCNN",
        "crossvit": "CrossViT (frozen backbone)",
        "convnext": "ConvNeXt (frozen backbone)",
    }

    console.rule(f"[bold cyan]{scenario_config['name']}")
    console.print(f"[dim]{scenario_config['description']}[/dim]")
    console.print(f"[dim]Prediction: {scenario_config['prediction']}[/dim]")
    console.print(
        f"Private: {scenario_config['private']} | Public: {scenario_config['public']} | "
        f"Image: {scenario_config['img_size']}x{scenario_config['img_size']} | "
        f"Classes: {scenario_config['classes']}"
    )
    console.print(f"Model: {model_display_names.get(model_type_str, model_type_str)}")
    console.print(f"Seeds: {seeds}")

    cached_loaders = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_len": train_len,
        "public_loader": public_loader,
        "model_type": model_type_str,
    }

    all_results = {}

    for precond_name, precond_type in precond_types.items():
        console.print(f"\n  [bold]{precond_name}[/bold]")
        seed_results = {}

        for seed in seeds:
            console.print(f"    [dim]Seed {seed}[/dim]")
            results = run_single_experiment(scenario_config, precond_name, precond_type, cached_loaders, seed)
            seed_results[seed] = results

        n_epochs = len(seed_results[seeds[0]])
        aggregated = []

        for epoch_idx in range(n_epochs):
            accs = [seed_results[s][epoch_idx]["accuracy"] for s in seeds]
            losses = [seed_results[s][epoch_idx]["loss"] for s in seeds]
            aggregated.append({
                "epoch": epoch_idx,
                "accuracy": np.mean(accs),
                "accuracy_std": np.std(accs),
                "loss": np.mean(losses),
                "loss_std": np.std(losses),
            })

        final_accs = [seed_results[s][-1]["accuracy"] for s in seeds]
        console.print(f"    [green]Final Acc: {np.mean(final_accs):.2f}% +/- {np.std(final_accs):.2f}%[/green]")

        all_results[precond_name] = {
            "aggregated": aggregated,
            "seeds": seed_results,
            "final_mean": np.mean(final_accs),
            "final_std": np.std(final_accs),
        }

    return all_results


def print_scenario_summary(scenario_key, scenario_config, all_results):
    console.rule(f"[bold green]RESULTS: {scenario_config['name']} ({scenario_key})")

    table = Table(title=f"Results: {scenario_config['name']}")
    table.add_column("Method", style="cyan")
    table.add_column("Final Acc (mean +/- std)", justify="right")
    table.add_column("Best Acc", justify="right")

    for name, result_data in all_results.items():
        final_mean = result_data["final_mean"]
        final_std = result_data["final_std"]
        best_acc = max(r["accuracy"] for r in result_data["aggregated"])
        table.add_row(
            name,
            f"{final_mean:.2f}% +/- {final_std:.2f}%",
            f"{best_acc:.2f}%",
        )

    console.print(table)

    def get_acc(key):
        if key in all_results:
            return all_results[key]["final_mean"], all_results[key]["final_std"]
        return None, None

    oracle_mean, oracle_std = get_acc("Oracle (Private)")
    identity_mean, identity_std = get_acc("Identity (No Precond)")

    A_pub_G_pub_pub, A_pub_G_pub_pub_std = get_acc("A_pub G_pub_pub")
    A_pub_G_pub_noise, A_pub_G_pub_noise_std = get_acc("A_pub G_pub_noise")
    A_pub_G_pink_noise, A_pub_G_pink_noise_std = get_acc("A_pub G_pink_noise")

    A_pink_G_pub_pub, A_pink_G_pub_pub_std = get_acc("A_pink G_pub_pub")
    A_pink_G_pub_noise, A_pink_G_pub_noise_std = get_acc("A_pink G_pub_noise")
    A_pink_G_pink_noise, A_pink_G_pink_noise_std = get_acc("A_pink G_pink_noise")

    A_syn_G_syn, A_syn_G_syn_std = get_acc("A_syn G_syn (Clustered Pink)")

    console.print("\n[bold]Key Insights:[/bold]")

    if A_pub_G_pub_pub is not None and A_pink_G_pub_pub is not None:
        diff = A_pub_G_pub_pub - A_pink_G_pub_pub
        console.print(f"  A effect (G=pub_pub):  A_pub={A_pub_G_pub_pub:.2f}+/-{A_pub_G_pub_pub_std:.2f}% vs A_pink={A_pink_G_pub_pub:.2f}+/-{A_pink_G_pub_pub_std:.2f}%  (D={diff:+.2f}%)")
    if A_pub_G_pub_noise is not None and A_pink_G_pub_noise is not None:
        diff = A_pub_G_pub_noise - A_pink_G_pub_noise
        console.print(f"  A effect (G=pub_noise): A_pub={A_pub_G_pub_noise:.2f}+/-{A_pub_G_pub_noise_std:.2f}% vs A_pink={A_pink_G_pub_noise:.2f}+/-{A_pink_G_pub_noise_std:.2f}%  (D={diff:+.2f}%)")
    if A_pub_G_pink_noise is not None and A_pink_G_pink_noise is not None:
        diff = A_pub_G_pink_noise - A_pink_G_pink_noise
        console.print(f"  A effect (G=pink_noise): A_pub={A_pub_G_pink_noise:.2f}+/-{A_pub_G_pink_noise_std:.2f}% vs A_pink={A_pink_G_pink_noise:.2f}+/-{A_pink_G_pink_noise_std:.2f}%  (D={diff:+.2f}%)")

    if A_pub_G_pub_pub is not None and A_pub_G_pub_noise is not None:
        diff = A_pub_G_pub_pub - A_pub_G_pub_noise
        console.print(f"  Label effect (A=pub, G_img=pub): pub_label={A_pub_G_pub_pub:.2f}% vs noise_label={A_pub_G_pub_noise:.2f}%  (D={diff:+.2f}%)")
    if A_pink_G_pub_pub is not None and A_pink_G_pub_noise is not None:
        diff = A_pink_G_pub_pub - A_pink_G_pub_noise
        console.print(f"  Label effect (A=pink, G_img=pub): pub_label={A_pink_G_pub_pub:.2f}% vs noise_label={A_pink_G_pub_noise:.2f}%  (D={diff:+.2f}%)")

    if A_pub_G_pub_noise is not None and A_pub_G_pink_noise is not None:
        diff = A_pub_G_pub_noise - A_pub_G_pink_noise
        console.print(f"  G_img effect (A=pub, noise_label): G_pub={A_pub_G_pub_noise:.2f}% vs G_pink={A_pub_G_pink_noise:.2f}%  (D={diff:+.2f}%)")
    if A_pink_G_pub_noise is not None and A_pink_G_pink_noise is not None:
        diff = A_pink_G_pub_noise - A_pink_G_pink_noise
        console.print(f"  G_img effect (A=pink, noise_label): G_pub={A_pink_G_pub_noise:.2f}% vs G_pink={A_pink_G_pink_noise:.2f}%  (D={diff:+.2f}%)")

    if A_syn_G_syn is not None:
        console.print(f"\n  [bold]Synthetic Clustered: {A_syn_G_syn:.2f}+/-{A_syn_G_syn_std:.2f}%[/bold]")
        if A_pink_G_pink_noise is not None:
            diff = A_syn_G_syn - A_pink_G_pink_noise
            console.print(f"    vs A_pink G_pink_noise: D={diff:+.2f}%")
        if oracle_mean is not None:
            gap = oracle_mean - A_syn_G_syn
            console.print(f"    Gap to Oracle: {gap:.2f}%")

    console.print(f"\n[dim]Prediction: {scenario_config['prediction']}[/dim]")

    methods = {}
    if A_pub_G_pub_pub is not None:
        methods["A_pub G_pub_pub"] = A_pub_G_pub_pub
    if A_pub_G_pub_noise is not None:
        methods["A_pub G_pub_noise"] = A_pub_G_pub_noise
    if A_pub_G_pink_noise is not None:
        methods["A_pub G_pink_noise"] = A_pub_G_pink_noise
    if A_pink_G_pub_pub is not None:
        methods["A_pink G_pub_pub"] = A_pink_G_pub_pub
    if A_pink_G_pub_noise is not None:
        methods["A_pink G_pub_noise"] = A_pink_G_pub_noise
    if A_pink_G_pink_noise is not None:
        methods["A_pink G_pink_noise"] = A_pink_G_pink_noise
    if A_syn_G_syn is not None:
        methods["A_syn G_syn (Clustered)"] = A_syn_G_syn

    if methods:
        winner = max(methods, key=methods.get)
        console.print(f"[bold green]Winner: {winner} ({methods[winner]:.2f}%)[/bold green]")
    else:
        winner = "N/A"

    def fmt_mean_std(mean, std):
        if mean is not None:
            return f"{mean:.2f}+/-{std:.2f}"
        return None

    return {
        "scenario": scenario_key,
        "oracle": fmt_mean_std(oracle_mean, oracle_std),
        "identity": fmt_mean_std(identity_mean, identity_std),
        "A_pub_G_pub_pub": fmt_mean_std(A_pub_G_pub_pub, A_pub_G_pub_pub_std),
        "A_pub_G_pub_noise": fmt_mean_std(A_pub_G_pub_noise, A_pub_G_pub_noise_std),
        "A_pub_G_pink_noise": fmt_mean_std(A_pub_G_pink_noise, A_pub_G_pink_noise_std),
        "A_pink_G_pub_pub": fmt_mean_std(A_pink_G_pub_pub, A_pink_G_pub_pub_std),
        "A_pink_G_pub_noise": fmt_mean_std(A_pink_G_pub_noise, A_pink_G_pub_noise_std),
        "A_pink_G_pink_noise": fmt_mean_std(A_pink_G_pink_noise, A_pink_G_pink_noise_std),
        "A_syn_G_syn_clustered": fmt_mean_std(A_syn_G_syn, A_syn_G_syn_std),
        "winner": winner,
        "prediction": scenario_config["prediction"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive preconditioning transfer experiment (A vs G decomposition)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick run with a single seed and reduced epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run with a single specific seed (e.g. --seed 42)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override epsilon budget (default: 2.0)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario key to run (e.g. 4_TotalMismatch) or 'all'",
    )
    parser.add_argument(
        "--precond_types",
        type=str,
        default="all",
        help="Comma-separated preconditioner types or 'all'",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Override model type: simple_cnn, crossvit, convnext, auto",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save result CSV (default: results)",
    )
    return parser.parse_args()


def main():
    global MODEL_TYPE, EPSILON, EPOCHS

    args = parse_args()

    if args.model_type is not None:
        MODEL_TYPE = args.model_type
    if args.epsilon is not None:
        EPSILON = args.epsilon

    seeds = SEEDS
    if args.fast:
        seeds = [42]
        EPOCHS = 2
    if args.seed is not None:
        seeds = [args.seed]

    if args.scenario == "all":
        scenarios_to_run = SCENARIOS
    else:
        if args.scenario not in SCENARIOS:
            console.print(f"[red]Unknown scenario: {args.scenario}[/red]")
            console.print(f"[dim]Available: {list(SCENARIOS.keys())}[/dim]")
            return
        scenarios_to_run = {args.scenario: SCENARIOS[args.scenario]}

    if args.precond_types == "all":
        precond_types = ALL_PRECOND_TYPES
    else:
        requested = [t.strip() for t in args.precond_types.split(",")]
        precond_types = {k: v for k, v in ALL_PRECOND_TYPES.items() if v in requested}
        if not precond_types:
            console.print(f"[red]No matching preconditioner types found.[/red]")
            console.print(f"[dim]Available: {list(ALL_PRECOND_TYPES.values())}[/dim]")
            return

    console.rule("[bold cyan]Comprehensive Preconditioning Transfer Experiment")
    console.print(f"Device:     {DEVICE}")
    console.print(f"Scenarios:  {list(scenarios_to_run.keys())}")
    console.print(f"Precond:    {list(precond_types.keys())}")
    console.print(f"Epsilon:    {EPSILON}")
    console.print(f"Epochs:     {EPOCHS}")
    console.print(f"Batch:      {BATCH_SIZE}")
    console.print(f"LR:         {LEARNING_RATE}")
    console.print(f"Seeds:      {seeds}")
    console.print(f"Model:      {MODEL_TYPE}")
    if PRECOND_UPDATE_EVERY and PRECOND_UPDATE_EVERY > 0:
        console.print(f"Precond update: every {PRECOND_UPDATE_EVERY} epoch(s)")
    else:
        console.print(f"Precond update: once at start")

    all_scenario_results = {}
    summaries = []

    for scenario_key, scenario_config in scenarios_to_run.items():
        results = run_scenario(scenario_key, scenario_config, precond_types, seeds)
        all_scenario_results[scenario_key] = results
        summary = print_scenario_summary(scenario_key, scenario_config, results)
        summaries.append(summary)

    if len(summaries) > 1:
        console.rule("[bold cyan]Grand Summary: All Scenarios")
        grand_table = Table(title="Grand Summary (mean +/- std)")
        grand_table.add_column("Scenario", style="cyan")
        grand_table.add_column("A_pub G_pp", justify="right")
        grand_table.add_column("A_pub G_pn", justify="right")
        grand_table.add_column("A_pub G_kn", justify="right")
        grand_table.add_column("A_pink G_pp", justify="right")
        grand_table.add_column("A_pink G_pn", justify="right")
        grand_table.add_column("A_pink G_kn", justify="right")
        grand_table.add_column("Winner", style="green")

        for s in summaries:
            grand_table.add_row(
                s["scenario"],
                s["A_pub_G_pub_pub"] or "N/A",
                s["A_pub_G_pub_noise"] or "N/A",
                s["A_pub_G_pink_noise"] or "N/A",
                s["A_pink_G_pub_pub"] or "N/A",
                s["A_pink_G_pub_noise"] or "N/A",
                s["A_pink_G_pink_noise"] or "N/A",
                s["winner"],
            )

        console.print(grand_table)

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = output_dir / f"transfer_results_{timestamp}.csv"

    fieldnames = [
        "scenario", "oracle", "identity",
        "A_pub_G_pub_pub", "A_pub_G_pub_noise", "A_pub_G_pink_noise",
        "A_pink_G_pub_pub", "A_pink_G_pub_noise", "A_pink_G_pink_noise",
        "A_syn_G_syn_clustered", "winner", "prediction",
    ]
    save_results_csv(summaries, csv_filename, columns=fieldnames)
    console.print(f"\n[bold green]Results saved to {csv_filename}[/bold green]")


if __name__ == "__main__":
    main()
