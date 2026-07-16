"""Ablation: Transfer alignment across domain match/mismatch scenarios."""

import sys
import copy
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
import numpy as np
from rich.console import Console
from rich.table import Table

from dp_kfac.trainer import set_seed, evaluate
from dp_kfac.privacy import clip_and_noise_gradients
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances as compute_kfac_covariances
from dp_kfac.covariance import compute_inverse_sqrt, accumulate_covariances
from dp_kfac.types import CovariancePair
from dp_kfac.precondition import precondition_per_sample_gradients
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.methods import estimate_adadps_preconditioner, precondition_per_sample_gradients_adadps
from dp_kfac.results import save_results_csv

SCENARIOS = {
    "1_DistMatch": {
        "name": "Ideal Transfer",
        "private": "fashionmnist",
        "public": "mnist",
        "img_size": 28,
        "classes": 10,
        "in_channels": 3,
        "prediction": "Public KFAC should win (close to Oracle)",
    },
    "2_DistDisjoint": {
        "name": "Texture Mismatch",
        "private": "pathmnist",
        "public": "mnist",
        "img_size": 28,
        "classes": 9,
        "in_channels": 3,
        "prediction": "Pink Noise should win (Public A hurts)",
    },
    "3_TaskHarder": {
        "name": "Task Shift",
        "private": "cifar100",
        "public": "cifar10",
        "img_size": 32,
        "classes": 100,
        "in_channels": 3,
        "prediction": "Public/Pink Noise competitive (A aligned, G differs)",
    },
    "4_TotalMismatch": {
        "name": "Total Mismatch",
        "private": "cifar100",
        "public": "mnist",
        "img_size": 32,
        "classes": 100,
        "in_channels": 3,
        "prediction": "Pink Noise should win (both A and G mismatched)",
    },
    "5_MedGlobal": {
        "name": "MedMNIST-Global",
        "private": "medmnist_global",
        "public": "mnist",
        "img_size": 28,
        "classes": 47,  # dynamically set
        "in_channels": 3,
        "prediction": "Pink Noise should win (no center bias, mixed domain)",
    },
}

SEEDS = [42, 7, 91, 23, 58, 134, 76, 3, 219, 65]
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 5e-3
EPSILON = 2.0
DELTA = 1e-5
CLIP_NORM = 0.5
PRECOND_BATCHES = 50  # batches used for covariance estimation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STANDARD_MEAN = [0.5, 0.5, 0.5]
STANDARD_STD = [0.5, 0.5, 0.5]


class SimpleCNN3ch(nn.Module):
    """SimpleCNN with 3 input channels for cross-domain experiments."""
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


def _make_transform(img_size, grayscale_to_rgb=False):
    t = []
    t.append(transforms.Resize((img_size, img_size)))
    if grayscale_to_rgb:
        t.append(transforms.Grayscale(num_output_channels=3))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(STANDARD_MEAN, STANDARD_STD))
    return transforms.Compose(t)


def _load_dataset(dataset_cls, img_size, grayscale_to_rgb=False, batch_size=256):
    transform = _make_transform(img_size, grayscale_to_rgb)
    train_ds = dataset_cls("./data", train=True, download=True, transform=transform)
    test_ds = dataset_cls("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=2)
    return train_loader, test_loader, len(train_ds)


def get_loaders(name, img_size, batch_size=256):
    """Load a dataset by name. Returns (train_loader, test_loader, train_size).
    For medmnist_global, returns (train_loader, test_loader, train_size, num_classes)."""
    if name == "mnist":
        return _load_dataset(datasets.MNIST, img_size, grayscale_to_rgb=True, batch_size=batch_size)
    elif name == "fashionmnist":
        return _load_dataset(datasets.FashionMNIST, img_size, grayscale_to_rgb=True, batch_size=batch_size)
    elif name == "cifar10":
        return _load_dataset(datasets.CIFAR10, img_size, batch_size=batch_size)
    elif name == "cifar100":
        return _load_dataset(datasets.CIFAR100, img_size, batch_size=batch_size)
    elif name == "pathmnist":
        return _load_pathmnist(img_size, batch_size)
    elif name == "medmnist_global":
        return _load_medmnist_global(img_size, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_pathmnist(img_size, batch_size):
    import medmnist
    info = medmnist.INFO["pathmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    transform = _make_transform(img_size)
    train_ds = DataClass(split="train", transform=transform, download=True, root="./data")
    test_ds = DataClass(split="test", transform=transform, download=True, root="./data")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=2)
    return train_loader, test_loader, len(train_ds)


class _MedMNISTSubset(torch.utils.data.Dataset):
    def __init__(self, medmnist_ds, label_offset, transform, max_samples=None):
        self.ds = medmnist_ds
        self.label_offset = label_offset
        self.transform = transform
        self.max_samples = min(max_samples, len(medmnist_ds)) if max_samples else len(medmnist_ds)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
        label = int(label.item() if hasattr(label, "item") else label) + self.label_offset
        return img, label


def _load_medmnist_global(img_size, batch_size):
    import medmnist
    transform = _make_transform(img_size, grayscale_to_rgb=True)
    flags = ["pathmnist", "bloodmnist", "dermamnist", "octmnist", "tissuemnist", "organamnist"]

    train_datasets, test_datasets = [], []
    offset = 0
    for flag in flags:
        info = medmnist.INFO[flag]
        n_classes = len(info["label"])
        DataClass = getattr(medmnist, info["python_class"])
        train_ds = DataClass(split="train", transform=None, download=True, root="./data")
        test_ds = DataClass(split="test", transform=None, download=True, root="./data")
        train_datasets.append(_MedMNISTSubset(train_ds, offset, transform, len(train_ds) // 2))
        test_datasets.append(_MedMNISTSubset(test_ds, offset, transform, len(test_ds) // 2))
        offset += n_classes

    full_train = ConcatDataset(train_datasets)
    full_test = ConcatDataset(test_datasets)
    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    test_loader = DataLoader(full_test, batch_size=256, num_workers=2)
    return train_loader, test_loader, len(full_train), offset


def estimate_kfac_from_loader(model, loader, device, num_classes, num_batches=50):
    """Estimate KFAC covariances from a data loader (public or private)."""
    recorder = KFACRecorder(model)
    recorder.enable()
    model.eval()

    cov_list = []
    count = 0

    for batch in loader:
        if count >= num_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
            if target.dim() > 1:
                target = target.squeeze()
        else:
            data = batch
            target = torch.randint(0, num_classes, (data.size(0),))
        data, target = data.to(device), target.long().to(device)

        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        cov = compute_kfac_covariances(model, recorder.activations, recorder.backprops)
        cov_list.append(cov)
        recorder.clear()
        count += 1

    recorder.disable()
    model.zero_grad(set_to_none=True)

    return accumulate_covariances(cov_list)


def estimate_kfac_from_noise(model, device, num_classes, input_shape,
                             num_batches=50, batch_size=64):
    """Estimate KFAC covariances from pink noise."""
    recorder = KFACRecorder(model)
    recorder.enable()
    model.eval()

    cov_list = []

    for _ in range(num_batches):
        noise_data = generate_pink_noise(batch_size, input_shape, device)
        noise_target = torch.randint(0, num_classes, (batch_size,), device=device)

        model.zero_grad()
        output = model(noise_data)
        loss = F.cross_entropy(output, noise_target)
        loss.backward()

        cov = compute_kfac_covariances(model, recorder.activations, recorder.backprops)
        cov_list.append(cov)
        recorder.clear()

    recorder.disable()
    model.zero_grad(set_to_none=True)

    return accumulate_covariances(cov_list)


def train_epoch_dp(model, train_loader, optimizer, inv_A, inv_G,
                   noise_multiplier, clip_norm, device):
    """One DP training epoch with optional KFAC preconditioning."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
            if target.dim() > 1:
                target = target.squeeze()
        else:
            data = batch
            target = torch.zeros(data.size(0), dtype=torch.long)
        data, target = data.to(device), target.long().to(device)
        bs = data.size(0)

        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output, target, reduction="sum")
        loss.backward()

        if inv_A and inv_G:
            precondition_per_sample_gradients(model, inv_A, inv_G)

        clip_and_noise_gradients(model, noise_multiplier, clip_norm, bs)
        optimizer.step()

        total_loss += loss.item() / bs
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, target = batch[0], batch[1]
                if target.dim() > 1:
                    target = target.squeeze()
            else:
                data = batch
                target = torch.zeros(data.size(0), dtype=torch.long)
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0


def _train_epoch_adadps(model, train_loader, optimizer, preconditioner,
                        noise_multiplier, clip_norm, device):
    """One DP training epoch with AdaDPS diagonal preconditioning."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
            if target.dim() > 1:
                target = target.squeeze()
        else:
            data = batch
            target = torch.zeros(data.size(0), dtype=torch.long)
        data, target = data.to(device), target.long().to(device)
        bs = data.size(0)

        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output, target, reduction="sum")
        loss.backward()

        precondition_per_sample_gradients_adadps(model, preconditioner)
        clip_and_noise_gradients(model, noise_multiplier, clip_norm, bs)
        optimizer.step()

        total_loss += loss.item() / bs
        n_batches += 1

    return total_loss / max(n_batches, 1)
def run_single(model_template, train_loader, test_loader, source_loader,
               device, num_classes, img_size, method, seed,
               epochs, epsilon, delta, clip_norm, lr, num_precond_batches):
    """Run a single (method, seed) experiment. Returns list of per-epoch dicts."""
    set_seed(seed)
    model = copy.deepcopy(model_template).to(device)
    model = GradSampleModule(model, batch_first=True, loss_reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_size = len(train_loader.dataset)
    sample_rate = BATCH_SIZE / train_size
    noise_mult = get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta,
        sample_rate=sample_rate, epochs=epochs, accountant="rdp",
    )

    inv_A, inv_G = {}, {}
    adadps_precond = None

    if method.startswith("AdaDPS"):
        if method == "AdaDPS (Public)":
            adadps_precond = estimate_adadps_preconditioner(model, source_loader, device)
        elif method == "AdaDPS (Pink)":
            # Estimate from pink noise batches
            from dp_kfac.optimizer import generate_pink_noise as _gen_pink
            pink_batches = []
            input_shape = (3, img_size, img_size)
            for _ in range(min(num_precond_batches, 20)):
                pink_batches.append((_gen_pink(64, input_shape, device),
                                     torch.randint(0, num_classes, (64,), device=device)))
            pink_loader = pink_batches
            adadps_precond = estimate_adadps_preconditioner(model, pink_loader, device)
    elif method != "DP-SGD":
        clean_model = copy.deepcopy(model_template).to(device)
        input_shape = (3, img_size, img_size)

        if method == "KFAC (Oracle)":
            covs = estimate_kfac_from_loader(clean_model, train_loader, device,
                                             num_classes, num_precond_batches)
        elif method == "KFAC (Public)":
            covs = estimate_kfac_from_loader(clean_model, source_loader, device,
                                             num_classes, num_precond_batches)
        elif method == "KFAC (Pink Noise)":
            covs = estimate_kfac_from_noise(clean_model, device, num_classes,
                                            input_shape, num_precond_batches)
        else:
            covs = CovariancePair(A={}, G={})

        if covs.A:
            inv_A, inv_G = compute_inverse_sqrt(covs)

        del clean_model

    history = []
    for epoch in range(epochs):
        if adadps_precond is not None:
            loss = _train_epoch_adadps(model, train_loader, optimizer, adadps_precond,
                                       noise_mult, clip_norm, device)
        else:
            loss = train_epoch_dp(model, train_loader, optimizer, inv_A, inv_G,
                                  noise_mult, clip_norm, device)
        acc = evaluate_model(model, test_loader, device)
        history.append({"epoch": epoch + 1, "loss": loss, "accuracy": acc})

    return history


def run_scenario(scenario_key, config, seeds, epochs, console,
                 num_precond_batches, output_dir):
    """Run all methods for a scenario. Returns results list."""
    console.rule(f"[bold cyan]{config['name']} ({scenario_key})")
    console.print(f"  Private: {config['private']}  Public: {config['public']}")
    console.print(f"  Classes: {config['classes']}  Image size: {config['img_size']}")
    console.print(f"  Prediction: {config['prediction']}")
    console.print()

    img_size = config["img_size"]
    num_classes = config["classes"]

    console.print("[yellow]  Loading datasets...[/yellow]")
    loader_result = get_loaders(config["private"], img_size, BATCH_SIZE)
    if config["private"] == "medmnist_global":
        train_loader, test_loader, train_size, num_classes = loader_result
        config["classes"] = num_classes  # update dynamically
    else:
        train_loader, test_loader, train_size = loader_result

    pub_loader, _, _ = get_loaders(config["public"], img_size, BATCH_SIZE)[:3]
    console.print(f"[green]  Private: {train_size} samples, {num_classes} classes[/green]")

    model_template = SimpleCNN3ch(num_classes=num_classes, img_size=img_size)

    methods = [
        "DP-SGD", "AdaDPS (Public)", "AdaDPS (Pink)",
        "KFAC (Oracle)", "KFAC (Public)", "KFAC (Pink Noise)",
    ]
    results = []

    for method in methods:
        console.print(f"\n  [bold]{method}[/bold]")
        for seed in seeds:
            console.print(f"    Seed {seed}...", end=" ")
            history = run_single(
                model_template, train_loader, test_loader, pub_loader,
                DEVICE, num_classes, img_size, method, seed,
                epochs, EPSILON, DELTA, CLIP_NORM, LEARNING_RATE,
                num_precond_batches,
            )
            final = history[-1]
            best_acc = max(h["accuracy"] for h in history)
            console.print(f"Final acc={final['accuracy']*100:.2f}%  Best={best_acc*100:.2f}%")
            results.append({
                "Scenario": scenario_key,
                "Method": method,
                "Seed": seed,
                "Final_Accuracy": final["accuracy"],
                "Best_Accuracy": best_acc,
                "Final_Loss": final["loss"],
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transfer alignment ablation across domain match/mismatch scenarios"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Quick run: 1 seed, 3 epochs, scenarios 1 & 4 only")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run with a single specific seed")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a specific scenario (e.g. 1_DistMatch)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results CSV")
    args = parser.parse_args()

    console = Console()

    seeds = SEEDS
    epochs = EPOCHS
    precond_batches = PRECOND_BATCHES

    if args.fast:
        seeds = [42]
        epochs = 3
        precond_batches = 10

    if args.seed is not None:
        seeds = [args.seed]

    if args.scenario:
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    elif args.fast:
        scenarios = {k: SCENARIOS[k] for k in ["1_DistMatch", "4_TotalMismatch"]}
    else:
        scenarios = SCENARIOS

    console.rule("[bold]Transfer Alignment Ablation")
    console.print(f"Device:   {DEVICE}")
    console.print(f"Epsilon:  {EPSILON}")
    console.print(f"Epochs:   {epochs}")
    console.print(f"Seeds:    {seeds}")
    console.print(f"Scenarios: {list(scenarios.keys())}")
    console.print()

    all_results = []

    for scenario_key, config in scenarios.items():
        scenario_results = run_scenario(
            scenario_key, config.copy(), seeds, epochs, console,
            precond_batches, args.output_dir,
        )
        all_results.extend(scenario_results)

    output_path = Path(args.output_dir) / "transfer_alignment_results.csv"
    save_results_csv(
        all_results, output_path,
        columns=["Scenario", "Method", "Seed", "Final_Accuracy", "Best_Accuracy", "Final_Loss"],
    )
    console.print(f"\n[green]Results saved to {output_path}[/green]")

    table = Table(title="Transfer Alignment Summary")
    table.add_column("Scenario", style="cyan")
    table.add_column("Method", style="white")
    table.add_column("Final Acc (mean +/- std)", justify="right")
    table.add_column("Best Acc (mean +/- std)", justify="right")

    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        key = (r["Scenario"], r["Method"])
        agg[key].append(r)

    for (scenario, method), runs in sorted(agg.items()):
        final_accs = [r["Final_Accuracy"] for r in runs]
        best_accs = [r["Best_Accuracy"] for r in runs]
        fm, fs = np.mean(final_accs) * 100, np.std(final_accs) * 100
        bm, bs = np.mean(best_accs) * 100, np.std(best_accs) * 100
        table.add_row(scenario, method, f"{fm:.2f} +/- {fs:.2f}", f"{bm:.2f} +/- {bs:.2f}")

    console.print(table)


if __name__ == "__main__":
    main()
