# -*- coding: utf-8 -*-
"""
ResNet backbone + Spiking FC head experiments.

Architecture:
    ResNet18/50 (small-image variant) -> features (extracted once per image)
    -> FC(feat_dim -> hidden) -> spiking neuron x T -> FC(hidden -> classes)

Features are extracted once from the ANN backbone, then the spiking head
runs T timesteps on the same features (static datasets) or processes T
event frames through the backbone (neuromorphic datasets).

Model naming (CLI key -> paper name):
    lif         -> LIF
    ultratlif   -> UltraLIF  (temporal, 2-term LSE)
    ultratplif  -> UltraPLIF (temporal, learnable tau)
    ultradlif   -> UltraDLIF (spatial, 3-term LSE)
    ultradplif  -> UltraDPLIF(spatial, learnable tau)

Usage:
    python experiments/train_resnet.py --dataset cifar10 --backbone resnet18
    python experiments/train_resnet.py --model ultratlif --timesteps 1 5 10
    python experiments/train_resnet.py --backbone resnet50 --hidden 2048 --batch-size 64
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet50
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralif.neurons.ultra import UltraLIF, UltraPLIF
from ultralif.neurons.ultradlif import UltraDLIF, UltraDPLIF
from ultralif.neurons.lif import LIF
from ultralif.datasets.utils import set_seed
from ultralif.training.logging import TeeLogger


# =============================================================================
# RESNET BACKBONES (small-image variants)
# =============================================================================

class ResNet18Small(nn.Module):
    """
    ResNet-18 backbone for small images (28x28 or 32x32).

    Modifications vs standard ResNet-18:
    - First conv: 3x3 stride=1 (was 7x7 stride=2)
    - MaxPool replaced with Identity (preserves spatial resolution)

    Args:
        in_channels: Input channels (1 for MNIST, 3 for CIFAR-10, 2 for DVS).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool,
        )
        self.feat_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h.view(h.size(0), -1)


class ResNet50Small(nn.Module):
    """
    ResNet-50 backbone for small images. Output: 2048-dim features.

    Args:
        in_channels: Input channels.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        base = resnet50(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool,
        )
        self.feat_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h.view(h.size(0), -1)


BACKBONE_REGISTRY = {"resnet18": ResNet18Small, "resnet50": ResNet50Small}


# =============================================================================
# RESNET-SNN MODEL
# =============================================================================

class ResNetSNN(nn.Module):
    """
    ResNet backbone with spiking FC head.

    Static datasets: features extracted once, spiking head runs T timesteps.
    Neuromorphic:    backbone processes each of the T event frames.

    Args:
        neuron: Instantiated spiking neuron.
        hidden_dim: FC hidden width.
        timesteps: Number of time steps T.
        in_channels: Input channels for backbone.
        num_classes: Output class count.
        neuromorphic: If True, treat input as [B, T, C, H, W] event frames.
        backbone_cls: ResNet18Small or ResNet50Small.
    """

    def __init__(
        self,
        neuron: nn.Module,
        hidden_dim: int = 256,
        timesteps: int = 10,
        in_channels: int = 3,
        num_classes: int = 10,
        neuromorphic: bool = False,
        backbone_cls=ResNet18Small,
    ):
        super().__init__()
        self.backbone = backbone_cls(in_channels=in_channels)
        self.fc1 = nn.Linear(self.backbone.feat_dim, hidden_dim)
        self.neuron = neuron
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.T = timesteps
        self.neuromorphic = neuromorphic
        self.last_spike_rate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point else torch.float32

        self.neuron.reset(batch, device)
        out_sum = torch.zeros(batch, self.fc2.out_features, device=device, dtype=dtype)
        spike_sum = 0.0

        if self.neuromorphic:
            T = x.shape[1] if x.dim() == 5 else self.T
            for t in range(T):
                feat = self.backbone(x[:, t].float())
                h = self.fc1(feat)
                spike = self.neuron(h)
                spike_sum = spike_sum + spike.mean()
                out_sum = out_sum + self.fc2(spike)
            self.last_spike_rate = spike_sum / T
            return out_sum / T
        else:
            feat = self.backbone(x.float())
            for t in range(self.T):
                h = self.fc1(feat)
                spike = self.neuron(h)
                spike_sum = spike_sum + spike.mean()
                out_sum = out_sum + self.fc2(spike)
            self.last_spike_rate = spike_sum / self.T
            return out_sum / self.T


# =============================================================================
# DATASET LOADING (ResNet-specific; backbone expects full images, not rate-encoded)
# =============================================================================

def _get_dataset(name: str, batch_size: int, timesteps: int, num_workers: int):
    data_dir = Path(__file__).parent.parent / "data"
    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    if name == "cifar10":
        tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
        ])
        te = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
        ])
        return (DataLoader(datasets.CIFAR10(data_dir, True, download=True, transform=tr), batch_size, shuffle=True, **kw),
                DataLoader(datasets.CIFAR10(data_dir, False, download=True, transform=te), batch_size, **kw))

    elif name == "mnist":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return (DataLoader(datasets.MNIST(data_dir, True, download=True, transform=t), batch_size, shuffle=True, **kw),
                DataLoader(datasets.MNIST(data_dir, False, download=True, transform=t), batch_size, **kw))

    elif name == "fashion":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        return (DataLoader(datasets.FashionMNIST(data_dir, True, download=True, transform=t), batch_size, shuffle=True, **kw),
                DataLoader(datasets.FashionMNIST(data_dir, False, download=True, transform=t), batch_size, **kw))

    elif name in ("nmnist", "dvs"):
        import tonic, tonic.transforms as TT
        def nl(ds, shuffle=False):
            return DataLoader(ds, batch_size, shuffle=shuffle,
                              collate_fn=tonic.collation.PadTensors(batch_first=True),
                              num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
        if name == "nmnist":
            sz = tonic.datasets.NMNIST.sensor_size
            ft = TT.Compose([TT.ToFrame(sensor_size=sz, n_time_bins=timesteps), torch.from_numpy])
            return nl(tonic.datasets.NMNIST(str(data_dir), True, transform=ft), True), nl(tonic.datasets.NMNIST(str(data_dir), False, transform=ft))
        else:
            sz = tonic.datasets.DVSGesture.sensor_size
            ft = TT.Compose([TT.ToFrame(sensor_size=sz, n_time_bins=timesteps), torch.from_numpy])
            return nl(tonic.datasets.DVSGesture(str(data_dir), True, transform=ft), True), nl(tonic.datasets.DVSGesture(str(data_dir), False, transform=ft))

    raise ValueError(f"Unknown dataset: {name!r}")


# =============================================================================
# TRAINING LOOP
# =============================================================================

DATASET_CONFIGS = {
    "cifar10": (3, 10, False),
    "mnist":   (1, 10, False),
    "fashion": (1, 10, False),
    "nmnist":  (2, 10, True),
    "dvs":     (2, 11, True),
}

NEURON_REGISTRY = {
    "lif":        LIF,
    "ultratlif":  UltraLIF,    # temporal, 2-term LSE (paper: UltraLIF)
    "ultratplif": UltraPLIF,   # temporal + learnable tau (paper: UltraPLIF)
    "ultradlif":  UltraDLIF,   # spatial, 3-term LSE (paper: UltraDLIF)
    "ultradplif": UltraDPLIF,  # spatial + learnable tau (paper: UltraDPLIF)
}

PAPER_NAMES = {
    "lif": "LIF", "ultratlif": "UltraLIF", "ultratplif": "UltraPLIF",
    "ultradlif": "UltraDLIF", "ultradplif": "UltraDPLIF",
}


def _train_one(model, train_loader, test_loader, epochs, lr, device, save_path=None, track_spikes=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_sr = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        srs = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                if track_spikes and model.last_spike_rate is not None:
                    v = model.last_spike_rate
                    srs.append(v.item() if hasattr(v, "item") else float(v))

        acc = 100.0 * correct / total
        sr = sum(srs) / len(srs) if srs else 0.0
        energy = sr / 0.5

        if acc > best_acc:
            best_acc = acc
            best_sr = sr
            if save_path:
                torch.save({"epoch": epoch, "acc": acc, "spike_rate": sr,
                            "state_dict": model.state_dict()}, save_path)

        if epoch % 10 == 0 or epoch == 1:
            eps_str = ""
            if hasattr(model.neuron, "eps"):
                eps_str = f"  eps={model.neuron.eps.item():.3f}"
            print(f"  Epoch {epoch:3d}: acc={acc:.4f}%  sr={sr:.4f}  E={energy:.2f}x  best={best_acc:.4f}%{eps_str}")

    return best_acc, best_sr, (best_sr / 0.5 if best_sr else float("nan"))


def main():
    parser = argparse.ArgumentParser(description="ResNet-SNN Training")
    parser.add_argument("--dataset", default="cifar10", choices=list(DATASET_CONFIGS))
    parser.add_argument("--model", nargs="+", default=list(NEURON_REGISTRY.keys()))
    parser.add_argument("--timesteps", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-track-spikes", dest="track_spikes", action="store_false")
    parser.set_defaults(track_spikes=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path(__file__).parent.parent
    ckpt_dir = root / "checkpoints" / f"{args.backbone}_{args.dataset}"
    results_dir = root / "results" / f"{args.backbone}_{args.dataset}"
    log_dir = root / "logs"
    for d in (ckpt_dir, results_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{args.backbone}_{args.dataset}_{ts}.log"
    tee = TeeLogger(str(log_path))
    sys.stdout = tee
    print(f"Logging to: {log_path}")

    in_channels, num_classes, neuromorphic = DATASET_CONFIGS[args.dataset]
    backbone_cls = BACKBONE_REGISTRY[args.backbone]

    print("=" * 70)
    print(f"{args.backbone.upper()}-SNN  Dataset={args.dataset.upper()}  backbone_cls={backbone_cls.__name__}")
    print(f"Models: {args.model}  T: {args.timesteps}  hidden={args.hidden}")
    print("=" * 70)

    neuro_T = args.timesteps[0] if neuromorphic else 10
    train_loader, test_loader = _get_dataset(args.dataset, args.batch_size, neuro_T, args.num_workers)

    all_results = {}

    for T in args.timesteps:
        print(f"\n{'='*70}\nT = {T}\n{'='*70}")
        for key in args.model:
            if key not in NEURON_REGISTRY:
                print(f"Unknown model: {key}, skipping")
                continue
            paper = PAPER_NAMES[key]
            print(f"\n--- {paper} | T={T} ---")
            set_seed(args.seed)

            neuron_cls = NEURON_REGISTRY[key]
            neuron = neuron_cls(args.hidden)
            model = ResNetSNN(
                neuron, hidden_dim=args.hidden, timesteps=T,
                in_channels=in_channels, num_classes=num_classes,
                neuromorphic=neuromorphic, backbone_cls=backbone_cls,
            ).to(device)

            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Params: {params:,}")

            ckpt = ckpt_dir / f"{key}_T{T}_h{args.hidden}_seed{args.seed}_{args.backbone}.pt"
            best_acc, best_sr, best_e = _train_one(
                model, train_loader, test_loader,
                epochs=args.epochs, lr=args.lr, device=device,
                save_path=str(ckpt), track_spikes=args.track_spikes,
            )
            print(f"\nRESULT: {paper} T={T} | Best={best_acc:.4f}% | SR={best_sr:.4f} | E={best_e:.2f}x")
            all_results.setdefault(key, {})[T] = {"acc": best_acc, "spike_rate": best_sr, "energy": best_e, "params": params}

    # Summary
    print("\n" + "=" * 70 + f"\nSUMMARY — {args.backbone.upper()}-SNN {args.dataset.upper()}\n" + "=" * 70)
    for key in args.model:
        if key not in all_results:
            continue
        row = f"{PAPER_NAMES[key]:<14}"
        for T in args.timesteps:
            r = all_results[key].get(T, {})
            row += f"  T={T}: {r.get('acc', float('nan')):.2f}%"
        print(row)

    out_path = results_dir / f"{args.backbone}_{args.dataset}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults: {out_path}")

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
