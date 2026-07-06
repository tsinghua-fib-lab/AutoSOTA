#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_frag_fault_to_vhdl.py

End-to-end example script:
  1) Train an SNN with (a) fragmentation and (b) fault-injection enabled.
  2) Convert the trained SNN into VHDL using Spiker+ (spikerplus).

What this script *does* generate in VHDL:
  - Only the SNN accelerator (Spiker+ network / optional FullAccelerator interface).
  - Fragmentation + Poisson/rate encoding are *input preprocessing*; they run on CPU
    (or your host) and produce the input spike train that you feed to the FPGA.

Key compatibility notes for “VHDL works like the trained model”:
  - Use Spiker+ NetBuilder to build the network (no bias terms, supported neuron models).
  - Keep beta such that (1 - beta) is an exact power-of-two (Spiker+ extracts a shift
    from beta to implement leak as a right-shift in VHDL).
  - Train with the same number of time-steps (n_cycles) you will synthesize in VHDL.

Requirements (minimal):
  - torch, torchvision
  - snntorch (required by spikerplus)
  - spikerplus (either pip install spikerplus OR point --spiker_root to Spiker-public/spiker)

Your additional local files (same folder as this script, or on PYTHONPATH):
  - algorithmic_fragmentation.py
  - fault_injection.py
  - (optional) learnable_fragmentation.py

Example:
  python train_frag_fault_to_vhdl.py \
      --data_dir ./data \
      --out_dir  ./out_run \
      --n_cycles 8 \
      --hidden 128 \
      --epochs 5 \
      --frag_method combo --overlap \
      --fault_ratio 0.05 --fault_type stuck --fault_start_epoch 2 \
      --weights_bw 8 --neurons_bw 12 --fp_dec 6 \
      --functional_vhdl False --interface False \
      --spiker_root ./Spiker-public/spiker

Outputs:
  out_run/
    checkpoint.pt
    net_dict.json
    vhdl/               (generated VHDL + ROM init files)
    sample_in_spikes.txt (example stimulus: one sample spike-train)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Your provided modules ---
from algorithmic_fragmentation import batch_dynamic_fragments
from fault_injection import build_fault_manager


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism trade-off: set True for reproducibility, False for speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_spikerplus(spiker_root: Optional[str]) -> None:
    """
    Try importing spikerplus. If it fails, attempt to add a local checkout
    (e.g., extracted Spiker-public/spiker folder) to sys.path.
    """
    try:
        import spikerplus  # noqa: F401
        return
    except Exception:
        pass

    candidates = []
    if spiker_root:
        candidates.append(Path(spiker_root))
    # Common relative layouts:
    candidates.append(Path(__file__).resolve().parent / "Spiker-public" / "spiker")
    candidates.append(Path.cwd() / "Spiker-public" / "spiker")

    for c in candidates:
        if (c / "spikerplus").is_dir():
            sys.path.insert(0, str(c))
            # If the path was previously cached as "not found", remove it.
            sys.path_importer_cache.pop(str(c), None)
            try:
                import spikerplus  # noqa: F401
                return
            except Exception:
                continue

    raise ImportError(
        "Cannot import spikerplus.\n"
        "Fix one of the following:\n"
        "  (1) pip install spikerplus snntorch\n"
        "  (2) Provide --spiker_root pointing to the extracted 'Spiker-public/spiker' directory\n"
    )


def beta_from_shift(shift: int) -> float:
    """
    Spiker+ VHDL generator derives a right-shift from (1 - beta).
    For correct hardware mapping, set:
      beta = 1 - 2^(-shift)
    """
    if shift <= 0:
        raise ValueError("beta_shift must be >= 1")
    return float(1.0 - 2.0 ** (-shift))


def make_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
    ])

    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@dataclass
class FragCfg:
    n_cycles: int
    method: str = "combo"
    overlap: bool = True
    kernel_size: int = 15
    overlap_iter: int = 3
    per_image: bool = False

    # Optional: pass extra configuration to batch_dynamic_fragments
    # (kept as Any to remain forward-compatible with your code)
    importance_cfg: Optional[Dict[str, Any]] = None
    power_norm: Optional[Dict[str, Any]] = None


def images_to_spike_train(
    images_bchw: torch.Tensor,
    frag_cfg: FragCfg,
    gain: float,
    rng: Optional[torch.Generator] = None,
    weak_model: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    images_bchw: [B,1,28,28] in [0,1]
    returns spikes_tbn: [T,B,N] float tensor (0/1)
    """
    B, C, H, W = images_bchw.shape
    assert C == 1, f"Expected MNIST [B,1,H,W], got {images_bchw.shape}"
    T = frag_cfg.n_cycles

    # Fragment: [B,T,1,H,W]
    frags = batch_dynamic_fragments(
        images_bchw,
        num_steps=T,
        overlap=frag_cfg.overlap,
        method=frag_cfg.method,
        weak_model=weak_model,
        importance_cfg=frag_cfg.importance_cfg,
        kernel_size=frag_cfg.kernel_size,
        overlap_iter=frag_cfg.overlap_iter,
        per_image=frag_cfg.per_image,
        device=images_bchw.device,
        power_norm=frag_cfg.power_norm,
    )

    # Flatten -> [B,T,N]
    frags_flat = frags.reshape(B, T, -1)

    # Convert to spikes by Bernoulli sampling.
    # Note: if you need deterministic behavior, pass a seeded torch.Generator.
    prob = (frags_flat * gain).clamp_(0.0, 1.0)
    spikes_btn = torch.bernoulli(prob, generator=rng)

    # [T,B,N] for spikerplus SNN forward
    spikes_tbn = spikes_btn.permute(1, 0, 2).contiguous()
    return spikes_tbn


def get_logits_from_spikerplus_net(net: torch.nn.Module) -> torch.Tensor:
    """
    Spiker+ SNN forward stores per-layer recordings in OrderedDicts:
      net.mem_rec / net.spk_rec
    We use the last layer membrane potential (mean over time) as logits.
    returns logits [B, n_classes]
    """
    # last recorded mem: [T,B,C]
    _, out_rec = list(net.mem_rec.items())[-1]
    logits = torch.mean(out_rec, dim=0)  # [B,C]
    return logits


@torch.no_grad()
def save_one_sample_spike_txt(
    out_path: Path,
    spikes_tbn: torch.Tensor,
) -> None:
    """
    Save one sample's spike-train as text (one line per cycle).
    Each line is a binary string of length N (input vector).
    This is useful as stimulus for the generated VHDL testbenches.
    """
    # spikes_tbn: [T,B,N]
    spikes = spikes_tbn[:, 0, :].detach().cpu().to(torch.int32)  # [T,N]
    lines = []
    for t in range(spikes.shape[0]):
        bitstr = "".join("1" if v.item() != 0 else "0" for v in spikes[t])
        lines.append(bitstr)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_one_epoch(
    net: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    frag_cfg: FragCfg,
    gain: float,
    rng: torch.Generator,
    epoch: int,
    fault_mgr: Optional[Any],
    fault_start_epoch: int,
) -> Tuple[float, float]:
    net.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        spikes_tbn = images_to_spike_train(
            images_bchw=images,
            frag_cfg=frag_cfg,
            gain=gain,
            rng=rng,
            weak_model=None,  # set to net if you want model-aware fragmentation (slower)
        )

        optimizer.zero_grad(set_to_none=True)
        net(spikes_tbn)  # forward (records in net.mem_rec / net.spk_rec)
        logits = get_logits_from_spikerplus_net(net)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Fault injection after update (same pattern as your simple_snn.py)
        if fault_mgr is not None and epoch >= fault_start_epoch:
            with torch.no_grad():
                fault_mgr.apply_(net)

        total_loss += float(loss.detach().cpu().item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu().item())
        total_seen += int(labels.size(0))

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


@torch.no_grad()
def evaluate(
    net: torch.nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    frag_cfg: FragCfg,
    gain: float,
    rng: torch.Generator,
) -> Tuple[float, float]:
    net.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        spikes_tbn = images_to_spike_train(
            images_bchw=images,
            frag_cfg=frag_cfg,
            gain=gain,
            rng=rng,
            weak_model=None,
        )

        net(spikes_tbn)
        logits = get_logits_from_spikerplus_net(net)
        loss = loss_fn(logits, labels)

        total_loss += float(loss.detach().cpu().item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu().item())
        total_seen += int(labels.size(0))

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def build_spikerplus_net_dict(
    n_cycles: int,
    n_inputs: int,
    hidden: int,
    n_classes: int,
    beta_shift: int,
    threshold: float,
) -> Dict[str, Any]:
    beta = beta_from_shift(beta_shift)

    # Spiker+ expects a dict with keys:
    #  - n_cycles, n_inputs
    #  - layer_0, layer_1, ... each with neuron_model, n_neurons, beta, threshold, reset_mechanism, etc.
    net_dict: Dict[str, Any] = {
        "n_cycles": int(n_cycles),
        "n_inputs": int(n_inputs),
        "layer_0": {
            "neuron_model": "lif",
            "n_neurons": int(hidden),
            "beta": float(beta),
            "learn_beta": False,
            "threshold": float(threshold),
            "learn_threshold": False,
            "reset_mechanism": "subtract",
        },
        "layer_1": {
            "neuron_model": "lif",
            "n_neurons": int(n_classes),
            "beta": float(beta),
            "learn_beta": False,
            "threshold": float(threshold),
            "learn_threshold": False,
            # Readout layer: do not reset membrane (Spiker+ typical classification setup)
            "reset_mechanism": "none",
        },
    }
    return net_dict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./out_run")
    p.add_argument("--spiker_root", type=str, default=None, help="Path to Spiker-public/spiker (local checkout)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Network
    p.add_argument("--n_cycles", type=int, default=8, help="Time-steps (also VHDL n_cycles)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--beta_shift", type=int, default=4, help="beta = 1 - 2^(-beta_shift)")
    p.add_argument("--threshold", type=float, default=1.0)

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)

    # Fragmentation
    p.add_argument("--frag_method", type=str, default="combo",
                   help="batch_dynamic_fragments method: e.g., combo|sobel|laplacian|grad|...")
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--kernel_size", type=int, default=15)
    p.add_argument("--overlap_iter", type=int, default=3)
    p.add_argument("--gain", type=float, default=8.0, help="Poisson/Bernoulli gain (scales firing probability)")

    # Fault injection
    p.add_argument("--fault_ratio", type=float, default=0.0)
    p.add_argument("--fault_type", type=str, default="stuck", choices=["stuck", "random", "connectivity"])
    p.add_argument("--fault_start_epoch", type=int, default=0)
    p.add_argument("--fault_dist", type=str, default="uniform", choices=["uniform", "per_layer"])
    p.add_argument("--stuck_at", type=float, default=0.0)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--fault_limit", type=float, default=0.5)

    # VHDL export
    p.add_argument("--weights_bw", type=int, default=8)
    p.add_argument("--neurons_bw", type=int, default=12)
    p.add_argument("--fp_dec", type=int, default=6)
    p.add_argument("--functional_vhdl", type=lambda x: x.lower() == "true", default=False,
                   help="True: pure VHDL ROMs (simulation). False: Xilinx IP + .coe (synthesis)")
    p.add_argument("--interface", type=lambda x: x.lower() == "true", default=False,
                   help="True: generate FullAccelerator interface (addr+spike). False: raw Network with in_spikes vector.")
    p.add_argument("--vhdl_debug", type=lambda x: x.lower() == "true", default=False)

    args = p.parse_args()

    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import spikerplus (pip or local checkout)
    ensure_spikerplus(args.spiker_root)
    import spikerplus  # noqa: E402
    from spikerplus import NetBuilder, VhdlGenerator  # noqa: E402
    from spikerplus.vhdl import write_vhdl  # noqa: E402

    # Build network dict + network
    n_inputs = 28 * 28
    n_classes = 10
    net_dict = build_spikerplus_net_dict(
        n_cycles=args.n_cycles,
        n_inputs=n_inputs,
        hidden=args.hidden,
        n_classes=n_classes,
        beta_shift=args.beta_shift,
        threshold=args.threshold,
    )

    builder = NetBuilder(net_dict)
    net = builder.build().to(device)

    # Fault manager
    fault_mgr = None
    if args.fault_ratio and args.fault_ratio > 0.0:
        fault_mgr = build_fault_manager(
            net,
            ratio=args.fault_ratio,
            fault_type=args.fault_type,
            distribution=args.fault_dist,
            stuck_at=args.stuck_at,
            noise_std=args.noise_std,
            limit=args.fault_limit,
            include_bias=False,
            verbose=True,
        )

    # Data
    train_loader, test_loader = make_mnist_loaders(args.data_dir, args.batch_size, args.num_workers)

    # Fragmentation cfg
    frag_cfg = FragCfg(
        n_cycles=args.n_cycles,
        method=args.frag_method,
        overlap=args.overlap,
        kernel_size=args.kernel_size,
        overlap_iter=args.overlap_iter,
        per_image=False,
        importance_cfg=None,  # optionally pass custom cfg
        power_norm=None,      # optionally pass dict for power_normalize_frags (see your code)
    )

    # Training setup
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # RNG for Bernoulli sampling
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            net=net,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            frag_cfg=frag_cfg,
            gain=args.gain,
            rng=rng,
            epoch=epoch,
            fault_mgr=fault_mgr,
            fault_start_epoch=args.fault_start_epoch,
        )
        val_loss, val_acc = evaluate(
            net=net,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            frag_cfg=frag_cfg,
            gain=args.gain,
            rng=rng,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train loss={train_loss:.4f}, acc={train_acc*100:.2f}% | "
            f"val loss={val_loss:.4f}, acc={val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    # Restore best
    if best_state is not None:
        net.load_state_dict(best_state)

    # Save checkpoint + net_dict
    ckpt = {
        "net_dict": net_dict,
        "state_dict": net.state_dict(),
        "args": vars(args),
        "best_val_acc": float(best_val_acc),
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")
    (out_dir / "net_dict.json").write_text(json.dumps(net_dict, indent=2), encoding="utf-8")

    # Save one sample spike-train as a stimulus example
    images0, labels0 = next(iter(test_loader))
    images0 = images0.to(device)
    spikes0 = images_to_spike_train(images0[:1], frag_cfg, gain=args.gain, rng=rng, weak_model=None)
    save_one_sample_spike_txt(out_dir / "sample_in_spikes.txt", spikes0)

    # Export VHDL
    optim_config = {"weights_bw": args.weights_bw, "neurons_bw": args.neurons_bw, "fp_dec": args.fp_dec}
    vhdl_gen = VhdlGenerator(net, optim_config)
    vhdl_top = vhdl_gen.generate(functional=args.functional_vhdl, interface=args.interface, debug=args.vhdl_debug)

    vhdl_dir = out_dir / "vhdl"
    write_vhdl(vhdl_top, output_dir=str(vhdl_dir), rm=True)

    print(f"\nDone. Outputs written to: {out_dir.resolve()}")
    print(f"- Checkpoint: {out_dir / 'checkpoint.pt'}")
    print(f"- VHDL dir:   {vhdl_dir}")
    print(f"- Stimulus:   {out_dir / 'sample_in_spikes.txt'}")


if __name__ == "__main__":
    main()
