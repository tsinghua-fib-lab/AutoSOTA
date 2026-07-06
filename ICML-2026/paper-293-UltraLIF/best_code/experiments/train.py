# -*- coding: utf-8 -*-
"""
UltraLIF Training Script

Train any spiking neuron variant on static or neuromorphic datasets.

Usage:
    # Train temporal UltraLIF (paper: UltraLIF) on MNIST
    python experiments/train.py --model ultratlif --dataset mnist --epochs 100 --hidden 64 --track-spikes

    # Train spatial UltraDLIF (paper: UltraDLIF) on SHD
    python experiments/train.py --model ultradlif --dataset shd --epochs 100 --hidden 64 --timesteps 10

    # Train all single-layer FC models on CIFAR-10
    python experiments/train.py --model all --dataset cifar10 --epochs 100 --hidden 64

    # 2-layer deep with BatchNorm
    python experiments/train.py --model all-deep-bn --dataset mnist --hidden 64

Model naming (CLI key -> paper name):
    ultratlif   -> UltraLIF  (temporal, main paper model)
    ultratplif  -> UltraPLIF (temporal + learnable tau)
    ultradlif   -> UltraDLIF (spatial diffusion)
    ultradplif  -> UltraDPLIF(spatial + learnable tau)
    lif         -> LIF
    plif        -> PLIF
    dspike      -> DSpike (Li et al. NeurIPS 2021)
    dspike+     -> DSpike+ (DSpike + learnable tau)
    sigmalif    -> SigmaLIF (ablation baseline)
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# Suppress torch.compile errors (no C compiler in runtime)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralif.neurons.ultra import UltraLIF, UltraPLIF, UltraLIF_DS, UltraPLIF_DS
from ultralif.neurons.ultradlif import UltraDLIF, UltraDPLIF
from ultralif.neurons.lif import LIF, PLIF, AdaLIF, FullPLIF
from ultralif.neurons.baselines import DSpike, DSpikePlus, SigmaLIF
from ultralif.networks.fc import SNN, DeepSNN, TripleSNN
from ultralif.networks.conv import ConvSNN, DeepConvSNN
from ultralif.networks.resnet import SpikingResNet18
from ultralif.datasets.loader import get_dataset
from ultralif.datasets.utils import set_seed
from ultralif.training.trainer import train_model
from ultralif.training.logging import TeeLogger
from ultralif.training.metrics import count_spikes_epoch, compute_energy_proxy

try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


# =============================================================================
# MODEL REGISTRIES
# CLI key -> (display name, neuron class)
# =============================================================================

# Single-layer FC
NEURONS = {
    "lif":        ("LIF",       LIF),
    "plif":       ("PLIF",      PLIF),
    "adalif":     ("AdaLIF",    AdaLIF),
    "fullplif":   ("FullPLIF",  FullPLIF),
    "ultratlif":  ("UltraLIF",  UltraLIF),
    "ultratplif": ("UltraPLIF", UltraPLIF),
    "ultradlif":  ("UltraDLIF", UltraDLIF),
    "ultradplif": ("UltraDPLIF",UltraDPLIF),
    "dspike":     ("DSpike",    DSpike),
    "dspike+":    ("DSpike+",   DSpikePlus),
    "sigmalif":   ("SigmaLIF",  SigmaLIF),
}

# 2-layer FC
DEEP_NEURONS = {
    "ultratlif2":  ("UltraLIF-2L",  UltraLIF),
    "ultratplif2": ("UltraPLIF-2L", UltraPLIF),
    "ultradlif2":  ("UltraDLIF-2L", UltraDLIF),
    "ultradplif2": ("UltraDPLIF-2L",UltraDPLIF),
    "lif2":        ("LIF-2L",       LIF),
}

# 3-layer FC
TRIPLE_NEURONS = {
    "ultratlif3":  ("UltraLIF-3L",  UltraLIF),
    "ultratplif3": ("UltraPLIF-3L", UltraPLIF),
    "ultradlif3":  ("UltraDLIF-3L", UltraDLIF),
    "ultradplif3": ("UltraDPLIF-3L",UltraDPLIF),
    "lif3":        ("LIF-3L",       LIF),
}

# 2L + residual connection
DEEP_NEURONS_RES = {
    "ultratlif2_res":  ("UltraLIF-2L-Res",  UltraLIF),
    "ultratplif2_res": ("UltraPLIF-2L-Res", UltraPLIF),
    "ultradlif2_res":  ("UltraDLIF-2L-Res", UltraDLIF),
    "ultradplif2_res": ("UltraDPLIF-2L-Res",UltraDPLIF),
    "lif2_res":        ("LIF-2L-Res",       LIF),
}

# 2L + BatchNorm
DEEP_NEURONS_BN = {
    "ultratlif2_bn":  ("UltraLIF-2L-BN",  UltraLIF),
    "ultratplif2_bn": ("UltraPLIF-2L-BN", UltraPLIF),
    "ultradlif2_bn":  ("UltraDLIF-2L-BN", UltraDLIF),
    "ultradplif2_bn": ("UltraDPLIF-2L-BN",UltraDPLIF),
    "lif2_bn":        ("LIF-2L-BN",       LIF),
}

# 2L + BN + residual
DEEP_NEURONS_BN_RES = {
    "ultratlif2_bn_res":  ("UltraLIF-2L-BN-Res",  UltraLIF),
    "ultratplif2_bn_res": ("UltraPLIF-2L-BN-Res", UltraPLIF),
    "ultradlif2_bn_res":  ("UltraDLIF-2L-BN-Res", UltraDLIF),
    "ultradplif2_bn_res": ("UltraDPLIF-2L-BN-Res",UltraDPLIF),
    "lif2_bn_res":        ("LIF-2L-BN-Res",       LIF),
}

# 3L + residual
TRIPLE_NEURONS_RES = {
    "ultratlif3_res":  ("UltraLIF-3L-Res",  UltraLIF),
    "ultratplif3_res": ("UltraPLIF-3L-Res", UltraPLIF),
    "ultradlif3_res":  ("UltraDLIF-3L-Res", UltraDLIF),
    "ultradplif3_res": ("UltraDPLIF-3L-Res",UltraDPLIF),
    "lif3_res":        ("LIF-3L-Res",       LIF),
}

# 3L + BN
TRIPLE_NEURONS_BN = {
    "ultratlif3_bn":  ("UltraLIF-3L-BN",  UltraLIF),
    "ultratplif3_bn": ("UltraPLIF-3L-BN", UltraPLIF),
    "ultradlif3_bn":  ("UltraDLIF-3L-BN", UltraDLIF),
    "ultradplif3_bn": ("UltraDPLIF-3L-BN",UltraDPLIF),
    "lif3_bn":        ("LIF-3L-BN",       LIF),
}

# 3L + BN + residual
TRIPLE_NEURONS_BN_RES = {
    "ultratlif3_bn_res":  ("UltraLIF-3L-BN-Res",  UltraLIF),
    "ultratplif3_bn_res": ("UltraPLIF-3L-BN-Res", UltraPLIF),
    "ultradlif3_bn_res":  ("UltraDLIF-3L-BN-Res", UltraDLIF),
    "ultradplif3_bn_res": ("UltraDPLIF-3L-BN-Res",UltraDPLIF),
    "lif3_bn_res":        ("LIF-3L-BN-Res",       LIF),
}

# 2-layer Conv
CONV_NEURONS = {
    "ultratlif_conv":        ("UltraLIF-Conv",        UltraLIF),
    "ultratplif_conv":       ("UltraPLIF-Conv",        UltraPLIF),
    "ultradlif_conv":        ("UltraDLIF-Conv",        UltraDLIF),
    "ultradplif_conv":       ("UltraDPLIF-Conv",       UltraDPLIF),
    "lif_conv":              ("LIF-Conv",              LIF),
    # Stateless ablation (V reset each timestep)
    "ultratlif_conv_sl":     ("UltraLIF-Conv-SL",     UltraLIF),
    "ultratplif_conv_sl":    ("UltraPLIF-Conv-SL",    UltraPLIF),
    "lif_conv_sl":           ("LIF-Conv-SL",           LIF),
    # Disentangled spike scale
    "ultratlif_conv_ds":     ("UltraLIF-Conv-DS",     UltraLIF_DS),
    "ultratplif_conv_ds":    ("UltraPLIF-Conv-DS",    UltraPLIF_DS),
    # BatchNorm variants
    "lif_conv_bn":           ("LIF-Conv-BN",           LIF),
    "ultratlif_conv_bn":     ("UltraLIF-Conv-BN",     UltraLIF),
    "ultratplif_conv_bn":    ("UltraPLIF-Conv-BN",    UltraPLIF),
    "ultradlif_conv_bn":     ("UltraDLIF-Conv-BN",    UltraDLIF),
    "ultradplif_conv_bn":    ("UltraDPLIF-Conv-BN",   UltraDPLIF),
    "ultratlif_conv_bn_ds":  ("UltraLIF-Conv-BN-DS",  UltraLIF_DS),
    "ultratplif_conv_bn_ds": ("UltraPLIF-Conv-BN-DS", UltraPLIF_DS),
}

# 4-layer Deep Conv
DEEP_CONV_NEURONS = {
    "lif_conv4":              ("LIF-Conv4",              LIF),
    "ultratlif_conv4":        ("UltraLIF-Conv4",        UltraLIF),
    "ultratplif_conv4":       ("UltraPLIF-Conv4",       UltraPLIF),
    "lif_conv4_bn":           ("LIF-Conv4-BN",           LIF),
    "ultratlif_conv4_bn":     ("UltraLIF-Conv4-BN",     UltraLIF),
    "ultratplif_conv4_bn":    ("UltraPLIF-Conv4-BN",    UltraPLIF),
    "ultratlif_conv4_bn_ds":  ("UltraLIF-Conv4-BN-DS",  UltraLIF_DS),
    "ultratplif_conv4_bn_ds": ("UltraPLIF-Conv4-BN-DS", UltraPLIF_DS),
    "ultradlif_conv4":        ("UltraDLIF-Conv4",       UltraDLIF),
    "ultradplif_conv4":       ("UltraDPLIF-Conv4",      UltraDPLIF),
    "ultradlif_conv4_bn":     ("UltraDLIF-Conv4-BN",    UltraDLIF),
    "ultradplif_conv4_bn":    ("UltraDPLIF-Conv4-BN",   UltraDPLIF),
}

# Fully spiking ResNet18
SPIKING_RESNET_NEURONS = {
    "lif_sresnet18":            ("LIF-SResNet18",            LIF),
    "ultratlif_sresnet18":      ("UltraLIF-SResNet18",      UltraLIF),
    "ultratplif_sresnet18":     ("UltraPLIF-SResNet18",     UltraPLIF),
    "ultratlif_sresnet18_ds":   ("UltraLIF-SResNet18-DS",   UltraLIF_DS),
    "ultratplif_sresnet18_ds":  ("UltraPLIF-SResNet18-DS",  UltraPLIF_DS),
}

# Keys that support sparsity penalty (models with learnable eps)
_ULTRA_KEYS = set(
    list(k for k in NEURONS if k.startswith(("ultra",)))
    + list(DEEP_NEURONS) + list(TRIPLE_NEURONS)
    + list(DEEP_NEURONS_RES) + list(DEEP_NEURONS_BN) + list(DEEP_NEURONS_BN_RES)
    + list(TRIPLE_NEURONS_RES) + list(TRIPLE_NEURONS_BN) + list(TRIPLE_NEURONS_BN_RES)
    + list(k for k in CONV_NEURONS if k.startswith(("ultra",)))
    + list(k for k in DEEP_CONV_NEURONS if k.startswith(("ultra",)))
    + list(k for k in SPIKING_RESNET_NEURONS if k.startswith(("ultra",)))
)


def main():
    all_registries = [
        NEURONS, DEEP_NEURONS, TRIPLE_NEURONS, CONV_NEURONS, DEEP_CONV_NEURONS,
        SPIKING_RESNET_NEURONS, DEEP_NEURONS_RES, DEEP_NEURONS_BN, DEEP_NEURONS_BN_RES,
        TRIPLE_NEURONS_RES, TRIPLE_NEURONS_BN, TRIPLE_NEURONS_BN_RES,
    ]
    all_model_keys = [k for reg in all_registries for k in reg]

    parser = argparse.ArgumentParser(description="UltraLIF Training")
    parser.add_argument(
        "--model", default="ultratlif",
        choices=[
            "all", "all-deep", "all-triple", "all-conv", "all-deep-conv", "all-sresnet18",
            "all-deep-res", "all-deep-bn", "all-deep-bn-res",
            "all-triple-res", "all-triple-bn", "all-triple-bn-res",
        ] + all_model_keys,
    )
    parser.add_argument("--dataset", default="mnist",
                        choices=["mnist", "fashion", "cifar10", "nmnist", "dvs_gesture",
                                 "cifar10_dvs", "shd", "ssc"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--tpu", action="store_true", help="Use TPU (requires torch_xla)")
    parser.add_argument("--no-save", action="store_true", help="Skip checkpoint saving")
    parser.add_argument("--track-spikes", action="store_true", help="Track spike rates every 10 epochs")
    parser.add_argument("--sparsity-lambda", type=float, default=0.0,
                        help="Sparsity penalty on spike rate (0 = disabled)")
    parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    args.dtype_torch = dtype_map[args.dtype]

    base_dir = Path(__file__).parent.parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tee_logger = None
    if not args.no_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"{args.dataset}_{args.model}_seed{args.seed}_{timestamp}.log"
        tee_logger = TeeLogger(log_path)
        sys.stdout = tee_logger
        print(f"Logging to: {log_path}")

    if args.tpu:
        if not TPU_AVAILABLE:
            print("ERROR: torch_xla not installed. Run: pip install torch_xla")
            sys.exit(1)
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print(f"UltraLIF Training — {args.dataset.upper()}")
    print("=" * 70)
    print(f"Device: {device}")

    ckpt_dir = base_dir / "checkpoints" / args.dataset
    results_dir = base_dir / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, in_dim, n_classes = get_dataset(
        args.dataset, args.batch_size, args.timesteps
    )
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print(f"Config: hidden={args.hidden}, T={args.timesteps}, epochs={args.epochs}, lr={args.lr}")
    if args.track_spikes:
        print("Spike tracking: ENABLED (every 10 epochs)")

    # Resolve model group
    group_map = {
        "all": NEURONS, "all-deep": DEEP_NEURONS, "all-triple": TRIPLE_NEURONS,
        "all-conv": CONV_NEURONS, "all-deep-conv": DEEP_CONV_NEURONS,
        "all-sresnet18": SPIKING_RESNET_NEURONS,
        "all-deep-res": DEEP_NEURONS_RES, "all-deep-bn": DEEP_NEURONS_BN,
        "all-deep-bn-res": DEEP_NEURONS_BN_RES, "all-triple-res": TRIPLE_NEURONS_RES,
        "all-triple-bn": TRIPLE_NEURONS_BN, "all-triple-bn-res": TRIPLE_NEURONS_BN_RES,
    }
    if args.model in group_map:
        models_to_run = list(group_map[args.model].keys())
    else:
        models_to_run = [args.model]

    is_neuromorphic = args.dataset in ["nmnist", "dvs_gesture", "cifar10_dvs", "shd", "ssc"]
    if args.dataset in ["mnist", "fashion"]:
        input_size, in_channels = 28, 1
    elif args.dataset == "cifar10":
        input_size, in_channels = 32, 3
    else:
        input_size, in_channels = 32, 2

    results = {}

    for key in models_to_run:
        # Determine registry and flags
        is_deep        = key in DEEP_NEURONS
        is_deep_res    = key in DEEP_NEURONS_RES
        is_deep_bn     = key in DEEP_NEURONS_BN
        is_deep_bn_res = key in DEEP_NEURONS_BN_RES
        is_triple      = key in TRIPLE_NEURONS
        is_triple_res  = key in TRIPLE_NEURONS_RES
        is_triple_bn   = key in TRIPLE_NEURONS_BN
        is_triple_bn_res = key in TRIPLE_NEURONS_BN_RES
        is_conv        = key in CONV_NEURONS
        is_deep_conv   = key in DEEP_CONV_NEURONS
        is_sresnet18   = key in SPIKING_RESNET_NEURONS

        if is_sresnet18:
            display, neuron_cls = SPIKING_RESNET_NEURONS[key]
        elif is_deep_conv:
            display, neuron_cls = DEEP_CONV_NEURONS[key]
        elif is_conv:
            display, neuron_cls = CONV_NEURONS[key]
        elif is_triple_bn_res:
            display, neuron_cls = TRIPLE_NEURONS_BN_RES[key]
        elif is_triple_bn:
            display, neuron_cls = TRIPLE_NEURONS_BN[key]
        elif is_triple_res:
            display, neuron_cls = TRIPLE_NEURONS_RES[key]
        elif is_triple:
            display, neuron_cls = TRIPLE_NEURONS[key]
        elif is_deep_bn_res:
            display, neuron_cls = DEEP_NEURONS_BN_RES[key]
        elif is_deep_bn:
            display, neuron_cls = DEEP_NEURONS_BN[key]
        elif is_deep_res:
            display, neuron_cls = DEEP_NEURONS_RES[key]
        elif is_deep:
            display, neuron_cls = DEEP_NEURONS[key]
        else:
            display, neuron_cls = NEURONS[key]

        print(f"\n--- {display} ---")
        set_seed(args.seed)

        if is_sresnet18:
            model = SpikingResNet18(neuron_cls, in_channels, n_classes, args.timesteps, input_size)
            neuron = model.stem_neuron
        elif is_deep_conv:
            use_bn = "_bn" in key
            model = DeepConvSNN(neuron_cls, in_channels, n_classes, args.timesteps, input_size, use_bn=use_bn)
            neuron = model.neuron1
        elif is_conv:
            stateless = key.endswith("_sl")
            use_bn = "_bn" in key
            model = ConvSNN(neuron_cls, in_channels, n_classes, args.timesteps, input_size,
                            stateless=stateless, use_bn=use_bn)
            neuron = model.neuron1
        elif is_triple_bn_res or is_triple_bn or is_triple_res or is_triple:
            use_res = is_triple_res or is_triple_bn_res
            use_bn  = is_triple_bn or is_triple_bn_res
            neuron1 = neuron_cls(args.hidden)
            neuron2 = neuron_cls(args.hidden)
            neuron3 = neuron_cls(args.hidden)
            model = TripleSNN(neuron1, neuron2, neuron3, in_dim, args.hidden, n_classes,
                              args.timesteps, neuromorphic=is_neuromorphic, use_res=use_res, use_bn=use_bn)
            neuron = neuron1
        elif is_deep_bn_res or is_deep_bn or is_deep_res or is_deep:
            use_res = is_deep_res or is_deep_bn_res
            use_bn  = is_deep_bn or is_deep_bn_res
            neuron1 = neuron_cls(args.hidden)
            neuron2 = neuron_cls(args.hidden)
            model = DeepSNN(neuron1, neuron2, in_dim, args.hidden, n_classes,
                            args.timesteps, neuromorphic=is_neuromorphic, use_res=use_res, use_bn=use_bn)
            neuron = neuron1
        else:
            neuron = neuron_cls(args.hidden)
            model = SNN(neuron, in_dim, args.hidden, n_classes, args.timesteps,
                        neuromorphic=is_neuromorphic)

        if not args.tpu and torch.cuda.is_available() and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("  torch.compile: enabled")
            except Exception as e:
                print(f"  torch.compile: disabled ({e})")

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {params:,}")

        use_sparsity = args.sparsity_lambda if key in _ULTRA_KEYS else 0.0
        if use_sparsity > 0:
            print(f"  Sparsity penalty: {use_sparsity}")

        sparse_suffix = f"_sp{use_sparsity}" if use_sparsity > 0 else ""
        save_path = (
            None if args.no_save
            else ckpt_dir / f"{key}_T{args.timesteps}{sparse_suffix}_seed{args.seed}.pt"
        )

        acc, history, final_info = train_model(
            model, train_loader, test_loader, args.epochs, args.lr, device,
            verbose=not args.quiet, use_tpu=args.tpu, save_path=save_path,
            track_spikes=args.track_spikes, neuromorphic=is_neuromorphic,
            timesteps=args.timesteps, sparsity_lambda=use_sparsity,
            dtype=args.dtype_torch,
        )

        # Collect learned params
        learned = {}
        check_neurons = [neuron1, neuron2] if (is_deep and not is_sresnet18 and not is_conv) else [neuron]
        for n in check_neurons:
            for attr in ("eps", "k", "spike_scale"):
                if hasattr(n, attr):
                    v = getattr(n, attr)
                    learned[attr] = v.item()
                    print(f"  Learned {attr}: {learned[attr]:.3f}")
            if hasattr(n, "tau") and isinstance(n.tau, torch.Tensor):
                learned["tau"] = n.tau.item()
                print(f"  Learned tau: {learned['tau']:.3f}")

        if args.track_spikes:
            final_sr, _ = count_spikes_epoch(
                model, test_loader, device, args.timesteps, is_neuromorphic, args.dtype_torch
            )
            learned["final_spike_rate"] = final_sr
            learned["energy_proxy"] = compute_energy_proxy(final_sr, args.hidden, args.timesteps)
            print(f"  Final spike rate: {final_sr:.4f}")
            print(f"  Energy proxy: {learned['energy_proxy']:.3f}x baseline")

        print(f"  Best: {acc:.2%}")

        results[key] = {
            "name": display, "acc": acc, "params": params,
            "learned": learned, "history": history,
            **{k: final_info[k] for k in ["eps_history", "k_history", "tau_history",
                                           "spike_scale_history", "spike_rate_history"]},
        }

    # Summary table
    print("\n" + "=" * 70 + "\nRESULTS\n" + "=" * 70)
    print(f"{'Model':15} {'Acc':>8} {'Params':>10} {'eps/k':>8} {'Spk Rate':>10} {'Energy':>8}")
    print("-" * 65)
    for i, (k, r) in enumerate(sorted(results.items(), key=lambda x: -x[1]["acc"])):
        marker = "*" if i == 0 else " "
        eps_k = r["learned"].get("eps", r["learned"].get("k", "-"))
        eps_k_str = f"{eps_k:.2f}" if isinstance(eps_k, float) else "-"
        spk = r["learned"].get("final_spike_rate")
        energy = r["learned"].get("energy_proxy")
        print(f"{marker} {r['name']:14} {r['acc']:>7.2%} {r['params']:>10,} "
              f"{eps_k_str:>8} {(spk or 0):>10.4f} {(energy or 0):>8.2f}x")

    results_file = results_dir / (
        f"{args.dataset}_{args.model}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        safe_args = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                     for k, v in vars(args).items()}
        json.dump({"args": safe_args, "results": results}, f, indent=2)
    print(f"\nResults saved: {results_file}")

    if tee_logger:
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
