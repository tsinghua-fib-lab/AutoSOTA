from __future__ import annotations

"""One-touch ImageNet train / eval script for SEW-ResNet with optional learnable fragmentation.

Features
--------
- ImageNet-style training and validation from ``ImageFolder`` directories.
- Optional SEW-ResNet + learnable fragmentation integration using the package created earlier.
- Optional AMP via ``torch.amp.autocast`` and ``torch.amp.GradScaler``.
- Optional DDP via ``torchrun`` + ``DistributedDataParallel``.
- Checkpoint save / resume / best-checkpoint evaluation.

Dataset layout
--------------
Expected by default:
    <data_root>/train/<class_name>/*.JPEG
    <data_root>/val/<class_name>/*.JPEG

Typical usage
-------------
Single GPU training + validation:
    python imagenet_train_testing.py \
        --data-root /path/to/imagenet \
        --output-dir ./runs/sew_r34_imagenet \
        --depth 34 \
        --mode train_eval

Multi-GPU DDP training + validation:
    torchrun --standalone --nproc_per_node=8 imagenet_train_testing.py \
        --data-root /path/to/imagenet \
        --output-dir ./runs/sew_r34_imagenet_ddp \
        --depth 34 \
        --amp \
        --sync-bn \
        --mode train_eval

Dynamic fragmentation:
    torchrun --standalone --nproc_per_node=8 imagenet_train_testing.py \
        --data-root /path/to/imagenet \
        --output-dir ./runs/sew_r34_imagenet_frag \
        --depth 34 \
        --fragmentation dynamic \
        --dynamic-candidates 2 4 8 \
        --frag-balance-weight 0.01 \
        --amp \
        --sync-bn \
        --mode train_eval
"""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sew_resnet_fragmentation_wrapper import (  # noqa: E402
    FragmentedSEWOutput,
    build_fragmented_sew_resnet,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _import_torchvision() -> tuple[Any, Any, Any]:
    try:
        from torchvision import datasets, transforms
        from torchvision.transforms import InterpolationMode
    except Exception as e:  # pragma: no cover - depends on local torchvision build
        raise ImportError(
            "torchvision could not be imported. Please install a matching torch/torchvision build "
            "before running ImageNet training or evaluation."
        ) from e
    return datasets, transforms, InterpolationMode


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":.4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: Sequence[AverageMeter], prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = list(meters)
        self.prefix = prefix

    def display(self, batch: int) -> str:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return "\t".join(entries)

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = "{" + f":{num_digits}d" + "}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet train/eval for SEW-ResNet + optional fragmentation")

    # I/O
    parser.add_argument("--data-root", type=str, required=True, help="ImageNet root containing train/ and val/")
    parser.add_argument("--train-dir", type=str, default="", help="Optional explicit train directory")
    parser.add_argument("--val-dir", type=str, default="", help="Optional explicit val directory")
    parser.add_argument("--output-dir", type=str, default="./runs/imagenet_sew")
    parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    parser.add_argument("--eval-checkpoint", type=str, default="", help="Checkpoint to evaluate in --mode eval")
    parser.add_argument("--save-freq", type=int, default=1, help="Epoch frequency for last-epoch checkpoints")
    parser.add_argument("--print-freq", type=int, default=50)

    # Model
    parser.add_argument("--depth", type=int, default=34, choices=[18, 34, 50])
    parser.add_argument("--cnf", type=str, default="ADD", choices=["ADD", "AND", "IAND"])
    parser.add_argument("--neuron-name", type=str, default="if", choices=["if", "lif"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--crop-pct", type=float, default=224.0 / 256.0)
    parser.add_argument("--plain-time-steps", type=int, default=4, help="Time steps when fragmentation is off")
    parser.add_argument("--decoder", type=str, default="entropy", choices=["entropy", "mean"])
    parser.add_argument("--entropy-gamma", type=float, default=1.0)
    parser.add_argument("--zero-init-residual", action="store_true", default=True)
    parser.add_argument("--no-zero-init-residual", action="store_false", dest="zero_init_residual")
    parser.add_argument("--ann-warmup", action="store_true", help="Load torchvision ImageNet ResNet weights into the SEW backbone")
    parser.add_argument("--use-expected-poisson", action="store_true", help="Use expected-rate Poisson encoding inside the wrapper")

    # Fragmentation
    parser.add_argument("--fragmentation", type=str, default="off", choices=["off", "fixed", "dynamic"])
    parser.add_argument("--fixed-num-fragments", type=int, default=4)
    parser.add_argument("--dynamic-candidates", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--init-direction", type=str, default="horizontal", choices=["horizontal", "vertical", "diag_lr", "diag_rl"])
    parser.add_argument("--mask-scale", type=float, default=1.0)
    parser.add_argument("--frag-balance-weight", type=float, default=0.01)

    # Optimization
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-process batch size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)

    # Precision / parallelism
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--sync-bn", action="store_true", help="Convert BatchNorm to SyncBatchNorm before DDP wrapping")
    parser.add_argument("--channels-last", action="store_true")

    # Misc
    parser.add_argument("--normalize", type=str, default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-cudnn-benchmark", action="store_true")

    return parser.parse_args()


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args: argparse.Namespace) -> Tuple[torch.device, bool, int]:
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        dist.barrier()
        return device, True, local_rank

    if torch.cuda.is_available():
        return torch.device("cuda"), False, 0
    return torch.device("cpu"), False, 0


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0) -> None:
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)


@torch.no_grad()
def reduce_float(value: float, device: torch.device, average: bool = True) -> float:
    if not is_dist_avail_and_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= float(get_world_size())
    return float(tensor.item())


@torch.no_grad()
def reduce_meter(meter: AverageMeter, device: torch.device) -> None:
    if not is_dist_avail_and_initialized():
        return
    stats = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    meter.sum = float(stats[0].item())
    meter.count = int(stats[1].item())
    meter.avg = meter.sum / max(meter.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Sequence[torch.Tensor]:
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        k_eff = min(k, output.size(1))
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_transforms(args: argparse.Namespace) -> Tuple[Any, Any]:
    _, transforms, InterpolationMode = _import_torchvision()
    resize_size = int(math.floor(args.input_size / args.crop_pct))

    train_tf = [
        transforms.RandomResizedCrop(args.input_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    val_tf = [
        transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
    ]

    if args.normalize == "imagenet":
        train_tf.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        val_tf.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(train_tf), transforms.Compose(val_tf)


def build_dataloaders(args: argparse.Namespace, ddp: bool) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler], int]:
    train_root = Path(args.train_dir) if args.train_dir else Path(args.data_root) / "train"
    val_root = Path(args.val_dir) if args.val_dir else Path(args.data_root) / "val"

    if not train_root.exists():
        raise FileNotFoundError(f"train directory not found: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"val directory not found: {val_root}")

    datasets, _, _ = _import_torchvision()
    train_tf, val_tf = build_transforms(args)
    train_dataset = datasets.ImageFolder(root=str(train_root), transform=train_tf)
    val_dataset = datasets.ImageFolder(root=str(val_root), transform=val_tf)

    num_classes = len(train_dataset.classes)
    if len(val_dataset.classes) != num_classes:
        raise RuntimeError("train/val class counts differ. Use matching ImageFolder layouts.")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler, val_sampler, num_classes


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    fragmentation = args.fragmentation
    fixed_num_fragments: Optional[int] = None
    dynamic_candidates: Optional[Sequence[int]] = None
    if fragmentation == "fixed":
        fixed_num_fragments = int(args.fixed_num_fragments)
    elif fragmentation == "dynamic":
        dynamic_candidates = tuple(int(x) for x in args.dynamic_candidates)

    model = build_fragmented_sew_resnet(
        depth=args.depth,
        num_classes=num_classes,
        image_size=(args.input_size, args.input_size),
        in_channels=3,
        stem="imagenet",
        cnf=args.cnf,
        neuron_name=args.neuron_name,
        zero_init_residual=args.zero_init_residual,
        use_expected_poisson=args.use_expected_poisson,
        fixed_num_fragments=fixed_num_fragments,
        dynamic_candidates=dynamic_candidates,
        init_direction=args.init_direction,
        mask_scale=args.mask_scale,
        decoder=args.decoder,
        entropy_gamma=args.entropy_gamma,
    )
    return model


def maybe_load_ann_warmup(model: nn.Module) -> None:
    base = unwrap_model(model)
    backbone = getattr(base, "backbone", None)
    if backbone is None:
        raise RuntimeError("Expected wrapper model with a `.backbone` attribute.")
    loader = getattr(backbone, "load_from_torchvision_resnet", None)
    if loader is None or not callable(loader):
        raise RuntimeError("Backbone does not expose load_from_torchvision_resnet().")
    loader(strict=False)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


def load_checkpoint_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    base = unwrap_model(model)
    try:
        base.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    if any(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        base.load_state_dict(stripped, strict=True)
        return

    prefixed = {f"module.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(prefixed, strict=True)


def save_checkpoint(
    args: argparse.Namespace,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    best_acc1: float,
    is_best: bool,
) -> None:
    if not is_main_process():
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_acc1": float(best_acc1),
        "args": vars(args),
    }

    last_path = out_dir / "checkpoint_last.pth"
    torch.save(payload, last_path)

    if epoch % max(int(args.save_freq), 1) == 0:
        torch.save(payload, out_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    if is_best:
        torch.save(payload, out_dir / "checkpoint_best.pth")


def load_training_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> Tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location=device)
    load_checkpoint_flexible(model, ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_acc1 = float(ckpt.get("best_acc1", 0.0))
    return start_epoch, best_acc1


def load_eval_checkpoint(ckpt_path: str, model: nn.Module, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_checkpoint_flexible(model, state)


def resolve_amp_dtype(args: argparse.Namespace, device: torch.device) -> Tuple[bool, Optional[torch.dtype], Optional[torch.amp.GradScaler]]:
    if not args.amp or device.type != "cuda":
        return False, None, None

    if args.amp_dtype == "bf16":
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not bf16_ok:
            if is_main_process():
                print("[WARN] bf16 AMP requested, but this CUDA device does not report bf16 support. Falling back to fp16.")
            dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            dtype = torch.bfloat16
            scaler = None
    else:
        dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    return True, dtype, scaler


def maybe_to_channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and x.dim() == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def compute_total_loss(args: argparse.Namespace, out: FragmentedSEWOutput, criterion: nn.Module, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    main_loss = criterion(out.logits, target)
    balance_loss = out.fragmentation.balance_loss
    total_loss = main_loss + float(args.frag_balance_weight) * balance_loss
    return total_loss, main_loss, balance_loss


def train_one_epoch(
    args: argparse.Namespace,
    epoch: int,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    train_loader: DataLoader,
    sampler: Optional[DistributedSampler],
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    batch_time = AverageMeter("Time", ":.3f")
    data_time = AverageMeter("Data", ":.3f")
    losses = AverageMeter("Loss", ":.4e")
    ce_losses = AverageMeter("CE", ":.4e")
    bal_losses = AverageMeter("Bal", ":.4e")
    top1 = AverageMeter("Acc@1", ":.2f")
    top5 = AverageMeter("Acc@5", ":.2f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, ce_losses, bal_losses, top1, top5], prefix=f"Epoch: [{epoch}]")

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        images = maybe_to_channels_last(images, args.channels_last)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
            if amp_enabled and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            out = model(images, return_aux=True, plain_num_steps=args.plain_time_steps, sample_selector=True)
            loss, ce_loss, bal_loss = compute_total_loss(args, out, criterion, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        acc1, acc5 = accuracy(out.logits.detach(), target, topk=(1, 5))
        bsz = images.size(0)
        losses.update(float(loss.detach().item()), bsz)
        ce_losses.update(float(ce_loss.detach().item()), bsz)
        bal_losses.update(float(bal_loss.detach().item()), bsz)
        top1.update(float(acc1.item()), bsz)
        top5.update(float(acc5.item()), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (i % args.print_freq == 0 or i == len(train_loader) - 1):
            extra = ""
            if out.fragmentation.selector_probs is not None:
                selector = [round(float(x), 4) for x in out.fragmentation.selector_probs.detach().cpu().tolist()]
                extra = f"\tT_sel={out.fragmentation.selected_t}\tselector={selector}"
            print(progress.display(i) + extra)

    for meter in (losses, ce_losses, bal_losses, top1, top5):
        reduce_meter(meter, device)

    return {
        "loss": losses.avg,
        "ce_loss": ce_losses.avg,
        "bal_loss": bal_losses.avg,
        "acc1": top1.avg,
        "acc5": top5.avg,
    }


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    model: nn.Module,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    checkpoint_path: str = "",
) -> Dict[str, float]:
    model.eval()

    if checkpoint_path:
        load_eval_checkpoint(checkpoint_path, model, device)
        model.eval()

    batch_time = AverageMeter("Time", ":.3f")
    losses = AverageMeter("Loss", ":.4e")
    ce_losses = AverageMeter("CE", ":.4e")
    bal_losses = AverageMeter("Bal", ":.4e")
    top1 = AverageMeter("Acc@1", ":.2f")
    top5 = AverageMeter("Acc@5", ":.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, ce_losses, bal_losses, top1, top5], prefix="Test: ")

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        images = maybe_to_channels_last(images, args.channels_last)

        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
            if amp_enabled and device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            out = model(images, return_aux=True, plain_num_steps=args.plain_time_steps, sample_selector=False)
            loss, ce_loss, bal_loss = compute_total_loss(args, out, criterion, target)

        acc1, acc5 = accuracy(out.logits, target, topk=(1, 5))
        bsz = images.size(0)
        losses.update(float(loss.item()), bsz)
        ce_losses.update(float(ce_loss.item()), bsz)
        bal_losses.update(float(bal_loss.item()), bsz)
        top1.update(float(acc1.item()), bsz)
        top5.update(float(acc5.item()), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (i % args.print_freq == 0 or i == len(val_loader) - 1):
            extra = ""
            if out.fragmentation.selector_probs is not None:
                selector = [round(float(x), 4) for x in out.fragmentation.selector_probs.detach().cpu().tolist()]
                extra = f"\tT_sel={out.fragmentation.selected_t}\tselector={selector}"
            print(progress.display(i) + extra)

    for meter in (losses, ce_losses, bal_losses, top1, top5):
        reduce_meter(meter, device)

    if is_main_process():
        print(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.5f}")

    return {
        "loss": losses.avg,
        "ce_loss": ce_losses.avg,
        "bal_loss": bal_losses.avg,
        "acc1": top1.avg,
        "acc5": top5.avg,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, ddp, local_rank = setup_distributed(args)
    rank = get_rank()
    set_seed(args.seed, rank=rank)

    if device.type == "cuda" and not args.disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if is_main_process():
        save_json(output_dir / "run_config.json", vars(args))

    train_loader, val_loader, train_sampler, _, num_classes = build_dataloaders(args, ddp=ddp)

    model = build_model(args, num_classes=num_classes)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if args.ann_warmup:
        maybe_load_ann_warmup(model)
        if is_main_process():
            print("[INFO] Loaded torchvision ANN warm-start weights into the SEW backbone.")

    if ddp and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if ddp:
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing)).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    amp_enabled, amp_dtype, scaler = resolve_amp_dtype(args, device)

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        start_epoch, best_acc1 = load_training_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        if is_main_process():
            print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch} (best_acc1={best_acc1:.3f}).")

    if args.mode == "eval":
        ckpt = args.eval_checkpoint or str(output_dir / "checkpoint_best.pth")
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Evaluation checkpoint not found: {ckpt}")
        metrics = evaluate(args, model, criterion, val_loader, device, amp_enabled, amp_dtype, checkpoint_path=ckpt)
        if is_main_process():
            save_json(output_dir / "eval_metrics.json", metrics)
        cleanup_distributed()
        return

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            args=args,
            epoch=epoch,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            train_loader=train_loader,
            sampler=train_sampler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        val_metrics = evaluate(
            args=args,
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        scheduler.step()

        acc1 = float(val_metrics["acc1"])
        is_best = acc1 > best_acc1
        best_acc1 = max(best_acc1, acc1)
        save_checkpoint(args, epoch, model, optimizer, scheduler, scaler, best_acc1, is_best=is_best)

        if is_main_process():
            lr_now = optimizer.param_groups[0]["lr"]
            summary = {
                "epoch": epoch,
                "lr": lr_now,
                "train": train_metrics,
                "val": val_metrics,
                "best_acc1": best_acc1,
            }
            print(json.dumps(summary, ensure_ascii=False))
            save_json(output_dir / "last_metrics.json", summary)

    if args.mode == "train_eval":
        best_ckpt = str(output_dir / "checkpoint_best.pth")
        if Path(best_ckpt).exists():
            if is_main_process():
                print(f"[INFO] Final evaluation using best checkpoint: {best_ckpt}")
            final_metrics = evaluate(args, model, criterion, val_loader, device, amp_enabled, amp_dtype, checkpoint_path=best_ckpt)
            if is_main_process():
                save_json(output_dir / "best_eval_metrics.json", final_metrics)

    cleanup_distributed()


if __name__ == "__main__":
    main()
