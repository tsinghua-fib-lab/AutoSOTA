from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Work around environments where torchvision tries to register a fake NMS op
# even when the op is not compiled into the local build.
try:
    _tv_lib = torch.library.Library("torchvision", "DEF")
    _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
except Exception:
    pass

from torchvision import datasets, transforms

# Support both:
#   python imagenet_train_testing.py ...
# and
#   python -m spikformer_fragmentation_addon.imagenet_train_testing ...
if __package__ is None or __package__ == "":
    import sys
    _ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(_ROOT))
    from spikformer_fragmentation_addon import (
        build_spikformer_preset,
        FixedLearnableFragmenter,
        DynamicLearnableFragmenter,
        FragmentedSpikformer,
    )
else:
    from . import (
        build_spikformer_preset,
        FixedLearnableFragmenter,
        DynamicLearnableFragmenter,
        FragmentedSpikformer,
    )


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-touch ImageNet train/test for Spikformer with optional learnable fragmentation"
    )
    parser.add_argument("--data-root", type=str, required=True, help="ImageNet root with train/ and val/ subdirectories")
    parser.add_argument("--output-dir", type=str, default="./outputs_imagenet_spikformer")
    parser.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"])
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume training from")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path for test mode")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--model-preset", type=str, default="spikformer-8-512")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-process train batch size")
    parser.add_argument("--val-batch-size", type=int, default=128, help="Per-process val batch size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--fake-data", action="store_true", help="Use torchvision FakeData instead of ImageFolder")
    parser.add_argument("--fake-train-size", type=int, default=2048)
    parser.add_argument("--fake-val-size", type=int, default=512)

    parser.add_argument("--amp", action="store_true", help="Enable AMP")
    parser.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--ddp", action="store_true", help="Enable DDP when launched by torchrun")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)

    parser.add_argument("--spike-backend", type=str, default="auto", choices=["native", "auto", "spikingjelly"])
    parser.add_argument("--spike-compute-backend", type=str, default="torch", choices=["torch", "cupy", "triton"])
    parser.add_argument("--time-steps", type=int, default=0, help="Override preset time steps when fragmentation is off")

    parser.add_argument("--use-fragmentation", action="store_true")
    parser.add_argument("--fragmentation-mode", type=str, default="dynamic", choices=["fixed", "dynamic"])
    parser.add_argument("--fragment-steps", type=int, default=4)
    parser.add_argument("--fragment-candidates", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--fragment-selector-init", type=int, default=4)
    parser.add_argument("--fragment-sharpness", type=float, default=1.0)
    parser.add_argument("--fragment-gumbel-tau", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--decode", type=str, default="entropy", choices=["mean", "entropy"])
    parser.add_argument("--gamma", type=float, default=1.0)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    distributed = bool(args.ddp) and is_distributed_env()
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl" if args.backend == "nccl" else "gloo"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        dist.barrier()
        return True, rank, world_size, local_rank, device

    rank = 0
    world_size = 1
    local_rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, rank, world_size, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def maybe_reset_spiking_state(model: nn.Module) -> None:
    # Safe for native implementation and required for stateful SpikingJelly modules.
    try:
        from spikingjelly.activation_based import functional as sj_functional
        sj_functional.reset_net(model)
        return
    except Exception:
        pass

    for module in model.modules():
        reset = getattr(module, "reset", None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass


class AverageMeter:
    def __init__(self) -> None:
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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1, 5)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res: List[torch.Tensor] = []
        for k in topk:
            k_eff = min(k, output.size(1))
            correct_k = correct[:k_eff].reshape(-1).float().sum(0)
            res.append(correct_k * (100.0 / target.size(0)))
        return res


@dataclass
class EpochMetrics:
    loss: float
    top1: float
    top5: float
    balance: float
    samples: int


def reduce_epoch_metrics(
    loss_sum: torch.Tensor,
    top1_sum: torch.Tensor,
    top5_sum: torch.Tensor,
    balance_sum: torch.Tensor,
    sample_count: torch.Tensor,
    distributed: bool,
) -> EpochMetrics:
    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(top1_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(top5_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(balance_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

    n = max(int(sample_count.item()), 1)
    return EpochMetrics(
        loss=float(loss_sum.item() / n),
        top1=float(top1_sum.item() / n),
        top5=float(top5_sum.item() / n),
        balance=float(balance_sum.item() / n),
        samples=n,
    )


def build_dataloaders(
    args: argparse.Namespace,
    distributed: bool,
) -> tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    image_size = int(args.image_size)
    resize_size = int(math.floor(image_size / 0.875))

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if args.fake_data:
        train_dataset = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, image_size, image_size),
            num_classes=args.num_classes,
            transform=train_transform,
        )
        val_dataset = datasets.FakeData(
            size=args.fake_val_size,
            image_size=(3, image_size, image_size),
            num_classes=args.num_classes,
            transform=val_transform,
        )
    else:
        train_dir = Path(args.data_root) / "train"
        val_dir = Path(args.data_root) / "val"
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def build_model(args: argparse.Namespace) -> nn.Module:
    overrides = {}
    if args.time_steps > 0 and not args.use_fragmentation:
        overrides["time_steps"] = args.time_steps

    backbone = build_spikformer_preset(
        args.model_preset,
        image_size=(args.image_size, args.image_size),
        in_channels=3,
        num_classes=args.num_classes,
        spike_backend=args.spike_backend,
        backend=args.spike_compute_backend,
        **overrides,
    )

    fragmenter = None
    if args.use_fragmentation:
        if args.fragmentation_mode == "fixed":
            fragmenter = FixedLearnableFragmenter(
                image_size=(args.image_size, args.image_size),
                num_steps=args.fragment_steps,
                sharpness=args.fragment_sharpness,
                straight_through=True,
            )
        else:
            fragmenter = DynamicLearnableFragmenter(
                image_size=(args.image_size, args.image_size),
                candidates=tuple(args.fragment_candidates),
                gumbel_tau=args.fragment_gumbel_tau,
                sharpness=args.fragment_sharpness,
                straight_through=True,
                selector_init=args.fragment_selector_init,
            )

    return FragmentedSpikformer(backbone, fragmenter, decode=args.decode, gamma=args.gamma)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def create_optimizer_and_scheduler(args: argparse.Namespace, model: nn.Module, steps_per_epoch: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = max(args.warmup_epochs * steps_per_epoch, 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def get_autocast_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def make_scaler(args: argparse.Namespace) -> torch.amp.GradScaler:
    enabled = bool(args.amp) and torch.cuda.is_available() and args.amp_dtype == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def save_checkpoint(
    state: dict,
    output_dir: Path,
    filename: str,
    rank: int,
    is_best: bool = False,
) -> None:
    if not is_main_process(rank):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / filename
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy2(ckpt_path, output_dir / "best.pth")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float]:
    if not path:
        return 0, 0.0
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    unwrap_model(model).load_state_dict(state_dict, strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_top1 = float(ckpt.get("best_top1", 0.0))
    return start_epoch, best_top1


def train_one_epoch(
    *,
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    distributed: bool,
    rank: int,
) -> EpochMetrics:
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    loss_sum = torch.zeros(1, device=device)
    top1_sum = torch.zeros(1, device=device)
    top5_sum = torch.zeros(1, device=device)
    balance_sum = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)

    amp_dtype = get_autocast_dtype(args.amp_dtype)

    for step, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(args.amp) and device.type == "cuda"):
            logits, aux = model(images, return_aux=True)
            ce_loss = criterion(logits, target)
            balance_term = aux["balance_loss"] if aux["balance_loss"] is not None else logits.new_tensor(0.0)
            loss = ce_loss + float(args.balance_weight) * balance_term

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad))
            optimizer.step()

        scheduler.step()

        acc1, acc5 = accuracy(logits.detach(), target, topk=(1, 5))
        loss_sum += loss.detach() * batch_size
        top1_sum += acc1.detach() * batch_size / 100.0
        top5_sum += acc5.detach() * batch_size / 100.0
        balance_sum += balance_term.detach() * batch_size
        sample_count += batch_size

        maybe_reset_spiking_state(model)

        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process(rank) and (step % args.print_freq == 0 or step == len(train_loader) - 1):
            selected_steps = aux.get("selected_steps", None)
            selector_probs = aux.get("selector_probs", None)
            selector_text = ""
            if selector_probs is not None:
                selector_text = f", selector_probs={selector_probs.detach().cpu().tolist()}"
            print(
                f"[train] epoch={epoch:03d} step={step:04d}/{len(train_loader)-1:04d} "
                f"loss={loss.item():.4f} ce={ce_loss.item():.4f} bal={balance_term.item():.4f} "
                f"top1={acc1.item():.2f} top5={acc5.item():.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.6e} "
                f"data={data_time.val:.3f}s iter={batch_time.val:.3f}s"
                + (f", selected_steps={selected_steps}" if selected_steps is not None else "")
                + selector_text
            )

    return reduce_epoch_metrics(loss_sum, top1_sum, top5_sum, balance_sum, sample_count, distributed)


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    distributed: bool,
    rank: int,
    split_name: str = "val",
) -> EpochMetrics:
    model.eval()

    loss_sum = torch.zeros(1, device=device)
    top1_sum = torch.zeros(1, device=device)
    top5_sum = torch.zeros(1, device=device)
    balance_sum = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)

    amp_dtype = get_autocast_dtype(args.amp_dtype)

    for step, (images, target) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=bool(args.amp) and device.type == "cuda"):
            logits, aux = model(images, return_aux=True)
            ce_loss = criterion(logits, target)
            balance_term = aux["balance_loss"] if aux["balance_loss"] is not None else logits.new_tensor(0.0)
            loss = ce_loss + float(args.balance_weight) * balance_term

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        loss_sum += loss * batch_size
        top1_sum += acc1 * batch_size / 100.0
        top5_sum += acc5 * batch_size / 100.0
        balance_sum += balance_term * batch_size
        sample_count += batch_size

        maybe_reset_spiking_state(model)

        if is_main_process(rank) and (step % args.print_freq == 0 or step == len(val_loader) - 1):
            print(
                f"[{split_name}] step={step:04d}/{len(val_loader)-1:04d} "
                f"loss={loss.item():.4f} top1={acc1.item():.2f} top5={acc5.item():.2f}"
            )

    return reduce_epoch_metrics(loss_sum, top1_sum, top5_sum, balance_sum, sample_count, distributed)


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed(args)

    seed_everything(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(args, distributed)
    model = build_model(args).to(device)

    if distributed:
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=ddp_device_ids, output_device=local_rank if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer, scheduler = create_optimizer_and_scheduler(args, model, max(len(train_loader), 1))
    scaler = make_scaler(args)

    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    start_epoch = 0
    best_top1 = 0.0

    resume_path = args.resume if args.resume else ""
    test_ckpt = args.checkpoint if args.checkpoint else ""

    if resume_path:
        start_epoch, best_top1 = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        if is_main_process(rank):
            print(f"Resumed from {resume_path} at epoch={start_epoch}, best_top1={best_top1:.2f}")

    if args.mode == "test":
        ckpt_path = test_ckpt or resume_path
        if not ckpt_path:
            raise ValueError("--mode test requires --checkpoint or --resume")
        load_checkpoint(
            ckpt_path,
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            map_location="cpu",
        )
        metrics = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            distributed=distributed,
            rank=rank,
            split_name="test",
        )
        if is_main_process(rank):
            print(f"[test] loss={metrics.loss:.4f} top1={metrics.top1:.2f} top5={metrics.top5:.2f} balance={metrics.balance:.4f}")
        cleanup_distributed(distributed)
        return

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            device=device,
            args=args,
            distributed=distributed,
            rank=rank,
        )

        if is_main_process(rank):
            print(
                f"[train][epoch={epoch:03d}] loss={train_metrics.loss:.4f} "
                f"top1={train_metrics.top1:.2f} top5={train_metrics.top5:.2f} balance={train_metrics.balance:.4f}"
            )

        do_eval = ((epoch + 1) % args.eval_freq == 0) or (epoch + 1 == args.epochs) or (args.mode == "train_test")
        if do_eval:
            val_metrics = evaluate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                args=args,
                distributed=distributed,
                rank=rank,
                split_name="val",
            )
            if is_main_process(rank):
                print(
                    f"[val][epoch={epoch:03d}] loss={val_metrics.loss:.4f} "
                    f"top1={val_metrics.top1:.2f} top5={val_metrics.top5:.2f} balance={val_metrics.balance:.4f}"
                )

            is_best = val_metrics.top1 > best_top1
            best_top1 = max(best_top1, val_metrics.top1)

            state = {
                "epoch": epoch,
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_top1": best_top1,
                "args": vars(args),
            }
            if (epoch + 1) % args.save_every == 0 or is_best or (epoch + 1 == args.epochs):
                save_checkpoint(state, output_dir, f"checkpoint_epoch_{epoch:03d}.pth", rank, is_best=is_best)
            if distributed:
                dist.barrier()

    if args.mode == "train_test":
        if distributed:
            dist.barrier()
        best_path = output_dir / "best.pth"
        if best_path.exists():
            load_checkpoint(str(best_path), model=model, optimizer=None, scheduler=None, scaler=None, map_location="cpu")
        final_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
            distributed=distributed,
            rank=rank,
            split_name="final_test",
        )
        if is_main_process(rank):
            print(
                f"[final_test] loss={final_metrics.loss:.4f} "
                f"top1={final_metrics.top1:.2f} top5={final_metrics.top5:.2f} balance={final_metrics.balance:.4f}"
            )

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
