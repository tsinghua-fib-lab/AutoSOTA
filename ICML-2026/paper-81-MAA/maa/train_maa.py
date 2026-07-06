#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import warnings
from contextlib import nullcontext

HELP_REQUESTED = any(arg in {"-h", "--help"} for arg in sys.argv[1:])

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    from .checkpoint import save_maa_adapter_state
    from .dataset import PairedDegradedImageDataset, paired_image_collate_fn
    from .modeling import prepare_maa_model
except ModuleNotFoundError:
    if not HELP_REQUESTED:
        raise

    class _MissingModule:
        pass

    class _MissingNN:
        Module = _MissingModule

    torch = None
    nn = _MissingNN()
    F = None
    AdamW = None
    DataLoader = None
    tqdm = None
    get_scheduler = None
    save_maa_adapter_state = None
    PairedDegradedImageDataset = None
    paired_image_collate_fn = None
    prepare_maa_model = None


warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f"maa.train.{os.path.abspath(output_dir)}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(os.path.join(output_dir, "training_log.log"))
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def module_dtype(module: nn.Module, fallback=None):
    if fallback is None:
        fallback = torch.float16
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return fallback


def project_tokens_chunked(mm_projector, tokens: torch.Tensor, dtype: torch.dtype, chunk_size: int):
    tokens = tokens.to(tokens.device, dtype=dtype)
    if chunk_size is None or chunk_size <= 0:
        return mm_projector(tokens)

    outputs = []
    for start in range(0, tokens.shape[1], chunk_size):
        end = min(tokens.shape[1], start + chunk_size)
        outputs.append(mm_projector(tokens[:, start:end, :]))
    return torch.cat(outputs, dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class TokenDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, num_res_blocks: int = 5):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)


def feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student, teacher)


def adversarial_loss(logits: torch.Tensor, target_label: float) -> torch.Tensor:
    targets = torch.full_like(logits, target_label)
    return F.binary_cross_entropy_with_logits(logits, targets)


def evaluate_l2(
    model: nn.Module,
    vision_tower_pristine: nn.Module,
    eval_loader: DataLoader,
    device: str,
    mm_projector: nn.Module,
    projector_dtype: torch.dtype,
    token_chunk_size: int = 0,
    autocast_ctx=None,
) -> float:
    model.eval()
    vision_tower_pristine.eval()
    total_loss = 0.0
    num_batches = 0
    autocast_ctx = autocast_ctx if autocast_ctx is not None else nullcontext

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            if batch is None:
                continue
            degraded = batch["degraded_pixel_values"].to(device, non_blocking=True)
            clean = batch["clean_pixel_values"].to(device, non_blocking=True)

            with autocast_ctx():
                teacher_out = vision_tower_pristine.maa_forward(clean, output_hidden_states=False)
                teacher_tokens = project_tokens_chunked(
                    mm_projector,
                    teacher_out.last_hidden_state[:, 1:, :],
                    projector_dtype,
                    token_chunk_size,
                )
                student_out = model.get_vision_tower().maa_forward(degraded, output_hidden_states=False)
                student_tokens = project_tokens_chunked(
                    mm_projector,
                    student_out.last_hidden_state[:, 1:, :],
                    projector_dtype,
                    token_chunk_size,
                )
                loss = feature_distillation_loss(student_tokens, teacher_tokens)

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches else float("inf")


def main(args) -> None:
    set_random_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.disable_amp) and device.startswith("cuda")

    logger = build_logger(args.output_dir)
    logger.info("Using device: %s", device)
    logger.info("Arguments: %s", vars(args))

    model, _, image_processor, vision_tower_pristine = prepare_maa_model(
        args.model_name_or_path,
        kernel_size=args.kernel_size,
        trainable=True,
        with_teacher=True,
    )
    model.to(device)
    vision_tower_pristine.to(device)
    vision_tower_pristine.eval()

    mm_projector = getattr(model.get_model(), "mm_projector", None)
    if mm_projector is None:
        raise RuntimeError("The loaded LLaVA model does not expose mm_projector.")
    projector_dtype = module_dtype(mm_projector)
    hidden_size = int(model.get_model().config.hidden_size)

    discriminator = TokenDiscriminator(
        input_dim=hidden_size,
        hidden_dim=args.discriminator_hidden_dim,
        num_res_blocks=args.discriminator_num_blocks,
    ).to(device)
    discriminator_params = sum(param.numel() for param in discriminator.parameters())
    logger.info("Discriminator parameters: %.2fM", discriminator_params / 1e6)

    full_dataset = PairedDegradedImageDataset(args.dataset_path, image_processor)
    total_count = len(full_dataset)
    val_count = max(1, int(total_count * args.validation_split_ratio))
    train_count = total_count - val_count
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=paired_image_collate_fn,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=paired_image_collate_fn,
        pin_memory=True,
    )

    adapter_params = [param for param in model.parameters() if param.requires_grad]
    optimizer_adapter = AdamW(adapter_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_discriminator = AdamW(
        discriminator.parameters(),
        lr=args.discriminator_learning_rate,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = max(1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    num_update_steps = updates_per_epoch * args.num_epochs
    num_warmup_steps = int(num_update_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer_adapter,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        def autocast_ctx():
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        model.train()
        discriminator.train()
        optimizer_adapter.zero_grad(set_to_none=True)
        optimizer_discriminator.zero_grad(set_to_none=True)

        running_l2 = 0.0
        running_adv = 0.0
        running_disc = 0.0
        running_steps = 0

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            if batch is None:
                continue

            degraded = batch["degraded_pixel_values"].to(device, non_blocking=True)
            clean = batch["clean_pixel_values"].to(device, non_blocking=True)

            with autocast_ctx():
                with torch.no_grad():
                    teacher_out = vision_tower_pristine.maa_forward(clean, output_hidden_states=False)
                    clean_features = project_tokens_chunked(
                        mm_projector,
                        teacher_out.last_hidden_state[:, 1:, :],
                        projector_dtype,
                        args.token_chunk_size,
                    )

                student_out = model.get_vision_tower().maa_forward(degraded, output_hidden_states=False)
                corrected_features = project_tokens_chunked(
                    mm_projector,
                    student_out.last_hidden_state[:, 1:, :],
                    projector_dtype,
                    args.token_chunk_size,
                )

                real_logits = discriminator(clean_features.detach())
                fake_logits = discriminator(corrected_features.detach())
                loss_disc = 0.5 * (
                    adversarial_loss(real_logits, 1.0) + adversarial_loss(fake_logits, 0.0)
                )

            scaler.scale(loss_disc / args.gradient_accumulation_steps).backward()

            for param in discriminator.parameters():
                param.requires_grad = False
            with autocast_ctx():
                fake_logits_for_adapter = discriminator(corrected_features)
                loss_adv = adversarial_loss(fake_logits_for_adapter, 1.0)
                loss_l2 = feature_distillation_loss(corrected_features, clean_features)
                loss_adapter = loss_l2 + args.adversarial_weight * loss_adv

            scaler.scale(loss_adapter / args.gradient_accumulation_steps).backward()
            for param in discriminator.parameters():
                param.requires_grad = True

            should_step = (
                (step + 1) % args.gradient_accumulation_steps == 0
                or (step + 1) == len(train_loader)
            )
            if should_step:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer_adapter)
                    torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=args.grad_clip)
                    scaler.unscale_(optimizer_discriminator)
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(),
                        max_norm=args.grad_clip,
                    )

                scaler.step(optimizer_adapter)
                scaler.step(optimizer_discriminator)
                scaler.update()
                scheduler.step()
                optimizer_adapter.zero_grad(set_to_none=True)
                optimizer_discriminator.zero_grad(set_to_none=True)
                global_step += 1

            running_l2 += loss_l2.item()
            running_adv += loss_adv.item()
            running_disc += loss_disc.item()
            running_steps += 1

            if args.logging_steps > 0 and global_step > 0 and global_step % args.logging_steps == 0:
                logger.info(
                    "Step %d: L2=%.4f Adv=%.4f D=%.4f LR=%.2e",
                    global_step,
                    running_l2 / max(1, running_steps),
                    running_adv / max(1, running_steps),
                    running_disc / max(1, running_steps),
                    scheduler.get_last_lr()[0],
                )
                running_l2 = running_adv = running_disc = 0.0
                running_steps = 0

        eval_autocast_ctx = autocast_ctx if use_amp else None
        val_loss = evaluate_l2(
            model,
            vision_tower_pristine,
            eval_loader,
            device,
            mm_projector,
            projector_dtype,
            token_chunk_size=args.token_chunk_size,
            autocast_ctx=eval_autocast_ctx,
        )
        logger.info("Epoch %d validation L2 loss: %.4f", epoch + 1, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_maa_adapter_state(model, os.path.join(args.output_dir, "best_adapter.pth"))
        save_maa_adapter_state(model, os.path.join(args.output_dir, "last_adapter.pth"))
        torch.save(
            discriminator.state_dict(),
            os.path.join(args.output_dir, "last_discriminator.pth"),
        )

    logger.info("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAA with paired distillation and adversarial alignment.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/maa")
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--discriminator_learning_rate", type=float, default=1e-6)
    parser.add_argument("--discriminator_hidden_dim", type=int, default=2048)
    parser.add_argument("--discriminator_num_blocks", type=int, default=5)
    parser.add_argument("--adversarial_weight", type=float, default=0.1)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--validation_split_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--token_chunk_size", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
