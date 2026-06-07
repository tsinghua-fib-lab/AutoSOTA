import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cofida.data import (
    SourceDataset,
    TargetDataset,
    TargetValDataset,
    find_images,
    load_monet_lookup,
    make_eval_transform,
    make_source_transform,
    make_target_strong_transform,
    make_target_weak_transform,
)
from cofida.evaluate import eval_source, eval_target
from cofida.models import CoFIDAMonet, EMA
from cofida.utils import device_and_flags, get_label_from_path, set_seed


class FocalLossCE(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor([1.0 - alpha, alpha], dtype=torch.float32)

    def forward(self, logits, labels):
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        idx = torch.arange(labels.size(0), device=logits.device)
        log_prob_target = log_prob[idx, labels]
        prob_target = prob[idx, labels]
        loss = -((1 - prob_target) ** self.gamma) * log_prob_target
        return (self.alpha.to(logits.device)[labels] * loss).mean()


def sym_kl_sharpen(logits_1, logits_2, temperature: float):
    with torch.no_grad():
        prob_1 = F.softmax(logits_1 / temperature, dim=1)
        prob_2 = F.softmax(logits_2 / temperature, dim=1)
    kl_12 = F.kl_div(F.log_softmax(logits_1, dim=1), prob_2, reduction="none").sum(1)
    kl_21 = F.kl_div(F.log_softmax(logits_2, dim=1), prob_1, reduction="none").sum(1)
    return 0.5 * (kl_12 + kl_21).mean()


def mse_rows(a, b):
    return ((a - b) ** 2).mean(dim=1).mean()


def orthogonality_penalty(edit: torch.Tensor, cls_weight: torch.Tensor):
    projection = (edit @ cls_weight.t()) @ cls_weight
    return (projection**2).mean()


def norm_bound_penalty(edit: torch.Tensor, r_max: float):
    return torch.clamp(edit.norm(dim=1) - r_max, min=0).mean()


def confidence_threshold(epoch: int, warmup_epochs: int, threshold_start: float, threshold_end: float, threshold_end_epoch: int):
    if epoch <= warmup_epochs:
        return 1.1
    if epoch >= threshold_end_epoch:
        return threshold_end
    span = threshold_end_epoch - warmup_epochs
    drop = (threshold_start - threshold_end) * (epoch - warmup_epochs) / max(1, span)
    return max(threshold_end, threshold_start - drop)


def train_one_epoch(model, ema, src_loader, tgt_loader, optimiser, runtime, epoch, config):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=runtime.use_amp)
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    steps = min(len(src_loader), len(tgt_loader))

    in_warmup = epoch <= 3
    ramp = 0.0 if in_warmup else min(1.0, (epoch - 3) / max(1, (10 - 3)))
    weight_kl = config.w_kl_max * ramp
    weight_feat = config.w_feat_max * ramp
    weight_edit = config.w_edit_max * ramp
    pseudo_threshold = confidence_threshold(
        epoch,
        config.pseudo_warmup_epochs,
        config.pseudo_t_start,
        config.pseudo_t_end,
        config.pseudo_t_end_epoch,
    )

    for parameter in model.backbone.parameters():
        parameter.requires_grad = not in_warmup

    focal = FocalLossCE(gamma=1.5, alpha=0.9)
    stats = {"sup": 0.0, "kl": 0.0, "feat": 0.0, "edit": 0.0, "pseudo": 0.0, "ortho": 0.0, "norm": 0.0, "total": 0.0, "n": 0, "pseudo_cov": 0.0}

    teacher_for_ema = copy.deepcopy(model).to(runtime.device)

    for _ in range(steps):
        src_batch = next(src_iter)
        tgt_batch = next(tgt_iter)
        src_images = src_batch["img"].to(runtime.device)
        src_labels = src_batch["label"].to(runtime.device)
        src_monet = src_batch["monet"].to(runtime.device)
        tgt_images_weak = tgt_batch["img_w"].to(runtime.device)
        tgt_images_strong = tgt_batch["img_s"].to(runtime.device)
        tgt_monet = tgt_batch["monet"].to(runtime.device)

        optimiser.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=runtime.use_amp):
            src_logits, _, _, _ = model.forward_full(src_images, src_monet)
            loss_sup = focal(src_logits, src_labels)

            weak_logits, _, weak_u, weak_e = model.forward_full(tgt_images_weak, tgt_monet)
            strong_logits, _, strong_u, strong_e = model.forward_full(tgt_images_strong, tgt_monet)

            with torch.no_grad():
                ema.load_shadow(teacher_for_ema)
                weak_logits_ema, _, _, _ = teacher_for_ema.forward_eval(tgt_images_weak, tgt_monet)

            loss_kl = sym_kl_sharpen(strong_logits, weak_logits_ema, config.temp)
            loss_feat = mse_rows(strong_u, weak_u)
            loss_edit = mse_rows(strong_e, weak_e)

            with torch.no_grad():
                pseudo_prob = torch.softmax(weak_logits_ema, dim=1)
                confidence, pseudo_labels = pseudo_prob.max(dim=1)
                mask = confidence >= pseudo_threshold
            stats["pseudo_cov"] += float(mask.float().mean().item())
            if mask.any() and epoch > config.pseudo_warmup_epochs:
                loss_pseudo = F.cross_entropy(strong_logits[mask], pseudo_labels[mask])
            else:
                loss_pseudo = strong_logits.sum() * 0.0

            cls_axes = model.head.mlp[0].weight
            loss_ortho = orthogonality_penalty(strong_e, cls_axes)
            loss_norm = norm_bound_penalty(strong_e, config.r_max)
            loss_uda = weight_kl * loss_kl + weight_feat * loss_feat + weight_edit * loss_edit
            total = loss_sup + loss_uda + loss_pseudo + config.lambda_ortho * loss_ortho + config.lambda_norm * loss_norm

        scaler.scale(total).backward()
        scaler.step(optimiser)
        scaler.update()
        ema.update(model)

        batch_size = src_images.size(0)
        stats["sup"] += float(loss_sup.item()) * batch_size
        stats["kl"] += float(loss_kl.item()) * batch_size
        stats["feat"] += float(loss_feat.item()) * batch_size
        stats["edit"] += float(loss_edit.item()) * batch_size
        stats["pseudo"] += float(loss_pseudo.item()) * batch_size
        stats["ortho"] += float(loss_ortho.item()) * batch_size
        stats["norm"] += float(loss_norm.item()) * batch_size
        stats["total"] += float(total.item()) * batch_size
        stats["n"] += batch_size

    for key in ("sup", "kl", "feat", "edit", "pseudo", "ortho", "norm", "total"):
        stats[key] /= max(1, stats["n"])
    stats["pseudo_cov"] /= max(1, steps)
    print(
        "Train:"
        f" total={stats['total']:.4f}"
        f" sup={stats['sup']:.4f}"
        f" kl={stats['kl']:.4f}"
        f" feat={stats['feat']:.4f}"
        f" edit={stats['edit']:.4f}"
        f" pseudo={stats['pseudo']:.4f}"
        f" ortho={stats['ortho']:.4f}"
        f" norm={stats['norm']:.4f}"
        f" pseudo_cov={stats['pseudo_cov']:.3f}"
    )


def train_teacher(config):
    set_seed(config.seed)
    runtime = device_and_flags()
    print(f"Device: {runtime.device.type} | AMP={runtime.use_amp}")
    os.makedirs(config.save_dir, exist_ok=True)

    monet_lookup, monet_cols = load_monet_lookup(config.monet_csv)
    num_concepts = len(monet_cols)
    print(f"Loaded MONET with {num_concepts} concepts")

    source_paths = find_images(config.source_dir)
    target_paths = find_images(config.target_dir)
    source_labels = [get_label_from_path(path) for path in source_paths]
    source_train, source_val = train_test_split(
        source_paths,
        test_size=0.2,
        stratify=source_labels,
        random_state=config.seed,
    )
    target_val_paths = find_images(config.target_val_dir) if config.target_val_dir else []
    print(f"Source images: {len(source_paths)} | Target images: {len(target_paths)}")
    if target_val_paths:
        print(f"Target val images: {len(target_val_paths)}")

    source_transform = make_source_transform(config.img_size)
    target_weak_transform = make_target_weak_transform(config.img_size)
    target_strong_transform = make_target_strong_transform(config.img_size)
    eval_transform = make_eval_transform(config.img_size)

    ds_src_train = SourceDataset(source_train, monet_lookup, source_transform, num_concepts)
    ds_src_val = SourceDataset(source_val, monet_lookup, eval_transform, num_concepts)
    ds_tgt_train = TargetDataset(target_paths, monet_lookup, target_weak_transform, target_strong_transform, num_concepts)

    dl_src_train = DataLoader(
        ds_src_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=runtime.pin_memory,
        drop_last=True,
    )
    dl_src_val = DataLoader(
        ds_src_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=runtime.pin_memory,
    )

    if target_val_paths:
        ds_tgt_val = TargetValDataset(target_val_paths, eval_transform, monet_lookup, num_concepts)
        dl_tgt_val = DataLoader(
            ds_tgt_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=runtime.pin_memory,
        )
    else:
        dl_tgt_val = None

    model = CoFIDAMonet(num_concepts=num_concepts, gini_pow=config.conf_gate_pow).to(runtime.device)
    if runtime.device.type == "mps":
        model = model.to(torch.float32)
    ema = EMA(model, decay=config.ema_decay)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_metric = -1.0
    best_path = os.path.join(config.save_dir, "best_cofida_monet.pt")

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs} | pseudo-thresh={confidence_threshold(epoch, config.pseudo_warmup_epochs, config.pseudo_t_start, config.pseudo_t_end, config.pseudo_t_end_epoch):.2f}")
        dl_tgt_train = DataLoader(
            ds_tgt_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=runtime.pin_memory,
            drop_last=True,
        )
        train_one_epoch(model, ema, dl_src_train, dl_tgt_train, optimiser, runtime, epoch, config)

        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        ema.load_shadow(model)
        src_metrics = eval_source(model, dl_src_val, runtime.device)
        print(
            f"Val (source, EMA): Acc={src_metrics['acc']:.4f}"
            f" BAcc={src_metrics['bacc']:.4f}"
            f" AUROC={src_metrics['auroc']:.4f}"
        )

        if dl_tgt_val is not None:
            tgt_metrics = eval_target(
                model,
                dl_tgt_val,
                runtime.device,
                use_recall_floor=config.use_recall_floor,
                mel_recall_floor=config.mel_recall_floor,
                report_opt=True,
            )
            message = (
                f"Val (target, EMA): Acc={tgt_metrics['acc']:.4f}"
                f" BAcc={tgt_metrics['bacc']:.4f}"
                f" AUROC={tgt_metrics['auroc']:.4f}"
            )
            if "thr_opt" in tgt_metrics:
                message += f" | BAcc@opt={tgt_metrics['bacc_opt']:.4f} thr_opt={tgt_metrics['thr_opt']:.3f}"
            print(message)
            score = tgt_metrics.get("bacc_opt", tgt_metrics["bacc"])
        else:
            tgt_metrics = None
            score = src_metrics["auroc"]

        if score > best_metric:
            best_metric = score
            torch.save(
                {
                    "epoch": epoch,
                    "ema_state": ema.shadow,
                    "model_state": model.state_dict(),
                    "monet_cols": monet_cols,
                    "src_val": src_metrics,
                    "tgt_val": tgt_metrics,
                    "config": vars(config),
                },
                best_path,
            )
            print(f"Saved best model to {best_path}")

        model.load_state_dict(backup, strict=False)

    print("\nTraining complete.")
    if os.path.exists(best_path):
        print(f"Best snapshot saved at {best_path}")
