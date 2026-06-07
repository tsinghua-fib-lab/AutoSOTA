import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cofida.checkpointing import load_checkpoint_safe
from cofida.data import TargetWithMONET, find_images, load_monet_lookup, make_eval_transform, make_source_transform
from cofida.models import CoFIDAMonet, StudentImageOnly
from cofida.utils import device_and_flags, extract_lesion_id, set_seed


def kd_loss(student_logits, teacher_logits, temperature: float):
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    prob_teacher = F.softmax(teacher_logits / temperature, dim=1)
    return (temperature * temperature) * F.kl_div(log_student, prob_teacher, reduction="batchmean")


def split_target_paths(paths: list[str], monet_lookup: dict, seed: int, val_split: float):
    kept = [path for path in paths if (extract_lesion_id(path), "clinical") in monet_lookup]
    total = len(kept)
    val_len = max(1, int(val_split * total))
    train_len = total - val_len
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    train_paths = [kept[idx] for idx in train_indices]
    val_paths = [kept[idx] for idx in val_indices]
    return kept, train_paths, val_paths


def train_one_epoch(student, teacher, loader, optimiser, runtime, epoch: int, config):
    student.train()
    teacher.eval()
    scaler = torch.cuda.amp.GradScaler(enabled=runtime.use_amp)
    stats = {"kd": 0.0, "feat": 0.0, "total": 0.0, "entropy": 0.0, "proxy_acc": 0.0, "n": 0}
    step = 0
    for batch in loader:
        images = batch["img"].to(runtime.device)
        monet = batch["monet"].to(runtime.device)
        with torch.no_grad():
            teacher_logits, _, teacher_u, _ = teacher.forward_full(images, monet)
            teacher_pred = teacher_logits.argmax(dim=1)
        student_logits, student_u = student(images)
        loss_kd = kd_loss(student_logits, teacher_logits, config.kd_temperature)
        loss_feat = F.mse_loss(student_u, teacher_u) if config.feat_align_w > 0 else torch.tensor(0.0, device=runtime.device)
        loss = config.kd_weight * loss_kd + config.feat_align_w * loss_feat

        optimiser.zero_grad(set_to_none=True)
        if runtime.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            prob = torch.softmax(student_logits, dim=1)
            entropy = (-(prob * torch.log(prob + 1e-8)).sum(dim=1)).mean().item()
            proxy_acc = (student_logits.argmax(1) == teacher_pred).float().mean().item()

        batch_size = images.size(0)
        stats["kd"] += float(loss_kd.item()) * batch_size
        stats["feat"] += float(loss_feat.item()) * batch_size
        stats["total"] += float(loss.item()) * batch_size
        stats["entropy"] += entropy * batch_size
        stats["proxy_acc"] += proxy_acc * batch_size
        stats["n"] += batch_size
        step += 1
        if step % config.print_freq == 0:
            print(
                f"  step {step:5d}: kd={loss_kd.item():.4f}"
                f" feat={loss_feat.item():.4f}"
                f" tot={loss.item():.4f}"
                f" ent={entropy:.3f}"
                f" proxy={proxy_acc:.3f}"
            )

    for key in ("kd", "feat", "total", "entropy", "proxy_acc"):
        stats[key] /= max(1, stats["n"])
    print(
        f"Epoch {epoch:02d} | total={stats['total']:.4f}"
        f" kd={stats['kd']:.4f}"
        f" feat={stats['feat']:.4f}"
        f" entropy={stats['entropy']:.3f}"
        f" proxy-acc={stats['proxy_acc']:.3f}"
    )


@torch.no_grad()
def validate(student, teacher, loader, runtime, epoch: int, config, split: str = "val"):
    student.eval()
    teacher.eval()
    kd_sum = 0.0
    feat_sum = 0.0
    ent_sum = 0.0
    proxy_sum = 0.0
    num_samples = 0
    for batch in loader:
        images = batch["img"].to(runtime.device)
        monet = batch["monet"].to(runtime.device)
        teacher_logits, _, teacher_u, _ = teacher.forward_full(images, monet)
        student_logits, student_u = student(images)
        kd = kd_loss(student_logits, teacher_logits, config.kd_temperature).item()
        feat = F.mse_loss(student_u, teacher_u).item() if config.feat_align_w > 0 else 0.0
        prob = torch.softmax(student_logits, dim=1)
        entropy = (-(prob * torch.log(prob + 1e-8)).sum(dim=1)).mean().item()
        proxy_acc = (student_logits.argmax(1) == teacher_logits.argmax(1)).float().mean().item()
        batch_size = images.size(0)
        kd_sum += kd * batch_size
        feat_sum += feat * batch_size
        ent_sum += entropy * batch_size
        proxy_sum += proxy_acc * batch_size
        num_samples += batch_size
    output = {
        "kd": kd_sum / max(1, num_samples),
        "feat": feat_sum / max(1, num_samples),
        "entropy": ent_sum / max(1, num_samples),
        "proxy_acc": proxy_sum / max(1, num_samples),
    }
    print(
        f"[{split}] Epoch {epoch:02d} | kd={output['kd']:.4f}"
        f" feat={output['feat']:.4f}"
        f" entropy={output['entropy']:.3f}"
        f" proxy-acc={output['proxy_acc']:.3f}"
    )
    return output


def train_student(config):
    set_seed(config.seed)
    runtime = device_and_flags()
    print(f"Device: {runtime.device.type} | AMP={runtime.use_amp}")
    os.makedirs(config.save_dir, exist_ok=True)

    monet_lookup, monet_cols = load_monet_lookup(config.monet_csv)
    num_concepts = len(monet_cols)
    all_paths = find_images(config.target_dir)
    kept, train_paths, val_paths = split_target_paths(all_paths, monet_lookup, config.seed, config.val_split)
    print(f"Found {len(all_paths)} target images; keeping {len(kept)} with MONET metadata")
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_transform = make_source_transform(config.img_size)
    eval_transform = make_eval_transform(config.img_size)
    ds_train = TargetWithMONET(train_paths, monet_lookup, train_transform)
    ds_val = TargetWithMONET(val_paths, monet_lookup, eval_transform)
    dl_train = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=runtime.pin_memory,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=runtime.pin_memory,
    )

    checkpoint = load_checkpoint_safe(config.teacher_checkpoint)
    state_key = "ema_state" if "ema_state" in checkpoint else "model_state"
    teacher = CoFIDAMonet(num_concepts=num_concepts).to(runtime.device)
    missing, unexpected = teacher.load_state_dict(checkpoint[state_key], strict=False)
    if missing or unexpected:
        print("Teacher state dict info:", {"missing": missing, "unexpected": unexpected})
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = StudentImageOnly(num_classes=2, hidden=512).to(runtime.device)
    optimiser = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_kd = float("inf")
    best_path = os.path.join(config.save_dir, "best_student.pt")

    for epoch in range(1, config.epochs + 1):
        train_one_epoch(student, teacher, dl_train, optimiser, runtime, epoch, config)
        val_stats = validate(student, teacher, dl_val, runtime, epoch, config)
        if val_stats["kd"] < best_kd:
            best_kd = val_stats["kd"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": student.state_dict(),
                    "optim_state": optimiser.state_dict(),
                    "kd_val": best_kd,
                    "config": vars(config),
                },
                best_path,
            )
            print(f"[Save] New best student (KD={best_kd:.4f}) -> {best_path}")

    print("Training complete.")
    if os.path.exists(best_path):
        print(f"Best student checkpoint saved at {best_path}")
