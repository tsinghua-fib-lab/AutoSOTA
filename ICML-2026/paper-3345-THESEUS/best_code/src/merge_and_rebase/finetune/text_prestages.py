from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .schedulers import build_lr_scheduler


def _optimizer_from_name(params, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    opt = str(name).strip().lower()
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _resolve_text_embeddings_finetune_cfg(
    strategy_cfg: dict[str, Any] | None,
    *,
    default_epochs: int = 1,
    default_lr: float,
    default_weight_decay: float,
    default_warmup_length: int,
    default_clip_grad_norm: float,
    default_accumulate_grad_batches: int,
) -> dict[str, Any] | None:
    if not isinstance(strategy_cfg, dict):
        return None

    raw = strategy_cfg.get("text_embeddings_finetune", None)
    if raw is None:
        return None

    if isinstance(raw, bool):
        if not raw:
            return None
        cfg = {}
    elif isinstance(raw, dict):
        cfg = dict(raw)
    else:
        raise ValueError("strategy.text_embeddings_finetune must be a bool or dict when provided.")

    if not bool(cfg.get("enabled", True)):
        return None

    epochs_raw = cfg.get("epochs", None)
    epochs = int(default_epochs if epochs_raw is None else epochs_raw)
    if epochs <= 0:
        raise ValueError("strategy.text_embeddings_finetune.epochs must be >= 1.")

    warmup_length_raw = cfg.get("warmup_length", None)
    warmup_length = int(default_warmup_length if warmup_length_raw is None else warmup_length_raw)
    if warmup_length < 0:
        raise ValueError("strategy.text_embeddings_finetune.warmup_length must be >= 0.")

    accumulate_raw = cfg.get("accumulate_grad_batches", None)
    accumulate_grad_batches = int(default_accumulate_grad_batches if accumulate_raw is None else accumulate_raw)
    if accumulate_grad_batches <= 0:
        raise ValueError("strategy.text_embeddings_finetune.accumulate_grad_batches must be >= 1.")

    early_stopping_patience_raw = cfg.get("early_stopping_patience", None)
    early_stopping_patience = int(5 if early_stopping_patience_raw is None else early_stopping_patience_raw)
    if early_stopping_patience <= 0:
        raise ValueError("strategy.text_embeddings_finetune.early_stopping_patience must be >= 1.")

    return {
        "epochs": epochs,
        "optimizer": str(cfg.get("optimizer", "adamw")),
        "lr": float(cfg.get("lr", default_lr)),
        "weight_decay": float(cfg.get("weight_decay", default_weight_decay)),
        "scheduler_name": str(cfg.get("scheduler_name", "cosine")),
        "warmup_length": warmup_length,
        "clip_grad_norm": float(cfg.get("grad_clip_norm", default_clip_grad_norm)),
        "accumulate_grad_batches": accumulate_grad_batches,
        "early_stopping": bool(cfg.get("early_stopping", False)),
        "early_stopping_patience": early_stopping_patience,
    }


def _resolve_text_prompt_tuning_cfg(
    strategy_cfg: dict[str, Any] | None,
    *,
    default_epochs: int = 1,
    default_lr: float,
    default_weight_decay: float,
    default_warmup_length: int,
    default_clip_grad_norm: float,
    default_accumulate_grad_batches: int,
) -> dict[str, Any] | None:
    if not isinstance(strategy_cfg, dict):
        return None

    raw = strategy_cfg.get("text_prompt_tuning", None)
    if raw is None:
        return None

    if isinstance(raw, bool):
        if not raw:
            return None
        cfg = {}
    elif isinstance(raw, dict):
        cfg = dict(raw)
    else:
        raise ValueError("strategy.text_prompt_tuning must be a bool or dict when provided.")

    if not bool(cfg.get("enabled", True)):
        return None

    if "epochs" in cfg:
        epochs_raw = cfg.get("epochs", None)
        epochs = int(default_epochs if epochs_raw is None else epochs_raw)
    else:
        epochs = 1
    if epochs <= 0:
        raise ValueError("strategy.text_prompt_tuning.epochs must be >= 1.")

    warmup_length_raw = cfg.get("warmup_length", None)
    warmup_length = int(default_warmup_length if warmup_length_raw is None else warmup_length_raw)
    if warmup_length < 0:
        raise ValueError("strategy.text_prompt_tuning.warmup_length must be >= 0.")

    accumulate_raw = cfg.get("accumulate_grad_batches", None)
    accumulate_grad_batches = int(default_accumulate_grad_batches if accumulate_raw is None else accumulate_raw)
    if accumulate_grad_batches <= 0:
        raise ValueError("strategy.text_prompt_tuning.accumulate_grad_batches must be >= 1.")

    early_stopping_patience_raw = cfg.get("early_stopping_patience", None)
    early_stopping_patience = int(5 if early_stopping_patience_raw is None else early_stopping_patience_raw)
    if early_stopping_patience <= 0:
        raise ValueError("strategy.text_prompt_tuning.early_stopping_patience must be >= 1.")

    context_length_raw = cfg.get("context_length", None)
    context_length = int(16 if context_length_raw is None else context_length_raw)
    if context_length <= 0:
        raise ValueError("strategy.text_prompt_tuning.context_length must be >= 1.")

    ctx_init = cfg.get("ctx_init", None)
    if ctx_init is not None and not isinstance(ctx_init, str):
        raise ValueError("strategy.text_prompt_tuning.ctx_init must be a string when provided.")

    return {
        "epochs": epochs,
        "optimizer": str(cfg.get("optimizer", "adamw")),
        "lr": float(cfg.get("lr", default_lr)),
        "weight_decay": float(cfg.get("weight_decay", default_weight_decay)),
        "scheduler_name": str(cfg.get("scheduler_name", "cosine")),
        "warmup_length": warmup_length,
        "clip_grad_norm": float(cfg.get("grad_clip_norm", default_clip_grad_norm)),
        "accumulate_grad_batches": accumulate_grad_batches,
        "early_stopping": bool(cfg.get("early_stopping", False)),
        "early_stopping_patience": early_stopping_patience,
        "context_length": context_length,
        "ctx_init": str(ctx_init) if isinstance(ctx_init, str) and ctx_init.strip() else None,
        "init_std": float(cfg.get("init_std", 0.02)),
    }


def _find_eot_positions(tokenized: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    if tokenized.ndim != 2:
        raise ValueError(f"tokenized text must be 2D [B, L], got shape={tuple(tokenized.shape)}")
    if eos_token_id is None:
        return tokenized.argmax(dim=-1)

    mask = tokenized.eq(int(eos_token_id))
    has_eot = mask.any(dim=-1)
    first_eot = mask.int().argmax(dim=-1)
    fallback = tokenized.argmax(dim=-1)
    return torch.where(has_eot, first_eot, fallback)


def _build_coop_tokenized_prompts(
    *,
    tokenizer: Any,
    classnames: list[str],
    context_length: int,
    device: torch.device,
) -> torch.Tensor:
    prefix = " ".join(["X"] * int(context_length))
    prompts = [f"{prefix} {name}".strip() for name in classnames]
    tokenized = tokenizer(prompts)
    if not isinstance(tokenized, torch.Tensor):
        tokenized = torch.as_tensor(tokenized, dtype=torch.long)
    tokenized = tokenized.to(device=device)
    if tokenized.ndim != 2:
        raise ValueError(f"Tokenizer must return shape [B, L]; got {tuple(tokenized.shape)}")
    return tokenized


def _encode_openclip_text_with_context(
    *,
    clip_model: nn.Module,
    tokenized_prompts: torch.Tensor,
    context_vectors: torch.Tensor,
    normalize: bool,
) -> torch.Tensor:
    transformer = clip_model.transformer
    cast_dtype = torch.float32
    if hasattr(transformer, "get_cast_dtype"):
        cast_dtype_candidate = transformer.get_cast_dtype()
        if cast_dtype_candidate is not None:
            cast_dtype = cast_dtype_candidate

    x = clip_model.token_embedding(tokenized_prompts).to(dtype=cast_dtype)
    n_ctx = int(context_vectors.shape[0])
    if x.shape[1] <= n_ctx:
        raise ValueError(
            f"context_length={n_ctx} is too large for tokenizer output length={int(x.shape[1])}. "
            "Reduce strategy.text_prompt_tuning.context_length."
        )

    ctx = context_vectors.to(device=x.device, dtype=cast_dtype)
    x[:, 1 : 1 + n_ctx, :] = ctx.unsqueeze(0)

    pos = clip_model.positional_embedding[: x.shape[1]].to(device=x.device, dtype=cast_dtype)
    x = x + pos

    attn_mask = getattr(clip_model, "attn_mask", None)
    if isinstance(attn_mask, torch.Tensor):
        attn = attn_mask[: x.shape[1], : x.shape[1]]
        x = clip_model.transformer(x, attn_mask=attn)
    else:
        x = clip_model.transformer(x)

    x = clip_model.ln_final(x)

    eos_id_raw = getattr(clip_model, "text_eos_id", None)
    eos_id = int(eos_id_raw) if eos_id_raw is not None else None
    eot_pos = _find_eot_positions(tokenized_prompts, eos_id)
    pooled = x[torch.arange(x.shape[0], device=x.device), eot_pos]

    projection = getattr(clip_model, "text_projection", None)
    if projection is not None:
        if isinstance(projection, nn.Linear):
            pooled = projection(pooled)
        else:
            pooled = pooled @ projection.to(device=pooled.device, dtype=pooled.dtype)

    if normalize:
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-12)
    return pooled


@torch.no_grad()
def _top1_with_text_features(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    text_features: torch.Tensor,
) -> float:
    model.to(device)
    model.eval()

    text_feats = text_features.to(device=device)
    if text_feats.ndim != 2:
        raise ValueError("text_features must be a 2D matrix [C, D].")
    if model.clip_model.normalize:
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)

    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        img_feats = model.clip_model.model.visual(x)
        if model.clip_model.normalize:
            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)
        logits = model.clip_model.logit_scale * (img_feats @ text_feats.t())
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return float(correct / max(1, total))


def _run_text_feature_optimization_stage(
    *,
    task: str,
    stage_name: str,
    progress_label: str,
    model: nn.Module,
    loaders: Any,
    device: torch.device,
    cfg: dict[str, Any],
    trainable_params: list[nn.Parameter],
    build_train_text_features: Callable[[], torch.Tensor],
    build_eval_text_features: Callable[[], torch.Tensor],
    capture_best_state: Callable[[], Any] | None = None,
) -> tuple[dict[str, Any], torch.Tensor, Any]:
    opt = _optimizer_from_name(
        trainable_params,
        cfg["optimizer"],
        float(cfg["lr"]),
        float(cfg["weight_decay"]),
    )

    accumulate_grad_batches = int(cfg["accumulate_grad_batches"])
    steps_per_epoch = math.ceil(len(loaders.train) / accumulate_grad_batches)
    total_steps = max(1, int(cfg["epochs"]) * steps_per_epoch)
    scheduler = build_lr_scheduler(
        opt,
        name=str(cfg.get("scheduler_name", "cosine")),
        base_lrs=float(cfg["lr"]),
        warmup_length=int(cfg["warmup_length"]),
        steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()
    patience_left = int(cfg["early_stopping_patience"])

    with torch.no_grad():
        init_text = build_eval_text_features()
    init_val = (
        _top1_with_text_features(model=model, loader=loaders.val, device=device, text_features=init_text)
        if hasattr(loaders, "val") and loaders.val is not None
        else float("nan")
    )
    init_test = _top1_with_text_features(model=model, loader=loaders.test, device=device, text_features=init_text)

    best_metric = -1.0
    best_val = float("nan")
    best_test = float("nan")
    best_epoch = 0
    best_text = init_text.detach().clone()
    best_aux = capture_best_state() if capture_best_state is not None else None
    best_elapsed_seconds = 0.0
    last_val = float("nan")
    last_test = float("nan")
    last_epoch = 0
    last_elapsed_seconds = 0.0
    global_step = 0
    t_start = time.time()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.eval()
        running_loss = 0.0
        n_seen = 0
        train_loss = float("nan")
        opt.zero_grad(set_to_none=True)
        window_batch_count = 0
        window_size = 1

        with tqdm(
            total=len(loaders.train),
            desc=f"[{task}] {progress_label} epoch {epoch}/{cfg['epochs']}",
            unit="batch",
        ) as pbar:
            for i, (x, y) in enumerate(loaders.train):
                if window_batch_count == 0:
                    remaining = len(loaders.train) - i
                    window_size = min(accumulate_grad_batches, remaining)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.no_grad():
                    img_feats = model.clip_model.model.visual(x)
                    if model.clip_model.normalize:
                        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-12)

                text_feats = build_train_text_features()
                logits = model.clip_model.logit_scale * (img_feats @ text_feats.t())
                raw_loss = loss_fn(logits, y)
                loss = raw_loss / window_size
                loss.backward()

                window_batch_count += 1
                if window_batch_count == window_size:
                    if float(cfg["clip_grad_norm"]) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(cfg["clip_grad_norm"]))
                    scheduler(global_step)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                    window_batch_count = 0

                bs = int(y.numel())
                running_loss += float(raw_loss.item()) * bs
                n_seen += bs
                train_loss = running_loss / max(1, n_seen)
                pbar.update(1)
                pbar.set_postfix({"loss": f"{train_loss:.4f}", "lr": f"{opt.param_groups[0]['lr']:.6f}"})

        with torch.no_grad():
            eval_text = build_eval_text_features()
        val_acc = (
            _top1_with_text_features(model=model, loader=loaders.val, device=device, text_features=eval_text)
            if hasattr(loaders, "val") and loaders.val is not None
            else float("nan")
        )
        test_acc = _top1_with_text_features(model=model, loader=loaders.test, device=device, text_features=eval_text)

        last_epoch = epoch
        last_val = float(val_acc)
        last_test = float(test_acc)
        epoch_elapsed_seconds = float(time.time() - t_start)
        last_elapsed_seconds = epoch_elapsed_seconds
        monitor = float(val_acc) if not math.isnan(val_acc) else float(test_acc)

        if monitor > best_metric:
            best_metric = monitor
            best_val = float(val_acc)
            best_test = float(test_acc)
            best_epoch = int(epoch)
            best_text = eval_text.detach().clone()
            best_aux = capture_best_state() if capture_best_state is not None else None
            best_elapsed_seconds = epoch_elapsed_seconds
            patience_left = int(cfg["early_stopping_patience"])
        else:
            patience_left -= 1
            if bool(cfg["early_stopping"]) and patience_left <= 0:
                print(f"[{task}] {stage_name} stage early-stopped.")
                break

        print(
            f"[{task}] {stage_name} epoch {epoch:03d}/{int(cfg['epochs'])} "
            f"loss={train_loss:.4f} val={val_acc:.4f} test={test_acc:.4f} "
            f"patience={patience_left}/{int(cfg['early_stopping_patience'])}"
        )

    summary = {
        "enabled": True,
        "trainable_params": int(sum(p.numel() for p in trainable_params)),
        "seconds": float(time.time() - t_start),
        "optimizer": str(cfg["optimizer"]),
        "epochs": int(cfg["epochs"]),
        "lr": float(cfg["lr"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_length": int(cfg["warmup_length"]),
        "clip_grad_norm": float(cfg["clip_grad_norm"]),
        "accumulate_grad_batches": int(cfg["accumulate_grad_batches"]),
        "initial_val_top1": float(init_val),
        "initial_test_top1": float(init_test),
        "best_epoch": int(best_epoch),
        "best_elapsed_seconds": float(best_elapsed_seconds),
        "best_val_top1": float(best_val),
        "best_test_top1": float(best_test),
        "last_epoch": int(last_epoch),
        "last_elapsed_seconds": float(last_elapsed_seconds),
        "last_val_top1": float(last_val),
        "last_test_top1": float(last_test),
    }
    return summary, best_text, best_aux


def _run_text_prompt_tuning_stage(
    *,
    task: str,
    model: nn.Module,
    loaders: Any,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    clip = model.clip_model
    clip_model = clip.model
    tokenizer = clip.tokenizer
    if tokenizer is None:
        raise RuntimeError("Text prompt tuning requires a tokenizer on model.clip_model.tokenizer.")

    context_length = int(cfg["context_length"])
    if context_length <= 0:
        raise ValueError("context_length must be >= 1.")

    clip_params = list(clip_model.parameters())
    clip_requires_grad = [bool(p.requires_grad) for p in clip_params]
    for p in clip_params:
        p.requires_grad = False

    try:
        tokenized_prompts = _build_coop_tokenized_prompts(
            tokenizer=tokenizer,
            classnames=list(loaders.classnames),
            context_length=context_length,
            device=device,
        )

        eos_id_raw = getattr(clip_model, "text_eos_id", None)
        eos_id = int(eos_id_raw) if eos_id_raw is not None else None
        eot_pos = _find_eot_positions(tokenized_prompts, eos_id)
        if torch.any(eot_pos <= context_length):
            raise ValueError(
                f"[{task}] strategy.text_prompt_tuning.context_length={context_length} is too large for one or more class prompts. "
                "Reduce context_length to keep classname tokens in the prompt."
            )

        width = int(clip_model.token_embedding.weight.shape[1])
        ctx = nn.Parameter(torch.empty(context_length, width, device=device))
        torch.nn.init.normal_(ctx, std=float(cfg["init_std"]))

        ctx_init = cfg.get("ctx_init", None)
        if isinstance(ctx_init, str) and ctx_init.strip():
            init_ids = tokenizer([ctx_init.strip()])
            if not isinstance(init_ids, torch.Tensor):
                init_ids = torch.as_tensor(init_ids, dtype=torch.long)
            init_ids = init_ids.to(device=device)
            init_emb = clip_model.token_embedding(init_ids)[0].detach().to(device=device, dtype=ctx.dtype)
            init_eot = int(_find_eot_positions(init_ids, eos_id)[0].item())
            usable = max(0, min(context_length, init_eot - 1))
            if usable > 0:
                ctx.data[:usable].copy_(init_emb[1 : 1 + usable])

        def _train_text_features() -> torch.Tensor:
            return _encode_openclip_text_with_context(
                clip_model=clip_model,
                tokenized_prompts=tokenized_prompts,
                context_vectors=ctx,
                normalize=bool(clip.normalize),
            )

        def _eval_text_features() -> torch.Tensor:
            return _encode_openclip_text_with_context(
                clip_model=clip_model,
                tokenized_prompts=tokenized_prompts,
                context_vectors=ctx.detach(),
                normalize=bool(clip.normalize),
            )

        base_summary, best_text, best_ctx = _run_text_feature_optimization_stage(
            task=task,
            stage_name="text-prompt",
            progress_label="Prompt",
            model=model,
            loaders=loaders,
            device=device,
            cfg=cfg,
            trainable_params=[ctx],
            build_train_text_features=_train_text_features,
            build_eval_text_features=_eval_text_features,
            capture_best_state=lambda: ctx.detach().clone(),
        )

        with torch.no_grad():
            model.clip_model._zs_text_features = best_text.to(device=device)
            model.clip_model._zs_text_fingerprint = None
            if isinstance(best_ctx, torch.Tensor):
                model.clip_model._tuned_prompt_context = best_ctx.detach().cpu()  # type: ignore[attr-defined]

        summary = dict(base_summary)
        summary.update(
            {
                "context_length": int(context_length),
                "ctx_init": cfg.get("ctx_init", None),
                "init_std": float(cfg["init_std"]),
            }
        )
        return summary
    finally:
        for p, flag in zip(clip_params, clip_requires_grad, strict=False):
            p.requires_grad = flag


def _run_text_embeddings_finetune_stage(
    *,
    task: str,
    model: nn.Module,
    loaders: Any,
    device: torch.device,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if model.clip_model._zs_text_features.numel() == 0:
        raise RuntimeError("Text-embedding fine-tuning requires prebuilt _zs_text_features.")

    text_param = nn.Parameter(model.clip_model._zs_text_features.detach().to(device=device).clone())

    def _train_text_features() -> torch.Tensor:
        text_feats = text_param
        if model.clip_model.normalize:
            text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)
        return text_feats

    def _eval_text_features() -> torch.Tensor:
        text_feats = text_param.detach()
        if model.clip_model.normalize:
            text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)
        return text_feats

    summary, best_text, _ = _run_text_feature_optimization_stage(
        task=task,
        stage_name="text-emb",
        progress_label="TextEmb",
        model=model,
        loaders=loaders,
        device=device,
        cfg=cfg,
        trainable_params=[text_param],
        build_train_text_features=_train_text_features,
        build_eval_text_features=_eval_text_features,
    )

    with torch.no_grad():
        final_text = best_text
        if model.clip_model.normalize:
            final_text = final_text / (final_text.norm(dim=-1, keepdim=True) + 1e-12)
        model.clip_model._zs_text_features = final_text.to(device=device)
        model.clip_model._zs_text_fingerprint = None

    return summary
