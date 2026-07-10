from __future__ import annotations

import argparse
import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml  # type: ignore
from tqdm import tqdm

from merge_and_rebase.cli_args import add_logging_args, build_logging_overrides
from merge_and_rebase.run_logging import default_summary_path, finish_with_error, merge_logging_config, start_run
from merge_and_rebase.utils.helpers import parse_csv

from ..data.text_loaders import (
    NLI_TASKS,
    NLITokenizedData,
    build_nli_task_data,
    build_nli_tokenized_loader,
    default_head_class_ids_for_task,
)
from ..models.text_lm import TextBuildConfig, TextLM
from .forward_mode import apply_training_forward_mode, resolve_training_forward_mode
from .schedulers import build_lr_scheduler

NLI_SUITES: dict[str, tuple[str, ...]] = {
    "nli6": tuple(NLI_TASKS),
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _set_seed(seed: int, *, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    if p.suffix.lower() in [".yaml", ".yml"]:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("YAML config must be a mapping at top-level.")
        return cfg

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("JSON config must be an object at top-level.")
        return cfg

    raise ValueError(f"Unsupported config extension: {p.suffix} (use .yaml/.yml or .json)")


def _get_common_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    common = cfg.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("config['common'] must be a dict.")
    return common


def _get_dataset_override(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    ds = cfg.get("datasets", {})
    if ds is None:
        return {}
    if not isinstance(ds, dict):
        raise ValueError("config['datasets'] must be a dict mapping task -> overrides.")
    ov = ds.get(task, {})
    if ov is None:
        return {}
    if not isinstance(ov, dict):
        raise ValueError(f"config['datasets']['{task}'] must be a dict.")
    return ov


def _resolve_tasks_from_cfg(cfg: dict[str, Any]) -> list[str] | None:
    order = cfg.get("datasets_order", None)
    if order is None:
        return None
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise ValueError("config['datasets_order'] must be a list[str].")
    return [str(x).strip().lower() for x in order]


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _safe_model_tag(model_name_or_path: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "__", str(model_name_or_path).strip())
    return cleaned.strip("_") or "model"


def _canonical_param_name(name: str) -> str:
    out = str(name)
    if out.startswith("base_model.model."):
        out = out[len("base_model.model.") :]
    out = out.replace(".modules_to_save.default", "")
    return out


def _detect_head_roots(model: nn.Module) -> list[str]:
    roots: list[str] = []
    seen: set[str] = set()
    for n, p in model.named_parameters():
        if not p.requires_grad and p.numel() == 0:
            continue
        cn = _canonical_param_name(n)
        root = cn.split(".")[0] if "." in cn else cn
        if root in {"score", "classifier", "classification_head"} and root not in seen:
            seen.add(root)
            roots.append(root)
    if roots:
        # Stable preferred order.
        order = {"score": 0, "classifier": 1, "classification_head": 2}
        roots.sort(key=lambda x: order.get(x, 99))
    return roots


def _extract_task_head(model: nn.Module) -> dict[str, torch.Tensor]:
    roots = _detect_head_roots(model)
    named = list(model.named_parameters())

    if roots:
        root = roots[0]
        out: dict[str, torch.Tensor] = {}
        for n, p in named:
            cn = _canonical_param_name(n)
            if cn == root or cn.startswith(root + "."):
                out[cn] = p.detach().cpu().clone()
        if out:
            return out

    # Fallback: pick the smallest matrix with out-dim == num_labels and related bias.
    num_labels = int(getattr(getattr(model, "config", None), "num_labels", 0))
    matrix_candidates: list[tuple[str, torch.Tensor]] = []
    if num_labels > 0:
        for n, p in named:
            cn = _canonical_param_name(n)
            if p.ndim == 2 and int(p.shape[0]) == num_labels:
                matrix_candidates.append((cn, p))
    if not matrix_candidates:
        raise RuntimeError("Unable to extract task head: no known classification-head parameters found.")

    matrix_candidates.sort(key=lambda kv: int(kv[1].numel()))
    weight_name, weight_param = matrix_candidates[0]
    out = {weight_name: weight_param.detach().cpu().clone()}

    bias_name = weight_name.rsplit(".", 1)[0] + ".bias" if "." in weight_name else "bias"
    for n, p in named:
        cn = _canonical_param_name(n)
        if cn == bias_name:
            out[cn] = p.detach().cpu().clone()
            break
    return out


def _resolve_head_class_ids(
    task: str,
    *,
    task_num_labels: int,
    head_num_labels: int,
    task_cfg: dict[str, Any],
) -> list[int]:
    explicit = _get(task_cfg, "data.head_class_ids", None)
    if explicit is not None:
        if not isinstance(explicit, list) or not all(isinstance(x, int) for x in explicit):
            raise ValueError(f"[{task}] data.head_class_ids must be a list[int] when provided.")
        out = [int(x) for x in explicit]
        if len(out) != int(task_num_labels):
            raise ValueError(
                f"[{task}] data.head_class_ids length mismatch. "
                f"got={len(out)} expected={int(task_num_labels)}"
            )
        return out

    mask_class = _get(task_cfg, "data.mask_class", None)
    if mask_class is not None:
        masked = int(mask_class)
        if masked < 0 or masked >= int(head_num_labels):
            raise ValueError(f"[{task}] data.mask_class={masked} is out of range for num_labels={head_num_labels}.")
        out = [i for i in range(int(head_num_labels)) if i != masked]
        if len(out) != int(task_num_labels):
            raise ValueError(
                f"[{task}] data.mask_class={masked} yields {len(out)} classes, "
                f"expected {int(task_num_labels)}."
            )
        return out

    if int(head_num_labels) == int(task_num_labels):
        return list(range(int(task_num_labels)))

    # Common 3-way shared head conventions used by NLI tasks.
    t = str(task).strip().lower()
    if int(task_num_labels) == 2 and int(head_num_labels) >= 3:
        if t in {"qnli", "rte"}:
            return [0, 2]
        if t == "scitail":
            return [0, 1]

    out = default_head_class_ids_for_task(task, num_labels=int(head_num_labels))
    if len(out) == int(task_num_labels):
        return out
    raise ValueError(
        f"[{task}] could not infer valid head_class_ids: "
        f"task_num_labels={int(task_num_labels)}, head_num_labels={int(head_num_labels)}. "
        "Set data.head_class_ids or data.mask_class explicitly."
    )


def _build_task_loaders(
    *,
    task: str,
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    max_length: int,
    head_num_labels: int,
    task_cfg: dict[str, Any],
) -> tuple[NLITokenizedData, NLITokenizedData, NLITokenizedData, dict[str, Any]]:
    max_train_samples = _get(task_cfg, "data.max_train_samples", None)
    max_val_samples = _get(task_cfg, "data.max_val_samples", None)
    max_test_samples = _get(task_cfg, "data.max_test_samples", None)

    train_data = build_nli_task_data(task=task, split="train", max_samples=max_train_samples)
    val_data = build_nli_task_data(task=task, split="validation", max_samples=max_val_samples)
    test_data = build_nli_task_data(task=task, split="test", max_samples=max_test_samples)

    head_class_ids = _resolve_head_class_ids(
        task,
        task_num_labels=len(train_data.labels),
        head_num_labels=head_num_labels,
        task_cfg=task_cfg,
    )
    if len(head_class_ids) != len(train_data.labels):
        raise ValueError(
            f"[{task}] head_class_ids length mismatch. got={len(head_class_ids)} expected={len(train_data.labels)}"
        )

    train_loader = build_nli_tokenized_loader(
        task_data=train_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=True,
        head_class_ids=head_class_ids,
    )
    val_loader = build_nli_tokenized_loader(
        task_data=val_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=False,
        head_class_ids=head_class_ids,
    )
    test_loader = build_nli_tokenized_loader(
        task_data=test_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        shuffle=False,
        head_class_ids=head_class_ids,
    )

    meta = {
        "train": train_loader.meta,
        "validation": val_loader.meta,
        "test": test_loader.meta,
        "labels": list(train_data.labels),
        "label_texts": list(train_data.label_texts),
        "head_class_ids": list(head_class_ids),
    }
    return train_loader, val_loader, test_loader, meta


def _optimizer_from_name(params, name: str, lr: float, weight_decay: float) -> optim.Optimizer:
    opt = str(name).strip().lower()
    if opt == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if opt == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _configure_text_strategy(
    *,
    model: nn.Module,
    strategy: str,
    strategy_cfg: dict[str, Any] | None,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    warmup_length: int,
    scheduler_name: str = "cosine",
    steps: int,
    device: torch.device,
) -> tuple[nn.Module, optim.Optimizer, Any, dict[str, int], dict[str, Any]]:
    cfg = dict(strategy_cfg or {})
    name = str(strategy).strip().lower()
    peft_cfg_out: dict[str, Any] = {}

    if name == "full":
        for p in model.parameters():
            p.requires_grad = True

    elif name == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        roots = _detect_head_roots(model)
        if not roots:
            raise RuntimeError("linear_probe requested, but no classifier head root was found.")
        active_root = roots[0]
        for n, p in model.named_parameters():
            cn = _canonical_param_name(n)
            if cn == active_root or cn.startswith(active_root + "."):
                p.requires_grad = True

    elif name == "peft_lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:  # pragma: no cover - env dependent
            raise ImportError("PEFT LoRA strategy requires `peft` to be installed.") from e

        peft_cfg = cfg.get("peft", {}) if isinstance(cfg, dict) else {}
        if not isinstance(peft_cfg, dict):
            raise ValueError("strategy.peft must be a dict when using strategy.name='peft_lora'.")

        target_modules = peft_cfg.get("target_modules", None)
        if not isinstance(target_modules, list) or not all(isinstance(x, str) for x in target_modules):
            raise ValueError("strategy.peft.target_modules must be a list[str].")

        roots = _detect_head_roots(model)
        modules_to_save = peft_cfg.get("modules_to_save", None)
        if modules_to_save is None:
            modules_to_save = roots if roots else None
        if modules_to_save is not None and (
            not isinstance(modules_to_save, list) or not all(isinstance(x, str) for x in modules_to_save)
        ):
            raise ValueError("strategy.peft.modules_to_save must be a list[str] when provided.")

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=int(peft_cfg.get("r", 16)),
            lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),
            target_modules=[str(x) for x in target_modules],
            bias=str(peft_cfg.get("bias", "none")),
            modules_to_save=[str(x) for x in modules_to_save] if modules_to_save else None,
        )
        model = get_peft_model(model, lora_cfg)
        peft_cfg_out = {
            "task_type": "SEQ_CLS",
            "inference_mode": False,
            "r": int(peft_cfg.get("r", 16)),
            "lora_alpha": int(peft_cfg.get("lora_alpha", 16)),
            "lora_dropout": float(peft_cfg.get("lora_dropout", 0.0)),
            "target_modules": [str(x) for x in target_modules],
            "bias": str(peft_cfg.get("bias", "none")),
            "modules_to_save": [str(x) for x in modules_to_save] if modules_to_save else [],
        }

    else:
        raise ValueError("strategy.name must be one of: full, linear_probe, peft_lora")

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(f"Strategy '{name}' produced zero trainable parameters.")

    opt = _optimizer_from_name(trainable_params, optimizer_name, lr, weight_decay)
    scheduler = build_lr_scheduler(
        opt,
        name=scheduler_name,
        base_lrs=lr,
        warmup_length=warmup_length,
        steps=steps,
    )

    info: dict[str, int] = {
        "trainable_params": int(sum(p.numel() for p in trainable_params)),
    }
    if name == "peft_lora":
        info["lora_params"] = int(
            sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower())
        )
    info["scheduler_name"] = scheduler_name

    return model, opt, scheduler, info, peft_cfg_out


@torch.no_grad()
def _top1(model: nn.Module, loader, device: str) -> float:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(dev)
    model.eval()

    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(dev, non_blocking=True)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev, non_blocking=True)
        labels = batch["labels"].to(dev, non_blocking=True).long()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())

    return float(correct / max(1, total))


def _save_peft_text_adapter(
    *,
    model: nn.Module,
    tokenizer: Any,
    task_dir: Path,
    strategy: str,
    suffix: str | None,
    peft_cfg: dict[str, Any] | None,
    build_cfg: TextBuildConfig,
) -> dict[str, Any]:
    if not hasattr(model, "save_pretrained"):
        raise ValueError("save_format='peft' expects a PEFT-wrapped model with .save_pretrained().")

    adapter_name = f"{strategy}_adapter" if suffix is None else f"{strategy}_{suffix}_adapter"
    adapter_dir = task_dir / adapter_name
    _ensure_dir(adapter_dir)
    model.save_pretrained(adapter_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(adapter_dir)

    meta = {
        "format": "peft",
        "peft_target": "text",
        "peft_adapter_dir": str(adapter_dir),
        "peft_cfg": peft_cfg if peft_cfg is not None else {},
        "backbone": {
            "kind": "hf_text",
            "model_name_or_path": build_cfg.model_name_or_path,
            "model_arch": build_cfg.model_arch,
            "model_kind": build_cfg.model_kind,
            "dtype": build_cfg.dtype,
        },
    }
    _save_json(adapter_dir / "merge_and_rebase_meta.json", meta)
    return meta


def train_task(
    *,
    task: str,
    build_cfg: TextBuildConfig,
    strategy: str,
    strategy_cfg: dict[str, Any] | None,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_length: int,
    scheduler_name: str = "cosine",
    optimizer_name: str,
    clip_grad_norm: float,
    accumulate_grad_batches: int,
    batch_size: int,
    num_workers: int,
    max_length: int,
    head_num_labels: int,
    early_stopping: bool,
    early_stopping_patience: int,
    seed: int,
    deterministic: bool,
    device: str,
    out_dir: Path,
    save_format: str,
    save_last_epoch: bool = False,
    task_cfg: dict[str, Any] | None = None,
    log_every_n_steps: int = 50,
    run_logger: Any | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if accumulate_grad_batches <= 0:
        raise ValueError("accumulate_grad_batches must be >= 1.")

    dev = _device(device)
    _set_seed(seed, deterministic=deterministic)
    forward_mode = resolve_training_forward_mode(strategy_cfg)

    if build_cfg.model_kind != "sequence_classification":
        raise ValueError("train_text currently supports backbone.model_kind='sequence_classification' only.")

    llm = TextLM.build(build_cfg)
    model = llm.model
    tokenizer = llm.tokenizer

    train_loader, val_loader, test_loader, task_meta = _build_task_loaders(
        task=task,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        head_num_labels=head_num_labels,
        task_cfg=task_cfg or {},
    )
    expected_num_labels = int(len(task_meta.get("labels", [])))
    model_num_labels = int(getattr(model.config, "num_labels", expected_num_labels))
    if model_num_labels != expected_num_labels:
        raise ValueError(
            f"[{task}] model head/logits mismatch: model_num_labels={model_num_labels} "
            f"but dataset_num_labels={expected_num_labels}. "
            "Ensure backbone.num_labels matches the dataset label space for this task."
        )

    task_dir = out_dir / _safe_model_tag(build_cfg.model_name_or_path) / task
    _ensure_dir(task_dir)
    if run_logger is not None:
        run_logger.log_event(
            "task_start",
            metrics={},
            context={
                "task": task,
                "strategy": strategy,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "effective_batch_size": int(batch_size * accumulate_grad_batches),
                "task_dir": str(task_dir),
            },
        )

    steps_per_epoch = math.ceil(len(train_loader.loader) / accumulate_grad_batches)
    total_steps = max(1, epochs * steps_per_epoch)
    model, opt, scheduler, trainable_info, peft_cfg_out = _configure_text_strategy(
        model=model,
        strategy=strategy,
        strategy_cfg=strategy_cfg,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        warmup_length=warmup_length,
        scheduler_name=scheduler_name,
        steps=total_steps,
        device=dev,
    )
    trainable_info = dict(trainable_info)
    trainable_info["forward_mode"] = forward_mode
    trainable_info.update(
        apply_training_forward_mode(
            model=model,
            forward_mode=forward_mode,
            device=dev,
            output_transform=lambda out: out.logits,
            output_builder=lambda logits: SimpleNamespace(loss=None, logits=logits),
        )
    )

    best_val = -1.0
    best_state: dict[str, Any] | None = None
    best_head_payload: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    last_epoch = 0
    last_val = float("nan")
    last_test = float("nan")
    patience_left = int(early_stopping_patience)

    t_start = time.time()
    global_update_step = 0
    ckpt_stem = str(strategy) if forward_mode == "standard" else f"{strategy}__{forward_mode}"

    def _build_checkpoint_payload(
        *,
        epoch_i: int,
        val_acc_i: float,
        test_acc_i: float,
        kind: str,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        payload: dict[str, Any] = {
            "task": task,
            "strategy": strategy,
            "forward_mode": forward_mode,
            "backbone": {
                "kind": "hf_text",
                "model_name_or_path": build_cfg.model_name_or_path,
                "model_arch": build_cfg.model_arch,
                "model_kind": build_cfg.model_kind,
                "dtype": build_cfg.dtype,
            },
            "num_labels": int(getattr(model.config, "num_labels", head_num_labels)),
            "labels": list(task_meta.get("labels", [])),
            "label_texts": list(task_meta.get("label_texts", [])),
            "head_class_ids": list(task_meta.get("head_class_ids", [])),
            "metrics": {
                "val_top1": float(val_acc_i),
                "test_top1": float(test_acc_i),
            },
        }
        if kind == "best_ep":
            payload["best_epoch"] = int(epoch_i)
        elif kind == "last_ep":
            payload["last_epoch"] = int(epoch_i)
            payload["best_epoch"] = int(best_epoch)
        else:
            raise ValueError("kind must be 'best_ep' or 'last_ep'")

        head_payload = _extract_task_head(model)

        if save_format == "full":
            payload["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            payload["format"] = "full"
        elif save_format == "head":
            payload["head"] = head_payload
            payload["format"] = "head"
        elif save_format == "peft":
            payload.update(
                _save_peft_text_adapter(
                    model=model,
                    tokenizer=tokenizer,
                    task_dir=task_dir,
                    strategy=ckpt_stem,
                    suffix=kind,
                    peft_cfg=peft_cfg_out,
                    build_cfg=build_cfg,
                )
            )
        else:
            raise ValueError("save_format must be 'full', 'head', or 'peft'")

        return payload, head_payload

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)

        window_batch_count = 0
        window_size = 1
        with tqdm(total=len(train_loader.loader), desc=f"[{task}] Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for i, batch in enumerate(train_loader.loader):
                if window_batch_count == 0:
                    remaining = len(train_loader.loader) - i
                    window_size = min(accumulate_grad_batches, remaining)

                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(dev, non_blocking=True)
                labels = batch["labels"].to(dev, non_blocking=True).long()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                raw_loss = outputs.loss
                if raw_loss is None:
                    logits = outputs.logits
                    raw_loss = nn.CrossEntropyLoss()(logits, labels)
                loss = raw_loss / window_size
                loss.backward()

                window_batch_count += 1
                should_step = window_batch_count == window_size
                if window_batch_count == window_size:
                    if clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                    scheduler(global_update_step)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    global_update_step += 1
                    window_batch_count = 0

                bs = int(labels.numel())
                running_loss += float(raw_loss.item()) * bs
                n_seen += bs

                train_loss = running_loss / max(1, n_seen)
                pbar.update(1)
                pbar.set_postfix({"loss": f"{train_loss:.4f}", "lr": f"{opt.param_groups[0]['lr']:.6f}"})
                if (
                    run_logger is not None
                    and log_every_n_steps > 0
                    and global_update_step > 0
                    and global_update_step % log_every_n_steps == 0
                    and should_step
                ):
                    run_logger.log_event(
                        "train_step",
                        metrics={
                            f"train/{task}/loss": float(train_loss),
                            f"train/{task}/lr": float(opt.param_groups[0]["lr"]),
                        },
                        step=int(global_update_step),
                        context={
                            "task": task,
                            "epoch": int(epoch),
                        },
                    )

        val_acc = _top1(model, val_loader.loader, str(dev))
        test_acc = _top1(model, test_loader.loader, str(dev))

        last_epoch = epoch
        last_val = float(val_acc)
        last_test = float(test_acc)

        if not math.isnan(val_acc) and val_acc > best_val:
            patience_left = int(early_stopping_patience)
            best_epoch = int(epoch)
            best_val = float(val_acc)
            best_state, best_head_payload = _build_checkpoint_payload(
                epoch_i=best_epoch,
                val_acc_i=float(val_acc),
                test_acc_i=float(test_acc),
                kind="best_ep",
            )
        else:
            patience_left -= 1
            if early_stopping and patience_left <= 0:
                print(f"[{task}] Early stopping triggered.")
                break

        print(
            f"[{task}] epoch {epoch:03d}/{epochs}  "
            f"loss={train_loss:.4f}  val={val_acc:.4f}  test={test_acc:.4f} "
            f"patience={patience_left}/{early_stopping_patience}"
        )
        if run_logger is not None:
            run_logger.log_event(
                "epoch_end",
                metrics={
                    f"train/{task}/loss": float(train_loss),
                    f"train/{task}/lr": float(opt.param_groups[0]["lr"]),
                    f"val/{task}/top1": float(val_acc),
                    f"test/{task}/top1": float(test_acc),
                    f"train/{task}/seconds": float(time.time() - t_start),
                },
                step=int(epoch),
                context={
                    "task": task,
                    "epoch": int(epoch),
                    "patience_left": int(patience_left),
                },
            )

    seconds = time.time() - t_start

    if best_state is None or best_head_payload is None:
        fallback_best_epoch = best_epoch if best_epoch > 0 else last_epoch
        fallback_test = last_test if not math.isnan(last_test) else _top1(model, test_loader.loader, str(dev))
        best_state, best_head_payload = _build_checkpoint_payload(
            epoch_i=fallback_best_epoch,
            val_acc_i=last_val,
            test_acc_i=float(fallback_test),
            kind="best_ep",
        )

    best_ckpt_path = task_dir / f"{ckpt_stem}_best_ep.pt"
    torch.save(best_state, best_ckpt_path)

    last_ckpt_path: Path | None = None
    if save_last_epoch:
        if last_epoch <= 0:
            last_epoch = epochs
        last_state, _ = _build_checkpoint_payload(
            epoch_i=last_epoch,
            val_acc_i=last_val,
            test_acc_i=last_test,
            kind="last_ep",
        )
        last_ckpt_path = task_dir / f"{ckpt_stem}_last_ep.pt"
        torch.save(last_state, last_ckpt_path)

    summary = {
        "task": task,
        "strategy": strategy,
        "forward_mode": forward_mode,
        "save_format": save_format,
        "save_last_epoch": bool(save_last_epoch),
        "ckpt_path": str(best_ckpt_path),
        "best_ckpt_path": str(best_ckpt_path),
        "last_ckpt_path": str(last_ckpt_path) if last_ckpt_path is not None else None,
        "metrics": best_state.get("metrics", {}),
        "seconds": float(seconds),
        "trainable": trainable_info,
        "best_epoch": int(best_state.get("best_epoch", -1)),
        "last_epoch": int(last_epoch),
        "meta": task_meta,
        "hparams": {
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "optimizer": str(optimizer_name),
            "warmup_length": int(warmup_length),
            "clip_grad_norm": float(clip_grad_norm),
            "accumulate_grad_batches": int(accumulate_grad_batches),
            "batch_size": int(batch_size),
            "effective_batch_size": int(batch_size * accumulate_grad_batches),
            "num_workers": int(num_workers),
            "max_length": int(max_length),
            "seed": int(seed),
        },
    }
    _save_json(task_dir / f"{ckpt_stem}.json", summary)

    print(f"[{task}] saved best: {best_ckpt_path}")
    if last_ckpt_path is not None:
        print(f"[{task}] saved last: {last_ckpt_path}")
    if run_logger is not None:
        run_logger.log_event(
            "task_end",
            metrics={
                f"val/{task}/top1": float(summary["metrics"].get("val_top1", float("nan"))),
                f"test/{task}/top1": float(summary["metrics"].get("test_top1", float("nan"))),
                f"train/{task}/seconds": float(summary["seconds"]),
            },
            context={
                "task": task,
                "summary": summary,
            },
        )

    return summary, best_head_payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fine-tune text sequence-classification models from a config file (YAML/JSON).")

    g = p.add_argument_group("Config")
    g.add_argument("--text-config", type=str, required=True, help="Path to text config (.yaml/.yml/.json).")

    g = p.add_argument_group("Task selection overrides (optional)")
    g.add_argument("--suite", type=str, default=None, choices=sorted(NLI_SUITES.keys()))
    g.add_argument("--tasks", type=str, default=None, help="Comma-separated task names (overrides suite/order).")

    g = p.add_argument_group("Runtime overrides (optional)")
    g.add_argument("--device", type=str, default=None, help="Override config device, e.g. cuda, cuda:0, cpu.")
    add_logging_args(p)

    return p


def resolve_tasks(args, cfg_file: dict[str, Any]) -> list[str]:
    if args.tasks and args.tasks.strip():
        tasks = [str(x).strip().lower() for x in parse_csv(args.tasks)]
        return tasks
    if args.suite is not None:
        return list(NLI_SUITES[args.suite])

    tasks = _resolve_tasks_from_cfg(cfg_file)
    return tasks if tasks is not None else list(NLI_SUITES["nli6"])


def main() -> None:
    run_logger = None
    try:
        parser = build_parser()
        args = parser.parse_args()

        cfg_file = _load_config(args.text_config)
        common = _get_common_cfg(cfg_file)

        tasks = resolve_tasks(args, cfg_file)

        global_cfg = deepcopy(common)

        backbone_name = str(_get(global_cfg, "backbone.name", "hf_text"))
        if backbone_name != "hf_text":
            raise ValueError(f"Unsupported backbone '{backbone_name}' (only hf_text is supported).")

        model_name_or_path = _get(global_cfg, "backbone.model_name_or_path", None)
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ValueError("common.backbone.model_name_or_path is required.")

        model_arch = str(_get(global_cfg, "backbone.model_arch", "auto"))
        model_kind = str(_get(global_cfg, "backbone.model_kind", "sequence_classification"))
        if model_kind != "sequence_classification":
            raise ValueError("train_text currently requires common.backbone.model_kind='sequence_classification'.")

        trust_remote_code = bool(_get(global_cfg, "backbone.trust_remote_code", False))
        use_fast_tokenizer = bool(_get(global_cfg, "backbone.use_fast_tokenizer", True))

        device = str(args.device) if args.device is not None else str(_get(global_cfg, "device", "cuda"))
        dtype = _get(global_cfg, "dtype", None)
        deterministic = bool(_get(global_cfg, "deterministic", False))

        out_dir = Path(_get(global_cfg, "output.out_dir", "src/checkpoints/finetune_text"))
        save_format_default = str(_get(global_cfg, "output.save_format", "full"))
        save_last_epoch_default = bool(_get(global_cfg, "output.save_last_epoch", False))
        extract_heads_default = bool(_get(global_cfg, "output.extract_heads", False))
        heads_path_default = _get(global_cfg, "output.heads_path", None)
        logging_cfg = merge_logging_config(_get(global_cfg, "logging", {}), build_logging_overrides(args))

        model_tag = _safe_model_tag(model_name_or_path)
        run_ts = int(time.time())
        run_path = default_summary_path(
            entrypoint="finetune.train_text",
            logging_cfg=logging_cfg,
            default_parent=out_dir / model_tag,
            timestamp=run_ts,
        )
        startup_cfg = deepcopy(common)
        startup_cfg["config"] = args.text_config
        startup_cfg["tasks"] = list(tasks)
        startup_cfg["device"] = device
        startup_cfg["dtype"] = dtype
        startup_cfg["deterministic"] = deterministic
        startup_cfg["logging"] = logging_cfg
        startup_cfg["summary"] = str(run_path)
        startup_cfg.setdefault("backbone", {})
        startup_cfg["backbone"]["name"] = backbone_name
        startup_cfg["backbone"]["model_name_or_path"] = model_name_or_path
        startup_cfg["backbone"]["model_arch"] = model_arch
        startup_cfg["backbone"]["model_kind"] = model_kind
        startup_cfg["backbone"]["trust_remote_code"] = trust_remote_code
        startup_cfg["backbone"]["use_fast_tokenizer"] = use_fast_tokenizer
        startup_cfg.setdefault("output", {})
        startup_cfg["output"]["out_dir"] = str(out_dir)
        startup_cfg["output"]["save_format"] = save_format_default
        startup_cfg["output"]["save_last_epoch"] = save_last_epoch_default
        startup_cfg["output"]["extract_heads"] = extract_heads_default
        startup_cfg["output"]["heads_path"] = heads_path_default

        all_summaries: dict[str, Any] = {
            "config_path": args.text_config,
            "common": common,
            "cli": {
                "suite": args.suite,
                "tasks": args.tasks,
                "device": args.device,
                "logging": build_logging_overrides(args),
            },
            "resolved": {
                "tasks": tasks,
                "build_cfg": {
                    "backbone": backbone_name,
                    "model_name_or_path": model_name_or_path,
                    "model_arch": model_arch,
                    "model_kind": model_kind,
                    "dtype": dtype,
                    "device": device,
                },
                "run_path": str(run_path),
            },
            "results": {},
        }
        run_logger = start_run(
            entrypoint="finetune.train_text",
            logging_cfg=logging_cfg,
            summary_path=run_path,
            metadata={
                "config_path": args.text_config,
                "summary_path": str(run_path),
                "resolved_config": startup_cfg,
            },
        )

        extracted_heads: dict[str, dict[str, torch.Tensor]] = {}

        for task in tasks:
            task = str(task).strip().lower()
            if task not in NLI_TASKS:
                raise ValueError(f"Unknown task '{task}'. Supported: {list(NLI_TASKS)}")

            task_cfg = deepcopy(common)
            _deep_update(task_cfg, _get_dataset_override(cfg_file, task))
            task_logging_cfg = merge_logging_config(_get(task_cfg, "logging", {}), build_logging_overrides(args))

            epochs = _get(task_cfg, "train.epochs", None)
            if epochs is None:
                raise ValueError(f"[{task}] train.epochs missing. Set common.train.epochs or datasets.{task}.train.epochs.")
            epochs = int(epochs)

            strategy_cfg = _get(task_cfg, "strategy", {})
            if not isinstance(strategy_cfg, dict):
                raise ValueError(f"[{task}] strategy must be a dict.")
            resolve_training_forward_mode(strategy_cfg)
            strategy = str(_get(task_cfg, "strategy.name", "full"))
            if strategy not in {"full", "linear_probe", "peft_lora"}:
                raise ValueError(f"[{task}] Unsupported strategy '{strategy}'. Use one of: full, linear_probe, peft_lora")

            optimizer_name = str(_get(task_cfg, "train.optimizer.name", "adamw"))
            lr = float(_get(task_cfg, "train.lr", 1e-4))
            weight_decay = float(_get(task_cfg, "train.weight_decay", 0.0))
            warmup_length = int(_get(task_cfg, "train.lr_scheduler.warmup_steps", 500))
            scheduler_name = str(_get(task_cfg, "train.lr_scheduler.name", "cosine"))
            clip_grad_norm = float(_get(task_cfg, "train.grad_clip_norm", 1.0))
            accumulate_grad_batches = int(_get(task_cfg, "train.accumulate_grad_batches", 1))
            if accumulate_grad_batches <= 0:
                raise ValueError(f"[{task}] train.accumulate_grad_batches must be >= 1.")

            batch_size = int(_get(task_cfg, "data.batch_size", 8))
            num_workers = int(_get(task_cfg, "data.num_workers", 0))
            max_length = int(_get(task_cfg, "data.max_length", 512))
            task_num_labels = int(len(build_nli_task_data(task=task, split="train", max_samples=1).labels))
            cfg_head_num_labels = int(
                _get(task_cfg, "backbone.num_labels", _get(global_cfg, "backbone.num_labels", task_num_labels))
            )
            if cfg_head_num_labels != task_num_labels:
                print(
                    f"[{task}] overriding backbone.num_labels from {cfg_head_num_labels} "
                    f"to {task_num_labels} to match dataset labels."
                )
            head_num_labels = int(task_num_labels)

            seed = int(_get(task_cfg, "seed", 42))
            early_stopping = bool(_get(task_cfg, "train.early_stopping", False))
            early_stopping_patience = int(_get(task_cfg, "train.early_stopping_patience", 5))

            task_out_dir = Path(_get(task_cfg, "output.out_dir", str(out_dir)))
            save_format = str(_get(task_cfg, "output.save_format", save_format_default))
            save_last_epoch = bool(_get(task_cfg, "output.save_last_epoch", save_last_epoch_default))
            extract_heads = bool(_get(task_cfg, "output.extract_heads", extract_heads_default))

            if save_format not in {"full", "head", "peft"}:
                raise ValueError(f"[{task}] output.save_format must be one of: full, head, peft")
            if save_format == "peft" and strategy != "peft_lora":
                raise ValueError(f"[{task}] save_format='peft' requires strategy.name='peft_lora'.")

            build_cfg = TextBuildConfig(
                model_name_or_path=str(model_name_or_path),
                model_arch=str(_get(task_cfg, "backbone.model_arch", model_arch)),
                device=str(device),
                dtype=_get(task_cfg, "dtype", dtype),
                model_kind=str(_get(task_cfg, "backbone.model_kind", model_kind)),
                num_labels=int(head_num_labels),
                trust_remote_code=bool(_get(task_cfg, "backbone.trust_remote_code", trust_remote_code)),
                use_fast_tokenizer=bool(_get(task_cfg, "backbone.use_fast_tokenizer", use_fast_tokenizer)),
            )

            summary, head_payload = train_task(
                task=task,
                build_cfg=build_cfg,
                strategy=strategy,
                strategy_cfg=strategy_cfg,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                warmup_length=warmup_length,
                scheduler_name=scheduler_name,
                optimizer_name=optimizer_name,
                clip_grad_norm=clip_grad_norm,
                accumulate_grad_batches=accumulate_grad_batches,
                batch_size=batch_size,
                num_workers=num_workers,
                max_length=max_length,
                head_num_labels=head_num_labels,
                early_stopping=early_stopping,
                early_stopping_patience=early_stopping_patience,
                seed=seed,
                deterministic=deterministic,
                device=str(device),
                out_dir=task_out_dir,
                save_format=save_format,
                save_last_epoch=save_last_epoch,
                task_cfg=task_cfg,
                log_every_n_steps=int(task_logging_cfg.get("log_every_n_steps", 50)),
                run_logger=run_logger,
            )

            all_summaries["results"][task] = summary
            if extract_heads:
                extracted_heads[task] = head_payload

        _save_json(run_path, all_summaries)
        run_logger.log_summary(all_summaries)
        print(f"\nSaved run summary: {run_path}")

        if extracted_heads:
            heads_path_raw = heads_path_default
            if isinstance(heads_path_raw, str) and heads_path_raw.strip():
                heads_path = Path(heads_path_raw)
            else:
                heads_path = out_dir / model_tag / "heads.pt"
            _ensure_dir(heads_path.parent)
            torch.save(extracted_heads, heads_path)
            print(f"Saved extracted task heads: {heads_path}")
            run_logger.log_event(
                "artifact_saved",
                metrics={},
                context={
                    "artifact": "heads.pt",
                    "path": str(heads_path),
                },
            )
        run_logger.finish("success")
    except Exception as exc:
        finish_with_error(run_logger, exc)
        raise


if __name__ == "__main__":
    main()
