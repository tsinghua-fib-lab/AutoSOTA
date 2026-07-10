from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model

from merge_and_rebase.finetune.strategies.base import build_optimizer
from merge_and_rebase.finetune.schedulers import build_lr_scheduler
from merge_and_rebase.models.parameter_subsets import resolve_visual_parameter_plan
from merge_and_rebase.models.patch_openclip_attention import LoRAableLinearMHA, LoRAableMHA, split_openclip_vit_attn
from merge_and_rebase.models.patch_openclip_projection import has_lora_compatible_proj_surface, patch_openclip_visual_proj

from ._delta_parameterization import bind_delta_parameterization
from .registry import register


def infer_linear_leaf_names(m: nn.Module) -> set[str]:
    leafs = set()
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            leafs.add(name.split(".")[-1])
    return leafs


def _resolve_attn_patch_cfg(
    *,
    peft_cfg: dict[str, Any],
    strategy_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if isinstance(strategy_cfg, dict):
        attention_cfg = strategy_cfg.get("attention", {})
        if attention_cfg is None:
            attention_cfg = {}
        if not isinstance(attention_cfg, dict):
            raise ValueError("strategy.attention must be a dict when provided.")
        raw.update(attention_cfg)

    # Backward-compat convenience: allow these keys under strategy.peft too.
    for k in (
        "attn_impl",
        "kernel",
        "eps",
        "ramp_fraction",
        "linear_rule",
        "delta_eta",
        "delta_exclude_cls_from_store",
        "delta_cls_only_readout",
        "delta_learn_w0",
        "delta_w0_rank",
    ):
        if k in peft_cfg and k not in raw:
            raw[k] = peft_cfg[k]

    attn_impl = str(raw.get("attn_impl", "softmax"))
    ramp_fraction_default = 0.2 if attn_impl == "linear" else 0.0
    ramp_fraction = float(raw.get("ramp_fraction", ramp_fraction_default))
    if ramp_fraction < 0.0 or ramp_fraction > 1.0:
        raise ValueError("attention.ramp_fraction must be in [0, 1].")

    return {
        "attn_impl": attn_impl,
        "kernel": str(raw.get("kernel", "elu_plus_one")),
        "eps": float(raw.get("eps", 1e-6)),
        "ramp_fraction": ramp_fraction,
        "linear_rule": str(raw.get("linear_rule", "kernel")),
        "delta_eta": float(raw.get("delta_eta", 1.0)),
        "delta_exclude_cls_from_store": bool(raw.get("delta_exclude_cls_from_store", True)),
        "delta_cls_only_readout": bool(raw.get("delta_cls_only_readout", False)),
        "delta_learn_w0": bool(raw.get("delta_learn_w0", False)),
        "delta_w0_rank": int(raw.get("delta_w0_rank", 0)),
    }


def _visual_has_lora_compatible_attn_surface(visual: nn.Module) -> bool:
    transformer = getattr(visual, "transformer", None)
    if transformer is None:
        return False
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        return False

    saw_block = False
    for blk in resblocks:
        saw_block = True
        attn = getattr(blk, "attn", None)
        if not isinstance(attn, (LoRAableMHA, LoRAableLinearMHA)):
            return False
        if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "out_proj")):
            return False
    return saw_block


def _cfg_value(strategy_cfg: dict | None, key: str, default: Any) -> Any:
    if not isinstance(strategy_cfg, dict):
        return default
    params = strategy_cfg.get("params", None)
    if isinstance(params, dict) and key in params:
        return params[key]
    return strategy_cfg.get(key, default)


def _resolve_target_modules(
    *,
    peft_cfg: dict[str, Any],
    strategy_cfg: dict[str, Any] | None,
    visual: nn.Module,
) -> tuple[list[str], bool]:
    raw = peft_cfg.get("target_modules", None)
    if raw is None or (isinstance(raw, str) and str(raw).strip().lower() == "auto"):
        trainable_params_mode = str(_cfg_value(strategy_cfg, "trainable_params", "all_trainable")).strip().lower()
        if trainable_params_mode != "regularized_only":
            raise ValueError("strategy.peft.target_modules is required unless strategy.params.trainable_params='regularized_only'.")
        plan = resolve_visual_parameter_plan(visual, "regularized_only")
        if not plan.supported or not plan.lora_target_modules:
            raise ValueError("Could not derive LoRA target_modules from strategy.params.trainable_params='regularized_only'.")
        return list(plan.lora_target_modules), True
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise ValueError("strategy.peft.target_modules must be a list[str], 'auto', or omitted.")
    return [str(x) for x in raw], False


@dataclass(frozen=True)
class PeftLoraVision:
    """
    Apply LoRA to *vision encoder only* (OpenCLIP visual), keeping everything else frozen
    except optionally logit_scale and/or other explicitly allowed params.

    Expected model layout:
      model.clip_model.model.visual  -> vision module (patched with PEFT)
      model.clip_model.model.transformer -> text encoder (kept frozen)
      model.clip_model.logit_scale -> scalar parameter (optionally trainable)

    Config expected in vision yaml:
      strategy:
        name: peft_lora
        peft:
          r: 16
          lora_alpha: 16
          lora_dropout: 0.0
          bias: "none"
          target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]  # REQUIRED: list of module names to target with LoRA
          train_logit_scale: false                    # optional
    """

    name: str = "peft_lora"

    def configure(
        self,
        *,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        dense_lr: float | None = None,
        warmup_length: int,
        scheduler_name: str = "cosine",
        optimizer: str = "adamw",
        steps: int,
        device: torch.device,
        peft_cfg: dict[str, Any] | None = None,
        strategy_cfg: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, Any]]:
        if peft_cfg is None:
            peft_cfg = {}

        # --- locate OpenCLIP pieces ---
        clip = getattr(model, "clip_model", None)
        if clip is None:
            raise ValueError("PeftLoraVision expects `model.clip_model` (your OpenClipClassifier).")

        if not hasattr(clip, "model"):
            raise ValueError("PeftLoraVision expects `model.clip_model.model` (OpenCLIP model).")

        openclip_model = clip.model
        if not hasattr(openclip_model, "visual"):
            raise ValueError("PeftLoraVision expects `model.clip_model.model.visual` to exist.")

        visual = model.clip_model.model.visual

        # Optionally train logit_scale
        train_logit_scale = bool(peft_cfg.get("train_logit_scale", False))
        if train_logit_scale and hasattr(clip, "logit_scale"):
            try:
                clip.logit_scale.requires_grad_(True)
            except Exception:
                pass

        parameterization = str(_cfg_value(strategy_cfg, "parameterization", "weights")).strip().lower()
        if parameterization not in {"weights", "delta"}:
            raise ValueError("strategy.params.parameterization must be one of: weights, delta")

        trainable_params_mode = str(_cfg_value(strategy_cfg, "trainable_params", "all_trainable")).strip().lower()
        if trainable_params_mode not in {"all_trainable", "regularized_only"}:
            raise ValueError("strategy.params.trainable_params must be one of: all_trainable, regularized_only")
        dense_lr_value = float(lr if dense_lr is None else dense_lr)

        # --- build LoRA config ---
        target_modules, target_modules_auto = _resolve_target_modules(
            peft_cfg=peft_cfg,
            strategy_cfg=strategy_cfg,
            visual=visual,
        )

        # leafs = infer_linear_leaf_names(clip.model.visual)
        # print("Linear leaf names:", sorted(leafs))

        attn_patch_cfg: dict[str, Any] | None = None
        if any(tm in ("q_proj", "k_proj", "v_proj", "out_proj") for tm in target_modules):
            attn_patch_cfg = _resolve_attn_patch_cfg(peft_cfg=peft_cfg, strategy_cfg=strategy_cfg)

        patched_proj = 0
        if "lin_proj" in target_modules:
            patched_proj = patch_openclip_visual_proj(visual)
            if patched_proj == 0 and not has_lora_compatible_proj_surface(visual):
                raise RuntimeError(
                    "No OpenCLIP projection module was patched, and the existing visual projection "
                    "surface is not LoRA-compatible. Check that the model is a ViT with `visual.proj`."
                )

        modules_to_save = peft_cfg.get("modules_to_save", None)
        if modules_to_save is not None and (not isinstance(modules_to_save, list) or not all(isinstance(x, str) for x in modules_to_save)):
            raise ValueError("strategy.peft.modules_to_save must be a list[str] when provided.")
        mts = list(modules_to_save or [])
        if isinstance(attn_patch_cfg, dict) and bool(attn_patch_cfg.get("delta_learn_w0", False)):
            if "delta_mem" not in mts:
                mts.append("delta_mem")

        resolved_peft_cfg = dict(peft_cfg)
        resolved_peft_cfg["target_modules"] = list(target_modules)
        lora_cfg = LoraConfig(
            r=int(resolved_peft_cfg.get("r", 16)),
            lora_alpha=int(resolved_peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(resolved_peft_cfg.get("lora_dropout", 0.0)),
            target_modules=target_modules,
            bias=str(resolved_peft_cfg.get("bias", "none")),
            modules_to_save=mts if mts else None,
        )

        # if target_modules contains "qkv" or "proj", we need to patch attention modules
        if any(tm in ("q_proj", "k_proj", "v_proj", "out_proj") for tm in target_modules):
            assert attn_patch_cfg is not None
            ramp_steps = int(round(float(attn_patch_cfg.get("ramp_fraction", 0.0)) * max(1, int(steps))))
            attn_patch_cfg["ramp_steps"] = int(ramp_steps)
            n = split_openclip_vit_attn(
                visual,
                proj_dropout=0.0,
                attn_impl=attn_patch_cfg["attn_impl"],
                kernel=attn_patch_cfg["kernel"],
                eps=attn_patch_cfg["eps"],
                ramp_steps=ramp_steps,
                linear_rule=str(attn_patch_cfg.get("linear_rule", "kernel")),
                delta_eta=float(attn_patch_cfg.get("delta_eta", 1.0)),
                delta_exclude_cls_from_store=bool(attn_patch_cfg.get("delta_exclude_cls_from_store", True)),
                delta_cls_only_readout=bool(attn_patch_cfg.get("delta_cls_only_readout", False)),
                delta_learn_w0=bool(attn_patch_cfg.get("delta_learn_w0", False)),
                delta_w0_rank=int(attn_patch_cfg.get("delta_w0_rank", 0)),
            )
            if n == 0 and not _visual_has_lora_compatible_attn_surface(visual):
                raise RuntimeError(
                    "No OpenCLIP attention modules were patched, and the existing visual attention "
                    "surface is not LoRA-compatible. Check that the model is a ViT and "
                    "target_modules are correct."
                )
            model.peft_patched_attn = True  # type: ignore[attr-defined]
            model.peft_attn_patch_cfg = attn_patch_cfg  # type: ignore[attr-defined]

        # --- freeze everything by default ---
        for p in model.parameters():
            p.requires_grad = False

        # Ensure text encoder stays frozen
        if hasattr(openclip_model, "transformer"):
            for p in openclip_model.transformer.parameters():
                p.requires_grad = False

        # --- wrap ONLY visual with PEFT ---
        peft_visual = get_peft_model(visual, lora_cfg)
        openclip_model.visual = peft_visual
        model.to(device)

        if train_logit_scale and hasattr(clip, "logit_scale"):
            clip.logit_scale.requires_grad_(True)

        dense_trainable_names: tuple[str, ...] = ()
        params_mode_supported = trainable_params_mode == "all_trainable"
        if trainable_params_mode == "regularized_only":
            wrapped_plan = resolve_visual_parameter_plan(model.clip_model.model.visual, "regularized_only")
            if wrapped_plan.supported:
                dense_trainable = set(wrapped_plan.dense_parameter_names)
                dense_trainable_names = wrapped_plan.dense_parameter_names
                for name, param in model.clip_model.model.visual.named_parameters():
                    if "lora_" in name.lower():
                        continue
                    param.requires_grad_(name in dense_trainable)
                params_mode_supported = True

        dense_visual_trainable_keys = tuple(
            name
            for name, param in model.clip_model.model.visual.named_parameters()
            if param.requires_grad and "lora_" not in name.lower()
        )

        named_params = list(model.named_parameters())
        dense_visual_trainable_full_names = tuple(
            name
            for name, param in named_params
            if param.requires_grad and "lora_" not in name.lower() and name.startswith("clip_model.model.visual.")
        )
        dense_visual_trainable_full_name_set = set(dense_visual_trainable_full_names)
        lora_param_names = {
            name
            for name, param in named_params
            if param.requires_grad and "lora_" in name.lower()
        }
        other_trainable_param_names = {
            name
            for name, param in named_params
            if param.requires_grad and name not in lora_param_names and name not in dense_visual_trainable_full_name_set
        }

        # Ensure LoRA params exist and require_grad
        lora = [(n, p) for n, p in model.named_parameters() if "lora_" in n.lower()]
        print("n_lora_params:", len(lora), "any_trainable:", any(p.requires_grad for _, p in lora))

        # Double-check we really injected something trainable
        lora_trainables = [p for p in model.clip_model.model.visual.parameters() if p.requires_grad]
        if len(lora_trainables) == 0:
            # If you hit this, target_modules likely didn't match the module names.
            names = [n for n, _ in model.clip_model.model.visual.named_modules()]
            raise RuntimeError(
                "No trainable LoRA parameters were created. "
                "Likely target_modules mismatch for OpenCLIP visual.\n"
                f"Example visual submodule names (first 40): {names[:40]}"
            )

        model.to(device)

        dense_delta_params: list[nn.Parameter] = []
        if parameterization == "delta" and dense_visual_trainable_full_names:
            dense_delta_params = bind_delta_parameterization(
                model=model,
                named_params=named_params,
                target_names=dense_visual_trainable_full_name_set,
                device=device,
            )

        if parameterization == "delta":
            lora_trainable_params = [
                param for name, param in named_params if name in lora_param_names
            ]
            dense_trainable_params = list(dense_delta_params)
            other_trainable_params = [
                param for name, param in named_params if name in other_trainable_param_names
            ]
        else:
            lora_trainable_params = [
                param for name, param in named_params if name in lora_param_names
            ]
            dense_trainable_params = [
                param for name, param in named_params if name in dense_visual_trainable_full_name_set
            ]
            other_trainable_params = [
                param for name, param in named_params if name in other_trainable_param_names
            ]

        trainable_params = [*lora_trainable_params, *dense_trainable_params, *other_trainable_params]

        if not trainable_params:
            raise RuntimeError("No trainable parameters found (after LoRA injection).")

        optimizer_param_groups: list[dict[str, Any]] = []
        base_lrs: list[float] = []
        if lora_trainable_params:
            optimizer_param_groups.append(
                {
                    "name": "lora",
                    "params": lora_trainable_params,
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                }
            )
            base_lrs.append(float(lr))
        if dense_trainable_params:
            optimizer_param_groups.append(
                {
                    "name": "dense",
                    "params": dense_trainable_params,
                    "lr": float(dense_lr_value),
                    "weight_decay": float(weight_decay),
                }
            )
            base_lrs.append(float(dense_lr_value))
        if other_trainable_params:
            optimizer_param_groups.append(
                {
                    "name": "other",
                    "params": other_trainable_params,
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                }
            )
            base_lrs.append(float(lr))

        opt = build_optimizer(optimizer_param_groups, optimizer, lr, weight_decay)
        scheduler = build_lr_scheduler(
            opt,
            name=scheduler_name,
            base_lrs=base_lrs,
            warmup_length=warmup_length,
            steps=steps,
        )

        model.peft_cfg_resolved = resolved_peft_cfg  # type: ignore[attr-defined]
        model.peft_dense_trainable_visual_keys = dense_visual_trainable_keys  # type: ignore[attr-defined]
        model.peft_patched_proj = bool(patched_proj or has_lora_compatible_proj_surface(model.clip_model.model.visual))  # type: ignore[attr-defined]
        model.peft_trainable_plan = {  # type: ignore[attr-defined]
            "target_modules_auto": bool(target_modules_auto),
            "target_modules": list(target_modules),
            "dense_trainable_names": list(dense_trainable_names),
            "parameterization": parameterization,
            "trainable_params_mode": trainable_params_mode,
            "lr_lora": float(lr),
            "lr_dense": float(dense_lr_value),
            "optimizer_groups": [str(group.get("name", "")).strip() for group in optimizer_param_groups],
        }

        info: dict[str, Any] = {
            "trainable_params": sum(p.numel() for p in trainable_params),
            "lora_params": sum(
                param.numel() for name, param in named_params if name in lora_param_names
            ),
            "dense_trainable_params": sum(
                param.numel() for name, param in named_params if name in dense_visual_trainable_full_name_set
            ),
            "dense_delta_params": sum(param.numel() for param in dense_delta_params),
            "optimizer_group_count": len(optimizer_param_groups),
            "lora_group_params": sum(param.numel() for param in lora_trainable_params),
            "dense_group_params": sum(param.numel() for param in dense_trainable_params),
            "other_group_params": sum(param.numel() for param in other_trainable_params),
            "trainable_params_fallback": int(trainable_params_mode == "regularized_only" and not params_mode_supported),
        }
        info["parameterization"] = parameterization
        info["scheduler_name"] = scheduler_name
        info["lr_lora"] = float(lr)
        info["lr_dense"] = float(dense_lr_value)
        if train_logit_scale and hasattr(clip, "logit_scale"):
            info["logit_scale_params"] = int(clip.logit_scale.numel())
        if any(tm in ("q_proj", "k_proj", "v_proj", "out_proj") for tm in target_modules):
            info["attn_ramp_steps"] = int(attn_patch_cfg.get("ramp_steps", 0))
        info["patched_proj"] = int(bool(getattr(model, "peft_patched_proj", False)))

        return opt, scheduler, info


register(PeftLoraVision())
