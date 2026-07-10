import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from merge_and_rebase.io.utils import read_json_silent

_DEFAULT_ATTN_PATCH_CFG: dict[str, Any] = {
    "attn_impl": "softmax",
    "kernel": "elu_plus_one",
    "eps": 1e-6,
    "linear_rule": "kernel",
    "delta_eta": 1.0,
    "delta_exclude_cls_from_store": True,
    "delta_cls_only_readout": False,
    "delta_learn_w0": False,
    "delta_w0_rank": 0,
}
_SPLIT_ATTN_MARKERS = (
    ".attn.q_proj.",
    ".attn.k_proj.",
    ".attn.v_proj.",
    ".attn.out_proj.",
)
_FUSED_ATTN_MARKERS = (
    ".attn.in_proj_weight",
    ".attn.in_proj_bias",
)
_ADAPTER_REF_SUFFIXES = (".pt", ".bin", ".safetensors", ".ckpt", ".pth")
_HF_ADAPTER_DIR_CACHE: dict[str, str] = {}
_PATCHED_ATTN_TARGET_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "out_proj"})


def is_peft_adapter_dir_ckpt(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("format") == "peft" and isinstance(obj.get("peft_adapter_dir"), str)


def is_peft_adapter_reference(ref: str) -> bool:
    path = Path(ref)
    if path.exists():
        return path.is_dir() and (path / "adapter_config.json").exists()
    return "/" in ref and not ref.endswith(_ADAPTER_REF_SUFFIXES)


def resolve_peft_adapter_dir(ref: str) -> str:
    path = Path(ref)
    if path.exists():
        if path.is_dir() and (path / "adapter_config.json").exists():
            return str(path)
        raise ValueError(f"PEFT adapter reference is not an adapter directory: {ref}")

    if not is_peft_adapter_reference(ref):
        raise ValueError(f"Not a PEFT adapter reference: {ref}")

    if ref not in _HF_ADAPTER_DIR_CACHE:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise ImportError("Resolving Hugging Face PEFT adapters requires `huggingface_hub`.") from exc

        _HF_ADAPTER_DIR_CACHE[ref] = snapshot_download(
            repo_id=ref,
            repo_type="model",
            allow_patterns=(
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
                "merge_and_rebase_meta.json",
            ),
        )
    return _HF_ADAPTER_DIR_CACHE[ref]


def is_peft_adapter_reference(ref: str) -> bool:
    path = Path(ref)
    if path.exists():
        return path.is_dir() and (path / "adapter_config.json").exists()
    return "/" in ref and not ref.endswith(_ADAPTER_REF_SUFFIXES)


def resolve_peft_adapter_dir(ref: str) -> str:
    path = Path(ref)
    if path.exists():
        if path.is_dir() and (path / "adapter_config.json").exists():
            return str(path)
        raise ValueError(f"PEFT adapter reference is not an adapter directory: {ref}")

    if not is_peft_adapter_reference(ref):
        raise ValueError(f"Not a PEFT adapter reference: {ref}")

    if ref not in _HF_ADAPTER_DIR_CACHE:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise ImportError("Resolving Hugging Face PEFT adapters requires `huggingface_hub`.") from exc

        _HF_ADAPTER_DIR_CACHE[ref] = snapshot_download(
            repo_id=ref,
            repo_type="model",
            allow_patterns=(
                "adapter_config.json",
                "adapter_model.safetensors",
                "adapter_model.bin",
                "merge_and_rebase_meta.json",
            ),
        )
    return _HF_ADAPTER_DIR_CACHE[ref]



def resolve_peft_adapter_dir(adapter_dir: str, *, checkpoint_path: str | None = None) -> Path:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add_candidate(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    _add_candidate(Path(adapter_dir))
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        _add_candidate(ckpt_path.with_name(f"{ckpt_path.stem}_adapter"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"PEFT adapter_dir not found. Tried: {tried}")


def normalize_peft_adapter_dir_checkpoint(
    obj: Any,
    *,
    checkpoint_path: str | None = None,
) -> Any:
    if not is_peft_adapter_dir_ckpt(obj):
        return obj

    resolved = resolve_peft_adapter_dir(str(obj["peft_adapter_dir"]), checkpoint_path=checkpoint_path)
    if str(resolved) == str(obj["peft_adapter_dir"]):
        return obj

    normalized = dict(obj)
    normalized["peft_adapter_dir"] = str(resolved)
    return normalized


def load_peft_adapter_dir_components(
    adapter_dir: str,
    *,
    checkpoint_path: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Returns (peft_state, peft_cfg_map) compatible with your existing helpers:
      - peft_state: state dict of adapter params (cpu tensors)
      - peft_cfg_map: dict like {"default": <adapter_config_dict>}
    """
    ad = resolve_peft_adapter_dir(adapter_dir, checkpoint_path=checkpoint_path)

    # 1) adapter config
    cfg_path = ad / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {ad}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"adapter_config.json is not a dict: {cfg_path}")

    # 2) adapter weights
    # PEFT commonly writes either:
    #  - adapter_model.safetensors
    #  - adapter_model.bin (older)
    st_path = ad / "adapter_model.safetensors"
    bin_path = ad / "adapter_model.bin"

    if st_path.exists():
        try:
            from safetensors.torch import load_file as _st_load_file  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Found adapter_model.safetensors but safetensors is not installed. "
                "Install `safetensors` or save adapters as .bin."
            ) from e
        peft_state = _st_load_file(str(st_path))
    elif bin_path.exists():
        peft_state = torch.load(str(bin_path), map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No adapter weights found in {ad} (expected adapter_model.safetensors or .bin)")

    if not isinstance(peft_state, dict):
        raise ValueError(f"Adapter weights are not a dict in {ad}")

    # ensure CPU tensors
    peft_state = {k: v.detach().cpu() for k, v in peft_state.items() if torch.is_tensor(v)}
    peft_state = normalize_peft_visual_state_dict_keys(peft_state)

    # Your downstream expects a map of adapter-name -> config-dict
    peft_cfg_map = {"default": cfg_dict}
    return peft_state, peft_cfg_map


def infer_target_modules_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    prefixes: set[str] = set()
    for raw_key in state_dict:
        key = str(raw_key)
        matched = False
        for suffix in (
            ".lora_A.weight",
            ".lora_A.default.weight",
            ".lora_B.weight",
            ".lora_B.default.weight",
        ):
            if key.endswith(suffix):
                prefix = key[: -len(suffix)]
                if prefix.startswith("base_model.model."):
                    prefix = prefix[len("base_model.model.") :]
                prefixes.add(prefix)
                matched = True
                break
        if not matched:
            continue
    return sorted(prefixes)


def save_peft_adapter_dir(
    output_dir: str | Path,
    *,
    state_dict: Mapping[str, torch.Tensor],
    adapter_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
    diagnostics_key: str | None = None,
) -> Path:
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - dependency error
        raise ImportError("Saving merged adapters requires safetensors.") from exc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_to_write = dict(adapter_config)
    inferred_target_modules = infer_target_modules_from_state_dict(state_dict)
    if inferred_target_modules:
        config_to_write["target_modules"] = inferred_target_modules
    with (out / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_to_write, handle, indent=2, sort_keys=True)
        handle.write("\n")
    save_file({k: v.detach().cpu().contiguous() for k, v in state_dict.items()}, str(out / "adapter_model.safetensors"))
    metadata = {"format": "peft", "peft_target": "mllm", **dict(provenance)}
    if diagnostics is not None:
        metadata[str(diagnostics_key or "merge_diagnostics")] = dict(diagnostics)
    with (out / "merge_and_rebase_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out


def _extract_target_modules_from_cfg(cfg: Any) -> set[str]:
    if not isinstance(cfg, Mapping):
        return set()
    raw = cfg.get("target_modules", None)
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {str(item) for item in raw}


def _ckpt_obj_target_modules(ckpt_obj: dict[str, Any]) -> set[str]:
    targets = _extract_target_modules_from_cfg(ckpt_obj.get("peft_cfg", None))
    if targets:
        return targets

    peft_cfg_map = ckpt_obj.get("peft_config", None)
    if isinstance(peft_cfg_map, Mapping):
        for cfg in peft_cfg_map.values():
            targets = _extract_target_modules_from_cfg(cfg)
            if targets:
                return targets

    ad = ckpt_obj.get("peft_adapter_dir", None)
    if isinstance(ad, str):
        cfg_path = Path(ad) / "adapter_config.json"
        cfg = read_json_silent(str(cfg_path))
        targets = _extract_target_modules_from_cfg(cfg)
        if targets:
            return targets
    return set()


def normalize_peft_visual_state_dict_keys(peft_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Normalize vision adapter keyspaces across equivalent backbones.

    Some Hugging Face ViT adapters use encoder/self_attn names, while this repo's
    OpenCLIP patched attention uses transformer/resblocks/attn names.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in peft_state.items():
        new_key = str(key)
        if ".encoder.layers." in new_key and ".self_attn." in new_key:
            new_key = new_key.replace("base_model.model.encoder.layers.", "base_model.model.transformer.resblocks.")
            new_key = new_key.replace(".self_attn.", ".attn.")
        out[new_key] = value
    return out


def get_patched_attn_flag(ckpt_obj: dict[str, Any]) -> bool:
    # Prefer explicit key in the .pt payload
    if "patched_attn" in ckpt_obj:
        return bool(ckpt_obj["patched_attn"])

    # Fallback: read your meta json inside adapter dir
    ad = ckpt_obj.get("peft_adapter_dir", None)
    if isinstance(ad, str):
        meta = read_json_silent(str(Path(ad) / "merge_and_rebase_meta.json"))
        if "patched_attn" in meta:
            return bool(meta["patched_attn"])

    targets = _ckpt_obj_target_modules(ckpt_obj)
    if _PATCHED_ATTN_TARGET_MODULES.issubset(targets):
        return True
    return False


def normalize_attn_patch_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(cfg or {})
    attn_impl = str(raw.get("attn_impl", _DEFAULT_ATTN_PATCH_CFG["attn_impl"])).strip().lower()
    if attn_impl not in {"softmax", "linear"}:
        raise ValueError(f"Unknown attn_impl '{attn_impl}'. Choose from: softmax, linear")
    linear_rule = str(raw.get("linear_rule", _DEFAULT_ATTN_PATCH_CFG["linear_rule"])).strip().lower()
    if linear_rule not in {"kernel", "delta"}:
        raise ValueError(f"Unknown linear_rule '{linear_rule}'. Choose from: kernel, delta")
    return {
        "attn_impl": attn_impl,
        "kernel": str(raw.get("kernel", _DEFAULT_ATTN_PATCH_CFG["kernel"])),
        "eps": float(raw.get("eps", _DEFAULT_ATTN_PATCH_CFG["eps"])),
        "linear_rule": linear_rule,
        "delta_eta": float(raw.get("delta_eta", _DEFAULT_ATTN_PATCH_CFG["delta_eta"])),
        "delta_exclude_cls_from_store": bool(
            raw.get("delta_exclude_cls_from_store", _DEFAULT_ATTN_PATCH_CFG["delta_exclude_cls_from_store"])
        ),
        "delta_cls_only_readout": bool(
            raw.get("delta_cls_only_readout", _DEFAULT_ATTN_PATCH_CFG["delta_cls_only_readout"])
        ),
        "delta_learn_w0": bool(raw.get("delta_learn_w0", _DEFAULT_ATTN_PATCH_CFG["delta_learn_w0"])),
        "delta_w0_rank": int(raw.get("delta_w0_rank", _DEFAULT_ATTN_PATCH_CFG["delta_w0_rank"])),
    }


def state_dict_looks_patched_attn(sd: Mapping[str, Any]) -> bool:
    keys = tuple(str(k) for k in sd.keys())
    has_split_attn = any(any(marker in k for marker in _SPLIT_ATTN_MARKERS) for k in keys)
    has_fused_attn = any(any(marker in k for marker in _FUSED_ATTN_MARKERS) for k in keys)
    return has_split_attn and not has_fused_attn


def get_attn_patch_cfg(ckpt_obj: dict[str, Any]) -> dict[str, Any]:
    cfg = ckpt_obj.get("attn_patch_cfg", None)
    if isinstance(cfg, dict):
        return normalize_attn_patch_cfg(cfg)

    ad = ckpt_obj.get("peft_adapter_dir", None)
    if isinstance(ad, str):
        meta = read_json_silent(str(Path(ad) / "merge_and_rebase_meta.json"))
        cfg2 = meta.get("attn_patch_cfg", None)
        if isinstance(cfg2, dict):
            return normalize_attn_patch_cfg(cfg2)

    return normalize_attn_patch_cfg(_DEFAULT_ATTN_PATCH_CFG)
