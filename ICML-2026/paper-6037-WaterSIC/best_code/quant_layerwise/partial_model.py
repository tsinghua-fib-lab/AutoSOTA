from __future__ import annotations

from pathlib import Path

import torch

from quant_layerwise.methods.gptq import dequantize_gptq
from quant_layerwise.methods.zsic import dequantize_zsic
from quant_layerwise.storage import LayerArtifact, RunManifest


def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    mods = dict(model.named_modules())
    if module_name not in mods:
        raise KeyError(
            f"Module '{module_name}' not found in model.named_modules(). "
            "Double check naming (expected 'layers.<id>.<attention|feed_forward>.<wq|wk|...>')."
        )
    return mods[module_name]


def _set_module_weight(module: torch.nn.Module, W: torch.Tensor):
    if not hasattr(module, "weight"):
        raise AttributeError(f"Target module has no .weight attribute: {type(module)}")

    with torch.no_grad():
        module.weight.data = W.to(device=module.weight.device, dtype=module.weight.dtype)


def dequantize_artifact(artifact: LayerArtifact, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize artifact -> float weight in the *original* basis."""
    method = artifact.method.lower()
    payload = artifact.payload

    if method == "gptq":
        Qint = payload["Qint"]
        scales = payload["scales"]
        zeros = payload["zeros"]
        groupsize = int(payload["groupsize"])

        W_hat = dequantize_gptq(Qint, scales, zeros, groupsize=groupsize, dtype=dtype)

    elif method in ("zsic", "sic"):
        Z = payload["Z"]
        alpha = payload["alpha"]
        alpha_base = payload.get("alpha_base", None)
        zero_point = payload.get("zero_point", None)
        apply_tgamma = bool(payload.get("apply_tgamma", False))
        t_vec = payload.get("t_vec", None)
        g_vec = payload.get("g_vec", None)
        dead_indices = payload.get("dead_indices", None)
        n_original = payload.get("n_original", None)
        dead_row_indices = payload.get("dead_row_indices", None)
        a_original = payload.get("a_original", None)

        W_hat = dequantize_zsic(
            Z,
            alpha,
            alpha_base=alpha_base,
            zero_point=zero_point,
            apply_tgamma=apply_tgamma,
            t_vec=t_vec,
            g_vec=g_vec,
            dtype=dtype,
            dead_indices=dead_indices,
            n_original=n_original,
            dead_row_indices=dead_row_indices,
            a_original=a_original,
        )

    elif method == "fullprec":
        W_hat = payload["W_full"].to(dtype)

    else:
        raise ValueError(f"Unknown artifact method: {artifact.method!r}")

    return W_hat.to(device=device, dtype=dtype)


def apply_layer_artifact(model: torch.nn.Module, artifact: LayerArtifact):
    module = get_module_by_name(model, artifact.module_name)

    W_hat = dequantize_artifact(
        artifact,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )

    _set_module_weight(module, W_hat)


def load_and_apply_manifest(
    model: torch.nn.Module,
    run_dir: str | Path,
    *,
    rank: int = 0,
    map_location: str = "cpu",
) -> RunManifest:
    """Load run manifest and apply all currently-saved quantized layers.

    Args:
        rank: Which tensor-parallel rank to load artifacts for.
              For world_size==1 manifests (legacy), this is ignored.
              For world_size>1, each rank loads its shard artifact.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    manifest = RunManifest.load(manifest_path)

    for module_name in manifest.artifacts:
        relpath = manifest.artifact_relpath_for_rank(module_name, rank)
        art_path = run_dir / relpath
        art = LayerArtifact.load(art_path, map_location=map_location)
        apply_layer_artifact(model, art)

    return manifest
