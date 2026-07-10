from __future__ import annotations

from typing import Any


def resolve_rebase_method_config(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    method_name = str(cfg.get("method", "gradfix"))
    method_params = cfg.get("method_params", {})
    if method_params is None:
        method_params = {}
    if not isinstance(method_params, dict):
        raise ValueError("config['method_params'] must be a dict when provided.")

    out = dict(method_params)
    if method_name == "gradfix":
        for legacy_key in ("mask_mode", "vote"):
            if legacy_key not in out and cfg.get(legacy_key) is not None:
                out[legacy_key] = cfg[legacy_key]
    return method_name, out


def format_rebase_method_label(method_name: str, method_params: dict[str, Any]) -> str:
    if method_name == "gradfix":
        mask_mode = str(method_params.get("mask_mode", "normal"))
        vote = str(method_params.get("vote", "mean"))
        return f"gradfix(mask={mask_mode}, vote={vote})"
    if method_name == "theseus":
        batches = int(method_params.get("num_batches", 1))
        seq_align = str(method_params.get("seq_align", "interpolate2d"))
        whiten_power = float(method_params.get("whiten_power", 0.0))
        covariance_mode = str(method_params.get("covariance_mode", "activations"))
        return f"theseus(batches={batches}, align={seq_align}, cov={covariance_mode}, whiten={whiten_power:g})"
    if method_name == "theseus_reference":
        batches = int(method_params.get("num_batches", 1))
        token_strategy = str(method_params.get("token_strategy", "interpolate_2d"))
        transport = str(method_params.get("method", "svd"))
        return f"theseus_reference(batches={batches}, token={token_strategy}, transport={transport})"
    if method_name == "transfusion":
        max_iter = int(method_params.get("max_iter", 100))
        intra_head = bool(method_params.get("intra_head", True))
        return f"transfusion(iter={max_iter}, intra_head={intra_head})"
    return method_name
