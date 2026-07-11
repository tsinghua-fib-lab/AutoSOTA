from __future__ import annotations

import torch

from scgfm.models.geometric_bases import GeometricBasesModel


def load_model_from_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("model_config", {})
    model = GeometricBasesModel(
        K=int(model_cfg.get("K", 16)),
        M=int(model_cfg.get("M", 32)),
        feature_dim=int(model_cfg.get("feature_dim", 50)),
        tau=float(model_cfg.get("tau", 0.1)),
        lambda_gw=float(model_cfg.get("lambda_gw", 1.0)),
        lambda_recon=float(model_cfg.get("lambda_recon", 1.0)),
        lambda_div=float(model_cfg.get("lambda_div", 0.02)),
        div_margin=float(model_cfg.get("div_margin", 8.0)),
        num_projections=int(model_cfg.get("num_projections", 50)),
        device=device,
    ).to(device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    # Handle backward compatibility: older checkpoints may not have gw_theta buffer
    missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Note: missing keys in checkpoint (expected): {missing_keys}")
    model.eval()
    return model, model_cfg

