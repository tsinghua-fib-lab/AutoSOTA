import os
import warnings
from typing import Optional, Union

import lightning
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from preprocessing import (
    gauss_noise_padding,
    gauss_noise_padding_gpu,
    softrank_preprocessing_correct,
    softrank_preprocessing_correct_gpu,
    whiten_blocks,
)
from infonet.decoder import Decoder
from infonet.encoder import Encoder
from infonet.infonet import InfoNet
from infonet.query import Query_Gen_transformer

warnings.filterwarnings("ignore")


class InferLightningWrapper(lightning.LightningModule):
    """
    Lightweight wrapper for checkpoint loading and inference only.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        input_dim_x = cfg.input_dim_x
        input_dim_y = cfg.input_dim_y
        latent_num = cfg.latent_num
        latent_dim = cfg.latent_dim
        decoder_query_dim = cfg.decoder_query_dim
        targetnet_hiddim = cfg.targetnet_hiddim

        self.encoder = Encoder(
            input_dim_x=input_dim_x,
            input_dim_y=input_dim_y,
            latent_num=latent_num,
            latent_dim=latent_dim,
            cross_attn_heads=int(cfg.get("cross_attn_heads", 8)),
            self_attn_heads=int(cfg.get("self_attn_heads", 16)),
            num_self_attn_per_block=int(cfg.get("num_self_attn_per_block", 8)),
            num_self_attn_blocks=int(cfg.get("num_self_attn_blocks", 2)),
        )

        self.decoder = Decoder(
            q_dim=decoder_query_dim,
            latent_dim=latent_dim,
        )

        self.query_gen = Query_Gen_transformer(
            input_dim_x=input_dim_x,
            input_dim_y=input_dim_y,
            dim=decoder_query_dim,
        )

        hypogen_kwargs = {}
        for k in [
            "weight_dim",
            "enc_dec_dim",
            "opt_block_dim",
            "opt_mid_dim",
            "num_opt_mlp_layer",
            "num_enc_dec_layer",
            "num_layers",
            "weight_split_dim",
            "nhead",
            "lr_scheme_method",
            "use_compile",
            "replicate_blocks",
            "target_layers_num",
            "ablation",
        ]:
            if hasattr(cfg, k):
                hypogen_kwargs[k] = getattr(cfg, k)

        self.model = InfoNet(
            encoder=self.encoder,
            decoder=self.decoder,
            query_gen=self.query_gen,
            decoder_query_dim=decoder_query_dim,
            input_dim_x=input_dim_x,
            input_dim_y=input_dim_y,
            targetnet_hiddim=targetnet_hiddim,
            **hypogen_kwargs,
        )

    def forward(self, x):
        return self.model(x.squeeze(1), early_sup=False)


def load_cfg(cfg_path: str) -> DictConfig:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg_path not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def load_ckpt(
    ckpt_path: str,
    cfg: Optional[DictConfig] = None,
    cfg_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
    compile_model: bool = False,
    num_layers_override: int = 0,
):
    """
    Load a checkpoint.

    Config resolution order:
        1. Explicit ``cfg`` argument
        2. ``cfg`` embedded inside the checkpoint (saved by pretrain.py or clean_ckpt.py)
        3. External ``cfg_path`` YAML file

    Args:
        ckpt_path: path to the .ckpt file
        cfg: OmegaConf config (highest priority, overrides everything)
        cfg_path: path to the .yaml config file (fallback if ckpt has no embedded cfg)
        device: device to load model onto

    Returns:
        module_wrapper, infonet_model, cfg, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise KeyError(f"'state_dict' not found in checkpoint: {ckpt_path}")

    # Resolve config
    if cfg is not None:
        pass  # use the explicitly provided cfg
    elif "cfg" in ckpt:
        cfg = OmegaConf.create(ckpt["cfg"])
        if verbose:
            print("[load_ckpt] cfg loaded from checkpoint.")
    elif cfg_path is not None:
        cfg = load_cfg(cfg_path)
        if verbose:
            print(f"[load_ckpt] cfg loaded from file: {cfg_path}")
    else:
        raise ValueError(
            "No config available. The checkpoint does not contain an embedded cfg, "
            "and neither cfg nor cfg_path was provided. "
            "Use clean_ckpt.py --cfg_path to embed config into the checkpoint."
        )

    module = InferLightningWrapper(cfg)
    missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)

    # Unexpected keys (present in the ckpt, absent from the model) are harmless
    # — the model simply ignores them. Missing keys, however, mean the model has
    # parameters the checkpoint does not, so those would stay randomly initialized;
    # that silently returns a wrong model, so we refuse to load.
    if unexpected:
        print(f"[load_ckpt] note: {len(unexpected)} unexpected key(s) in the checkpoint "
              f"were ignored (e.g. {unexpected[:5]}).")
    if missing:
        raise RuntimeError(
            f"load_ckpt: {len(missing)} model parameter(s) are missing from the checkpoint "
            f"(e.g. {missing[:5]}). The embedded config does not match the checkpoint's weights, "
            f"so part of the model would be left randomly initialized. Refusing to return a "
            f"partially-initialized model."
        )

    module = module.to(device)
    module.eval()
    module.model.eval()

    if compile_model:
        module.model = torch.compile(module.model, mode='reduce-overhead', fullgraph=False)

    # Stamp the preprocessing flags onto the model so EVERY eval path
    # (estimate_mi_batch / _gpu and everything that calls them) auto-applies the
    # SAME whitening the model was trained with. Default "none" -> unchanged.
    module.model._whiten = str(cfg.get("whiten", "none"))
    module.model._whiten_eps = float(cfg.get("whiten_eps", 1e-3))

    if verbose:
        print(f"[load_ckpt] ckpt_path = {ckpt_path}")
        print(f"[load_ckpt] device    = {device}")
        print(f"[load_ckpt] missing keys    = {len(missing)}")
        print(f"[load_ckpt] unexpected keys = {len(unexpected)}")
        if len(missing) > 0:
            print("[load_ckpt] first few missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("[load_ckpt] first few unexpected keys:", unexpected[:10])

    return module, module.model, cfg, device


def _ensure_3d_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if not torch.is_tensor(x):
        raise TypeError(f"Input must be np.ndarray or torch.Tensor, got {type(x)}")

    if x.dim() == 2:
        x = x.unsqueeze(0)
    elif x.dim() != 3:
        raise ValueError(f"Expected input dim 2 or 3, got shape {tuple(x.shape)}")

    return x.float()


def estimate_mi_batch(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    model,
    max_dim: int,
    softrank_reg: float = 1e-3,
    gauss_copula: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    whiten: Optional[str] = None,
    whiten_eps: Optional[float] = None,
) -> torch.Tensor:
    """
    Batch estimate mutual information.

    Args:
        X: [B, N, d] or [N, d]
        Y: [B, N, d] or [N, d]
        whiten: per-side whitening to match training ("none"/"eig"). If None, read
            from the model (stamped by load_ckpt from the ckpt's cfg).

    Returns:
        mi_est: [B]
    """
    if device is None:
        param_device = next(model.parameters()).device
        device = param_device
    else:
        device = torch.device(device)

    X = _ensure_3d_tensor(X)
    Y = _ensure_3d_tensor(Y)

    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got X={X.shape}, Y={Y.shape}")

    B, _, d = X.shape
    if d > max_dim:
        raise ValueError(f"Input dim d={d} cannot be greater than max_dim={max_dim}")

    X = X.cpu()
    Y = Y.cpu()

    X_padded = gauss_noise_padding(X, aim_dim=max_dim, perm=False)
    Y_padded = gauss_noise_padding(Y, aim_dim=max_dim, perm=False)

    sample_xy = torch.cat([X_padded, Y_padded], dim=-1)

    sample_xy = softrank_preprocessing_correct(
        sample_xy,
        regularization_strength=softrank_reg,
        gauss_copula=gauss_copula,
    ).to(device)

    _whiten = getattr(model, "_whiten", "none") if whiten is None else whiten
    if _whiten == "eig":
        _eps = getattr(model, "_whiten_eps", 1e-3) if whiten_eps is None else whiten_eps
        sample_xy = whiten_blocks(sample_xy, max_dim=max_dim, eps_floor=float(_eps))

    model.eval()
    with torch.no_grad():
        mi_est = model(sample_xy)

    if not torch.is_tensor(mi_est):
        mi_est = torch.tensor([mi_est], device=device, dtype=torch.float32)

    mi_est = mi_est.reshape(-1).detach().cpu()

    if mi_est.shape[0] != B:
        if mi_est.numel() == 1 and B == 1:
            return mi_est
        raise RuntimeError(
            f"Model output batch mismatch: expected B={B}, got shape={tuple(mi_est.shape)}"
        )

    return mi_est


def estimate_mi(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    model,
    max_dim: int,
    softrank_reg: float = 1e-3,
    gauss_copula: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    Estimate mutual information for a single pair.

    Args:
        X: [N, d]
        Y: [N, d]

    Returns:
        float
    """
    mi = estimate_mi_batch(
        X=X, Y=Y, model=model, max_dim=max_dim,
        softrank_reg=softrank_reg, gauss_copula=gauss_copula, device=device,
    )
    return float(mi[0].item())


def compute_ksmi_mean(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    projection_dim: int,
    model,
    proj_num: int,
    batchsize: int,
    max_dim: int = 5,
    softrank_reg: float = 1e-3,
    normalize_input: bool = True,
    gauss_copula: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    early_exit: bool = False,
    early_exit_cv: float = 0.05,
    min_slices: int = 8,
) -> float:
    """
    K-sliced MI mean estimation via random projections.

    Args:
        X: [N, dx]
        Y: [N, dy]

    Returns:
        float
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)

    X = X.float().cpu()
    Y = Y.float().cpu()

    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(f"X, Y must be [N, d], got X={X.shape}, Y={Y.shape}")

    seq_len, dx = X.shape
    seq_len_y, dy = Y.shape
    if seq_len != seq_len_y:
        raise ValueError(f"X and Y must have same number of samples, got X={X.shape}, Y={Y.shape}")

    if normalize_input:
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
        Y = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-6)

    results = []
    welford_n = 0
    welford_mean = 0.0
    welford_M2 = 0.0

    for i in range(0, proj_num, batchsize):
        current_batch_size = min(batchsize, proj_num - i)

        X_proj_batch = []
        Y_proj_batch = []

        for _ in range(current_batch_size):
            proj_matrix_x = torch.randn(dx, projection_dim, device="cpu")
            proj_matrix_y = torch.randn(dy, projection_dim, device="cpu")

            proj_matrix_x = proj_matrix_x / (torch.norm(proj_matrix_x, dim=0, keepdim=True) + 1e-12)
            proj_matrix_y = proj_matrix_y / (torch.norm(proj_matrix_y, dim=0, keepdim=True) + 1e-12)

            X_proj = X @ proj_matrix_x
            Y_proj = Y @ proj_matrix_y

            X_proj_batch.append(X_proj.unsqueeze(0))
            Y_proj_batch.append(Y_proj.unsqueeze(0))

        X_proj_batch = torch.cat(X_proj_batch, dim=0)
        Y_proj_batch = torch.cat(Y_proj_batch, dim=0)

        mi_est = estimate_mi_batch(
            X_proj_batch, Y_proj_batch,
            model=model, max_dim=max_dim,
            softrank_reg=softrank_reg, gauss_copula=gauss_copula, device=device,
        )
        results.append(mi_est)

        if early_exit:
            for val in mi_est.tolist():
                welford_n += 1
                delta = val - welford_mean
                welford_mean += delta / welford_n
                delta2 = val - welford_mean
                welford_M2 += delta * delta2

            num_processed = welford_n
            if num_processed >= min_slices and welford_n >= 2:
                variance = welford_M2 / (welford_n - 1) if welford_n > 1 else 0.0
                std = variance ** 0.5
                cv = std / (abs(welford_mean) + 1e-8)
                if cv < early_exit_cv:
                    break

    results = torch.cat(results, dim=0)
    return float(results.mean().item())


# ============================================================
# Fully-GPU inference (no CPU round-trip, vectorized softrank)
# ============================================================

def estimate_mi_batch_gpu(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    model,
    max_dim: int,
    softrank_reg: float = 1e-3,
    gauss_copula: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    whiten: Optional[str] = None,
    whiten_eps: Optional[float] = None,
) -> torch.Tensor:
    """
    GPU-only counterpart to estimate_mi_batch.

    Moves X/Y to ``device`` once and keeps everything (padding, softrank,
    forward) on GPU. Requires CUDA.

    Args:
        X: [B, N, d] or [N, d]
        Y: [B, N, d] or [N, d]

    Returns:
        mi_est: [B] (on CPU, like the original variant)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if device.type != "cuda":
        raise ValueError(f"estimate_mi_batch_gpu requires a CUDA device, got {device}")

    X = _ensure_3d_tensor(X).to(device, non_blocking=True)
    Y = _ensure_3d_tensor(Y).to(device, non_blocking=True)

    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape, got X={X.shape}, Y={Y.shape}")

    B, _, d = X.shape
    if d > max_dim:
        raise ValueError(f"Input dim d={d} cannot be greater than max_dim={max_dim}")

    X_padded = gauss_noise_padding_gpu(X, aim_dim=max_dim, perm=False)
    Y_padded = gauss_noise_padding_gpu(Y, aim_dim=max_dim, perm=False)

    sample_xy = torch.cat([X_padded, Y_padded], dim=-1)

    sample_xy = softrank_preprocessing_correct_gpu(
        sample_xy,
        regularization_strength=softrank_reg,
        gauss_copula=gauss_copula,
    )

    _whiten = getattr(model, "_whiten", "none") if whiten is None else whiten
    if _whiten == "eig":
        _eps = getattr(model, "_whiten_eps", 1e-3) if whiten_eps is None else whiten_eps
        sample_xy = whiten_blocks(sample_xy, max_dim=max_dim, eps_floor=float(_eps))

    model.eval()
    with torch.no_grad():
        mi_est = model(sample_xy)

    if not torch.is_tensor(mi_est):
        mi_est = torch.tensor([mi_est], device=device, dtype=torch.float32)

    mi_est = mi_est.reshape(-1).detach().cpu()

    if mi_est.shape[0] != B:
        if mi_est.numel() == 1 and B == 1:
            return mi_est
        raise RuntimeError(
            f"Model output batch mismatch: expected B={B}, got shape={tuple(mi_est.shape)}"
        )

    return mi_est


def estimate_mi_gpu(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    model,
    max_dim: int,
    softrank_reg: float = 1e-3,
    gauss_copula: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """GPU-only counterpart to estimate_mi (single (X, Y) pair)."""
    mi = estimate_mi_batch_gpu(
        X=X, Y=Y, model=model, max_dim=max_dim,
        softrank_reg=softrank_reg, gauss_copula=gauss_copula, device=device,
    )
    return float(mi[0].item())


def compute_ksmi_mean_gpu(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    projection_dim: int,
    model,
    proj_num: int,
    batchsize: int,
    max_dim: int = 5,
    softrank_reg: float = 1e-3,
    normalize_input: bool = True,
    gauss_copula: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """
    GPU-only counterpart to compute_ksmi_mean.

    All ``proj_num`` random projections are sampled and applied on GPU in one
    batched matmul per chunk, then fed through ``estimate_mi_batch_gpu``.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if device.type != "cuda":
        raise ValueError(f"compute_ksmi_mean_gpu requires a CUDA device, got {device}")

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)

    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError(f"X, Y must be [N, d], got X={X.shape}, Y={Y.shape}")

    X = X.float().to(device, non_blocking=True)
    Y = Y.float().to(device, non_blocking=True)

    seq_len, dx = X.shape
    seq_len_y, dy = Y.shape
    if seq_len != seq_len_y:
        raise ValueError(f"X and Y must have same number of samples, got X={X.shape}, Y={Y.shape}")

    if normalize_input:
        X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)
        Y = (Y - Y.mean(dim=0, keepdim=True)) / (Y.std(dim=0, keepdim=True) + 1e-6)

    results = []

    for i in range(0, proj_num, batchsize):
        current_batch_size = min(batchsize, proj_num - i)

        proj_x = torch.randn(current_batch_size, dx, projection_dim, device=device)
        proj_y = torch.randn(current_batch_size, dy, projection_dim, device=device)
        proj_x = proj_x / (torch.norm(proj_x, dim=1, keepdim=True) + 1e-12)
        proj_y = proj_y / (torch.norm(proj_y, dim=1, keepdim=True) + 1e-12)

        # X: [N, dx] @ proj_x: [B, dx, P] -> X_proj_batch: [B, N, P]
        X_proj_batch = torch.einsum("nd,bdp->bnp", X, proj_x)
        Y_proj_batch = torch.einsum("nd,bdp->bnp", Y, proj_y)

        mi_est = estimate_mi_batch_gpu(
            X_proj_batch, Y_proj_batch,
            model=model, max_dim=max_dim,
            softrank_reg=softrank_reg, gauss_copula=gauss_copula, device=device,
        )
        results.append(mi_est)

    results = torch.cat(results, dim=0)
    return float(results.mean().item())
