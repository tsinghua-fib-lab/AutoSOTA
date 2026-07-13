from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.autograd.functional import jvp

from cellbridge.dynamics.models import SinusoidalTimeEmbedding

log = logging.getLogger(__name__)


class GeoPathMLP(nn.Module):
    """
    Simple geodesic path network that consumes (x0, x1, t) and predicts a
    data-dimension sized correction.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        time_embed_dim: int = 64,
        time_embedding: str = "sin",
        use_time: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if data_dim <= 0:
            raise ValueError("data_dim must be > 0")
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.time_embed_dim = time_embed_dim
        self.time_embedding = time_embedding
        self.use_time = use_time
        self.dropout = dropout

        if self.use_time:
            if self.time_embedding != "sin":
                raise ValueError("Only sinusoidal time embedding is supported.")
            self.t_embed = SinusoidalTimeEmbedding(self.time_embed_dim)
        # Use 1 for direct time concatenation instead of time_embed_dim
        input_dim = 2 * self.data_dim + (1 if self.use_time else 0)

        layers: list[nn.Module] = []
        feat_dim = hidden_dim
        layers.append(nn.Linear(input_dim, feat_dim))
        for _ in range(self.depth - 1):
            layers.append(nn.SELU())
            layers.append(nn.Linear(feat_dim, feat_dim))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        layers.append(nn.SELU())
        layers.append(nn.Linear(feat_dim, self.data_dim))
        self.model = nn.Sequential(*layers)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x0, x1: tensors of shape (B, D)
            t: tensor of shape (B,) or (B, 1)
        """
        if t.ndim == 1:
            t_in = t[:, None]
        elif t.ndim == 2 and t.shape[1] == 1:
            t_in = t
        else:
            raise ValueError("t must be of shape (B,) or (B,1)")
        feats = [x0, x1]
        if self.use_time:
            # Directly concatenate time instead of using embedding
            feats.append(t_in)
        h = torch.cat(feats, dim=-1)
        return self.model(h)


class GeoPathBridge(nn.Module):
    """
    Implements the GeoPath correction used in Metric Flow Matching.
    """

    def __init__(
        self,
        geopath_net: nn.Module,
        alpha: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.geopath_net = geopath_net
        self.alpha = float(alpha)
        self.trainable = bool(trainable)

    @staticmethod
    def gamma(t: torch.Tensor) -> torch.Tensor:
        """Return bridge envelope value."""
        return 1.0 - t**2 - (1.0 - t) ** 2

    @staticmethod
    def d_gamma(t: torch.Tensor) -> torch.Tensor:
        """Return bridge envelope derivative."""
        return 2.0 - 4.0 * t

    def _ensure_col(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t[:, None]
        if t.ndim == 2 and t.shape[1] == 1:
            return t
        raise ValueError("t must be of shape (B,) or (B,1)")

    def sample(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        *,
        detach_outputs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x_t, v_t) for the GeoPath-adjusted bridge.
        """
        if x0.shape != x1.shape:
            raise ValueError("x0 and x1 must share the same shape")
        t_col = self._ensure_col(t).to(x0.device).clamp(0.0, 1.0)
        x0 = x0.to(t_col.device)
        x1 = x1.to(t_col.device)
        base_xt = (1.0 - t_col) * x0 + t_col * x1

        def net_with_t(t_arg: torch.Tensor) -> torch.Tensor:
            return self.geopath_net(x0, x1, t_arg)

        correction, d_corr_dt = jvp(
            net_with_t,
            (t_col,),
            (torch.ones_like(t_col),),
            create_graph=self.trainable,
            strict=True,
        )
        correction = self.alpha * correction
        d_corr_dt = self.alpha * d_corr_dt

        gamma = self.gamma(t_col)
        d_gamma = self.d_gamma(t_col)

        xt = base_xt + gamma * correction
        vt = (x1 - x0) + d_gamma * correction + gamma * d_corr_dt

        if detach_outputs and not self.trainable:
            xt = xt.detach()
            vt = vt.detach()
        return xt, vt


def load_or_train_geopath(
    geopath_cfg: DictConfig,
    data_dim: int,
    artifacts_folder: Path,
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
    G: np.ndarray | None = None,
    logger: logging.Logger | None = None,
    save_trained_checkpoint: bool = True,
    output_dir: Path | None = None,
) -> nn.Module:
    """
    Load a pretrained GeoPath checkpoint or train a new model if not found.

    Args:
        geopath_cfg: Configuration for geopath (must have .model field)
        data_dim: Data dimensionality
        artifacts_folder: Folder to search for existing checkpoints
        X: Start distribution data (needed if training from scratch)
        Y: End distribution data (needed if training from scratch)
        G: Coupling matrix (needed if training from scratch)
        logger: Logger instance
        save_trained_checkpoint: Whether to save newly trained checkpoint
        output_dir: Directory to save newly trained checkpoint

    Returns:
        Loaded or trained geopath network
    """
    if logger is None:
        logger = log

    pretrained_ckpt = geopath_cfg.get("pretrained_ckpt")
    ckpt_path = None

    if pretrained_ckpt:
        # User provided explicit checkpoint path
        ckpt_path = Path(pretrained_ckpt)
        if not ckpt_path.is_absolute():
            logger.warning(
                "Relative checkpoint path provided: %s. "
                "This may not resolve correctly.",
                ckpt_path,
            )
        if not ckpt_path.exists():
            logger.warning(
                "GeoPath checkpoint %s not found; will search or train.", ckpt_path
            )
            ckpt_path = None
    else:
        # Try to find checkpoint in artifacts folder
        logger.warning(
            "GeoPath enabled but no pretrained_ckpt provided, "
            "trying to localize a geopath model."
        )
        geopath_ckpts = list(artifacts_folder.glob("*geopath*.ckpt"))

        if len(geopath_ckpts) > 0:
            ckpt_path = geopath_ckpts[0]
            logger.info("Found GeoPath checkpoint %s in artifacts folder.", ckpt_path)
        else:
            logger.info(
                "No GeoPath checkpoint found in artifacts folder. "
                "Will train a new model."
            )

    # Load checkpoint if found
    if ckpt_path is not None:
        logger.info("Loading GeoPath checkpoint from %s", ckpt_path)
        geopath_net = instantiate(geopath_cfg.model)(data_dim=data_dim)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Try to extract geopath_net parameters with prefix first
        geopath_state = {
            k.replace("geopath_net.", ""): v
            for k, v in state_dict.items()
            if k.startswith("geopath_net.")
        }

        # If no keys with prefix found, use state_dict directly
        if len(geopath_state) == 0:
            logger.info("No 'geopath_net.' prefix found, loading directly")
            geopath_state = state_dict

        logger.info(
            "Loading %d parameter tensors into GeoPath model", len(geopath_state)
        )
        missing, unexpected = geopath_net.load_state_dict(geopath_state, strict=False)
        if missing:
            logger.warning("GeoPath missing params: %s", missing)
        if unexpected:
            logger.warning("GeoPath unexpected params: %s", unexpected)

        return geopath_net

    # Train from scratch
    logger.info("Training a GeoPath model from scratch.")
    if X is None or Y is None or G is None:
        raise ValueError("X, Y, and G must be provided to train GeoPath from scratch")

    from cellbridge.dynamics.geopath_training import GeoPathTrainer

    geopath_trainer = GeoPathTrainer(geopath_cfg, logger=logger, X=X, Y=Y, G=G)
    geopath_net = geopath_trainer.run()

    # Save checkpoint if requested
    if save_trained_checkpoint and output_dir is not None:
        out_ckpt = output_dir / "geopath.ckpt"
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving GeoPath checkpoint to %s", out_ckpt)
        torch.save({"state_dict": geopath_net.state_dict()}, out_ckpt)

    return geopath_net
