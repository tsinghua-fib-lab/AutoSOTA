"""Spectral-decoupled scale trainer."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from conformal_model.scale.arch.losses import QuantileLoss
from conformal_model.scale.arch.config import QuantileModelConfig
from conformal_model.scale.arch.dataset import ResidualSequenceDataset
from conformal_model.scale.arch.spec_decoupled_components import resolve_scale_model

_DEFAULT_TRAIN = {
    "epochs": 250,
    "batch_size": 128,
    "lr": 8e-4,
    "weight_decay": 1e-5,
    "val_ratio": 0.2,
    "patience": 4,
    "grad_clip": 5.0,
    "device": "cuda",
    "seed": 2026,
}

_DEFAULT_MODEL = {
    "n_scales": 4,
    "kernel_type": "mexican_hat",
    "n_high_scales": None,
    "backbone_out_dim": 128,
    "enable_gating": True,
    "backbone_kwargs": {
        "node_dim": 64,
        "embed_dim": 64,
        "num_layer": 5,
        "temp_dim_tid": 128,
        "temp_dim_diw": 128,
        "time_of_day_size": 288,
        "day_of_week_size": 7,
    },
}

_DEFAULT_LOSS = {"crossing_penalty": 15.0}


class SpectralExchangeablescale:
    """Trainer/manager for SpectralExchangeableQuantileNet."""

    def __init__(
        self,
        cfg: QuantileModelConfig,
        *,
        adjacency: np.ndarray,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        device: str | torch.device | None = None,
        weight_decay: float | None = None,
        val_ratio: float | None = None,
        patience: int | None = None,
        grad_clip: float | None = None,
        crossing_penalty: float | None = None,
        n_scales: int | None = None,
        kernel_type: str | None = None,
        n_high_scales: int | None = None,
        backbone_out_dim: int | None = None,
        backbone_kwargs: dict | None = None,
        enable_gating: bool | None = None,
        model_name: str | None = None,
        best_ckpt_path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg

        resolved_device = device if device is not None else _DEFAULT_TRAIN["device"]
        if str(resolved_device).lower().startswith("cuda") and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)

        self.seed = int(_DEFAULT_TRAIN["seed"])
        self._set_seed(self.seed)
        self._split_generator = torch.Generator()
        self._split_generator.manual_seed(self.seed)

        self.epochs = int(_DEFAULT_TRAIN["epochs"] if epochs is None else epochs)
        self.batch_size = int(_DEFAULT_TRAIN["batch_size"] if batch_size is None else batch_size)
        self.lr = float(_DEFAULT_TRAIN["lr"] if lr is None else lr)
        self.weight_decay = float(_DEFAULT_TRAIN["weight_decay"] if weight_decay is None else weight_decay)
        self.val_ratio = float(_DEFAULT_TRAIN["val_ratio"] if val_ratio is None else val_ratio)
        self.patience = int(_DEFAULT_TRAIN["patience"] if patience is None else patience)
        self.grad_clip = float(_DEFAULT_TRAIN["grad_clip"] if grad_clip is None else grad_clip)

        gso_tensor = torch.from_numpy(adjacency).float().to(self.device)

        resolved_n_scales = int(_DEFAULT_MODEL["n_scales"] if n_scales is None else n_scales)
        resolved_kernel_type = str(_DEFAULT_MODEL["kernel_type"] if kernel_type is None else kernel_type)
        resolved_n_high_scales = _DEFAULT_MODEL["n_high_scales"] if n_high_scales is None else n_high_scales
        resolved_backbone_out_dim = int(_DEFAULT_MODEL["backbone_out_dim"] if backbone_out_dim is None else backbone_out_dim)
        resolved_backbone_kwargs = dict(_DEFAULT_MODEL["backbone_kwargs"]) if backbone_kwargs is None else backbone_kwargs
        resolved_enable_gating = bool(_DEFAULT_MODEL["enable_gating"] if enable_gating is None else enable_gating)

        self.best_ckpt_path = Path(best_ckpt_path) if best_ckpt_path is not None else None
        if self.best_ckpt_path is not None:
            self.best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        model_cls = resolve_scale_model(model_name)
        self.model = model_cls(
            cfg=cfg,
            adjacency=adjacency,
            gso=gso_tensor,
            n_scales=resolved_n_scales,
            kernel_type=resolved_kernel_type,
            n_high_scales=resolved_n_high_scales,
            backbone_out_dim=resolved_backbone_out_dim,
            backbone_kwargs=resolved_backbone_kwargs,
            enable_gating=resolved_enable_gating,
        ).to(self.device)

        resolved_crossing = float(_DEFAULT_LOSS["crossing_penalty"] if crossing_penalty is None else crossing_penalty)
        self.criterion = QuantileLoss(cfg.quantiles, crossing_penalty_weight=resolved_crossing)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            import torch.backends.cudnn as cudnn  # type: ignore
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass

    def _prepare_dataset(
        self,
        context: torch.Tensor,
        residuals: torch.Tensor,
        context_exog: torch.Tensor | None,
        future_exog: torch.Tensor | None,
    ) -> ResidualSequenceDataset:
        return ResidualSequenceDataset(context, residuals, context_exog, future_exog)

    def _require_tod_dow(self, context_exog: torch.Tensor | None) -> None:
        if context_exog is None or context_exog.shape[-1] < 2:
            raise ValueError(
                "Backbone requires `context_exog` to include tod/dow scalars in the last 2 dims."
            )

    def fit(
        self,
        context: torch.Tensor,
        residuals: torch.Tensor,
        context_exog: torch.Tensor | None = None,
        future_exog: torch.Tensor | None = None,
        train_indices: np.ndarray | None = None,
        val_indices: np.ndarray | None = None,
    ) -> None:
        """Train on (context, residuals) pairs."""
        self._require_tod_dow(context_exog)
        dataset = self._prepare_dataset(context, residuals, context_exog, future_exog)

        if train_indices is not None and val_indices is not None:
            train_set = Subset(dataset, train_indices.tolist())
            val_set = Subset(dataset, val_indices.tolist())
        else:
            val_size = int(len(dataset) * self.val_ratio)
            train_size = len(dataset) - val_size
            train_set, val_set = random_split(dataset, [train_size, val_size], generator=self._split_generator)

        self._set_seed(self.seed)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        total_steps = int(self.epochs) * (n_train_batches + n_val_batches)
        overall = tqdm(total=total_steps, desc="Training", unit="batch")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_accum = 0.0
            for past, resid, past_ex, _ in train_loader:
                past = past.to(self.device)
                resid = resid.to(self.device)
                past_ex = past_ex.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                preds = self.model(past, past_ex)
                loss = self.criterion(preds, resid)
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

                train_loss_accum += float(loss.item()) * past.size(0)
                overall.update(1)
                overall.set_postfix({"epoch": f"{epoch}/{self.epochs}", "batch_loss": f"{loss.item():.4e}"})

            avg_train_loss = train_loss_accum / max(1, len(train_set))

            self.model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for past, resid, past_ex, _ in val_loader:
                    past = past.to(self.device)
                    resid = resid.to(self.device)
                    past_ex = past_ex.to(self.device)

                    preds = self.model(past, past_ex)
                    batch_loss = float(self.criterion(preds, resid).item())
                    val_loss_accum += batch_loss * past.size(0)

                    overall.update(1)
                    overall.set_postfix({"epoch": f"{epoch}/{self.epochs}", "val_batch_loss": f"{batch_loss:.4e}"})

            avg_val_loss = val_loss_accum / max(1, len(val_set))
            scheduler.step(avg_val_loss)

            overall.set_postfix({"epoch": f"{epoch}/{self.epochs}", "train": f"{avg_train_loss:.4e}", "val": f"{avg_val_loss:.4e}"})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if self.best_ckpt_path is not None:
                    torch.save(best_state, self.best_ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        overall.close()

        if self.best_ckpt_path is not None and self.best_ckpt_path.exists():
            state = torch.load(self.best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)
        elif best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def predict_residual_quantiles(
        self,
        context: torch.Tensor,
        context_exog: torch.Tensor | None = None,
        future_exog: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict residual quantiles for each sample."""
        self._require_tod_dow(context_exog)
        dummy_resid = torch.zeros(context.shape[0], self.cfg.horizon, self.cfg.num_nodes)

        dataset = self._prepare_dataset(context, dummy_resid, context_exog, future_exog)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        self.model.eval()
        out = []
        for past, _, past_ex, _ in loader:
            past = past.to(self.device)
            past_ex = past_ex.to(self.device)
            out.append(self.model(past, past_ex).cpu())

        return torch.cat(out, dim=0)

    def quantile_indices(self, alphas: Sequence[float]) -> List[Tuple[int, int]]:
        """Map miscoverage levels alpha to (q_low_idx, q_high_idx) indices."""
        q_list = [round(float(x), 6) for x in list(self.cfg.quantiles)]
        out: List[Tuple[int, int]] = []
        for alpha in alphas:
            lo = round(float(alpha) / 2.0, 6)
            hi = round(1.0 - float(alpha) / 2.0, 6)
            try:
                out.append((q_list.index(lo), q_list.index(hi)))
            except ValueError as exc:
                raise ValueError(
                    f"Model was not trained for alpha={alpha} (q_low={lo}, q_high={hi}). "
                    f"Available: {q_list}"
                ) from exc
        return out
