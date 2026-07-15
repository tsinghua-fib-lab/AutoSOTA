from __future__ import annotations

import copy
import re
from typing import Optional, Tuple, Dict, Any, Iterable, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

from data import create_dataloaders
from train.utils import (
    _steps_from_percentiles,
    finish_wandb,
    get_dataset_data_name_lower,
    make_checkpoint_callback,
    make_run_name,
    make_trainer,
    make_wandb_logger,
    should_share_xy_normalizer,
    split_dataset_result,
    strip_name_keys,
    StripMetadataModule,
)
from train.metrics import (
    compute_nrmse,
    compute_psrmse_three_bands,
    compute_stochastic_mean_std_metrics,
)


@torch.no_grad()
def compute_ensemble_metrics(
    y_samples: Tensor,
    y_true: Tensor,
    eps: float = 1e-7,
    compute_crps: bool = True,
) -> Dict[str, Tensor]:
    """
    Args:
        y_samples: [K, B, ...]
        y_true:    [B, ...]
    Returns:
        Dict of scalar tensors averaged over B and spatial dims:
            rmse, spread, ssr, (optional) crps

    Notes:
    - CRPS for empirical ensemble:
        CRPS = E|X - y| - 0.5 E|X - X'|
      We compute E|X - X'| elementwise in O(K log K) using sorting:
        (1/K^2) sum_{i,j} |x_i - x_j| = (2/K^2) * sum_{i=1}^K (2i-K-1) x_(i)
    """
    if y_samples.ndim < 2:
        raise ValueError(
            f"y_samples must have shape [K, B, ...], got {tuple(y_samples.shape)}"
        )
    if y_true.ndim != y_samples.ndim - 1:
        raise ValueError(
            f"y_true must have shape [B, ...], got {tuple(y_true.shape)} vs {tuple(y_samples.shape)}"
        )

    K = int(y_samples.shape[0])

    y_mean = y_samples.mean(dim=0)
    y_var = y_samples.var(dim=0, unbiased=False).clamp_min(eps)

    mse = (y_mean - y_true).pow(2).mean()
    rmse = mse.sqrt()

    spread = y_var.mean().sqrt()
    ssr = spread / (rmse + eps)

    out: Dict[str, Tensor] = {"rmse": rmse, "spread": spread, "ssr": ssr}

    if compute_crps:
        term1 = (y_samples - y_true.unsqueeze(0)).abs().mean()
        ys, _ = torch.sort(y_samples, dim=0)  # [K, B, ...]
        i = torch.arange(1, K + 1, device=ys.device, dtype=ys.dtype).view(
            K, *([1] * (ys.ndim - 1))
        )
        w = 2 * i - K - 1  # [K, 1, 1, ...]
        term2 = (2.0 / (K * K)) * (w * ys).sum(dim=0).mean()

        out["crps"] = term1 - 0.5 * term2

    return out


def _format_horizon_tag(step: int, width: int = 3) -> str:
    """Stable lexicographic tag for horizons: 1 -> t001."""
    return f"t{int(step):0{int(width)}d}"


class DiffusionModel(StripMetadataModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_config: DictConfig,
        scheduler_config: Optional[DictConfig] = None,
        T: int = 50,
        samples_per_example: int = 1,
        ema_decay: float = 0.9999,
        x_normalizer: Any = None,
        y_normalizer: Any = None,
        test_samples_per_example: int | None = None,
        rollout_k_chunk: int = 8,
        rollout_traj_chunk: int | None = 16,
        stochastic_n_chunk: int = 32,
        stochastic_k_chunk: int = 4,
        viz_num_trajectories_default: int = 3,
        viz_horizon_percentiles_default: Sequence[float] = (1, 20, 40, 60, 80, 100),
        eval_mode: str | None = None,
        eval_solver: str = "euler",
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "operatorencoder",
                "autoencoder",
                "cond_autoencoder",
                "encoder",
                "decoder",
                "output_embedder",
                "optimizer_config",
                "scheduler_config",
                "x_normalizer",
                "y_normalizer",
            ]
        )

        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.loss_fn = nn.MSELoss()

        self.T = int(T)
        self.samples_per_example = int(samples_per_example)
        self.test_samples_per_example = test_samples_per_example
        self.eval_solver = str(eval_solver)

        self.eval_mode = self._normalize_eval_mode(eval_mode)
        self.ema_decay = float(ema_decay) if ema_decay is not None else None
        self.use_ema = self.ema_decay is not None
        if self.use_ema:
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.requires_grad_(False)
            self.model_ema.eval()
            self._ema_keys_checked = False
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.rollout_k_chunk = (
            int(rollout_k_chunk) if rollout_k_chunk is not None else 8
        )
        self.rollout_traj_chunk = (
            int(rollout_traj_chunk) if rollout_traj_chunk is not None else 16
        )
        self.stochastic_n_chunk = max(1, int(stochastic_n_chunk))
        self.stochastic_k_chunk = max(1, int(stochastic_k_chunk))

        self.data_shape: Optional[Tuple[int, ...]] = None
        self.register_buffer(
            "_data_shape", torch.zeros(0, dtype=torch.long), persistent=True
        )
        self.viz_num_trajectories_default = int(viz_num_trajectories_default)
        self.viz_horizon_percentiles_default = tuple(
            float(p) for p in viz_horizon_percentiles_default
        )
        self._last_rollout_viz_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._last_stochastic_viz_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize_eval_mode(mode: str | None) -> str:
        if mode is None:
            return "none"
        m = str(mode).strip().lower()
        if m == "":
            return "none"
        if m not in {"none", "rollout", "stochastic"}:
            raise ValueError(
                f"Unknown eval_mode={mode!r}. Expected 'rollout' or 'stochastic'."
            )
        return m

    @staticmethod
    def _first_dataloader(dl_or_list: Any) -> Any:
        if dl_or_list is None:
            return None
        if isinstance(dl_or_list, (list, tuple)):
            return dl_or_list[0] if len(dl_or_list) > 0 else None
        return dl_or_list

    def _move_normalizers_to_device(self) -> None:
        for n in (
            getattr(self, "x_normalizer", None),
            getattr(self, "y_normalizer", None),
        ):
            if n is None:
                continue
            to_fn = getattr(n, "to", None)
            if callable(to_fn):
                n.to(self.device)

    def on_fit_start(self) -> None:
        self._move_normalizers_to_device()

    @torch.no_grad()
    def _update_ema(self) -> None:
        """
        Stable EMA update over state_dict (parameters + buffers), with:
        - one-time key equality check
        - float and complex support
        - non-float tensors copied verbatim
        """
        if not self.use_ema:
            return

        msd = self.model.state_dict()
        esd = self.model_ema.state_dict()

        if not self._ema_keys_checked:
            if msd.keys() != esd.keys():
                missing_in_ema = sorted(set(msd.keys()) - set(esd.keys()))
                extra_in_ema = sorted(set(esd.keys()) - set(msd.keys()))
                raise RuntimeError(
                    "EMA state_dict mismatch between model and model_ema. "
                    f"missing_in_ema={missing_in_ema[:20]}{'...' if len(missing_in_ema) > 20 else ''}, "
                    f"extra_in_ema={extra_in_ema[:20]}{'...' if len(extra_in_ema) > 20 else ''}"
                )
            self._ema_keys_checked = True

        decay = float(self.ema_decay)
        one_minus = 1.0 - decay

        for k, v in msd.items():
            ev = esd[k]
            if not isinstance(v, torch.Tensor) or not isinstance(ev, torch.Tensor):
                continue

            v_detached = v.detach()

            if torch.is_floating_point(ev):
                ev.copy_(ev.mul(decay).add(v_detached, alpha=one_minus))
            elif torch.is_complex(ev):
                ev.copy_(ev * decay + v_detached * one_minus)
            else:
                ev.copy_(v_detached)

    def forward(
        self, x: Tensor, t: Tensor, cond: Tensor, use_ema: Optional[bool] = None
    ) -> Tensor:
        if use_ema is None:
            use_ema = self.use_ema and (not self.training)
        model = self.model_ema if (use_ema and self.use_ema) else self.model
        return model(x, t, cond)

    def compute_diffusion_loss(
        self, x_cond: Tensor, y_target: Tensor, use_ema: bool = False
    ) -> Tensor:
        if self.data_shape is None:
            self.data_shape = tuple(y_target.shape[1:])
            self._data_shape = torch.tensor(
                self.data_shape, device=self.device, dtype=torch.long
            )

        B = int(y_target.shape[0])
        t = torch.rand((B,), device=self.device, dtype=y_target.dtype)

        x0 = torch.randn_like(y_target)
        x1 = y_target

        t_view = t.view(B, *([1] * (y_target.ndim - 1)))
        x_t = torch.lerp(x0, x1, t_view)

        v_pred = self.forward(x_t, t, x_cond, use_ema=use_ema)
        v_target = x1 - x0
        return self.loss_fn(v_pred, v_target)

    def training_step(self, batch, batch_idx) -> Tensor:
        self._move_normalizers_to_device()
        x_cond, y_target = batch
        loss = self.compute_diffusion_loss(x_cond, y_target, use_ema=False)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.use_ema:
            self._update_ema()

    def validation_step(self, batch, batch_idx) -> Tensor:
        self._move_normalizers_to_device()
        x_cond, y_true = batch
        loss = self.compute_diffusion_loss(x_cond, y_true, use_ema=False)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        self._move_normalizers_to_device()
        x_cond, y_true = batch
        loss = self.compute_diffusion_loss(x_cond, y_true, use_ema=False)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return loss

    def _log_test_metric_tables(self, metrics: Dict[str, torch.Tensor]) -> None:
        """
        Create separate W&B tables for per-horizon TEST metrics:
          test_rollout_rmse_table, test_rollout_nrmse_table, test_rollout_crps_table, ...

        Each table has columns: step, value
        """
        if not isinstance(self.logger, WandbLogger):
            return
        try:
            import wandb
        except Exception:
            return
        bases = [
            "rollout_rmse",
            "rollout_nrmse",
            "rollout_vrmse",
            "rollout_crps",
            "rollout_spread",
            "rollout_ssr",
            "rollout_psrmse_low",
            "rollout_psrmse_mid",
            "rollout_psrmse_high",
        ]
        pat = re.compile(r"^(rollout_[A-Za-z0-9_]+)_(t\d+)$")
        per_base: Dict[str, Dict[int, float]] = {b: {} for b in bases}

        for k, v in metrics.items():
            m = pat.match(k)
            if m is None:
                continue
            base, tag = m.group(1), m.group(2)
            if base not in per_base:
                continue
            step = int(tag[1:])

            if isinstance(v, torch.Tensor):
                val = float(v.detach().cpu().item())
            else:
                val = float(v)
            per_base[base][step] = val
        for base in bases:
            d = per_base[base]
            if not d:
                continue
            table = wandb.Table(columns=["step", "value"])
            for step in sorted(d.keys()):
                table.add_data(int(step), float(d[step]))
            self.logger.experiment.log({f"test_{base}_table": table})

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        if self.eval_mode == "stochastic":
            payload = getattr(self, "val_stochastic", None)
            if payload is None:
                return
            metrics = self.evaluate_stochastic_operator_learning(
                payload, collect_viz=True, viz_num_examples=3, solver=self.eval_solver
            )
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
            self.log_stochastic_mean_std_viz_cache(
                self._last_stochastic_viz_cache, prefix="val_stochastic_viz"
            )
        elif self.eval_mode == "inverse":
            val_loader = self._first_dataloader(
                getattr(self.trainer, "val_dataloaders", None)
            )
            if val_loader is None:
                return
            metrics = self.evaluate_inverse_operator_learning(val_loader)
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
        elif self.eval_mode == "rollout":
            val_trjs = getattr(self, "val_trajectories", None)
            if val_trjs is None:
                return

            metrics = self.evaluate_autoregressive_rollout_ensemble(
                val_trjs,
                log_all_horizons=False,  # percentiles only => not too many scalars
                horizon_percentiles=self.viz_horizon_percentiles_default,
                collect_viz=True,
                viz_num_trajectories=self.viz_num_trajectories_default,
                viz_horizon_percentiles=self.viz_horizon_percentiles_default,
            )
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)

            self.log_rollout_viz_cache(
                self._last_rollout_viz_cache, prefix="val_rollout_viz"
            )
        else:
            return

    def on_test_epoch_end(self) -> None:
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        if self.eval_mode == "stochastic":
            payload = getattr(self, "test_stochastic", None)
            if payload is None:
                return
            metrics = self.evaluate_stochastic_operator_learning(
                payload, collect_viz=True, viz_num_examples=3, solver=self.eval_solver
            )
            for k, v in metrics.items():
                self.log(f"test_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
            self.log_stochastic_mean_std_viz_cache(
                self._last_stochastic_viz_cache, prefix="test_stochastic_viz"
            )
        elif self.eval_mode == "inverse":
            test_loader = self._first_dataloader(
                getattr(self.trainer, "test_dataloaders", None)
            )
            if test_loader is None:
                return
            metrics = self.evaluate_inverse_operator_learning(test_loader)
            for k, v in metrics.items():
                self.log(f"test_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
        elif self.eval_mode == "rollout":
            test_trjs = getattr(self, "test_trajectories", None)
            if test_trjs is None:
                return

            metrics = self.evaluate_autoregressive_rollout_ensemble(
                test_trjs,
                num_samples_per_example=self.test_samples_per_example,
                log_all_horizons=True,  # produces per-step metrics for ALL horizons
                horizon_percentiles=self.viz_horizon_percentiles_default,  # unused when log_all_horizons=True
                collect_viz=True,
                viz_num_trajectories=self.viz_num_trajectories_default,
                viz_horizon_percentiles=self.viz_horizon_percentiles_default,  # viz still percentiles only
            )
            for k, v in metrics.items():
                if "_t" in k:
                    continue
                self.log(f"test_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
            self._log_test_metric_tables(metrics)
            self.log_rollout_viz_cache(
                self._last_rollout_viz_cache, prefix="test_rollout_viz"
            )
        else:
            return

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_config, params=self.model.parameters())
        if (
            self.scheduler_config is None
            or getattr(self.scheduler_config, "scheduler", None) is None
        ):
            return optimizer

        scheduler = instantiate(self.scheduler_config.scheduler, optimizer=optimizer)
        lr_sched = {
            "scheduler": scheduler,
            "interval": str(getattr(self.scheduler_config, "interval", "epoch")),
            "frequency": int(getattr(self.scheduler_config, "frequency", 1)),
            "strict": bool(getattr(self.scheduler_config, "strict", True)),
            "name": str(getattr(self.scheduler_config, "name", "lr")),
        }
        monitor = getattr(self.scheduler_config, "monitor", None)
        if monitor is not None:
            lr_sched["monitor"] = str(monitor)

        return {"optimizer": optimizer, "lr_scheduler": lr_sched}

    @torch.inference_mode()
    def sample(
        self,
        cond: Tensor,
        num_steps: Optional[int] = None,
        solver: str = "euler",
        use_ema=None,
    ) -> Tensor:
        steps = int(num_steps or self.T)
        B = int(cond.shape[0])
        dt = 1.0 / steps

        if self.data_shape is None:
            raise RuntimeError(
                "data_shape is unknown. Run a forward/loss once before sampling."
            )

        y = torch.randn((B, *self.data_shape), device=self.device, dtype=cond.dtype)
        t_grid = torch.linspace(
            0.0, 1.0, steps + 1, device=self.device, dtype=cond.dtype
        )

        for i in range(steps):
            t_curr = t_grid[i].expand(B)
            v_curr = self.forward(y, t_curr, cond, use_ema=use_ema)

            if solver == "euler":
                y = y + v_curr * dt
            elif solver == "heun":
                y_guess = y + v_curr * dt
                v_next = self.forward(
                    y_guess, t_grid[i + 1].expand(B), cond, use_ema=use_ema
                )
                y = y + 0.5 * (v_curr + v_next) * dt
            else:
                raise ValueError(f"Unknown solver={solver!r}. Use 'euler' or 'heun'.")
        return y

    @torch.inference_mode()
    def evaluate_stochastic_operator_learning(
        self,
        payload: Any,
        *,
        num_pred_samples: int | None = None,
        max_examples: int | None = 256,
        solver: str = "euler",
        collect_viz: bool = False,
        viz_num_examples: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate stochastic operator learning on a set-valued validation/test payload.

        Expected payload form:
          - {"x": x_orig, "y": y_set_orig}

        Shapes:
          x_orig: (N, C, ...) or (N, ...)
          y_set_orig: (N, S, C, ...) or (N, S, ...)  (original scale)
        """
        if not isinstance(payload, dict):
            raise ValueError(
                "stochastic payload must be a dict with keys {'x','y'}; "
                f"got type={type(payload)}"
            )

        x_orig = payload.get("x", None)
        y_set = payload.get("y", None)
        if x_orig is None or y_set is None:
            return {}

        if (
            getattr(self, "x_normalizer", None) is None
            or getattr(self, "y_normalizer", None) is None
        ):
            raise ValueError(
                "Stochastic evaluation requires both x_normalizer and y_normalizer (for encode/decode)."
            )

        x_cpu = torch.as_tensor(x_orig, device="cpu", dtype=torch.float32)
        y_cpu = torch.as_tensor(y_set, device="cpu", dtype=torch.float32)
        if x_cpu.ndim == 2:
            x_cpu = x_cpu.unsqueeze(1)  # (N,L)->(N,1,L)
        if y_cpu.ndim == 3:
            y_cpu = y_cpu.unsqueeze(2)
        elif y_cpu.ndim == 4:
            n, s, d2, d3 = y_cpu.shape
            if int(d2) > 8 and int(d3) > 8:
                y_cpu = y_cpu.unsqueeze(2)
        if y_cpu.ndim not in (4, 5):
            raise ValueError(
                f"Expected y_set to be (N,S,C,...) got {tuple(y_cpu.shape)}"
            )

        N = int(x_cpu.shape[0])
        if max_examples is not None:
            N = min(N, int(max_examples))
            x_cpu = x_cpu[:N]
            y_cpu = y_cpu[:N]

        S = int(y_cpu.shape[1])
        if S <= 0:
            return {}

        K = int(
            num_pred_samples
            or (self.test_samples_per_example or self.samples_per_example)
        )
        K = max(1, K)

        self._move_normalizers_to_device()
        n_chunk = max(1, int(getattr(self, "stochastic_n_chunk", 32)))
        k_chunk = max(1, int(getattr(self, "stochastic_k_chunk", 4)))
        total_n = 0
        sums: Dict[str, torch.Tensor] = {}
        self._last_stochastic_viz_cache = {}

        def _sample_pred_cpu(
            xb_cpu: torch.Tensor,  # (B,C,...)
            *,
            k_chunk_local: int,
        ) -> torch.Tensor:
            """
            Returns y_pred on CPU: (K,B,C,...).
            Uses chunked sampling over K and will recursively split B / K if CUDA OOM occurs.
            """
            B = int(xb_cpu.shape[0])
            if B <= 0:
                raise ValueError("Empty batch in stochastic evaluation.")

            try:
                xb_model = self.x_normalizer.encode(
                    xb_cpu.to(self.device, non_blocking=True)
                )

                y_chunks_cpu: list[torch.Tensor] = []
                for k0 in range(0, K, int(k_chunk_local)):
                    kk = int(min(int(k_chunk_local), K - k0))
                    y_model_samples = self.generate_y_samples(
                        xb_model,
                        num_samples=kk,
                        use_ema=None,
                        num_steps=None,
                        solver=solver,
                    )  # (kk,B,C,...)

                    y_model_flat = y_model_samples.reshape(
                        kk * B, *y_model_samples.shape[2:]
                    )
                    y_orig_flat = self.y_normalizer.decode(y_model_flat).to(
                        torch.float32
                    )
                    y_pred_k = y_orig_flat.reshape(kk, B, *y_orig_flat.shape[1:]).to(
                        "cpu"
                    )
                    y_chunks_cpu.append(y_pred_k)
                    del y_model_samples, y_model_flat, y_orig_flat, y_pred_k
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                return (
                    torch.cat(y_chunks_cpu, dim=0)
                    if len(y_chunks_cpu) > 1
                    else y_chunks_cpu[0]
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if B > 1:
                    mid = max(1, B // 2)
                    y0 = _sample_pred_cpu(xb_cpu[:mid], k_chunk_local=k_chunk_local)
                    y1 = _sample_pred_cpu(xb_cpu[mid:], k_chunk_local=k_chunk_local)
                    return torch.cat([y0, y1], dim=1)  # concat along B
                if int(k_chunk_local) > 1:
                    return _sample_pred_cpu(
                        xb_cpu, k_chunk_local=max(1, int(k_chunk_local) // 2)
                    )
                raise

        for i0 in range(0, N, n_chunk):
            i1 = min(N, i0 + n_chunk)
            xb = x_cpu[i0:i1]
            yb = y_cpu[i0:i1]  # (B,S,C,...)
            y_pred = _sample_pred_cpu(xb, k_chunk_local=k_chunk)  # (K,B,C,...), CPU
            y_true = yb.transpose(0, 1).contiguous()

            if collect_viz and (not self._last_stochastic_viz_cache):
                Np = min(int(viz_num_examples), int(y_pred.shape[1]))
                gen_mean = y_pred.mean(dim=0)[:Np].detach().to("cpu")
                gen_std = y_pred.std(dim=0, unbiased=False)[:Np].detach().to("cpu")
                tgt_mean = y_true.mean(dim=0)[:Np].detach().to("cpu")
                tgt_std = y_true.std(dim=0, unbiased=False)[:Np].detach().to("cpu")
                self._last_stochastic_viz_cache = {
                    "gen_mean": gen_mean,
                    "gen_std": gen_std,
                    "tgt_mean": tgt_mean,
                    "tgt_std": tgt_std,
                }
                if y_pred.ndim == 4 and y_true.ndim == 4:
                    k_viz = min(int(y_pred.shape[0]), 8)
                    s_viz = min(int(y_true.shape[0]), 8)
                    self._last_stochastic_viz_cache["gen_samples_1d"] = (
                        y_pred[:k_viz, :Np].detach().to("cpu")
                    )
                    self._last_stochastic_viz_cache["tgt_samples_1d"] = (
                        y_true[:s_viz, :Np].detach().to("cpu")
                    )

            m = compute_stochastic_mean_std_metrics(y_pred, y_true)

            bsz = int(i1 - i0)
            for k, v in m.items():
                sums[k] = sums.get(k, torch.zeros_like(v)) + v.detach().to(
                    sums.get(k, v).device
                ) * float(bsz)
            total_n += bsz

        if total_n <= 0:
            return {}
        return {k: v / float(total_n) for k, v in sums.items()}

    @torch.inference_mode()
    def evaluate_inverse_operator_learning(
        self,
        dataloader: Any,
        *,
        num_pred_samples: int | None = None,
        max_examples: int | None = 256,
        solver: str = "euler",
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate inverse operator learning with deterministic targets.

        Metrics are ensemble-based but use a single true target per condition:
          - inverse_nrmse: NRMSE(mean(pred), y_true)
          - inverse_crps: CRPS(pred_ensemble, y_true)
          - inverse_ssr: spread-to-skill ratio

        Also logs rmse/spread for completeness.
        """
        if dataloader is None:
            return {}
        if (
            getattr(self, "x_normalizer", None) is None
            or getattr(self, "y_normalizer", None) is None
        ):
            raise ValueError(
                "Inverse evaluation requires both x_normalizer and y_normalizer."
            )

        K = int(
            num_pred_samples
            or (self.test_samples_per_example or self.samples_per_example)
        )
        K = max(1, K)
        n_chunk = max(1, int(getattr(self, "stochastic_n_chunk", 32)))
        k_chunk = max(1, int(getattr(self, "stochastic_k_chunk", 4)))

        self.eval()
        self._move_normalizers_to_device()

        sums: Dict[str, torch.Tensor] = {}
        total_n = 0
        remaining = None if max_examples is None else int(max_examples)

        def _sample_pred_cpu(
            xb_model_cpu: torch.Tensor, *, k_chunk_local: int
        ) -> torch.Tensor:
            B = int(xb_model_cpu.shape[0])
            if B <= 0:
                raise ValueError("Empty batch in inverse evaluation.")
            y_chunks_cpu: list[torch.Tensor] = []
            for k0 in range(0, K, int(k_chunk_local)):
                kk = int(min(int(k_chunk_local), K - k0))
                xb_model_gpu = xb_model_cpu.to(
                    self.device, dtype=torch.float32, non_blocking=True
                )
                y_model_samples = self.generate_y_samples(
                    xb_model_gpu,
                    num_samples=kk,
                    use_ema=None,
                    num_steps=None,
                    solver=solver,
                )
                y_model_flat = y_model_samples.reshape(
                    kk * B, *y_model_samples.shape[2:]
                )
                y_orig_flat = self.y_normalizer.decode(y_model_flat).to(torch.float32)
                y_pred_k = y_orig_flat.reshape(kk, B, *y_orig_flat.shape[1:]).to("cpu")
                y_chunks_cpu.append(y_pred_k)
                del xb_model_gpu, y_model_samples, y_model_flat, y_orig_flat, y_pred_k
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                torch.cat(y_chunks_cpu, dim=0)
                if len(y_chunks_cpu) > 1
                else y_chunks_cpu[0]
            )

        for batch in dataloader:
            if remaining is not None and remaining <= 0:
                break

            x_model, y_model = batch[0], batch[1]
            if x_model.ndim == 2:
                x_model = x_model.unsqueeze(1)
            if y_model.ndim == 2:
                y_model = y_model.unsqueeze(1)

            x_model = x_model.detach().to("cpu", dtype=torch.float32)
            y_model = y_model.detach().to("cpu", dtype=torch.float32)

            B0 = int(x_model.shape[0])
            use_B = B0 if remaining is None else min(B0, int(remaining))
            x_model = x_model[:use_B]
            y_model = y_model[:use_B]

            for i0 in range(0, use_B, n_chunk):
                i1 = min(use_B, i0 + n_chunk)
                xb = x_model[i0:i1]
                yb = y_model[i0:i1]
                bsz = int(i1 - i0)
                if bsz <= 0:
                    continue

                y_true_orig = (
                    self.y_normalizer.decode(yb.to(self.device, non_blocking=True))
                    .to(torch.float32)
                    .to("cpu")
                )
                y_pred = _sample_pred_cpu(xb, k_chunk_local=k_chunk)  # (K,B,C,...), CPU

                ensemble_m = compute_ensemble_metrics(
                    y_pred, y_true_orig, compute_crps=True
                )
                y_mean = y_pred.mean(dim=0)
                nrmse = compute_nrmse(y_mean, y_true_orig)

                metrics = {
                    "inverse_rmse": ensemble_m["rmse"],
                    "inverse_nrmse": nrmse,
                    "inverse_crps": ensemble_m["crps"],
                    "inverse_spread": ensemble_m["spread"],
                    "inverse_ssr": ensemble_m["ssr"],
                }

                for mk, mv in metrics.items():
                    mv_cpu = mv.detach().to("cpu")
                    sums[mk] = sums.get(mk, torch.zeros_like(mv_cpu)) + mv_cpu * float(
                        bsz
                    )
                total_n += bsz

            if remaining is not None:
                remaining -= use_B

        if total_n <= 0:
            return {}
        return {k: v / float(total_n) for k, v in sums.items()}

    def log_stochastic_mean_std_viz_cache(
        self, viz_cache: Dict[str, torch.Tensor], prefix: str
    ) -> None:
        """
        Log stochastic operator-learning snapshots as a figure:
          generated mean/std vs target mean/std.
        Uses cached tensors; does not trigger any extra sampling.
        """
        if not viz_cache:
            return
        if not isinstance(self.logger, WandbLogger):
            return

        import matplotlib.pyplot as plt

        def to_field(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().cpu()
            if t.ndim >= 2 and int(t.shape[0]) <= 8:
                t = t[0]
            return t.squeeze()

        try:
            import wandb
        except Exception:
            wandb = None

        gen_mean = viz_cache.get("gen_mean", None)
        gen_std = viz_cache.get("gen_std", None)
        tgt_mean = viz_cache.get("tgt_mean", None)
        tgt_std = viz_cache.get("tgt_std", None)
        gen_samples_1d = viz_cache.get("gen_samples_1d", None)  # (Kviz, Np, C, X)
        tgt_samples_1d = viz_cache.get("tgt_samples_1d", None)  # (Sviz, Np, C, X)
        if gen_mean is None or gen_std is None or tgt_mean is None or tgt_std is None:
            return

        Np = int(gen_mean.shape[0])
        fig, axes = plt.subplots(Np, 4, figsize=(4.8 * 4, 3.6 * Np), squeeze=False)
        titles = ["Generated mean", "Generated std", "Target mean", "Target std"]

        for r in range(Np):
            row = [gen_mean[r], gen_std[r], tgt_mean[r], tgt_std[r]]
            for c, ten in enumerate(row):
                ax = axes[r, c]
                field = to_field(ten)

                if field.ndim == 1:
                    if c == 0 and gen_samples_1d is not None:
                        for k in range(int(gen_samples_1d.shape[0])):
                            sfield = to_field(gen_samples_1d[k, r])
                            if sfield.ndim == 1:
                                ax.plot(
                                    sfield.numpy(),
                                    color="tab:blue",
                                    alpha=0.18,
                                    linewidth=1.0,
                                )
                    if c == 2 and tgt_samples_1d is not None:
                        for k in range(int(tgt_samples_1d.shape[0])):
                            sfield = to_field(tgt_samples_1d[k, r])
                            if sfield.ndim == 1:
                                ax.plot(
                                    sfield.numpy(),
                                    color="tab:orange",
                                    alpha=0.18,
                                    linewidth=1.0,
                                )

                    ax.plot(field.numpy(), color="black", linewidth=2.0)
                    ax.grid(True, which="major", alpha=0.3, linewidth=0.8)
                else:
                    im = ax.imshow(field.numpy(), origin="lower")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.tick_params(axis="both", which="both", labelsize=9)
                if r == 0:
                    ax.set_title(titles[c])

        fig.tight_layout()
        key = f"{prefix}/mean_std"

        if wandb is not None:
            self.logger.experiment.log({key: wandb.Image(fig)})
        else:
            self.logger.log_image(key=key, images=[fig])

        plt.close(fig)

    @torch.inference_mode()
    def generate_y_samples(
        self, x: Tensor, num_samples: int = 1, use_ema=None, **kwargs
    ) -> Tensor:
        """
        Returns:
            [K, B, ...] where K=num_samples
        """
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        if num_samples == 1:
            return self.sample(cond=x, use_ema=use_ema, **kwargs).unsqueeze(0)

        x_rep = x.repeat_interleave(num_samples, dim=0)
        y_rep = self.sample(cond=x_rep, use_ema=use_ema, **kwargs)

        B = int(x.shape[0])
        return y_rep.view(B, num_samples, *y_rep.shape[1:]).transpose(0, 1).contiguous()

    def log_rollout_viz_cache(
        self, viz_cache: Dict[int, Dict[str, torch.Tensor]], prefix: str
    ) -> None:
        if not viz_cache:
            return
        if not isinstance(self.logger, WandbLogger):
            return

        import matplotlib.pyplot as plt

        def to_field(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().cpu()
            if t.ndim >= 2 and int(t.shape[0]) <= 8:
                t = t[0]
            return t.squeeze()

        try:
            import wandb
        except Exception:
            wandb = None

        for step in sorted(viz_cache.keys()):
            tag = _format_horizon_tag(step)
            snap = viz_cache[step]

            target = snap["target"]  # (Np, C, ...)
            mean = snap["mean"]
            var = snap["var"]
            err = snap["err"]

            Np = int(target.shape[0])
            fig, axes = plt.subplots(Np, 4, figsize=(4.8 * 4, 3.6 * Np), squeeze=False)
            titles = ["Target", "Ensemble mean", "Ensemble var", "(Mean - Target)^2"]

            for r in range(Np):
                row = [target[r], mean[r], var[r], err[r]]
                for c, ten in enumerate(row):
                    ax = axes[r, c]
                    field = to_field(ten)

                    if field.ndim == 1:
                        ax.plot(field.numpy())
                        ax.grid(True, which="major", alpha=0.3, linewidth=0.8)
                    else:
                        im = ax.imshow(field.numpy(), origin="lower")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.tick_params(axis="both", which="both", labelsize=9)
                    if r == 0:
                        ax.set_title(titles[c])

            fig.tight_layout()
            key = f"{prefix}/{tag}"

            if wandb is not None:
                self.logger.experiment.log({key: wandb.Image(fig)})
            else:
                self.logger.log_image(key=key, images=[fig])

            plt.close(fig)

    @torch.inference_mode()
    def evaluate_autoregressive_rollout_ensemble(
        self,
        trajectories: torch.Tensor,
        *,
        max_trajectories: int | None = 64,
        horizons: Iterable[int] | None = None,
        horizon_percentiles: Iterable[float] = (1, 20, 40, 60, 80, 100),
        use_ema: bool = True,
        num_samples_per_example: int | None = None,
        log_all_horizons: bool = False,
        collect_viz: bool = False,
        viz_num_trajectories: int = 3,
        viz_horizons: Iterable[int] | None = None,
        viz_horizon_percentiles: Iterable[float] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Exact rollout evaluation for ensemble predictions (includes CRPS + PSRMSE bands).

        Intended for large ensembles / large fields:
        - Keeps rollout state on CPU.
        - Sends conditioning to GPU in chunks, samples next states, decodes, and computes exact CPU metrics.
        - (Optional) caches target/mean/var/error at selected horizons for visualization (no extra sampling).

        Args:
            trajectories: (N, T, ...) original-scale
        """
        if trajectories is None or trajectories.numel() == 0:
            return {}

        self.eval()
        self._move_normalizers_to_device()

        self._last_rollout_viz_cache = {}
        trjs_in = trajectories.detach().to(dtype=torch.float32)

        if trjs_in.ndim == 3:
            trjs = trjs_in.unsqueeze(2)  # (N,T,X)->(N,T,1,X)
        elif trjs_in.ndim == 4:
            n, t, d2, d3 = trjs_in.shape
            if int(d2) <= 8 and int(d3) >= 16:
                trjs = trjs_in  # (N,T,C,X)
            else:
                trjs = trjs_in.unsqueeze(2)  # (N,T,1,H,W)
        elif trjs_in.ndim == 5:
            trjs = trjs_in  # (N,T,C,H,W)
        else:
            raise ValueError(f"Unexpected trajectories shape {tuple(trjs_in.shape)}")

        if max_trajectories is not None:
            trjs = trjs[: int(max_trajectories)]

        N_traj, T = int(trjs.shape[0]), int(trjs.shape[1])
        if T < 2:
            return {}

        if self.data_shape is None:
            self.data_shape = tuple(trjs.shape[2:])
            self._data_shape = torch.tensor(
                self.data_shape, device=self.device, dtype=torch.long
            )

        total_steps = T - 1
        K = max(1, int(num_samples_per_example or self.samples_per_example))
        if log_all_horizons:
            horizons_set = set(range(1, total_steps + 1))
        elif horizons is None:
            pct_to_step = _steps_from_percentiles(total_steps, horizon_percentiles)
            horizons_set = set(pct_to_step.values())
        else:
            horizons_set = set(int(h) for h in horizons)
        if collect_viz:
            if viz_horizons is not None:
                viz_horizons_set = set(int(h) for h in viz_horizons)
            else:
                vp = (
                    horizon_percentiles
                    if viz_horizon_percentiles is None
                    else viz_horizon_percentiles
                )
                viz_pct_to_step = _steps_from_percentiles(total_steps, vp)
                viz_horizons_set = set(viz_pct_to_step.values())
        else:
            viz_horizons_set = set()

        k_chunk = max(1, int(self.rollout_k_chunk))
        traj_chunk = max(1, int(self.rollout_traj_chunk))
        y0_orig_cpu = trjs[:, 0].to("cpu")  # (N,C,...)
        if getattr(self, "x_normalizer", None) is None:
            raise ValueError(
                "Autoregressive rollout requires `x_normalizer` to encode conditioning inputs."
            )
        x0_model = self.x_normalizer.encode(
            y0_orig_cpu.to(self.device, non_blocking=True)
        )
        x0_model_cpu = x0_model.detach().to("cpu")

        x_prev_model_cpu = (
            x0_model_cpu.unsqueeze(0)
            .repeat(K, 1, *([1] * (x0_model_cpu.ndim - 1)))
            .contiguous()
        )

        y_true_shape = trjs[:, 1].shape[1:]  # (C,...)
        y_next_orig_cpu = torch.empty(
            (K, N_traj, *y_true_shape), device="cpu", dtype=torch.float32
        )

        eps_norm = 1e-12
        sums: Dict[str, torch.Tensor] = {}
        count = 0

        horizon_metrics: Dict[int, Dict[str, torch.Tensor]] = {}
        horizon_rmse: Dict[int, torch.Tensor] = {}
        horizon_nrmse: Dict[int, torch.Tensor] = {}
        horizon_vrmse: Dict[int, torch.Tensor] = {}
        horizon_psrmse: Dict[int, Dict[str, torch.Tensor]] = {}

        rmse_sum: torch.Tensor | None = None
        nrmse_sum: torch.Tensor | None = None
        vrmse_sum: torch.Tensor | None = None

        if collect_viz:
            viz_cache: Dict[int, Dict[str, torch.Tensor]] = {}
            viz_num = max(1, int(viz_num_trajectories))
        else:
            viz_cache = {}

        for t in range(total_steps):
            y_true_cpu = trjs[:, t + 1].detach().to("cpu")

            for k0 in range(0, K, k_chunk):
                k1 = min(K, k0 + k_chunk)
                for s0 in range(0, N_traj, traj_chunk):
                    s1 = min(N_traj, s0 + traj_chunk)

                    cond_cpu = x_prev_model_cpu[k0:k1, s0:s1]
                    cond_gpu = cond_cpu.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )

                    flat = (k1 - k0) * (s1 - s0)
                    cond_flat = cond_gpu.reshape(flat, *cond_gpu.shape[2:])

                    y_model_flat = self.generate_y_samples(
                        cond_flat,
                        num_samples=1,
                        use_ema=use_ema,
                        num_steps=None,
                        solver="euler",
                    )[0]

                    y_model_block = y_model_flat.reshape(
                        (k1 - k0), (s1 - s0), *y_model_flat.shape[1:]
                    )
                    x_prev_model_cpu[k0:k1, s0:s1] = y_model_block.detach().to("cpu")

                    if getattr(self, "y_normalizer", None) is None:
                        raise ValueError(
                            "Rollout requires `y_normalizer` to decode predictions for metrics."
                        )
                    y_orig_flat = self.y_normalizer.decode(y_model_flat).to(
                        torch.float32
                    )
                    y_orig_block = y_orig_flat.reshape(
                        (k1 - k0), (s1 - s0), *y_orig_flat.shape[1:]
                    )
                    y_next_orig_cpu[k0:k1, s0:s1] = y_orig_block.detach().to("cpu")

                    del (
                        cond_gpu,
                        cond_flat,
                        y_model_flat,
                        y_model_block,
                        y_orig_flat,
                        y_orig_block,
                    )

            m = compute_ensemble_metrics(y_next_orig_cpu, y_true_cpu, compute_crps=True)
            ps = compute_psrmse_three_bands(y_next_orig_cpu, y_true_cpu)

            for mk, mv in m.items():
                mv = mv.detach().to("cpu")
                sums[mk] = mv if mk not in sums else (sums[mk] + mv)

            for pk, pv in ps.items():
                pv = pv.detach().to("cpu")
                sums[pk] = pv if pk not in sums else (sums[pk] + pv)

            y_mean = y_next_orig_cpu.mean(dim=0)
            mse_t = (y_mean - y_true_cpu).pow(2).mean()
            rmse_t = torch.sqrt(mse_t + 1e-12)
            step_l2 = torch.sqrt(y_true_cpu.pow(2).mean()).clamp_min(eps_norm)
            step_var = y_true_cpu.var(unbiased=False).clamp_min(eps_norm)
            nrmse_t = rmse_t / step_l2
            vrmse_t = rmse_t / torch.sqrt(step_var)

            rmse_sum = rmse_t if rmse_sum is None else (rmse_sum + rmse_t)
            nrmse_sum = nrmse_t if nrmse_sum is None else (nrmse_sum + nrmse_t)
            vrmse_sum = vrmse_t if vrmse_sum is None else (vrmse_sum + vrmse_t)
            count += 1

            step = t + 1

            if collect_viz and (step in viz_horizons_set):
                Np = min(viz_num, N_traj)
                y_true_plot = y_true_cpu[:Np].clone()
                y_samples_plot = y_next_orig_cpu[:, :Np]
                y_mean_plot = y_samples_plot.mean(dim=0).clone()
                y_var_plot = y_samples_plot.var(dim=0, unbiased=False).clone()
                y_err_plot = (y_mean_plot - y_true_plot).pow(2).clone()

                viz_cache[step] = {
                    "target": y_true_plot,
                    "mean": y_mean_plot,
                    "var": y_var_plot,
                    "err": y_err_plot,
                }

            if step in horizons_set:
                horizon_metrics[step] = {k: v.detach().to("cpu") for k, v in m.items()}
                horizon_rmse[step] = rmse_t.detach().to("cpu")
                horizon_nrmse[step] = nrmse_t.detach().to("cpu")
                horizon_vrmse[step] = vrmse_t.detach().to("cpu")
                horizon_psrmse[step] = {k: v.detach().to("cpu") for k, v in ps.items()}

        out: Dict[str, torch.Tensor] = {}
        if count > 0:
            for k, v in sums.items():
                out[f"rollout_{k}"] = v / float(count)
            if rmse_sum is not None:
                out["rollout_rmse"] = rmse_sum / float(count)
            if nrmse_sum is not None:
                out["rollout_nrmse"] = nrmse_sum / float(count)
            if vrmse_sum is not None:
                out["rollout_vrmse"] = vrmse_sum / float(count)

        for step, mm in sorted(horizon_metrics.items(), key=lambda kv: kv[0]):
            tag = _format_horizon_tag(step)
            for k, v in mm.items():
                out[f"rollout_{k}_{tag}"] = v
            out[f"rollout_rmse_{tag}"] = horizon_rmse[step]
            out[f"rollout_nrmse_{tag}"] = horizon_nrmse[step]
            out[f"rollout_vrmse_{tag}"] = horizon_vrmse[step]
            for k, v in horizon_psrmse.get(step, {}).items():
                out[f"rollout_{k}_{tag}"] = v

        if collect_viz:
            self._last_rollout_viz_cache = viz_cache

        return out


def train_dm(cfg: DictConfig) -> None:
    if "seed" in cfg:
        seed_everything(int(cfg.seed), workers=True)

    dataset_result = instantiate(cfg.dataset)
    (
        train_set,
        val_set,
        test_set,
        val_trjs,
        test_trjs,
        val_stochastic,
        test_stochastic,
    ) = split_dataset_result(dataset_result)

    data_name_lower = get_dataset_data_name_lower(cfg)
    share_xy = should_share_xy_normalizer(
        data_name_lower=data_name_lower, training_cfg=cfg.training
    )

    train_loader, val_loader, test_loader, xn, yn = create_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        normalization_mode=str(cfg.training.normalization_mode),
        share_xy_normalizer=share_xy,
    )

    backbone = instantiate(strip_name_keys(cfg.model))

    viz_pcts = getattr(cfg.dm, "viz_horizon_percentiles", (1, 20, 40, 60, 80, 100))

    dm_module = DiffusionModel(
        model=backbone,
        optimizer_config=cfg.optimizer,
        scheduler_config=getattr(cfg, "scheduler", None),
        T=int(cfg.dm.T),
        samples_per_example=int(cfg.dm.samples_per_example),
        ema_decay=float(getattr(cfg.dm, "ema_decay", 0.9999)),
        x_normalizer=xn,
        y_normalizer=yn,
        test_samples_per_example=getattr(cfg.dm, "test_samples_per_example", None),
        rollout_k_chunk=int(getattr(cfg.dm, "rollout_k_chunk", 8)),
        rollout_traj_chunk=getattr(cfg.dm, "rollout_traj_chunk", None),
        stochastic_n_chunk=int(getattr(cfg.dm, "stochastic_n_chunk", 32)),
        stochastic_k_chunk=int(getattr(cfg.dm, "stochastic_k_chunk", 4)),
        viz_num_trajectories_default=int(getattr(cfg.dm, "viz_num_trajectories", 3)),
        viz_horizon_percentiles_default=viz_pcts,
        eval_mode=getattr(getattr(cfg, "evaluation", None), "mode", None),
    )

    if val_trjs is not None:
        dm_module.val_trajectories = val_trjs
    if test_trjs is not None:
        dm_module.test_trajectories = test_trjs
    if val_stochastic is not None:
        dm_module.val_stochastic = val_stochastic
    if test_stochastic is not None:
        dm_module.test_stochastic = test_stochastic

    model_name = getattr(cfg.model, "model_name", "DiffusionModel")
    run_name = make_run_name(model_name, "DM")
    project_name = f"{cfg.dataset.data_name}_GEN"
    wandb_logger = make_wandb_logger(cfg, project_name=project_name, run_name=run_name)

    eval_mode = DiffusionModel._normalize_eval_mode(
        getattr(getattr(cfg, "evaluation", None), "mode", None)
    )
    if eval_mode == "stochastic":
        if val_stochastic is None or test_stochastic is None:
            raise ValueError(
                "evaluation.mode='stochastic' requires dataset extras: val_stochastic and test_stochastic."
            )
        monitor_metric = "val_stochastic_ed"
    elif eval_mode == "rollout":
        if val_trjs is None or test_trjs is None:
            raise ValueError(
                "evaluation.mode='rollout' requires dataset extras: val_trjs and test_trjs."
            )
        monitor_metric = "val_rollout_nrmse"
    else:
        monitor_metric = "val_loss"
    ckpt_best = make_checkpoint_callback(
        project_name=project_name,
        run_name=run_name,
        filename_base=model_name,
        filename=f"{model_name}-best-{{epoch:02d}}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = make_trainer(
        training_cfg=cfg.training,
        max_epochs=int(cfg.training.epochs),
        train_loader_len=len(train_loader),
        logger=wandb_logger,
        callbacks=[ckpt_best],
    )

    trainer.fit(dm_module, train_loader, val_loader)

    ckpt_path = getattr(ckpt_best, "best_model_path", None) or "best"
    try:
        trainer.test(
            dm_module, dataloaders=test_loader, ckpt_path=ckpt_path, weights_only=False
        )
    except TypeError:
        trainer.test(dm_module, dataloaders=test_loader, ckpt_path=ckpt_path)

    finish_wandb(wandb_logger)
