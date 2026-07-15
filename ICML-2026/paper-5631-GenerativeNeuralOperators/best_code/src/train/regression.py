import torch
import torch.nn as nn
import re
from typing import Dict

from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

from data import create_dataloaders
from train.utils import (
    _steps_from_percentiles,
    finish_wandb,
    make_checkpoint_callback,
    make_run_name,
    make_trainer,
    make_wandb_logger,
    should_share_xy_normalizer,
    split_dataset_result,
    get_dataset_data_name_lower,
    strip_name_keys,
    StripMetadataModule,
)
from train.metrics import (
    compute_nrmse,
    compute_psrmse_three_bands,
    compute_stochastic_mean_std_metrics,
)


def _format_horizon_tag(step: int, width: int = 3) -> str:
    """Format horizon step as a zero-padded tag (e.g., 1 -> 't001') for stable lexicographic sorting."""
    return f"t{int(step):0{int(width)}d}"


class Regression(StripMetadataModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig | None = None,
        x_normalizer=None,
        y_normalizer=None,
        eval_mode: str | None = None,
        stochastic_n_chunk: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.eval_mode = self._normalize_eval_mode(eval_mode)
        self.stochastic_n_chunk = max(1, int(stochastic_n_chunk))
        self.criterion = nn.MSELoss()
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

    def forward(self, x):
        out = self.model(x)
        return out

    def _move_normalizers_to_device(self) -> None:
        """Ensure normalizer stats are on the same device as the module."""
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

    def training_step(self, batch, batch_idx):
        self._move_normalizers_to_device()
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self._move_normalizers_to_device()
        x, y = batch
        out = self(x)
        if self.y_normalizer is not None:
            out_original = self.y_normalizer.decode(out)
            y_original = self.y_normalizer.decode(y)
            mse_original = self.criterion(out_original, y_original)
            self.log("val_loss", mse_original, prog_bar=True, sync_dist=True)

            rmse_original = torch.sqrt(mse_original + 1e-12)
            self.log("val_rmse", rmse_original, prog_bar=True, sync_dist=True)

            nrmse_original = compute_nrmse(out_original, y_original)
            self.log("val_nrmse", nrmse_original, prog_bar=True, sync_dist=True)
            return mse_original

        mse = self.criterion(out, y)
        self.log("val_loss", mse, prog_bar=True, sync_dist=True)
        return mse

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        if self.eval_mode == "stochastic":
            payload = getattr(self, "val_stochastic", None)
            if payload is None:
                return
            metrics = self.evaluate_stochastic_operator_learning(
                payload, collect_viz=True, viz_num_examples=3
            )
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=False, on_epoch=True, sync_dist=True)
            self.log_stochastic_mean_std_viz_cache(
                self._last_stochastic_viz_cache, prefix="val_stochastic_viz"
            )
        elif self.eval_mode == "rollout":
            val_trjs = getattr(self, "val_trajectories", None)
            if val_trjs is None:
                return
            metrics = self.evaluate_autoregressive_rollout(
                val_trjs,
                log_all_horizons=False,
                collect_viz=True,
                viz_num_trajectories=3,
            )
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=False, on_epoch=True, sync_dist=True)
            self.log_rollout_viz_cache(
                self._last_rollout_viz_cache, prefix="val_rollout_viz"
            )
        else:
            return

    def test_step(self, batch, batch_idx):
        self._move_normalizers_to_device()
        x, y = batch
        out = self(x)
        if self.y_normalizer is not None:
            out_original = self.y_normalizer.decode(out)
            y_original = self.y_normalizer.decode(y)
            mse_original = self.criterion(out_original, y_original)
            self.log("test_loss", mse_original, prog_bar=True, sync_dist=True)
            rmse_original = torch.sqrt(mse_original + 1e-12)
            self.log("test_rmse", rmse_original, prog_bar=True, sync_dist=True)
            nrmse_original = compute_nrmse(out_original, y_original)
            self.log("test_nrmse", nrmse_original, prog_bar=True, sync_dist=True)
            return mse_original

        mse = self.criterion(out, y)
        self.log("test_loss", mse, prog_bar=True, sync_dist=True)
        return mse

    def on_test_epoch_end(self) -> None:
        if getattr(self.trainer, "global_rank", 0) != 0:
            return
        if self.eval_mode == "stochastic":
            payload = getattr(self, "test_stochastic", None)
            if payload is None:
                return
            metrics = self.evaluate_stochastic_operator_learning(
                payload, collect_viz=True, viz_num_examples=3
            )
            for k, v in metrics.items():
                self.log(f"test_{k}", v, prog_bar=False, on_epoch=True, sync_dist=True)
            self.log_stochastic_mean_std_viz_cache(
                self._last_stochastic_viz_cache, prefix="test_stochastic_viz"
            )
        elif self.eval_mode == "rollout":
            test_trjs = getattr(self, "test_trajectories", None)
            if test_trjs is None:
                return
            metrics = self.evaluate_autoregressive_rollout(
                test_trjs,
                log_all_horizons=True,
                collect_viz=True,
                viz_num_trajectories=3,
            )
            for k, v in metrics.items():
                if "_t" in k:
                    continue
                self.log(f"test_{k}", v, prog_bar=False, on_epoch=True, sync_dist=True)
            self._log_test_metric_tables(metrics)

            self.log_rollout_viz_cache(
                self._last_rollout_viz_cache, prefix="test_rollout_viz"
            )
        else:
            return

    @torch.inference_mode()
    def evaluate_stochastic_operator_learning(
        self,
        payload,
        *,
        max_examples: int | None = 256,
        eps: float = 1e-12,
        collect_viz: bool = False,
        viz_num_examples: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate deterministic predictions against set-valued targets."""
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
            x_cpu = x_cpu.unsqueeze(1)
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

        self._move_normalizers_to_device()
        n_chunk = max(1, int(getattr(self, "stochastic_n_chunk", 64)))
        total_n = 0
        sums: Dict[str, torch.Tensor] = {}

        self._last_stochastic_viz_cache = {}
        self.model.eval()

        def _predict_y_cpu(xb_cpu: torch.Tensor) -> torch.Tensor:
            """Return decoded predictions on CPU."""
            B = int(xb_cpu.shape[0])
            if B <= 0:
                raise ValueError("Empty batch in stochastic evaluation.")
            try:
                xb_model = self.x_normalizer.encode(
                    xb_cpu.to(self.device, non_blocking=True)
                )
                y_pred_model = self(xb_model)
                return (
                    self.y_normalizer.decode(y_pred_model).to(torch.float32).to("cpu")
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if B > 1:
                    mid = max(1, B // 2)
                    y0 = _predict_y_cpu(xb_cpu[:mid])
                    y1 = _predict_y_cpu(xb_cpu[mid:])
                    return torch.cat([y0, y1], dim=0)
                raise

        for i0 in range(0, N, n_chunk):
            i1 = min(N, i0 + n_chunk)
            xb = x_cpu[i0:i1]
            yb = y_cpu[i0:i1]  # (B,S,C,...)

            y_pred_orig = _predict_y_cpu(xb)
            y_pred = y_pred_orig.unsqueeze(0)
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

            m = compute_stochastic_mean_std_metrics(y_pred, y_true, eps=eps)

            bsz = int(i1 - i0)
            for k, v in m.items():
                sums[k] = sums.get(k, torch.zeros_like(v)) + v.detach().to(
                    "cpu"
                ) * float(bsz)
            total_n += bsz

        if total_n <= 0:
            return {}
        return {k: v / float(total_n) for k, v in sums.items()}

    def log_stochastic_mean_std_viz_cache(
        self, viz_cache: Dict[str, torch.Tensor], prefix: str
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

    def _log_test_metric_tables(self, metrics: Dict[str, torch.Tensor]) -> None:
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
            fig, axes = plt.subplots(Np, 3, figsize=(4.8 * 3, 3.6 * Np), squeeze=False)
            titles = ["Target", "Prediction", "(Pred - Target)^2"]

            for r in range(Np):
                row = [target[r], mean[r], err[r]]
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

    def evaluate_autoregressive_rollout(
        self,
        trajectories: torch.Tensor,
        *,
        horizons: list[int] | None = None,
        horizon_percentiles: tuple[float, ...] = (1, 20, 40, 60, 80, 100),
        log_all_horizons: bool = False,
        collect_viz: bool = False,
        viz_num_trajectories: int = 3,
        viz_horizons: list[int] | None = None,
        viz_horizon_percentiles: tuple[float, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate autoregressive rollout error on original-scale trajectories."""
        if trajectories is None or trajectories.numel() == 0:
            return {}

        self.model.eval()
        self._move_normalizers_to_device()
        trajectories = trajectories.to(self.device)
        self._last_rollout_viz_cache = {}
        if trajectories.ndim == 3:
            trjs = trajectories.unsqueeze(2)
        elif trajectories.ndim == 4:
            n, t, d2, d3 = trajectories.shape
            if int(d2) <= 8 and int(d3) >= 16:
                trjs = trajectories  # (N, T, C, X)
            else:
                trjs = trajectories.unsqueeze(2)  # (N, T, 1, H, W)
        elif trajectories.ndim == 5:
            trjs = trajectories  # (N, T, C, H, W)
        else:
            raise ValueError(
                "Expected trajectories shaped (N_traj, T, X), (N_traj, T, C, X), (N_traj, T, H, W), or "
                f"(N_traj, T, C, H, W); got {tuple(trajectories.shape)}"
            )

        N_traj, T = int(trjs.shape[0]), int(trjs.shape[1])
        total_steps = T - 1

        if log_all_horizons:
            pct_to_step = {}
            horizons_set = set(range(1, total_steps + 1))
        elif horizons is None:
            pct_to_step = _steps_from_percentiles(total_steps, horizon_percentiles)
            horizons_set = set(pct_to_step.values())
        else:
            pct_to_step = {}
            horizons_set = set(int(h) for h in horizons)

        if collect_viz:
            if viz_horizons is not None:
                viz_horizons_set = set(int(h) for h in viz_horizons)
            else:
                vhp = (
                    viz_horizon_percentiles
                    if viz_horizon_percentiles is not None
                    else horizon_percentiles
                )
                viz_horizons_set = set(
                    _steps_from_percentiles(total_steps, vhp).values()
                )
        else:
            viz_horizons_set = set()

        rollout_predictions = torch.zeros_like(trjs)
        rollout_predictions[:, 0] = trjs[:, 0]
        with torch.no_grad():
            for t in range(T - 1):
                x_t = rollout_predictions[:, t]  # (N_traj, C, *spatial)
                if self.x_normalizer is not None:
                    x_t = self.x_normalizer.encode(x_t)
                elif self.y_normalizer is not None:
                    x_t = self.y_normalizer.encode(x_t)
                pred_t_plus_1_normalized = self.model(x_t)  # (N_traj, C_out, *spatial)
                if self.y_normalizer is not None:
                    pred_t_plus_1 = self.y_normalizer.decode(pred_t_plus_1_normalized)
                else:
                    pred_t_plus_1 = pred_t_plus_1_normalized
                rollout_predictions[:, t + 1] = pred_t_plus_1
        rollout_errors = rollout_predictions - trjs
        reduce_dims = (0, 2) + tuple(range(3, rollout_errors.ndim))
        timestep_mse = torch.mean(
            rollout_errors[:, 1:].pow(2), dim=reduce_dims
        )  # Shape: (T-1,)
        timestep_rmse = torch.sqrt(timestep_mse + 1e-12)
        eps_norm = 1e-12
        timestep_l2 = torch.sqrt(
            torch.mean(trjs[:, 1:].pow(2), dim=reduce_dims)
        ).clamp_min(
            eps_norm
        )  # (T-1,)
        timestep_var = torch.var(
            trjs[:, 1:], dim=reduce_dims, unbiased=False
        ).clamp_min(
            eps_norm
        )  # (T-1,)

        metrics = {}
        mse_rollout = torch.mean(rollout_errors[:, 1:].pow(2))  # Skip initial condition
        rmse_rollout = torch.sqrt(mse_rollout + 1e-12)
        metrics["rollout_rmse"] = rmse_rollout
        metrics["rollout_nrmse"] = torch.mean(timestep_rmse / timestep_l2)
        metrics["rollout_vrmse"] = torch.mean(timestep_rmse / torch.sqrt(timestep_var))
        ps_sums: Dict[str, torch.Tensor] = {}
        horizon_ps: Dict[int, Dict[str, torch.Tensor]] = {}
        viz_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        with torch.no_grad():
            for step in range(1, total_steps + 1):
                y_pred = rollout_predictions[:, step].detach().to("cpu")
                y_true = trjs[:, step].detach().to("cpu")
                y_samples = y_pred.unsqueeze(0)  # (K=1, N, C, ...)
                ps = compute_psrmse_three_bands(y_samples, y_true)

                for k, v in ps.items():
                    v = v.detach().to("cpu")
                    ps_sums[k] = v if k not in ps_sums else (ps_sums[k] + v)

                if step in horizons_set:
                    horizon_ps[step] = {k: v.detach().to("cpu") for k, v in ps.items()}

                if collect_viz and (step in viz_horizons_set):
                    Np = min(int(viz_num_trajectories), N_traj)
                    y_true_plot = y_true[:Np].clone()
                    y_pred_plot = y_pred[:Np].clone()
                    y_var_plot = torch.zeros_like(y_pred_plot)
                    y_err_plot = (y_pred_plot - y_true_plot).pow(2).clone()
                    viz_cache[step] = {
                        "target": y_true_plot,
                        "mean": y_pred_plot,
                        "var": y_var_plot,
                        "err": y_err_plot,
                    }

        if total_steps > 0:
            for k, v in ps_sums.items():
                metrics[f"rollout_{k}"] = v / float(total_steps)
        for step in sorted(horizons_set):
            if step <= 0 or step > len(timestep_mse):
                continue
            tag = _format_horizon_tag(step)
            metrics[f"rollout_rmse_{tag}"] = timestep_rmse[step - 1]
            metrics[f"rollout_nrmse_{tag}"] = (
                timestep_rmse[step - 1] / timestep_l2[step - 1]
            )
            metrics[f"rollout_vrmse_{tag}"] = timestep_rmse[step - 1] / torch.sqrt(
                timestep_var[step - 1]
            )

            for k, v in horizon_ps.get(step, {}).items():
                metrics[f"rollout_{k}_{tag}"] = v

        if collect_viz:
            self._last_rollout_viz_cache = viz_cache
        return metrics

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


def train_regression(cfg: DictConfig) -> None:
    """Train the regression baseline."""
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

    batch_size = int(cfg.training.batch_size)
    max_epochs = int(cfg.training.epochs)
    num_workers = int(cfg.training.num_workers)
    normalization_mode = str(cfg.training.normalization_mode)
    data_name_lower = get_dataset_data_name_lower(cfg)
    share_xy = should_share_xy_normalizer(
        data_name_lower=data_name_lower, training_cfg=cfg.training
    )

    train_loader, val_loader, test_loader, xn, yn = create_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        normalization_mode=normalization_mode,
        share_xy_normalizer=share_xy,
    )
    model = instantiate(strip_name_keys(cfg.model))
    regressor = Regression(
        model,
        cfg.optimizer,
        scheduler_config=getattr(cfg, "scheduler", None),
        x_normalizer=xn,
        y_normalizer=yn,
        eval_mode=getattr(getattr(cfg, "evaluation", None), "mode", None),
        stochastic_n_chunk=int(getattr(cfg.training, "stochastic_n_chunk", 64)),
    )
    if val_trjs is not None:
        regressor.val_trajectories = val_trjs
    if test_trjs is not None:
        regressor.test_trajectories = test_trjs
    if val_stochastic is not None:
        regressor.val_stochastic = val_stochastic
    if test_stochastic is not None:
        regressor.test_stochastic = test_stochastic
    dataset = getattr(cfg, "dataset", None)
    base_name = getattr(dataset, "data_name", "Experiment") if dataset else "Experiment"
    model_cfg = getattr(cfg, "model", None)
    model_name = (
        getattr(model_cfg, "model_name", None)
        or getattr(model_cfg, "mode_name", None)
        or getattr(cfg, "model_name", "RegressionModel")
    )
    task_name = "Regression"
    logging_cfg = getattr(cfg, "logging", None)
    project_name = f"{base_name}_GEN"
    run_name = make_run_name(str(model_name), task_name)

    wandb_logger = make_wandb_logger(cfg, project_name=project_name, run_name=run_name)
    eval_mode = Regression._normalize_eval_mode(
        getattr(getattr(cfg, "evaluation", None), "mode", None)
    )
    if eval_mode == "stochastic":
        if val_stochastic is None or test_stochastic is None:
            raise ValueError(
                "evaluation.mode='stochastic' requires dataset extras: val_stochastic and test_stochastic."
            )
        monitor_metric = "val_stochastic_mean_nrmse"
    elif eval_mode == "rollout":
        if val_trjs is None or test_trjs is None:
            raise ValueError(
                "evaluation.mode='rollout' requires dataset extras: val_trjs and test_trjs."
            )
        monitor_metric = "val_rollout_nrmse"
    else:
        monitor_metric = "val_loss"
    checkpoint_callback = make_checkpoint_callback(
        project_name=project_name,
        run_name=run_name,
        filename_base=str(model_name),
        filename=f"{model_name}-best-{{epoch:02d}}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = make_trainer(
        training_cfg=cfg.training,
        max_epochs=int(max_epochs),
        train_loader_len=len(train_loader),
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(regressor, train_loader, val_loader)
    ckpt_path = getattr(checkpoint_callback, "best_model_path", None) or "best"
    try:
        trainer.test(
            regressor, dataloaders=test_loader, ckpt_path=ckpt_path, weights_only=False
        )
    except TypeError:
        trainer.test(regressor, dataloaders=test_loader, ckpt_path=ckpt_path)
    finish_wandb(wandb_logger)
