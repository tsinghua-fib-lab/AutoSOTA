import re
import warnings
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from data import create_dataloaders
from train.diffusion.dm import compute_ensemble_metrics
from train.metrics import (
    compute_nrmse,
    compute_psrmse_three_bands,
    compute_stochastic_mean_std_metrics,
)
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


def _format_horizon_tag(step: int, width: int = 3) -> str:
    """Stable lexicographic tag for horizons: 1 -> t001."""
    return f"t{int(step):0{int(width)}d}"


def _enable_dropout_only(m: nn.Module) -> None:
    """Enable dropout layers during eval for MC-dropout (keep everything else in eval)."""
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        m.train()


def _count_dropout_modules(model: nn.Module) -> int:
    return sum(
        1
        for m in model.modules()
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout))
    )


class DropoutRegression(StripMetadataModule):
    """Regression baseline with MC-dropout ensembles at evaluation time."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig | None = None,
        *,
        x_normalizer: Any = None,
        y_normalizer: Any = None,
        eval_mode: str | None = None,
        samples_per_example: int = 8,
        test_samples_per_example: int | None = None,
        rollout_k_chunk: int = 8,
        rollout_traj_chunk: int | None = 16,
        stochastic_n_chunk: int = 32,
        stochastic_k_chunk: int = 4,
        viz_num_trajectories_default: int = 3,
        viz_horizon_percentiles_default: Sequence[float] = (1, 20, 40, 60, 80, 100),
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "optimizer_config",
                "scheduler_config",
                "x_normalizer",
                "y_normalizer",
            ]
        )

        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.eval_mode = self._normalize_eval_mode(eval_mode)

        self.samples_per_example = int(samples_per_example)
        self.test_samples_per_example = test_samples_per_example
        self.rollout_k_chunk = (
            int(rollout_k_chunk) if rollout_k_chunk is not None else 8
        )
        self.rollout_traj_chunk = (
            int(rollout_traj_chunk) if rollout_traj_chunk is not None else 16
        )
        self.stochastic_n_chunk = max(1, int(stochastic_n_chunk))
        self.stochastic_k_chunk = max(1, int(stochastic_k_chunk))
        self.viz_num_trajectories_default = int(viz_num_trajectories_default)
        self.viz_horizon_percentiles_default = tuple(
            float(p) for p in viz_horizon_percentiles_default
        )

        self.criterion = nn.MSELoss()
        self.dropout_module_count = _count_dropout_modules(self.model)
        if self.dropout_module_count <= 0:
            warnings.warn(
                "DropoutRegression was initialized with no nn.Dropout/Dropout2d/Dropout3d/AlphaDropout "
                "modules. MC-dropout samples will be deterministic unless the model implements stochastic "
                "behavior outside standard PyTorch dropout modules.",
                RuntimeWarning,
                stacklevel=2,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
            return mse_original

        mse = self.criterion(out, y)
        self.log("val_loss", mse, prog_bar=True, sync_dist=True)
        return mse

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
            return mse_original

        mse = self.criterion(out, y)
        self.log("test_loss", mse, prog_bar=True, sync_dist=True)
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
                self.log(f"val_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
            self.log_stochastic_mean_std_viz_cache(
                self._last_stochastic_viz_cache, prefix="val_stochastic_viz"
            )
        elif self.eval_mode == "rollout":
            val_trjs = getattr(self, "val_trajectories", None)
            if val_trjs is None:
                return
            metrics = self.evaluate_autoregressive_rollout_ensemble(
                val_trjs,
                log_all_horizons=False,
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
        elif self.eval_mode == "inverse":
            val_loader = self._first_dataloader(
                getattr(self.trainer, "val_dataloaders", None)
            )
            if val_loader is None:
                return
            metrics = self.evaluate_inverse_operator_learning(val_loader)
            for k, v in metrics.items():
                self.log(f"val_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)

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
                self.log(f"test_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
            self.log_stochastic_mean_std_viz_cache(
                self._last_stochastic_viz_cache, prefix="test_stochastic_viz"
            )
        elif self.eval_mode == "rollout":
            test_trjs = getattr(self, "test_trajectories", None)
            if test_trjs is None:
                return
            metrics = self.evaluate_autoregressive_rollout_ensemble(
                test_trjs,
                num_samples_per_example=self.test_samples_per_example,
                log_all_horizons=True,
                horizon_percentiles=self.viz_horizon_percentiles_default,
                collect_viz=True,
                viz_num_trajectories=self.viz_num_trajectories_default,
                viz_horizon_percentiles=self.viz_horizon_percentiles_default,
            )
            for k, v in metrics.items():
                if "_t" in k:
                    continue
                self.log(f"test_{k}", v, prog_bar=False, on_epoch=True, sync_dist=False)
            self._log_test_metric_tables(metrics)
            self.log_rollout_viz_cache(
                self._last_rollout_viz_cache, prefix="test_rollout_viz"
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
            val = (
                float(v.detach().cpu().item())
                if isinstance(v, torch.Tensor)
                else float(v)
            )
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
            target = snap["target"]
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
        gen_samples_1d = viz_cache.get("gen_samples_1d", None)
        tgt_samples_1d = viz_cache.get("tgt_samples_1d", None)
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
    def _mc_predict_y_model(
        self, x_model: torch.Tensor, *, num_samples: int
    ) -> torch.Tensor:
        """Return MC-dropout predictions in model space with shape (K,B,C,...)."""
        K = max(1, int(num_samples))
        self.model.eval()
        self.model.apply(_enable_dropout_only)

        if K == 1:
            return self.model(x_model).unsqueeze(0)

        x_rep = x_model.repeat_interleave(K, dim=0)
        y_rep = self.model(x_rep)
        B = int(x_model.shape[0])
        return y_rep.view(B, K, *y_rep.shape[1:]).transpose(0, 1).contiguous()

    @torch.inference_mode()
    def evaluate_stochastic_operator_learning(
        self,
        payload: Any,
        *,
        num_pred_samples: int | None = None,
        max_examples: int | None = 256,
        eps: float = 1e-12,
        collect_viz: bool = False,
        viz_num_examples: int = 3,
    ) -> Dict[str, torch.Tensor]:
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
            """Return decoded predictions on CPU with shape (K,B,C,...)."""
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
                    y_model_samples = self._mc_predict_y_model(
                        xb_model, num_samples=kk
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
                    return torch.cat([y0, y1], dim=1)
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

    @torch.inference_mode()
    def evaluate_inverse_operator_learning(
        self,
        dataloader: Any,
        *,
        num_pred_samples: int | None = None,
        max_examples: int | None = 256,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate inverse operator learning with ensemble predictions."""
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

        self.eval()
        self._move_normalizers_to_device()

        sums: Dict[str, torch.Tensor] = {}
        total_n = 0
        remaining = None if max_examples is None else int(max_examples)

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

                xb_model = xb.to(self.device, non_blocking=True)
                y_model_samples = self._mc_predict_y_model(
                    xb_model, num_samples=K
                )  # (K,B,C,...)
                y_model_flat = y_model_samples.reshape(
                    K * bsz, *y_model_samples.shape[2:]
                )

                y_pred_flat = self.y_normalizer.decode(y_model_flat).to(torch.float32)
                y_pred = y_pred_flat.reshape(K, bsz, *y_pred_flat.shape[1:]).to("cpu")

                y_true_orig = (
                    self.y_normalizer.decode(yb.to(self.device, non_blocking=True))
                    .to(torch.float32)
                    .to("cpu")
                )

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

                del xb_model, y_model_samples, y_model_flat, y_pred_flat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if remaining is not None:
                remaining -= use_B

        if total_n <= 0:
            return {}
        return {k: v / float(total_n) for k, v in sums.items()}

    @torch.inference_mode()
    def evaluate_autoregressive_rollout_ensemble(
        self,
        trajectories: torch.Tensor,
        *,
        max_trajectories: int | None = 64,
        horizons: Iterable[int] | None = None,
        horizon_percentiles: Iterable[float] = (1, 20, 40, 60, 80, 100),
        num_samples_per_example: int | None = None,
        log_all_horizons: bool = False,
        collect_viz: bool = False,
        viz_num_trajectories: int = 3,
        viz_horizons: Iterable[int] | None = None,
        viz_horizon_percentiles: Iterable[float] | None = None,
    ) -> Dict[str, torch.Tensor]:
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
            trjs = trjs_in
        else:
            raise ValueError(f"Unexpected trajectories shape {tuple(trjs_in.shape)}")

        if max_trajectories is not None:
            trjs = trjs[: int(max_trajectories)]

        N_traj, T = int(trjs.shape[0]), int(trjs.shape[1])
        if T < 2:
            return {}

        total_steps = T - 1
        K = max(1, int(num_samples_per_example or self.samples_per_example))
        if log_all_horizons:
            horizons_set = set(range(1, total_steps + 1))
        elif horizons is None:
            pct_to_step = _steps_from_percentiles(
                total_steps, tuple(horizon_percentiles)
            )
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
                viz_pct_to_step = _steps_from_percentiles(total_steps, tuple(vp))
                viz_horizons_set = set(viz_pct_to_step.values())
            viz_cache: Dict[int, Dict[str, torch.Tensor]] = {}
            viz_num = max(1, int(viz_num_trajectories))
        else:
            viz_horizons_set = set()
            viz_cache = {}
            viz_num = 0
        cond_norm = (
            self.x_normalizer if self.x_normalizer is not None else self.y_normalizer
        )
        if cond_norm is None or self.y_normalizer is None:
            raise ValueError(
                "Rollout requires both a conditioning normalizer (x_normalizer or y_normalizer) and y_normalizer."
            )
        y0_orig_cpu = trjs[:, 0].to("cpu")  # (N,C,...)
        x_prev_orig_cpu = (
            y0_orig_cpu.unsqueeze(0)
            .repeat(K, 1, *([1] * (y0_orig_cpu.ndim - 1)))
            .contiguous()
        )

        y_true_shape = trjs[:, 1].shape[1:]
        y_next_orig_cpu = torch.empty(
            (K, N_traj, *y_true_shape), device="cpu", dtype=torch.float32
        )

        k_chunk = max(1, int(self.rollout_k_chunk))
        traj_chunk = max(1, int(self.rollout_traj_chunk))

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
        self.model.eval()
        self.model.apply(_enable_dropout_only)

        for t in range(total_steps):
            y_true_cpu = trjs[:, t + 1].detach().to("cpu")

            for k0 in range(0, K, k_chunk):
                k1 = min(K, k0 + k_chunk)
                for s0 in range(0, N_traj, traj_chunk):
                    s1 = min(N_traj, s0 + traj_chunk)

                    cond_orig_cpu = x_prev_orig_cpu[k0:k1, s0:s1]  # (kB, sB, C,...)
                    flat = (k1 - k0) * (s1 - s0)
                    cond_flat_cpu = cond_orig_cpu.reshape(
                        flat, *cond_orig_cpu.shape[2:]
                    )

                    cond_flat_gpu = cond_flat_cpu.to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    cond_model = cond_norm.encode(cond_flat_gpu)
                    y_model_flat = self.model(cond_model)
                    y_orig_flat = self.y_normalizer.decode(y_model_flat).to(
                        torch.float32
                    )
                    y_orig_block = y_orig_flat.reshape(
                        (k1 - k0), (s1 - s0), *y_orig_flat.shape[1:]
                    )
                    y_next_orig_cpu[k0:k1, s0:s1] = y_orig_block.detach().to("cpu")

                    del (
                        cond_flat_gpu,
                        cond_model,
                        y_model_flat,
                        y_orig_flat,
                        y_orig_block,
                    )
            x_prev_orig_cpu.copy_(y_next_orig_cpu)

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


def train_dropout(cfg: DictConfig) -> None:
    """Train the MC-dropout regression baseline."""
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

    drop_cfg = getattr(cfg, "dropout", None)
    samples_per_example = (
        int(getattr(drop_cfg, "samples_per_example", 8)) if drop_cfg is not None else 8
    )
    test_samples_per_example = (
        getattr(drop_cfg, "test_samples_per_example", None)
        if drop_cfg is not None
        else None
    )
    rollout_k_chunk = (
        int(getattr(drop_cfg, "rollout_k_chunk", 8)) if drop_cfg is not None else 8
    )
    rollout_traj_chunk = (
        getattr(drop_cfg, "rollout_traj_chunk", None) if drop_cfg is not None else 16
    )
    stochastic_n_chunk = (
        int(getattr(drop_cfg, "stochastic_n_chunk", 32)) if drop_cfg is not None else 32
    )
    stochastic_k_chunk = (
        int(getattr(drop_cfg, "stochastic_k_chunk", 4)) if drop_cfg is not None else 4
    )
    viz_num_trajectories = (
        int(getattr(drop_cfg, "viz_num_trajectories", 3)) if drop_cfg is not None else 3
    )
    viz_pcts = (
        getattr(drop_cfg, "viz_horizon_percentiles", (1, 20, 40, 60, 80, 100))
        if drop_cfg is not None
        else (
            1,
            20,
            40,
            60,
            80,
            100,
        )
    )

    module = DropoutRegression(
        model=model,
        optimizer_config=cfg.optimizer,
        scheduler_config=getattr(cfg, "scheduler", None),
        x_normalizer=xn,
        y_normalizer=yn,
        eval_mode=getattr(getattr(cfg, "evaluation", None), "mode", None),
        samples_per_example=samples_per_example,
        test_samples_per_example=test_samples_per_example,
        rollout_k_chunk=rollout_k_chunk,
        rollout_traj_chunk=rollout_traj_chunk,
        stochastic_n_chunk=stochastic_n_chunk,
        stochastic_k_chunk=stochastic_k_chunk,
        viz_num_trajectories_default=viz_num_trajectories,
        viz_horizon_percentiles_default=viz_pcts,
    )

    if val_trjs is not None:
        module.val_trajectories = val_trjs
    if test_trjs is not None:
        module.test_trajectories = test_trjs
    if val_stochastic is not None:
        module.val_stochastic = val_stochastic
    if test_stochastic is not None:
        module.test_stochastic = test_stochastic

    dataset = getattr(cfg, "dataset", None)
    base_name = getattr(dataset, "data_name", "Experiment") if dataset else "Experiment"
    model_cfg = getattr(cfg, "model", None)
    model_name = (
        getattr(model_cfg, "model_name", None)
        or getattr(model_cfg, "mode_name", None)
        or getattr(cfg, "model_name", "DropoutModel")
    )
    task_name = "Dropout"
    project_name = f"{base_name}_GEN"
    run_name = make_run_name(str(model_name), task_name)
    wandb_logger = make_wandb_logger(cfg, project_name=project_name, run_name=run_name)

    eval_mode = DropoutRegression._normalize_eval_mode(
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
        callbacks=[ckpt_best],
    )

    trainer.fit(module, train_loader, val_loader)

    ckpt_path = getattr(ckpt_best, "best_model_path", None) or "best"
    try:
        trainer.test(
            module, dataloaders=test_loader, ckpt_path=ckpt_path, weights_only=False
        )
    except TypeError:
        trainer.test(module, dataloaders=test_loader, ckpt_path=ckpt_path)

    finish_wandb(wandb_logger)
