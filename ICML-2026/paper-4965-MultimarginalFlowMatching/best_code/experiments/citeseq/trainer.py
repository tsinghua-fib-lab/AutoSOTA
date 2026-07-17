"""
CITE specific trainer for NeurIPS 2022 multimodal single-cell data.

Follows the evaluation protocol from previous work:
leave-one-out evaluation with all metrics computed
in original PCA space (after inverse normalization).

Author(s): Raghav Kansal
"""

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments import Trainer
from experiments.citeseq.data import CITE_IDX_TO_DAY
from experiments.evaluation import (
    compute_fgd,
    compute_mmd,
    compute_swd,
    compute_w1_distance,
    compute_w2_distance,
)
from experiments.plotting import plot_target_vs_learned
from experiments.singlecell import plotting


class CiteSeqTrainer(Trainer):
    """
    Trainer for CITE single-cell experiments. Extends the base Trainer with CITE-specific evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
        lr: float,
        # Training parameters
        epochs: int = 10,
        optimizer: str = "adam",
        grad_clip: float = 0.0,
        weight_decay: float = 0.0,
        lr_schedule: str | None = None,
        do_otp: bool = True,
        # Progressive loss weighting
        otp_alpha_type: str = "sigmoid",
        otp_alpha_slope: float = 6.0,
        otp_alpha_mean_scale: float = 1.0,
        # Sampling/evaluation
        sampling_steps: int = 50,
        ema_eval: bool = True,
        # Model
        potentials: OrderedDict | None = None,
        device: str = "cpu",
        # CITE-seq-specific
        scaler=None,
        reshuffle_each_epoch: bool = True,
        marginals: dict[int, Tensor] | None = None,
        train_times: list[int] | None = None,
        holdout_times: list[int] | None = None,
        eval_n_steps: int = 50,
        eval_num_samples: int | None = None,
        eval_metrics: list[str] | None = None,
        traj_skips: int | None = None,
        save_skips: int | None = None,
        num_1d_plot_samples: int = 5,
        animation_duration: int = 500,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            lr=lr,
            epochs=epochs,
            optimizer=optimizer,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            do_otp=do_otp,
            otp_alpha_type=otp_alpha_type,
            otp_alpha_slope=otp_alpha_slope,
            otp_alpha_mean_scale=otp_alpha_mean_scale,
            sampling_steps=sampling_steps,
            ema_eval=ema_eval,
            traj_skips=traj_skips,
            save_skips=save_skips,
            potentials=potentials,
            device=device,
        )

        self.scaler = scaler
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.marginals = marginals
        self.train_times = train_times
        self.holdout_times = holdout_times
        self.eval_n_steps = eval_n_steps
        self.eval_num_samples = eval_num_samples
        self.eval_metrics = set(eval_metrics or ["w1", "swd", "mmd", "fgd", "w2"])
        self.num_1d_plot_samples = num_1d_plot_samples
        self.animation_duration = animation_duration

        if traj_skips is None:
            self.traj_skips = 1 if epochs <= 10 else max(1, math.ceil(epochs / 10))
        else:
            self.traj_skips = traj_skips

        self.epoch_trajectories: list[np.ndarray] = []
        self.trajectory_t_eval: np.ndarray | None = None

        self.xtk_plot_dir = self.save_dir / "xtk_plots"
        self.xtk_plot_dir.mkdir(parents=True, exist_ok=True)

        self._init_marginal_metrics()

        if reshuffle_each_epoch:
            self.logger.info("Reshuffling cell pairings each epoch")

        # Log CITE-seq-specific info
        if train_times:
            self.logger.info(
                f"  Train times: {train_times} "
                f"(days {[CITE_IDX_TO_DAY[t] for t in train_times]})"
            )
        if holdout_times:
            self.logger.info(
                f"  Holdout times: {holdout_times} "
                f"(days {[CITE_IDX_TO_DAY[t] for t in holdout_times]})"
            )

    def _get_underlying_dataset(self):
        dataset = self.train_loader.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def _init_marginal_metrics(self):
        """Initialize metric tracking for all marginal times."""
        all_times = sorted(self.marginals.keys()) if self.marginals else []
        time_keys = [f"t{t}" for t in all_times]
        for m in self.eval_metrics:
            for t in time_keys:
                self.losses[f"{m}_{t}"] = []
        self.losses["metric_epochs"] = []

    def on_train_start(self):
        self._generate_trajectories(epoch=0, save_to_disk=True)
        self._plot_xtk_comparison(epoch=0)
        self._compute_metrics(epoch=0, do_mmd=True)

    def on_epoch_start(self, epoch: int, batch: Tensor | None = None):
        if self.reshuffle_each_epoch:
            dataset = self._get_underlying_dataset()
            if hasattr(dataset, "reshuffle"):
                dataset.reshuffle()

    def on_epoch_end(self, epoch: int, batch: Tensor | None = None):
        ep = epoch + 1
        is_eval = ((ep % self.traj_skips) == 0) or (epoch == self.epochs - 1)
        is_save = ((ep % self.save_skips) == 0) or (epoch == self.epochs - 1)
        if is_eval:
            self._generate_trajectories(epoch=ep, save_to_disk=is_save)
            if is_save:
                self._plot_xtk_comparison(epoch=ep)
            if epoch != self.epochs - 1:
                self._compute_metrics(epoch=ep, do_mmd=True)

    @torch.no_grad()
    def _generate_trajectories(self, epoch: int, save_to_disk: bool = True):
        if self.marginals is None or self.train_times is None:
            return

        self.model.eval()
        source_time = min(self.train_times)
        source = self.marginals[source_time]
        num_samples = (
            len(source)
            if self.eval_num_samples is None
            else min(self.eval_num_samples, len(source))
        )
        x0 = source[:num_samples].to(self.device)

        trajectories, t_eval = self.model.sample(x0, n_steps=self.eval_n_steps, ema=self.ema_eval)

        self.epoch_trajectories.append(trajectories.cpu().numpy())
        if self.trajectory_t_eval is None:
            self.trajectory_t_eval = t_eval.cpu().numpy()

        if save_to_disk:
            traj_dir = self.save_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                traj_dir / f"trajectories_epoch{epoch:04d}.npz",
                trajectories=trajectories.cpu().numpy(),
                t_eval=t_eval.cpu().numpy(),
            )

    @torch.no_grad()
    def _plot_xtk_comparison(self, epoch: int):
        if self.potentials is None or len(self.potentials) == 0:
            return

        n_samples = self.num_1d_plot_samples
        dataset = self._get_underlying_dataset()

        if hasattr(dataset, "get_ot_aligned_samples"):
            try:
                batch = dataset.get_ot_aligned_samples(n_samples=n_samples)
            except ValueError:
                batch = self._get_random_samples(n_samples)
        else:
            batch = self._get_random_samples(n_samples)

        otp_alpha = self.curriculum(epoch * len(self.train_loader))

        plot_target_vs_learned(
            model=self.model,
            batch=batch,
            potentials=self.potentials,
            otp_alpha=otp_alpha,
            n_samples=n_samples,
            name=f"xtk_epoch{epoch:04d}",
            plot_dir=self.xtk_plot_dir,
            show=False,
            device=self.device,
        )

    def _get_random_samples(self, n_samples: int) -> torch.Tensor:
        samples = []
        for t in sorted(self.train_times):
            marginal = self.marginals[t]
            indices = np.random.choice(len(marginal), size=n_samples, replace=False)
            samples.append(marginal[indices].cpu())
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def _compute_metrics(self, epoch: int, do_mmd: bool = False):
        """Compute metrics at all time points in original PCA space.

        All metrics (W1, SWD, MMD, FGD, W2) are computed after inverse-transforming
        both generated and ground truth samples back to original PCA space, matching
        the evaluation protocol of prior work.
        """
        if self.marginals is None or not self.epoch_trajectories:
            return

        metrics = self.eval_metrics
        raw_trajectories = self.epoch_trajectories[-1]
        t_eval = self.trajectory_t_eval

        if self.scaler is not None:
            shape = raw_trajectories.shape
            inv_trajectories = self.scaler.inverse_transform(
                raw_trajectories.reshape(-1, shape[-1])
            ).reshape(shape)
        else:
            inv_trajectories = raw_trajectories

        all_times = sorted(self.marginals.keys())
        time_min, time_max = min(all_times), max(all_times)

        def normalize_time(t):
            return (t - time_min) / (time_max - time_min)

        for t in all_times:
            t_norm = normalize_time(t)
            target_idx = np.argmin(np.abs(t_eval - t_norm))

            num_gt = (
                len(self.marginals[t])
                if self.eval_num_samples is None
                else min(self.eval_num_samples, len(self.marginals[t]))
            )
            gt_norm = self.marginals[t][:num_gt]
            gen_inv = torch.from_numpy(inv_trajectories[target_idx])

            if self.scaler is not None:
                gt_inv = torch.from_numpy(self.scaler.inverse_transform(gt_norm.cpu().numpy()))
            else:
                gt_inv = gt_norm

            if "w1" in metrics:
                self.losses[f"w1_t{t}"].append(compute_w1_distance(gen_inv, gt_inv))
            if "swd" in metrics:
                self.losses[f"swd_t{t}"].append(compute_swd(gen_inv, gt_inv))
            if "mmd" in metrics and do_mmd:
                self.losses[f"mmd_t{t}"].append(compute_mmd(gen_inv, gt_inv))
            if "fgd" in metrics:
                self.losses[f"fgd_t{t}"].append(compute_fgd(gen_inv, gt_inv))
            if "w2" in metrics:
                w2_dim = min(gen_inv.shape[-1], 10)
                self.losses[f"w2_t{t}"].append(
                    compute_w2_distance(gen_inv[:, :w2_dim], gt_inv[:, :w2_dim])
                )

        self.losses["metric_epochs"].append(epoch)

        metric_order = ["swd", "mmd", "fgd", "w1"]
        log_parts = [f"Epoch {epoch} metrics:"]
        for t in all_times:
            vals = []
            for m in metric_order:
                key = f"{m}_t{t}"
                if self.losses.get(key):
                    vals.append(f"{m.upper()}={self.losses[key][-1]:.4f}")
            if vals:
                day_str = CITE_IDX_TO_DAY.get(t, t)
                log_parts.append(f"t{t}(day{day_str}: {', '.join(vals)})")
        self.logger.info(" ".join(log_parts))

    def post_training(self, show: bool = False, create_animation: bool = False) -> Path:
        self._compute_metrics(epoch=self.epochs, do_mmd=True)
        save_path = super().post_training(show=show)

        if self.marginals is not None and self.epoch_trajectories:
            self._plot_final_trajectories()

        if create_animation and self.epoch_trajectories:
            self.create_animations()

        self.append_to_master_csv()
        return save_path

    def _plot_final_trajectories(self):
        if not self.epoch_trajectories or self.trajectory_t_eval is None:
            return

        trajectories = torch.from_numpy(self.epoch_trajectories[-1])
        all_times = sorted(self.marginals.keys())
        gt_marginals = {t: self.marginals[t] for t in all_times}

        plotting.plot_pca_trajectories(
            trajectories=trajectories,
            time_points=self.trajectory_t_eval,
            ground_truth_marginals=gt_marginals,
            plot_times=all_times,
            pcs=(0, 1),
            num_trajectories=0,
            save_path=self.save_dir / "trajectories_pc1_pc2.pdf",
            show=False,
        )

        dim = trajectories.shape[-1]
        if dim >= 4:
            plotting.plot_pca_trajectories(
                trajectories=trajectories,
                time_points=self.trajectory_t_eval,
                ground_truth_marginals=gt_marginals,
                plot_times=all_times,
                pcs=(2, 3),
                num_trajectories=0,
                save_path=self.save_dir / "trajectories_pc3_pc4.pdf",
                show=False,
            )

    def create_animations(self, num_trajectories: int = 100, pcs: tuple[int, int] = (0, 1)):
        if not self.epoch_trajectories or self.marginals is None or self.trajectory_t_eval is None:
            self.logger.warning("Skipping animation: missing data")
            return

        all_times = sorted(self.marginals.keys())
        gt_marginals = {t: self.marginals[t] for t in all_times}

        plotting.create_trajectory_animation(
            epoch_trajectories=self.epoch_trajectories,
            ground_truth_marginals=gt_marginals,
            trajectory_t_eval=self.trajectory_t_eval,
            save_path=self.save_dir / "trajectories_animation.gif",
            traj_skips=self.traj_skips,
            num_trajectories=num_trajectories,
            pcs=pcs,
            duration=self.animation_duration,
        )

    def plot_losses(self, log: bool = False, show: bool = False):
        plotting.plot_losses(
            self.losses,
            name="losses" + ("_log" if log else ""),
            plot_dir=self.save_dir,
            log=log,
            show=show,
        )

    def _find_best_epoch_idx(self) -> int | None:
        """Find the best epoch based on holdout time FGD/W1."""
        holdout_keys = [f"fgd_t{t}" for t in self.holdout_times]
        valid_keys = [k for k in holdout_keys if k in self.losses and self.losses[k]]

        if not valid_keys:
            # Fall back to W1 at holdout times
            holdout_keys = [f"w1_t{t}" for t in self.holdout_times]
            valid_keys = [k for k in holdout_keys if k in self.losses and self.losses[k]]

        if not valid_keys:
            return None

        n = min(len(self.losses[k]) for k in valid_keys)
        avg_metric = [
            sum(self.losses[k][i] for k in valid_keys) / len(valid_keys) for i in range(n)
        ]
        return int(np.argmin(avg_metric))

    def append_to_master_csv(self, master_csv_path: Path | None = None):
        """Append best-epoch metrics to master results CSV."""
        if master_csv_path is None:
            master_csv_path = self.save_dir.parent.parent / "master_results.csv"

        model_dir = f"{self.save_dir.parent.name}/{self.save_dir.name}"
        best_idx = self._find_best_epoch_idx()

        if best_idx is None:
            self.logger.warning("No holdout metrics found, skipping master CSV")
            return

        metric_epochs = self.losses.get("metric_epochs", [])
        best_epoch = metric_epochs[best_idx] if best_idx < len(metric_epochs) else best_idx

        row = {"model_dir": model_dir, "best_epoch": best_epoch}

        all_times = sorted(self.marginals.keys()) if self.marginals else []
        for t in all_times:
            for metric in ["w1", "swd", "fgd", "w2"]:
                key = f"{metric}_t{t}"
                if key in self.losses and best_idx < len(self.losses[key]):
                    row[key] = self.losses[key][best_idx]

        if self.losses.get("train_loss"):
            row["train_loss"] = self.losses["train_loss"][
                min(best_epoch, len(self.losses["train_loss"]) - 1)
            ]
        if self.losses.get("val_loss"):
            row["val_loss"] = self.losses["val_loss"][
                min(best_epoch, len(self.losses["val_loss"]) - 1)
            ]

        holdout_fgd_keys = [f"fgd_t{t}" for t in self.holdout_times]
        holdout_vals = [row.get(k, 0) for k in holdout_fgd_keys if k in row]
        avg_holdout = sum(holdout_vals) / max(len(holdout_vals), 1)
        self.logger.info(f"Best epoch: {best_epoch} (avg holdout FGD = {avg_holdout:.4f})")

        df_row = pd.DataFrame([row])

        if master_csv_path.exists():
            existing_df = pd.read_csv(master_csv_path)
            combined_df = pd.concat([existing_df, df_row], ignore_index=True)
            combined_df.to_csv(master_csv_path, index=False)
        else:
            master_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_row.to_csv(master_csv_path, index=False)

        self.logger.info(f"Appended results to {master_csv_path}")
