"""
Single-cell specific trainer for Embryoid Body (EB) data.

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
from experiments.evaluation import (
    compute_fgd,
    compute_mmd,
    compute_swd,
    compute_w1_distance,
    compute_w2_distance,
)
from experiments.plotting import plot_target_vs_learned
from experiments.singlecell import plotting


class EBTrainer(Trainer):
    """
    Trainer for Embryoid Body (EB) single-cell experiments.

    Extends the base Trainer with EB-specific plotting and evaluation metrics.
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
        # EB-specific
        scaler=None,
        reshuffle_each_epoch: bool = True,
        marginals: dict[int, Tensor] | None = None,
        train_times: list[int] | None = None,
        holdout_times: list[int] | None = None,
        eval_n_steps: int = 50,
        eval_num_samples: int = 2000,
        eval_metrics: list[str] | None = None,
        traj_skips: int | None = None,
        num_1d_plot_samples: int = 5,
        animation_duration: int = 500,
    ):
        """Initialize the EB trainer."""
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

        # Auto-compute traj_skips
        if traj_skips is None:
            self.traj_skips = 1 if epochs <= 10 else max(1, math.ceil(epochs / 10))
        else:
            self.traj_skips = traj_skips

        # Trajectory storage
        self.epoch_trajectories: list[np.ndarray] = []
        self.trajectory_t_eval: np.ndarray | None = None

        # Plotting directories
        self.xtk_plot_dir = self.save_dir / "xtk_plots"
        self.xtk_plot_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metric tracking
        self._init_marginal_metrics()

        # Log configuration
        if reshuffle_each_epoch:
            self.logger.info("Reshuffling cell pairings each epoch")

    def _get_underlying_dataset(self):
        """Get the underlying dataset, handling random_split wrapper."""
        dataset = self.train_loader.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def _init_marginal_metrics(self):
        """Initialize metric tracking for all marginal times."""
        all_times = sorted(self.marginals.keys()) if self.marginals else []
        time_keys = [f"t{t}" for t in all_times]
        if 2 in (self.marginals or {}) and 4 in (self.marginals or {}):
            time_keys.append("t2_t4")
        for m in self.eval_metrics:
            for t in time_keys:
                self.losses[f"{m}_{t}"] = []
        self.losses["metric_epochs"] = []

    def on_train_start(self):
        """Save initial state before training."""
        self._save_epoch_trajectories(epoch=0)
        self._plot_xtk_comparison(epoch=0)
        self._compute_metrics(epoch=0, do_mmd=True)

    def on_epoch_start(self, epoch: int, batch: Tensor | None = None):
        """Reshuffle cell pairings."""
        if self.reshuffle_each_epoch:
            dataset = self._get_underlying_dataset()
            if hasattr(dataset, "reshuffle"):
                dataset.reshuffle()

    def on_epoch_end(self, epoch: int, batch: Tensor | None = None):
        """Save trajectories and compute metrics."""
        if ((epoch + 1) % self.traj_skips == 0) or (epoch == self.epochs - 1):
            self._save_epoch_trajectories(epoch=epoch + 1)
            self._plot_xtk_comparison(epoch=epoch + 1)
            if epoch != self.epochs - 1:
                self._compute_metrics(epoch=epoch + 1, do_mmd=True)

    @torch.no_grad()
    def _save_epoch_trajectories(self, epoch: int):
        """Sample and save trajectories."""
        if self.marginals is None or self.train_times is None:
            return

        self.model.eval()
        source_time = min(self.train_times)
        source = self.marginals[source_time]
        num_samples = min(self.eval_num_samples, len(source))
        x0 = source[:num_samples].to(self.device)

        trajectories, t_eval = self.model.sample(x0, n_steps=self.eval_n_steps, ema=self.ema_eval)

        self.epoch_trajectories.append(trajectories.cpu().numpy())
        if self.trajectory_t_eval is None:
            self.trajectory_t_eval = t_eval.cpu().numpy()

        # Save to disk
        traj_dir = self.save_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            traj_dir / f"trajectories_epoch{epoch:04d}.npz",
            trajectories=trajectories.cpu().numpy(),
            t_eval=t_eval.cpu().numpy(),
        )

    @torch.no_grad()
    def _plot_xtk_comparison(self, epoch: int):
        """Plot X_tk comparison."""
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
        """Get random samples from each marginal."""
        samples = []
        for t in sorted(self.train_times):
            marginal = self.marginals[t]
            indices = np.random.choice(len(marginal), size=n_samples, replace=False)
            samples.append(marginal[indices].cpu())
        return torch.stack(samples, dim=1)  # (n_samples, n_times, dim)

    @torch.no_grad()
    def _compute_metrics(self, epoch: int, do_mmd: bool = False):
        """Compute metrics at all time points.

        W1 is computed in normalized (standardized) space to match the protocol of Neklyudov et al. (2024), WLF.
        All other metrics use inverse-transformed
        (original PCA) space. Only metrics listed in ``self.eval_metrics`` are computed.
        """
        if self.marginals is None or not self.epoch_trajectories:
            return

        metrics = self.eval_metrics
        raw_trajectories = self.epoch_trajectories[-1]  # normalized space
        t_eval = self.trajectory_t_eval

        need_inv = metrics & {"swd", "mmd", "fgd"}
        if need_inv and self.scaler is not None:
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

        generated_samples = {}
        generated_samples_norm = {}

        for t in all_times:
            t_norm = normalize_time(t)
            target_idx = np.argmin(np.abs(t_eval - t_norm))

            gt_norm = self.marginals[t][: self.eval_num_samples]
            gen_norm = torch.from_numpy(raw_trajectories[target_idx])

            if "w1" in metrics:
                self.losses[f"w1_t{t}"].append(compute_w1_distance(gen_norm, gt_norm))
            if "w2" in metrics:
                self.losses[f"w2_t{t}"].append(compute_w2_distance(gen_norm, gt_norm))

            if need_inv:
                if self.scaler is not None:
                    gt_inv = torch.from_numpy(self.scaler.inverse_transform(gt_norm.cpu().numpy()))
                else:
                    gt_inv = gt_norm
                gen_inv = torch.from_numpy(inv_trajectories[target_idx])

                if "swd" in metrics:
                    self.losses[f"swd_t{t}"].append(compute_swd(gen_inv, gt_inv))
                if "mmd" in metrics and do_mmd:
                    self.losses[f"mmd_t{t}"].append(compute_mmd(gen_inv, gt_inv))
                if "fgd" in metrics:
                    self.losses[f"fgd_t{t}"].append(compute_fgd(gen_inv, gt_inv))

                generated_samples[t] = gen_inv
            generated_samples_norm[t] = gen_norm

        # Combined t2+t4 metrics
        if 2 in generated_samples_norm and 4 in generated_samples_norm:
            if "w1" in metrics:
                gen_norm_combined = torch.cat(
                    [generated_samples_norm[2], generated_samples_norm[4]]
                )
                gt_norm_combined = torch.cat(
                    [
                        self.marginals[2][: self.eval_num_samples],
                        self.marginals[4][: self.eval_num_samples],
                    ]
                )
                self.losses["w1_t2_t4"].append(
                    compute_w1_distance(gen_norm_combined, gt_norm_combined)
                )

            if need_inv and 2 in generated_samples and 4 in generated_samples:
                gen_combined = torch.cat([generated_samples[2], generated_samples[4]])
                gt_2 = self.marginals[2][: self.eval_num_samples]
                gt_4 = self.marginals[4][: self.eval_num_samples]
                if self.scaler is not None:
                    gt_2 = torch.from_numpy(self.scaler.inverse_transform(gt_2.cpu().numpy()))
                    gt_4 = torch.from_numpy(self.scaler.inverse_transform(gt_4.cpu().numpy()))
                gt_combined = torch.cat([gt_2, gt_4])

                if "swd" in metrics:
                    self.losses["swd_t2_t4"].append(compute_swd(gen_combined, gt_combined))
                if "mmd" in metrics and do_mmd:
                    self.losses["mmd_t2_t4"].append(compute_mmd(gen_combined, gt_combined))
                if "fgd" in metrics:
                    self.losses["fgd_t2_t4"].append(compute_fgd(gen_combined, gt_combined))

        self.losses["metric_epochs"].append(epoch)

        # Log summary — show all computed metrics per time
        metric_order = ["w2", "swd", "mmd", "fgd", "w1"]
        log_parts = [f"Epoch {epoch} metrics:"]
        for t in all_times:
            vals = []
            for m in metric_order:
                key = f"{m}_t{t}"
                if self.losses.get(key):
                    vals.append(f"{m.upper()}={self.losses[key][-1]:.4f}")
            if vals:
                log_parts.append(f"t{t}({', '.join(vals)})")
        self.logger.info(" ".join(log_parts))

    def post_training(self, show: bool = False, create_animation: bool = False) -> Path:
        """Run post-training tasks."""
        self._compute_metrics(epoch=self.epochs, do_mmd=True)
        save_path = super().post_training(show=show)

        # Plot trajectories
        if self.marginals is not None and self.epoch_trajectories:
            self._plot_final_trajectories()

        # Create animation
        if create_animation and self.epoch_trajectories:
            self.create_animations()

        # Append to master CSV
        self.append_to_master_csv()

        return save_path

    def _plot_final_trajectories(self):
        """Plot final trajectory visualization."""
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

        # Also plot PC3 vs PC4 if dimension is high enough
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
        """Create animated GIF showing trajectory evolution across training."""
        if not self.epoch_trajectories or self.marginals is None or self.trajectory_t_eval is None:
            self.logger.warning("Skipping animation: missing trajectories, marginals, or t_eval")
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
        """Override to use single-cell specific loss plotting."""
        plotting.plot_losses(
            self.losses,
            name="losses" + ("_log" if log else ""),
            plot_dir=self.save_dir,
            log=log,
            show=show,
        )

    def _find_best_epoch_idx(self) -> int | None:
        """Index of epoch with lowest avg MMD = (mmd_t1 + mmd_t3 + 2*mmd_t2_t4) / 4.

        Falls back to (fgd_t1 + fgd_t3)/2 if MMD is unavailable.
        """
        if all(self.losses.get(k) for k in ("mmd_t1", "mmd_t3", "mmd_t2_t4")):
            mmd_t1 = self.losses["mmd_t1"]
            mmd_t3 = self.losses["mmd_t3"]
            mmd_rest = self.losses["mmd_t2_t4"]
            n = min(len(mmd_t1), len(mmd_t3), len(mmd_rest))
            avg = [(mmd_t1[i] + mmd_t3[i] + 2 * mmd_rest[i]) / 4 for i in range(n)]
            return int(np.argmin(avg))

        fgd_t1 = self.losses.get("fgd_t1", [])
        fgd_t3 = self.losses.get("fgd_t3", [])
        if not fgd_t1 or not fgd_t3:
            return None
        n = min(len(fgd_t1), len(fgd_t3))
        avg_fgd = [(fgd_t1[i] + fgd_t3[i]) / 2 for i in range(n)]
        return int(np.argmin(avg_fgd))

    def append_to_master_csv(self, master_csv_path: Path | None = None):
        """Append best-epoch metrics to master results CSV.

        Selects the epoch with the lowest avg MMD = (mmd_t1+mmd_t3+2*mmd_t2_t4)/4
        (falls back to avg of fgd_t1, fgd_t3 if MMD unavailable) and saves all
        metrics from that epoch.
        """
        if master_csv_path is None:
            master_csv_path = self.save_dir.parent.parent / "master_results.csv"

        model_dir = f"{self.save_dir.parent.name}/{self.save_dir.name}"
        best_idx = self._find_best_epoch_idx()

        if best_idx is None:
            self.logger.warning("No metric history found, skipping master CSV")
            return

        metric_epochs = self.losses.get("metric_epochs", [])
        best_epoch = metric_epochs[best_idx] if best_idx < len(metric_epochs) else best_idx

        row = {"model_dir": model_dir, "best_epoch": best_epoch}

        # Add all metric values from the best epoch
        metric_prefixes = ["w1", "swd", "mmd", "fgd", "w2"]
        time_suffixes = ["t1", "t2", "t3", "t4", "t2_t4"]
        for metric in metric_prefixes:
            for time in time_suffixes:
                key = f"{metric}_{time}"
                if key in self.losses and best_idx < len(self.losses[key]):
                    row[key] = self.losses[key][best_idx]

        # Average MMD across all 4 marginals (t1, t3 and rest counts as 2)
        if all(f"mmd_{t}" in row for t in ("t1", "t3", "t2_t4")):
            row["avg_mmd"] = (row["mmd_t1"] + row["mmd_t3"] + 2 * row["mmd_t2_t4"]) / 4

        # Add training metrics from the best epoch (use closest available)
        if self.losses.get("train_loss"):
            row["train_loss"] = self.losses["train_loss"][
                min(best_epoch, len(self.losses["train_loss"]) - 1)
            ]
        if self.losses.get("val_loss"):
            row["val_loss"] = self.losses["val_loss"][
                min(best_epoch, len(self.losses["val_loss"]) - 1)
            ]

        if "avg_mmd" in row:
            self.logger.info(f"Best epoch: {best_epoch} (avg MMD = {row['avg_mmd']:.4f})")
        else:
            self.logger.info(
                f"Best epoch: {best_epoch} "
                f"(avg fgd_t1+t3 = {(row.get('fgd_t1', 0) + row.get('fgd_t3', 0)) / 2:.4f})"
            )

        df_row = pd.DataFrame([row])

        if master_csv_path.exists():
            existing_df = pd.read_csv(master_csv_path)
            combined_df = pd.concat([existing_df, df_row], ignore_index=True)
            combined_df.to_csv(master_csv_path, index=False)
        else:
            master_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_row.to_csv(master_csv_path, index=False)

        self.logger.info(f"Appended results to {master_csv_path}")
