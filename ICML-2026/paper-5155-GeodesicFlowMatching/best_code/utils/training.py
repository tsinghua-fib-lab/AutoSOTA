from pathlib import Path
import torch
from utils.wandb_utils import log_metrics
import numpy as np
import plotly.graph_objects as go
from cleanup_ssps.model import MLP_Small, ResidualMLP
from cleanup_ssps.run import FlowTrainer, FeedforwardTrainer

_DEFAULT_SAMPLING_MODES = [
    "geo_det",
    "geo_amb_const",
    "geo_tan_const",
    "geo_amb_sb",
    "geo_tan_sb",
    "euc_det",
    "euc_ot",
    "euc_sb",
]


class TrainingManager:
    def __init__(self, ssp_space, trainer_configs, ssp_config,
                 sampling_modes=None):
        self.ssp_space       = ssp_space
        self.trainer_configs = trainer_configs
        self.ssp_config      = ssp_config
        self.device          = trainer_configs.get("device", "cpu")

        if sampling_modes is not None:
            self.sampling_modes = list(sampling_modes)
        else:
            self.sampling_modes = list(
                trainer_configs.get("sampling_modes") or _DEFAULT_SAMPLING_MODES
            )

        self.train_feedforward = bool(trainer_configs.get("train_feedforward", True))

        ck = trainer_configs.get("checkpoint_dir")
        self.checkpoint_dir = Path(ck).resolve() if ck else None

    # Decide COUPLING per sampling mode
    # Returns: (use_ot_train: bool, ot_method: str|None, ot_reg: float|None)
    def _ot_for_mode(self, sampling):
        # Random coupling (no OT)
        if sampling in ("geo_det", "euc_det"):
            return (False, None, None)

        # Exact OT
        if sampling in ("geo_amb_const", "geo_tan_const", "euc_ot"):
            return (True, "emd", None)

        # Sinkhorn OT
        if sampling in ("geo_amb_sb", "geo_tan_sb", "euc_sb"):
            return (True, "sinkhorn", self.trainer_configs.get("ot_reg_sb", 0.05))

        return False, None, None  # random/independent coupling

    def _save_checkpoint(self, filename: str, module: torch.nn.Module) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        torch.save(module.state_dict(), path)
        print(f"  Saved checkpoint {path}")

    def train(self):
        results = {}

        # ---------------- 1) Feed-forward baseline (optional) ----------------
        if self.train_feedforward:
            ff_arch = ResidualMLP(self.ssp_space.ssp_dim, flow=False).to(self.device)

            ff_use_ot = True
            ff_ot_method = self.trainer_configs.get("ot_method", "sinkhorn")
            ff_ot_reg    = self.trainer_configs.get("ot_reg",    0.005)

            ff_trainer = FeedforwardTrainer(
                encoded_dim = self.ssp_space.ssp_dim,
                data_dir    = self.trainer_configs["data_dir"],
                batch_size  = self.trainer_configs["batch_size"],
                epochs      = self.trainer_configs["epochs"],
                lr          = self.trainer_configs["lr"],
                weight_decay= self.trainer_configs["weight_decay"],
                val_split   = self.trainer_configs["val_split"],
                noise_type  = self.trainer_configs["noise_type"],
                target_type = self.trainer_configs["target_type"],
                architecture= ff_arch,
                device      = self.device,

                use_ot_train = ff_use_ot,
                ot_method    = ff_ot_method,
                ot_reg       = ff_ot_reg,
                dataloader_num_workers=self.trainer_configs.get("dataloader_num_workers"),
                dataloader_prefetch_factor=self.trainer_configs.get(
                    "dataloader_prefetch_factor", 2
                ),
            )
            print("Training FeedForward (with OT pairing)" if ff_trainer.use_ot_train else "FeedForward (random pairing)")
            model_ff, loss_ff, val_ff = ff_trainer.train()
            results[("ResidualMLP_FF", "euc_det")] = ((model_ff,), loss_ff, val_ff)
            self._save_checkpoint("feedforward.pt", model_ff)

            for epoch, (tr, vl) in enumerate(zip(loss_ff, val_ff)):
                log_metrics({
                    "trainer":   "ResidualMLP_FF",
                    "sampling":  "euc_det",
                    "epoch":     epoch,
                    "train_loss": tr,
                    "val_loss":   vl
                })

        # ---------------- 2) Flow-matching / diffusion variants ----------------
        E = self.trainer_configs["epochs"]
        for sampling in self.sampling_modes:
            print(f"--- Sampling mode: {sampling} ---")
            rf_arch = ResidualMLP(self.ssp_space.ssp_dim, flow=True).to(self.device)

            use_ot_train, ot_method, ot_reg = self._ot_for_mode(sampling)

            rf_trainer = FlowTrainer(
                encoded_dim   = self.ssp_space.ssp_dim,
                architecture  = rf_arch,
                data_dir      = self.trainer_configs["data_dir"],
                batch_size    = self.trainer_configs["batch_size"],
                epochs        = E,
                lr            = self.trainer_configs["lr"],
                weight_decay  = self.trainer_configs["weight_decay"],
                val_split     = self.trainer_configs["val_split"],
                noise_type    = self.trainer_configs["noise_type"],
                target_type   = self.trainer_configs["target_type"],
                device        = self.device,

                sampling_mode = sampling,  # supports both geodesic & euclidean names
                sigma_min     = self.trainer_configs.get("sigma_min", 0.1),
                beta_min      = self.trainer_configs.get("beta_min", 0.1),
                beta_max      = self.trainer_configs.get("beta_max", 20.0),

                # Pairing control
                use_ot_train  = use_ot_train,
                ot_method     = ot_method,
                ot_reg        = ot_reg,
                dataloader_num_workers=self.trainer_configs.get("dataloader_num_workers"),
                dataloader_prefetch_factor=self.trainer_configs.get(
                    "dataloader_prefetch_factor", 2
                ),
            )

            models_rf, loss_rf, val_rf = rf_trainer.train()
            results[("ResidualMLP_RF", sampling)] = (models_rf, loss_rf, val_rf)
            self._save_checkpoint(f"drift_{sampling}.pt", models_rf[0])

            for epoch, (tr, vl) in enumerate(zip(loss_rf, val_rf)):
                log_metrics({
                    "trainer":    "ResidualMLP_RF",
                    "sampling":   sampling,
                    "epoch":      epoch,
                    "train_loss": tr,
                    "val_loss":   vl
                })

        # ---------------- 3) Combined loss curves ----------------
        self.plot_training_results(results)
        return results

    def plot_training_results(self, training_results):
        def get_label(name, sampling):
            if name.endswith("_FF"):
                return "FeedForward"

            # Geodesic labels you provided
            if sampling == "geo_det":        return "GeoDetFM"
            if sampling == "geo_amb_const":  return "GeoAmbConst (exact OT)"
            if sampling == "geo_tan_const":  return "GeoTanConst (exact OT)"
            if sampling == "geo_amb_sb":     return "GeoAmbSB (Sinkhorn)"
            if sampling == "geo_tan_sb":     return "GeoTanSB (Sinkhorn)"

            # Euclidean labels you provided
            if sampling == "euc_det":  return "Det_CFM"
            if sampling == "euc_ot":   return "OT_CFM (exact OT)"
            if sampling == "euc_sb":   return "SB_CFM (Sinkhorn)"

            return f"{name} ({sampling})"

        fig_train = go.Figure()
        fig_val   = go.Figure()

        for (name, sampling), (_, train_l, val_l) in training_results.items():
            label = get_label(name, sampling)
            fig_train.add_trace(go.Scatter(
                x=list(range(len(train_l))),
                y=train_l,
                mode='lines+markers',
                name=label
            ))
            fig_val.add_trace(go.Scatter(
                x=list(range(len(val_l))),
                y=val_l,
                mode='lines+markers',
                name=label
            ))

        fig_train.update_layout(
            title="Training Losses: All Methods",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Method"
        )
        fig_val.update_layout(
            title="Validation Losses: All Methods",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Method"
        )

        log_metrics({"All_Train_Losses": fig_train})
        log_metrics({"All_Val_Losses":   fig_val})




