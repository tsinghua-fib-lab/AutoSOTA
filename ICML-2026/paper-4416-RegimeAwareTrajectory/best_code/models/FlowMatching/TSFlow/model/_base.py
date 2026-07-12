from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from torchdyn.core import NeuralODE
from torchtyping import TensorType
from typeguard import typechecked

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.utils.gaussian_process import Q0Dist
from models.FlowMatching.TSFlow.utils.optimal_transport import OTPlanSampler
from models.FlowMatching.TSFlow.utils.util import LongScaler
from models.FlowMatching.TSFlow.utils.variables import Prior, get_lags_for_freq, get_season_length

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values", "mean"]


class TSFlowBase(pl.LightningModule):
    def __init__(
        self,
        context_length,
        prediction_length,
        optimizer_params: dict,
        prior_params: dict,
        frequency: str,
        matching: str = "random",
        normalization: str | None = None,
        use_lags: bool = True,
        use_ema: bool = False,
        num_steps: int = 16,
        sigm: float = 0.001,
        solver: str = "euler",

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.logs = {}
        if normalization == "mean":
            self.scaler = MeanScaler(dim=1, keepdim=True, minimum_scale=1)
        elif normalization == "longmean" or normalization == "longmax":
            self.scaler = LongScaler()
        elif normalization == "zscore":
            self.scaler = StdScaler(dim=1, keepdim=True, minimum_scale=1)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        self.use_lags = use_lags
        self.lags_seq = get_lags_for_freq(frequency) if use_lags else [0]
        self.freq = get_season_length(frequency)
        self.context_length = context_length
        self.prior_context_length = (
            context_length
            if "context_freqs" not in prior_params.keys()
            else prior_params["context_freqs"] * prediction_length
        )
        self.prediction_length = prediction_length
        self.optimizer_params = optimizer_params
        self.best_crps = np.inf
        self.prior = Prior(prior_params["kernel"])
        self.ot_sampler = OTPlanSampler(method="exact")
        self.matching = matching
        self.use_ema = use_ema
        self.num_steps = num_steps
        self.solver = solver
        self.num_samples = 1
        self.times = []
        self.sigmin = sigm
        self.sigmax = 1 if self.prior != Prior.ISO else self.sigmin
        self.q0 = Q0Dist(
            **prior_params,
            prediction_length=prediction_length,
            freq=self.freq,
            iso=1e-2 if self.prior != Prior.ISO else 0,
        )

    def _extract_features(self, data):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        return [optimizer]

    @typechecked
    def forward_path(
        self,
        x1: TensorType[float, "batch", "length", "num_series"],
        x0: TensorType[float, "batch", "length", "num_series"],
        t: TensorType[float],
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
    ]:
        eps = torch.randn_like(x0)
        sig_t = (1 - t) * self.sigmax + t * self.sigmin
        psi = t * x1 + (1 - t) * x0 + sig_t * eps  # xt
        dpsi = x1 - x0 + (self.sigmin - self.sigmax) * eps  # uts
        return psi, dpsi

    @typechecked
    def p_losses(
        self,
        x1: TensorType[float, "batch", "length", "num_series"],
        x0: TensorType[float, "batch", "length", "num_series"],
        t: TensorType[float, "batch", 1],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
    ) -> TensorType[float]:
        num_dims_to_add = x1.dim() - t.dim()
        t = t.unsqueeze(-1) if num_dims_to_add == 1 else t.unsqueeze(-1).unsqueeze(-1)

        psi, dpsi = self.forward_path(x1, x0, t)
        predicted_flow = self.backbone(t, psi, features)

        loss = F.mse_loss(dpsi, predicted_flow)
        return loss

    @typechecked
    @torch.no_grad()
    def get_vf(
        self,
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        observation: TensorType[float, "batch", "length", "num_series"] | None = None,
        observation_mask: TensorType[float, "batch", "length", "num_series"] | None = None,
        guidance_scale: float = 0,
    ):
        def quantile_loss(y_prediction, y_target):
            assert y_target.shape == y_prediction.shape
            device = y_prediction.device
            batch_size_x_num_samples, _, _ = y_target.shape
            batch_size = batch_size_x_num_samples // self.num_samples
            q = torch.linspace(0.1, 0.9, self.num_samples, device=device).repeat(batch_size)
            q = q[:, None, None]
            e = y_target - y_prediction
            loss = torch.max(q * e, (q - 1) * e)
            return loss

        def score_func(t, x, model, args):
            with torch.enable_grad():
                x.requires_grad_(True)
                dxt = model(t, x, features, args)
                pred = x + (1 - t) * dxt
                E = quantile_loss(pred, observation)[observation_mask == 1].sum()
                return dxt, -torch.autograd.grad(E, x)[0]

        class vf(torch.nn.Module):
            def __init__(self, backbone, guidance_scale, sigmin, sigmax):
                super().__init__()
                self.backbone = backbone
                self.guidance_scale = guidance_scale
                self.sigmin = sigmin
                self.sigmax = sigmax

            def forward(self, t, x, args):
                if guidance_scale > 0:
                    dxt, score = score_func(t, x, self.backbone, args)
                    sig_t = (1 - t) * self.sigmax + t * self.sigmin
                    dsig_t = self.sigmin - self.sigmax
                    dxt = dxt - dsig_t * sig_t * self.guidance_scale * score

                else:
                    dxt = self.backbone(t, x, features, args)
                return dxt

        return vf(self.backbone if not self.use_ema else self.ema_backbone, guidance_scale, self.sigmin, self.sigmax)

    @typechecked
    @torch.no_grad()
    def sample(
        self,
        noise: TensorType[float, "batch", "length", "num_series"],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        observation: TensorType[float, "batch", "length", "num_series"] | None = None,
        observation_mask: TensorType[float, "batch", "length", "num_series"] | None = None,
        guidance_scale: float = 0,
    ) -> TensorType[float, "batch", "length", "num_series"]:
        if self.num_steps == 0:
            return noise.to(self.device)
        t_span = torch.linspace(0, 1, self.num_steps + 1)
        node = NeuralODE(self.get_vf(features, observation, observation_mask, guidance_scale), solver=self.solver)
        return node.trajectory(noise.to(self.device), t_span)[-1]

    @typechecked
    def sample_n(
        self,
        num_samples: int,
        features: TensorType[float, "batch", "length", "num_features"] | None = None,
    ) -> TensorType[float, "batch", "length", 1]:
        noise = self.q0(num_samples).to(self.device).unsqueeze(-1)
        noise = noise + self.sigmax * torch.randn_like(noise)
        return self.sample(noise, features).cpu() + 1

    def fast_denoise(self, xt, t, features=None, noise=None, steps=4):
        t_span = torch.linspace(t, 1, steps + 1, device=self.device)[:-1]
        node = NeuralODE(
            self.get_vf(features),
            solver=self.solver,
            sensitivity="adjoint",
        )

        return node.trajectory(xt.to(self.device), t_span)[-1]

    def fast_noise(self, xt, t, features=None, noise=None, steps=4):
        t_span = torch.linspace(1, t, steps + 1, device=self.device)[:-1]
        node = NeuralODE(
            self.get_vf(features),
            solver=self.solver,
            sensitivity="adjoint",
        )
        return node.trajectory(xt.to(self.device), t_span)[-1]

    def forward(self, *args):
        raise NotImplementedError()

    def training_step(self, *args):
        raise NotImplementedError

    def on_train_batch_end(self, *args):
        if self.ema_backbone is not None and self.use_ema:
            self.ema_backbone.update()
