from typing import Tuple

import torch
from einops import rearrange
from ema_pytorch import EMA
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values
from gluonts.transform.split import InstanceSplitter
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.arch import BackboneModel
from models.FlowMatching.TSFlow.model._base import PREDICTION_INPUT_NAMES, TSFlowBase
from models.FlowMatching.TSFlow.utils.gaussian_process import Q0Dist
from models.FlowMatching.TSFlow.utils.util import LongScaler
from models.FlowMatching.TSFlow.utils.variables import Prior, Setting

patch_typeguard()


class TSFlowCond(TSFlowBase):
    def __init__(
        self,
        setting: str,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        backbone_params: dict,
        prior_params: dict,
        optimizer_params: dict,
        ema_params: dict,
        frequency: str,
        normalization: str | None = None,
        use_lags: bool = True,
        use_ema: bool = False,
        num_steps: int = 16,
        solver: str = "euler",
        matching: str = "random",
    ):
        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            prior_params=prior_params,
            optimizer_params=optimizer_params,
            frequency=frequency,
            normalization=normalization,
            use_lags=use_lags,
            use_ema=use_ema,
            num_steps=num_steps,
            solver=solver,
            matching=matching,
        )
        num_features = 2 + (len(self.lags_seq) if use_lags else 0)

        target_dim = target_dim if setting == Setting.MULTIVARIATE else 1

        if setting == Setting.UNIVARIATE:
            self.backbone = BackboneModel(
                **backbone_params,
                num_features=num_features,
                target_dim=target_dim,
            )
        else:
            # not applied
            raise NotImplementedError
            # self.backbone = BackboneModelMultivariate(
            #     **backbone_params,
            #     num_features=num_features,
            #     target_dim=target_dim,
            # )
        self.ema_backbone = EMA(self.backbone, **ema_params)
        self.setting = setting
        self.guidance_scale = 0
        self.sigmax = self.sigmin
        self.q0 = Q0Dist(
            **prior_params,
            prediction_length=prediction_length,
            freq=self.freq,
            iso=1e-1 if self.prior != Prior.ISO else 0,
        )

    @typechecked
    def _extract_features(
        self, data: dict
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", "length", "num_series", "num_features"],
    ]:
        ####### Add for biological Dataset #######
        x = data['x']
        x_target = data['x_target']
        condition_dict = data.get('condition_dict')
        past_obs = data.get('past_obs', None)
        data['past_target'] = past_obs if past_obs is not None else x
        data["future_target"] = x_target
        data["past_observed_values"] = torch.ones_like(x, device=x.device)
        ####### Add for biological Dataset #######
        past = data["past_target"]
        future = data["future_target"]
        context_observed = data["past_observed_values"]
        mean = data["mean"]
        # id = data["id"]
        # if self.setting == Setting.UNIVARIATE:
        #     past = rearrange(past, "... -> ... 1")
        #     future = rearrange(future, "... -> ... 1")
        #     context_observed = rearrange(context_observed, "... -> ... 1")
        #     mean = rearrange(data["mean"], "... -> ... 1")

        context = past[:, -self.context_length :]
        long_context = past[:, : -self.context_length]
        prior_context = past[:, -self.prior_context_length :]

        if isinstance(self.scaler, LongScaler):
            scaled_context, loc, scale = self.scaler(context, scale=mean)
        else:
            _, loc, scale = self.scaler(past, context_observed)
            scaled_context = context / scale
        scaled_long_context = (long_context - loc) / scale
        scaled_prior_context = (prior_context - loc) / scale
        scaled_future = (future - loc) / scale

        x1 = torch.cat([scaled_context, scaled_future], dim=-2)
        batch_size, length, c = x1.shape

        observation_mask = torch.zeros_like(x1)
        observation_mask[:, : -self.prediction_length] = context_observed[:, -self.context_length :]

        features = []
        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_long_context,
                x1,
                dim=1,
            )
            features.append(lags)

        dist = self.q0.gp_regression(rearrange(scaled_prior_context, "b l c -> (b c) l"), self.prediction_length)

        fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
        fut_mean = rearrange(dist.mean, "(b c) l -> b l c", c=c)
        fut_std = torch.diagonal(dist.covariance_matrix, dim1=-2, dim2=-1)
        fut_std = rearrange(fut_std, "(b c) ... -> b ... c", c=c)
        features.append(torch.cat([scaled_context, fut_mean], dim=-2).unsqueeze(-1))
        features.append(observation_mask.unsqueeze(-1))
        x0 = torch.cat([scaled_context, fut], dim=-2)

        features = torch.cat(features, dim=-1)
        return x1, x0, observation_mask, loc, scale, features

    @typechecked
    def training_step(self, data: dict, idx: int) -> dict:
        assert self.training is True
        x1, x0, _, _, _, features = self._extract_features(data)
        t = torch.rand((x1.shape[0], 1), device=self.device)
        loss = self.p_losses(x1, x0, t, features)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            batch_size=x1.shape[0],
            on_epoch=True,
            logger=True,
        )
        return {"loss": loss}

    @typechecked
    def forward(
        self,
        past_target: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        past_observed_values: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        mean: TensorType[float, "batch", 1] | TensorType[float, "batch", 1, "num_series"] = None,
    ) -> (
        TensorType[float, "batch", "num_samples", "prediction_length"]
        | TensorType[float, "batch", "num_samples", "prediction_length", "num_series"]
    ):
        # This is only used during prediction
        past_target = past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        mean = mean.to(self.device).repeat_interleave(self.num_samples, dim=0)
        future_target = torch.zeros_like(past_target[:, -self.prediction_length :])
        data = dict(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            future_target=future_target,
        )
        observation, x0, observation_mask, loc, scale, features = self._extract_features(data)
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        pred = self.sample(
            x0.to(self.device),
            features=features,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
        )
        if self.setting == Setting.UNIVARIATE:
            pred = rearrange(pred * scale + loc, "(b n) l 1 -> b n l", n=self.num_samples)
        else:
            pred = rearrange(pred * scale + loc, "(b n) l k -> b n l k", n=self.num_samples)
        return pred[:, :, observation.shape[1] - self.prediction_length :]

    @typechecked
    def get_predictor(self, input_transform: InstanceSplitter, batch_size: int = 40, device: str | torch.device = None):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )
