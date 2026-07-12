from typing import Iterator, Optional

import torch
from einops import rearrange
from ema_pytorch import EMA
from gluonts.dataset import Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model import Forecast
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values
from torchtyping import TensorType
from typeguard import typechecked

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.arch import BackboneModel
from models.FlowMatching.TSFlow.model._base import PREDICTION_INPUT_NAMES, TSFlowBase
from models.FlowMatching.TSFlow.utils.variables import get_lags_for_freq


class TSFlowPS(TSFlowBase):
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
        alpha: float = 0.005,
        iterations: int = 4,
        noise_level: float = 0.5,
        guidance_scale: int = 4,
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

        self.setting = setting
        self.alpha = alpha
        self.iterations = iterations
        self.noise_level = noise_level
        self.guidance_scale = guidance_scale
        if use_lags:
            self.lags_seq = get_lags_for_freq(frequency)
            backbone_params = backbone_params.copy()
            backbone_params["input_dim"] += len(self.lags_seq)
            backbone_params["output_dim"] += len(self.lags_seq)
        else:
            self.lags_seq = [0]
        self.input_dim = backbone_params["input_dim"]
        self.backbone = BackboneModel(**backbone_params, num_features=1)
        self.num_steps = num_steps
        self.solver = solver
        self.ema_backbone = EMA(self.backbone, **ema_params)

    def _extract_features(self, data):
        prior = data["past_target"][:, : -self.context_length]
        context = data["past_target"][:, -self.context_length :]
        context_observed = data["past_observed_values"][:, -self.context_length :]
        scaled_context, _, scale = self.scaler(context, data["mean"])
        if prior.shape[1] != 0:
            scaled_prior = prior / scale

        scaled_future = data["future_target"] / scale
        x = torch.cat([scaled_context, scaled_future], dim=1)

        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_prior,
                torch.cat([scaled_context, scaled_future], dim=1),
                dim=1,
            )
            x = torch.cat([x[:, :, None], lags], dim=-1)
        else:
            x = x[:, :, None]

        observation_mask = torch.zeros_like(x)
        observation_mask[:, : -self.prediction_length, 0] = context_observed[:, -self.context_length :]

        return x, observation_mask, scale[:, :, None]

    def training_step(self, data, idx):
        assert self.training is True
        x1, _, _ = self._extract_features(data)

        if self.prior == "isotropic":
            x0 = torch.randn_like(x1)
        else:
            x0 = self.q0(x1.shape[0]).to(self.device).unsqueeze(-1)

        if self.matching == "ot":
            x0, x1, _ = self.ot_sampler.sample_plan(x0, x1 - 1, replace=False)

        t = torch.rand((x1.shape[0], 1), device=self.device)
        loss = self.p_losses(x1, x0, t, None)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            batch_size=x1.shape[0],
            on_epoch=True,
            logger=True,
        )
        return {"loss": loss}

    def prior_sample(self, input, observation, observation_mask, alpha=0.1, iterations=10, noise_level=1.0):
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

        noise = self.sigmax * torch.randn_like(input)
        for i in range(iterations):
            self.backbone.zero_grad()
            input.requires_grad_(True)
            pred = self.fast_denoise(input + noise, torch.tensor(0.0, device=self.device))
            Ey = quantile_loss(pred, observation)[observation_mask == 1].sum()
            reg = self.q0.log_likelihood(input.squeeze()).sum()
            grad = 16 * torch.autograd.grad(Ey, input)[0] + torch.autograd.grad(reg, input)[0]
            input = input - alpha * grad + noise_level * torch.sqrt(torch.tensor(2 * alpha)) * torch.rand_like(grad)
            input.grad = None
        return input + noise

    @typechecked
    def forward(
        self,
        past_target: TensorType[float, "batch", "length"],
        past_observed_values: TensorType[float, "batch", "length"],
        mean: TensorType[float, "batch", 1] = None,
    ) -> TensorType[float, "batch", "num_samples", "prediction_length"]:
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

        observation, observation_mask, scale = self._extract_features(data)
        observation = observation.to(self.device) - 1

        batch_size, length, ch = observation.shape
        noise = self.q0(observation.shape[0]).to(self.device).unsqueeze(-1)
        noise = self.prior_sample(
            noise,
            observation,
            observation_mask,
            alpha=self.alpha,
            iterations=self.iterations,
            noise_level=self.noise_level,
        )

        pred = (
            self.sample(
                noise=noise,
                observation=observation,
                observation_mask=observation_mask,
                guidance_scale=self.guidance_scale,
            )
            + 1
        )
        pred = rearrange(pred * scale, "(b n) l 1 -> b n l", n=self.num_samples)
        return pred[..., length - self.prediction_length :]

    def get_predictor(self, input_transform, batch_size=40, device=None):
        return PyTorchPredictorWGrads(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )


class PyTorchPredictorWGrads(PyTorchPredictor):
    def predict(self, dataset: Dataset, num_samples: Optional[int] = None) -> Iterator[Forecast]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=lambda data: batchify(data, self.device),
        )

        self.prediction_net.eval()

        yield from self.forecast_generator(
            inference_data_loader=inference_data_loader,
            prediction_net=self.prediction_net,
            input_names=self.input_names,
            output_transform=self.output_transform,
            num_samples=num_samples,
        )
