import torch
from ema_pytorch import EMA
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.arch import BackboneModel
from models.FlowMatching.TSFlow.model._base import PREDICTION_INPUT_NAMES, TSFlowBase
from models.FlowMatching.TSFlow.utils.variables import get_lags_for_freq


class TSFlowUncond(TSFlowBase):
    def __init__(
        self,
        setting,
        target_dim,
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

        if use_lags:
            self.lags_seq = get_lags_for_freq(frequency)
            backbone_params = backbone_params.copy()
            backbone_params["input_dim"] += len(self.lags_seq)
            backbone_params["output_dim"] += len(self.lags_seq)
        else:
            self.lags_seq = [0]
        self.input_dim = backbone_params["input_dim"]
        self.backbone = BackboneModel(
            **backbone_params,
            num_features=1,
        )
        self.num_steps = num_steps
        self.solver = solver
        self.best_w2 = torch.inf
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

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_static_cat: torch.Tensor = None,
        feat_static_real: torch.Tensor = None,
        past_time_feat: torch.Tensor = None,
        future_time_feat: torch.Tensor = None,
        mean: torch.Tensor = None,
    ):
        # This is only used during prediction
        device = next(self.backbone.parameters()).device
        data = dict(
            feat_static_cat=(feat_static_cat.to(device) if feat_static_cat is not None else None),
            feat_static_real=(feat_static_real.to(device) if feat_static_real is not None else None),
            past_time_feat=(past_time_feat.to(device) if past_time_feat is not None else None),
            past_target=past_target.to(device),
            future_target=torch.zeros(past_target.shape[0], self.prediction_length, device=device),
            past_observed_values=(past_observed_values.to(device) if past_observed_values is not None else None),
            future_time_feat=(future_time_feat.to(device) if future_time_feat is not None else None),
        )
        observation, scale, features = self._extract_features(data)
        observation = observation.to(device)
        batch_size, length, ch = observation.shape
        pred = self.sample(torch.randn_like(observation))
        pred = pred * scale
        return pred[:, None, length - self.prediction_length :, 0]

    def get_predictor(self, input_transform, batch_size=40, device=None):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )
