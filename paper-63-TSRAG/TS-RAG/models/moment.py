import logging
import warnings
from argparse import Namespace
from copy import deepcopy
from math import ceil
from typing import Union

import math
import warnings
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
# import torch.nn as nn
import numpy.typing as npt
from dataclasses import dataclass
from transformers import T5Config, T5EncoderModel, T5Model

# from momentfm.common import TASKS
# from momentfm.data.base import TimeseriesOutputs
# from momentfm.models.layers.embed import PatchEmbedding, Patching
# from momentfm.models.layers.revin import RevIN
# from momentfm.utils.masking import Masking
# from momentfm.utils.utils import (
#     NamespaceWithDefaults,
#     get_anomaly_criterion,
#     get_huggingface_model_dimensions,
# )
from models.utils import NamespaceWithDefaults, get_anomaly_criterion

@dataclass
class TASKS:
    RECONSTRUCTION: str = "reconstruction"
    FORECASTING: str = "forecasting"
    CLASSIFICATION: str = "classification"
    EMBED: str = "embedding"

@dataclass
class TimeseriesOutputs:
    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    logits: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False

SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.dropout(x))
        x = x.flatten(start_dim=2, end_dim=3)
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
        reduction: str = "concat",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        if reduction == "mean":
            self.linear = nn.Linear(d_model, n_classes)
        elif reduction == "concat":
            self.linear = nn.Linear(n_channels * d_model, n_classes)
        else:
            raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")

    def forward(self, x, input_mask: torch.Tensor = None):
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        y = self.linear(x)
        return y


class ForecastingHead(nn.Module):
    def __init__(
        self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MOMENT(nn.Module):
    def __init__(self, config: Union[Namespace, dict], **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        self.normalizer = RevIN(
            num_features=1, affine=config.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=config.patch_len, stride=config.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            patch_dropout=config.getattr("patch_dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        self.encoder = self._get_transformer_backbone(config)
        self.head = self._get_head(self.task_name)

        # Frozen parameters
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)

    def _update_inputs(
        self, config: Union[Namespace, dict], **kwargs: dict
    ) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            config.d_model is None
            and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            config.d_model = config.t5_config['d_model']
            logging.info(f"Setting d_model to {config.d_model}")
        elif config.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone "
                "unless transformer backbone is a Huggingface model."
            )

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name != TASKS.RECONSTRUCTION:
            warnings.warn("Only reconstruction head is pre-trained. Classification and forecasting heads must be fine-tuned.")
        if task_name == TASKS.RECONSTRUCTION:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("head_dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction = self.config.getattr("reduction", "concat"),
            )
        elif task_name == TASKS.FORECASTING:
            num_patches = (
                max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
            ) // self.config.patch_stride_len + 1
            self.head_nf = self.config.d_model * num_patches
            return ForecastingHead(
                self.head_nf,
                self.config.forecast_horizon,
                self.config.getattr("head_dropout", 0.1),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        model_config = T5Config.from_dict(config.t5_config)
        if config.getattr("randomly_initialize_backbone", False):
            transformer_backbone = T5Model(model_config)
            logging.info(
                f"Initializing randomly initialized transformer from {config.transformer_backbone}."
            )
        else:
            transformer_backbone = T5EncoderModel(model_config)
            logging.info(
                f"Initializing pre-trained transformer from {config.transformer_backbone}."
            )

        transformer_backbone = transformer_backbone.get_encoder()

        if config.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)

    def embed(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        """
        Embeds the input time series data into a latent representation using the model's encoder.
    
        Args:
            x_enc (torch.Tensor): The input tensor of shape (batch_size, n_channels, seq_len).
            input_mask (torch.Tensor, optional): A mask tensor of shape (batch_size, seq_len) indicating the valid input positions. Default is None, which means all positions are valid.
            reduction (str, optional): The reduction method to apply on the output embeddings. Options are "mean" (default) or "none". The `mean` reduction averages embeddings over channels and patches. 
            **kwargs: Additional keyword arguments.
    
        Returns:
            TimeseriesOutputs: An object containing the embeddings, input mask, and metadata regarding the reduction method used.
        """
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.config.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)

        elif reduction == "none":
            pass
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction
        )

    def reconstruction(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # Prevent too short time-series from causing NaNs
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        if self.config.getattr("debug", False):
            illegal_output = self._check_model_weights_for_illegal_values()
        else:
            illegal_output = None

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        )

    def reconstruct(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0).to(
            x_enc.device
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)
            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def detect_anomalies(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        anomaly_criterion: str = "mse",
        **kwargs,
    ) -> TimeseriesOutputs:
        outputs = self.reconstruct(x_enc=x_enc, input_mask=input_mask)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    def forecast(
        self,
        *,
        x_enc: torch.Tensor, 
        input_mask: torch.Tensor = None, 
        **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        dec_out = self.head(enc_out)  # [batch_size x n_channels x forecast_horizon]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

    def short_forecast(
        self,
        *, 
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        forecast_horizon: int = 1,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape
        num_masked_patches = ceil(forecast_horizon / self.patch_len)
        num_masked_timesteps = num_masked_patches * self.patch_len

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        # Shift the time-series and mask the last few timesteps for forecasting
        x_enc = torch.roll(x_enc, shifts=-num_masked_timesteps, dims=2)
        input_mask = torch.roll(input_mask, shifts=-num_masked_timesteps, dims=1)

        # Attending to mask tokens
        input_mask[:, -num_masked_timesteps:] = 1
        mask = torch.ones_like(input_mask)
        mask[:, -num_masked_timesteps:] = 0

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]

        end = -num_masked_timesteps + forecast_horizon
        end = None if end == 0 else end

        dec_out = self.normalizer(x=dec_out, mode="denorm")
        forecast = dec_out[:, :, -num_masked_timesteps:end]

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            forecast=forecast,
            metadata={"forecast_horizon": forecast_horizon},
        )

    def classify(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "concat",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Mean across channels
        if reduction == "mean":
            # [batch_size x n_patches x d_model]
            enc_out = enc_out.mean(dim=1, keepdim=False)  
        # Concatenate across channels
        elif reduction == "concat":
            # [batch_size x n_patches x d_model * n_channels]
            enc_out = enc_out.permute(0, 2, 3, 1).reshape(
                batch_size, n_patches, self.config.d_model * n_channels)

        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        logits = self.head(enc_out, input_mask=input_mask)

        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction)

    def forward(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == TASKS.RECONSTRUCTION:
            return self.reconstruction(
                x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
            )
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.FORECASTING:
            return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class MOMENTPipeline(MOMENT, PyTorchModelHubMixin):
    def __init__(self, config: Union[Namespace, dict], **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop(
            "task_name", TASKS.RECONSTRUCTION
        )
        super().__init__(config, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.RECONSTRUCTION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.RECONSTRUCTION)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == TASKS.FORECASTING:
            if not hasattr(config, "forecast_horizon"):
                raise ValueError(
                    "forecast_horizon must be specified for long-horizon forecasting."
                )

        if config.task_name == TASKS.CLASSIFICATION:
            if not hasattr(config, "n_channels"):
                raise ValueError("n_channels must be specified for classification.")
            if not hasattr(config, "num_class"):
                raise ValueError("num_class must be specified for classification.")

    def init(self) -> None:
        if self.new_task_name != TASKS.RECONSTRUCTION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)

def freeze_parameters(model):
    """
    Freeze parameters of the model
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        """
        :param x: input tensor of shape (batch_size, n_channels, seq_len)
        :param mode: 'norm' or 'denorm'
        :param mask: input mask of shape (batch_size, seq_len)
        :return: RevIN transformed tensor
        """
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        """
        x    : batch_size x n_channels x seq_len
        mask : batch_size x seq_len
        """
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        n_channels = x.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
        # Set masked positions to NaN, and unmasked positions are taken from x
        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps
        # self.stdev = torch.sqrt(
        #     torch.var(masked_x, dim=-1, keepdim=True) + self.eps).get_data().detach()
        # NOTE: By default not bessel correction

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output



class PatchEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        seq_len: int = 512,
        patch_len: int = 8,
        stride: int = 8,
        patch_dropout: int = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: float = 1.41,
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.stride = stride
        self.d_model = d_model
        self.add_positional_embedding = add_positional_embedding

        self.value_embedding = nn.Linear(patch_len, d_model, bias=value_embedding_bias)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if value_embedding_bias:
                self.value_embedding.bias.data.zero_()
            # torch.nn.init.orthogonal_(self.mask_embedding, gain=orth_gain) # Fails

        # Positional embedding
        if self.add_positional_embedding:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(patch_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mask = Masking.convert_seq_to_patch_view(
            mask, patch_len=self.patch_len
        ).unsqueeze(-1)
        # mask : [batch_size x n_patches x 1]
        n_channels = x.shape[1]
        mask = (
            mask.repeat_interleave(self.d_model, dim=-1)
            .unsqueeze(1)
            .repeat(1, n_channels, 1, 1)
        )
        # mask : [batch_size x n_channels x n_patches x d_model]

        # Input encoding
        x = mask * self.value_embedding(x) + (1 - mask) * self.mask_embedding
        if self.add_positional_embedding:
            x = x + self.position_embedding(x)

        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        if self.stride != self.patch_len:
            warnings.warn(
                "Stride and patch length are not equal. "
                "This may lead to unexpected behavior."
            )

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x : [batch_size x n_channels x num_patch x patch_len]
        return x
    

class Masking:
    def __init__(
        self, mask_ratio: float = 0.3, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Indices with 0 mask are hidden, and with 1 are observed.
        """
        self.mask_ratio = mask_ratio
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride

    @staticmethod
    def convert_seq_to_patch_view(
        mask: torch.Tensor, patch_len: int = 8, stride: Optional[int] = None
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x seq_len]
        Output
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        stride = patch_len if stride is None else stride
        mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
        # mask : [batch_size x n_patches x patch_len]
        return (mask.sum(dim=-1) == patch_len).long()

    @staticmethod
    def convert_patch_to_seq_view(
        mask: torch.Tensor,
        patch_len: int = 8,
    ):
        """
        Input:
            mask : torch.Tensor of shape [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len] or
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len] or
            [batch_size x n_patches]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        if x.ndim == 4:
            return self._mask_patch_view(x, input_mask=input_mask)
        elif x.ndim == 3:
            return self._mask_seq_view(x, input_mask=input_mask)

    def _mask_patch_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x n_patches x patch_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x n_patches]
        """
        input_mask = self.convert_seq_to_patch_view(
            input_mask, self.patch_len, self.stride
        )
        n_observed_patches = input_mask.sum(dim=-1, keepdim=True)  # batch_size x 1

        batch_size, _, n_patches, _ = x.shape
        len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
        noise = torch.rand(
            batch_size, n_patches, device=x.device
        )  # noise in [0, 1], batch_size x n_channels x n_patches
        noise = torch.where(
            input_mask == 1, noise, torch.ones_like(noise)
        )  # only keep the noise of observed patches

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(
            ids_shuffle, dim=1
        )  # ids_restore: [batch_size x n_patches]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros(
            [batch_size, n_patches], device=x.device
        )  # mask: [batch_size x n_patches]
        for i in range(batch_size):
            mask[i, : len_keep[i]] = 1

        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.long()

    def _mask_seq_view(self, x, input_mask=None):
        """
        Input:
            x : torch.Tensor of shape
            [batch_size x n_channels x seq_len]
            input_mask: torch.Tensor of shape [batch_size x seq_len]
        Output:
            mask : torch.Tensor of shape [batch_size x seq_len]
        """
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        mask = self._mask_patch_view(x, input_mask=input_mask)
        return self.convert_patch_to_seq_view(mask, self.patch_len).long()
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, model_name="MOMENT"):
        super(PositionalEmbedding, self).__init__()
        self.model_name = model_name

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if (
            self.model_name == "MOMENT"
            or self.model_name == "TimesNet"
            or self.model_name == "GPT4TS"
        ):
            return self.pe[:, : x.size(2)]
        else:
            return self.pe[:, : x.size(1)]

class InstanceNorm(nn.Module):
    """
    Instance Normalization with handling for constant inputs.
    For constant inputs, the normalized output is set to 1, and inverse restores the original constant value.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        loc_scale: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if loc_scale is None:
            # Compute loc and scale
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(),
                nan=1.0
            )

            # Detect constant inputs
            is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True)  # Batch-wise constant input detection

            # For constant inputs, set scale = 1
            scale = torch.where(is_constant, torch.ones_like(scale), scale)
        else:
            loc, scale = loc_scale

        # Normalize input
        normalized = (x - loc) / scale

        # For constant inputs, override normalized result to 1
        is_constant = torch.all(x == x[..., :1], dim=-1, keepdim=True) if loc_scale is None else (scale == 1)
        normalized = torch.where(is_constant, torch.ones_like(normalized), normalized)

        return normalized, (loc, scale)

class MOMENTWithRetrieval(nn.Module):
    def __init__(self, config: Union[Namespace, dict], **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)
        config = self._validate_inputs(config)
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len

        self.normalizer = RevIN(
            num_features=1, affine=config.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=config.patch_len, stride=config.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            seq_len=config.seq_len,
            patch_len=config.patch_len,
            stride=config.patch_stride_len,
            patch_dropout=config.getattr("patch_dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0))
        self.encoder = self._get_transformer_backbone(config)
        self.head = self._get_head(self.task_name)

        # Retrieval Augmentation Layer
        # if 'moe' in self.augment:
        # moe: moe with topk+1 experts, use mha
        n_patches = 64
        hidden_dim = 512
        bottleneck_dim = 1024
        self.encode_mlp = nn.Sequential(
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
        )
        self.project_before_fusion = nn.Linear(n_patches * config.d_model, hidden_dim)
        self.project_after_fusion = nn.Linear(hidden_dim, n_patches * config.d_model)

        self.instance_norm = InstanceNorm()
        self.dropout = nn.Dropout(p=0.2)

        # Frozen parameters
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            self.patch_embedding = freeze_parameters(self.patch_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)
    
    def init_extra_weights(self, layers):
        """
        Initialize weights for multiple layers.
        
        Args:
            layers (list): List of layers (e.g., [self.retrieve_yin_layer, ...]).
        """
        factor = 0.05
        
        for layer in layers:
            if isinstance(layer, nn.Sequential):
                self.init_extra_weights(layer)
            if isinstance(layer, nn.Linear):
                # Initialize weights using normal distribution
                layer.weight.data.normal_(mean=0.0, std=factor * ((16 * 2) ** -0.5))
                
                # Initialize biases to zero if they exist
                if layer.bias is not None:
                    layer.bias.data.zero_()
            if isinstance(layer, nn.MultiheadAttention):
                # Initialize weights using normal distribution
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                
                # Initialize biases to zero if they exist
                if layer.in_proj_bias is not None:
                    layer.in_proj_bias.data.zero_()
                if layer.out_proj.bias is not None:
                    layer.out_proj.bias.data.zero_()

    def _update_inputs(
        self, config: Union[Namespace, dict], **kwargs: dict
    ) -> NamespaceWithDefaults:
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            config.d_model is None
            and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            config.d_model = config.t5_config['d_model']
            logging.info(f"Setting d_model to {config.d_model}")
        elif config.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone "
                "unless transformer backbone is a Huggingface model."
            )

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name != TASKS.RECONSTRUCTION:
            warnings.warn("Only reconstruction head is pre-trained. Classification and forecasting heads must be fine-tuned.")
        if task_name == TASKS.RECONSTRUCTION:
            return PretrainHead(
                self.config.d_model,
                self.config.patch_len,
                self.config.getattr("head_dropout", 0.1),
                self.config.getattr("orth_gain", 1.41),
            )
        elif task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction = self.config.getattr("reduction", "concat"),
            )
        elif task_name == TASKS.FORECASTING:
            num_patches = (
                max(self.config.seq_len, self.config.patch_len) - self.config.patch_len
            ) // self.config.patch_stride_len + 1
            self.head_nf = self.config.d_model * num_patches
            return ForecastingHead(
                self.head_nf,
                self.config.forecast_horizon,
                self.config.getattr("head_dropout", 0.1),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        model_config = T5Config.from_dict(config.t5_config)
        if config.getattr("randomly_initialize_backbone", False):
            transformer_backbone = T5Model(model_config)
            logging.info(
                f"Initializing randomly initialized transformer from {config.transformer_backbone}."
            )
        else:
            transformer_backbone = T5EncoderModel(model_config)
            logging.info(
                f"Initializing pre-trained transformer from {config.transformer_backbone}."
            )

        transformer_backbone = transformer_backbone.get_encoder()

        if config.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)

    def embed(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        """
        Embeds the input time series data into a latent representation using the model's encoder.
    
        Args:
            x_enc (torch.Tensor): The input tensor of shape (batch_size, n_channels, seq_len).
            input_mask (torch.Tensor, optional): A mask tensor of shape (batch_size, seq_len) indicating the valid input positions. Default is None, which means all positions are valid.
            reduction (str, optional): The reduction method to apply on the output embeddings. Options are "mean" (default) or "none". The `mean` reduction averages embeddings over channels and patches. 
            **kwargs: Additional keyword arguments.
    
        Returns:
            TimeseriesOutputs: An object containing the embeddings, input mask, and metadata regarding the reduction method used.
        """
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.config.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)

        elif reduction == "none":
            pass
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction
        )

    def reconstruction(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # Prevent too short time-series from causing NaNs
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        if self.config.getattr("debug", False):
            illegal_output = self._check_model_weights_for_illegal_values()
        else:
            illegal_output = None

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        )

    def reconstruct(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0).to(
            x_enc.device
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)
            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def detect_anomalies(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        anomaly_criterion: str = "mse",
        **kwargs,
    ) -> TimeseriesOutputs:
        outputs = self.reconstruct(x_enc=x_enc, input_mask=input_mask)
        self.anomaly_criterion = get_anomaly_criterion(anomaly_criterion)

        anomaly_scores = self.anomaly_criterion(x_enc, outputs.reconstruction)

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=outputs.reconstruction,
            anomaly_scores=anomaly_scores,
            metadata={"anomaly_criterion": anomaly_criterion},
        )

    def forecast(
        self,
        *,
        x_enc: torch.Tensor, 
        retrieved_seq: torch.Tensor,
        input_mask: torch.Tensor = None, 
        **kwargs
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=torch.ones_like(input_mask))

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        # flatten enc_out
        enc_out = nn.Flatten(start_dim=-2)(enc_out)

        # fuse retrieved seqs
        # scaling
        retrieved_seq, loc_scale_retrieved = self.instance_norm(retrieved_seq)
        # import pdb; pdb.set_trace()
        # retrieved_seq = self.normalizer(x=retrieved_seq, mode="norm")
        L = 64
        r_B, r_M, r_L = retrieved_seq.shape
        assert r_L % 2 == 0, "L of retrieved_seq should be even"
        retrieved_x, retrieved_y = retrieved_seq.split((r_L-L, L), dim=2)

        # moe fusion
        # Step 1: concat embeddings of retrieved_y and enc_out
        retrieved_y_enc = []
        for i in range(r_M):
            retrieved_y_enc.append(self.encode_mlp(retrieved_y[:, i, :]))
        retrieved_y_enc = torch.stack(retrieved_y_enc, dim=1)
        all_enc = torch.cat([self.project_before_fusion(enc_out).unsqueeze(1), retrieved_y_enc], dim=1)
        # Step 2: MHA
        att_output, attn_weights = self.mha(all_enc, all_enc, all_enc)
        att_output = all_enc + att_output
        att_output = att_output + self.dropout(self.ffn(att_output))
        # Step 3: gate
        scores = []
        for i in range(r_M+1):
            gate = torch.sigmoid(self.gate_layer(att_output[:,i,:]))
            scores.append(gate)
        scores = torch.stack(scores, dim=1)     # B, k+1, 1
        alpha = F.softmax(scores, dim=1)       # B, k+1, 1
        # Step 4: fuse
        fused_sequance_output = torch.sum(alpha * att_output, dim=1)    # B, d_model
        fused_sequance_output = self.dropout(fused_sequance_output)
        # Step 5: skip connection
        enc_out = enc_out + self.project_after_fusion(fused_sequance_output)            # B, 1, d_model

        # reverse flatten
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        dec_out = self.head(enc_out)  # [batch_size x n_channels x forecast_horizon]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, forecast=dec_out)

    def short_forecast(
        self,
        *, 
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        forecast_horizon: int = 1,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape
        num_masked_patches = ceil(forecast_horizon / self.patch_len)
        num_masked_timesteps = num_masked_patches * self.patch_len

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        # Shift the time-series and mask the last few timesteps for forecasting
        x_enc = torch.roll(x_enc, shifts=-num_masked_timesteps, dims=2)
        input_mask = torch.roll(input_mask, shifts=-num_masked_timesteps, dims=1)

        # Attending to mask tokens
        input_mask[:, -num_masked_timesteps:] = 1
        mask = torch.ones_like(input_mask)
        mask[:, -num_masked_timesteps:] = 0

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]

        end = -num_masked_timesteps + forecast_horizon
        end = None if end == 0 else end

        dec_out = self.normalizer(x=dec_out, mode="denorm")
        forecast = dec_out[:, :, -num_masked_timesteps:end]

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            forecast=forecast,
            metadata={"forecast_horizon": forecast_horizon},
        )

    def classify(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "concat",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Mean across channels
        if reduction == "mean":
            # [batch_size x n_patches x d_model]
            enc_out = enc_out.mean(dim=1, keepdim=False)  
        # Concatenate across channels
        elif reduction == "concat":
            # [batch_size x n_patches x d_model * n_channels]
            enc_out = enc_out.permute(0, 2, 3, 1).reshape(
                batch_size, n_patches, self.config.d_model * n_channels)

        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        logits = self.head(enc_out, input_mask=input_mask)

        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction)

    def forward(
        self,
        *,
        x_enc: torch.Tensor,
        retrieved_seq: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == TASKS.RECONSTRUCTION:
            return self.reconstruction(
                x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
            )
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.FORECASTING:
            return self.forecast(x_enc=x_enc, retrieved_seq=retrieved_seq, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
        

class MOMENTPipelineWithRetrieval(MOMENTWithRetrieval, PyTorchModelHubMixin):
    def __init__(self, config: Union[Namespace, dict], **kwargs: dict):
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop(
            "task_name", TASKS.RECONSTRUCTION
        )
        super().__init__(config, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.RECONSTRUCTION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.RECONSTRUCTION)
        config = Namespace(**kwargs["model_kwargs"])

        if config.task_name == TASKS.FORECASTING:
            if not hasattr(config, "forecast_horizon"):
                raise ValueError(
                    "forecast_horizon must be specified for long-horizon forecasting."
                )

        if config.task_name == TASKS.CLASSIFICATION:
            if not hasattr(config, "n_channels"):
                raise ValueError("n_channels must be specified for classification.")
            if not hasattr(config, "num_class"):
                raise ValueError("num_class must be specified for classification.")

    def init(self) -> None:
        if self.new_task_name != TASKS.RECONSTRUCTION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)