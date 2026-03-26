# ---------------------------------------------------------------------------------
# Portions of this file are derived from uni2ts
# - Source: https://github.com/SalesforceAIResearch/uni2ts
# - Paper: Unified Training of Universal Time Series Forecasting Transformers
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import math
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Generator, Optional
import sys

import lightning as L
import torch
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from uni2ts.common.torch_util import safe_div
from uni2ts.loss.packed import PackedNLLLoss as _PackedNLLLoss
from uni2ts.model.moirai.module import MoiraiModule
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler


class SampleNLLLoss(_PackedNLLLoss):
    def reduce_loss(
        self,
        loss: Float[torch.Tensor, "batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "batch seq_len #dim"]],
        sample_id: Optional[Int[torch.Tensor, "batch seq_len"]],
        variate_id: Optional[Int[torch.Tensor, "batch seq_len"]],
    ) -> Float[torch.Tensor, "batch"]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loss = safe_div(loss, tobs)
        return (loss * mask).sum(dim=(-1, -2))


class MoiraiBackbone(L.LightningModule):
    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        context_length: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiModule] = None,
        patch_size: int | str = "auto",
        num_samples: int = 100,
        scaling: bool = True,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = MoiraiModule(**module_kwargs) if module is None else module
        self.module.scaling = scaling
        self.module.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.per_sample_loss_func = SampleNLLLoss()

    @contextmanager
    def hparams_context(
        self,
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        context_length: Optional[int] = None,
        patch_size: Optional[int | str] = None,
        num_samples: Optional[int] = None,
    ) -> Generator["MoiraiForecast", None, None]:
        kwargs = {
            "prediction_length": prediction_length,
            "target_dim": target_dim,
            "context_length": context_length,
            "patch_size": patch_size,
            "num_samples": num_samples,
        }
        old_hparams = deepcopy(self.hparams)
        for kw, arg in kwargs.items():
            if arg is not None:
                self.hparams[kw] = arg

        yield self

        for kw in kwargs:
            self.hparams[kw] = old_hparams[kw]

    @property
    def past_length(self) -> int:
        return (
            self.hparams.context_length + self.hparams.prediction_length
            if self.hparams.patch_size == "auto"
            else self.hparams.context_length
        )

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.hparams.context_length / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        return math.ceil(self.hparams.prediction_length / patch_size)

    @property
    def max_patch_size(self) -> int:
        return max(self.module.patch_sizes)

    def forward(
        self,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        num_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        
        if self.hparams.patch_size == "auto":
            val_loss = []
            preds = []
            for patch_size in self.module.patch_sizes:
                val_loss.append(
                    self._val_loss(
                        patch_size=patch_size,
                        target=past_target[..., : self.past_length, :],
                        observed_target=past_observed_target[
                            ..., : self.past_length, :
                        ],
                        is_pad=past_is_pad[..., : self.past_length]
                    )
                )
                distr = self._get_distr(
                    patch_size,
                    past_target[..., -self.hparams.context_length :, :],
                    past_observed_target[..., -self.hparams.context_length :, :],
                    past_is_pad[..., -self.hparams.context_length :]
                )
                preds.append(
                    self._format_preds(
                        patch_size,
                        distr.sample(
                            torch.Size((num_samples or self.hparams.num_samples,))
                        ),
                        past_target.shape[-1],
                    )
                )
            val_loss = torch.stack(val_loss)
            preds = torch.stack(preds)
            idx = val_loss.argmin(dim=0)
            return preds[idx, torch.arange(len(idx), device=idx.device)]
        else:
            distr = self._get_distr(
                self.hparams.patch_size,
                past_target[..., -self.hparams.context_length :, :],
                past_observed_target[..., -self.hparams.context_length :, :],
                past_is_pad[..., -self.hparams.context_length :],
            )
            preds = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))
            return self._format_preds(
                self.hparams.patch_size, preds, past_target.shape[-1]
            )

    def _val_loss(
        self,
        patch_size: int,
        target: Float[torch.Tensor, "batch time tgt"],
        observed_target: Bool[torch.Tensor, "batch time tgt"],
        is_pad: Bool[torch.Tensor, "batch time"]
    ) -> Float[torch.Tensor, "batch"]:
        # convert format
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            past_target=target[..., : self.hparams.context_length, :],
            past_observed_target=observed_target[..., : self.hparams.context_length, :],
            past_is_pad=is_pad[..., : self.hparams.context_length],
            future_target=target[..., self.hparams.context_length :, :],
            future_observed_target=observed_target[
                ..., self.hparams.context_length :, :
            ],
            future_is_pad=is_pad[..., self.hparams.context_length :]
        )
        # get predictions
        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            torch.ones_like(time_id, dtype=torch.long) * patch_size,
        )
        val_loss = self.per_sample_loss_func(
            pred=distr,
            target=target,
            prediction_mask=prediction_mask,
            observed_mask=observed_mask,
            sample_id=sample_id,
            variate_id=variate_id,
        )
        return val_loss

    def _get_distr(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"]
    ) -> Distribution:
        # convert format
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            patch_size,
            past_target,
            past_observed_target,
            past_is_pad
        )
        # get predictions
        distr = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            torch.ones_like(time_id, dtype=torch.long) * patch_size,
        )
        return distr

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: Bool[torch.Tensor, "batch past_seq tgt"],
        future_target: Float[torch.Tensor, "batch future_seq tgt"],
    ) -> tuple[
        Int[torch.Tensor, "batch past_token"], Int[torch.Tensor, "batch future_token"]
    ]:
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(past_seq_id.cumsum(dim=-1) - 1, min=0)
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    math.ceil(future_target.shape[-2] / patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        past_seq_id = past_seq_id.to(dtype=torch.int32)
        future_seq_id = future_seq_id.to(dtype=torch.int32)
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None
    ) -> tuple[
        Float[torch.Tensor, "batch combine_seq patch"],  # target
        Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
        Int[torch.Tensor, "batch combine_seq"],  # sample_id
        Int[torch.Tensor, "batch combine_seq"],  # time_id
        Int[torch.Tensor, "batch combine_seq"],  # variate_id
        Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        if future_target is None:
            future_target = torch.zeros(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_target.shape[-1],
                ),
                dtype=past_target.dtype,
                device=device,
            )
        
        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target, future_target
        )

        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_observed_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.hparams.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    # future=self.prediction_token_length(patch_size),
                    future = math.ceil(future_target.shape[-2] / patch_size)
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        # self.prediction_token_length(patch_size)
                        math.ceil(future_target.shape[-2] / patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    def _format_preds(
        self,
        patch_size: int,
        preds: Float[torch.Tensor, "sample batch combine_seq patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :patch_size]
        preds = rearrange(
            preds,
            "sample ... (dim seq) patch -> ... sample (seq patch) dim",
            dim=target_dim,
        )[..., : self.hparams.prediction_length, :]
        return preds.squeeze(-1)