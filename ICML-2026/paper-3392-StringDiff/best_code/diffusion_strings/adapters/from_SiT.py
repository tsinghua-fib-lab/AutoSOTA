from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

Time = float | torch.Tensor
Label = int | torch.Tensor
VectorField = Callable[[torch.Tensor, Time, Label | None], torch.Tensor]


def _batch_time_like(t: Time, x: torch.Tensor) -> torch.Tensor:
    """Convert a scalar or batch time to a tensor matching the batch size of x."""
    if torch.is_tensor(t):
        t = t.to(device=x.device, dtype=x.dtype)
    else:
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)

    if t.ndim == 0:
        return t.expand(x.shape[0])

    if t.shape == (x.shape[0],):
        return t

    if t.numel() == 1:
        return t.reshape(()).expand(x.shape[0])

    raise ValueError(
        f"t must be a scalar or have shape ({x.shape[0]},), got {tuple(t.shape)}"
    )


def _batch_label_like(y: Label, x: torch.Tensor) -> torch.Tensor:
    """Convert a scalar or batch label to a tensor matching the batch size of x."""
    if torch.is_tensor(y):
        y = y.to(device=x.device, dtype=torch.long)
    else:
        y = torch.as_tensor(y, device=x.device, dtype=torch.long)

    if y.ndim == 0:
        return y.expand(x.shape[0])

    if y.shape == (x.shape[0],):
        return y

    if y.numel() == 1:
        return y.reshape(()).expand(x.shape[0])

    raise ValueError(
        f"y must be a scalar or have shape ({x.shape[0]},), got {tuple(y.shape)}"
    )


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Infer the device and dtype from a model parameter or buffer."""
    tensor = next(model.parameters(), None)
    if tensor is None:
        tensor = next(model.buffers(), None)
    if tensor is None:
        return torch.device("cpu"), torch.float32
    return tensor.device, tensor.dtype


def _sit_input_shape(model: nn.Module) -> tuple[int, int, int]:
    """Infer the single-example SiT input shape ``(C, H, W)`` from the model."""
    in_channels = getattr(model, "in_channels", None)
    if in_channels is None:
        x_embedder = getattr(model, "x_embedder", None)
        proj = getattr(x_embedder, "proj", None)
        if proj is not None and hasattr(proj, "weight"):
            in_channels = proj.weight.shape[1]
    if in_channels is None:
        raise ValueError("Could not infer SiT input channels from the model")

    x_embedder = getattr(model, "x_embedder", None)
    img_size = getattr(x_embedder, "img_size", None)
    if img_size is not None:
        if isinstance(img_size, int):
            height = width = img_size
        else:
            height, width = img_size
        return int(in_channels), int(height), int(width)

    num_patches = getattr(x_embedder, "num_patches", None)
    patch_size = getattr(x_embedder, "patch_size", getattr(model, "patch_size", None))
    if num_patches is not None and patch_size is not None:
        patches_per_side = int(num_patches**0.5)
        if patches_per_side * patches_per_side != num_patches:
            raise ValueError("Could not infer square SiT input size from num_patches")
        if isinstance(patch_size, int):
            patch_height = patch_width = patch_size
        else:
            patch_height, patch_width = patch_size
        return (
            int(in_channels),
            patches_per_side * int(patch_height),
            patches_per_side * int(patch_width),
        )

    raise ValueError("Could not infer SiT input spatial shape from the model")


def randn_sit_input(
    model: nn.Module,
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create Gaussian noise with shape ``(n, *SiT_input_shape)``.

    ``model`` is expected to be an initialized SiT model. If ``device`` or
    ``dtype`` are not provided, they are inferred from the model parameters.
    """
    model_device, model_dtype = _model_device_and_dtype(model)
    return torch.randn(
        (n, *_sit_input_shape(model)),
        device=model_device if device is None else device,
        dtype=model_dtype if dtype is None else dtype,
        generator=generator,
    )


def _load_checkpoint(model: nn.Module, checkpoint_path: str | Path, strict: bool) -> None:
    """Load a checkpoint into a model, accepting common checkpoint wrapper keys."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint

    if isinstance(checkpoint, Mapping):
        for key in ("ema", "model", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                state_dict = candidate
                break

    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint must be a state_dict or contain ema/model/state_dict")

    cleaned_state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=strict)


def _call_model(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    use_cfg: bool,
    cfg_scale: float,
    **model_kwargs: Any,
) -> torch.Tensor:
    """Call a SiT model with optional classifier-free guidance."""
    if use_cfg:
        return model.forward_with_cfg(x, t, cfg_scale=cfg_scale, **model_kwargs)

    return model(x, t, **model_kwargs)


@torch.no_grad()
def _vector_field(
    x: torch.Tensor,
    t: Time,
    y: Label | None = None,
    *,
    fn: Callable[..., torch.Tensor],
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model_kwargs: Mapping[str, Any],
) -> torch.Tensor:
    """Evaluate a SiT transport vector-field wrapper at state x and time t."""
    batch_model_kwargs = dict(model_kwargs)
    if y is not None:
        batch_model_kwargs["y"] = y
    if "y" in batch_model_kwargs:
        batch_model_kwargs["y"] = _batch_label_like(batch_model_kwargs["y"], x)
    return fn(x, _batch_time_like(t, x), model_fn, **batch_model_kwargs)


def sit_vector_fields_from_model(
    model: nn.Module,
    *,
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: str | None = None,
    train_eps: float | None = None,
    sample_eps: float | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    use_cfg: bool = False,
    cfg_scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[VectorField, VectorField]:
    """Expose ``b`` and ``score`` callables from an already-loaded SiT model.

    The returned callables accept ``(x, t, y=None)``. ``t`` and ImageNet class
    label ``y`` may each be scalars or batch-shaped tensors and are expanded to
    ``(x.shape[0],)`` before the SiT model is evaluated.
    """
    from transport import create_transport

    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    model.eval()

    transport = create_transport(
        path_type=path_type,
        prediction=prediction,
        loss_weight=loss_weight,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )

    model_fn = partial(
        _call_model,
        model,
        use_cfg=use_cfg,
        cfg_scale=cfg_scale,
    )
    field_kwargs = {
        "model_fn": model_fn,
        "model_kwargs": {} if model_kwargs is None else dict(model_kwargs),
    }
    b = partial(_vector_field, fn=transport.get_drift(), **field_kwargs)
    score = partial(_vector_field, fn=transport.get_score(), **field_kwargs)

    return b, score


def load_sit_vector_fields(
    *,
    model: nn.Module | None = None,
    model_name: str = "SiT-XL/2",
    checkpoint_path: str | Path | None = None,
    image_size: int = 256,
    num_classes: int = 1000,
    learn_sigma: bool | None = None,
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: str | None = None,
    train_eps: float | None = None,
    sample_eps: float | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
    use_cfg: bool = False,
    cfg_scale: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> tuple[nn.Module, VectorField, VectorField]:
    """Load a SiT model and expose stochastic-interpolant fields.

    The returned ``b`` callable is the probability-flow drift/velocity field
    from the SiT transport object. The returned ``score`` callable is
    ``nabla_x log p_t(x)``. Both callables accept ``(x, t, y=None)``, where
    ``t`` and ImageNet class label ``y`` may be scalars or batch-shaped tensors.

    This follows the official SiT implementation:
    ``transport.create_transport(...).get_drift()`` and ``get_score()`` convert
    velocity, score, and noise parameterizations to the stochastic interpolant
    vector fields.
    """
    if model is None:
        from models import SiT_models

        if learn_sigma is None:
            learn_sigma = image_size == 256

        latent_size = image_size // 8
        model = SiT_models[model_name](
            input_size=latent_size,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
        )

    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, strict=strict)

    b, score = sit_vector_fields_from_model(
        model,
        path_type=path_type,
        prediction=prediction,
        loss_weight=loss_weight,
        train_eps=train_eps,
        sample_eps=sample_eps,
        model_kwargs=model_kwargs,
        use_cfg=use_cfg,
        cfg_scale=cfg_scale,
        device=device,
        dtype=dtype,
    )

    return model, b, score
