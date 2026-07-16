from typing import Dict, Tuple, Iterator, List, Optional
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .types import Tensor, CovarianceDict, KFACConfig
from .recorder import KFACRecorder
from .covariance import compute_covariances, compute_inverse_sqrt, CovariancePair
from .optimizer import generate_pink_noise


class ActivationSource(Enum):
    PUBLIC = "public"
    PINK_NOISE = "pink"
    WHITE_NOISE = "white"


class GradientSource(Enum):
    PUBLIC_WITH_LABELS = "public_labels"
    PUBLIC_WITH_NOISE = "public_noise"
    PINK_NOISE = "pink_noise"
    WHITE_NOISE = "white_noise"


@dataclass
class PrecondMethod:
    name: str
    activation_source: ActivationSource
    gradient_source: GradientSource
    description: str


METHODS = {
    "dp_sgd": None,

    "dp_kfac_public": PrecondMethod(
        name="dp_kfac_public",
        activation_source=ActivationSource.PUBLIC,
        gradient_source=GradientSource.PUBLIC_WITH_LABELS,
        description="A from public, G from public with real labels",
    ),

    "dp_kfac_noise": PrecondMethod(
        name="dp_kfac_noise",
        activation_source=ActivationSource.WHITE_NOISE,
        gradient_source=GradientSource.WHITE_NOISE,
        description="A from white noise, G from white noise",
    ),

    "dp_kfac_pink": PrecondMethod(
        name="dp_kfac_pink",
        activation_source=ActivationSource.PINK_NOISE,
        gradient_source=GradientSource.PINK_NOISE,
        description="A from pink noise, G from pink noise",
    ),

    "dp_kfac_hybrid_pub_pub_noise": PrecondMethod(
        name="dp_kfac_hybrid_pub_pub_noise",
        activation_source=ActivationSource.PUBLIC,
        gradient_source=GradientSource.PUBLIC_WITH_NOISE,
        description="A from public, G from public images with random labels",
    ),

    "dp_kfac_hybrid_pub_pink_noise": PrecondMethod(
        name="dp_kfac_hybrid_pub_pink_noise",
        activation_source=ActivationSource.PUBLIC,
        gradient_source=GradientSource.PINK_NOISE,
        description="A from public, G from pink noise with random labels",
    ),

    "dp_kfac_hybrid_pink_pub_pub": PrecondMethod(
        name="dp_kfac_hybrid_pink_pub_pub",
        activation_source=ActivationSource.PINK_NOISE,
        gradient_source=GradientSource.PUBLIC_WITH_LABELS,
        description="A from pink noise, G from public with real labels",
    ),

    "dp_kfac_hybrid_pink_pub_noise": PrecondMethod(
        name="dp_kfac_hybrid_pink_pub_noise",
        activation_source=ActivationSource.PINK_NOISE,
        gradient_source=GradientSource.PUBLIC_WITH_NOISE,
        description="A from pink noise, G from public images with random labels",
    ),
}


def generate_white_noise(
    batch_size: int,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if len(input_shape) == 1:
        # Flat features (e.g. TF-IDF)
        return torch.randn(batch_size, input_shape[0], device=device)
    if len(input_shape) == 3:
        channels, height, width = input_shape
    else:
        channels, height, width = 1, input_shape[0], input_shape[0]
    return torch.randn(batch_size, channels, height, width, device=device)


def compute_activation_covariance(
    model: nn.Module,
    recorder: KFACRecorder,
    source: ActivationSource,
    public_iter: Optional[Iterator],
    batch_size: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device,
    steps: int = 10,
) -> CovarianceDict:
    cov_A_list = []

    recorder.enable()
    for _ in range(steps):
        if source == ActivationSource.PUBLIC:
            if public_iter is None:
                raise ValueError("public_iter required for PUBLIC activation source")
            data, labels = next(public_iter)
            data = data.to(device)
            labels = labels.to(device)
        elif source == ActivationSource.PINK_NOISE:
            data = generate_pink_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            data = generate_white_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        cov = compute_covariances(model, recorder.activations, recorder.backprops)
        cov_A_list.append(cov.A)
        recorder.clear()

    recorder.disable()
    model.zero_grad()

    if not cov_A_list:
        return {}

    avg_A: CovarianceDict = {}
    for name in cov_A_list[0]:
        stacked = torch.stack([c[name] for c in cov_A_list])
        avg_A[name] = stacked.mean(dim=0)

    return avg_A


def compute_gradient_covariance(
    model: nn.Module,
    recorder: KFACRecorder,
    source: GradientSource,
    public_iter: Optional[Iterator],
    batch_size: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device,
    steps: int = 10,
) -> CovarianceDict:
    cov_G_list = []

    recorder.enable()
    for _ in range(steps):
        if source == GradientSource.PUBLIC_WITH_LABELS:
            if public_iter is None:
                raise ValueError("public_iter required for PUBLIC_WITH_LABELS gradient source")
            data, labels = next(public_iter)
            data = data.to(device)
            labels = labels.to(device)
        elif source == GradientSource.PUBLIC_WITH_NOISE:
            if public_iter is None:
                raise ValueError("public_iter required for PUBLIC_WITH_NOISE gradient source")
            data, _ = next(public_iter)
            data = data.to(device)
            labels = torch.randint(0, num_classes, (data.size(0),), device=device)
        elif source == GradientSource.PINK_NOISE:
            data = generate_pink_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            data = generate_white_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        cov = compute_covariances(model, recorder.activations, recorder.backprops)
        cov_G_list.append(cov.G)
        recorder.clear()

    recorder.disable()
    model.zero_grad()

    if not cov_G_list:
        return {}

    avg_G: CovarianceDict = {}
    for name in cov_G_list[0]:
        stacked = torch.stack([c[name] for c in cov_G_list])
        avg_G[name] = stacked.mean(dim=0)

    return avg_G


def compute_preconditioner(
    model: nn.Module,
    recorder: KFACRecorder,
    method: PrecondMethod,
    public_iter: Optional[Iterator],
    batch_size: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device,
    kfac_config: KFACConfig,
    steps: int = 10,
) -> Tuple[CovarianceDict, CovarianceDict]:
    cov_A = compute_activation_covariance(
        model=model,
        recorder=recorder,
        source=method.activation_source,
        public_iter=public_iter,
        batch_size=batch_size,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
        steps=steps,
    )

    cov_G = compute_gradient_covariance(
        model=model,
        recorder=recorder,
        source=method.gradient_source,
        public_iter=public_iter,
        batch_size=batch_size,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
        steps=steps,
    )

    if not cov_A or not cov_G:
        return {}, {}

    cov_pair = CovariancePair(A=cov_A, G=cov_G)
    inv_A, inv_G = compute_inverse_sqrt(cov_pair, damping=kfac_config.damping)

    return inv_A, inv_G


def get_method(method_name: str) -> Optional[PrecondMethod]:
    return METHODS.get(method_name)


def is_kfac_method(method_name: str) -> bool:
    return method_name.startswith("dp_kfac")


def list_methods() -> list[str]:
    return list(METHODS.keys())


def estimate_adadps_preconditioner(
    model: nn.Module,
    public_loader: DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None,
    is_text: bool = False,
) -> Dict[str, Tensor]:
    """Estimate AdaDPS diagonal preconditioner E[g^2] from public data."""
    target = model._module if hasattr(model, "_module") else model
    precond: Dict[str, Tensor] = {}
    count = 0

    model.train()
    for batch_idx, batch in enumerate(public_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        model.zero_grad()
        if is_text and isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = target(input_ids=input_ids, attention_mask=attention_mask)
        else:
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            output = target(data)

        loss = F.cross_entropy(output, labels)
        loss.backward()

        for name, param in target.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue
            sq_grad = param.grad.detach().pow(2)
            if name in precond:
                precond[name] += sq_grad
            else:
                precond[name] = sq_grad.clone()

        count += 1

    for name in precond:
        precond[name] /= max(count, 1)

    model.zero_grad()
    return precond


def precondition_per_sample_gradients_adadps(
    model: nn.Module,
    preconditioner: Dict[str, Tensor],
    eps: float = 1e-10,
) -> None:
    """Apply AdaDPS preconditioning: g_scaled = g / sqrt(E[g^2] + eps)."""
    target = model._module if hasattr(model, "_module") else model

    for name, module in target.named_modules():
        if not hasattr(module, "weight"):
            continue
        if not hasattr(module.weight, "grad_sample") or module.weight.grad_sample is None:
            continue

        # Build full parameter name for weight
        weight_name = f"{name}.weight" if name else "weight"
        if weight_name in preconditioner:
            scale = (preconditioner[weight_name] + eps).rsqrt()
            g_sample = module.weight.grad_sample
            if isinstance(g_sample, list):
                g_sample = g_sample[-1]
            module.weight.grad_sample = g_sample * scale.unsqueeze(0)

        if module.bias is not None and hasattr(module.bias, "grad_sample") and module.bias.grad_sample is not None:
            bias_name = f"{name}.bias" if name else "bias"
            if bias_name in preconditioner:
                scale_b = (preconditioner[bias_name] + eps).rsqrt()
                g_sample_b = module.bias.grad_sample
                if isinstance(g_sample_b, list):
                    g_sample_b = g_sample_b[-1]
                module.bias.grad_sample = g_sample_b * scale_b.unsqueeze(0)


def generate_clustered_pink_noise(
    batch_size: int,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_centroids: int = 100,
    jitter_scale: float = 0.1,
    alpha: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Generate clustered pink noise using GMM-style centroid assignment.

    1. Generate num_centroids pink noise images as cluster centers.
    2. Assign each sample to a random centroid.
    3. Add small Gaussian jitter around the centroid.
    Returns (images, pseudo_labels) where labels are centroid indices.
    """
    # Generate centroids
    centroids = generate_pink_noise(num_centroids, input_shape, device, alpha=alpha)

    # Assign each sample to a random centroid
    assignments = torch.randint(0, num_centroids, (batch_size,), device=device)
    images = centroids[assignments]

    # Add jitter
    jitter = torch.randn_like(images) * jitter_scale
    images = images + jitter

    return images, assignments
