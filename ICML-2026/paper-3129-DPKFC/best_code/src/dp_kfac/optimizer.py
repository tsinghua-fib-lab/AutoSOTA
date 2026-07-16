from typing import Optional, Iterator, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import Tensor, CovarianceDict, CovariancePair, KFACConfig, DPConfig
from .recorder import KFACRecorder
from .covariance import compute_covariances, compute_inverse_sqrt, accumulate_covariances
from .precondition import precondition_per_sample_gradients
from .privacy import clip_and_noise_gradients


class DPKFACOptimizer:
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        dp_config: DPConfig,
        kfac_config: Optional[KFACConfig] = None,
        only_trainable: bool = False,
    ) -> None:
        self.model = model
        self.base_optimizer = base_optimizer
        self.dp_config = dp_config
        self.kfac_config = kfac_config or KFACConfig()
        self.only_trainable = only_trainable

        self._recorder: Optional[KFACRecorder] = None
        self._inv_A: CovarianceDict = {}
        self._inv_G: CovarianceDict = {}
        self._step_count: int = 0

    def compute_preconditioner(
        self,
        data_source: Iterator[Tuple[Tensor, Tensor]],
        num_steps: int = 10,
        num_classes: Optional[int] = None,
        use_public_labels: bool = True,
    ) -> None:
        if self._recorder is None:
            self._recorder = KFACRecorder(
                self.model,
                only_trainable=self.only_trainable,
            )

        device = next(self.model.parameters()).device
        cov_list: List[CovariancePair] = []

        self._recorder.enable()
        for i, (data, labels) in enumerate(data_source):
            if i >= num_steps:
                break

            data = data.to(device)
            labels = labels.to(device)

            if not use_public_labels and num_classes is not None:
                labels = torch.randint(0, num_classes, (data.size(0),), device=device)

            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, labels)
            loss.backward()

            cov = compute_covariances(
                self.model,
                self._recorder.activations,
                self._recorder.backprops,
            )
            cov_list.append(cov)
            self._recorder.clear()

        self._recorder.disable()
        self.model.zero_grad()

        aggregated = accumulate_covariances(cov_list)
        self._inv_A, self._inv_G = compute_inverse_sqrt(
            aggregated,
            damping=self.kfac_config.damping,
        )

    def compute_preconditioner_from_noise(
        self,
        batch_size: int,
        input_shape: Tuple[int, ...],
        num_steps: int = 10,
        num_classes: int = 10,
    ) -> None:
        if self._recorder is None:
            self._recorder = KFACRecorder(
                self.model,
                only_trainable=self.only_trainable,
            )

        device = next(self.model.parameters()).device
        cov_list: List[CovariancePair] = []

        self._recorder.enable()
        for _ in range(num_steps):
            data = generate_pink_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, labels)
            loss.backward()

            cov = compute_covariances(
                self.model,
                self._recorder.activations,
                self._recorder.backprops,
            )
            cov_list.append(cov)
            self._recorder.clear()

        self._recorder.disable()
        self.model.zero_grad()

        aggregated = accumulate_covariances(cov_list)
        self._inv_A, self._inv_G = compute_inverse_sqrt(
            aggregated,
            damping=self.kfac_config.damping,
        )

    def step(self, batch_size: int) -> None:
        if self._inv_A and self._inv_G:
            precondition_per_sample_gradients(
                self.model,
                self._inv_A,
                self._inv_G,
            )

        clip_and_noise_gradients(
            self.model,
            noise_multiplier=self.dp_config.noise_multiplier,
            max_grad_norm=self.dp_config.max_grad_norm,
            batch_size=batch_size,
        )

        self.base_optimizer.step()
        self._step_count += 1

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    @property
    def has_preconditioner(self) -> bool:
        return bool(self._inv_A and self._inv_G)

    def remove_hooks(self) -> None:
        if self._recorder is not None:
            self._recorder.remove()
            self._recorder = None


def generate_pink_noise(
    batch_size: int,
    input_shape: Tuple[int, ...],
    device: torch.device,
    alpha: float = 1.0,
) -> Tensor:
    if len(input_shape) == 1:
        # Flat features (e.g. TF-IDF): use 1D pink noise via FFT
        n_features = input_shape[0]
        white_noise = torch.randn(batch_size, n_features, dtype=torch.cfloat, device=device)
        freqs = torch.fft.fftfreq(n_features, device=device)
        f = freqs.abs()
        f[0] = 1.0
        scale = 1.0 / (f ** alpha)
        scale[0] = 0.0
        pink_freq = white_noise * scale.unsqueeze(0)
        pink_noise = torch.fft.ifft(pink_freq).real
        std = pink_noise.std(dim=1, keepdim=True)
        pink_noise = pink_noise / (std + 1e-8) * 0.5
        return pink_noise

    if len(input_shape) == 3:
        channels, height, width = input_shape
    else:
        channels, height, width = 3, input_shape[0], input_shape[0]

    white_noise = torch.randn(
        batch_size, channels, height, width,
        dtype=torch.cfloat, device=device
    )

    freqs = torch.fft.fftfreq(height, device=device)
    fx, fy = torch.meshgrid(freqs, freqs, indexing="ij")
    f = torch.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0

    scale = 1.0 / (f ** alpha)
    scale[0, 0] = 0.0
    scale = scale.view(1, 1, height, width)

    pink_freq = white_noise * scale
    pink_noise = torch.fft.ifft2(pink_freq).real

    std = pink_noise.view(batch_size, -1).std(dim=1, keepdim=True)
    pink_noise = pink_noise / (std.view(batch_size, 1, 1, 1) + 1e-8) * 0.5

    return pink_noise
