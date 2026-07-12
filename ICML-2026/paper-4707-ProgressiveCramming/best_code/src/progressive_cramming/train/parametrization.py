"""Parametrizations of the compression embedding owned by trainers."""

import torch


class DirectParametrization:
    """Embedding optimized directly: parameters == embedding."""

    def __init__(self, init_embedding: torch.Tensor, device: torch.device):
        self.embedding = torch.nn.Parameter(init_embedding.data.to(device))
        self._initialization_snapshot = self.embedding.detach().clone().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.embedding]

    @property
    def shared_parameters(self) -> list[torch.nn.Parameter]:
        return []

    @property
    def optimizable_tensor(self) -> torch.Tensor:
        return self.embedding

    def materialize(self) -> torch.Tensor:
        return self.embedding

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list | None:
        return None

    def shared_state_dict(self) -> dict | None:
        return None


class PretrainedPCAParametrization:
    """Embedding parametrized by low-rank PCA coefficients: e = z @ W + mu."""

    def __init__(
        self,
        batch_size: int,
        num_compression_tokens: int,
        hidden_size: int,
        pca_components: torch.Tensor,
        pca_mean: torch.Tensor,
        device: torch.device,
    ):
        flattened = pca_mean.shape[0]
        expected = num_compression_tokens * hidden_size
        if flattened != expected:
            raise ValueError(
                f"PCA dim mismatch: pretrained has {flattened}, expected {expected} "
                f"(num_tokens={num_compression_tokens}, hidden_size={hidden_size})"
            )
        self._components = pca_components.to(device)
        self._mean = pca_mean.to(device)
        self._batch_size = batch_size
        self._num_compression_tokens = num_compression_tokens
        self._hidden_size = hidden_size
        n_components = self._components.shape[0]
        self.coefficients = torch.nn.Parameter(
            torch.randn([batch_size, n_components], dtype=torch.float32, device=device) * 0.1
        )
        self._initialization_snapshot = self.materialize().detach().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.coefficients]

    @property
    def shared_parameters(self) -> list[torch.nn.Parameter]:
        return []

    @property
    def optimizable_tensor(self) -> torch.Tensor:
        return self.coefficients

    def materialize(self) -> torch.Tensor:
        flat = torch.matmul(self.coefficients, self._components) + self._mean.unsqueeze(0)
        return flat.reshape(self._batch_size, self._num_compression_tokens, self._hidden_size)

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list:
        return self.coefficients.clone().detach().to(torch.float32).cpu().numpy().tolist()

    def shared_state_dict(self) -> dict | None:
        return None


class LowDimProjectedParametrization:
    """Compression embedding in a rank-k subspace: ``e = projection(z)``."""

    def __init__(
        self,
        *,
        init_coefficients: torch.Tensor,  # [batch, num_compression_tokens, low_dim_size]
        low_dim_size: int,
        hidden_size: int,
        device: torch.device,
        projection_state_dict: dict | None = None,
        train_projection: bool = True,
        projection_module: torch.nn.Linear | None = None,
    ):
        if init_coefficients.dim() != 3 or init_coefficients.shape[-1] != low_dim_size:
            raise ValueError(
                f"init_coefficients must have shape [batch, num_compression_tokens, {low_dim_size}], "
                f"got {tuple(init_coefficients.shape)}"
            )
        self._low_dim_size = low_dim_size
        self._hidden_size = hidden_size
        self.coefficients = torch.nn.Parameter(init_coefficients.data.to(device))
        if projection_module is not None:
            # Caller-owned Linear (checkpoint loading + requires_grad are
            # assumed to be configured by the caller).
            self.projection = projection_module
        else:
            self.projection = torch.nn.Linear(low_dim_size, hidden_size).to(device)
            if projection_state_dict is not None:
                self.projection.load_state_dict(projection_state_dict)
            if not train_projection:
                for p in self.projection.parameters():
                    p.requires_grad = False
        self._initialization_snapshot = self.materialize().detach().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        # Only the batch-level coefficients here — projection lives in
        # `shared_parameters` so per-sample-aware trainers (Progressive) can
        # route it through a dedicated optimizer.
        return [self.coefficients]

    @property
    def shared_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.projection.parameters() if parameter.requires_grad]

    @property
    def optimizable_tensor(self) -> torch.Tensor:
        # ConvergedSamplesGuard freezes converged samples' coefficients only;
        # the projection is shared and is updated batch-wide.
        return self.coefficients

    def materialize(self) -> torch.Tensor:
        return self.projection(self.coefficients)

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list:
        return self.coefficients.clone().detach().to(torch.float32).cpu().numpy().tolist()

    def shared_state_dict(self) -> dict:
        return self.projection.state_dict()


def build_parametrization(
    *,
    init_method: str,
    batch_size: int,
    num_compression_tokens: int,
    hidden_size: int,
    device: torch.device,
    init_helper,
    pca_components: torch.Tensor | None,
    pca_mean: torch.Tensor | None,
    low_dim_train: bool = False,
    low_dim_size: int | None = None,
    projection_state_dict: dict | None = None,
    train_projection: bool = True,
    projection_module: torch.nn.Linear | None = None,
):
    """Create the parametrization owning the optimizable parameters of the compression embedding."""
    if low_dim_train:
        if low_dim_size is None:
            raise ValueError("low_dim_size is required when low_dim_train=True!")
        return LowDimProjectedParametrization(
            init_coefficients=init_helper(),
            low_dim_size=low_dim_size,
            hidden_size=hidden_size,
            device=device,
            projection_state_dict=projection_state_dict,
            train_projection=train_projection,
            projection_module=projection_module,
        )
    if init_method == "pretrained_pca":
        assert pca_components is not None and pca_mean is not None
        return PretrainedPCAParametrization(
            batch_size=batch_size,
            num_compression_tokens=num_compression_tokens,
            hidden_size=hidden_size,
            pca_components=pca_components,
            pca_mean=pca_mean,
            device=device,
        )
    return DirectParametrization(init_embedding=init_helper(), device=device)


class PerSampleDirectParametrization:
    """Per-sample compression embeddings as separate [1, compression, hidden] Parameters."""

    def __init__(self, init_embedding: torch.Tensor, device: torch.device):
        """init_embedding: [batch, compression, hidden] on CPU."""
        batch_size = init_embedding.size(0)
        self._params = [torch.nn.Parameter(init_embedding.data[j : j + 1].clone().to(device)) for j in range(batch_size)]
        self._initialization_snapshot = init_embedding.detach().clone().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self._params)

    @property
    def shared_parameters(self) -> list[torch.nn.Parameter]:
        return []

    def materialize(self) -> torch.Tensor:
        """[batch, compression, hidden] — concatenation across samples."""
        return torch.cat(self._params, dim=0)

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list | None:
        return None

    def shared_state_dict(self) -> dict | None:
        return None


class PerSamplePretrainedPCAParametrization:
    """Per-sample low-rank coefficients backed by a frozen PCA basis: e_j = z_j @ W + mu."""

    def __init__(
        self,
        batch_size: int,
        num_compression_tokens: int,
        hidden_size: int,
        pca_components: torch.Tensor,
        pca_mean: torch.Tensor,
        device: torch.device,
    ):
        flattened = pca_mean.shape[0]
        expected = num_compression_tokens * hidden_size
        if flattened != expected:
            raise ValueError(
                f"PCA dim mismatch: pretrained has {flattened}, expected {expected} "
                f"(num_tokens={num_compression_tokens}, hidden_size={hidden_size})!"
            )
        self._components = pca_components.to(device)
        self._mean = pca_mean.to(device)
        self._batch_size = batch_size
        self._num_compression_tokens = num_compression_tokens
        self._hidden_size = hidden_size
        n_components = self._components.shape[0]
        self._coefficients = [
            torch.nn.Parameter(torch.randn([1, n_components], dtype=torch.float32, device=device) * 0.1)
            for _ in range(batch_size)
        ]
        self._initialization_snapshot = self.materialize().detach().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self._coefficients)

    @property
    def shared_parameters(self) -> list[torch.nn.Parameter]:
        return []

    def materialize(self) -> torch.Tensor:
        """[batch, compression, hidden] — reconstruct from per-sample coefficients."""
        coefficients = torch.cat(self._coefficients, dim=0)
        flat = torch.matmul(coefficients, self._components) + self._mean.unsqueeze(0)
        return flat.reshape(self._batch_size, self._num_compression_tokens, self._hidden_size)

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list:
        return [c.clone().detach().to(torch.float32).cpu().numpy().tolist() for c in self._coefficients]

    def shared_state_dict(self) -> dict | None:
        return None


class PerSampleLowDimProjectedParametrization:
    """Per-sample low-rank coefficients with a shared trainable Linear projection.

    Coefficients live per-sample in ``R^{low_dim_size}``; a single
    ``torch.nn.Linear(low_dim_size, hidden_size)`` is shared across the batch
    and lifts them to the model's hidden dimension.  This is the per-sample
    analogue of :class:`LowDimProjectedParametrization`, used by Progressive
    cramming where each sample has its own optimizer.

    The projection can either be **owned** (the parametrization creates its own
    ``nn.Linear`` and optionally warm-starts it from ``projection_state_dict``),
    or **provided** by the caller via ``projection_module`` — the latter is how
    ``--low_dim_projection_global`` is implemented: one Linear is created
    before the data loop, kept alive across batches together with its
    AdamW state + LR scheduler, and passed in here on every batch.
    """

    def __init__(
        self,
        *,
        init_coefficients: torch.Tensor,  # [batch, num_compression_tokens, low_dim_size]
        low_dim_size: int,
        hidden_size: int,
        device: torch.device,
        projection_state_dict: dict | None = None,
        train_projection: bool = True,
        projection_module: torch.nn.Linear | None = None,
    ):
        if init_coefficients.dim() != 3 or init_coefficients.shape[-1] != low_dim_size:
            raise ValueError(
                f"init_coefficients must have shape [batch, num_compression_tokens, {low_dim_size}], "
                f"got {tuple(init_coefficients.shape)}"
            )
        batch_size = init_coefficients.size(0)
        self._low_dim_size = low_dim_size
        self._hidden_size = hidden_size
        self._params = [torch.nn.Parameter(init_coefficients.data[j : j + 1].clone().to(device)) for j in range(batch_size)]
        if projection_module is not None:
            # Caller-owned Linear (checkpoint loading + requires_grad are
            # assumed to be configured by the caller).
            self.projection = projection_module
        else:
            self.projection = torch.nn.Linear(low_dim_size, hidden_size).to(device)
            if projection_state_dict is not None:
                self.projection.load_state_dict(projection_state_dict)
            if not train_projection:
                for p in self.projection.parameters():
                    p.requires_grad = False
        self._initialization_snapshot = self.materialize().detach().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self._params)

    @property
    def shared_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.projection.parameters() if p.requires_grad]

    def materialize(self) -> torch.Tensor:
        """[batch, num_compression_tokens, hidden_size] — concat per-sample coefficients then project."""
        coefficients = torch.cat(self._params, dim=0)  # [batch, num_comp, low_dim]
        return self.projection(coefficients)

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list:
        return [c.clone().detach().to(torch.float32).cpu().numpy().tolist() for c in self._params]

    def shared_state_dict(self) -> dict:
        return self.projection.state_dict()


def build_per_sample_parametrization(
    *,
    init_method: str,
    batch_size: int,
    num_compression_tokens: int,
    hidden_size: int,
    device: torch.device,
    init_helper,
    pca_components: torch.Tensor | None,
    pca_mean: torch.Tensor | None,
    low_dim_train: bool = False,
    low_dim_size: int | None = None,
    projection_state_dict: dict | None = None,
    train_projection: bool = True,
    projection_module: torch.nn.Linear | None = None,
):
    """Per-sample variant of :func:`build_parametrization`, used by progressive cramming.

    Dispatch order mirrors :func:`build_parametrization` exactly (low-dim,
    pretrained PCA, direct), so trainers can branch on
    ``args.low_dim_projection`` / ``args.embedding_init_method`` symmetrically
    in both code paths.

    ``projection_module`` (low-dim only) lets the caller pass a pre-built
    ``nn.Linear`` to be reused — used in ``--low_dim_projection_global`` mode
    where a single Linear lives through the entire run.
    """
    if low_dim_train:
        if low_dim_size is None:
            raise ValueError("low_dim_size is required when low_dim_train=True")
        return PerSampleLowDimProjectedParametrization(
            init_coefficients=init_helper(),
            low_dim_size=low_dim_size,
            hidden_size=hidden_size,
            device=device,
            projection_state_dict=projection_state_dict,
            train_projection=train_projection,
            projection_module=projection_module,
        )
    if init_method == "pretrained_pca":
        assert pca_components is not None and pca_mean is not None
        return PerSamplePretrainedPCAParametrization(
            batch_size=batch_size,
            num_compression_tokens=num_compression_tokens,
            hidden_size=hidden_size,
            pca_components=pca_components,
            pca_mean=pca_mean,
            device=device,
        )
    return PerSampleDirectParametrization(init_embedding=init_helper(), device=device)
