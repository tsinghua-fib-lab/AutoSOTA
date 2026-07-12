from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
from datasets import Dataset
from sklearn.decomposition import PCA


def _get_input_embedding_weight(model) -> torch.Tensor | None:
    """Best-effort retrieval of the token-embedding weight tensor across architectures."""
    try:
        return model.get_input_embeddings().weight
    except AttributeError:
        pass
    state_dict = model.state_dict()
    if "transformer.wte.weight" in state_dict:
        return state_dict["transformer.wte.weight"]
    for name, parameters in state_dict.items():
        if name.endswith("embed_tokens.weight") or name.endswith("wte.weight"):
            return parameters
    return None


def _fit_mvnormal_from_model(model) -> torch.distributions.MultivariateNormal:
    """Fit MVN over the model's input embeddings; falls back to diagonal covariance on failure."""
    weight = _get_input_embedding_weight(model)
    if weight is None:
        raise ValueError("Cannot run mvnormal initialization: input embedding weight not found!")
    with torch.no_grad():
        embeddings = (weight[:-3, :] if weight.shape[0] > 3 else weight).cpu().to(torch.float32)
        mu = embeddings.mean(dim=0)
        n = embeddings.size(0)
        centered = embeddings - mu
        sigma = (centered.T @ centered) / max(n, 1)
        sigma = sigma + 1e-6 * torch.eye(sigma.shape[0], dtype=sigma.dtype)
        covariance = 1e-5 * sigma
    try:
        return torch.distributions.MultivariateNormal(mu, covariance_matrix=covariance)
    except Exception:
        covariance_matrix = torch.clamp(torch.diag(covariance), min=1e-8)
        return torch.distributions.MultivariateNormal(mu, covariance_matrix=torch.diag(covariance_matrix))


def _fit_pca_from_dataset_path(path: str, n_components_target: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit PCA on flattened ``embedding`` field of a HuggingFace Dataset stored at ``path``."""
    if not path:
        raise ValueError("pretrained_pca_path must be specified for pretrained_pca init!")
    if not os.path.exists(path):
        raise ValueError(f"pretrained_pca_path does not exist: {path}!")

    progressive_dataset = Dataset.load_from_disk(path)
    flat_embeddings: list[np.ndarray] = []
    for i in range(len(progressive_dataset)):
        embedding = progressive_dataset[i].get("embedding")
        if embedding is None:
            continue
        flat_embeddings.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
    if not flat_embeddings:
        raise ValueError(f"No embeddings found in {path}!")

    X = np.stack(flat_embeddings, axis=0)
    n_components = min(n_components_target, X.shape[0] - 1, X.shape[1])
    if n_components < 1:
        raise ValueError(f"Cannot fit PCA: need at least 2 samples, got {X.shape[0]}!")

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    print(f"Fitted PCA from {path}: {n_components} components, explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return (
        torch.tensor(pca.components_, dtype=torch.float32),
        torch.tensor(pca.mean_, dtype=torch.float32),
    )


def _load_embeddings_from_disk(path: str) -> torch.Tensor:
    """Load a compression-embedding tensor saved as either a raw tensor or a state_dict."""
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        if "compression_embeddings" in loaded:
            loaded = loaded["compression_embeddings"]
        elif "state_dict" in loaded:
            for key in loaded["state_dict"].keys():
                if "compression" in key.lower() or "embedding" in key.lower():
                    loaded = loaded["state_dict"][key]
                    break
            else:
                raise ValueError(f"Could not find compression embeddings in state_dict at {path}!")
        else:
            loaded = next(iter(loaded.values()))
    if not isinstance(loaded, torch.Tensor):
        loaded = torch.tensor(loaded, dtype=torch.float32)
    print(f"Loaded embeddings from {path}: shape {tuple(loaded.shape)}")
    return loaded.to(torch.float32)


def _resolve_load_from_disk_save_path(path: str, output_dir: str) -> str:
    """Decide where to persist a freshly generated load_from_disk seed."""
    if path:
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        return path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, "generated_compression_embeddings.pt")
    return "generated_compression_embeddings.pt"


def prepare_embedding_init(args, model):
    """Fit / load whatever the chosen init_method needs. Run once per training run.

    Returns:
        (init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings) — a tuple
        whose entries are ``None`` unless required by ``init_method``.
    """
    init_method = args.embedding_init_method
    mvn_dist = None
    pca_components = None
    pca_mean = None
    loaded_embeddings = None

    if init_method == "load_from_disk":
        if args.embedding_init_path and os.path.exists(args.embedding_init_path):
            loaded_embeddings = _load_embeddings_from_disk(args.embedding_init_path)
        else:
            # Generate a fresh seed using the secondary init method, persist, and load.
            save_path = _resolve_load_from_disk_save_path(args.embedding_init_method, args.output_dir)
            gen_method = args.load_from_disk_embedding_init_method
            gen_mvn_dist = None
            gen_pca_components = None
            gen_pca_mean = None
            if gen_method == "mvnormal":
                gen_mvn_dist = _fit_mvnormal_from_model(model)
            elif gen_method == "pretrained_pca":
                gen_pca_components, gen_pca_mean = _fit_pca_from_dataset_path(
                    args.pretrained_pca_path, args.pretrained_pca_num_components
                )
            seed = create_compression_embedding(
                batch_size=1,
                num_compression_tokens=args.number_of_mem_tokens,
                hidden_size=model.config.hidden_size,
                init_method=gen_method,
                mvn_dist=gen_mvn_dist,
                pca_components=gen_pca_components,
                pca_mean=gen_pca_mean,
            )
            loaded_embeddings = seed.data.detach().clone().cpu()
            torch.save(loaded_embeddings, save_path)
            print(f"Generated embeddings via '{gen_method}' and saved to {save_path}: shape {tuple(loaded_embeddings.shape)}")
    elif init_method == "pretrained_pca":
        pca_components, pca_mean = _fit_pca_from_dataset_path(args.pretrained_pca_path, args.pretrained_pca_num_components)
    elif init_method == "mvnormal":
        mvn_dist = _fit_mvnormal_from_model(model)

    return init_method, mvn_dist, pca_components, pca_mean, loaded_embeddings


def _uniform(shape: tuple, scale: float = 1.0) -> torch.Tensor:
    return torch.rand(list(shape), dtype=torch.float32) * scale


def _normal(shape: tuple, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(list(shape), dtype=torch.float32) * scale


def _signed_uniform(shape: tuple, scale: float = 1.0) -> torch.Tensor:
    return (torch.rand(list(shape), dtype=torch.float32) * 2 - 1) * scale


# Direct-init strategies: name -> sampler producing tensor of shape ``shape`` directly.
_DIRECT_INIT_STRATEGIES: dict[str, Callable] = {
    "zeros": lambda shape: torch.zeros(list(shape), dtype=torch.float32),
    "random": lambda shape: _uniform(shape, 1.0),
    "random0.2": lambda shape: _uniform(shape, 0.2),
    "random0.02": lambda shape: _uniform(shape, 0.02),
    "random0.002": lambda shape: _uniform(shape, 0.002),
    "random0.0002": lambda shape: _uniform(shape, 0.0002),
    "random5": lambda shape: _uniform(shape, 5.0),
    "random_norm": lambda shape: _normal(shape, 1.0),
    "random_norm_0.2": lambda shape: _normal(shape, 0.2),
    "random_norm_0.02": lambda shape: _normal(shape, 0.02),
    "neg_random": lambda shape: _signed_uniform(shape, 1.0),
    "neg_random0.2": lambda shape: _signed_uniform(shape, 0.2),
    "neg_random5": lambda shape: _signed_uniform(shape, 5.0),
}


def create_compression_embedding(
    *,
    batch_size: int,
    num_compression_tokens: int,
    hidden_size: int,
    init_method: str,
    mvn_dist: torch.distributions.MultivariateNormal | None = None,
    token_embeddings: torch.Tensor | None = None,
    single_compression_token_embeddings_initialization: torch.Tensor | None = None,
    pca_components: torch.Tensor | None = None,
    pca_mean: torch.Tensor | None = None,
    loaded_embeddings: torch.Tensor | None = None,
) -> torch.nn.Parameter:
    """Sample one compression-embedding ``Parameter`` of shape ``[batch_size, num_tokens, hidden_size]``."""
    shape = (batch_size, num_compression_tokens, hidden_size)

    if init_method == "mvnormal":
        if mvn_dist is None:
            raise ValueError(f'Initialization "{init_method}" requires mvn_dist to be fitted in advance!')
        samples = mvn_dist.sample((batch_size, num_compression_tokens))
        return torch.nn.Parameter(samples.to(dtype=torch.float32))

    if init_method.startswith("single_"):
        single_shape = (1, num_compression_tokens, hidden_size)
        if single_compression_token_embeddings_initialization is not None:
            data = single_compression_token_embeddings_initialization.detach().clone().repeat(batch_size, 1, 1)
        else:
            # Fallback: sample one seed once and broadcast.
            data = _uniform(single_shape, 1.0).repeat(batch_size, 1, 1)
        return torch.nn.Parameter(data)

    if init_method == "mean_token_embeds":
        if token_embeddings is None:
            raise ValueError(f'Initialization "{init_method}" requires token_embeddings!')
        return torch.nn.Parameter(token_embeddings.mean(1, keepdim=True).repeat(1, num_compression_tokens, 1))

    if init_method == "pretrained_pca":
        if pca_components is None or pca_mean is None:
            raise ValueError(f'Initialization "{init_method}" requires pca_components and pca_mean!')
        flattened_dim = pca_mean.shape[0]
        expected_flattened_dim = num_compression_tokens * hidden_size
        if flattened_dim != expected_flattened_dim:
            raise ValueError(
                f"PCA dimension mismatch: pretrained has {flattened_dim} but current needs {expected_flattened_dim} "
                f"(num_tokens={num_compression_tokens}, hidden_size={hidden_size})!"
            )
        n_components_to_use = min(pca_components.shape[0], num_compression_tokens)
        flat = torch.matmul(
            _normal((batch_size, n_components_to_use), 0.1), pca_components[:n_components_to_use]
        ) + pca_mean.unsqueeze(0)
        return torch.nn.Parameter(flat.reshape(*shape))

    if init_method == "load_from_disk":
        if loaded_embeddings is None:
            raise ValueError(f'Initialization "{init_method}" requires loaded_embeddings!')
        data = _broadcast_loaded_embeddings(loaded_embeddings, batch_size, num_compression_tokens, hidden_size)
        return torch.nn.Parameter(data.to(torch.float32))

    if init_method == "compression_head_forward":
        raise ValueError(
            "init_method='compression_head_forward' is only supported by the progressive cramming trainer, "
            "which runs the compression head per sample to seed each embedding; it cannot be sampled here."
        )

    sampler = _DIRECT_INIT_STRATEGIES.get(init_method)
    if sampler is not None:
        return torch.nn.Parameter(sampler(shape))

    raise ValueError(f"Unsupported initialization method: {init_method}!")


def _broadcast_loaded_embeddings(
    loaded_embeddings: torch.Tensor, batch_size: int, num_compression_tokens: int, hidden_size: int
) -> torch.Tensor:
    """Validate shape and broadcast ``loaded`` to ``[batch, compression, hidden]``."""
    if loaded_embeddings.ndim == 2:
        if loaded_embeddings.shape != (num_compression_tokens, hidden_size):
            raise ValueError(
                f"Loaded embeddings shape mismatch: got {tuple(loaded_embeddings.shape)}, "
                f"expected ({num_compression_tokens}, {hidden_size})!"
            )
        return loaded_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
    if loaded_embeddings.ndim == 3:
        if loaded_embeddings.shape[1] != num_compression_tokens or loaded_embeddings.shape[2] != hidden_size:
            raise ValueError(
                f"Loaded embeddings shape mismatch: got {tuple(loaded_embeddings.shape)}, "
                f"expected (1 or {batch_size}, {num_compression_tokens}, {hidden_size})!"
            )
        if loaded_embeddings.shape[0] == 1:
            return loaded_embeddings.repeat(batch_size, 1, 1)
        if loaded_embeddings.shape[0] == batch_size:
            return loaded_embeddings
        raise ValueError(
            f"Loaded embeddings batch size mismatch: got {loaded_embeddings.shape[0]}, expected 1 or {batch_size}!"
        )
    raise ValueError(f"Loaded embeddings must be 2D or 3D tensor, got shape {tuple(loaded_embeddings.shape)}!")
