"""Model lifecycle helpers: loading with retry, pre-loading sets of models, cleanup."""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig

from ..infrastructure.retry import cuda_retry, is_cuda_error
from .model_client import ModelClient


logger = logging.getLogger(__name__)


def init_model_client_with_retry(
    model_name_or_path: str,
    max_new_tokens: int,
    _logger: logging.Logger | None = None,
    temperature: float = 1.0,
) -> ModelClient:
    """Initialize a ModelClient with CUDA exponential-backoff retry.

    Attempts to create a ModelClient on CUDA up to 5 times with exponential
    backoff. Falls back to CPU if all CUDA attempts fail.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        max_new_tokens: Maximum number of tokens to generate.
        _logger: Logger instance for status messages.
        temperature: Temperature for likelihood computation scaling.

    Returns:
        An initialized ModelClient on CUDA (or CPU as fallback).
    """
    _logger = _logger or logger

    try:
        # stagger=False because ModelClient.__init__ already adds its own
        # stagger delay and calls wait_for_gpu_availability internally.
        return cuda_retry(
            lambda: ModelClient(
                model_name_or_path=model_name_or_path,
                logger=_logger,
                max_generate_length=max_new_tokens,
                temperature=temperature,
                device="cuda",
            ),
            stagger=False,
            logger=_logger,
            operation="ModelClient CUDA init",
        )
    except Exception as e:
        if not is_cuda_error(e):
            raise
        _logger.warning(f"CUDA init failed, falling back to CPU: {e}")

    # All CUDA attempts exhausted — fall back to CPU.
    # If CPU init also fails, ModelClient.__init__ raises naturally.
    return ModelClient(
        model_name_or_path=model_name_or_path,
        logger=_logger,
        max_generate_length=max_new_tokens,
        temperature=temperature,
        device="cpu",
    )


def preload_model_clients(
    cfg: DictConfig,
    model_dir_list: list[str],
    proposal: str | None = None,
) -> tuple[ModelClient | dict[str, ModelClient] | None, dict[str, ModelClient]]:
    """Pre-load likelihood and generation model clients for in-memory sampling.

    Loads all likelihood models into a dict keyed by model path, then selects
    the appropriate generation client (reusing a likelihood client when the
    model path matches).

    Args:
        cfg: Hydra config (needs compute_likelihooods_all_models.args and
            iterative_generation.args sub-configs).
        model_dir_list: List of model directory paths.
        proposal: Proposal distribution type ("unconstrained", "safe", or
            "mixture"). Determines which model is used for generation.

    Returns:
        Tuple of (gen_model_client, lik_model_clients). gen_model_client is
        a single ModelClient, a dict (for mixture), or None (for safe/sequential
        where generation happens via recursion).
    """
    lik_model_clients: dict[str, ModelClient] = {}

    # Pre-load likelihood models
    max_new_tokens_lik = (
        cfg.compute_likelihooods_all_models.args.generation_config.max_new_tokens
    )
    lik_temperature = cfg.temperature
    for model_path in model_dir_list:
        if model_path not in lik_model_clients:
            logger.info(f"Pre-loading likelihood model: {model_path}")
            lik_model_clients[model_path] = init_model_client_with_retry(
                model_path, max_new_tokens_lik, logger, temperature=lik_temperature
            )

    # Reuse lik clients for generation when the model path matches
    max_new_tokens_gen = cfg.iterative_generation.args.generation_config.max_new_tokens
    gen_model_client: ModelClient | dict[str, ModelClient] | None = None

    if proposal == "unconstrained":
        if model_dir_list[-1] in lik_model_clients:
            gen_model_client = lik_model_clients[model_dir_list[-1]]
        else:
            gen_model_client = init_model_client_with_retry(
                model_dir_list[-1], max_new_tokens_gen, logger
            )
    elif proposal == "safe":
        if cfg.conformal_policy_control.constrain_against == "init":
            if model_dir_list[0] in lik_model_clients:
                gen_model_client = lik_model_clients[model_dir_list[0]]
            else:
                gen_model_client = init_model_client_with_retry(
                    model_dir_list[0], max_new_tokens_gen, logger
                )
        # For safe/non-init, generation happens via recursion (no pre-load needed)
    elif proposal == "mixture":

        def _get_or_load(path: str) -> ModelClient:
            if path in lik_model_clients:
                return lik_model_clients[path]
            return init_model_client_with_retry(path, max_new_tokens_gen, logger)

        gen_model_client = {
            "unconstrained": _get_or_load(model_dir_list[-1]),
            "safe": _get_or_load(model_dir_list[0]),
        }

    return gen_model_client, lik_model_clients


def cleanup_model_clients(
    lik_model_clients: dict[str, ModelClient],
    gen_model_client: ModelClient | dict[str, ModelClient] | None = None,
) -> None:
    """Free pre-loaded models and reclaim GPU memory.

    Clears the likelihood clients dict (releasing all references) and
    empties the CUDA cache. The gen_model_client parameter is accepted
    for API compatibility but is not used — gen clients are typically
    aliases into lik_model_clients and freed by the clear().

    Args:
        lik_model_clients: Dict of likelihood model clients to clear.
        gen_model_client: Unused; kept for call-site compatibility.
    """
    lik_model_clients.clear()
    torch.cuda.empty_cache()
