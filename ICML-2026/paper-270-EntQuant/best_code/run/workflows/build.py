import logging
import time
from math import inf
from typing import Any

import torch
from hydra_zen.typing import Partial
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HqqConfig, PreTrainedModel

from entquant.model.entquant_model import EntQuantModel
from entquant.model.tokenizer import save_tokenizer, Tokenizer
from entquant.super_weights import detect_fallback_layers, SuperWeightsConfig
from entquant.utils import clear_cache, get_memory_stats, str_to_dtype
from run.hydra_zen import register_workflow

logger = logging.getLogger(__name__)


@register_workflow("build")
def build_base_model(
    model_cfg: dict = "${cfg.model}",
) -> tuple[PreTrainedModel, dict[str, Any]]:
    model_cls = model_cfg["model_cls"] or AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_cfg["base_model_name_or_path"],
        device_map=model_cfg["device_map"],
        dtype=str_to_dtype(model_cfg["dtype"]),
        **model_cfg.get("model_kwargs", {}),
    )
    if model_cfg.get("generation_config") is not None:
        model.generation_config = model_cfg["generation_config"]

    logger.info(f"Created pretrained model: {model.name_or_path}")
    logger.debug(f"Model structure:\n{model}")

    results: dict[str, Any] = {}

    clear_cache()
    mem_allocated, mem_reserved = get_memory_stats()
    logger.info(f"Memory allocated: {mem_allocated:8.3f} GiB")
    logger.info(f"Memory reserved:  {mem_reserved:8.3f} GiB")
    results["memory_allocated"] = mem_allocated
    results["memory_reserved"] = mem_reserved

    return model, results


@register_workflow("build")
def build_entquant_model(
    model_cfg: dict = "${cfg.model}",
    super_weight_config: SuperWeightsConfig | None = "${cfg.super_weights}",
    entquant: dict = "${cfg.entquant}",
    save_dir: str | None = "${run.save_model_dir}",
    tokenizer: Tokenizer = "${cfg.tokenizer}",
) -> tuple[EntQuantModel, dict[str, Any]]:
    results: dict[str, Any] = {}
    _build_t0 = time.perf_counter()

    # Detect super weight layers (fallback handling)
    super_weight_layers: set[str] | None = None
    if super_weight_config is not None and super_weight_config.spike_threshold != inf:
        _sw_t0 = time.perf_counter()
        super_weight_layers = detect_fallback_layers(
            model_cfg["base_model_name_or_path"],
            config=super_weight_config,
            device_map="cpu",
            dtype=str_to_dtype(model_cfg["dtype"]),
        )
        results["super_weight_layers"] = sorted(super_weight_layers)
        results["super_weight_layers_time_s"] = time.perf_counter() - _sw_t0
        logger.info(f"Super weight layers: {sorted(super_weight_layers)}")
        logger.info(f"Super weight layer detection time: {results['super_weight_layers_time_s']:.3f}s")

    # Build via EntQuantModel.from_pretrained
    model = EntQuantModel.from_pretrained(
        model_cfg["base_model_name_or_path"],
        quantize=entquant["quantize"],
        compress=entquant["compress"],
        save_dir=save_dir,
        block_pattern=entquant.get("block_pattern", "model.layers.*"),
        device_map=model_cfg["device_map"],
        weight_qtype=entquant["weight_qtype"],
        activation_qtype=entquant.get("activation_qtype"),
        fallback_layers=super_weight_layers,
        include=entquant.get("include"),
        exclude=entquant.get("exclude"),
        optimizer=entquant.get("optimizer"),
        optimizer_fallback=entquant.get("optimizer_fallback"),
        backend=entquant.get("backend"),
        dtype=str_to_dtype(model_cfg["dtype"]),
        model_cls=model_cfg.get("model_cls"),
        model_kwargs=model_cfg.get("model_kwargs"),
    )

    # Save tokenizer alongside checkpoint
    if save_dir:
        save_tokenizer(tokenizer, save_dir)

    # Set generation config on the inner model (if configured)
    if model_cfg.get("generation_config") is not None:
        model.model.generation_config = model_cfg["generation_config"]

    results["build_time_s"] = time.perf_counter() - _build_t0
    logger.info(f"Build time: {results['build_time_s']:.3f}s")

    # Log compression stats if compressed
    if entquant["compress"] and model.compressor is not None:
        stats = model.compression_stats()
        GiB = 1024**3
        logger.info(
            f"\n{'=' * 80}\n"
            f"COMPRESSION STATISTICS\n"
            f"{'=' * 80}\n"
            f"Injected parameters:\n"
            f"  Original ({stats['original_dtype']}):  "
            f"{stats['injected_original_bytes'] / GiB:8.3f} GiB\n"
            f"  Quantized (qtype): "
            f"{stats['injected_quantized_bytes'] / GiB:8.3f} GiB\n"
            f"  Compressed (ANS):  "
            f"{stats['injected_compressed_bytes'] / GiB:8.3f} GiB\n"
            f"  vs original:        "
            f"{stats['injected_vs_original_ratio']:7.2f}x\n\n"
            f"Full model:\n"
            f"  Original ({stats['original_dtype']}):  "
            f"{stats['full_original_bytes'] / GiB:8.3f} GiB\n"
            f"  Compressed:        "
            f"{stats['full_compressed_bytes'] / GiB:8.3f} GiB\n"
            f"  vs original:        "
            f"{stats['full_vs_original_ratio']:7.2f}x\n\n"
            f"Average entropy: {stats['avg_entropy']:.3f}\n"
            f"{'=' * 80}"
        )
        results["compression_stats"] = stats

    clear_cache()
    mem_allocated, mem_reserved = get_memory_stats()
    logger.info(f"Memory allocated: {mem_allocated:8.3f} GiB")
    logger.info(f"Memory reserved:  {mem_reserved:8.3f} GiB")
    results["memory_allocated"] = mem_allocated
    results["memory_reserved"] = mem_reserved

    return model, results


def _patch_hqq_compute_dtype(compute_dtype: torch.dtype):
    from transformers.quantizers.quantizer_hqq import HqqHfQuantizer

    _original_init = HqqHfQuantizer.__init__

    def _patched_init(self, quantization_config, **kwargs):
        _original_init(self, quantization_config, **kwargs)
        self.dtype = compute_dtype

    HqqHfQuantizer.__init__ = _patched_init


@register_workflow("build")
def build_quantized_model(
    model_cfg: dict = "${cfg.model}",
    super_weight_config: SuperWeightsConfig | None = "${cfg.super_weights}",
    quantization_config: Partial = "${cfg.quantization.config}",
    modules_to_exclude: list[str] = "${cfg.quantization.modules_to_exclude}",
) -> tuple[PreTrainedModel, dict[str, Any]]:
    results: dict[str, Any] = {}

    modules_to_exclude = list(modules_to_exclude)  # may be a ListConfig

    if super_weight_config is not None and super_weight_config.spike_threshold != inf:
        super_weight_layers = detect_fallback_layers(
            model_cfg["base_model_name_or_path"],
            config=super_weight_config,
            device_map="cpu",
            dtype=str_to_dtype(model_cfg["dtype"]),
        )
        modules_to_exclude.extend(super_weight_layers)
        results["super_weight_layers"] = sorted(super_weight_layers)
    else:
        results["super_weight_layers"] = None

    if isinstance(quantization_config(), BitsAndBytesConfig):
        quantization_config = quantization_config(llm_int8_skip_modules=modules_to_exclude)
    elif isinstance(quantization_config(), HqqConfig):
        quantization_config = quantization_config(skip_modules=modules_to_exclude)
    elif hasattr(quantization_config(), "modules_to_not_convert"):
        quantization_config = quantization_config(modules_to_not_convert=modules_to_exclude)
    else:
        logger.warning(f"Quantization config {quantization_config} is not supported, ignoring excluded modules.")
        quantization_config = quantization_config()

    # HOTFIX: For some reason, compute_dtype is not properly passed to
    # HqqHfQuantizer. Alternative: use native hqq backend.
    if isinstance(quantization_config, HqqConfig):
        _patch_hqq_compute_dtype(str_to_dtype(model_cfg["dtype"]))

    logger.info(f"Building quantized model with config: {quantization_config}")
    model_cfg["model_kwargs"]["quantization_config"] = quantization_config
    model, results_base = build_base_model(model_cfg)
    results.update(results_base)

    return model, results
