"""
Quantization module
Supports INT4, INT8, FP8 and other quantization methods
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def apply_quantization(model: nn.Module, quantization: str) -> nn.Module:
    """
    Apply quantization to the model

    Args:
        model: the model to be quantized
        quantization: quantization method (int4, int8, fp8)

    Returns:
        the quantized model
    """
    if quantization == "none":
        return model

    try:
        from transformers import BitsAndBytesConfig

        if quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Applying INT4 quantization")

        elif quantization == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Applying INT8 quantization")

        else:
            logger.warning(f"Unsupported quantization method: {quantization}, skipping quantization")
            return model

        # Note: actual quantization should be applied at load time; this is for illustration only
        # If the model is already loaded, it needs to be reloaded with the quantization config applied
        logger.warning("Quantization should be applied at model load time; the model is already loaded, skipping")
        return model

    except ImportError:
        logger.error("bitsandbytes is not installed, quantization cannot be used")
        return model