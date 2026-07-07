"""Tokenizer utilities for loading and saving alongside checkpoints."""

import logging
from pathlib import Path
from typing import Any, Type

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def load_tokenizer(
    model_name_or_path: str,
    tokenizer_cls: Type[Tokenizer] = AutoTokenizer,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> Tokenizer:
    """Load a tokenizer from a model identifier or local path."""
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path, **(tokenizer_kwargs or {}))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, save_dir: str | Path) -> None:
    """Save a tokenizer to a directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved tokenizer to {save_dir}")
