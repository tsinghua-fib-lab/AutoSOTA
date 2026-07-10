from __future__ import annotations

from . import register
from .base import Tokenizer


@register("transformers")
class TokenizerTransformers(Tokenizer):
	"""Tokenizer class."""

	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		return self._tokenizer.pad_token_id

	@classmethod
	def build(cls, cfg: TokenizerTransformers.Config) -> TokenizerTransformers:
		"""Build an tokenizer class.

		Args:
			cfg (TokenizerTransformers.Config): Tokenizer configuration.

		Returns:
			TokenizerTransformers: This class.
		"""
		from transformers import AutoTokenizer
		from transformers.tokenization_utils import PreTrainedTokenizer

		tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path)
		return cls(cfg, tokenizer, tokenizer.get_vocab())

	def tokenize(self, line: str) -> list[str]:
		"""Tokenize the input line.

		Args:
			line (str): An input line.

		Returns:
			list[str]: The tokenized line.
		"""
		return self._tokenizer.tokenize(line, verbose=False)
