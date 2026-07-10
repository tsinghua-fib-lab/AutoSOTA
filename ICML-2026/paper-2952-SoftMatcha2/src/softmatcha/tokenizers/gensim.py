from __future__ import annotations

from dataclasses import dataclass

from . import register
from .base import Tokenizer


@register("gensim")
class TokenizerGensim(Tokenizer):
	"""TokenizerGensim class."""

	@dataclass
	class Config(Tokenizer.Config):
		"""Configuration for tokenizer.

		name_or_path (str): Model name or path.
		split_hyphen (bool): Split hyphens.
		"""

		# Split hyphens.
		split_hyphen: bool = False

	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		return self.dictionary[self.UNK_TOKEN]

	@classmethod
	def build(cls, cfg: TokenizerGensim.Config) -> Tokenizer:
		"""Build an tokenizer class.

		Args:
			cfg (TokenizerGensim.Config): Tokenizer configuration.

		Returns:
			TokenizerGensim: This class.
		"""
		from softmatcha.tokenizers.moses import TokenizerMoses
		from softmatcha.utils import gensim as gensim_utils

		tokenizer = TokenizerMoses.build(
			TokenizerMoses.Config(
				gensim_utils.download_gensim_model(cfg.name_or_path),
				split_hyphen=cfg.split_hyphen,
			)
		)
		return tokenizer
