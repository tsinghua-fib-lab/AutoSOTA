from __future__ import annotations

import re

from . import register
from .base import Tokenizer


@register("fasttext")
class TokenizerFasttext(Tokenizer):
	"""TokenizerFasttext class."""

	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		return self.dictionary[self.UNK_TOKEN]

	@classmethod
	def build(cls, cfg: TokenizerFasttext.Config) -> Tokenizer:
		"""Build an tokenizer class.

		Args:
			cfg (TokenizerFasttext.Config): Tokenizer configuration.

		Returns:
			Tokenizer: This class.
		"""
		from softmatcha.utils import fasttext as fasttext_utils

		save_dir = fasttext_utils.download_fasttext_model(cfg.name_or_path)

		lang_match = re.search(r"fasttext-(.+)-vectors", cfg.name_or_path)
		if lang_match is None:
			raise ValueError(f"Cannot load the model: {cfg.name_or_path}")
		lang = lang_match.group(1)

		match lang:
			case "ja":
				from .mecab import TokenizerMecab

				return TokenizerMecab.build(TokenizerMecab.Config(save_dir))
			case "el" | "la" | "he":
				# TODO(deguchi): Add other languages
				from .moses import TokenizerMoses

				return TokenizerMoses.build(TokenizerMoses.Config(save_dir, lang=lang))
			case _:
				from .icu import TokenizerICU

				return TokenizerICU.build(TokenizerICU.Config(save_dir, lang=lang))
