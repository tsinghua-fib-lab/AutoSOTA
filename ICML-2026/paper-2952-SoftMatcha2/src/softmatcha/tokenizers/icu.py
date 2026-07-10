from __future__ import annotations

import os.path
from dataclasses import dataclass

import simdjson

from .base import Tokenizer


class TokenizerICU(Tokenizer):
	@dataclass
	class Config(Tokenizer.Config):
		"""Configuration for tokenizer.

		name_or_path (str): Model name or path.
		lang (str): Language code.
		"""

		lang: str = "en"

	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		return self.dictionary[self.UNK_TOKEN]

	@classmethod
	def build(cls, cfg: TokenizerICU.Config) -> TokenizerICU:
		"""Build an tokenizer class.

		Args:
			cfg (TokenizerICU.Config): Tokenizer configuration.

		Returns:
			TokenizerICU: This class.
		"""
		parser = simdjson.Parser()
		dictionary = parser.load(os.path.join(cfg.name_or_path, "vocab.json"), True)
		dictionary[cls.UNK_TOKEN] = max(dictionary.values()) + 1

		import icu_tokenizer

		return cls(cfg, icu_tokenizer.Tokenizer(lang=cfg.lang), dictionary)

	def tokenize(self, line: str) -> list[str]:
		"""Tokenize the input line.

		Args:
			line (str): An input line.

		Returns:
			list[str]: The tokenized line.
		"""
		line = line.strip()
		line = line.lower()
		return self._tokenizer.tokenize(line)

	def tokenize_raw(self, line: str) -> list[str]:
		line = line.strip()
		return self._tokenizer.tokenize(line)