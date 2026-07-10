from __future__ import annotations

import os.path

import simdjson

from .base import Tokenizer


class TokenizerMecab(Tokenizer):
	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		return self.dictionary[self.UNK_TOKEN]

	@classmethod
	def build(cls, cfg: TokenizerMecab.Config) -> TokenizerMecab:
		"""Build an tokenizer class.

		Args:
			cfg (TokenizerMecab.Config): Tokenizer configuration.

		Returns:
			TokenizerMecab: This class.
		"""

		parser = simdjson.Parser()
		dictionary = parser.load(os.path.join(cfg.name_or_path, "vocab.json"), True)
		dictionary[cls.UNK_TOKEN] = max(dictionary.values()) + 1

		import ipadic
		import MeCab

		return cls(cfg, MeCab.Tagger(f"-Owakati {ipadic.MECAB_ARGS}"), dictionary)

	def tokenize(self, line: str) -> list[str]:
		"""Tokenize the input line.

		Args:
			line (str): An input line.

		Returns:
			list[str]: The tokenized line.
		"""
		line = line.strip()
		parsed_result = self._tokenizer.parse(line)
		if parsed_result is None:
			return []
		return parsed_result.rstrip().split(" ")
	
	def tokenize_raw(self, line: str) -> list[str]:
		line = line.strip()
		parsed_result = self._tokenizer.parse(line)
		if parsed_result is None:
			return []
		return parsed_result.rstrip().split(" ")