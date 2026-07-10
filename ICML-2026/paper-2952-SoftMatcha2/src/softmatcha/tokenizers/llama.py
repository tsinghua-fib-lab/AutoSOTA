from __future__ import annotations
from dataclasses import dataclass
from . import register
from .base import Tokenizer

@register("llama")
class TokenizerLlama(Tokenizer):
	"""Tokenizer for Llama (and other Hugging Face models)."""

	@dataclass
	class Config(Tokenizer.Config):
		"""Configuration for tokenizer.
		
		name_or_path (str): Hugging Face model name or path.
		"""
		pass

	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		if self._tokenizer.unk_token_id is not None:
			return self._tokenizer.unk_token_id
		return 0

	@classmethod
	def build(cls, cfg: TokenizerLlama.Config) -> Tokenizer:
		import os
		import simdjson
		from transformers import AutoTokenizer
		from softmatcha.utils.llama import download_hf_model
		"""Build the tokenizer class."""
		save_dir = download_hf_model(cfg.name_or_path)
		parser = simdjson.Parser()
		dictionary = parser.load(os.path.join(save_dir, "vocab.json"), True)
		if cls.UNK_TOKEN not in dictionary:
			dictionary[cls.UNK_TOKEN] = max(dictionary.values()) + 1
		hf_tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, use_fast=True)

		return cls(cfg, hf_tokenizer, dictionary)

	def tokenize(self, line: str) -> list[str]:
		"""Tokenize the input line using Hugging Face tokenizer.

		Args:
			line (str): An input line.

		Returns:
			list[str]: The tokenized line (list of sub-words).
		"""
		tokens = self._tokenizer.tokenize(line.strip())
		return tokens
	
	def tokenize_raw(self, line: str) -> list[str]:
		tokens = self._tokenizer.tokenize(line.strip())
		return tokens