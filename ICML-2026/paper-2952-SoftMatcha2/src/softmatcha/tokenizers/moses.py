from __future__ import annotations

import os
import re
from dataclasses import dataclass

import simdjson
from sacremoses.tokenize import MosesTokenizer

from .base import Tokenizer


class MosesTokenizerFast(MosesTokenizer):
	IsAlphaSet = set(MosesTokenizer.IsAlpha)
	IsLowerSet = set(MosesTokenizer.IsLower)

	DOTMULTI_SUBSTITUTION = re.compile(r"\.([\.]+)")
	DOTMULTI = re.compile(r"DOTMULTI\.")
	DOTMULTI_REPLACE_RULE = re.compile(r"DOTMULTI\.([^\.])")
	DOTDOTMULTI = re.compile(r"DOTDOTMULTI")
	DOTMULTI_RESTORE = re.compile(r"DOTMULTI")

	NUMERIC_ONLY = re.compile(r"[\s]+(\#NUMERIC_ONLY\#)")

	TOKEN_ENDS_WITH_PERIOD = re.compile(r"^(\S+)\.$")
	NUMERIC = re.compile(r"^[0-9]+")

	def replace_multidots(self, text):
		text = self.DOTMULTI_SUBSTITUTION.sub(r" DOTMULTI\1", text)
		while self.DOTMULTI.search(text):
			text = self.DOTMULTI_REPLACE_RULE.sub(r"DOTDOTMULTI \1", text)
			text = self.DOTMULTI.sub("DOTDOTMULTI", text)
		return text

	def restore_multidots(self, text):
		while self.DOTDOTMULTI.search(text):
			text = self.DOTDOTMULTI.sub(r"DOTMULTI.", text)
		return self.DOTMULTI_RESTORE.sub(r".", text)
		return text

	def has_numeric_only(self, text):
		return bool(self.NUMERIC_ONLY.search(text))

	def handles_nonbreaking_prefixes(self, text):
		# Splits the text into tokens to check for nonbreaking prefixes.
		tokens = text.split()
		num_tokens = len(tokens)
		for i, token in enumerate(tokens):
			# Checks if token ends with a fullstop.
			token_ends_with_period = self.TOKEN_ENDS_WITH_PERIOD.search(token)
			if token_ends_with_period:
				prefix = token_ends_with_period.group(1)
				# Checks for 3 conditions if
				# i.   the prefix contains a fullstop and
				#      any char in the prefix is within the IsAlpha charset
				# ii.  the prefix is in the list of nonbreaking prefixes and
				#      does not contain #NUMERIC_ONLY#
				# iii. the token is not the last token and that the
				#      next token contains all lowercase.
				if (
					("." in prefix and self.isanyalpha(prefix))
					or (
						prefix in self.NONBREAKING_PREFIXES
						and prefix not in self.NUMERIC_ONLY_PREFIXES
					)
					or (
						i != num_tokens - 1
						and tokens[i + 1]
						and self.islower(tokens[i + 1][0])
					)
				):
					pass  # No change to the token.
				# Checks if the prefix is in NUMERIC_ONLY_PREFIXES
				# and ensures that the next word is a digit.
				elif (
					prefix in self.NUMERIC_ONLY_PREFIXES
					and (i + 1) < num_tokens
					and self.NUMERIC.search(tokens[i + 1])
				):
					pass  # No change to the token.
				else:  # Otherwise, adds a space after the tokens before a dot.
					tokens[i] = prefix + " ."
		return " ".join(tokens)  # Stitch the tokens back.

	def islower(self, text):
		# return not set(text).difference(self.IsLowerSet)
		return bool(set(text) <= self.IsLowerSet)

	def isanyalpha(self, text):
		# return any(set(text).intersection(self.IsAlphaSet))
		return bool(set(text) & self.IsAlphaSet)


class TokenizerMoses(Tokenizer):
	"""Tokenizer class."""

	@dataclass
	class Config(Tokenizer.Config):
		"""Configuration for tokenizer.

		name_or_path (str): Model name or path.
		lang (str): Language code.
		split_hyphen (bool): Split hyphens.
		"""

		lang: str = "en"
		split_hyphen: bool = False

	@property
	def unk_idx(self) -> int:
		"""Return the unknown index."""
		return self.dictionary[self.UNK_TOKEN]

	@classmethod
	def build(cls, cfg: TokenizerMoses.Config) -> TokenizerMoses:
		"""Build an tokenizer class.

		Args:
			cfg (TokenizerMoses.Config): Tokenizer configuration.

		Returns:
			TokenizerMoses: This class.
		"""
		parser = simdjson.Parser()
		dictionary = parser.load(os.path.join(cfg.name_or_path, "vocab.json"), True)
		dictionary[cls.UNK_TOKEN] = max(dictionary.values()) + 1
		tokenizer = MosesTokenizerFast(lang=cfg.lang)
		if cfg.split_hyphen:
			tokenizer.AGGRESSIVE_HYPHEN_SPLIT = (
				tokenizer.AGGRESSIVE_HYPHEN_SPLIT[0],
				r"\1 - ",
			)
		return cls(cfg, tokenizer, dictionary)

	def tokenize(self, line: str) -> list[str]:
		"""Tokenize the input line.

		Args:
			line (str): An input line.

		Returns:
			list[str]: The tokenized line.
		"""

		tokens = self._tokenizer.tokenize(
			line.strip(),
			aggressive_dash_splits=self.cfg.split_hyphen,
			return_str=False,
			escape=False,
		)
		return [t.lower() for t in tokens]

	def tokenize_raw(self, line: str) -> list[str]:
		tokens = self._tokenizer.tokenize(
			line.strip(),
			aggressive_dash_splits=self.cfg.split_hyphen,
			return_str=False,
			escape=False,
		)
		return [t for t in tokens]