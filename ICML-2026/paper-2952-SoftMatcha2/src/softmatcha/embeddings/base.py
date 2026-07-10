from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from softmatcha.typing import NDArrayF32


@dataclass
class Embedding(abc.ABC):
	"""Embedding base class.

	embeddings (NDArrayF32): Embedding vectors.
	"""

	embeddings: NDArrayF32

	@dataclass
	class Config:
		"""Configuration for the embedding.

		name_or_path (str): Model name or path.
		mmap (bool): Load the embedding via mmap.
		"""

		name_or_path: str = field(default="", metadata={"cmd": False})
		mmap: bool = False

	def __len__(self) -> int:
		return len(self.embeddings)

	@classmethod
	def build(cls, cfg: Config) -> Embedding:
		"""Build an embedding class.

		Args:
			cfg (Embedding.Config): Embedding configuration.

		Returns:
			Embedding: This class.
		"""
		return cls(cls.load(cfg.name_or_path, mmap=cfg.mmap))

	@classmethod
	@abc.abstractmethod
	def load(cls, name_or_path: str, mmap: bool = False) -> NDArrayF32:
		"""Load an embedding table.

		Args:
			name_or_path (str): Model name or path.
			mmap (bool): Open the embedding file via mmap.

		Returns:
			NDArrayF32: The embedding table.
		"""

	def __call__(self, tokens: list[int]) -> NDArrayF32:
		"""Embed tokens into their vector representations.

		Args:
			tokens (list[int]): Input tokens.

		Returns:
			NDArrayF32: Token embeddings.
		"""
		if len(tokens) == 0:
			return np.array([], dtype=np.float32)
		return self.embeddings[tokens]
