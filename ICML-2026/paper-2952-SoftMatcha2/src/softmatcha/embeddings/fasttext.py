from __future__ import annotations

import os.path
from dataclasses import dataclass

import numpy as np

from softmatcha.typing import NDArrayF32

from . import register
from .base import Embedding


@register("fasttext")
@dataclass
class EmbeddingFasttext(Embedding):
	"""EmbeddingFasttext class wraps fasttext models.

	embeddings (NDArrayF32): Embedding vectors.
	"""

	@classmethod
	def load(cls, name_or_path: str, mmap: bool = False) -> NDArrayF32:
		"""Load an embedding table.

		Args:
			name_or_path (str): Model name.
			mmap (bool): Open the embedding file via mmap.

		Returns:
			NDArrayF32: The embedding table.
		"""
		from softmatcha.utils import fasttext as fasttext_utils

		save_dir = fasttext_utils.download_fasttext_model(name_or_path)
		embed = np.load(os.path.join(save_dir, "embedding.npy"), mmap_mode="r" if mmap else None)
		return embed
