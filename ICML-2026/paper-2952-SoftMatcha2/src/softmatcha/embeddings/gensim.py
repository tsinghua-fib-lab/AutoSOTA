from __future__ import annotations

import os.path
from dataclasses import dataclass

import numpy as np

from softmatcha.typing import NDArrayF32

from . import register
from .base import Embedding


@register("gensim")
@dataclass
class EmbeddingGensim(Embedding):
	"""EmbeddingGensim class wraps various gensim models, e.g., glove.

	embeddings (NDArrayF32): Embedding vectors.
	"""

	@classmethod
	def load(cls, name_or_path: str, mmap: bool = False) -> NDArrayF32:
		"""Load an embedding table.

		Args:
			name_or_path (str): Model name or path.
			mmap (bool): Open the embedding file via mmap.

		Returns:
			NDArrayF32: The embedding table.
		"""
		from softmatcha.utils import gensim as gensim_utils

		save_dir = gensim_utils.download_gensim_model(name_or_path)
		embed = np.load(
			os.path.join(save_dir, "embedding.npy"), mmap_mode="r" if mmap else None
		)
		return embed
