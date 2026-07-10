from __future__ import annotations

import os.path
from dataclasses import dataclass

import numpy as np

from softmatcha.typing import NDArrayF32

from . import register
from .base import Embedding


@register("llama")
@dataclass
class EmbeddingLlama(Embedding):
	"""EmbeddingLlama class wraps various llama models, e.g., glove.

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
		from softmatcha.utils import llama as llama_utils

		save_dir = llama_utils.download_hf_model(name_or_path)
		embed = np.load(
			os.path.join(save_dir, "embedding.npy"), mmap_mode="r" if mmap else None
		)
		return embed
