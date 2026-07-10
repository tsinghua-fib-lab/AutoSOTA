from __future__ import annotations

import os.path
from dataclasses import dataclass

import numpy as np

from softmatcha.typing import NDArrayF32

from . import register
from .base import Embedding


@register("transformers")
@dataclass
class EmbeddingTransformers(Embedding):
	"""EmbeddingTransformers class wraps encoder models of huggingface/transformers.

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
		import huggingface_hub
		from huggingface_hub.constants import CONFIG_NAME
		from transformers import AutoModel

		import softmatcha.functional as F

		hf_config_path = huggingface_hub.hf_hub_download(name_or_path, CONFIG_NAME)
		save_dir = os.path.dirname(hf_config_path)
		embedding_path = os.path.join(save_dir, "embedding.npy")
		if not os.path.exists(embedding_path):
			embeddings = AutoModel.from_pretrained(name_or_path).get_input_embeddings()
			embeddings = F.normalize(embeddings.weight.detach().numpy(), axis=-1)
			with open(embedding_path, mode="wb") as f:
				np.save(f, embeddings)
		return np.load(embedding_path, mmap_mode="r" if mmap else None)
