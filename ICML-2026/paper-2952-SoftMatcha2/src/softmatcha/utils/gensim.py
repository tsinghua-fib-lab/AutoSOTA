import json
import os.path

import numpy as np

import softmatcha.functional as F


def download_gensim_model(name: str) -> str:
	"""Download and extract a gensim model and its vocabulary.

	Args:
		name (str): A model name.

	Returns:
		str: Path to the saved directory.
	"""
	# c.f. gensim.download.BASE_DIR
	gensim_data_dir = os.environ.get(
		"GENSIM_DATA_DIR", os.path.expanduser("~/gensim-data")
	)
	save_dir = os.path.join(gensim_data_dir, name)
	vocab_file = os.path.join(save_dir, "vocab.json")
	embedding_file = os.path.join(save_dir, "embedding.npy")
	if not os.path.exists(vocab_file) or not os.path.exists(embedding_file):
		import gensim.downloader

		gensim_model = gensim.downloader.load(name)

		with open(vocab_file, mode="w") as f:
			json.dump(gensim_model.key_to_index, f, ensure_ascii=False, indent="")

		embeddings = gensim_model.vectors.astype(np.float32)
		embeddings = np.concatenate(
			[
				embeddings,
				np.zeros((1, embeddings.shape[1]), dtype=np.float32),
			]
		)
		with open(embedding_file, mode="wb") as f:
			np.save(f, embeddings)

	return save_dir