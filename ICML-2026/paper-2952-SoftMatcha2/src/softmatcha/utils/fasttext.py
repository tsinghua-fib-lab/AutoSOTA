import json
import os.path

import numpy as np
import tqdm

import softmatcha.functional as F


def download_fasttext_model(name: str) -> str:
	"""Download and extract a fasttext model and its vocabulary.

	Args:
		name (str): A model name.

	Returns:
		str: Path to the saved directory.
	"""
	import fasttext
	import huggingface_hub

	path = huggingface_hub.hf_hub_download(f"facebook/{name}", filename="model.bin")
	save_dir = os.path.dirname(os.path.abspath(path))
	vocab_file = os.path.join(save_dir, "vocab.json")
	embedding_file = os.path.join(save_dir, "embedding.npy")

	if not os.path.exists(vocab_file) or not os.path.exists(embedding_file):
		model = fasttext.load_model(path)
		words = model.get_words(on_unicode_error="replace")
		with open(vocab_file, mode="w") as f:
			json.dump(
				{word: idx for idx, word in enumerate(words)},
				f,
				ensure_ascii=False,
				indent="",
			)

		embeddings = np.zeros((len(words) + 1, model.get_dimension()), dtype=np.float32)
		for idx in tqdm.tqdm(range(len(words))):
			embeddings[idx] = model.get_input_vector(idx)
		with open(embedding_file, mode="wb") as f:
			np.save(f, embeddings)

	return save_dir
