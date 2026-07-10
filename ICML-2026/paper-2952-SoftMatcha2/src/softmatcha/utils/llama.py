import json
import os.path
import numpy as np
import softmatcha.functional as F
import torch
from transformers import AutoModel, AutoTokenizer

def download_hf_model(name_or_path: str) -> str:
	"""Download and extract a Hugging Face model and its vocabulary.

	Args:
		name_or_path (str): Model name or path (e.g., "meta-llama/Llama-2-7b-hf").

	Returns:
		str: Path to the saved directory.
	"""
	llama_data_dir = os.environ.get(
		"LLAMA_DATA_DIR", os.path.expanduser("~/llama-data")
	)
	safe_name = name_or_path.replace("/", "_")
	save_dir = os.path.join(llama_data_dir, safe_name)
	
	vocab_file = os.path.join(save_dir, "vocab.json")
	embedding_file = os.path.join(save_dir, "embedding.npy")
	if os.path.exists(vocab_file) and os.path.exists(embedding_file):
		return save_dir

	os.makedirs(save_dir, exist_ok=True)
	print(f"Loading model: {name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(name_or_path)
	model = AutoModel.from_pretrained(name_or_path)
	vocab = tokenizer.get_vocab()
	with open(vocab_file, mode="w") as f:
		json.dump(vocab, f, ensure_ascii=False, indent="")

	embeddings = model.get_input_embeddings().weight.detach().numpy().astype(np.float32)
	
	embeddings = np.concatenate(
		[
			embeddings,
			np.zeros((1, embeddings.shape[1]), dtype=np.float32),
		]
	)

	with open(embedding_file, mode="wb") as f:
		np.save(f, embeddings)

	return save_dir