from __future__ import annotations
import os
import logging
import shutil
import numpy as np
from pathlib import Path
from softmatcha import stopwatch
from softmatcha_rs import build_sa_rs
from softmatcha.embeddings import Embedding
from softmatcha.utils.makefile import make_file
logger = logging.getLogger(__name__)

def build_index(
	index_path  : str,
	embedding   : Embedding,
	write_thread: int,
	chunk_size  : int,
	pair_cons   : int,
	trio_cons   : int,
	rough_div   : int,
	num_shards  : int
) -> None:
	
	
	# =================================================================================================================
	# 1. load the token information
	# =================================================================================================================
	# 1-1. load previous files
	index_path_ = Path(index_path)
	index_path_.mkdir(parents=True, exist_ok=True)
	if True:
		bin_path = index_path_ / "tokens.bin"
		tokens = np.memmap(bin_path, dtype=np.uint32, mode='r+', offset=0, shape=(os.path.getsize(bin_path) // 4,))
	if True:
		bin_path = index_path_ / "metadata.bin"
		initial = np.memmap(bin_path, dtype=np.uint64, mode='r+', offset=0, shape=(os.path.getsize(bin_path) // 8,))

	# 1-2. define constants
	num_tokens = initial[0]
	max_tokens = initial[4]
	MAX_VOCAB  = initial[6]
	DATA_BEGIN = initial[7]
	FILE_PAIR  = (pair_cons * pair_cons + 63) // 64
	FILE_TRIO  = ((trio_cons * (trio_cons + 1) * (trio_cons + 2)) // 6 + 63) // 64
	SA_SIZE    = 0
	for i in range(8):
		if (2 ** (i * 8)) >= max_tokens:
			SA_SIZE = i
			break
	initial[ 8] = pair_cons
	initial[ 9] = trio_cons
	initial[10] = rough_div
	initial[11] = SA_SIZE
	initial[12] = FILE_PAIR
	initial[13] = FILE_TRIO
	initial.flush()
	del initial

	# 1-3. update a metadata file
	OFFSET_FREQ = DATA_BEGIN
	OFFSET_NORM = OFFSET_FREQ + MAX_VOCAB
	OFFSET_PAIR = OFFSET_NORM + MAX_VOCAB // 2
	OFFSET_TRIO = OFFSET_PAIR + FILE_PAIR
	DATA_SIZE   = OFFSET_TRIO + FILE_TRIO
	with open(bin_path, "r+b") as f:
		f.truncate(DATA_SIZE * 8)
	arr = np.memmap(bin_path, dtype=np.uint64, mode='r+', offset=0, shape=(os.path.getsize(bin_path) // 8,))
	
	# 1-4. get arrays from metadata
	freq = arr[OFFSET_FREQ : OFFSET_NORM]
	norm = arr[OFFSET_NORM : OFFSET_PAIR].view(np.float32)
	pair = arr[OFFSET_PAIR : OFFSET_TRIO]
	trio = arr[OFFSET_TRIO : DATA_SIZE]
	

	# =================================================================================================================
	# 2. make the index file
	# =================================================================================================================
	# 2-1. make index
	with stopwatch.timers["index"]:
		build_sa_rs(
			tokens, freq, pair, trio, str(index_path_),
			max_tokens, num_tokens, rough_div,
			pair_cons, trio_cons,
			chunk_size, write_thread, num_shards, SA_SIZE
		)
	
	# 2-2. the function to get norm (zipfian whitening)
	def get_norm(embedding: np.ndarray, freq: np.ndarray, eps: float = 1e-11):
		embedding = np.asarray(embedding, dtype=float)
		freq = np.asarray(freq, dtype=float)
		V_total, D = embedding.shape
		norm = np.full(V_total, 1e10, dtype=np.float32)
		whitened = np.zeros((V_total, D), dtype=np.float32)

		# enumerate freq > 0 words
		mask = freq > 0
		if not np.any(mask):
			return norm, whitened
		emb_valid = embedding[mask]
		freq_valid = freq[mask]

		# parameter
		p = freq_valid / freq_valid.sum()
		mu_hat = (p[:, None] * emb_valid).sum(axis=0)
		centered = emb_valid - mu_hat
		
		# zipfian whitening
		sqrt_p = np.sqrt(p)
		W_p = centered * sqrt_p[:, None]
		covariance = W_p.T @ W_p
		eigvals, eigvecs = np.linalg.eigh(covariance)
		inv_S = np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps)))
		whitening_mat = eigvecs @ inv_S @ eigvecs.T
		whitening_mat = eigvecs @ inv_S
		whitened_valid = centered @ whitening_mat
		norm_valid = np.linalg.norm(whitened_valid, axis=1)
		whitened[mask] = whitened_valid
		norm[mask] = norm_valid
		return norm
	
	# 2-3. get norm
	with stopwatch.timers["norm"]:
		logger.info(f"Calculating norm... (1-3 min.)")
		max_len = min(MAX_VOCAB, len(embedding.embeddings))
		norm[0:max_len] = get_norm(embedding.embeddings[0:max_len], freq[0:max_len])
		for i in range(MAX_VOCAB):
			norm[i] = norm[i] * norm[i]
		logger.info(f"Calculating norm finished")

	# 2-4. flush
	with stopwatch.timers["flush"]:
		arr.flush()
		del arr