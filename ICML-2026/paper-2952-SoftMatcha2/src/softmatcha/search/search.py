from __future__ import annotations
import warnings
import numba as nb
import numba.typed.typedlist as tl_mod
import numpy as np
import os
import ctypes
import mmap
import bisect
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaTypeSafetyWarning
from numba.typed.typedlist import List
import softmatcha.functional as F
from softmatcha import stopwatch
from softmatcha.embeddings import Embedding
from softmatcha.struct import Pattern
from softmatcha.tokenizers import Tokenizer
from softmatcha.typing import NDArrayF32, NDArrayI64
from softmatcha_rs import enumerate_candidates_rs, get_match_range_rs
warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)
numba_array_type = nb.types.Array(nb.int64, 1, "C")



# =====================================================================================================================
# 1. function for numba speedup
# =====================================================================================================================
@nb.njit(nb.types.void(nb.types.ListType(numba_array_type), numba_array_type))
def append_jit(lst: nb.types.ListType, elm: nb.types.Array) -> None:
	lst.append(elm)
append = append_jit.overloads[
	(nb.types.ListType(numba_array_type), numba_array_type)
].entry_point

def monkey_patch_caching(mod, exclude=[]):
	for name, val in mod.__dict__.items():
		if isinstance(val, Dispatcher) and name not in exclude:
			val.enable_caching()
monkey_patch_caching(tl_mod, ["_sort"])
List.empty_list(numba_array_type)



# =====================================================================================================================
# 2. compute similarity & norm
# =====================================================================================================================
def compute_similarity(pat: Pattern, qs: Searcher) -> NDArrayF32:
	scores = np.zeros((len(pat), len(qs.vocabulary_embeddings)), dtype=np.float32)
	with stopwatch.timers["similarity"]:
		scores[:, range(len(qs.vocabulary_embeddings))] = F.matmul(pat.embeddings, qs.vocabulary_embeddings)
	return scores


# =====================================================================================================================
# 3. main part (searcher class)
# =====================================================================================================================
class Searcher:
	tokens: NDArrayI64
	vocabs: NDArrayI64

	# we read the information of the built index in this __init__ function
	def __init__(self, index_path: str,
				 tokenizer: Tokenizer, embedding: Embedding) -> None:
		
		# 3-1. load metadata
		print("loading begin...")
		meta_path = index_path + "/metadata.bin"
		meta = np.memmap(meta_path, dtype=np.uint64, mode='r', shape=(os.path.getsize(meta_path)//8,))
		self.num_tokens = meta[ 0]
		TOKEN_SIZE      = meta[ 4]
		self.max_vocab  = meta[ 6]
		DATA_BEGIN      = meta[ 7]
		self.pair_cons  = meta[ 8]
		self.trio_cons  = meta[ 9]
		self.rough_div  = meta[10]
		self.sa_size    = meta[11]
		self.pair_file  = meta[12]
		self.trio_file  = meta[13]
		OFFSET_FREQ = DATA_BEGIN
		OFFSET_NORM = OFFSET_FREQ + self.max_vocab
		OFFSET_PAIR = OFFSET_NORM + self.max_vocab // 2
		OFFSET_TRIO = OFFSET_PAIR + self.pair_file
		DATA_SIZE   = OFFSET_TRIO + self.trio_file
		self.pair_  = np.array(meta[OFFSET_PAIR : OFFSET_TRIO], copy=True)
		self.trio_  = np.array(meta[OFFSET_TRIO : DATA_SIZE  ], copy=True)
		div_num = (self.num_tokens + self.rough_div - 1) // self.rough_div + 1

		# 3-2. load offsets
		offsets_path = index_path + "/offset.bin"
		offsets = np.memmap(offsets_path, dtype=np.uint8, mode='r', shape=(os.path.getsize(offsets_path),))
		self.byte_offset1  = offsets[0 : TOKEN_SIZE]
		self.byte_offset2  = offsets[TOKEN_SIZE : TOKEN_SIZE + TOKEN_SIZE // 32].view(np.uint64)
		self.black_list    = np.array(offsets[TOKEN_SIZE + TOKEN_SIZE // 32 : ].view(np.uint64), copy=True)

		# 3-3. load rough
		rough_path = index_path + "/rough.bin"
		roughs = np.memmap(rough_path, dtype=np.uint64, mode='r', shape=(os.path.getsize(rough_path)//8,))
		self.rough = np.array(roughs[0 : div_num * 5], copy=True)

		# 3-4. load sa
		sa_path = index_path + "/sa.bin"
		self.sa = np.memmap(sa_path, dtype=np.uint8, mode='r', shape=(os.path.getsize(sa_path),))

		# 3-5. load index
		self.idx_fname = str(index_path) + "/index.bin"
		self.idx_length = os.path.getsize(index_path + "/index.bin") // (4 * 8)

		# 3-6. load norm (special)
		self.norm_  = meta[OFFSET_NORM : OFFSET_PAIR]
		self.norm__ = self.norm_.view(np.float32)
		self.norm   = np.zeros(self.max_vocab, dtype=np.float32)
		for i in range(self.max_vocab):
			if i >= 50_000:
				self.norm[i] = 10.0 ** 10
			else:
				self.norm[i] = self.norm__[i]
		self.vocabulary_embeddings = self.normalize(embedding.embeddings[range(min(self.max_vocab, len(embedding.embeddings)))])
		print("loading finished")

		# 3-7. load text-file name
		self.text_file = ""
		for i in range(512, 768):
			if meta[i] == 0:
				break
			self.text_file += chr(meta[i])
		self.tokenizer = tokenizer



	# =================================================================================================================
	# 4. utils
	# =================================================================================================================
	# 4-0. normalize embedding vectors
	@staticmethod
	def normalize(emb: NDArrayF32) -> NDArrayF32:
		emb2 = emb.copy()
		for i in range(len(emb2)):
			emb2[i] = F.normalize(emb2[i])
		return emb2

	# 4-1. get a string including padding from the corpus text
	def get_contexts(self, path: str, l: int, r: int, padding: int, encoding: str = "utf-8") -> list[str]:
		file_size = os.path.getsize(path)
		l = max(0, min(l, file_size))
		r = max(0, min(r, file_size))
		if r < l:
			r = l

		with open(path, "rb") as f:
			# (a) calculate left
			left_start = max(0, l - padding)
			left_len = l - left_start
			f.seek(left_start)
			left_bytes = f.read(left_len)
			left_text_full = left_bytes.decode(encoding, errors="ignore")
			last_newline_idx = left_text_full.rfind("\n")
			if last_newline_idx != -1:
				left_text = left_text_full[last_newline_idx + 1 :]
				cut_by_newline = True
			else:
				left_text = left_text_full
				cut_by_newline = False
			left_text = left_text.lstrip(" ")
			if (not cut_by_newline and left_text) and left_start != 0:
				left_text = "..." + left_text

			# (b) calculate center
			f.seek(l)
			middle_bytes = f.read(r - l)
			middle_text = middle_bytes.decode(encoding, errors="ignore")
			middle_text = middle_text.replace("\n", " ")

			# (c) calculate right
			right_end = min(file_size, r + padding)
			right_len = right_end - r
			f.seek(r)
			right_bytes = f.read(right_len)
			right_text_full = right_bytes.decode(encoding, errors="ignore")
			first_newline_idx = right_text_full.find("\n")
			if first_newline_idx != -1:
				right_text = right_text_full[:first_newline_idx]
				cut_by_newline = True
			else:
				right_text = right_text_full
				cut_by_newline = False
			right_text = right_text.rstrip(" ")
			if (not cut_by_newline and right_text) and right_end != file_size:
				right_text = right_text + "..."
		
		# return
		return [left_text, middle_text, right_text]

	# 4-2. get bytes
	def get_bytes_in_token(self, token_position: int) -> int:
		black_list_cur = bisect.bisect_left(self.black_list[0:len(self.black_list)//2], (token_position >> 8) << 8)
		sum = 0
		for i in range(token_position % 256):
			v = int(self.byte_offset1[((token_position >> 8) << 8) + i + 1])
			if v == 255:
				v = self.black_list[len(self.black_list)//2 + black_list_cur]
				black_list_cur += 1
			sum += v
		return self.byte_offset2[token_position >> 8] + sum

	# 4-3. get match position "randomly"
	def get_match_position(self, left_: int, right_: int, search_num: int) -> NDArrayI64:
		if right_ == left_:
			return np.zeros(0, dtype=np.int64)
		seed1 = 86_912_010_012_073
		seed2 =  1_234_567_890_123
		ret = np.zeros(min(right_ - left_, search_num), dtype=np.int64)
		for i in range(len(ret)):
			idx = (seed2 + seed1 * i) % (right_ - left_)
			for j in range(self.sa_size):
				ret[i] += (2 ** (8 * j)) * int(self.sa[(left_ + idx) * self.sa_size + j])
		return ret
	
	# 4-4. get exact match
	def get_exact_match(self, pattern: Pattern, search_num: int, padding: int):
		l, r = get_match_range_rs(np.asarray(pattern.tokens, dtype=np.uint64), self.rough, self.idx_fname, self.rough_div, self.num_tokens, self.idx_length)
		list_pos = self.get_match_position(l, r, search_num)
		list_str = []
		for pos in list_pos:
			last = len(pattern.tokens) - 1
			bytes_left = self.get_bytes_in_token(pos)
			bytes_right = self.get_bytes_in_token(pos + last)
			bytes_right += len(self.tokenizer.tokens[pattern.tokens[last]].encode("utf-8"))
			qlist = self.get_contexts(self.text_file, int(bytes_left), int(bytes_right), padding)
			list_str.append(qlist)
		return list_str, r - l
	


	# =================================================================================================================
	# 5. search function
	# =================================================================================================================
	def search(
			self,
			pattern: Pattern,
			num_candidates: int,
			min_similarity: float,
			max_runtime: float,          
		) -> tuple[NDArrayI64]:

		# 5-1. get similarity
		with stopwatch.timers["similarity"]:
			score_matrix = compute_similarity(pattern, self)
			score_matrix = np.asarray(score_matrix, dtype=np.float32, order="C")
		
		# 5-2. searching
		with stopwatch.timers["search"]:
			cand, cand_score, count, thres = enumerate_candidates_rs(
				np.asarray(pattern.tokens, dtype=np.uint64),
				score_matrix,
				self.rough,
				self.pair_,
				self.trio_,
				self.norm,
				self.idx_fname,
				self.num_tokens,
				self.idx_length,
				self.pair_cons,
				self.trio_cons,
				self.rough_div,
				num_candidates,
				min_similarity,
				max_runtime,
			)
		cand = cand[0:num_candidates]
		cand_score = cand_score[0:num_candidates]
		count = count[0:num_candidates]
		return (cand, cand_score, count, thres)