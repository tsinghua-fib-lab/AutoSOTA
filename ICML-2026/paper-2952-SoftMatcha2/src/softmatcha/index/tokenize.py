from __future__ import annotations
import os
import gc
import random
import logging
import numba as nb
import numpy as np
import concurrent.futures
from pathlib import Path
from softmatcha import stopwatch
from softmatcha.tokenizers import Tokenizer
from softmatcha.utils.io import buffer_lines
from softmatcha.utils.makefile import make_file
from softmatcha.utils.custom_tqdm import CustomTqdm
logger = logging.getLogger(__name__)
_worker_tokenizer = None


# =====================================================================================================================
# Preparation
# =====================================================================================================================
def init_worker(tokenizer: Tokenizer, cfg):
	global _worker_tokenizer
	_worker_tokenizer = tokenizer
	tokenizer.build(cfg)

def tokenize_count(line: str):
	global _worker_tokenizer
	symbols = _worker_tokenizer.tokenize(line)
	return len(symbols)

def tokenize_encode_offsets(line: str):
	global _worker_tokenizer
	symbols = _worker_tokenizer.tokenize_raw(line)
	token_ids = _worker_tokenizer.encode([sym.lower() for sym in symbols])
	offsets = _worker_tokenizer.get_span_start_positions(
		line,
		symbols,
	)
	return token_ids, offsets

def get_custom_tqdm(num):
	return CustomTqdm(
		total=num,
		bar_format="{bar:64} {n_fmt}/{total_fmt} ETA {remaining}",
		ascii="░█",
		dynamic_ncols=True
	)

def read_random_chunk_safe(file_path, start_pos, chunk_size):
	with open(file_path, "rb") as f:
		f.seek(start_pos)
		if start_pos > 0:
			while True:
				byte = f.read(1)
				if not byte:
					return "", 0
				if not (0x80 <= byte[0] <= 0xBF):
					break
		buffer = f.read(chunk_size)
		while True:
			byte = f.read(1)
			if not byte:
				break
			if 0x80 <= byte[0] <= 0xBF:
				buffer += byte
			else:
				f.seek(-1, 1)
				break
		actual_byte_count = len(buffer)
		text = buffer.decode("utf-8", errors="replace")
		return text, actual_byte_count

def return_number_of_tokens(lines, num_workers, tokenizer):
	with concurrent.futures.ProcessPoolExecutor(
		max_workers=num_workers,
		initializer=init_worker,
		initargs=(tokenizer, tokenizer.cfg)
	) as executor:
		return sum(list(executor.map(tokenize_count, lines, chunksize=(len(lines) + num_workers - 1) // num_workers)))



# =================================================================================================================
# Main Tokenize Function
# =================================================================================================================
def tokenize(
	index_path: str,
	input_file: str,
	tokenizer: Tokenizer,
	num_workers: int,
	buffer_size: int,
	max_vocab: int,
) -> None:
	
	MAX_VOCAB = max_vocab
	chunk = 1_000_000
	num_chunk = (os.path.getsize(input_file) + chunk - 1) // chunk
	num_retries = 0

	while True:
		# =============================================================================================================
		# 1. count lines & estimate #tokens
		# =============================================================================================================
		# 1-0. make temporary files
		LINES_SIZE = 4_096
		index_path_ = Path(index_path)
		index_path_.mkdir(parents=True, exist_ok=True)
		if True:
			bin_path1 = index_path_ / "tmp1.bin"
			make_file(bin_path1, LINES_SIZE * 8)
			lines_tkn = np.memmap(bin_path1, dtype=np.uint64, mode='w+', shape=(LINES_SIZE,))
			bin_path2 = index_path_ / "tmp2.bin"
			make_file(bin_path2, LINES_SIZE * 8)
			lines_byt = np.memmap(bin_path2, dtype=np.uint64, mode='w+', shape=(LINES_SIZE,))

		# 1-1. count number of lines
		bar1 = get_custom_tqdm(num_chunk)
		num_lines = 0
		with open(input_file, mode="rb") as f:
			while True:
				pos = f.tell()
				bar1.update(pos // chunk - bar1.n)

				# [a] update
				lines_byt[num_lines] = pos
				line = f.readline()
				if not line:
					break
				num_lines += 1

				# [b] change filesize
				if num_lines + 1024 > LINES_SIZE:
					lines_tkn.flush()
					lines_byt.flush()
					gc.collect()
					del lines_byt
					del lines_tkn
					LINES_SIZE *= 2
					with open(bin_path1, "r+b") as f_resize:
						f_resize.truncate(LINES_SIZE * 8)
					with open(bin_path2, "r+b") as f_resize:
						f_resize.truncate(LINES_SIZE * 8)
					lines_tkn = np.memmap(bin_path1, dtype=np.uint64, mode='r+', shape=(LINES_SIZE,))
					lines_byt = np.memmap(bin_path2, dtype=np.uint64, mode='r+', shape=(LINES_SIZE,))
		bar1.update(bar1.total - bar1.n)
		bar1.close()

		# 1-2. estimate number of tokens (small)
		logger.info(f"Estimating number of tokens... (<3 min)")
		total_bytes = os.path.getsize(input_file)
		sub_bytes = 0
		sub_token = 0
		if os.path.getsize(input_file) < 400_000_000:
			text, sub_bytes = read_random_chunk_safe(input_file, 0, os.path.getsize(input_file))
			lines = text.splitlines()
			sub_token = return_number_of_tokens(lines, num_workers, tokenizer)
			est_tokens = sub_token + 1024
		
		# 1-3. estimate number of tokens (large)
		else:
			sub_chunk_est = 10_000
			lines = []
			for i in range(40_000):
				pos = random.randint(0, os.path.getsize(input_file) - sub_chunk_est)
				v, bytes = read_random_chunk_safe(input_file, pos, sub_chunk_est)
				sub_bytes += bytes
				for u in v.splitlines():
					lines.append(u)
			sub_token = return_number_of_tokens(lines, num_workers, tokenizer)
			safe_ratio = 1.003 + 0.03 * ((2 ** num_retries) - 1)
			est_tokens = int(safe_ratio * total_bytes * sub_token / sub_bytes)


		# =============================================================================================================
		# 2. make a file
		# =============================================================================================================
		# 2-1. decide filesize
		TOKEN_SIZE = (est_tokens // 1024 + 2) * 1024
		
		# 2-2. make binary files
		if True:
			bin_path = index_path_ / "tokens.bin"
			make_file(bin_path, TOKEN_SIZE * 4)
			tokens = np.memmap(bin_path, dtype=np.uint32, mode='w+', shape=(TOKEN_SIZE,))
		if True:
			bin_path = index_path_ / "offset.bin"
			make_file(bin_path, TOKEN_SIZE + TOKEN_SIZE // 32)
			offsets = np.memmap(bin_path, dtype=np.uint8, mode='w+', shape=(TOKEN_SIZE + TOKEN_SIZE // 32,))
			byte_offset1  = offsets[0 : TOKEN_SIZE]
			byte_offset2  = offsets[TOKEN_SIZE : ].view(np.uint64)
		if True:
			bin_path = index_path_ / "metadata.bin"
			make_file(bin_path, 2048 * 8)
			initial = np.memmap(bin_path, dtype=np.uint64, mode='w+', shape=(2048,))
		
		# 2-3. make a temporary file
		if True:
			bin_path = index_path_ / "tmp3.bin"
			bytes_rec = np.memmap(bin_path, dtype=np.uint32, mode='w+', shape=(TOKEN_SIZE,))
		
		# 2-4. output number of lines & tokens
		logger.info(f"Tokenize pre-phase finished")
		logger.info(f"#Lines     : {num_lines:,}")
		logger.info(f"#Tokens    : {est_tokens:,} est.")

		# 2-5. estimate the maximum vocabulary
		max_vocab_in_list = 0
		for i in tokenizer.tokens:
			max_vocab_in_list = max(max_vocab_in_list, i)
		if len(tokenizer.tokens) <= MAX_VOCAB:
			logger.info(f"#Vocabulary: {max_vocab_in_list + 1:,}")
		else:
			logger.info(f"#Vocabulary: {max_vocab_in_list + 1:,} \x1b[31m(Capped to {MAX_VOCAB:,})\x1b[0m")


		# =============================================================================================================
		# 3. tokenize all
		# =============================================================================================================
		logger.info(f"Tokenize begins...")
		ctokens = 0
		clines = 0
		ct = 0
		failed = False
		with concurrent.futures.ProcessPoolExecutor(
			max_workers=num_workers,
			initializer=init_worker,
			initargs=(tokenizer, tokenizer.cfg)
		) as executor:
			with stopwatch.timers["tokenize"]:
				for buffer in buffer_lines(input_file, buffer_size*num_workers, num_chunk, chunk):
					results = list(executor.map(tokenize_encode_offsets, buffer, chunksize=buffer_size))
					token_sequences, offsets_sequences = zip(*results)
					init_ctokens = ctokens

					# copy the token sequence to tokens[]
					for seq in token_sequences:
						length = seq.shape[0]
						if ctokens + length > TOKEN_SIZE:
							failed = True
							break
						tokens[ctokens : ctokens+length] = seq
						ctokens += length
					if failed == True:
						break
					
					# copy the byte offsets to byte_offsets1[]
					for seq in offsets_sequences:
						length = seq.shape[0]
						bytes_rec[ct : ct+length] = seq
						ct += length

					# copy the line numbers
					a = np.fromiter((len(seq) for seq in token_sequences), dtype=np.int64)
					s = np.empty(len(a) + 1, dtype=np.int64)
					s[0] = 0
					s[1:] = np.cumsum(a)
					lines_tkn[clines : clines + len(token_sequences)] = init_ctokens + s[0 : len(token_sequences)]
					clines += len(token_sequences)
		if failed == True:
			logger.info(f"Failed to estimate the number of tokens. repeating...")
			num_retries += 1
			continue
		num_tokens = ctokens
		lines_tkn[clines] = num_tokens
		logger.info(f"#Tokens    : {ctokens:,}")
		break


	# =================================================================================================================
	# 4. get other informations
	# =================================================================================================================
	# 4-1. functions to fill remaining tokens
	@nb.njit(cache=True, parallel=True)
	def fill_max(token, fst, lst, MAX):
		chunk = (lst - fst + num_workers - 1) // num_workers
		for worker_id in nb.prange(num_workers):
			stt = min(lst, fst + chunk * (worker_id + 0))
			end = min(lst, fst + chunk * (worker_id + 1))
			for i in range(stt, end):
				token[i] = MAX
	
	# 4-2. function to cap tokens
	@nb.njit(cache=True, parallel=True)
	def cap_tokens(token, num_tokens, MAX):
		chunk = (num_tokens + num_workers - 1) // num_workers
		for worker_id in nb.prange(num_workers):
			stt = min(num_tokens, chunk * (worker_id + 0))
			end = min(num_tokens, chunk * (worker_id + 1))
			for i in range(stt, end):
				token[i] = min(token[i], MAX)
	
	# 4-2. function to fill byte offsets
	@nb.njit(cache=True, parallel=True)
	def fill_all(num_tokens, rec_i32, byte_offset1, byte_offset2, lines_tkn, lines_byt, black_cnt, black_list):
		chunk = (num_tokens + num_workers - 1) // num_workers
		chunk = ((chunk + 255) // 256) * 256
		for worker_id in nb.prange(num_workers):
			stt = min(num_tokens, chunk * (worker_id + 0))
			end = min(num_tokens, chunk * (worker_id + 1))
			ok = 0
			ng = num_lines
			while ng - ok > 1:
				mid = (ok + ng) // 2
				if lines_tkn[mid] <= stt:
					ok = mid
				else:
					ng = mid
			current_line = ok
			while current_line < num_lines and lines_tkn[current_line + 1] < stt:
				current_line += 1

			# Process by 256 bytes
			for i in range(stt, end, 256):
				byte_offset2[i >> 8] = lines_byt[current_line] + rec_i32[i]
				prv = 0
				for j in range(i, min(num_tokens, i + 256)):
					cur = lines_byt[current_line] + rec_i32[j]
					if j != i:
						byte_offset1[j] = min(255, cur - prv)
						if cur - prv >= 255 and black_cnt[worker_id] < len(black_list[worker_id]):
							black_list[worker_id][2 * black_cnt[worker_id] + 0] = j - 1
							black_list[worker_id][2 * black_cnt[worker_id] + 1] = cur - prv
							black_cnt[worker_id] += 1
						prv = cur
					else:
						prv = cur
					while current_line < num_lines and lines_tkn[current_line + 1] <= j + 1:
						current_line += 1
	
	# 4-3. registration
	with stopwatch.timers["register"]:
		black_cnt  = np.zeros((num_workers), dtype=np.uint64)
		black_list = np.zeros((num_workers, (num_tokens // (num_workers * 10_000)) + 1_000), dtype=np.uint64)
		logger.info(f"Tokenize final processing begins..")
		logger.info(f"<this may take 5-10% of tokenize time>")
		fill_max(tokens, num_tokens, TOKEN_SIZE, MAX_VOCAB - 1)
		cap_tokens(tokens, num_tokens, MAX_VOCAB - 1)
		logger.info(f"Tokenize final processing 1/2 finished")
		fill_all(num_tokens, bytes_rec, byte_offset1, byte_offset2, lines_tkn, lines_byt, black_cnt, black_list)
		logger.info(f"Tokenize final processing 2/2 finished")
		logger.info(f"=====================================================")

		# black list (word length >= 256)
		black_sum = 0
		for i in range(num_workers):
			black_sum += black_cnt[i]
		byte_offset1.flush()
		byte_offset2.flush()
		offsets.flush()
		offsets._mmap.close()
		del offsets
		del byte_offset1
		del byte_offset2
		bin_path = index_path_ / "offset.bin"
		with open(bin_path, "ab") as f:
			f.truncate((TOKEN_SIZE + TOKEN_SIZE // 32) + black_sum * 16)
		offsets = np.memmap(bin_path, dtype=np.uint8, mode='r+', shape=(TOKEN_SIZE + TOKEN_SIZE // 32 + black_sum * 16,))
		byte_offset3  = offsets[TOKEN_SIZE + TOKEN_SIZE // 32 : ].view(np.uint64)
		cnts = 0
		for i in range(num_workers):
			for j in range(black_cnt[i]):
				byte_offset3[0 * black_sum + cnts] = black_list[i][2 * j + 0]
				byte_offset3[1 * black_sum + cnts] = black_list[i][2 * j + 1]
				cnts += 1
		logger.info(f"Exceptions = {cnts:,}")
	

	# =================================================================================================================
	# 5. final
	# =================================================================================================================
	# 5-1. record final data
	initial[ 0] = num_tokens
	initial[ 1] = num_lines
	initial[ 4] = TOKEN_SIZE
	initial[ 5] = LINES_SIZE
	initial[ 6] = MAX_VOCAB
	initial[ 7] = os.path.getsize(index_path_ / "metadata.bin")
	for i in range(len(input_file)):
		initial[512+i] = ord(input_file[i])
	initial.flush()
	offsets.flush()
	tokens.flush()
	del initial
	del offsets
	del tokens

	# 5-2. delete temporary file
	if os.path.exists(index_path_ / "tmp1.bin"):
		os.remove(index_path_ / "tmp1.bin")
	if os.path.exists(index_path_ / "tmp2.bin"):
		os.remove(index_path_ / "tmp2.bin")
	if os.path.exists(index_path_ / "tmp3.bin"):
		os.remove(index_path_ / "tmp3.bin")

	# 5-3. return number of tokens
	return num_tokens