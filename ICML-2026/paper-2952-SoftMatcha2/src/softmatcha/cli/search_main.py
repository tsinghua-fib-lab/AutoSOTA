#!/usr/bin/env python3
import logging
import sys
import math
import time
from pathlib import Path
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Sequence
from simple_parsing import field
from softmatcha import configs, stopwatch
from softmatcha.embeddings import Embedding, get_embedding
from softmatcha.search import Searcher
from softmatcha.struct import Pattern
from softmatcha.tokenizers import Tokenizer, get_tokenizer


# =====================================================================================================================
# 1. arguments
# =====================================================================================================================
@dataclass
class SearcherArguments:
	pattern: str = field(positional=True)   # pattern string
	index_path: str = field()               # path to "index files"
	num_candidates: int = 20                # number of candidates
	max_runtime: float = 10.0               # maximum runtime to search
	min_similarity: float = 0.45            # minimum similarity to search
	mmap : bool = False                     # load index on disk

def get_argparser(args: Sequence[str] | None = None) -> configs.ArgumentParser:
	parser = configs.get_argparser(args=args)
	parser.add_arguments(SearcherArguments, "searcher")
	return parser

def format_argparser() -> configs.ArgumentParser:
	parser = get_argparser()
	parser.preprocess_parser()
	return parser


# =====================================================================================================================
# 2. CLI output
# =====================================================================================================================
def output_cli(matched_pattern, match_score, match_count, dicts):
	pat_str_list = []
	pat_str_maxlen = 6
	max_match = 0
	if len(match_count) >= 1:
		max_match = max(match_count)
	max_len = max(6, len(format(max_match, ',')))

	# convert tokens to string
	for i in range(len(matched_pattern)):
		pat_str = ""
		for j in range(len(matched_pattern[i])):
			if matched_pattern[i][j] >= 1_000_000_000:
				break
			if j >= 1:
				pat_str += " "
			pat_str += dicts[matched_pattern[i][j]]
		pat_str_list.append(pat_str)
		pat_str_maxlen = max(pat_str_maxlen, len(pat_str))
	
	# print first two lines
	first_str = ""
	first_str += "| Rank "
	first_str += "| Score "
	first_str += "| " + (" " * ((max_len - 5) // 2)) + "#Match" + (" " * ((max_len - 6) // 2)) + " "
	first_str += "| String"
	second_str = ("-" * (20 + max_len + pat_str_maxlen))
	print(first_str)
	print(second_str)

	# print third and later lines
	for i in range(len(matched_pattern)):
		print_str = ""
		print_str += "| "
		print_str += str(i + 1).rjust(4) + " | "
		print_str += str(round(match_score[i] * 100.0, 1)).rjust(5) + " | "
		print_str += format(match_count[i], ',').rjust(max_len) + " | "
		print_str += pat_str_list[i]
		print(print_str)



# =====================================================================================================================
# 3. main function
# =====================================================================================================================
def main(args: Namespace) -> None:
	stopwatch.timers.reset()
	
	# 3-1. configure output
	logging.basicConfig(
		format="| %(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		level="INFO",
		force=True,
		stream=sys.stderr,
	)
	logger = logging.getLogger("search")

	# 3-2. load embedding & tokenizer
	embedding_class = get_embedding(args.common.backend)
	embedding: Embedding = embedding_class.build(
		embedding_class.Config(args.common.model, mmap=args.searcher.mmap)
	)
	tokenizer_class = get_tokenizer(args.common.backend)
	tokenizer: Tokenizer = tokenizer_class.build(
		tokenizer_class.Config(
			args.common.model,
			**{
				k: v
				for k, v in asdict(args.tokenizer).items()
				if k != "name_or_path"
			},
		)
	)
	
	# 3-3. load searcher
	searcher = Searcher(args.searcher.index_path, tokenizer, embedding)

	# 3-4. answer query
	def _query(pattern_str: str) -> int:
		pattern_tokens = tokenizer(pattern_str)
		pattern_embeddings = searcher.normalize(embedding(pattern_tokens))
		pattern = Pattern.build(pattern_tokens, pattern_embeddings, [0.0] * len(pattern_embeddings),)

		if len(pattern) > 12:
			print("\x1b[31mError: The number of tokens should be 12 or less (Current: " + str(len(pattern)) + ")\x1b[0m")
		else:
			unknown = False
			c = 0
			for i in pattern.tokens:
				if i >= min(searcher.max_vocab, len(embedding.embeddings)) - 1:
					unknown = True
					list_words = tokenizer.tokenize(pattern_str)
					print("\x1b[31mWord \"" + list_words[c] + "\" is unknown, so no match found\x1b[0m")
					break
				c += 1
			if unknown == False:
				num = 0
				matched_pattern, match_score, match_count, thres = searcher.search(
					pattern,
					args.searcher.num_candidates,
					args.searcher.min_similarity,
					args.searcher.max_runtime,
				)
				num = len(matched_pattern)
				if True:
					output_cli(matched_pattern, match_score, match_count, tokenizer.tokens)
					if num < args.searcher.num_candidates:
						print("\x1b[34m[Output " + str(num) + " hits with similarity >=" + str(int(100.0 * thres + 0.5)) + "%]\x1b[0m")
					else:
						print("\x1b[34m[Output " + str(num) + " hits]\x1b[0m")
				return num

	# 3-5. actual execution
	start = time.perf_counter()
	_query(args.searcher.pattern)
	print("\x1b[34m[Search Time: " + str(time.perf_counter() - start) + " sec]\x1b[0m")
		


# =====================================================================================================================
# 4. functions which are called first
# =====================================================================================================================
def cli_main() -> None:
	args = get_argparser().parse_args()
	main(args)

if __name__ == "__main__":
	print("Query begins")
	cli_main()