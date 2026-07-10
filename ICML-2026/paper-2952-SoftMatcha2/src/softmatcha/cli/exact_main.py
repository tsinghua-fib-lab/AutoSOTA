#!/usr/bin/env python3
import logging
import sys
import math
import time
import numpy as np
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Sequence
from pathlib import Path
from simple_parsing import field
from softmatcha import configs
from softmatcha.embeddings import Embedding, get_embedding
from softmatcha.search import Searcher
from softmatcha.struct import Pattern
from softmatcha.tokenizers import Tokenizer, get_tokenizer
from softmatcha_rs import get_match_range_rs


# =====================================================================================================================
# 1. arguments
# =====================================================================================================================
@dataclass
class ExactArguments:
	pattern: str = field(positional=True)   # pattern string
	index_path: str = field()               # path to "index files"
	display: int = 10                       # how many strings should we display
	padding: int = 100                      # padding of displayed strings
	mmap : bool = True                      # load index on disk

def get_argparser(args: Sequence[str] | None = None) -> configs.ArgumentParser:
	parser = configs.get_argparser(args=args)
	parser.add_arguments(ExactArguments, "exact")
	return parser

def format_argparser() -> configs.ArgumentParser:
	parser = get_argparser()
	parser.preprocess_parser()
	return parser



# =====================================================================================================================
# 2. main function
# =====================================================================================================================
def main(args: Namespace) -> None:
	logger = logging.getLogger("exact")

	# 2-1. load embedding & tokenizer
	embedding_class = get_embedding(args.common.backend)
	embedding: Embedding = embedding_class.build(
		embedding_class.Config(args.common.model, mmap=args.exact.mmap)
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
	
	# 2-2. load searcher
	searcher = Searcher(args.exact.index_path, tokenizer, embedding)
	
	# 2-3. output function
	def print_highlighted_concat(str1: str, str2: str, str3: str) -> None:
		RED = "\x1b[31m"
		RESET = "\x1b[0m"
		result = f"{str1}{RED}{str2}{RESET}{str3}"
		print(result)

	# 2-4. answer query
	def _query(pattern_str: str):
		pattern_tokens = tokenizer(pattern_str)
		pattern_embeddings = embedding(pattern_tokens)
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
				logger.info(f"Pattern length: {len(pattern):,}")
				list_str, match_num = searcher.get_exact_match(pattern, args.exact.display, args.exact.padding)
				cnt = 0
				for v in list_str:
					cnt += 1
					print("[" + str(cnt) + "] ", end="")
					print_highlighted_concat(v[0], v[1], v[2])
					print("")
				if match_num <= args.exact.display:
					print("#Hits: " + format(match_num, ','))
				else:
					print("#Hits: " + format(match_num, ',') + " (only " + str(args.exact.display) + " are displayed)")

	# 2-5. execution
	_query(args.exact.pattern)
		


# =====================================================================================================================
# 3. functions which are called first
# =====================================================================================================================
def cli_main() -> None:
	args = get_argparser().parse_args()
	main(args)

if __name__ == "__main__":
	cli_main()