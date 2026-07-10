#!/usr/bin/env python3
import logging
import os
import sys
import numpy as np
import simple_parsing
from pathlib import Path
from typing import Sequence
from argparse import Namespace
from simple_parsing import field
from dataclasses import asdict, dataclass
from softmatcha import configs
from softmatcha import stopwatch
from softmatcha.index import tokenize
from softmatcha.index import build_index
from softmatcha.tokenizers import get_tokenizer
from softmatcha.embeddings import get_embedding
simple_parsing.parsing.logger.setLevel(logging.ERROR)
simple_parsing.wrappers.dataclass_wrapper.logger.setLevel(logging.ERROR)


# =====================================================================================================================
# 1. basic configuration
# =====================================================================================================================
# 1-1. config logs
logging.basicConfig(
	format="| %(asctime)s | %(levelname)s | %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level="INFO",
	stream=sys.stderr,
)
logger = logging.getLogger("softmatcha.cli.build_inverted_index")

# 1-2. config args
@dataclass
class IndexerArguments:
	inputs: list[str] = field(positional=True, nargs="+")
	index      : str = field(alias=["--index"])
	buffer_size: int = field(default = 2_500)   # buffer size.
	mem_size   : int = field(default = 500)     # used memory size (MB). Higher is faster but consumes memory.
	mem_size_ex: int = field(default = 100)     # used memory size (MB) when searching.
	max_vocab  : int = field(default = 2 ** 19) # maximum vocabulary size.

# 1-3. parse args
def get_argparser(args: Sequence[str] | None = None) -> configs.ArgumentParser:
	parser = configs.get_argparser(args=args)
	parser.add_arguments(IndexerArguments, "indexer")
	return parser
def format_argparser() -> configs.ArgumentParser:
	parser = get_argparser()
	parser.preprocess_parser()
	return parser



def main(args: Namespace) -> None:
	# =================================================================================================================
	# 2. main part (tokenize)
	# =================================================================================================================
	stopwatch.timers.reset(profile=True)
	os.environ["RUST_LOG"] = "info"
	workers = os.cpu_count()

	# 2-1. load tokenizers
	input_paths = [os.path.abspath(input_path) for input_path in args.indexer.inputs]
	tokenizer_class = get_tokenizer(args.common.backend)
	tokenizer = tokenizer_class.build(
		tokenizer_class.Config(
			name_or_path=args.common.model,
			**{k: v for k, v in asdict(args.tokenizer).items() if k != "name_or_path"},
		)
	)

	# 2-2. load embeddings
	embedding_class = get_embedding(args.common.backend)
	embedding = embedding_class.build(
		embedding_class.Config(args.common.model, mmap=True)
	)

	# 2-3. tokenize
	logger.info(f"Tokenize pre-phase begins..")
	logger.info(f"\x1b[31mNote: {max(args.indexer.mem_size, args.indexer.mem_size_ex):,}MB memory is required\x1b[0m")
	num_tokens = tokenize(
		args.indexer.index,
		input_paths[0],
		tokenizer,
		num_workers = workers,
		buffer_size = args.indexer.buffer_size,
		max_vocab   = args.indexer.max_vocab,
	)


	# =================================================================================================================
	# 3. main part (index)
	# =================================================================================================================
	# 3-1. decide index size (pair/trio)
	dict_size = min(num_tokens * 2, 1_000_000 * args.indexer.mem_size_ex // 2) * 8
	pair_cons = min(100_000, max(32, int((dict_size // 4) ** (1.0 / 2.0))))
	pair_cons = (pair_cons // 32) * 32
	trio_cons = min( 10_000, max(32, int((max(1, dict_size - pair_cons * pair_cons) * 6) ** (1.0 / 3.0))))
	trio_cons = (trio_cons // 32) * 32

	# 3-2. decide index size (rough)
	avail_rough = 1_000_000 * args.indexer.mem_size_ex - (
		pair_cons * pair_cons + trio_cons * (trio_cons - 1) * (trio_cons - 2) // 6
	) // 8
	rough_size = max(avail_rough, 0) // 32
	rough_div = 65_536
	while rough_div > 128:
		rough_div //= 2
		if (num_tokens - rough_div + 1) // rough_div >= rough_size:
			rough_div *= 2
			break

	# 3-3. build
	build_index(
		args.indexer.index,
		embedding,
		write_thread = 4,
		chunk_size   = args.indexer.mem_size * (1_000_000 // 120),
		pair_cons    = pair_cons,
		trio_cons    = trio_cons,
		rough_div    = rough_div,
		num_shards   = 3
	)

	# 3-4. delete file
	if os.path.exists(Path(args.indexer.index) / "tokens.bin"):
		os.remove(Path(args.indexer.index) / "tokens.bin")

	# 3-5. output logs
	logger.info(f"Total time: {sum(stopwatch.timers.elapsed_time.values())}s")



# =====================================================================================================================
# 4. functions which are called first
# =====================================================================================================================
def cli_main() -> None:
	args = get_argparser().parse_args()
	main(args)


if __name__ == "__main__":
	cli_main()
