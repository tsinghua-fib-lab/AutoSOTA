import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
parser = make_arg_parser(parser)
args = parser.parse_args([
    "--model", "/models/Qwen3-8B",
    "--port", "11014",
    "--max-num-seqs", "1536",
    "--max-num-batched-tokens", "16384",
    "--dtype", "bfloat16",
    "--gpu-memory-utilization", "0.9",
    "--enforce-eager",
])
validate_parsed_serve_args(args)
uvloop.run(run_server(args))
