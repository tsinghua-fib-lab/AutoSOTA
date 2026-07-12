import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("EB_GPU_DEVICE", "1")
os.environ["VLLM_USE_PD_SCHEDULER"] = "1"

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

B = int(os.environ.get("EB_MAX_NUM_BATCHED_TOKENS", "14336"))
N = int(os.environ.get("EB_MAX_NUM_SEQS", "1536"))
PORT = int(os.environ.get("EB_PORT", "11015"))

parser = FlexibleArgumentParser()
parser = make_arg_parser(parser)
args = parser.parse_args([
    "--model", "/models/Qwen3-8B",
    "--port", str(PORT),
    "--max-num-seqs", str(N),
    "--max-num-batched-tokens", str(B),
    "--dtype", "bfloat16",
    "--gpu-memory-utilization", os.environ.get("EB_GPU_MEM_UTIL", "0.9"),
    "--enforce-eager",
])
validate_parsed_serve_args(args)
uvloop.run(run_server(args))
