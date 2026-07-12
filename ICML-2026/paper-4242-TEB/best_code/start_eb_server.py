import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_USE_PD_SCHEDULER"] = "1"
os.environ["VLLM_PD_K_MODE"] = "ifr"
os.environ["VLLM_PD_CALIBRATION_FILE"] = "/repo/reproduce/outputs/pd_calibration_Qwen3-8B.json"
os.environ["VLLM_COLLECT_SCHEDULE_STATS"] = "1"

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
parser = make_arg_parser(parser)
args = parser.parse_args([
    "--model", "/models/Qwen3-8B",
    "--port", "11015",
    "--max-num-seqs", "1536",
    "--max-num-batched-tokens", "14336",
    "--dtype", "bfloat16",
    "--gpu-memory-utilization", "0.9",
    "--enforce-eager",
])
validate_parsed_serve_args(args)
uvloop.run(run_server(args))
