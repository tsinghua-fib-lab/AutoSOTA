import sys, os, asyncio
sys.argv = [
    'vllm', 'bench', 'serve',
    '--model', '/models/Qwen3-8B',
    '--base-url', 'http://localhost:11014',
    '--dataset-name', 'custom',
    '--dataset-path', 'reproduce/outputs/sharegpt_prompts.jsonl',
    '--custom-output-len', '-1',
    '--num-prompts', '4000',
    '--num-warmups', '20',
    '--request-rate', 'inf',
    '--max-concurrency', '2048',
    '--save-result',
    '--save-detailed',
    '--result-dir', 'reproduce/outputs/bench_v1_sharegpt',
    '--result-filename', 'bench_v1.json',
    '--ignore-eos',
]

# Parse args like the CLI does
from vllm.benchmarks.serve import add_cli_args
from vllm.utils.argparse_utils import FlexibleArgumentParser
parser = FlexibleArgumentParser()
parser = add_cli_args(parser)
args = parser.parse_args(sys.argv[3:])  # skip vllm bench serve

print("Parsed args:", args)
print("Running benchmark...")

from vllm.benchmarks.serve import main
result = main(args)
print("Result:", result)
