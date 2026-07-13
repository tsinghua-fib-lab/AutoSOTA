# Throughput Benchmarks

Code for context-length throughput measurements.

## Install

From the repository root:

```bash
pip install -r experiments/throughput_benchmarks/requirements.txt
```

`benchmark_throughput.py` imports the repository-root model modules
(`prototype_attn.py`, `llama_baseline.py`, `mamba.py`, and `deltanet.py`), so run
the commands from the repository root.

## Run

```bash
PYTHONPATH="$PWD" python experiments/throughput_benchmarks/benchmark_throughput.py \
  --models /path/to/model_a /path/to/model_b \
  --names ModelA ModelB \
  --tokenizer tok/fineweb_bpe_16000.json \
  --context_lengths 2048 4096 8192 16384 \
  --steps 50 \
  --output throughput_results.json
```

The output JSON contains model name, context length, and iterations per second.

## Aggregate

```bash
python experiments/throughput_benchmarks/benchmark_throughput.py \
  --aggregate throughput_results.json
```
