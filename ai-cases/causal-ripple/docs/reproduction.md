# Reproducing the selected configurations

## Generation environment

Follow the installation commands in the [README](../README.md). Generation uses Python 3.12, PyTorch 2.11.0 with CUDA 12.8, Transformers 4.46.3, BF16 weights, batch size one, temperature zero, and seed 1234. The exact package versions are in [requirements/runtime.txt](../requirements/runtime.txt).

The LLaDA-8B-Base checkpoint is pinned to `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`. You can pass a local snapshot path through `--model` and disable downloads with `--local-files-only`.

## Protocol and selected profiles

All main runs use 128 ordinary steps and one output block. Revision operates during the middle part of decoding, protects the last 16 output positions, and limits a joint update to one seed plus three affected positions. The seed is retained when the current token-transfer allowance is smaller than the proposed update.

| Profile | Candidate cap | Probe cap | Transfer truncation | Probe acceptance |
| :--- | ---: | ---: | :--- | :--- |
| `base` | 32 | 12 | Seed, then position order | Prediction flip or support drop ≥ 0.15 |
| `ds1000` | 16 | 12 | Seed, then influence rank | Same as base |
| `bigcodebench` | 32 | 48 | Seed, then influence rank | Base gate, changed token, confidence ≥ 0.30 |
| `ds1000-cascade` | 32 | 12 | Seed, then influence rank | Same as base |

The main DS-1000 result is **55/1,000** with `ds1000`. Its cascade experiment uses `ds1000-cascade`, **52 → 53/1,000**. BigCodeBench uses **73 → 74/1,140**. HumanEval, MBPP, and MBPP+ use `base` for both main and cascade runs.

CORE uses 32 candidates, one revised seed, and a verification interval of eight ordinary steps. Causal-Ripple uses an interval of 16 steps and paired probes. Both have a ceiling of 136 serial forward calls; paired probes can raise Causal-Ripple's sequence-forward count to 140. Confidence and top-k margin filling use 128 ordinary calls without verification. Every JSONL generation record includes both cost measures and processed-token count.

## Prepare the additional benchmarks

```bash
causal-ripple-prepare --benchmark mbppplus --output-dir data/mbppplus
causal-ripple-prepare --benchmark ds1000 --output-dir data/ds1000
causal-ripple-prepare --benchmark bigcodebench --output-dir data/bigcodebench
```

`--source PATH` accepts a local official JSONL or JSONL.GZ file. Data versions are fixed:

| Benchmark | Source version | Tasks |
| :--- | :--- | ---: |
| MBPP+ | EvalPlus `v0.2.0` | 378 |
| DS-1000 | Git commit `b39aab71da6d23ef8d3cac59a7c5f834516ab334` | 1,000 |
| BigCodeBench Complete | Dataset `v0.1.4`, revision `b74c0d0bf70d2c0bc459be537895cca163007f1a` | 1,140 |

DS-1000 preparation extracts the public setup, expected output-variable names, and library requirements into a completion prompt. MBPP+ uses its public prompt; BigCodeBench uses `complete_prompt`. Preparation writes `public.jsonl`, `problems.jsonl`, and a manifest. Generation reads `public.jsonl`; scoring reads `problems.jsonl` and the corresponding official evaluator.

## Generate full runs

```bash
CUDA_VISIBLE_DEVICES=0 causal-ripple --benchmark mbppplus \
  --prompts data/mbppplus/public.jsonl --output outputs/mbppplus/ripple.jsonl

CUDA_VISIBLE_DEVICES=1 causal-ripple --benchmark ds1000 \
  --prompts data/ds1000/public.jsonl --output outputs/ds1000/ripple.jsonl

CUDA_VISIBLE_DEVICES=2 causal-ripple --benchmark bigcodebench \
  --prompts data/bigcodebench/public.jsonl --output outputs/bigcodebench/ripple.jsonl
```

Each command is independent. On one GPU, run them sequentially. Output length defaults to 256 for DS-1000 and 512 for the other benchmarks. Inputs exceeding the 4,096-token context window remain in the full denominator and receive a `context_limit` record.

Repeat a command to resume a greedy run. A manifest checks the source files, checkpoint, configuration, and prompt hashes. Changing any of them requires a new output file. A completed `--limit` smoke run can be extended by removing the limit. Random filling uses a trajectory-wide RNG stream and requires a fresh output file on restart.

Run baselines by changing `--method` and the output name:

```bash
CUDA_VISIBLE_DEVICES=0 causal-ripple --benchmark bigcodebench \
  --prompts data/bigcodebench/public.jsonl \
  --method core --output outputs/bigcodebench/core.jsonl

CUDA_VISIBLE_DEVICES=0 causal-ripple --benchmark bigcodebench \
  --prompts data/bigcodebench/public.jsonl \
  --method topk --output outputs/bigcodebench/topk.jsonl

CUDA_VISIBLE_DEVICES=0 causal-ripple --benchmark bigcodebench \
  --prompts data/bigcodebench/public.jsonl \
  --method confidence --output outputs/bigcodebench/confidence.jsonl
```

Visible-Cascade is a complete inference method, including its input Causal-Ripple generation:

```bash
CUDA_VISIBLE_DEVICES=0 causal-ripple --benchmark bigcodebench \
  --prompts data/bigcodebench/public.jsonl \
  --method visible-cascade --output outputs/bigcodebench/cascade.jsonl
```

Replace the benchmark and input path for MBPP+ or DS-1000. DS-1000 automatically selects `ds1000-cascade`. Refinement uses public assertions when present, at most two suspicious code units, eight local rediffusion steps, and certificate-gated structural/name repair. DS-1000 uses the supplied-variable contract; its refinement has no executable visible-test gate.

## Official evaluation environments

Use separate Python 3.10 environments for benchmark library compatibility. The following commands start at the repository root. Generated programs are executed by the benchmark evaluators.

### DS-1000

```bash
python3.10 -m venv .venv-ds1000
.venv-ds1000/bin/pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
.venv-ds1000/bin/pip install -r requirements/ds1000.txt
.venv-ds1000/bin/pip install -e . --no-deps

git clone https://github.com/xlang-ai/DS-1000.git external/ds1000
git -C external/ds1000 checkout b39aab71da6d23ef8d3cac59a7c5f834516ab334

.venv-ds1000/bin/python -m causal_ripple.evaluate \
  --benchmark ds1000 --data-dir data/ds1000 \
  --samples outputs/ds1000/ripple.jsonl \
  --output outputs/ds1000/ripple.scores.jsonl \
  --evaluator-dir external/ds1000 --workers 2
```

Each DS-1000 task runs its official execution and optional string checks with a 120-second timeout.

### MBPP+

```bash
python3.10 -m venv .venv-evalplus
.venv-evalplus/bin/pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
.venv-evalplus/bin/pip install -r requirements/evalplus.txt
.venv-evalplus/bin/pip install -e . --no-deps

.venv-evalplus/bin/python -m causal_ripple.evaluate \
  --benchmark mbppplus --data-dir data/mbppplus \
  --samples outputs/mbppplus/ripple.jsonl \
  --output outputs/mbppplus/ripple.scores.jsonl --workers 2
```

EvalPlus is pinned to `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`. A task counts as successful only when both base and expanded tests pass. EvalPlus caches reference outputs on the first evaluation.

### BigCodeBench

```bash
python3.10 -m venv .venv-bigcodebench
.venv-bigcodebench/bin/pip install -r requirements/bigcodebench-eval.txt
.venv-bigcodebench/bin/pip install --no-deps -r requirements/bigcodebench.txt
.venv-bigcodebench/bin/pip install -e . --no-deps

.venv-bigcodebench/bin/python -m causal_ripple.evaluate \
  --benchmark bigcodebench --data-dir data/bigcodebench \
  --samples outputs/bigcodebench/ripple.jsonl \
  --output outputs/bigcodebench/ripple.scores.jsonl \
  --groundtruth results/bigcodebench_groundtruth_times.jsonl --workers 2
```

BigCodeBench is pinned to `09dd993f46c3fbf3a799465bb96d524edcb0b199`. `bigcodebench-eval.txt` records the installed evaluation environment, including the benchmark's library dependencies. Installing the evaluator with `--no-deps` keeps this environment independent of upstream model-serving dependencies. The scoring entry point imports no generation backend and requires no PyTorch.

Scoring applies the official sanitizer and the archived `code_prompt + pass + generated program` assembly. The included timing cache calibrates task timeouts; entries with unavailable reference timing use the original 20-second fallback. Resource arguments match the reported evaluation: 30 GiB memory, 10-second minimum time limit, and ground-truth time factor 1. All 1,140 tasks remain in the denominator.

## HumanEval and historical MBPP

These experiments use the original lm-evaluation-harness tasks: zero-shot HumanEval and three-shot MBPP with 500 evaluation tasks. MBPP+ is evaluated separately above. The integration preserves their task-specific stop strings and token cleanup.

```bash
CUDA_VISIBLE_DEVICES=0 python -m causal_ripple.lm_eval \
  --model causal_ripple \
  --model_args remasking=ripple,steps=128,gen_length=512,block_length=512 \
  --tasks humaneval --num_fewshot 0 --batch_size 1 \
  --seed 0,1234,1234,1234 --confirm_run_unsafe_code \
  --log_samples --output_path outputs/humaneval/ripple

CUDA_VISIBLE_DEVICES=1 python -m causal_ripple.lm_eval \
  --model causal_ripple \
  --model_args remasking=ripple,steps=128,gen_length=512,block_length=512 \
  --tasks mbpp --num_fewshot 3 --batch_size 1 \
  --seed 0,1234,1234,1234 --confirm_run_unsafe_code \
  --log_samples --output_path outputs/mbpp/ripple
```

Use `remasking=core`, `remasking=topk_margin`, or `remasking=low_confidence` for the baselines. Use `remasking=visible_cascade` for refinement. A local checkpoint can be supplied with `model_path=/path/to/snapshot` in `--model_args`.

## Output records

The JSONL runner saves the task ID, prompt hash, raw completion, cleaned program, generated token IDs, revision statistics, forward-call accounting, elapsed time, and peak allocated memory. Companion manifests fix run identity. Scoring requires exactly one completion for every prepared task, checks prompt hashes, resumes through a score manifest, and writes a final `.summary.json`.

For token-level analysis, pass `trace_callback` to `generate`. Each accepted revision records the decoding step, root position, selected positions, old and proposed token IDs, and influence scores. Tracing preserves sampling behavior.

## Release validation

The CPU suite compares 36 deterministic cases with the frozen samplers underlying the selected experiments. Twelve Ripple cases exercise active paired probes. Real A100 replays additionally check `Mbpp/2`, `DS1000/156`, `BigCodeBench/939`, and its Visible-Cascade refinement. The checks compare output token IDs, completion cleanup, and inference costs. See [release_validation.json](../results/release_validation.json).
