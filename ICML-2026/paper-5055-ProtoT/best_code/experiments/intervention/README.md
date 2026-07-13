# Prototype Intervention Experiments

This repository contains the public code needed to reproduce the prototype intervention tables (Table.5, 7-10) produced by `withppl_all.py`.

## What the experiment does

`withppl_all.py` evaluates targeted prototype interventions on a trained `ProtoBroadcastLM` model. For each concept, it measures:

- baseline target-token probability from the prefix context;
- baseline sentence perplexity;
- target-token probability and sentence perplexity after write-mask, read-mask, and random re-initialization interventions;
- relative changes in probability and perplexity.

The script writes per-sentence JSON records and Markdown/CSV tables to `withppl_results/`.

## Minimal code files

The experiment should be run from the repository root. The intervention folder only keeps the experiment runner:

```text
experiments/intervention/withppl_all.py
```

`withppl_all.py` imports `ProtoBroadcastLM` from the repository-root `prototype_attn.py`, which imports shared helpers from the repository-root `utils.py`.

## Python dependencies

Install the minimal Python dependencies with:

```bash
pip install -r requirements_intervention.txt
```

The script was developed with PyTorch and `tokenizers`. `transformers` is needed because it is imported by `utils.py`.

## Required checkpoint layout

The script expects three checkpoint directories under `experiments/intervention/`:

```text
ProtoAttn_FineWeb_Large/
trial000/seed_124/
trial000/seed_325/
```

Each directory must contain:

```text
model_state_dict.pth
model_config.json
fineweb_bpe_16000.json
```

The public branch includes checkpoint configuration and tokenizer files. It does not include `model_state_dict.pth` weights by default.

## Run

From the repository root:

```bash
python experiments/intervention/withppl_all.py
```

## Outputs

The script creates `withppl_results/` and writes:

- one JSON file per experiment;
- one Markdown table per experiment;
- one CSV table per experiment.

Existing result files with the same names are appended to rather than overwritten. Remove old JSON files first if you need a clean run.
