# Text-Generation Performance

Code for generation samples, BLEU/ROUGE summaries, and Elo judging.

## Install

From the repository root:

```bash
pip install -r experiments/text_generation_performance/requirements.txt
```

`generate_samples.py` imports the repository-root model modules (`prototype_attn.py`,
`llama_baseline.py`, `mamba.py`, and `deltanet.py`), so run the commands from the
repository root.

## Make FineWeb Prompts

```bash
python experiments/text_generation_performance/make_fineweb_prompts.py \
  --npz /share/datasets/prototype_data/val.npz \
  --tokenizer tok/fineweb_bpe_16000.json \
  --num-prompts 100 \
  --output generation_prompts.json
```

## Generate Samples

```bash
PYTHONPATH="$PWD" python experiments/text_generation_performance/generate_samples.py \
  --models /path/to/model_a /path/to/model_b \
  --names ModelA ModelB \
  --tokenizer tok/fineweb_bpe_16000.json \
  --dataset generation_prompts.json \
  --num_samples 100 \
  --output generation_samples.json
```

The output JSON contains generated text and per-sample BLEU/ROUGE-L.

## Elo Judge

```bash
python experiments/text_generation_performance/judge_elo.py \
  --input_files generation_samples.json \
  --output_file elo_results.json \
  --model_id google/gemma-3-4b-it \
  --seed 0
```

The output JSON contains Elo scores and pairwise comparison records.
