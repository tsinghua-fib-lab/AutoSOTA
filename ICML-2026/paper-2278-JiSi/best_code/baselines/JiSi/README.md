# JiSi Runner

This package contains the JiSi routing and aggregation implementation.

## Main Entry

```bash
python -m baselines.JiSi.run_jisi \
  --config baselines/JiSi/config/jisi/main.example.json
```

For local experiments, copy the example config and edit the copy:

```bash
cp baselines/JiSi/config/jisi/main.example.json baselines/JiSi/config/jisi/main.local.json
cp baselines/JiSi/config/jisi/api_config.example.json baselines/JiSi/config/jisi/api_config.local.json
```

## Modes

| Mode | Behavior |
| --- | --- |
| `router` | Route each query to the strongest selected model and evaluate from cached model records |
| `aggregator` | Select references and call an aggregator model to produce new answers |

## Config Files

| File | Purpose |
| --- | --- |
| `config/jisi/main.example.json` | JiSi runtime template |
| `config/jisi/api_config.example.json` | OpenAI-compatible model endpoint template |

The runner validates data paths at startup. Make sure `data/jisi/.../train.jsonl`, `test.jsonl`, and `baseline_scores.json` exist before launching a run.

## Post Evaluation

Aggregation writes `result.jsonl`. Score it with:

```bash
python -m baselines.JiSi.post_eval \
  --res_path results/jisi/<run_name>/result.jsonl \
  --datasets paper
```

SWE-Bench is verified separately because it requires patch submission rather than normal answer extraction:

```bash
python -m baselines.JiSi.test_swe \
  --res_path results/jisi/<run_name>/result.jsonl \
  --index-map path/to/swe_imap.json \
  --model-name jisi_run_name \
  --run-id jisi_swe_verified \
  --submit
```
