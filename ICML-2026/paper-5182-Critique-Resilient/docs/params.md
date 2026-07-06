# run_full_pipeline parameters

This documents the arguments `run_full_pipeline.py` passes to each step. Defaults
shown are the `run_full_pipeline.py` defaults when not overridden on the CLI.
Optional flags are only included in the subcommand when the corresponding value
is set (non-None or True).

## Pipeline settings (not passed to subcommands)
- `--timeout-hours`: `3.0`
- `--run-parallel`: `False`

## Generate benchmark questions (`generate_benchmark.py`)
- `--runs-file`: `configs/runs.json`
- `--topic-info-file`: `configs/topic_info.json`
- `--config`: `configs/models.json`
- `--output-dir`: `benchmarks`
- `--log-level`: `INFO`
- `--limit`: `None` (omitted when None)
- `--disable-batch`: `False` (omitted when False)
- `--max-rounds`: `None` (omitted when None)
- `--max-question-attempts`: `None` (omitted when None)
- `--models`: `None` (omitted when None; from `--benchmark-models`)

## Generate answers (`generate_answers.py`)
- `--config`: `configs/models.json`
- `--benchmark-dir`: `benchmarks`
- `--output-dir`: `answers`
- `--log-level`: `INFO`
- `--limit`: `None` (omitted when None)
- `--disable-batch`: `False` (omitted when False)
- `--max-rounds`: `None` (omitted when None)
- `--allow-self-answering`: `False` (omitted when False)
- `--rerun-failures`: `False` (omitted when False; from `--rerun-answer-failures`)
- `--models`: `None` (omitted when None; from `--answer-models`)

## Generate critiques (`generate_critiques.py`)
Runs once per mode in `--critique-modes` (default: `["contradictor", "evaluator"]`).
- `--mode`: `<mode>` (each entry from `--critique-modes`)
- `--config`: `configs/models.json`
- `--benchmark-dir`: `benchmarks`
- `--answers-dir`: `answers`
- `--output-dir`: `critiques`
- `--log-level`: `INFO`
- `--limit`: `None` (omitted when None)
- `--disable-batch`: `False` (omitted when False)
- `--max-rounds`: `None` (omitted when None)
- `--self-improve`: `False` (omitted when False; from `--self-improve-critiques`)
- `--models`: `None` (omitted when None; from `--critique-models`)

## Debate ill-posed claims (`debate.py`)
- `--mode`: `ill-posed`
- `--config`: `configs/models.json`
- `--benchmark-dir`: `benchmarks`
- `--answers-dir`: `answers`
- `--critiques-dir`: `critiques`
- `--output-dir`: `debates`
- `--rounds`: `5` (from `--debate-rounds`)
- `--log-level`: `INFO`
- `--limit`: `None` (omitted when None)
- `--no-allow-concede`: `False` (omitted when False)

## Debate critiques (`debate.py`)
- `--mode`: `critique`
- `--config`: `configs/models.json`
- `--benchmark-dir`: `benchmarks`
- `--answers-dir`: `answers`
- `--critiques-dir`: `critiques`
- `--output-dir`: `debates`
- `--rounds`: `5` (from `--debate-rounds`)
- `--log-level`: `INFO`
- `--limit`: `None` (omitted when None)
- `--no-allow-concede`: `False` (omitted when False)

## Automated judging (`automated_judge.py`)
- `--mode`: `all`
- `--config`: `configs/models.json`
- `--benchmark-dir`: `benchmarks`
- `--answers-dir`: `answers`
- `--critiques-dir`: `critiques`
- `--debates-dir`: `debates`
- `--output-dir`: `automated_evaluations`
- `--log-level`: `INFO`
- `--limit`: `None` (omitted when None)
- `--disable-batch`: `False` (omitted when False)
- `--batch-size`: `None` (omitted when None)
- `--overwrite`: `False` (omitted when False; from `--overwrite-judgments`)
- `--allow-no-debate`: `False` (omitted when False)
- `--force-correct-critiques`: `False` (omitted when False)
- `--models`: `None` (omitted when None; from `--judge-models`)
