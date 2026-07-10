# Evaluation

This directory contains benchmark evaluators used by the JiSi data collector and post-evaluation script. Evaluators expose a common interface:

```python
from evaluation.factory import EvaluatorFactory

evaluator = EvaluatorFactory().get_evaluator("aime")
data = evaluator.load_data(split="test")
result = evaluator.evaluate(data[0], "The answer is 42.")
```

Each evaluator returns a dictionary containing fields such as `prediction`, `ground_truth`, and `is_correct`.

## JiSi Paper Set

The default post-evaluation set used by `baselines.JiSi.post_eval --datasets paper` is:

```text
aime, gpqa, livemathbench, mmlupro, livecodebench, hle, simpleqa, arenahard
```

Some datasets require external assets or an LLM grader. Put benchmark data under `data/` and configure grader endpoints through environment variables:

```bash
export GRADER_MODEL_NAME="gpt-4.1-mini"
export GRADER_BASE_URL="https://api.openai.com/v1"
export GRADER_API_KEY="your-key"
```

ArenaHard can use separate variables:

```bash
export ARENA_GRADER_MODEL_NAME="gpt-4.1-mini"
export ARENA_GRADER_BASE_URL="https://api.openai.com/v1"
export ARENA_GRADER_API_KEY="your-key"
```

## Add an Evaluator

1. Create `evaluation/<Dataset>/`.
2. Implement a class that inherits `BaseEvaluator`.
3. Register it in `evaluation/factory.py`.
4. Keep large dataset files under `data/` and out of git.
