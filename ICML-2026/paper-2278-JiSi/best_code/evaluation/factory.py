"""Evaluator factory for the JiSi paper benchmark set."""

from enum import Enum
from typing import Any, Dict, Optional

from evaluation.AIME import AIMEEvaluator
from evaluation.ArenaHard import ArenaHardEvaluator
from evaluation.GPQA import GPQAEvaluator
from evaluation.HLE import HLEEvaluator
from evaluation.LiveCodeBench import LiveCodeBenchEvaluator
from evaluation.LiveMathBench import LiveMathBenchEvaluator
from evaluation.MMLUPro import MMLUProEvaluator
from evaluation.SimpleQA import SimpleQAEvaluator


class Benchmark(Enum):
    AIME = "aime"
    AIME2024 = "aime2024"
    AIME2025 = "aime2025"
    AIMETOTAL = "aime_total"
    GPQA = "gpqa"
    HLE = "hle"
    LIVECODEBENCH = "livecodebench"
    LIVEMATHBENCH = "livemathbench"
    MMLUPRO = "mmlupro"
    SIMPLEQA = "simpleqa"
    ARENAHARD = "arenahard"


class EvaluatorFactory:
    """Create evaluators for the benchmarks retained in this JiSi release."""

    def __init__(
        self,
        max_workers: int = 8,
        mode: str = "test",
        grader_cache_config: Optional[Dict[str, Any]] = None,
    ):
        self.max_workers = max_workers
        if mode not in {"test", "full"}:
            raise ValueError("mode must be either 'test' or 'full'")
        self.mode = mode
        self.grader_cache_config = grader_cache_config

    def get_evaluator(self, task: str | Benchmark):
        if isinstance(task, str):
            task = Benchmark(task)
        if not isinstance(task, Benchmark):
            raise TypeError(f"Invalid task type: {type(task)}, task: {task}")

        if task == Benchmark.AIME:
            return AIMEEvaluator(split="hybrid")
        if task == Benchmark.AIME2024:
            return AIMEEvaluator(split="2024")
        if task == Benchmark.AIME2025:
            return AIMEEvaluator(split="2025")
        if task == Benchmark.AIMETOTAL:
            return AIMEEvaluator(split="total")
        if task == Benchmark.GPQA:
            return GPQAEvaluator()
        if task == Benchmark.HLE:
            return HLEEvaluator(grader_cache_config=self.grader_cache_config)
        if task == Benchmark.LIVECODEBENCH:
            return LiveCodeBenchEvaluator(split="test")
        if task == Benchmark.LIVEMATHBENCH:
            return LiveMathBenchEvaluator()
        if task == Benchmark.MMLUPRO:
            return MMLUProEvaluator(split="test")
        if task == Benchmark.SIMPLEQA:
            return SimpleQAEvaluator(grader_cache_config=self.grader_cache_config)
        if task == Benchmark.ARENAHARD:
            return ArenaHardEvaluator(grader_cache_config=self.grader_cache_config)

        raise ValueError(f"Invalid task: {task}")
