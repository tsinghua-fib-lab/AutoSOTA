import time
import traceback
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import lm_eval.models.huggingface
import torch
from lm_eval import evaluator
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .evaluator import ModelEvaluator
from .utils import eval_mode

logger = getLogger(__name__)


def _cast_floats(value: Any) -> Any:
    """Recursively cast numpy/torch floats to native Python floats."""
    if isinstance(value, float):
        return float(value)
    elif isinstance(value, dict):
        return {k: _cast_floats(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_cast_floats(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(_cast_floats(v) for v in value)
    return value


@dataclass
class TaskConfig:
    """Configuration for an lm_eval task."""

    name: str | list[str]
    eval_kwargs: dict[str, Any] = field(default_factory=dict)
    instruct_mode: bool | str = "auto"


class LMEvalModelEvaluator(ModelEvaluator):
    """
    Evaluates a model on several zero/few-shot tasks using EleutherAI's Language Model Evaluation Harness (lm_eval).
    The returned dictionary has the following keys:
     - lm_eval/<task_name>/<metric_name>: All raw metrics for each task (e.g., acc,none, acc_stderr,none).
     - lm_eval/average/acc: The average accuracy over tasks that report acc,none.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        tasks: tuple[TaskConfig, ...] | list[TaskConfig],
        batch_size: int | None = "auto",
        prefix: str | None = "lm_eval",
        limit: int | None = None,
        hflm_kwargs: dict[str, Any] | None = None,
        log_samples: bool = False,
        max_retries: int = 3,
    ):
        """
        Args:
            tokenizer: Required to process samples of the datasets.
            tasks: List of TaskConfig objects specifying tasks to evaluate on.
                See https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks for a list
                of available tasks.
            batch_size: Batch size to use for evaluation.
            prefix: Prefix for the result keys.
            limit: Limit the number of samples per task. Useful for debugging.
            hflm_kwargs: Optional kwargs passed to lm_eval.models.huggingface.HFLM for additional arguments.
            log_samples: Whether to log samples for debugging.
            max_retries: Maximum number of retry attempts per task on failure (e.g., OOM).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.batch_size = batch_size
        self.prefix = prefix
        self.limit = limit
        self.hflm_kwargs = hflm_kwargs or {}
        self.log_samples = log_samples
        self.max_retries = max_retries

    def __call__(
        self,
        model: nn.Module | PreTrainedModel,
        prefix: str | None = "lm_eval",
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert isinstance(model, PreTrainedModel), (
            f"Expected PreTrainedModel, got {type(model).__name__}. If using EntQuantModel, pass entquant_model.model."
        )
        prefix = self.prefix if self.prefix is not None else prefix
        prefix = prefix + "/" if prefix is not None else ""
        results = {}
        with torch.no_grad(), eval_mode(model):
            task_names = [t.name if isinstance(t.name, str) else t.name for t in self.tasks]
            logger.info(f"Validating LM Eval on {task_names}...")
            lm_eval_result = evaluate_lm_eval(
                tasks=self.tasks,
                model=model,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                limit=self.limit,
                hflm_kwargs=self.hflm_kwargs,
                log_samples=self.log_samples,
                max_retries=self.max_retries,
            )
            logger.info(f"Validated LM Eval:\n{lm_eval_result}")
            for task_name, result in lm_eval_result.items():
                # Store all metrics with prefix
                for metric_name, value in result.items():
                    results[f"{prefix}{task_name}/{metric_name}"] = _cast_floats(value)
        return results


def evaluate_lm_eval(
    tasks: list[TaskConfig],
    model: PreTrainedModel,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast,
    batch_size: int | None = "auto",
    limit: int | None = None,
    hflm_kwargs: dict[str, Any] | None = None,
    log_samples: bool = False,
    max_retries: int = 3,
) -> dict[str, dict[str, float]]:
    """
    Script to evaluate a model on a list of tasks using lm_eval.

    Args:
        tasks: List of TaskConfig objects specifying tasks to evaluate on.
        model: An `PreTrainedModel` that supports causal language modeling.
        tokenizer: Suitable tokenizer for the model.
        batch_size: Batch size to use for evaluation.
        limit: Limit the number of samples per task. Useful for debugging.
        hflm_kwargs: Optional kwargs passed to lm_eval.models.huggingface.HFLM for additional arguments.
        log_samples: Whether to log samples for debugging.
        max_retries: Maximum number of retry attempts per task on failure (e.g., OOM).

    Returns: Nested dictionary of results in the format [<task_name>][<metric_name>], containing all raw metrics
        for each task. Also includes ["average"]["acc"] with the mean accuracy over tasks that report acc,none.
    """
    results = {}
    acc_tasks = []

    for task in tasks:
        _hflm_kwargs = dict(hflm_kwargs) if hflm_kwargs else {}
        if "batch_size" in task.eval_kwargs:
            _hflm_kwargs["batch_size"] = task.eval_kwargs["batch_size"]
            del task.eval_kwargs["batch_size"]
        else:
            _hflm_kwargs["batch_size"] = batch_size
        logger.debug(f"Using batch size {_hflm_kwargs['batch_size']} for task {task.name}")
        wrapped_model = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            device=None,  # device of the model is used
            trust_remote_code=True,
            **_hflm_kwargs,
        )

        if task.instruct_mode == "auto":
            if hasattr(tokenizer, "chat_template"):
                apply_chat_template = tokenizer.chat_template is not None
            else:
                apply_chat_template = False
        else:
            apply_chat_template = task.instruct_mode

        task_names = [task.name] if isinstance(task.name, str) else task.name
        if "limit" in task.eval_kwargs:
            _limit = task.eval_kwargs["limit"]
            del task.eval_kwargs["limit"]
        else:
            _limit = limit

        logger.info(f"Evaluating task(s) {task_names} (instruct_mode={apply_chat_template}, kwargs={task.eval_kwargs})")

        for attempt in range(max_retries):
            try:
                _task_t0 = time.perf_counter()
                lm_eval_result = evaluator.simple_evaluate(
                    model=wrapped_model,
                    tasks=task_names,
                    batch_size=None,
                    use_cache=None,
                    check_integrity=False,
                    limit=_limit,
                    apply_chat_template=apply_chat_template,
                    fewshot_as_multiturn=apply_chat_template,
                    **task.eval_kwargs,
                )
                _task_time_s = time.perf_counter() - _task_t0
                logger.info(f"Task(s) {task_names} completed in {_task_time_s:.3f}s")

                for task_name, result in lm_eval_result["results"].items():
                    # Store all metrics for this task
                    results[task_name] = dict(result)
                    results[task_name]["runtime_s"] = _task_time_s

                    # Track tasks that use accuracy for averaging
                    if "acc,none" in result:
                        acc_tasks.append(task_name)

                if log_samples and "samples" in lm_eval_result:
                    for task_name, samples in lm_eval_result["samples"].items():
                        results[task_name]["samples"] = samples
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Task '{task.name}' failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    torch.cuda.empty_cache()
                else:
                    logger.error(traceback.format_exc())
                    logger.error(f"Task '{task.name}' failed after {max_retries} attempts: {e}")

    if acc_tasks:
        results["average"] = {
            "acc": float(sum([results[task_name]["acc,none"] for task_name in acc_tasks]) / len(acc_tasks)),
        }

    return results
