from typing import List, Optional, Union
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import (
    simple_evaluate as _simple_evaluate,
)  # as evaluate_harness_downstream
from lm_eval.utils import make_table as harness_make_table
from lm_eval.tasks import TaskManager

from ..utils import QERA_SRC_DIR


@torch.no_grad()
def evaluate_harness_downstream(
    model,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker=None,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager=None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
):
    model.eval()
    model = HFLM(model)
    if task_manager is None:
        QERA_HARNESS_TASK_DIR = QERA_SRC_DIR.parent.joinpath("qera_harness_tasks")
        task_manager = TaskManager(
            verbosity=verbosity,
            include_path=QERA_HARNESS_TASK_DIR.absolute().as_posix(),
            include_defaults=True,
        )
    results = _simple_evaluate(
        model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        use_cache=use_cache,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        delete_requests_cache=delete_requests_cache,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        check_integrity=check_integrity,
        write_out=write_out,
        log_samples=log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=gen_kwargs,
        task_manager=task_manager,
        verbosity=verbosity,
        predict_only=predict_only,
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
    )
    table_view: str = harness_make_table(results)
    if "groups" in results:
        group_table_view = harness_make_table(results, "groups")

    _ = results.pop("samples")
    _ = results.pop("pretty_env_info")
    if "config" in results and "model_dtype" in results["config"]:
        results["config"]["model_dtype"] = str(results["config"]["model_dtype"])
    results["table_view"] = table_view
    return results
