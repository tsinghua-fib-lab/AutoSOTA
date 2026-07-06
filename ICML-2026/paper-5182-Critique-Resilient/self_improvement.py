from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

from model_api import query_llm_batch, query_llm_parallel, query_llm_single
from utils import _ensure_non_empty_responses, safe_load_json
from constants import STATUS_FAILED, STATUS_ILL_POSED, STATUS_SUCCEEDED


@dataclass
class Attempt:
    round: int
    answer: str
    raw_answer: Optional[str] = None
    evaluation: Optional[Dict] = None
    improved_from: Optional[int] = None


@dataclass
class ImprovementResult:
    final_answer: Optional[str]
    attempts: List[Attempt] = field(default_factory=list)
    status: str = STATUS_FAILED  # succeeded/failed
    last_feedback: Optional[Dict] = None


def _batched_query(model: str, prompts: Sequence[str], disable_batch: bool, **kwargs) -> List[str]:
    if not prompts:
        return []
    if len(prompts) == 1:
        responses = [query_llm_single(model, prompts[0], **kwargs)]
        _ensure_non_empty_responses(responses, f"LLM reply empty for {model}")
        return responses
    if disable_batch:
        responses = query_llm_parallel(model, list(prompts), **kwargs)
    else:
        responses = query_llm_batch(model, prompts, **kwargs)
    _ensure_non_empty_responses(responses, f"LLM reply empty for {model}")
    return responses


def self_improve_answers(
    model: str,
    questions: Sequence[str],
    initial_answers: Sequence[str],
    build_eval_prompt: Callable[[str, str, int], str],
    build_refine_prompt: Callable[[str, str, str], str],
    temperature: Optional[float],
    reasoning: Optional[str],
    max_rounds: int = 5,
    disable_batch: bool = False,
    raw_initial_answers: Optional[Sequence[str]] = None,
) -> List[ImprovementResult]:
    """
    Perform self-critique/improvement loops on a batch of answers.

    Args:
        raw_initial_answers: Optional raw (unparsed) versions of initial_answers
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    results = [ImprovementResult(final_answer=ans) for ans in initial_answers]
    active_indices = list(range(len(questions)))

    # Store raw versions for tracking
    raw_answers_map = {}
    if raw_initial_answers:
        for i, raw in enumerate(raw_initial_answers):
            raw_answers_map[i] = {0: raw}  # round 0 -> raw initial answer
    else:
        for i in range(len(initial_answers)):
            raw_answers_map[i] = {0: initial_answers[i]}

    for round_idx in tqdm(range(max_rounds), desc="Self-improve answers", leave=False):
        eval_prompts = [
            build_eval_prompt(questions[i], results[i].final_answer, i) for i in active_indices
        ]
        eval_responses = _batched_query(
            model,
            eval_prompts,
            disable_batch,
            temperature=temperature,
            reasoning=reasoning,
        )

        next_active = []
        refine_prompts = []
        refine_indices = []

        for idx, eval_text in zip(active_indices, eval_responses):
            evaluation = safe_load_json(
                eval_text,
                schema={
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["pass", "fail"]},
                        "ill_posed": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "improvements": {"type": "string"},
                    },
                    "required": ["verdict", "ill_posed", "issues", "improvements"]
                },
            )
            evaluation_missing = not isinstance(evaluation, dict)
            if evaluation_missing:
                evaluation = {
                    "verdict": "fail",
                    "issues": ["Could not parse evaluation JSON. Raw text: " + eval_text],
                    "ill_posed": False,
                    "improvements": "Self-check evaluation unknown due to parsing failure.",
                }
            # Get raw answer for this round
            raw_answer = raw_answers_map.get(idx, {}).get(round_idx, results[idx].final_answer)
            attempt = Attempt(
                round=round_idx + 1,
                answer=results[idx].final_answer,
                raw_answer=raw_answer,
                evaluation=evaluation,
                improved_from=round_idx,
            )
            results[idx].attempts.append(attempt)
            results[idx].last_feedback = evaluation

            if evaluation_missing:
                results[idx].status = STATUS_FAILED
                continue

            if evaluation.get("verdict") == "pass":
                results[idx].status = STATUS_SUCCEEDED
                continue

            if evaluation.get("ill_posed"):
                results[idx].status = STATUS_ILL_POSED
                continue

            if round_idx == max_rounds - 1:
                results[idx].status = STATUS_FAILED
                continue

            refine_indices.append(idx)
            refine_prompts.append(
                build_refine_prompt(
                    questions[idx],
                    results[idx].final_answer,
                    evaluation.get("improvements", "No specific improvement suggestions provided."),
                )
            )
            next_active.append(idx)

        if not refine_prompts:
            break

        refined_answers = _batched_query(
            model,
            refine_prompts,
            disable_batch,
            temperature=temperature,
            reasoning=reasoning,
        )
        for idx, new_answer in zip(refine_indices, refined_answers):
            # Store raw refined answer for next round
            if idx not in raw_answers_map:
                raw_answers_map[idx] = {}
            raw_answers_map[idx][round_idx + 1] = new_answer
            results[idx].final_answer = new_answer

        active_indices = next_active

    return results


def self_improve_critiques(
    model: str,
    questions: Sequence[str],
    initial_critiques: Sequence[str],
    build_eval_prompt: Callable[[str, str, int], str],
    build_refine_prompt: Callable[[str, str, str], str],
    temperature: Optional[float],
    reasoning: Optional[str],
    max_rounds: int = 5,
    disable_batch: bool = False,
    raw_initial_critiques: Optional[Sequence[str]] = None,
) -> List[ImprovementResult]:
    """
    Perform self-critique/improvement loops on a batch of critiques.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    results = [ImprovementResult(final_answer=crit) for crit in initial_critiques]
    active_indices = list(range(len(questions)))

    raw_critiques_map: Dict[int, Dict[int, str]] = {}
    if raw_initial_critiques:
        for i, raw in enumerate(raw_initial_critiques):
            raw_critiques_map[i] = {0: raw}
    else:
        for i in range(len(initial_critiques)):
            raw_critiques_map[i] = {0: initial_critiques[i]}

    for round_idx in tqdm(range(max_rounds), desc="Self-improve critiques", leave=False):
        eval_prompts = [
            build_eval_prompt(questions[i], results[i].final_answer, i) for i in active_indices
        ]
        eval_responses = _batched_query(
            model,
            eval_prompts,
            disable_batch,
            temperature=temperature,
            reasoning=reasoning,
        )

        next_active = []
        refine_prompts = []
        refine_indices = []

        for idx, eval_text in zip(active_indices, eval_responses):
            evaluation = safe_load_json(
                eval_text,
                schema={
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": ["pass", "fail"]},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "improvements": {"type": "string"},
                    },
                    "required": ["verdict", "issues", "improvements"]
                },
            )
            evaluation_missing = not isinstance(evaluation, dict)
            if evaluation_missing:
                evaluation = {
                    "verdict": "fail",
                    "issues": ["Could not parse evaluation JSON. Raw text: " + eval_text],
                    "improvements": "Self-check evaluation unknown due to parsing failure.",
                }
            raw_answer = raw_critiques_map.get(idx, {}).get(round_idx, results[idx].final_answer)
            attempt = Attempt(
                round=round_idx + 1,
                answer=results[idx].final_answer,
                raw_answer=raw_answer,
                evaluation=evaluation,
                improved_from=round_idx,
            )
            results[idx].attempts.append(attempt)
            results[idx].last_feedback = evaluation

            if evaluation_missing:
                results[idx].status = STATUS_FAILED
                continue

            if evaluation.get("verdict") == "pass":
                results[idx].status = STATUS_SUCCEEDED
                continue

            if round_idx == max_rounds - 1:
                results[idx].status = STATUS_FAILED
                continue

            refine_indices.append(idx)
            refine_prompts.append(
                build_refine_prompt(
                    questions[idx],
                    results[idx].final_answer,
                    evaluation.get("improvements", "No specific improvement suggestions provided."),
                )
            )
            next_active.append(idx)

        if not refine_prompts:
            break

        refined_critiques = _batched_query(
            model,
            refine_prompts,
            disable_batch,
            temperature=temperature,
            reasoning=reasoning,
        )
        for idx, new_critique in zip(refine_indices, refined_critiques):
            if idx not in raw_critiques_map:
                raw_critiques_map[idx] = {}
            raw_critiques_map[idx][round_idx + 1] = new_critique
            results[idx].final_answer = new_critique

        active_indices = next_active

    return results
