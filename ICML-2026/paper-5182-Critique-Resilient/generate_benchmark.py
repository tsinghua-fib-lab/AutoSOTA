import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from model_config import _slugify, load_registry
from prompt_library import (
    build_question_prompt,
    build_refine_prompt,
    build_self_check_prompt,
    load_answer_guidance,
    load_question_guidance,
    load_self_critique_guidance,
)
from self_improvement import self_improve_answers
from model_api import query_llm_batch, query_llm_parallel, query_llm_single
from constants import STATUS_FAILED, STATUS_ILL_POSED, STATUS_PENDING, STATUS_SUCCEEDED
from data_models import (
    BenchmarkEntry,
    GenerationRound,
    RefinementAttempt,
    load_benchmark_entries,
    save_benchmark_entries,
)
from utils import (
    _ensure_non_empty_responses,
    _load_parsing_config,
    clean_math,
    collect_invalid_questions,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    setup_logging,
)

logger = logging.getLogger(__name__)

load_dotenv()


def load_topic_info(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text())
    info = {}
    for item in data:
        slug = item.get("slug")
        if slug:
            info[slug] = item
    return info


def load_runs(path: Path, topic_info: Dict[str, Dict]) -> List[Dict]:
    data = json.loads(path.read_text())
    runs = []
    for run_id, payload in data.items():
        topic_slug = payload.get("topic")
        if topic_slug not in topic_info:
            raise ValueError(f"Invalid topic slug '{topic_slug}' in run {run_id}. Valid topics: {list(topic_info.keys())}")
        topic_name = topic_info[topic_slug]["name"]
        outer_attempt = payload.get("outer_attempt", 1)
        runs.append(
            {
                "run_id": str(run_id),
                "outer_attempt": outer_attempt,
                "topic_slug": topic_slug,
                "topic_name": topic_name,
            }
        )
    return runs


def _format_question_answer_with_llm(text: str) -> Optional[str]:
    cfg = _load_parsing_config()
    if not cfg:
        logger.warning("No parsing config found; cannot add [QUESTION]/[ANSWER] tags.")
        return None
    model = cfg.get("model")
    if not model:
        logger.warning("No model specified in parsing config; cannot add [QUESTION]/[ANSWER] tags.")
        return None
    temperature = cfg.get("temperature")
    reasoning = cfg.get("reasoning")
    prompt = (
        "You are a strict formatter. The input should contain a math problem and its solution but may be missing "
        "the [QUESTION] and [ANSWER] tags. Return the same content with those tags inserted. Do not change any "
        "wording, math, or formatting; only add the tags and minimal newlines. If no explicit boundary is present, "
        "treat the first paragraph as the question and the rest as the answer. Output only the tagged text."
    )
    try:
        response = query_llm_single(
            model,
            text,
            prompt=prompt,
            temperature=temperature,
            reasoning=reasoning,
        )
    except Exception:
        logger.exception("Failed to add [QUESTION] and [ANSWER] tags with LLM.")
        return None
    if response is None or not response.strip():
        raise ValueError("LLM reply empty when adding [QUESTION]/[ANSWER] tags")
    return response


def _parse_tagged_question_answer(text: str) -> Optional[Tuple[str, str]]:
    if "[QUESTION]" in text and "[ANSWER]" in text:
        question, answer = text.split("[ANSWER]", 1)
        question = question.replace("[QUESTION]", "").strip()
        answer = answer.strip()
        if not question:
            raise ValueError("Parsed question is empty")
        return question, answer
    return None


def parse_question_answer(text: str) -> Tuple[str, str]:
    parsed = _parse_tagged_question_answer(text)
    if parsed:
        return parsed
    logger.warning("Response missing required [QUESTION] and [ANSWER] tags; attempting LLM tag repair.")
    repaired = _format_question_answer_with_llm(text)
    if repaired:
        parsed = _parse_tagged_question_answer(repaired)
        if parsed:
            return parsed
    raise ValueError("Response missing required [QUESTION] and [ANSWER] tags")


def final_question(entry: Optional[BenchmarkEntry]) -> Optional[str]:
    if not entry:
        return None
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question




def upsert(
    entries: List[Optional[BenchmarkEntry]],
    run_id: str,
    outer_attempt: int,
    topic_slug: str,
    topic_name: str,
) -> BenchmarkEntry:
    for entry in entries:
        if entry and entry.run_id == run_id and entry.outer_attempt == outer_attempt:
            return entry
    entry = BenchmarkEntry(
        run_id=run_id,
        outer_attempt=outer_attempt,
        topic_slug=topic_slug,
        topic_name=topic_name,
        status=STATUS_PENDING,
        generation_rounds=[],
    )
    entries.append(entry)
    return entry


def generate_questions(
    model: str,
    runs: List[Dict],
    prev_map: Dict[Tuple[str, int], BenchmarkEntry],
    disable_batch: bool,
    guidance: str,
    temperature: float,
    reasoning: str,
):
    prompts = []
    for run in runs:
        run_id = run["run_id"]
        topic_name = run["topic_name"]
        outer_attempt = run.get("outer_attempt", 1)
        previous_attempts = list(run.get("previous_questions") or [])
        previous_context = run.get("previous_context")
        if not previous_attempts:
            prev_entry = prev_map.get((run_id, outer_attempt))
            if prev_entry and prev_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                for gen_round in prev_entry.generation_rounds or []:
                    refinements = gen_round.refinement_rounds or []
                    if refinements:
                        question = refinements[-1].question
                        if question:
                            previous_attempts.append(question)
                if (
                    previous_attempts
                    and previous_context is None
                    and outer_attempt > 1
                    and (prev_entry.generation_rounds or [])
                ):
                    prior_entry = prev_map.get((run_id, outer_attempt - 1))
                    prior_question = final_question(prior_entry)
                    if prior_question:
                        previous_context = (
                            "The following questions on this topic failed the self-solve gate or meaningfulness check.\n"
                            "Generate a materially different, well-posed replacement that avoids these issues.\n"
                            "Additionally, ensure the new question is simpler than the last question from "
                            f"outer_attempt {outer_attempt - 1}:\n"
                            f"{prior_question}"
                        )
        prompts.append(
            build_question_prompt(
                topic_name,
                guidance,
                previous_attempts if previous_attempts else None,
                previous_context,
            )
        )

    if len(prompts) == 1:
        responses = [query_llm_single(model, prompts[0], temperature=temperature, reasoning=reasoning)]
        _ensure_non_empty_responses(responses, f"LLM reply empty for {model}")
        return responses
    if disable_batch:
        responses = query_llm_parallel(
            model,
            prompts,
            temperature=temperature,
            reasoning=reasoning,
        )
    else:
        responses = query_llm_batch(model, prompts, temperature=temperature, reasoning=reasoning)
    _ensure_non_empty_responses(responses, f"LLM reply empty for {model}")
    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark questions with self-critique.")
    parser.add_argument("--runs-file", type=Path, required=True, help="JSON mapping run IDs to topic slugs.")
    parser.add_argument("--topic-info-file", type=Path, default=Path("configs/topic_info.json"), help="JSON with topic metadata.")
    parser.add_argument("--models", nargs="*", help="Subset of model names to run.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"), help="Model registry JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"), help="Where to store benchmark files.")
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"), help="Directory with critiques.")
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"), help="Directory with answers.")
    parser.add_argument("--auto-evals-dir", type=Path, default=Path("automated_evaluations"), help="Directory with automated evaluations.")
    parser.add_argument("--human-evals-dir", type=Path, default=Path("evaluations"), help="Directory with human evaluations.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Self-critique rounds.")
    parser.add_argument("--max-outer-attempts", type=int, default=1, help="Max outer attempts when self-answers are adjudicated incorrect.")
    parser.add_argument(
        "--max-question-attempts",
        type=int,
        default=5,
        help="Max full question-generation attempts per run (includes past runs).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of topics (for testing).")
    parser.add_argument("--disable-batch", action="store_true", help="Disable batching even when available.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    registry = load_registry(str(args.config))
    question_guidance = load_question_guidance()
    answer_guidance = load_answer_guidance()
    self_critique_guidance = load_self_critique_guidance()
    topic_info = load_topic_info(args.topic_info_file)
    runs = load_runs(args.runs_file, topic_info)
    if args.limit is not None:
        runs = runs[: args.limit]
    invalid_questions = collect_invalid_questions(
        args.critiques_dir,
        args.answers_dir,
        args.auto_evals_dir,
        args.human_evals_dir,
        registry,
        log_automated_disagreements=False,
    )

    models = registry.pick(args.models) if args.models else registry.by_role("benchmark")

    def process_model(model_spec):
        slug = _slugify(model_spec.name)
        output_path = args.output_dir / f"{slug}.json"
        current_entries = load_benchmark_entries(output_path)

        run_id_to_entry = {}
        for entry in current_entries:
            if not entry:
                continue
            outer_attempt = entry.outer_attempt if entry.outer_attempt is not None else 1
            run_id_to_entry[(entry.run_id, outer_attempt)] = entry

        latest_by_run = latest_outer_attempt_by_run(current_entries)
        run_info_by_id = {run["run_id"]: run for run in runs}
        model_runs = list(runs)
        run_keys = {(run["run_id"], run.get("outer_attempt", 1)) for run in model_runs}

        def previous_questions_for_run(run_id: str, max_attempt: int) -> List[str]:
            collected = []
            for entry in current_entries:
                if not entry or entry.run_id != run_id:
                    continue
                attempt_num = normalize_outer_attempt(entry.outer_attempt)
                if attempt_num is None or attempt_num > max_attempt:
                    continue
                question = final_question(entry)
                if question:
                    collected.append((attempt_num, question))
            collected.sort(key=lambda item: item[0])
            return [question for _attempt, question in collected]

        if args.max_outer_attempts and args.max_outer_attempts > 1:
            for run_id, run_info in run_info_by_id.items():
                base_attempt = normalize_outer_attempt(run_info.get("outer_attempt", 1))
                current_max = latest_by_run.get(run_id, base_attempt or 1)
                if current_max is None or current_max >= args.max_outer_attempts:
                    continue
                current_entry = run_id_to_entry.get((run_id, current_max))
                failed_status = current_entry is not None and current_entry.status == STATUS_FAILED
                q_key = question_key(slug, run_id, current_max)
                invalidated = q_key is not None and q_key in invalid_questions
                if not (invalidated or failed_status):
                    continue
                next_attempt = current_max + 1
                if (run_id, next_attempt) in run_keys:
                    continue
                previous_questions = previous_questions_for_run(run_id, current_max)
                if invalidated and failed_status:
                    previous_context = (
                        "The following questions on this topic had self-answers adjudicated incorrect or "
                        "failed the self-solve gate. Generate a materially different, well-posed replacement "
                        "that is simpler than these questions:"
                    )
                elif invalidated:
                    previous_context = (
                        "The following questions on this topic had self-answers adjudicated incorrect. "
                        "Generate a materially different, well-posed replacement that is simpler than these questions:"
                    )
                else:
                    previous_context = (
                        "The following questions on this topic failed the self-solve gate. "
                        "Generate a materially different, well-posed replacement that is simpler than these questions:"
                    )
                model_runs.append(
                    {
                        "run_id": run_id,
                        "outer_attempt": next_attempt,
                        "topic_slug": run_info["topic_slug"],
                        "topic_name": run_info["topic_name"],
                        "previous_questions": previous_questions,
                        "previous_context": previous_context,
                    }
                )
                run_keys.add((run_id, next_attempt))

        def question_attempts(entry: Optional[BenchmarkEntry]) -> int:
            if not entry or not entry.generation_rounds:
                return 0
            return len(entry.generation_rounds)

        def select_pending_runs() -> List[Dict]:
            pending = []
            for run in model_runs:
                entry = run_id_to_entry.get((run["run_id"], run.get("outer_attempt", 1)))
                if entry and entry.status == STATUS_SUCCEEDED:
                    continue
                if question_attempts(entry) >= args.max_question_attempts:
                    continue
                pending.append(run)
            return pending

        pending_runs = select_pending_runs()
        if not pending_runs:
            return f"No pending topics for {model_spec.name}"

        while pending_runs:
            raw_outputs = generate_questions(
                model_spec.name,
                pending_runs,
                run_id_to_entry,
                args.disable_batch,
                question_guidance,
                model_spec.temperature,
                model_spec.reasoning,
            )

            questions = []
            answers = []
            raw_answers_list = []
            for run, raw in zip(pending_runs, raw_outputs):
                q, a = parse_question_answer(raw)
                q = clean_math(q)
                a = clean_math(a)
                questions.append(q)
                answers.append(a)
                raw_answers_list.append(raw)  # Store raw output

            eval_prompts = lambda q, a, idx: build_self_check_prompt(
                q,
                a,
                self_critique_guidance,
                answer_guidance,
            )
            refine_prompts = lambda q, a, fb: build_refine_prompt(q, a, fb, answer_guidance)

            improvements = self_improve_answers(
                model_spec.name,
                questions,
                answers,
                eval_prompts,
                refine_prompts,
                temperature=model_spec.temperature,
                reasoning=model_spec.reasoning,
                max_rounds=args.max_rounds,
                disable_batch=args.disable_batch,
                raw_initial_answers=raw_answers_list,
            )

            for run, question, result in zip(pending_runs, questions, improvements):
                entry = upsert(
                    current_entries,
                    run["run_id"],
                    run.get("outer_attempt", 1),
                    run["topic_slug"],
                    run["topic_name"],
                )
                run_id_to_entry[(entry.run_id, entry.outer_attempt or 1)] = entry
                attempt_records = [
                    RefinementAttempt(
                        round=att.round,
                        question=question,
                        answer=att.answer,
                        raw_answer=att.raw_answer,
                        evaluation=att.evaluation,
                    )
                    for att in result.attempts
                ]
                generation_round = GenerationRound(
                    refinement_rounds=attempt_records,
                    status=result.status,
                )
                if entry.generation_rounds is None:
                    entry.generation_rounds = []
                entry.generation_rounds.append(generation_round)
                entry.status = result.status

            pending_runs = select_pending_runs()

        save_benchmark_entries(output_path, current_entries)
        return f"Wrote {output_path}"

    with ThreadPoolExecutor(max_workers=max(4, len(models))) as pool:
        futures = [pool.submit(process_model, spec) for spec in models]
        pbar = tqdm(total=len(futures), desc="Generate benchmarks") if futures else None
        for fut in as_completed(futures):
            try:
                msg = fut.result()
                logger.info(msg)
            except Exception as exc:
                logger.error(f"Generation task failed: {exc}")
            finally:
                if pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()


if __name__ == "__main__":
    main()
