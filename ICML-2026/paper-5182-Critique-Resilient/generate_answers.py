import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from model_config import _slugify, load_registry
from prompt_library import (
    build_answer_prompt,
    build_refine_prompt,
    build_self_check_prompt,
    load_answer_guidance,
    load_self_critique_guidance,
)
from self_improvement import self_improve_answers
from model_api import query_llm_batch, query_llm_parallel, query_llm_single
from constants import STATUS_FAILED, STATUS_ILL_POSED, STATUS_SUCCEEDED
from data_models import (
    AnswerAttempt,
    AnswerEntry,
    BenchmarkEntry,
    load_answer_entries,
    load_benchmark_entries,
    save_answer_entries,
)
from utils import (
    _ensure_non_empty_responses,
    answer_key,
    answer_key_from_entry,
    clean_math,
    collect_invalid_questions,
    is_latest_outer_attempt,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    setup_logging,
)

logger = logging.getLogger(__name__)

load_dotenv()


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def build_override_note(overrides: Dict, q_slug: str, a_slug: str, idx: int) -> Optional[str]:
    key = f"{q_slug}/{a_slug}"
    reason = overrides.get("overrides", {}).get(key, {}).get(str(idx))
    if reason:
        return f"Override present: do not mark this question as ill-posed. Reason: {reason}."
    return None


def final_question(entry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def prepare_batch(
    question_model: str,
    answer_model: str,
    benchmark_entries: List[Optional[BenchmarkEntry]],
    existing: List[Optional[AnswerEntry]],
    rerun_failures: bool,
    limit: Optional[int],
    allow_self_answering: bool,
    latest_by_run: Dict[str, int],
    invalid_questions: Set,
    question_slug: str,
) -> List[Tuple[int, Dict, str]]:
    if question_model == answer_model and not allow_self_answering:
        return []
    existing_by_key: Dict = {}
    for rec in existing:
        if not rec:
            continue
        key = answer_key_from_entry(rec)
        if not key:
            continue
        prior = existing_by_key.get(key)
        if not prior or (prior.status != STATUS_SUCCEEDED and rec.status == STATUS_SUCCEEDED):
            existing_by_key[key] = rec
    batch = []
    for idx, entry in enumerate(benchmark_entries):
        if limit is not None and len(batch) >= limit:
            break
        if not entry or entry.status != STATUS_SUCCEEDED:
            status = entry.status if entry else "missing"
            logger.info(f"Skipping question {question_model}-{idx} (status: {status})")
            continue
        outer_attempt = normalize_outer_attempt(entry.outer_attempt)
        if entry.run_id is not None and not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
            continue
        q_key = question_key(question_slug, entry.run_id, outer_attempt)
        if q_key and q_key in invalid_questions:
            continue
        question_text = final_question(entry)
        if not question_text:
            continue
        key = answer_key(
            _slugify(question_model),
            _slugify(answer_model),
            entry.run_id,
            normalize_outer_attempt(entry.outer_attempt),
        )
        prior = existing_by_key.get(key) if key else None
        if prior and prior.status == STATUS_SUCCEEDED:
            continue
        if prior and prior.status in {STATUS_FAILED, STATUS_ILL_POSED} and not rerun_failures:
            continue
        batch.append((idx, entry, question_text))
    return batch


def run_generation(
    question_model: str,
    answer_model: str,
    batch_items: List[Tuple[int, BenchmarkEntry, str]],
    answer_guidance: str,
    self_critique_guidance: str,
    max_rounds: int,
    disable_batch: bool,
    temperature: float,
    reasoning: Optional[str],
    override_payload: Dict,
):
    questions = [item[2] for item in batch_items]
    prompts = [build_answer_prompt(q, answer_guidance) for q in questions]

    q_slug = _slugify(question_model)
    a_slug = _slugify(answer_model)

    if len(prompts) == 1:
        raw_answers = [
            query_llm_single(answer_model, prompts[0], temperature=temperature, reasoning=reasoning)
        ]
    elif disable_batch:
        raw_answers = query_llm_parallel(
            answer_model,
            prompts,
            temperature=temperature,
            reasoning=reasoning,
            desc=f"{a_slug} answers",
        )
    else:
        raw_answers = query_llm_batch(answer_model, prompts, temperature=temperature, reasoning=reasoning)

    _ensure_non_empty_responses(raw_answers, f"LLM reply empty for {a_slug}")
    cleaned_answers = [clean_math(r) for r in raw_answers]

    def eval_prompt(question: str, answer: str, local_idx: int):
        note = build_override_note(override_payload, q_slug, a_slug, batch_items[local_idx][0])
        base = build_self_check_prompt(question, answer, self_critique_guidance, answer_guidance)
        return base + f"\n\n{note}" if note else base
    refine_prompts = lambda q, a, fb: build_refine_prompt(q, a, fb, answer_guidance)

    results = self_improve_answers(
        answer_model,
        questions,
        cleaned_answers,
        eval_prompt,
        refine_prompts,
        temperature=temperature,
        reasoning=reasoning,
        max_rounds=max_rounds,
        disable_batch=disable_batch,
        raw_initial_answers=raw_answers,
    )

    outputs = []
    for (idx, entry, question_text), improved in zip(batch_items, results):
        attempts = [
            AnswerAttempt(
                round=att.round,
                answer=att.answer,
                raw_answer=att.raw_answer,
                evaluation=att.evaluation,
            )
            for att in improved.attempts
        ]
        ill_posed_claim = None
        if improved.status == STATUS_ILL_POSED:
            if improved.attempts:
                ill_posed_claim = improved.attempts[-1].evaluation
            if ill_posed_claim is None:
                ill_posed_claim = improved.last_feedback
        record = AnswerEntry(
            question_model=q_slug,
            answer_model=a_slug,
            question=question_text,
            run_id=entry.run_id,
            outer_attempt=entry.outer_attempt,
            topic_slug=entry.topic_slug,
            ill_posed_claim=ill_posed_claim,
            status=improved.status,
            attempts=attempts,
        )
        outputs.append((idx, record))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate answers for benchmark questions with self-critique.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"), help="Model registry.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"), help="Directory with benchmark JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("answers"), help="Directory to store answers.")
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"), help="Directory with critiques.")
    parser.add_argument("--auto-evals-dir", type=Path, default=Path("automated_evaluations"), help="Directory with automated evaluations.")
    parser.add_argument("--human-evals-dir", type=Path, default=Path("evaluations"), help="Directory with human evaluations.")
    parser.add_argument("--models", nargs="*", help="Subset of models to use as answerers.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Self-improvement rounds.")
    parser.add_argument("--disable-batch", action="store_true", help="Disable batching.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions per pair.")
    parser.add_argument("--rerun-failures", action="store_true", help="Retry failed entries.")
    parser.add_argument("--allow-self-answering", action="store_true", help="Allow models to answer their own questions.")
    parser.add_argument("--illposed-overrides", type=Path, default=Path("configs/illposed_overrides.json"), help="Overrides JSON.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    registry = load_registry(str(args.config))
    answer_guidance = load_answer_guidance()
    self_critique_guidance = load_self_critique_guidance()
    overrides = load_json(args.illposed_overrides, {"overrides": {}})
    invalid_questions = collect_invalid_questions(
        args.critiques_dir,
        args.output_dir,
        args.auto_evals_dir,
        args.human_evals_dir,
        registry,
        log_automated_disagreements=False,
    )

    answer_models = registry.pick(args.models) if args.models else registry.by_role("answer")
    benchmark_files = list(args.benchmark_dir.glob("*.json"))

    tasks = []
    with ThreadPoolExecutor(max_workers=max(4, len(answer_models))) as pool:
        for bench_path in benchmark_files:
            q_slug = bench_path.stem
            question_model = next((spec.name for spec in registry.models.values() if _slugify(spec.name) == q_slug), None)
            if not question_model:
                continue
            benchmark_entries = load_benchmark_entries(bench_path)
            if not benchmark_entries:
                logger.info(f"Skipping empty benchmark: {bench_path}")
                continue
            latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
            for answer_spec in answer_models:
                if answer_spec.name == question_model and not args.allow_self_answering:
                    continue
                out_path = args.output_dir / q_slug / f"{answer_spec.slug}.json"
                existing = load_answer_entries(out_path)
                batch = prepare_batch(
                    question_model,
                    answer_spec.name,
                    benchmark_entries,
                    existing,
                    args.rerun_failures,
                    args.limit,
                    args.allow_self_answering,
                    latest_by_run,
                    invalid_questions,
                    q_slug,
                )
                if not batch:
                    continue

                def run_task(qm=question_model, am=answer_spec, b=batch, b_entries=benchmark_entries, out=out_path):
                    try:
                        outputs = run_generation(
                            qm,
                            am.name,
                            b,
                            answer_guidance,
                            self_critique_guidance,
                            args.max_rounds,
                            args.disable_batch,
                            am.temperature,
                            am.reasoning,
                            overrides,
                        )
                        existing_records = load_answer_entries(out)
                        index_by_key: Dict = {}
                        for rec_idx, rec in enumerate(existing_records):
                            if not rec:
                                continue
                            key = answer_key_from_entry(rec)
                            if key and key not in index_by_key:
                                index_by_key[key] = rec_idx
                        for idx, record in outputs:
                            key = answer_key_from_entry(record)
                            if key and key in index_by_key:
                                existing_records[index_by_key[key]] = record
                            else:
                                existing_records.append(record)
                        save_answer_entries(out, existing_records)
                        return qm, am.name, len(outputs)
                    except Exception:
                        logger.exception("Task failed for %s on %s (%s)", am.name, qm, out)
                        raise

                tasks.append(pool.submit(run_task))

        pbar = tqdm(total=len(tasks), desc="Generate answers") if tasks else None
        for fut in as_completed(tasks):
            try:
                qm, am, count = fut.result()
                logger.info(f"Finished {count} answers for {am} on {qm}")
            except Exception as exc:
                logger.error(f"Task failed: {exc}")
            finally:
                if pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()


if __name__ == "__main__":
    main()
