import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from model_config import ModelSpec, _slugify, load_registry
from prompt_library import (
    build_critique_prompt,
    build_critique_refine,
    build_critique_self_check,
    load_critique_guidance,
)
from self_improvement import self_improve_critiques
from utils import (
    _ensure_non_empty_responses,
    benchmark_answers_from_entries,
    clean_math,
    collect_invalid_questions,
    critique_key,
    is_latest_outer_attempt,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    safe_load_json,
    setup_logging,
)
from model_api import query_llm_batch, query_llm_parallel, query_llm_single
from constants import (
    CRITIQUE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
    VALID_CRITIQUE_VERDICTS,
)
from data_models import (
    AnswerEntry,
    BenchmarkEntry,
    CritiqueAttempt,
    CritiqueEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    save_critique_entries,
)

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass(frozen=True)
class CritiqueJob:
    question: str
    answer_text: str
    answer_author: str
    question_author: str
    run_id: Optional[str]
    outer_attempt: Optional[int]
    topic_slug: Optional[str]
    output_path: Path
    record_key: Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]


def final_question(entry: BenchmarkEntry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def final_answer(entry: AnswerEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].answer


def extract_structured_critique(text: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """
    Parse a critique JSON (if available) and return verdict and notes (single string).
    Falls back to marking verdict unknown and capturing the raw text as notes.

    Valid verdicts: "correct", "incorrect", "insufficient", "obscure"
    Invalid verdicts are treated as "unknown".
    """
    verdict = CRITIQUE_VERDICT_UNKNOWN
    notes: str = ""
    suggestions: Optional[str] = None
    if not text:
        return verdict, notes, suggestions

    parsed = safe_load_json(
        text,
        schema={
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["correct", "incorrect", "insufficient", "obscure"],
                },
                "notes": {"type": "string"},
                "suggestions": {"type": "string"},
            },
            "required": ["verdict", "notes"]
        },
    )
    if isinstance(parsed, dict):
        # Validate verdict against allowed set (from constants.py)
        parsed_verdict = parsed.get("verdict")
        if isinstance(parsed_verdict, str) and parsed_verdict in VALID_CRITIQUE_VERDICTS:
            verdict = parsed_verdict
        elif isinstance(parsed_verdict, str):
            # Log invalid verdicts for debugging
            logger.warning(f"Invalid critique verdict '{parsed_verdict}', treating as 'unknown'")
            verdict = CRITIQUE_VERDICT_UNKNOWN

        if "notes" not in parsed:
            logger.warning("Critique JSON missing required 'notes' field; treating as 'unknown'")
            return CRITIQUE_VERDICT_UNKNOWN, "", None

        raw_notes = parsed.get("notes")
        if isinstance(raw_notes, list):
            notes = "; ".join(str(n) for n in raw_notes)
        elif raw_notes is not None:
            notes = str(raw_notes)

        raw_suggestions = parsed.get("suggestions")
        if isinstance(raw_suggestions, list):
            suggestions = "; ".join(str(s) for s in raw_suggestions)
        elif raw_suggestions is not None:
            suggestions = str(raw_suggestions)

    if not notes and text.strip() and verdict == CRITIQUE_VERDICT_UNKNOWN:
        notes = text.strip()

    return verdict, notes, suggestions


def _batched_query(model: str, prompts: List[str], disable_batch: bool, temperature: float, reasoning: Optional[str]) -> List[str]:
    if not prompts:
        return []
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


def build_answer_map(
    entries: List[Optional[AnswerEntry]],
    latest_by_run: Optional[Dict[str, int]] = None,
    invalid_questions: Optional[Set[Tuple[Optional[str], Optional[str], Optional[str]]]] = None,
) -> Dict[Tuple[Optional[str], Optional[str], Optional[str]], AnswerEntry]:
    mapped: Dict[Tuple[Optional[str], Optional[str], Optional[str]], AnswerEntry] = {}
    for entry in entries:
        if not entry:
            continue
        outer_attempt = normalize_outer_attempt(entry.outer_attempt)
        if latest_by_run and entry.run_id is not None:
            if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                continue
        key = question_key(entry.question_model, entry.run_id, outer_attempt)
        if invalid_questions and key in invalid_questions:
            continue
        if not key:
            continue
        existing = mapped.get(key)
        if not existing or (existing.status != STATUS_SUCCEEDED and entry.status == STATUS_SUCCEEDED):
            mapped[key] = entry
    return mapped


def load_answer_records(
    answers_dir: Path,
    q_slug: str,
    answer_author: str,
    benchmark_entries: List[Optional[BenchmarkEntry]],
) -> List[Optional[AnswerEntry]]:
    a_slug = _slugify(answer_author)
    answer_path = answers_dir / q_slug / f"{a_slug}.json"
    if a_slug == q_slug:
        if answer_path.exists():
            raise RuntimeError(
                f"Self-answer file exists for {q_slug}. Remove {answer_path} to use benchmark answers."
            )
        return benchmark_answers_from_entries(
            q_slug,
            benchmark_entries,
        )
    return load_answer_entries(answer_path)


def prepare_pairs(
    mode: str,
    question_model: str,
    answer_models: List[str],
    answers_root: Path,
    limit: Optional[int],
    benchmark_entries: List[Optional[BenchmarkEntry]],
    critic_names: Optional[Set[str]],
    latest_by_run: Dict[str, int],
    invalid_questions: Set[Tuple[Optional[str], Optional[str], Optional[str]]],
) -> List[Tuple[Tuple[Optional[str], Optional[str], Optional[str]], str, str]]:
    pairs = []
    critic_name_set = set(critic_names or [])
    q_slug = _slugify(question_model)
    question_author_answers = load_answer_records(
        answers_root,
        q_slug,
        question_model,
        benchmark_entries,
    )
    question_author_map = build_answer_map(
        question_author_answers,
        latest_by_run=latest_by_run,
        invalid_questions=invalid_questions,
    )

    if mode == "contradictor":
        critic_list = sorted(critic_name_set) if critic_name_set else answer_models
        # Critics review the question author's self-answers
        for critic_model in critic_list:
            if critic_model == question_model:
                continue
            for key, author_entry in question_author_map.items():
                if limit is not None and len(pairs) >= limit:
                    break
                if not author_entry:
                    continue
                pairs.append((key, critic_model, question_model))
        return pairs

    if mode == "evaluator" and critic_name_set and question_model not in critic_name_set:
        return pairs

    for answer_model in answer_models:
        if mode in {"all", "custom"} and critic_name_set and answer_model not in critic_name_set:
            continue
        a_slug = _slugify(answer_model)
        answer_path = answers_root / q_slug / f"{a_slug}.json"
        answer_records = load_answer_entries(answer_path)
        if answer_model == question_model and not answer_records:
            answer_records = question_author_answers

        answer_map = build_answer_map(
            answer_records,
            latest_by_run=latest_by_run,
            invalid_questions=invalid_questions,
        )
        for key, target_entry in answer_map.items():
            if limit is not None and len(pairs) >= limit:
                break
            question_author_entry = question_author_map.get(key)
            if not question_author_entry or not target_entry:
                continue
            if mode == "evaluator":
                if answer_model == question_model:
                    continue
                critic = question_model
                pairs.append((key, critic, answer_model))
            elif mode in {"all", "custom"}:
                critic = answer_model
                pairs.append((key, critic, target_entry.answer_model))
    return pairs


def generate_critiques_batch(
    critic_model: str,
    jobs: List[CritiqueJob],
    guidance: str,
    max_rounds: int,
    self_improve: bool,
    disable_batch: bool,
    temperature: float,
    reasoning: Optional[str],
) -> List[List[CritiqueAttempt]]:
    prompts = [
        build_critique_prompt(job.question, job.answer_author, job.answer_text, guidance)
        for job in jobs
    ]
    raw_critiques = _batched_query(critic_model, prompts, disable_batch, temperature, reasoning)
    cleaned_critiques = [clean_math(r) for r in raw_critiques]

    if not self_improve:
        results: List[List[CritiqueAttempt]] = []
        for cleaned_text, raw_text in zip(cleaned_critiques, raw_critiques):
            verdict, notes, suggestions = extract_structured_critique(cleaned_text)
            results.append(
                [
                    CritiqueAttempt(
                        round=1,
                        cleaned_critique=cleaned_text,
                        raw_critique=raw_text,
                        verdict=verdict,
                        notes=notes,
                        suggestions=suggestions,
                        evaluation=None,
                    )
                ]
            )
        return results

    questions = [job.question for job in jobs]
    answer_texts = [job.answer_text for job in jobs]

    def eval_prompt(q: str, crit: str, idx: int):
        return build_critique_self_check(q, answer_texts[idx], crit, guidance)

    answer_lookup = {q: ans for q, ans in zip(questions, answer_texts)}

    def refine_prompt(q: str, crit: str, fb: str):
        base_answer = answer_lookup.get(q, "")
        return build_critique_refine(q, base_answer, crit, fb)

    results = self_improve_critiques(
        critic_model,
        questions,
        cleaned_critiques,
        eval_prompt,
        refine_prompt,
        temperature=temperature,
        reasoning=reasoning,
        max_rounds=max_rounds,
        disable_batch=disable_batch,
        raw_initial_critiques=raw_critiques,
    )

    enriched_all: List[List[CritiqueAttempt]] = []
    for res in results:
        enriched_attempts = []
        for att in res.attempts:
            verdict, notes, suggestions = extract_structured_critique(att.answer)
            enriched_attempts.append(
                CritiqueAttempt(
                    round=att.round,
                    cleaned_critique=att.answer,
                    raw_critique=att.raw_answer or "",
                    verdict=verdict,
                    notes=notes,
                    suggestions=suggestions,
                    evaluation=att.evaluation,
                )
            )
        enriched_all.append(enriched_attempts)

    return enriched_all


def generate_critique_single(
    spec: ModelSpec,
    job: CritiqueJob,
    guidance: str,
    max_rounds: int,
    self_improve: bool,
) -> List[CritiqueAttempt]:
    prompt = build_critique_prompt(job.question, job.answer_author, job.answer_text, guidance)
    raw_critique = query_llm_single(
        spec.name,
        prompt,
        temperature=spec.temperature,
        reasoning=spec.reasoning,
    )
    cleaned_critique = clean_math(raw_critique)

    if not self_improve:
        verdict, notes, suggestions = extract_structured_critique(cleaned_critique)
        return [
            CritiqueAttempt(
                round=1,
                cleaned_critique=cleaned_critique,
                raw_critique=raw_critique,
                verdict=verdict,
                notes=notes,
                suggestions=suggestions,
                evaluation=None,
            )
        ]

    question = job.question
    answer_text = job.answer_text

    def eval_prompt(q: str, crit: str, _idx: int) -> str:
        return build_critique_self_check(q, answer_text, crit, guidance)

    def refine_prompt(q: str, crit: str, fb: str) -> str:
        return build_critique_refine(q, answer_text, crit, fb)

    results = self_improve_critiques(
        spec.name,
        [question],
        [cleaned_critique],
        eval_prompt,
        refine_prompt,
        temperature=spec.temperature,
        reasoning=spec.reasoning,
        max_rounds=max_rounds,
        disable_batch=True,
        raw_initial_critiques=[raw_critique],
    )

    enriched_attempts: List[CritiqueAttempt] = []
    for att in (results[0].attempts if results else []):
        verdict, notes, suggestions = extract_structured_critique(att.answer)
        enriched_attempts.append(
            CritiqueAttempt(
                round=att.round,
                cleaned_critique=att.answer,
                raw_critique=att.raw_answer or "",
                verdict=verdict,
                notes=notes,
                suggestions=suggestions,
                evaluation=att.evaluation,
            )
        )

    return enriched_attempts


def main():
    parser = argparse.ArgumentParser(description="Generate critiques for answers.")
    parser.add_argument(
        "--mode",
        choices=["contradictor", "evaluator", "all", "custom"],
        required=True,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--output-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--auto-evals-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--human-evals-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--max-rounds", type=int, default=5, help="Max self-improvement rounds. Ignored if self-improve disabled.")
    parser.add_argument("--no-self-improve", action="store_false", dest="self_improve", help="Disable critique self-improvement.",)
    parser.add_argument(
        "--rerun-failures",
        action="store_true",
        help="Regenerate critiques that already exist but failed or are unknown.",
    )
    parser.add_argument("--disable-batch", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="Process critiques in parallel per task (no batch APIs).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of critiques per pair.")
    parser.add_argument("--custom-map", type=Path, help="JSON for custom mode: list of {question_author, answer_author, critic}.")
    parser.add_argument("--models", nargs="*", help="Subset of models to involve (default: all critique-capable).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    parser.set_defaults(self_improve=True)
    args = parser.parse_args()

    setup_logging(args.log_level)

    registry = load_registry(str(args.config))
    guidance = load_critique_guidance()

    critic_specs = registry.pick(args.models) if args.models else registry.by_role("critique")
    critic_names = {spec.name for spec in critic_specs}
    invalid_questions = collect_invalid_questions(
        args.output_dir,
        args.answers_dir,
        args.auto_evals_dir,
        args.human_evals_dir,
        registry,
        log_automated_disagreements=False,
    )

    custom_pairs: List[Tuple[str, str, str]] = []
    if args.mode == "custom":
        if not args.custom_map:
            raise ValueError("Custom mode requires --custom-map")
        payload = json.loads(args.custom_map.read_text()) if args.custom_map.exists() else []
        for item in payload:
            question_author = item.get("question_author") or item.get("question_owner")
            answer_author = item.get("answer_author") or item.get("answerer")
            critic = item.get("critic")
            if not question_author or not answer_author or not critic:
                continue
            custom_pairs.append((question_author, answer_author, critic))

    for bench_path in args.benchmark_dir.glob("*.json"):
        q_slug = bench_path.stem
        question_model = next((spec.name for spec in registry.models.values() if spec.slug == q_slug), None)
        if not question_model:
            continue
        benchmark_entries = load_benchmark_entries(bench_path)
        latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
        answer_models = [spec.name for spec in registry.models.values() if "answer" in spec.roles]

        pairs: List[Tuple[Tuple[Optional[str], Optional[str], Optional[str]], str, str]] = []
        if args.mode == "custom":
            for question_author, answerer, critic in custom_pairs:
                if question_author != question_model:
                    continue
                if critic not in critic_names:
                    continue
                answer_records = load_answer_records(
                    args.answers_dir,
                    q_slug,
                    answerer,
                    benchmark_entries,
                )
                answer_map = build_answer_map(
                    answer_records,
                    latest_by_run=latest_by_run,
                    invalid_questions=invalid_questions,
                )
                for key, rec in answer_map.items():
                    if args.limit is not None and len(pairs) >= args.limit:
                        break
                    pairs.append((key, critic, answerer))
        else:
            pairs = prepare_pairs(
                args.mode,
                question_model,
                answer_models,
                args.answers_dir,
                args.limit,
                benchmark_entries,
                critic_names,
                latest_by_run,
                invalid_questions,
            )

        if not pairs:
            continue

        jobs_by_critic: Dict[str, Dict] = {}
        benchmark_question_map: Dict[Tuple[Optional[str], Optional[str], Optional[str]], str] = {}
        for entry in benchmark_entries:
            if not entry:
                continue
            question_text = final_question(entry)
            if not question_text:
                continue
            outer_attempt = normalize_outer_attempt(entry.outer_attempt)
            if entry.run_id is not None and not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                continue
            key = question_key(q_slug, entry.run_id, outer_attempt)
            if key and key in invalid_questions:
                continue
            if key and key not in benchmark_question_map:
                benchmark_question_map[key] = question_text

        for key, critic_model, answer_author in pairs:
            critic_spec = registry.models.get(critic_model)
            if not critic_spec or "critique" not in critic_spec.roles:
                continue
            a_slug = _slugify(answer_author)
            critic_slug = critic_spec.slug
            answer_records = load_answer_records(
                args.answers_dir,
                q_slug,
                answer_author,
                benchmark_entries,
            )
            answer_map = build_answer_map(
                answer_records,
                latest_by_run=latest_by_run,
                invalid_questions=invalid_questions,
            )
            answer_entry = answer_map.get(key)
            if not answer_entry or answer_entry.status in {STATUS_FAILED, STATUS_ILL_POSED}:
                run_id = key[0] if key else None
                logger.info(
                    f"Skipping critique for failed/ill-posed answer {question_model}-{answer_author}-{run_id}"
                )
                continue
            question_text = answer_entry.question or benchmark_question_map.get(key)
            if not question_text:
                run_id = key[0] if key else None
                logger.warning(f"Skipping critique for missing question {question_model}-{answer_author}-{run_id}")
                continue
            output_path = args.output_dir / args.mode / q_slug / f"{critic_slug}__{a_slug}.json"
            existing = load_critique_entries(output_path)
            existing_map: Dict[
                Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]],
                CritiqueEntry,
            ] = {}
            for rec in existing:
                if not rec:
                    continue
                rec_key = critique_key(
                    rec.question_author,
                    rec.answer_author,
                    rec.critic,
                    args.mode,
                    rec.run_id,
                    rec.outer_attempt,
                )
                if rec_key and rec_key not in existing_map:
                    existing_map[rec_key] = rec
            record_key = critique_key(
                q_slug,
                a_slug,
                critic_slug,
                args.mode,
                answer_entry.run_id,
                answer_entry.outer_attempt,
            )
            prior = existing_map.get(record_key)
            if prior and (prior.status == STATUS_SUCCEEDED or not args.rerun_failures):
                continue

            answer_text = final_answer(answer_entry) or ""
            job = CritiqueJob(
                question=question_text,
                answer_text=answer_text,
                answer_author=answer_author,
                question_author=question_model,
                run_id=answer_entry.run_id,
                outer_attempt=answer_entry.outer_attempt,
                topic_slug=answer_entry.topic_slug,
                output_path=output_path,
                record_key=record_key,
            )
            bucket = jobs_by_critic.setdefault(critic_spec.name, {"spec": critic_spec, "jobs": []})
            bucket["jobs"].append(job)

        if args.parallel:
            tasks: List[Tuple[ModelSpec, CritiqueJob]] = []
            for payload in jobs_by_critic.values():
                spec = payload["spec"]
                for job in payload["jobs"]:
                    tasks.append((spec, job))

            if not tasks:
                continue

            path_locks: Dict[Path, threading.Lock] = {}
            locks_guard = threading.Lock()

            def lock_for(path: Path) -> threading.Lock:
                with locks_guard:
                    lock = path_locks.get(path)
                    if lock is None:
                        lock = threading.Lock()
                        path_locks[path] = lock
                    return lock

            def write_record(spec: ModelSpec, job: CritiqueJob, attempts: List[CritiqueAttempt]):
                lock = lock_for(job.output_path)
                with lock:
                    records = load_critique_entries(job.output_path)
                    index_by_key: Dict[
                        Tuple[
                            Optional[str],
                            Optional[str],
                            Optional[str],
                            Optional[str],
                            Optional[str],
                            Optional[str],
                        ],
                        int,
                    ] = {}
                    for rec_idx, rec in enumerate(records):
                        if not rec:
                            continue
                        rec_key = critique_key(
                            rec.question_author,
                            rec.answer_author,
                            rec.critic,
                            args.mode,
                            rec.run_id,
                            rec.outer_attempt,
                        )
                        if rec_key and rec_key not in index_by_key:
                            index_by_key[rec_key] = rec_idx
                    final_verdict = attempts[-1].verdict if attempts else None
                    status = (
                        STATUS_SUCCEEDED
                        if final_verdict and final_verdict != CRITIQUE_VERDICT_UNKNOWN
                        else STATUS_FAILED
                    )
                    record = CritiqueEntry(
                        question=job.question,
                        run_id=job.run_id,
                        outer_attempt=job.outer_attempt,
                        topic_slug=job.topic_slug,
                        question_author=_slugify(job.question_author),
                        critic=spec.slug,
                        answer_author=_slugify(job.answer_author),
                        status=status,
                        attempts=attempts,
                    )
                    if job.record_key and job.record_key in index_by_key:
                        records[index_by_key[job.record_key]] = record
                    else:
                        records.append(record)
                    save_critique_entries(job.output_path, records)

            def process_task(spec: ModelSpec, job: CritiqueJob):
                try:
                    attempts = generate_critique_single(
                        spec,
                        job,
                        guidance,
                        args.max_rounds,
                        args.self_improve,
                    )
                    write_record(spec, job, attempts)
                    logger.info(f"Critique done by {spec.name} for run {job.run_id}")
                except Exception as exc:
                    logger.error(
                        f"Critique failed for {spec.name} run {job.run_id}: {exc}"
                    )

            with ThreadPoolExecutor(max_workers=min(32, max(4, len(tasks)))) as pool:
                futures = [pool.submit(process_task, spec, job) for spec, job in tasks]
                pbar = tqdm(total=len(futures), desc="Generate critiques") if futures else None
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error(f"Critique task failed: {exc}")
                    finally:
                        if pbar:
                            pbar.update(1)
                if pbar:
                    pbar.close()
        else:
            with ThreadPoolExecutor(max_workers=max(4, len(jobs_by_critic))) as pool:
                futures = []

                def process_batch(spec: ModelSpec, jobs: List[CritiqueJob]):
                    attempts_list = generate_critiques_batch(
                        spec.name,
                        jobs,
                        guidance,
                        args.max_rounds,
                        args.self_improve,
                        args.disable_batch,
                        spec.temperature,
                        spec.reasoning,
                    )
                    for job, attempts in zip(jobs, attempts_list):
                        records = load_critique_entries(job.output_path)
                        index_by_key: Dict[
                            Tuple[
                                Optional[str],
                                Optional[str],
                                Optional[str],
                                Optional[str],
                                Optional[str],
                                Optional[str],
                            ],
                            int,
                        ] = {}
                        for rec_idx, rec in enumerate(records):
                            if not rec:
                                continue
                            rec_key = critique_key(
                                rec.question_author,
                                rec.answer_author,
                                rec.critic,
                                args.mode,
                                rec.run_id,
                                rec.outer_attempt,
                            )
                            if rec_key and rec_key not in index_by_key:
                                index_by_key[rec_key] = rec_idx
                        final_verdict = attempts[-1].verdict if attempts else None
                        status = (
                            STATUS_SUCCEEDED
                            if final_verdict and final_verdict != CRITIQUE_VERDICT_UNKNOWN
                            else STATUS_FAILED
                        )
                        record = CritiqueEntry(
                            question=job.question,
                            run_id=job.run_id,
                            outer_attempt=job.outer_attempt,
                            topic_slug=job.topic_slug,
                            question_author=_slugify(job.question_author),
                            critic=spec.slug,
                            answer_author=_slugify(job.answer_author),
                            status=status,
                            attempts=attempts,
                        )
                        if job.record_key and job.record_key in index_by_key:
                            records[index_by_key[job.record_key]] = record
                        else:
                            records.append(record)
                        save_critique_entries(job.output_path, records)
                        logger.info(f"Critique done by {spec.name} for run {job.run_id}")

                for payload in jobs_by_critic.values():
                    spec = payload["spec"]
                    jobs = payload["jobs"]
                    if not jobs:
                        continue
                    futures.append(pool.submit(process_batch, spec, jobs))

                pbar = tqdm(total=len(futures), desc="Generate critiques") if futures else None
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error(f"Critique batch failed: {exc}")
                    finally:
                        if pbar:
                            pbar.update(1)
                if pbar:
                    pbar.close()


if __name__ == "__main__":
    main()
