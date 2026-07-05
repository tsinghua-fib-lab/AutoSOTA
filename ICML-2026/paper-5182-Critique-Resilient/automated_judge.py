import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from model_api import query_llm_batch, query_llm_parallel, query_llm_single
from model_config import ModelSpec, load_registry
from prompt_library import (
    load_answer_guidance,
    load_critique_guidance,
    load_judgment_illposed_guidance,
    load_judgment_critique_guidance,
    load_question_guidance,
)
from utils import (
    answer_key,
    answer_key_from_entry,
    automated_evaluation_key_for_task,
    automated_evaluation_key_from_entry,
    benchmark_answers_from_entries,
    collect_invalid_questions,
    format_key,
    is_latest_outer_attempt,
    judging_task_key,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    safe_load_json,
    setup_logging,
)
from constants import (
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_UNKNOWN,
    JUDGE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
    VALID_CRITIQUE_DEBATE_VERDICTS,
    VALID_ILLPOSED_DEBATE_VERDICTS,
)
from data_models import (
    AnswerEntry,
    AutomatedEvaluation,
    CritiqueEntry,
    DebateMessage,
    EvaluationFile,
    JudgingTask,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
    save_evaluation_entries,
)

logger = logging.getLogger(__name__)

load_dotenv()

INPUT_TOO_LONG_MARKERS = ("context_length_exceeded", "input too long")


def load_decisions(path: Path) -> Dict[Tuple[str, str, str, str, str, str], AutomatedEvaluation]:
    payload = load_evaluation_entries(path)
    decisions: Dict[Tuple[str, str, str, str, str, str], AutomatedEvaluation] = {}
    for entry in payload.decisions:
        key = automated_evaluation_key_from_entry(entry)
        if key:
            decisions[key] = entry
    return decisions


def save_decisions(path: Path, decisions: Dict[str, AutomatedEvaluation]):
    save_evaluation_entries(path, EvaluationFile(decisions=list(decisions.values())))


def build_entry_map(entries, key_fn):
    mapped = {}
    for entry in entries:
        if not entry:
            continue
        key = key_fn(entry)
        if not key or key in mapped:
            continue
        mapped[key] = entry
    return mapped


def latest_outer_attempts_by_question(benchmark_dir: Path) -> Dict[str, Dict[str, int]]:
    latest_by_question: Dict[str, Dict[str, int]] = {}
    for bench_path in benchmark_dir.glob("*.json"):
        q_slug = bench_path.stem
        entries = load_benchmark_entries(bench_path)
        latest_by_question[q_slug] = latest_outer_attempt_by_run(entries)
    return latest_by_question


def final_answer(entry: AnswerEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].answer


def load_answers_with_benchmark_fallback(
    answer_path: Path,
    answer_slug: str,
    q_slug: str,
    fallback_answers: List[Optional[AnswerEntry]],
) -> List[Optional[AnswerEntry]]:
    if answer_slug == q_slug:
        if answer_path.exists():
            raise RuntimeError(
                f"Self-answer file exists for {q_slug}. Remove {answer_path} to use benchmark answers."
            )
        return fallback_answers
    return load_answer_entries(answer_path)


def is_input_length_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in INPUT_TOO_LONG_MARKERS)


def build_failed_decision(task: JudgingTask, judge_slug: str, error_message: str) -> AutomatedEvaluation:
    return AutomatedEvaluation(
        type=task.type,
        mode=task.mode,
        question_model=task.question_model,
        answer_model=task.answer_model,
        critic_model=task.critic_model,
        verdict=JUDGE_VERDICT_UNKNOWN,
        confidence=None,
        reasoning=None,
        status=STATUS_FAILED,
        error=error_message,
        raw_response=None,
        run_id=task.run_id,
        outer_attempt=task.outer_attempt,
        topic_slug=task.topic_slug,
        judge_model=judge_slug,
    )


def build_redactions(registry, alice_model: str, bob_model: str) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    redactions = []
    speaker_map = {"Alice": "Alice", "Bob": "Bob"}
    for name in registry.candidate_model_names(alice_model):
        redactions.append((name, "Alice"))
        speaker_map[name] = "Alice"
    for name in registry.candidate_model_names(bob_model):
        redactions.append((name, "Bob"))
        speaker_map[name] = "Bob"
    return redactions, speaker_map


def redact_text(text: str, redactions: Iterable[Tuple[str, str]]) -> str:
    if not text:
        return ""
    redacted = text
    for needle, replacement in sorted(redactions, key=lambda item: -len(item[0] or "")):
        if not needle:
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])"
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
    return redacted


def format_debate(
    history: Optional[List[DebateMessage]],
    redactions: Iterable[Tuple[str, str]],
    speaker_map: Dict[str, str],
) -> str:
    if not history:
        return "(No debate transcript available.)"
    lines = []
    for msg in history:
        speaker = speaker_map.get(msg.speaker, msg.speaker)
        message = redact_text(msg.message, redactions)
        if msg.round is not None:
            lines.append(f"- {speaker} (round {msg.round}): {message}")
        else:
            lines.append(f"- {speaker}: {message}")
    return "\n".join(lines)


def gather_illposed_tasks(
    debates_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    allow_no_debate: bool = False,
) -> List[JudgingTask]:
    tasks = []
    for debate_file in debates_dir.glob("illposed/*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
        fallback_answers = benchmark_answers_from_entries(
            q_slug,
            benchmark_entries,
        )
        debates = load_debate_entries(debate_file)
        answer_path = answers_dir / q_slug / f"{a_slug}.json"
        answers = load_answers_with_benchmark_fallback(
            answer_path,
            a_slug,
            q_slug,
            fallback_answers,
        )
        debate_map = build_entry_map(
            debates,
            lambda entry: answer_key(q_slug, a_slug, entry.run_id, entry.outer_attempt),
        )
        answer_map = build_entry_map(answers, answer_key_from_entry)
        fallback_map = build_entry_map(
            fallback_answers,
            lambda entry: question_key(q_slug, entry.run_id, entry.outer_attempt),
        )
        keys = set(debate_map) | set(answer_map)
        for key in keys:
            debate = debate_map.get(key)
            answer_record = answer_map.get(key)
            status = answer_record.status if answer_record else None
            if status and status != STATUS_ILL_POSED:
                continue
            question = (debate.question if debate else "") or (answer_record.question if answer_record else "")
            answer_text = final_answer(answer_record) if answer_record else ""
            if not answer_text:
                fallback_record = fallback_map.get(question_key(q_slug, key[0], key[3]))
                answer_text = final_answer(fallback_record) if fallback_record else ""
                if not question and fallback_record:
                    question = fallback_record.question
            history = debate.history if debate else []
            if not question and not answer_text and not history:
                continue
            if not history and not allow_no_debate:
                run_id = key[0] if key else None
                logger.warning(
                    f"Skipping illposed/{q_slug}/{a_slug}/{run_id}: no debate history (use --allow-no-debate to override)"
                )
                continue
            alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or a_slug)
            bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or q_slug)
            run_id = (debate.run_id if debate else None) or (answer_record.run_id if answer_record else None)
            outer_attempt = (debate.outer_attempt if debate else None) or (
                answer_record.outer_attempt if answer_record else None
            )
            topic_slug = (debate.topic_slug if debate else None) or (answer_record.topic_slug if answer_record else None)
            tasks.append(
                JudgingTask(
                    type="illposed",
                    question=question,
                    answer=answer_text,
                    debate_history=history,
                    question_model=q_slug,
                    answer_model=a_slug,
                    critic_model=None,
                    alice_model=alice_model,
                    bob_model=bob_model,
                    run_id=run_id,
                    outer_attempt=outer_attempt,
                    topic_slug=topic_slug,
                )
            )
    return tasks


def last_critique_text(crit_entry: CritiqueEntry, context: str) -> str:
    attempts = crit_entry.attempts or []
    if not attempts:
        raise ValueError(f"Missing critique attempts for judgment task: {context}")
    last = attempts[-1]
    if not last.verdict:
        raise ValueError(f"Missing critique verdict for judgment task: {context}")
    if last.notes is None:
        raise ValueError(f"Missing critique notes for judgment task: {context}")
    if not isinstance(last.notes, str):
        raise ValueError(f"Invalid critique notes type for judgment task: {context}")
    return f"Verdict: {last.verdict}\nNotes: {last.notes}"


def gather_critique_tasks(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    allow_no_debate: bool = False,
    include_correct: bool = False,
) -> List[JudgingTask]:
    tasks = []
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
            fallback_answers = benchmark_answers_from_entries(
                q_slug,
                benchmark_entries,
            )
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                critiques = load_critique_entries(crit_file)
                answer_path = answers_dir / q_slug / f"{answer_slug}.json"
                answers = load_answers_with_benchmark_fallback(
                    answer_path,
                    answer_slug,
                    q_slug,
                    fallback_answers,
                )
                debates = load_debate_entries(
                    debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                )
                critique_map = build_entry_map(
                    critiques,
                    lambda entry: answer_key(
                        entry.question_author,
                        entry.answer_author,
                        entry.run_id,
                        entry.outer_attempt,
                    ),
                )
                answer_map = build_entry_map(answers, answer_key_from_entry)
                debate_map = build_entry_map(
                    debates,
                    lambda entry: answer_key(q_slug, answer_slug, entry.run_id, entry.outer_attempt),
                )
                fallback_map = build_entry_map(
                    fallback_answers,
                    lambda entry: question_key(q_slug, entry.run_id, entry.outer_attempt),
                )
                for key, crit_entry in critique_map.items():
                    if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                        continue
                    attempts = crit_entry.attempts or []
                    last_attempt = attempts[-1] if attempts else None
                    if not last_attempt or last_attempt.verdict in {None, CRITIQUE_VERDICT_UNKNOWN}:
                        run_id = key[0] if key else None
                        logger.warning(
                            f"Skipping critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{run_id}: no valid final verdict"
                        )
                        continue
                    if last_attempt and last_attempt.verdict == CRITIQUE_VERDICT_CORRECT and not include_correct:
                        continue
                    debate = debate_map.get(key)
                    question = (debate.question if debate else "") or crit_entry.question
                    answer_record = answer_map.get(key)
                    answer_text = final_answer(answer_record) if answer_record else ""
                    if not answer_text:
                        fallback_record = fallback_map.get(question_key(q_slug, key[0], key[3]))
                        answer_text = final_answer(fallback_record) if fallback_record else ""
                        if not question and fallback_record:
                            question = fallback_record.question
                    context = f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{key[0] if key else 'unknown'}"
                    critique_text = last_critique_text(crit_entry, context)
                    history = debate.history if debate else []
                    if not question and not critique_text and not history:
                        continue
                    if not history and not allow_no_debate:
                        run_id = key[0] if key else None
                        logger.warning(
                            f"Skipping critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{run_id}: no debate history (use --allow-no-debate to override)"
                        )
                        continue
                    alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or critic_slug)
                    bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or answer_slug)
                    run_id = (debate.run_id if debate else None) or crit_entry.run_id
                    outer_attempt = (debate.outer_attempt if debate else None) or crit_entry.outer_attempt
                    topic_slug = (debate.topic_slug if debate else None) or crit_entry.topic_slug
                    tasks.append(
                        JudgingTask(
                            type="critique",
                            mode=mode,
                            question=question,
                            answer=answer_text,
                            critique=critique_text,
                            debate_history=history,
                            question_model=q_slug,
                            answer_model=answer_slug,
                            critic_model=critic_slug,
                            alice_model=alice_model,
                            bob_model=bob_model,
                            run_id=run_id,
                            outer_attempt=outer_attempt,
                            topic_slug=topic_slug,
                        )
                    )
    return tasks


def build_illposed_prompt(task: JudgingTask, guidance_q: str, guidance_j: str, registry) -> str:
    redactions, speaker_map = build_redactions(registry, task.alice_model or "", task.bob_model or "")
    question = redact_text(task.question or "", redactions)
    answer = redact_text(task.answer or "", redactions)
    debate = format_debate(task.debate_history, redactions, speaker_map)
    return (
        "You are a neutral judge. Your task: Determine if ALICE'S CLAIM is valid.\n"
        "Alice claims the question is ill-posed. Bob defends that it is well-posed.\n"
        "The debate helps you assess the claim. If Bob concedes or acknowledges the issue, this CONFIRMS Alice was right.\n"
        "Use the guidance below and ignore any party identity cues.\n\n"
        "Judgment guidance:\n"
        f"{guidance_j}\n\n"
        "Question quality guidance:\n"
        f"{guidance_q}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer (context only):\n"
        f"{answer}\n\n"
        "Debate transcript:\n"
        f"{debate}\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "verdict": "claimant_wins" | "defender_wins_incorrect" | "wrong_problem" | "mixed" | "unknown",\n'
        '  "confidence": 1-5,\n'
        '  "reasoning": "<concise justification>"\n'
        "}\n\n"
        "Verdict meanings:\n"
        '- "claimant_wins": Alice\'s claim is correct - the question is ill-posed (use this even if Bob concedes)\n'
        '- "defender_wins_incorrect": Alice\'s claim is incorrect - Bob shows the question is well-posed and Alice misidentified the issue\n'
        '- "wrong_problem": The question has issues, but Alice\'s specific diagnosis is incorrect\n'
        '- "mixed": Alice makes multiple claims, some correct and some incorrect\n'
        '- "unknown": Cannot determine if Alice\'s claim is valid with confidence\n\n'
        "Confidence scale: 1=very uncertain, 2=uncertain, 3=moderate, 4=confident, 5=very confident\n"
    )


def build_critique_prompt(task: JudgingTask, guidance_a: str, guidance_c: str, guidance_j: str, registry) -> str:
    redactions, speaker_map = build_redactions(registry, task.alice_model or "", task.bob_model or "")
    question = redact_text(task.question or "", redactions)
    answer = redact_text(task.answer or "", redactions)
    critique = redact_text(task.critique or "", redactions)
    debate = format_debate(task.debate_history, redactions, speaker_map)
    return (
        "You are a neutral judge. Your task: Determine if ALICE'S CRITIQUE is correct.\n"
        "Alice claims Bob's answer has errors/issues. Bob defends the answer.\n"
        "The debate helps you assess the critique. If Bob concedes, fixes the issue, or provides missing justification, this CONFIRMS Alice was right.\n"
        "Use the guidance below and ignore any party identity cues.\n\n"
        "Judgment guidance:\n"
        f"{guidance_j}\n\n"
        "Answer quality guidance:\n"
        f"{guidance_a}\n\n"
        "Critique guidance:\n"
        f"{guidance_c}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Critique:\n"
        f"{critique}\n\n"
        "Debate transcript:\n"
        f"{debate}\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "verdict": "claimant_wins" | "defender_wins_incorrect" | "defender_wins_minor" | "wrong_problem" | "mixed" | "unknown",\n'
        '  "confidence": 1-5,\n'
        '  "reasoning": "<concise justification>"\n'
        "}\n\n"
        "Verdict meanings:\n"
        '- "claimant_wins": Alice\'s critique is correct - the answer has substantive flaws (use this even if Bob concedes/fixes)\n'
        '- "defender_wins_incorrect": Alice\'s critique is incorrect - Bob shows the answer is correct and Alice misidentified a problem\n'
        '- "defender_wins_minor": Alice\'s critique is technically correct but about very minor (stylistic only) issues\n'
        '- "wrong_problem": There are issues with the answer, but Alice\'s specific diagnosis is incorrect\n'
        '- "mixed": Alice makes multiple claims, some correct and some incorrect\n'
        '- "unknown": Cannot determine if Alice\'s critique is valid with confidence\n\n'
        "Confidence scale: 1=very uncertain, 2=uncertain, 3=moderate, 4=confident, 5=very confident\n"
    )


def normalize_illposed_verdict(verdict: Optional[str]) -> str:
    """
    Normalize judge verdicts for ill-posedness debates.
    Only accepts canonical forms: claimant_wins, defender_wins_incorrect, wrong_problem, mixed, unknown
    """
    if not verdict:
        return JUDGE_VERDICT_UNKNOWN
    v = verdict.strip().lower()
    if v in VALID_ILLPOSED_DEBATE_VERDICTS:
        return v
    return JUDGE_VERDICT_UNKNOWN


def normalize_critique_verdict(verdict: Optional[str]) -> str:
    """
    Normalize judge verdicts for critique debates.
    Only accepts canonical forms: claimant_wins, defender_wins_incorrect, defender_wins_minor, wrong_problem, mixed, unknown
    """
    if not verdict:
        return JUDGE_VERDICT_UNKNOWN
    v = verdict.strip().lower()
    if v in VALID_CRITIQUE_DEBATE_VERDICTS:
        return v
    return JUDGE_VERDICT_UNKNOWN


def parse_confidence(raw) -> int:
    """
    Parse confidence. Only accepts integers 1-5.
    Raises ValueError if the value is invalid or missing.
    """
    if raw is None:
        raise ValueError("Confidence field is required")
    try:
        conf = int(raw)
        if 1 <= conf <= 5:
            return conf
        raise ValueError(f"Confidence must be 1-5, got: {conf}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid confidence value: {raw}") from e


def parse_judgment(text: str, task: JudgingTask, judge_slug: str) -> AutomatedEvaluation:
    if task.type == "illposed":
        schema = {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": [
                        "claimant_wins",
                        "defender_wins_incorrect",
                        "wrong_problem",
                        "mixed",
                        "unknown",
                    ],
                },
                "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
            "required": ["verdict", "confidence", "reasoning"]
        }
    else:
        schema = {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": [
                        "claimant_wins",
                        "defender_wins_incorrect",
                        "defender_wins_minor",
                        "wrong_problem",
                        "mixed",
                        "unknown",
                    ],
                },
                "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
            "required": ["verdict", "confidence", "reasoning"]
        }
    parsed = safe_load_json(text or "", schema=schema)
    verdict = None
    confidence = None
    reasoning = None
    if isinstance(parsed, dict):
        verdict = parsed.get("verdict")
        # Catch confidence parsing exceptions and mark judgment as failed
        try:
            confidence = parse_confidence(parsed.get("confidence"))
        except ValueError as e:
            logger.warning(f"Failed to parse confidence for task {format_key(judging_task_key(task) or [])}: {e}")
            confidence = None
            # Mark as failed if confidence is required but missing/invalid
            verdict = JUDGE_VERDICT_UNKNOWN
        reasoning = parsed.get("reasoning", None)
    if task.type == "illposed":
        verdict = normalize_illposed_verdict(verdict)
    else:
        verdict = normalize_critique_verdict(verdict)
    status = STATUS_SUCCEEDED if verdict not in {JUDGE_VERDICT_UNKNOWN, "invalid"} else STATUS_FAILED
    return AutomatedEvaluation(
        type=task.type,
        mode=task.mode,
        question_model=task.question_model,
        answer_model=task.answer_model,
        critic_model=task.critic_model,
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        status=status,
        raw_response=text,
        run_id=task.run_id,
        outer_attempt=task.outer_attempt,
        topic_slug=task.topic_slug,
        judge_model=judge_slug,
    )


def chunked(items: List[JudgingTask], size: Optional[int]) -> Iterable[List[JudgingTask]]:
    if not size or size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _batched_query(
    model: str,
    prompts: List[str],
    disable_batch: bool,
    temperature: Optional[float],
    reasoning: Optional[str],
) -> List[str]:
    if not prompts:
        return []
    if len(prompts) == 1:
        return [query_llm_single(model, prompts[0], temperature=temperature, reasoning=reasoning)]
    if disable_batch:
        return query_llm_parallel(
            model,
            prompts,
            temperature=temperature,
            reasoning=reasoning,
        )
    return query_llm_batch(model, prompts, temperature=temperature, reasoning=reasoning)


def main():
    parser = argparse.ArgumentParser(description="Automated judging of debate claims.")
    parser.add_argument("--mode", choices=["illposed", "critiques", "all"], default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--output-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--human-evals-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--disable-batch", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="Process judgments in parallel per task (no batch APIs).")
    parser.add_argument("--limit", type=int, default=None, help="Limit tasks per judge.")
    parser.add_argument("--models", nargs="*", help="Subset of judge models to use (default: all).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-no-debate", action="store_true", help="Allow judging even when there's no debate history.")
    parser.add_argument("--force-correct-critiques", action="store_true", help="Include critiques with verdict 'correct' in judging.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    registry = load_registry(str(args.config))
    guidance_q = load_question_guidance()
    guidance_a = load_answer_guidance()
    guidance_c = load_critique_guidance()
    guidance_j_illposed = load_judgment_illposed_guidance()
    guidance_j_critique = load_judgment_critique_guidance()

    tasks: List[JudgingTask] = []
    if args.mode in {"illposed", "all"}:
        tasks.extend(gather_illposed_tasks(args.debates_dir, args.answers_dir, args.benchmark_dir, registry, args.allow_no_debate))
    if args.mode in {"critiques", "all"}:
        tasks.extend(
            gather_critique_tasks(
                args.debates_dir,
                args.critiques_dir,
                args.answers_dir,
                args.benchmark_dir,
                registry,
                args.allow_no_debate,
                args.force_correct_critiques,
            )
        )

    invalid_questions = collect_invalid_questions(
        args.critiques_dir,
        args.answers_dir,
        args.output_dir,
        args.human_evals_dir,
        registry,
        log_automated_disagreements=False,
    )
    latest_by_question = latest_outer_attempts_by_question(args.benchmark_dir)
    if tasks:
        filtered_tasks: List[JudgingTask] = []
        for task in tasks:
            q_slug = task.question_model
            latest_by_run = latest_by_question.get(q_slug or "")
            outer_attempt = normalize_outer_attempt(task.outer_attempt)
            if latest_by_run and task.run_id is not None:
                if not is_latest_outer_attempt(task.run_id, outer_attempt, latest_by_run):
                    continue
            q_key = question_key(q_slug, task.run_id, outer_attempt)
            if q_key and q_key in invalid_questions:
                continue
            filtered_tasks.append(task)
        tasks = filtered_tasks

    if args.models:
        judges = registry.pick(args.models)
        non_judges = [spec.name for spec in judges if "judge" not in spec.roles]
        if non_judges:
            logger.warning("Skipping non-judge models: %s", ", ".join(non_judges))
            judges = [spec for spec in judges if "judge" in spec.roles]
    else:
        judges = registry.by_role("judge")
    jobs_by_judge: Dict[str, Dict[str, object]] = {spec.name: {"spec": spec, "tasks": []} for spec in judges}

    for task in tasks:
        participants = {task.alice_model, task.bob_model}
        for spec in judges:
            if spec.name in participants:
                continue
            jobs_by_judge[spec.name]["tasks"].append(task)

    def process_judge(spec: ModelSpec, tasks_for_judge: List[JudgingTask]) -> int:
        out_path = args.output_dir / f"{spec.slug}.json"
        decisions = load_decisions(out_path)
        pending = []
        for task in tasks_for_judge:
            eval_key = automated_evaluation_key_for_task(task, spec.slug)
            if not args.overwrite and eval_key in decisions:
                continue
            pending.append(task)
            if args.limit is not None and len(pending) >= args.limit:
                break
        if not pending:
            return 0
        processed = 0
        for batch in chunked(pending, args.batch_size):
            prompts = []
            for task in batch:
                if task.type == "illposed":
                    prompt = build_illposed_prompt(task, guidance_q, guidance_j_illposed, registry)
                else:
                    prompt = build_critique_prompt(task, guidance_a, guidance_c, guidance_j_critique, registry)
                prompts.append(prompt)
            try:
                responses = _batched_query(
                    spec.name,
                    prompts,
                    args.disable_batch,
                    spec.temperature,
                    spec.reasoning,
                )
            except Exception as exc:
                message = str(exc)
                logger.error(f"Judge batch failed for {spec.name}: {exc}")
                if is_input_length_error(message):
                    for task in batch:
                        decision = build_failed_decision(task, spec.slug, message)
                        eval_key = automated_evaluation_key_for_task(task, spec.slug)
                        if eval_key:
                            decisions[eval_key] = decision
                    save_decisions(out_path, decisions)
                    processed += len(batch)
                    logger.info(f"{spec.pretty}: marked {len(batch)} evaluations failed (input too long)")
                continue
            for task, response in zip(batch, responses):
                decision = parse_judgment(response, task, spec.slug)
                eval_key = automated_evaluation_key_for_task(task, spec.slug)
                if eval_key:
                    decisions[eval_key] = decision
            processed += len(batch)
            save_decisions(out_path, decisions)
            logger.info(f"{spec.pretty}: processed {len(batch)} evaluations")
        return len(pending)

    if args.parallel:
        jobs: List[Tuple[ModelSpec, JudgingTask, Path]] = []
        for payload in jobs_by_judge.values():
            spec = payload["spec"]
            tasks_for_judge = payload["tasks"]
            if not tasks_for_judge:
                continue
            out_path = args.output_dir / f"{spec.slug}.json"
            decisions = load_decisions(out_path)
            pending = []
            for task in tasks_for_judge:
                eval_key = automated_evaluation_key_for_task(task, spec.slug)
                if not args.overwrite and eval_key in decisions:
                    continue
                pending.append(task)
                if args.limit is not None and len(pending) >= args.limit:
                    break
            for task in pending:
                jobs.append((spec, task, out_path))

        if jobs:
            path_locks: Dict[Path, threading.Lock] = {}
            locks_guard = threading.Lock()

            def lock_for(path: Path) -> threading.Lock:
                with locks_guard:
                    lock = path_locks.get(path)
                    if lock is None:
                        lock = threading.Lock()
                        path_locks[path] = lock
                    return lock

            def process_task(spec: ModelSpec, task: JudgingTask, out_path: Path):
                try:
                    if task.type == "illposed":
                        prompt = build_illposed_prompt(task, guidance_q, guidance_j_illposed, registry)
                    else:
                        prompt = build_critique_prompt(task, guidance_a, guidance_c, guidance_j_critique, registry)
                    response = query_llm_single(
                        spec.name,
                        prompt,
                        temperature=spec.temperature,
                        reasoning=spec.reasoning,
                    )
                    decision = parse_judgment(response, task, spec.slug)
                except Exception as exc:
                    message = str(exc)
                    if is_input_length_error(message):
                        logger.error(
                            "Judge task failed for %s %s: %s",
                            spec.name,
                            format_key(judging_task_key(task) or []),
                            exc,
                        )
                        failure = build_failed_decision(task, spec.slug, message)
                        lock = lock_for(out_path)
                        with lock:
                            decisions = load_decisions(out_path)
                            eval_key = automated_evaluation_key_for_task(task, spec.slug)
                            if not args.overwrite and eval_key in decisions:
                                return
                            if eval_key:
                                decisions[eval_key] = failure
                            save_decisions(out_path, decisions)
                        logger.info(
                            "%s: recorded failed evaluation %s (input too long)",
                            spec.pretty,
                            format_key(judging_task_key(task) or []),
                        )
                    else:
                        logger.error(
                            "Judge task failed for %s %s: %s",
                            spec.name,
                            format_key(judging_task_key(task) or []),
                            exc,
                        )
                    return

                lock = lock_for(out_path)
                with lock:
                    decisions = load_decisions(out_path)
                    eval_key = automated_evaluation_key_for_task(task, spec.slug)
                    if not args.overwrite and eval_key in decisions:
                        return
                    if eval_key:
                        decisions[eval_key] = decision
                    save_decisions(out_path, decisions)
                logger.info(
                    "%s: processed evaluation %s",
                    spec.pretty,
                    format_key(judging_task_key(task) or []),
                )

            with ThreadPoolExecutor(max_workers=min(32, max(4, len(jobs)))) as pool:
                futures = [pool.submit(process_task, spec, task, out_path) for spec, task, out_path in jobs]
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.error(f"Judge task failed: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=max(4, len(jobs_by_judge))) as pool:
            futures = []
            for payload in jobs_by_judge.values():
                spec = payload["spec"]
                tasks_for_judge = payload["tasks"]
                if not tasks_for_judge:
                    continue
                futures.append(pool.submit(process_judge, spec, tasks_for_judge))
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"Judge worker failed: {exc}")


if __name__ == "__main__":
    main()
