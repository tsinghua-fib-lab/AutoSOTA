import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from model_config import _slugify, load_registry
from prompt_library import (
    load_answer_guidance,
    load_critique_guidance,
    load_debate_illposed_guidance,
    load_debate_critique_guidance,
    load_question_guidance,
)
from model_api import query_llm_single
from utils import (
    benchmark_answers_from_entries,
    clean_math,
    collect_invalid_questions,
    debate_key as debate_task_key,
    is_latest_outer_attempt,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    safe_load_json,
    setup_logging,
)
from constants import CRITIQUE_VERDICT_CORRECT, CRITIQUE_VERDICT_UNKNOWN, STATUS_ILL_POSED, STATUS_SUCCEEDED
from data_models import (
    AnswerEntry,
    BenchmarkEntry,
    CritiqueEntry,
    DebateEntry,
    DebateMessage,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    save_debate_entries,
)

logger = logging.getLogger(__name__)

load_dotenv()


def _debate_key_for_entry(
    entry: Optional[DebateEntry],
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
) -> Optional[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]]:
    if not entry:
        return None
    return debate_task_key(
        question_model,
        answer_model,
        critic_model,
        mode,
        entry.run_id,
        entry.outer_attempt,
    )


def collect_debate_keys(
    entries: List[Optional[DebateEntry]],
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
) -> Set[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]]:
    keys: Set[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]] = set()
    for entry in entries:
        key = _debate_key_for_entry(entry, question_model, answer_model, critic_model, mode)
        if key:
            keys.add(key)
    return keys


def build_entry_map(entries: List[Optional[Any]]) -> Dict[Tuple[Optional[str], Optional[str], Optional[str]], Any]:
    mapped: Dict[Tuple[Optional[str], Optional[str], Optional[str]], Any] = {}
    for entry in entries:
        if not entry:
            continue
        key = question_key(
            getattr(entry, "question_model", None),
            getattr(entry, "run_id", None),
            getattr(entry, "outer_attempt", None),
        )
        if not key:
            continue
        existing = mapped.get(key)
        if not existing:
            mapped[key] = entry
            continue
        existing_status = getattr(existing, "status", None)
        entry_status = getattr(entry, "status", None)
        if existing_status != STATUS_SUCCEEDED and entry_status == STATUS_SUCCEEDED:
            mapped[key] = entry
    return mapped


def index_by_key(
    entries: List[Optional[DebateEntry]],
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
) -> Dict[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]], int]:
    index: Dict[
        Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]], int
    ] = {}
    for idx, entry in enumerate(entries):
        if not entry:
            continue
        key = _debate_key_for_entry(entry, question_model, answer_model, critic_model, mode)
        if key and key not in index:
            index[key] = idx
    return index


def final_answer(entry: AnswerEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].answer


def is_latest_valid_question(
    question_slug: str,
    run_id: Optional[str],
    outer_attempt: Optional[int],
    latest_by_run: Dict[str, int],
    invalid_questions: Set[Tuple[Optional[str], Optional[str], Optional[str]]],
) -> bool:
    outer_value = normalize_outer_attempt(outer_attempt)
    if run_id is not None and not is_latest_outer_attempt(run_id, outer_value, latest_by_run):
        return False
    q_key = question_key(question_slug, run_id, outer_value)
    if q_key and q_key in invalid_questions:
        return False
    return True


def format_illposed_claim(claim: Dict[str, Any], context: str) -> str:
    required = {"verdict": str, "ill_posed": bool, "issues": list, "improvements": str}
    missing = [key for key in required if key not in claim]
    if missing:
        raise ValueError(f"Missing ill-posed claim fields for {context}: {missing}")
    if not isinstance(claim["verdict"], str):
        raise ValueError(f"Invalid ill-posed claim verdict type for {context}")
    if not isinstance(claim["ill_posed"], bool):
        raise ValueError(f"Invalid ill-posed claim ill_posed type for {context}")
    if not isinstance(claim["issues"], list):
        raise ValueError(f"Invalid ill-posed claim issues type for {context}")
    if not isinstance(claim["improvements"], str):
        raise ValueError(f"Invalid ill-posed claim improvements type for {context}")
    issues = claim["issues"] or []
    if any(not isinstance(issue, str) for issue in issues):
        raise ValueError(f"Invalid ill-posed claim issue entries for {context}")
    issue_lines = [f"- {issue}" for issue in issues] if issues else ["- (none)"]
    return (
        "Parsed ill-posed claim:\n"
        f"- verdict: {claim['verdict']}\n"
        f"- ill_posed: {claim['ill_posed']}\n"
        "- issues:\n"
        f"{chr(10).join(issue_lines)}\n"
        f"- improvements: {claim['improvements']}"
    )


def format_critique_for_debate(entry: CritiqueEntry, context: str) -> str:
    attempts = entry.attempts or []
    if not attempts:
        raise ValueError(f"Missing critique attempts for {context}")
    last = attempts[-1]
    if not last.verdict:
        raise ValueError(f"Missing critique verdict for {context}")
    if last.notes is None:
        raise ValueError(f"Missing critique notes for {context}")
    if not isinstance(last.notes, str):
        raise ValueError(f"Invalid critique notes type for {context}")
    return f"Verdict: {last.verdict}\nNotes: {last.notes}"


def final_critique_verdict(entry: CritiqueEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].verdict


def critique_status_details(entry: Optional[CritiqueEntry]) -> str:
    if not entry:
        return "entry=None"
    attempts = entry.attempts or []
    last = attempts[-1] if attempts else None
    last_verdict = last.verdict if last else None
    raw_len = len(last.raw_critique or "") if last else 0
    cleaned_len = len(last.cleaned_critique or "") if last else 0
    notes_len = len(last.notes or "") if last else 0
    return (
        "status="
        f"{entry.status}, attempts={len(attempts)}, last_verdict={last_verdict}, "
        f"raw_len={raw_len}, cleaned_len={cleaned_len}, notes_len={notes_len}"
    )


def load_answers_for_critique_debates(
    answers_dir: Path,
    q_slug: str,
    answer_slug: str,
    benchmark_entries: List[Optional[BenchmarkEntry]],
) -> List[Optional[AnswerEntry]]:
    answer_path = answers_dir / q_slug / f"{answer_slug}.json"
    if answer_slug == q_slug:
        if answer_path.exists():
            raise RuntimeError(
                f"Self-answer file exists for {q_slug}. Remove {answer_path} to use benchmark answers."
            )
        return benchmark_answers_from_entries(q_slug, benchmark_entries)
    return load_answer_entries(answer_path)


def run_round(
    speaker_model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    reasoning: Optional[str],
) -> Tuple[str, bool]:
    reply = query_llm_single(
        speaker_model,
        user_message,
        prompt=system_prompt + "\n\nIMPORTANT: Respond with JSON containing 'message' (your response text) and 'concede' (boolean, true if you concede/agree, false otherwise).",
        temperature=temperature,
        reasoning=reasoning,
        response_format={"type": "json_object"} if 'anthropic' not in speaker_model else None,
    )
    if reply is None or not reply.strip():
        raise ValueError(f"LLM reply empty for {speaker_model}")
    parsed = safe_load_json(reply)
    if isinstance(parsed, dict) and "message" in parsed:
        return clean_math(parsed["message"]), parsed.get("concede", False)
    # Fallback if JSON parsing fails
    return clean_math(reply), False


def illposed_debate(
    defender_model: str,
    claimant_model: str,
    question: str,
    claim_summary: str,
    rounds: int,
    guidance_q: str,
    guidance_d: str,
    defender_temp: float,
    defender_reason: Optional[str],
    claimant_temp: float,
    claimant_reason: Optional[str],
    allow_concede: bool,
) -> List[DebateMessage]:
    history: List[DebateMessage] = []
    system_prompt_defender = (
        "You are Bob, defending the well-posedness of your question. Use the guidance to rebut the ill-posedness claim."
        f"\nQuestion quality guidance:\n{guidance_q}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    system_prompt_claimant = (
        "You are Alice, justifying why the question is ill-posed. Stay concise and evidence-based."
        f"\nQuestion quality guidance:\n{guidance_q}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    last_message = (
        f"Alice argues your question is ill-posed.\nQuestion:\n{question}\n\nClaim:\n{claim_summary}\n"
        "Respond with a short defense as Bob."
    )
    for r in range(1, rounds + 1):
        defender_message, defender_concede = run_round(defender_model, system_prompt_defender, last_message, defender_temp, defender_reason)
        history.append(DebateMessage(round=r, speaker="Bob", message=defender_message, concede=defender_concede))
        if allow_concede and defender_concede:
            # Intentional asymmetry: defender concession ends the round immediately.
            break
        claimant_prompt = (
            f"Bob responded:\n{defender_message}\n\nQuestion:\n{question}\n\n"
            "Restate your ill-posedness reasoning or acknowledge if the defense resolves your concerns as Alice."
        )
        claimant_message, claimant_concede = run_round(claimant_model, system_prompt_claimant, claimant_prompt, claimant_temp, claimant_reason)
        history.append(DebateMessage(round=r, speaker="Alice", message=claimant_message, concede=claimant_concede))
        if allow_concede and claimant_concede:
            # Claimant concedes after both sides have spoken in this round.
            break
        last_message = (
            f"Alice replied:\n{claimant_message}\n\nQuestion:\n{question}\n"
            "Respond briefly to move the discussion forward as Bob."
        )
    return history


def critique_debate(
    defender_model: str,
    claimant_model: str,
    question: str,
    answer: str,
    critique: str,
    rounds: int,
    guidance_a: str,
    guidance_c: str,
    guidance_d: str,
    author_temp: float,
    author_reason: Optional[str],
    critic_temp: float,
    critic_reason: Optional[str],
    allow_concede: bool,
) -> List[DebateMessage]:
    history: List[DebateMessage] = []
    system_prompt_author = (
        "You are Bob, responding to Alice's critique of your answer. Be concise and correct errors when valid."
        f"\nAnswer quality guidance:\n{guidance_a}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    system_prompt_critic = (
        "You are Alice, continuing your critique. Acknowledge fixes if they address the issue. Stay factual."
        f"\nCritique quality guidance:\n{guidance_c}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    last_message = (
        f"Alice raised a critique about your answer.\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nCritique:\n{critique}\n"
        "Respond briefly as Bob."
    )
    for r in range(1, rounds + 1):
        author_message, author_concede = run_round(defender_model, system_prompt_author, last_message, author_temp, author_reason)
        history.append(DebateMessage(round=r, speaker="Bob", message=author_message, concede=author_concede))
        if allow_concede and author_concede:
            break
        critic_prompt = (
            f"Bob replied:\n{author_message}\n\nOriginal critique:\n{critique}\n\nQuestion:\n{question}\n"
            "Follow up concisely as Alice."
        )
        critic_message, critic_concede = run_round(claimant_model, system_prompt_critic, critic_prompt, critic_temp, critic_reason)
        history.append(DebateMessage(round=r, speaker="Alice", message=critic_message, concede=critic_concede))
        if allow_concede and critic_concede:
            break
        last_message = (
            f"Alice replied:\n{critic_message}\n\nQuestion:\n{question}\nAnswer:\n{answer}\n"
            "Respond briefly as Bob."
        )
    return history


def main():
    parser = argparse.ArgumentParser(description="Facilitate debates on ill-posedness claims or critiques.")
    parser.add_argument("--mode", choices=["ill-posed", "critique"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--output-dir", type=Path, default=Path("debates"))
    parser.add_argument("--auto-evals-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--human-evals-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-allow-concede", action="store_true", help="Disable early stop on concession.")
    parser.add_argument("--parallel", action="store_true", help="Process debates in parallel per task.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.rounds < 1:
        parser.error("--rounds must be >= 1")

    registry = load_registry(str(args.config))
    guidance_q = load_question_guidance()
    guidance_a = load_answer_guidance()
    guidance_c = load_critique_guidance()
    guidance_d_illposed = load_debate_illposed_guidance()
    guidance_d_critique = load_debate_critique_guidance()
    invalid_questions = collect_invalid_questions(
        args.critiques_dir,
        args.answers_dir,
        args.auto_evals_dir,
        args.human_evals_dir,
        registry,
        log_automated_disagreements=False,
    )

    debates = 0

    if args.mode == "ill-posed":
        if args.parallel:
            tasks = []
            for bench_path in args.benchmark_dir.glob("*.json"):
                q_slug = bench_path.stem
                question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                if not question_model:
                    continue
                benchmark_entries = load_benchmark_entries(bench_path)
                latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
                answers_dir = args.answers_dir / q_slug
                for answer_file in answers_dir.glob("*.json"):
                    answer_model_slug = answer_file.stem
                    answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_model_slug), None)
                    if not answer_model:
                        continue
                    if answer_model_slug == q_slug:
                        raise RuntimeError(
                            f"Self-answer file exists for {q_slug}. Remove {answer_file} to use benchmark answers."
                        )
                    records = load_answer_entries(answer_file)
                    debate_path = args.output_dir / "illposed" / q_slug / f"{answer_model_slug}.json"
                    existing = load_debate_entries(debate_path)
                    existing_keys = collect_debate_keys(existing, q_slug, answer_model_slug, None, None)
                    for idx, rec in enumerate(records):
                        if args.limit is not None and len(tasks) >= args.limit:
                            break
                        if not rec or rec.status != STATUS_ILL_POSED:
                            continue
                        if not is_latest_valid_question(
                            q_slug,
                            rec.run_id,
                            rec.outer_attempt,
                            latest_by_run,
                            invalid_questions,
                        ):
                            continue
                        if len(existing) > idx and existing[idx]:
                            continue
                        key = debate_task_key(
                            q_slug,
                            answer_model_slug,
                            None,
                            None,
                            rec.run_id,
                            rec.outer_attempt,
                        )
                        if key and key in existing_keys:
                            continue
                        tasks.append(
                            (
                                question_model,
                                answer_model,
                                answer_model_slug,
                                q_slug,
                                rec,
                                idx,
                                debate_path,
                            )
                        )
                    if args.limit is not None and len(tasks) >= args.limit:
                        break
                if args.limit is not None and len(tasks) >= args.limit:
                    break

            if tasks:
                path_locks: Dict[Path, threading.Lock] = {}
                locks_guard = threading.Lock()

                def lock_for(path: Path) -> threading.Lock:
                    with locks_guard:
                        lock = path_locks.get(path)
                        if lock is None:
                            lock = threading.Lock()
                            path_locks[path] = lock
                        return lock

                def process_task(
                    question_model,
                    answer_model,
                    answer_model_slug: str,
                    q_slug: str,
                    rec: AnswerEntry,
                    idx: int,
                    debate_path: Path,
                ) -> bool:
                    context = f"{q_slug}/{answer_model_slug}/{idx}"
                    try:
                        claim_summary = format_illposed_claim(rec.ill_posed_claim or {}, context)
                        history = illposed_debate(
                            question_model.name,
                            answer_model.name,
                            rec.question,
                            claim_summary,
                            args.rounds,
                            guidance_q,
                            guidance_d_illposed,
                            question_model.temperature,
                            question_model.reasoning,
                            answer_model.temperature,
                            answer_model.reasoning,
                            not args.no_allow_concede,
                        )
                    except Exception as exc:
                        logger.error(
                            f"Failed to generate ill-posed debate for {q_slug}/{answer_model_slug}/{idx}: {exc}"
                        )
                        return False

                    lock = lock_for(debate_path)
                    with lock:
                        existing = load_debate_entries(debate_path)
                        existing_keys = collect_debate_keys(existing, q_slug, answer_model_slug, None, None)
                        if len(existing) > idx and existing[idx]:
                            return False
                        key = debate_task_key(
                            q_slug,
                            answer_model_slug,
                            None,
                            None,
                            rec.run_id,
                            rec.outer_attempt,
                        )
                        if key and key in existing_keys:
                            return False
                        if len(existing) <= idx:
                            existing.extend([None for _ in range(idx - len(existing) + 1)])
                        existing[idx] = DebateEntry(
                            question=rec.question,
                            alice_model=answer_model.slug,
                            bob_model=question_model.slug,
                            claimant=answer_model.slug,
                            run_id=rec.run_id,
                            outer_attempt=rec.outer_attempt,
                            topic_slug=rec.topic_slug,
                            history=history,
                        )
                        save_debate_entries(debate_path, existing)
                    return True

                with ThreadPoolExecutor(max_workers=min(32, max(4, len(tasks)))) as pool:
                    futures = [
                        pool.submit(process_task, *task)
                        for task in tasks
                    ]
                    pbar = tqdm(total=len(futures), desc="Ill-posed debates")
                    for fut in as_completed(futures):
                        try:
                            if fut.result():
                                debates += 1
                        except Exception as exc:
                            logger.error(f"Ill-posed debate task failed: {exc}")
                        finally:
                            pbar.update(1)
                    pbar.close()
            else:
                logger.info("No ill-posed debates to generate.")
        else:
            # Count total tasks for progress bar
            total_tasks = 0
            for bench_path in args.benchmark_dir.glob("*.json"):
                q_slug = bench_path.stem
                question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                if not question_model:
                    continue
                benchmark_entries = load_benchmark_entries(bench_path)
                latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
                answers_dir = args.answers_dir / q_slug
                for answer_file in answers_dir.glob("*.json"):
                    answer_model_slug = answer_file.stem
                    answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_model_slug), None)
                    if not answer_model:
                        continue
                    if answer_model_slug == q_slug:
                        raise RuntimeError(
                            f"Self-answer file exists for {q_slug}. Remove {answer_file} to use benchmark answers."
                        )
                    records = load_answer_entries(answer_file)
                    for idx, rec in enumerate(records):
                        if args.limit is not None and total_tasks >= args.limit:
                            break
                        if not rec or rec.status != STATUS_ILL_POSED:
                            continue
                        if not is_latest_valid_question(
                            q_slug,
                            rec.run_id,
                            rec.outer_attempt,
                            latest_by_run,
                            invalid_questions,
                        ):
                            continue
                        debate_path = args.output_dir / "illposed" / q_slug / f"{answer_model_slug}.json"
                        existing = load_debate_entries(debate_path)
                        existing_keys = collect_debate_keys(existing, q_slug, answer_model_slug, None, None)
                        if len(existing) > idx and existing[idx]:
                            continue
                        key = debate_task_key(
                            q_slug,
                            answer_model_slug,
                            None,
                            None,
                            rec.run_id,
                            rec.outer_attempt,
                        )
                        if key and key in existing_keys:
                            continue
                        total_tasks += 1
                    if args.limit is not None and total_tasks >= args.limit:
                        break
                if args.limit is not None and total_tasks >= args.limit:
                    break

            # Process with progress bar
            pbar = tqdm(total=total_tasks, desc="Ill-posed debates")
            for bench_path in args.benchmark_dir.glob("*.json"):
                q_slug = bench_path.stem
                question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                if not question_model:
                    continue
                benchmark_entries = load_benchmark_entries(bench_path)
                latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
                answers_dir = args.answers_dir / q_slug
                for answer_file in answers_dir.glob("*.json"):
                    answer_model_slug = answer_file.stem
                    answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_model_slug), None)
                    if not answer_model:
                        continue
                    if answer_model_slug == q_slug:
                        raise RuntimeError(
                            f"Self-answer file exists for {q_slug}. Remove {answer_file} to use benchmark answers."
                        )
                    records = load_answer_entries(answer_file)

                    # Load debate file once per answer_file instead of per record
                    debate_path = args.output_dir / "illposed" / q_slug / f"{answer_model_slug}.json"
                    existing = load_debate_entries(debate_path)
                    existing_keys = collect_debate_keys(existing, q_slug, answer_model_slug, None, None)

                    for idx, rec in enumerate(records):
                        if args.limit is not None and debates >= args.limit:
                            break
                        if not rec or rec.status != STATUS_ILL_POSED:
                            continue
                        if not is_latest_valid_question(
                            q_slug,
                            rec.run_id,
                            rec.outer_attempt,
                            latest_by_run,
                            invalid_questions,
                        ):
                            continue
                        claim = rec.ill_posed_claim or {}
                        context = f"{q_slug}/{answer_model_slug}/{idx}"
                        claim_summary = format_illposed_claim(claim, context)

                        if len(existing) > idx and existing[idx]:
                            continue
                        key = debate_task_key(
                            q_slug,
                            answer_model_slug,
                            None,
                            None,
                            rec.run_id,
                            rec.outer_attempt,
                        )
                        if key and key in existing_keys:
                            continue

                        try:
                            history = illposed_debate(
                                question_model.name,
                                answer_model.name,
                                rec.question,
                                claim_summary,
                                args.rounds,
                                guidance_q,
                                guidance_d_illposed,
                                question_model.temperature,
                                question_model.reasoning,
                                answer_model.temperature,
                                answer_model.reasoning,
                                not args.no_allow_concede,
                            )
                            if len(existing) <= idx:
                                existing.extend([None for _ in range(idx - len(existing) + 1)])
                            existing[idx] = DebateEntry(
                                question=rec.question,
                                alice_model=answer_model.slug,
                                bob_model=question_model.slug,
                                claimant=answer_model.slug,
                                run_id=rec.run_id,
                                outer_attempt=rec.outer_attempt,
                                topic_slug=rec.topic_slug,
                                history=history,
                            )
                            save_debate_entries(debate_path, existing)
                            if key:
                                existing_keys.add(key)
                            debates += 1
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Failed to generate ill-posed debate for {q_slug}/{answer_model_slug}/{idx}: {e}")
                            pbar.update(1)
                            continue
                    if args.limit is not None and debates >= args.limit:
                        break
                if args.limit is not None and debates >= args.limit:
                    break
            pbar.close()
    else:
        # critique debates
        if args.parallel:
            tasks = []
            for crit_mode_dir in args.critiques_dir.glob("*"):
                mode = crit_mode_dir.name
                for q_dir in crit_mode_dir.glob("*"):
                    q_slug = q_dir.name
                    question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                    if not question_model:
                        continue
                    benchmark_entries = load_benchmark_entries(args.benchmark_dir / f"{q_slug}.json")
                    latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
                    latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
                    latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
                    for crit_file in q_dir.glob("*.json"):
                        parts = crit_file.stem.split("__")
                        if len(parts) != 2:
                            continue
                        critic_slug, answer_slug = parts
                        critic_model = next((spec for spec in registry.models.values() if spec.slug == critic_slug), None)
                        answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_slug), None)
                        if not critic_model or not answer_model:
                            continue
                        critiques = load_critique_entries(crit_file)
                        answers = load_answers_for_critique_debates(
                            args.answers_dir,
                            q_slug,
                            answer_slug,
                            benchmark_entries,
                        )
                        answer_map = build_entry_map(answers)
                        debate_path = args.output_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                        existing = load_debate_entries(debate_path)
                        existing_keys = collect_debate_keys(existing, q_slug, answer_slug, critic_slug, mode)
                        for idx, crit_entry in enumerate(critiques):
                            if args.limit is not None and len(tasks) >= args.limit:
                                break
                            if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                                logger.warning(
                                    "Skipping critique/"
                                    f"{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: "
                                    f"not succeeded ({critique_status_details(crit_entry)})"
                                )
                                continue
                            verdict = final_critique_verdict(crit_entry)
                            if verdict in {None, CRITIQUE_VERDICT_UNKNOWN}:
                                logger.warning(
                                    "Skipping critique/"
                                    f"{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: "
                                    f"unknown final verdict ({critique_status_details(crit_entry)})"
                                )
                                continue
                            if verdict == CRITIQUE_VERDICT_CORRECT:
                                continue
                            if not is_latest_valid_question(
                                q_slug,
                                crit_entry.run_id,
                                crit_entry.outer_attempt,
                                latest_by_run,
                                invalid_questions,
                            ):
                                continue
                            question_key_value = question_key(q_slug, crit_entry.run_id, crit_entry.outer_attempt)
                            if not question_key_value:
                                continue
                            debate_key_value = debate_task_key(
                                q_slug,
                                answer_slug,
                                critic_slug,
                                mode,
                                crit_entry.run_id,
                                crit_entry.outer_attempt,
                            )
                            if debate_key_value and debate_key_value in existing_keys:
                                continue
                            answer_entry = answer_map.get(question_key_value)
                            if not answer_entry:
                                continue
                            answer_text = final_answer(answer_entry) or ""
                            tasks.append(
                                (
                                    mode,
                                    q_slug,
                                    critic_slug,
                                    answer_slug,
                                    critic_model,
                                    answer_model,
                                    crit_entry,
                                    answer_text,
                                    idx,
                                    debate_path,
                                )
                            )
                        if args.limit is not None and len(tasks) >= args.limit:
                            break
                    if args.limit is not None and len(tasks) >= args.limit:
                        break
                if args.limit is not None and len(tasks) >= args.limit:
                    break

            if tasks:
                path_locks: Dict[Path, threading.Lock] = {}
                locks_guard = threading.Lock()

                def lock_for(path: Path) -> threading.Lock:
                    with locks_guard:
                        lock = path_locks.get(path)
                        if lock is None:
                            lock = threading.Lock()
                            path_locks[path] = lock
                        return lock

                def process_task(
                    mode: str,
                    q_slug: str,
                    critic_slug: str,
                    answer_slug: str,
                    critic_model,
                    answer_model,
                    crit_entry: CritiqueEntry,
                    answer_text: str,
                    idx: int,
                    debate_path: Path,
                ) -> bool:
                    context = f"{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}"
                    try:
                        critique_text = format_critique_for_debate(crit_entry, context)
                        history = critique_debate(
                            answer_model.name,
                            critic_model.name,
                            crit_entry.question,
                            answer_text,
                            critique_text,
                            args.rounds,
                            guidance_a,
                            guidance_c,
                            guidance_d_critique,
                            answer_model.temperature,
                            answer_model.reasoning,
                            critic_model.temperature,
                            critic_model.reasoning,
                            not args.no_allow_concede,
                        )
                    except Exception as exc:
                        logger.error(
                            f"Failed to generate critique debate for {mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: {exc}"
                        )
                        return False

                    lock = lock_for(debate_path)
                    with lock:
                        existing = load_debate_entries(debate_path)
                        key = debate_task_key(
                            q_slug,
                            answer_slug,
                            critic_slug,
                            mode,
                            crit_entry.run_id,
                            crit_entry.outer_attempt,
                        )
                        if not key:
                            return False
                        entry = DebateEntry(
                            question=crit_entry.question,
                            alice_model=critic_model.slug,
                            bob_model=answer_model.slug,
                            run_id=crit_entry.run_id,
                            outer_attempt=crit_entry.outer_attempt,
                            topic_slug=crit_entry.topic_slug,
                            answer_author=answer_model.slug,
                            critic=critic_model.slug,
                            history=history,
                        )
                        existing.append(entry)
                        save_debate_entries(debate_path, existing)
                    return True

                with ThreadPoolExecutor(max_workers=min(32, max(4, len(tasks)))) as pool:
                    futures = [
                        pool.submit(process_task, *task)
                        for task in tasks
                    ]
                    pbar = tqdm(total=len(futures), desc="Critique debates")
                    for fut in as_completed(futures):
                        try:
                            if fut.result():
                                debates += 1
                        except Exception as exc:
                            logger.error(f"Critique debate task failed: {exc}")
                        finally:
                            pbar.update(1)
                    pbar.close()
            else:
                logger.info("No critique debates to generate.")
        else:
            # Count total tasks for progress bar
            total_tasks = 0
            for crit_mode_dir in args.critiques_dir.glob("*"):
                mode = crit_mode_dir.name
                for q_dir in crit_mode_dir.glob("*"):
                    q_slug = q_dir.name
                    question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                    if not question_model:
                        continue
                    benchmark_entries = load_benchmark_entries(args.benchmark_dir / f"{q_slug}.json")
                    latest_by_run = latest_outer_attempt_by_run(benchmark_entries)
                    for crit_file in q_dir.glob("*.json"):
                        parts = crit_file.stem.split("__")
                        if len(parts) != 2:
                            continue
                        critic_slug, answer_slug = parts
                        critic_model = next((spec for spec in registry.models.values() if spec.slug == critic_slug), None)
                        answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_slug), None)
                        if not critic_model or not answer_model:
                            continue
                        critiques = load_critique_entries(crit_file)
                        answers = load_answers_for_critique_debates(
                            args.answers_dir,
                            q_slug,
                            answer_slug,
                            benchmark_entries,
                        )
                        answer_map = build_entry_map(answers)
                        for idx, crit_entry in enumerate(critiques):
                            if args.limit is not None and total_tasks >= args.limit:
                                break
                            if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                                logger.warning(
                                    "Skipping critique/"
                                    f"{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: "
                                    f"not succeeded ({critique_status_details(crit_entry)})"
                                )
                                continue
                            verdict = final_critique_verdict(crit_entry)
                            if verdict in {None, CRITIQUE_VERDICT_UNKNOWN}:
                                logger.warning(
                                    "Skipping critique/"
                                    f"{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: "
                                    f"unknown final verdict ({critique_status_details(crit_entry)})"
                                )
                                continue
                            if final_critique_verdict(crit_entry) == CRITIQUE_VERDICT_CORRECT:
                                continue
                            if not is_latest_valid_question(
                                q_slug,
                                crit_entry.run_id,
                                crit_entry.outer_attempt,
                                latest_by_run,
                                invalid_questions,
                            ):
                                continue
                            debate_path = args.output_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                            existing = load_debate_entries(debate_path)
                            existing_keys = collect_debate_keys(existing, q_slug, answer_slug, critic_slug, mode)
                            question_key_value = question_key(q_slug, crit_entry.run_id, crit_entry.outer_attempt)
                            if not question_key_value:
                                continue
                            debate_key_value = debate_task_key(
                                q_slug,
                                answer_slug,
                                critic_slug,
                                mode,
                                crit_entry.run_id,
                                crit_entry.outer_attempt,
                            )
                            if debate_key_value and debate_key_value in existing_keys:
                                continue
                            if not answer_map.get(question_key_value):
                                continue
                            total_tasks += 1
                        if args.limit is not None and total_tasks >= args.limit:
                            break
                    if args.limit is not None and total_tasks >= args.limit:
                        break
                if args.limit is not None and total_tasks >= args.limit:
                    break

            # Process with progress bar
            pbar = tqdm(total=total_tasks, desc="Critique debates")
            for crit_mode_dir in args.critiques_dir.glob("*"):
                mode = crit_mode_dir.name
                for q_dir in crit_mode_dir.glob("*"):
                    q_slug = q_dir.name
                    question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                    if not question_model:
                        continue
                    benchmark_entries = load_benchmark_entries(args.benchmark_dir / f"{q_slug}.json")
                    for crit_file in q_dir.glob("*.json"):
                        parts = crit_file.stem.split("__")
                        if len(parts) != 2:
                            continue
                        critic_slug, answer_slug = parts
                        critic_model = next((spec for spec in registry.models.values() if spec.slug == critic_slug), None)
                        answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_slug), None)
                        if not critic_model or not answer_model:
                            continue
                        critiques = load_critique_entries(crit_file)
                        answers = load_answers_for_critique_debates(
                            args.answers_dir,
                            q_slug,
                            answer_slug,
                            benchmark_entries,
                        )
                        answer_map = build_entry_map(answers)

                        # Load debate file once per critique file instead of per record
                        debate_path = args.output_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                        existing = load_debate_entries(debate_path)
                        existing_keys = collect_debate_keys(existing, q_slug, answer_slug, critic_slug, mode)

                        for idx, crit_entry in enumerate(critiques):
                            if args.limit is not None and debates >= args.limit:
                                break
                            if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                                continue
                            verdict = final_critique_verdict(crit_entry)
                            if verdict in {None, CRITIQUE_VERDICT_UNKNOWN}:
                                logger.warning(f"Skipping critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: unknown final verdict")
                                continue
                            if verdict == CRITIQUE_VERDICT_CORRECT:
                                continue
                            if not is_latest_valid_question(
                                q_slug,
                                crit_entry.run_id,
                                crit_entry.outer_attempt,
                                latest_by_run,
                                invalid_questions,
                            ):
                                continue
                            question_key_value = question_key(q_slug, crit_entry.run_id, crit_entry.outer_attempt)
                            if not question_key_value:
                                continue
                            debate_key_value = debate_task_key(
                                q_slug,
                                answer_slug,
                                critic_slug,
                                mode,
                                crit_entry.run_id,
                                crit_entry.outer_attempt,
                            )
                            if debate_key_value and debate_key_value in existing_keys:
                                continue
                            answer_entry = answer_map.get(question_key_value)
                            if not answer_entry:
                                continue
                            answer_text = final_answer(answer_entry) or ""
                            context = f"{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}"
                            critique_text = format_critique_for_debate(crit_entry, context)

                            try:
                                history = critique_debate(
                                    answer_model.name,
                                    critic_model.name,
                                    crit_entry.question,
                                    answer_text,
                                    critique_text,
                                    args.rounds,
                                    guidance_a,
                                    guidance_c,
                                    guidance_d_critique,
                                    answer_model.temperature,
                                    answer_model.reasoning,
                                    critic_model.temperature,
                                    critic_model.reasoning,
                                    not args.no_allow_concede,
                                )
                                entry = DebateEntry(
                                    question=crit_entry.question,
                                    alice_model=critic_model.slug,
                                    bob_model=answer_model.slug,
                                    run_id=crit_entry.run_id,
                                    outer_attempt=crit_entry.outer_attempt,
                                    topic_slug=crit_entry.topic_slug,
                                    answer_author=answer_model.slug,
                                    critic=critic_model.slug,
                                    history=history,
                                )
                                index = index_by_key(existing, q_slug, answer_slug, critic_slug, mode)
                                if debate_key_value and debate_key_value in index:
                                    existing[index[debate_key_value]] = entry
                                else:
                                    existing.append(entry)
                                save_debate_entries(debate_path, existing)
                                if debate_key_value:
                                    existing_keys.add(debate_key_value)
                                debates += 1
                                pbar.update(1)
                            except Exception as e:
                                logger.error(f"Failed to generate critique debate for {mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: {e}")
                                pbar.update(1)
                                continue
                        if args.limit is not None and debates >= args.limit:
                            break
                    if args.limit is not None and debates >= args.limit:
                        break
                if args.limit is not None and debates >= args.limit:
                    break
            pbar.close()

    logger.info(f"Generated {debates} debate transcripts.")


if __name__ == "__main__":
    main()
