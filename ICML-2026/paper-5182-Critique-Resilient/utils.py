import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Dict, List, Optional, Sequence, Tuple


from constants import (
    CRITIQUE_VERDICT_INCORRECT,
    CRITIQUE_VERDICT_INSUFFICIENT,
    CRITIQUE_VERDICT_OBSCURE,
    STATUS_FAILED,
    STATUS_ILL_POSED,
)
from data_models import (
    AnswerAttempt,
    AnswerEntry,
    AutomatedEvaluation,
    BenchmarkEntry,
    HumanEvaluation,
    load_answer_entries,
    load_critique_entries,
    load_evaluation_entries,
    load_human_evaluation_entries,
)
from model_config import ModelRegistry
from victory import (
    ConflictingHumanJudgmentError,
    VictorySide,
    resolve_victory_from_verdicts,
    verdict_to_victory_side,
)

logger = logging.getLogger(__name__)

QuestionKey = Tuple[Optional[str], Optional[str], Optional[str]]
AnswerKey = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
CritiqueKey = Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]
EvaluationKey = Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]


def _normalize_part(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def normalize_outer_attempt(value: Optional[object], default: Optional[int] = 1) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def latest_outer_attempt_by_run(
    entries: Sequence[Any],
    *,
    default_outer_attempt: int = 1,
) -> Dict[str, int]:
    latest: Dict[str, int] = {}
    for entry in entries:
        if not entry:
            continue
        run_id = entry.get("run_id") if isinstance(entry, dict) else getattr(entry, "run_id", None)
        if run_id is None:
            continue
        outer_attempt = entry.get("outer_attempt") if isinstance(entry, dict) else getattr(entry, "outer_attempt", None)
        attempt = normalize_outer_attempt(outer_attempt, default_outer_attempt)
        if attempt is None:
            continue
        run_key = _normalize_part(run_id)
        if run_key is None:
            continue
        existing = latest.get(run_key)
        if existing is None or attempt > existing:
            latest[run_key] = attempt
    return latest


def is_latest_outer_attempt(
    run_id: Optional[object],
    outer_attempt: Optional[object],
    latest_by_run: Dict[str, int],
    *,
    default_outer_attempt: int = 1,
) -> bool:
    if run_id is None:
        return True
    run_key = _normalize_part(run_id)
    if run_key is None:
        return True
    latest = latest_by_run.get(run_key)
    if latest is None:
        return True
    attempt = normalize_outer_attempt(outer_attempt, default_outer_attempt)
    if attempt is None:
        return True
    return attempt >= latest


def question_key(
    question_model: Optional[str], run_id: Optional[str], outer_attempt: Optional[str]
) -> Optional[QuestionKey]:
    if run_id is None or not question_model or outer_attempt is None:
        return None
    return (_normalize_part(run_id), _normalize_part(question_model), _normalize_part(outer_attempt))


def question_key_from_entry(entry: Any, question_model: Optional[str]) -> Optional[QuestionKey]:
    if not entry:
        return None
    run_id = entry.get("run_id") if isinstance(entry, dict) else getattr(entry, "run_id", None)
    outer_attempt = entry.get("outer_attempt") if isinstance(entry, dict) else getattr(entry, "outer_attempt", None)
    return question_key(question_model, run_id, outer_attempt)


def answer_key(
    question_model: Optional[str],
    answer_model: Optional[str],
    run_id: Optional[str],
    outer_attempt: Optional[str],
) -> Optional[AnswerKey]:
    if run_id is None or not question_model or not answer_model or outer_attempt is None:
        return None
    return (
        _normalize_part(run_id),
        _normalize_part(question_model),
        _normalize_part(answer_model),
        _normalize_part(outer_attempt),
    )


def answer_key_from_entry(entry: Any) -> Optional[AnswerKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return answer_key(
            entry.get("question_model"),
            entry.get("answer_model"),
            entry.get("run_id"),
            entry.get("outer_attempt"),
        )
    return answer_key(
        getattr(entry, "question_model", None),
        getattr(entry, "answer_model", None),
        getattr(entry, "run_id", None),
        getattr(entry, "outer_attempt", None),
    )


def critique_key(
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
    run_id: Optional[str],
    outer_attempt: Optional[str],
) -> Optional[CritiqueKey]:
    if run_id is None or not question_model or not answer_model or outer_attempt is None:
        return None
    return (
        _normalize_part(run_id),
        _normalize_part(question_model),
        _normalize_part(answer_model),
        _normalize_part(critic_model),
        _normalize_part(mode),
        _normalize_part(outer_attempt),
    )


def critique_key_from_entry(entry: Any, mode: Optional[str]) -> Optional[CritiqueKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return critique_key(
            entry.get("question_author"),
            entry.get("answer_author"),
            entry.get("critic"),
            mode,
            entry.get("run_id"),
            entry.get("outer_attempt"),
        )
    return critique_key(
        getattr(entry, "question_author", None),
        getattr(entry, "answer_author", None),
        getattr(entry, "critic", None),
        mode,
        getattr(entry, "run_id", None),
        getattr(entry, "outer_attempt", None),
    )


def debate_key(
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
    run_id: Optional[str],
    outer_attempt: Optional[str],
) -> Optional[CritiqueKey]:
    return critique_key(question_model, answer_model, critic_model, mode, run_id, outer_attempt)


def debate_key_from_entry(
    entry: Any,
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
) -> Optional[CritiqueKey]:
    if not entry:
        return None
    run_id = entry.get("run_id") if isinstance(entry, dict) else getattr(entry, "run_id", None)
    outer_attempt = entry.get("outer_attempt") if isinstance(entry, dict) else getattr(entry, "outer_attempt", None)
    return debate_key(question_model, answer_model, critic_model, mode, run_id, outer_attempt)


def automated_evaluation_key_from_entry(entry: Any) -> Optional[EvaluationKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return (
            _normalize_part(entry.get("run_id")),
            _normalize_part(entry.get("question_model")),
            _normalize_part(entry.get("answer_model")),
            _normalize_part(entry.get("critic_model")),
            _normalize_part(entry.get("mode")),
            _normalize_part(entry.get("judge_model")),
            _normalize_part(entry.get("outer_attempt")),
        )
    return (
        _normalize_part(getattr(entry, "run_id", None)),
        _normalize_part(getattr(entry, "question_model", None)),
        _normalize_part(getattr(entry, "answer_model", None)),
        _normalize_part(getattr(entry, "critic_model", None)),
        _normalize_part(getattr(entry, "mode", None)),
        _normalize_part(getattr(entry, "judge_model", None)),
        _normalize_part(getattr(entry, "outer_attempt", None)),
    )


def judging_task_key(entry: Any) -> Optional[CritiqueKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return critique_key(
            entry.get("question_model"),
            entry.get("answer_model"),
            entry.get("critic_model"),
            entry.get("mode"),
            entry.get("run_id"),
            entry.get("outer_attempt"),
        )
    return critique_key(
        getattr(entry, "question_model", None),
        getattr(entry, "answer_model", None),
        getattr(entry, "critic_model", None),
        getattr(entry, "mode", None),
        getattr(entry, "run_id", None),
        getattr(entry, "outer_attempt", None),
    )


def automated_evaluation_key_for_task(entry: Any, judge_model: Optional[str]) -> Optional[EvaluationKey]:
    base = judging_task_key(entry)
    if not base:
        return None
    run_id, question_model, answer_model, critic_model, mode, outer_attempt = base
    return (
        _normalize_part(run_id),
        _normalize_part(question_model),
        _normalize_part(answer_model),
        _normalize_part(critic_model),
        _normalize_part(mode),
        _normalize_part(judge_model),
        _normalize_part(outer_attempt),
    )


def human_evaluation_key_from_entry(entry: Any) -> Optional[CritiqueKey]:
    return judging_task_key(entry)


def _auto_claim_type_matches(decision: AutomatedEvaluation, claim_type: Optional[str]) -> bool:
    if not claim_type:
        return True
    if claim_type == "critique":
        return decision.type in {"critique", "critique_debate"}
    return decision.type == claim_type


def load_automated_decisions(
    auto_eval_dir: Path,
    *,
    claim_type: Optional[str] = None,
) -> Dict[CritiqueKey, List[AutomatedEvaluation]]:
    decisions_by_key: Dict[CritiqueKey, List[AutomatedEvaluation]] = defaultdict(list)
    if not auto_eval_dir.exists():
        return decisions_by_key
    for eval_file in auto_eval_dir.glob("*.json"):
        payload = load_evaluation_entries(eval_file)
        for decision in payload.decisions:
            if not decision or not _auto_claim_type_matches(decision, claim_type):
                continue
            key = judging_task_key(decision)
            if key:
                decisions_by_key[key].append(decision)
    return decisions_by_key


def load_human_decisions(
    human_eval_dir: Path,
    *,
    claim_type: Optional[str] = None,
) -> Dict[CritiqueKey, List[HumanEvaluation]]:
    decisions_by_key: Dict[CritiqueKey, List[HumanEvaluation]] = defaultdict(list)
    if not human_eval_dir.exists():
        return decisions_by_key
    for eval_file in human_eval_dir.glob("*.json"):
        payload = load_human_evaluation_entries(eval_file)
        for decision in payload.decisions:
            if not decision:
                continue
            if claim_type and decision.type != claim_type:
                continue
            key = human_evaluation_key_from_entry(decision)
            if key:
                decisions_by_key[key].append(decision)
    return decisions_by_key


def required_judge_quorum(
    registry: Optional[ModelRegistry],
    *,
    claim_type: str,
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
) -> Optional[int]:
    if registry is None:
        return None
    judge_specs = registry.by_role("judge")
    if not judge_specs:
        return None
    if claim_type == "illposed":
        participants: Iterable[Optional[str]] = (question_model, answer_model)
    else:
        participants = (answer_model, critic_model)
    participant_names = {
        registry.resolve_model_name(name)
        for name in participants
        if name
    }
    return sum(1 for spec in judge_specs if spec.name not in participant_names)


def adjudicate_claim(
    claim_type: str,
    *,
    automated_decisions: Sequence[AutomatedEvaluation],
    human_decisions: Sequence[HumanEvaluation],
    required_automated: Optional[int] = None,
    context: Optional[str] = None,
    log_automated_disagreements: bool = True,
) -> Optional[VictorySide]:
    human_verdicts = [d.verdict for d in human_decisions if d and d.verdict]
    if human_verdicts:
        try:
            return resolve_victory_from_verdicts(
                claim_type,
                human_verdicts=human_verdicts,
                context=context,
            )
        except ConflictingHumanJudgmentError:
            logger.warning("Conflicting human judgments for %s", context or "claim")
            return None

    if required_automated is not None and len(automated_decisions) < required_automated:
        return None

    automated_verdicts = [
        str(d.verdict).strip().lower()
        for d in automated_decisions
        if d and d.verdict and d.status != STATUS_FAILED
    ]
    if not automated_verdicts:
        return None
    unique = set(automated_verdicts)
    if len(unique) != 1:
        if log_automated_disagreements:
            logger.error(
                "Automated judgments disagree for %s: %s",
                context or "unknown task",
                sorted(unique),
            )
        return None
    verdict = next(iter(unique))
    return verdict_to_victory_side(claim_type, verdict)


def resolve_bt_adjudication(
    claim_type: str,
    *,
    automated_decisions: Sequence[AutomatedEvaluation],
    human_decisions: Sequence[HumanEvaluation],
    context: Optional[str] = None,
    log_automated_disagreements: bool = True,
    finegrained_unanimity: bool = False,
) -> VictorySide:
    """
    Resolve BT adjudication: use automated verdicts when unanimous; otherwise require exactly
    one human label and use it. Raise if zero or multiple human labels are present.
    """
    automated_verdicts = [
        str(decision.verdict).strip().lower()
        for decision in automated_decisions
        if decision and decision.verdict and decision.status != STATUS_FAILED
    ]
    if finegrained_unanimity:
        auto_outcome = None
        unique = set(automated_verdicts)
        if len(unique) == 1:
            verdict = next(iter(unique))
            auto_outcome = verdict_to_victory_side(claim_type, verdict)
    else:
        auto_outcome = resolve_victory_from_verdicts(
            claim_type,
            automated_verdicts=automated_verdicts,
            context=context,
            log_automated_disagreements=log_automated_disagreements,
        )
    if auto_outcome is not None:
        return auto_outcome

    human_verdicts = [decision.verdict for decision in human_decisions if decision and decision.verdict]
    valid_human = [
        verdict for verdict in human_verdicts
        if verdict_to_victory_side(claim_type, verdict) is not None
    ]
    if len(valid_human) != 1:
        raise RuntimeError(
            f"Non-unanimous automated judgments for {context or 'claim'}; "
            f"expected exactly 1 valid human label, got {len(valid_human)}"
        )
    human_outcome = verdict_to_victory_side(claim_type, valid_human[0])
    return human_outcome


def adjudicated_claim_outcome(
    claim_key: Optional[CritiqueKey],
    *,
    claim_type: str,
    auto_by_key: Dict[CritiqueKey, List[AutomatedEvaluation]],
    human_by_key: Dict[CritiqueKey, List[HumanEvaluation]],
    registry: Optional[ModelRegistry] = None,
    required_automated: Optional[int] = None,
    log_automated_disagreements: bool = True,
) -> Optional[VictorySide]:
    if not claim_key:
        return None
    if required_automated is None and registry is not None:
        required_automated = required_judge_quorum(
            registry,
            claim_type=claim_type,
            question_model=claim_key[1],
            answer_model=claim_key[2],
            critic_model=claim_key[3],
        )
    return adjudicate_claim(
        claim_type,
        automated_decisions=auto_by_key.get(claim_key, []),
        human_decisions=human_by_key.get(claim_key, []),
        required_automated=required_automated,
        context=format_key(claim_key),
        log_automated_disagreements=log_automated_disagreements,
    )


_NONCORRECTNESS_VERDICTS = {
    CRITIQUE_VERDICT_INCORRECT,
    CRITIQUE_VERDICT_INSUFFICIENT,
    CRITIQUE_VERDICT_OBSCURE,
}


def collect_self_answer_critique_claims(
    critiques_dir: Path,
    *,
    mode: str = "contradictor",
    noncorrect_verdicts: Optional[Iterable[str]] = None,
) -> Dict[QuestionKey, List[CritiqueKey]]:
    claims_by_question: Dict[QuestionKey, List[CritiqueKey]] = defaultdict(list)
    mode_dir = critiques_dir / mode
    if not mode_dir.exists():
        return claims_by_question
    allowed_verdicts = set(noncorrect_verdicts or _NONCORRECTNESS_VERDICTS)
    for q_dir in mode_dir.glob("*"):
        for crit_file in q_dir.glob("*.json"):
            entries = load_critique_entries(crit_file)
            for entry in entries:
                if not entry or entry.status != "succeeded":
                    continue
                if entry.question_author != entry.answer_author:
                    continue
                attempts = entry.attempts or []
                if not attempts:
                    continue
                verdict = attempts[-1].verdict
                if verdict not in allowed_verdicts:
                    continue
                q_key = question_key(entry.question_author, entry.run_id, entry.outer_attempt)
                claim_key = critique_key(
                    entry.question_author,
                    entry.answer_author,
                    entry.critic,
                    mode,
                    entry.run_id,
                    entry.outer_attempt,
                )
                if q_key and claim_key:
                    claims_by_question[q_key].append(claim_key)
    return claims_by_question


def adjudicated_self_answer_outcome(
    question_key_value: Optional[QuestionKey],
    *,
    claims_by_question: Dict[QuestionKey, List[CritiqueKey]],
    auto_by_key: Dict[CritiqueKey, List[AutomatedEvaluation]],
    human_by_key: Dict[CritiqueKey, List[HumanEvaluation]],
    registry: Optional[ModelRegistry] = None,
    log_automated_disagreements: bool = True,
) -> Optional[VictorySide]:
    return _adjudicated_question_outcome(
        question_key_value,
        claim_type="critique",
        claims_by_question=claims_by_question,
        auto_by_key=auto_by_key,
        human_by_key=human_by_key,
        registry=registry,
        log_automated_disagreements=log_automated_disagreements,
    )


def _adjudicated_question_outcome(
    question_key_value: Optional[QuestionKey],
    *,
    claim_type: str,
    claims_by_question: Dict[QuestionKey, List[CritiqueKey]],
    auto_by_key: Dict[CritiqueKey, List[AutomatedEvaluation]],
    human_by_key: Dict[CritiqueKey, List[HumanEvaluation]],
    registry: Optional[ModelRegistry] = None,
    log_automated_disagreements: bool = True,
) -> Optional[VictorySide]:
    if not question_key_value:
        return None
    claim_keys = claims_by_question.get(question_key_value, [])
    if not claim_keys:
        return None
    outcomes: List[VictorySide] = []
    for claim_key in claim_keys:
        outcome = adjudicated_claim_outcome(
            claim_key,
            claim_type=claim_type,
            auto_by_key=auto_by_key,
            human_by_key=human_by_key,
            registry=registry,
            log_automated_disagreements=log_automated_disagreements,
        )
        if outcome is not None:
            outcomes.append(outcome)
    if not outcomes:
        return None
    if VictorySide.ALICE in outcomes:
        return VictorySide.ALICE
    if VictorySide.BOB in outcomes:
        return VictorySide.BOB
    return VictorySide.DROP


def collect_self_answer_adjudications(
    critiques_dir: Path,
    auto_eval_dir: Path,
    human_eval_dir: Path,
    registry: Optional[ModelRegistry],
    *,
    mode: str = "contradictor",
    noncorrect_verdicts: Optional[Iterable[str]] = None,
    log_automated_disagreements: bool = True,
) -> Dict[QuestionKey, VictorySide]:
    claims_by_question = collect_self_answer_critique_claims(
        critiques_dir,
        mode=mode,
        noncorrect_verdicts=noncorrect_verdicts,
    )
    if not claims_by_question:
        return {}
    auto_by_key = load_automated_decisions(auto_eval_dir, claim_type="critique")
    human_by_key = load_human_decisions(human_eval_dir, claim_type="critique")
    adjudications: Dict[QuestionKey, VictorySide] = {}
    for q_key in claims_by_question:
        outcome = adjudicated_self_answer_outcome(
            q_key,
            claims_by_question=claims_by_question,
            auto_by_key=auto_by_key,
            human_by_key=human_by_key,
            registry=registry,
            log_automated_disagreements=log_automated_disagreements,
        )
        if outcome is not None:
            adjudications[q_key] = outcome
    return adjudications


def collect_illposed_claims_by_question(
    answers_dir: Path,
) -> Dict[QuestionKey, List[CritiqueKey]]:
    claims_by_question: Dict[QuestionKey, List[CritiqueKey]] = defaultdict(list)
    if not answers_dir.exists():
        return claims_by_question
    for q_dir in answers_dir.glob("*"):
        for ans_file in q_dir.glob("*.json"):
            entries = load_answer_entries(ans_file)
            for entry in entries:
                if not entry or entry.status != STATUS_ILL_POSED:
                    continue
                question_model = entry.question_model or q_dir.name
                answer_model = entry.answer_model or ans_file.stem
                q_key = question_key(question_model, entry.run_id, entry.outer_attempt)
                claim_key = critique_key(
                    question_model,
                    answer_model,
                    None,
                    None,
                    entry.run_id,
                    entry.outer_attempt,
                )
                if q_key and claim_key:
                    claims_by_question[q_key].append(claim_key)
    return claims_by_question


def collect_illposed_adjudications(
    answers_dir: Path,
    auto_eval_dir: Path,
    human_eval_dir: Path,
    registry: Optional[ModelRegistry],
    *,
    log_automated_disagreements: bool = True,
) -> Dict[QuestionKey, VictorySide]:
    claims_by_question = collect_illposed_claims_by_question(answers_dir)
    if not claims_by_question:
        return {}
    auto_by_key = load_automated_decisions(auto_eval_dir, claim_type="illposed")
    human_by_key = load_human_decisions(human_eval_dir, claim_type="illposed")
    adjudications: Dict[QuestionKey, VictorySide] = {}
    for q_key in claims_by_question:
        outcome = _adjudicated_question_outcome(
            q_key,
            claim_type="illposed",
            claims_by_question=claims_by_question,
            auto_by_key=auto_by_key,
            human_by_key=human_by_key,
            registry=registry,
            log_automated_disagreements=log_automated_disagreements,
        )
        if outcome is not None:
            adjudications[q_key] = outcome
    return adjudications


def collect_invalid_questions(
    critiques_dir: Path,
    answers_dir: Path,
    auto_eval_dir: Path,
    human_eval_dir: Path,
    registry: Optional[ModelRegistry],
    *,
    mode: str = "contradictor",
    noncorrect_verdicts: Optional[Iterable[str]] = None,
    log_automated_disagreements: bool = True,
) -> set[QuestionKey]:
    invalid_questions: set[QuestionKey] = set()
    self_answer_adjudications = collect_self_answer_adjudications(
        critiques_dir,
        auto_eval_dir,
        human_eval_dir,
        registry,
        mode=mode,
        noncorrect_verdicts=noncorrect_verdicts,
        log_automated_disagreements=log_automated_disagreements,
    )
    invalid_questions.update(
        q_key
        for q_key, outcome in self_answer_adjudications.items()
        if outcome == VictorySide.ALICE
    )
    illposed_adjudications = collect_illposed_adjudications(
        answers_dir,
        auto_eval_dir,
        human_eval_dir,
        registry,
        log_automated_disagreements=log_automated_disagreements,
    )
    invalid_questions.update(
        q_key
        for q_key, outcome in illposed_adjudications.items()
        if outcome == VictorySide.ALICE
    )
    return invalid_questions


def collect_invalid_self_answer_questions(
    critiques_dir: Path,
    auto_eval_dir: Path,
    human_eval_dir: Path,
    registry: Optional[ModelRegistry],
    *,
    mode: str = "contradictor",
    noncorrect_verdicts: Optional[Iterable[str]] = None,
    log_automated_disagreements: bool = True,
) -> set[QuestionKey]:
    """Collect questions invalidated by self-answer adjudications only (no ill-posed check)."""
    invalid_questions: set[QuestionKey] = set()
    self_answer_adjudications = collect_self_answer_adjudications(
        critiques_dir,
        auto_eval_dir,
        human_eval_dir,
        registry,
        mode=mode,
        noncorrect_verdicts=noncorrect_verdicts,
        log_automated_disagreements=log_automated_disagreements,
    )
    invalid_questions.update(
        q_key
        for q_key, outcome in self_answer_adjudications.items()
        if outcome == VictorySide.ALICE
    )
    return invalid_questions


def format_key(parts: Sequence[Optional[str]]) -> str:
    return "/".join("unknown" if part in (None, "") else str(part) for part in parts)


def task_key_from_prefix(prefix: str, run_id: Optional[str], outer_attempt: Optional[str]) -> Optional[CritiqueKey]:
    if run_id is None or outer_attempt is None:
        return None
    if not prefix:
        return None
    parts = prefix.split("/")
    if not parts:
        return None
    if parts[0] == "illposed":
        if len(parts) < 3:
            return None
        return critique_key(parts[1], parts[2], None, None, run_id, outer_attempt)
    if parts[0] == "critique":
        if len(parts) < 4:
            return None
        mode = parts[1]
        question_model = parts[2]
        pair = parts[3]
        if "__" not in pair:
            return None
        critic_model, answer_model = pair.split("__", 1)
        return critique_key(question_model, answer_model, critic_model, mode, run_id, outer_attempt)
    return None


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Disable httpx logging at INFO level to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _ensure_non_empty_responses(responses: Sequence[Optional[str]], context: str) -> None:
    for idx, response in enumerate(responses):
        if response is None or (isinstance(response, str) and not response.strip()):
            raise ValueError(f"{context} at index {idx}")


def clean_math(text: str) -> str:
    """Clean LaTeX math delimiters by converting to $ and $$ formats."""
    if not text:
        return text
    text = text.replace("\\( ", "$").replace("\\(", "$")
    text = text.replace(" \\)", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    for env in ("equation", "equation*", "align", "align*", "gather", "gather*"):
        text = text.replace(f"\\begin{{{env}}}", "$$")
        text = text.replace(f"\\end{{{env}}}", "$$")
    return text


def benchmark_answers_from_entries(
    question_model_slug: str,
    benchmark_entries: List[Optional[BenchmarkEntry]],
) -> List[Optional[AnswerEntry]]:
    """Build AnswerEntry records from benchmark entries when self-answer files are disallowed or missing."""
    answers: List[Optional[AnswerEntry]] = []
    for entry in benchmark_entries:
        if not entry:
            answers.append(None)
            continue
        gen_rounds = entry.generation_rounds or []
        if not gen_rounds:
            answers.append(None)
            continue
        refinements = gen_rounds[-1].refinement_rounds or []
        if not refinements:
            answers.append(None)
            continue
        last_ref = refinements[-1]
        attempts = [
            AnswerAttempt(
                round=att.round,
                answer=att.answer,
                raw_answer=att.raw_answer,
                evaluation=att.evaluation,
            )
            for att in refinements
        ]
        answers.append(
            AnswerEntry(
                question_model=question_model_slug,
                answer_model=question_model_slug,
                question=last_ref.question,
                run_id=entry.run_id,
                outer_attempt=entry.outer_attempt,
                topic_slug=entry.topic_slug,
                status=entry.status,
                attempts=attempts,
            )
        )
    return answers


def _load_parsing_config() -> Optional[Dict]:
    cfg_path = Path("configs/parsing.json")
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return None
    return None


def clean_json_text(text: str) -> str:
    text = text.replace("```json", "```")

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            # take first fenced block
            return parts[1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_with_model(text: str, schema: Optional[Dict[str, Any]]) -> Optional[Dict]:
    from model_api import query_llm_single

    cfg = _load_parsing_config()

    if not cfg:
        raise RuntimeError("No parsing config found for JSON repair")
    model = cfg.get("model")
    if not model:
        raise RuntimeError("No model specified in parsing config for JSON repair")
    temperature = cfg.get("temperature", None)
    reasoning = cfg.get("reasoning", None)
    prompt_lines = [
        "You are a strict JSON repair assistant.",
        "Given the malformed text below, output a valid JSON object only.",
        "Be careful to fix stuff like incorrect escaping, missing commas, unbalanced braces, and incorrect quotes.",
        "Some escapings might be done correctly, so be cautious not to over-escape. Use contextual judgment.",
        "You will handle what is mostly math, so use your knowledge of LaTeX syntax to avoid breaking math expressions.",
        "Do not add, remove or change the content of any fields. In particular, if a string field doesn't match an enum allowed value, fail to parse rather than changing it.",
        "Note: if a field is optional in the schema, but the value is null in the input, it should be dropped in the output (unless null is specifically allowed by the schema).",
        "If there are extra fields not in the schema, drop them.",
        "Note 2: If the input text is natural language instead of JSON, but it contains all the relevant information in sufficiently obvious format, try to convert it into JSON format. Just don't try to invent fields.",
    ]
    if schema:
        prompt_lines.append("JSON Schema:")
        prompt_lines.append(json.dumps(schema, indent=2, sort_keys=False))
    prompt_lines.append('If you cannot repair, respond with the following JSON: { \"parsing_error\" : \"Cannot parse\", \"reason\": ... }.')
    prompt = "\n".join(prompt_lines)
    prompt += "\n\nMalformed text:\n"

    for i in range(3): # Up to 3 attempts because we live in a world where determinism is dead
        try:
            repaired = query_llm_single(
                model,
                text,
                prompt=prompt,
                temperature=temperature,
                response_format={"type": "json_object"},
                reasoning=reasoning,
            )
            #repaired = clean_json_text(repaired)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                # Last attempt: clean and try again
                cleaned = clean_json_text(repaired)

                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    parsed = {"parsing_error": "Cannot parse"}

            if "parsing_error" in parsed:
                logger.warning(f"LLM JSON repair attempt {i+1} failed: {parsed.get('reason', 'no reason provided')}")
            else:
                return parsed

        except Exception:
            logger.exception("Error repairing JSON with model")

    return None


def safe_load_json(text: str, schema: Optional[Dict[str, Any]] = None, strict: bool = False) -> Optional[Dict]:
    """
    Attempt to parse JSON with progressive fallback strategies.

    Args:
        text: The text to parse as JSON
        schema: Optional JSON Schema for repair
        strict: If True, only allow direct JSON parsing (no cleaning or repair)

    Returns:
        Parsed JSON dict, or None if parsing failed
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if strict:
            logger.warning("Strict mode: Failed to parse JSON directly, not attempting fallback")
            return None

    # Fallback 1: Try cleaning common issues
    cleaned = clean_json_text(text)
    try:
        parsed = json.loads(cleaned)
        logger.debug("JSON parsed successfully after cleaning")
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback 2: Use LLM to repair
    logger.info("Attempting LLM-based JSON repair")
    repaired = _repair_with_model(text, schema)
    if repaired:
        logger.info("JSON repaired successfully with LLM")
    else:
        logger.warning("Failed to repair JSON even with LLM assistance")
    return repaired
