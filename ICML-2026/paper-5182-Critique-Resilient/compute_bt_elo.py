import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from constants import (
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
)
from data_models import (
    AutomatedEvaluation,
    BenchmarkEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_evaluation_entries,
)
from model_config import load_registry
from utils import (
    collect_invalid_self_answer_questions,
    collect_self_answer_adjudications,
    format_key,
    is_latest_outer_attempt,
    judging_task_key,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
    task_key_from_prefix,
)
from victory import VictorySide, resolve_automated_victory


def final_question(entry: BenchmarkEntry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def _task_key(
    prefix: str, run_id: Optional[str], outer_attempt: Optional[str], _topic_slug: Optional[str], _question: Optional[str]
):
    return task_key_from_prefix(prefix, run_id, outer_attempt)


def collect_decisions(auto_eval_dir: Path) -> Dict[Tuple, List[AutomatedEvaluation]]:
    decisions_by_claim: Dict[Tuple, List[AutomatedEvaluation]] = defaultdict(list)
    if not auto_eval_dir.exists():
        return decisions_by_claim
    for eval_file in auto_eval_dir.glob("*.json"):
        data = load_evaluation_entries(eval_file)
        for decision in data.decisions:
            key = judging_task_key(decision) if decision else None
            if key:
                decisions_by_claim[key].append(decision)
    return decisions_by_claim


QuestionKey = Tuple[Optional[str], Optional[str], Optional[str]]
CritiqueKey = Tuple[str, str, str, Union[int, QuestionKey]]


def load_critique_verdicts(
    critiques_dir: Path,
    *,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
    invalid_questions: Optional[Set[QuestionKey]] = None,
) -> Dict[CritiqueKey, Dict[str, Dict[str, Optional[str]]]]:
    verdicts: Dict[CritiqueKey, Dict[str, Dict[str, Optional[str]]]] = defaultdict(dict)
    if not critiques_dir.exists():
        return verdicts
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else {}
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__", 1)
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                entries = load_critique_entries(crit_file)
                for idx, entry in enumerate(entries):
                    if not entry or entry.status != STATUS_SUCCEEDED:
                        continue
                    outer_attempt = normalize_outer_attempt(entry.outer_attempt)
                    if entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(entry.run_id, outer_attempt, latest_by_run):
                            continue
                    attempts = entry.attempts or []
                    if not attempts:
                        continue
                    verdict = attempts[-1].verdict
                    if not verdict:
                        continue
                    info = {
                        "verdict": verdict,
                        "run_id": entry.run_id,
                        "outer_attempt": outer_attempt,
                        "topic_slug": entry.topic_slug,
                        "question": entry.question,
                    }
                    q_key = question_key(entry.question_author, entry.run_id, outer_attempt)
                    if invalid_questions and q_key in invalid_questions:
                        continue
                    if q_key:
                        verdicts[(q_slug, critic_slug, answer_slug, q_key)][mode] = info
                    verdicts[(q_slug, critic_slug, answer_slug, idx)][mode] = info
    return verdicts


def find_critique_verdict(
    verdicts: Dict[CritiqueKey, Dict[str, Dict[str, Optional[str]]]],
    q_slug: str,
    critic_slug: str,
    answer_slug: str,
    idx: int,
    q_key: Optional[QuestionKey],
    preferred_mode: Optional[str],
    fallback_any: bool,
) -> Tuple[Optional[str], Optional[Dict[str, Optional[str]]]]:
    modes = {}
    if q_key is not None:
        modes = verdicts.get((q_slug, critic_slug, answer_slug, q_key), {})
    if not modes:
        modes = verdicts.get((q_slug, critic_slug, answer_slug, idx), {})
    if preferred_mode and preferred_mode in modes:
        return preferred_mode, modes[preferred_mode]
    if not fallback_any:
        return None, None
    if not modes:
        return None, None
    mode = sorted(modes.keys())[0]
    return mode, modes[mode]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def log1pexp(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def fit_bipartite_bt(
    pairs: List[Tuple[int, int, int, int]],
    num_answerers: int,
    num_questioners: int,
    max_iter: int,
    lr: float,
    tol: float,
    l2: float,
) -> Tuple[List[float], List[float]]:
    theta = [0.0 for _ in range(num_answerers)]
    psi = [0.0 for _ in range(num_questioners)]
    last_ll = None
    for step_idx in range(max_iter):
        grad_theta = [0.0 for _ in range(num_answerers)]
        grad_psi = [0.0 for _ in range(num_questioners)]
        ll = 0.0
        for a_idx, q_idx, wins, losses in pairs:
            total = wins + losses
            if total <= 0:
                continue
            diff = theta[a_idx] - psi[q_idx]
            p = sigmoid(diff)
            ll += wins * diff - total * log1pexp(diff)
            grad = wins - total * p
            grad_theta[a_idx] += grad
            grad_psi[q_idx] -= grad
        for i in range(num_answerers):
            grad_theta[i] -= l2 * theta[i]
        for j in range(num_questioners):
            grad_psi[j] -= l2 * psi[j]
        scale = lr / math.sqrt(step_idx + 1.0)
        max_step = 0.0
        for i in range(num_answerers):
            delta = scale * grad_theta[i]
            theta[i] += delta
            max_step = max(max_step, abs(delta))
        for j in range(num_questioners):
            delta = scale * grad_psi[j]
            psi[j] += delta
            max_step = max(max_step, abs(delta))
        if num_questioners:
            shift = sum(psi) / num_questioners
            if shift:
                psi = [v - shift for v in psi]
                theta = [v - shift for v in theta]
        if last_ll is not None and abs(ll - last_ll) < tol and max_step < tol:
            break
        last_ll = ll
    return theta, psi


def resolve_model(registry, name_or_slug: Optional[str]) -> Optional[str]:
    if not name_or_slug:
        return None
    if registry is None:
        return name_or_slug
    return registry.resolve_model_name(name_or_slug)


def collect_games(
    benchmarks_dir: Path,
    answers_dir: Path,
    critiques_dir: Path,
    auto_eval_dir: Path,
    registry_path: Optional[Path],
    answer_critique_mode: str,
    self_answer_critique_mode: str,
    fallback_any_mode: bool,
    human_eval_dir: Optional[Path] = None,
    log_automated_disagreements: bool = True,
) -> Tuple[List[Tuple[str, str, int]], Counter]:
    registry = load_registry(str(registry_path)) if registry_path and registry_path.exists() else None
    if human_eval_dir is None:
        human_eval_dir = Path("evaluations")
    latest_by_question: Dict[str, Dict[str, int]] = {}
    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        entries = load_benchmark_entries(bench_path)
        latest_by_question[q_slug] = latest_outer_attempt_by_run(entries)
    invalid_questions = collect_invalid_self_answer_questions(
        critiques_dir,
        auto_eval_dir,
        human_eval_dir,
        registry,
        log_automated_disagreements=log_automated_disagreements,
    )
    self_answer_outcomes = collect_self_answer_adjudications(
        critiques_dir,
        auto_eval_dir,
        human_eval_dir,
        registry,
        log_automated_disagreements=log_automated_disagreements,
    )
    critique_verdicts = load_critique_verdicts(
        critiques_dir,
        latest_by_question=latest_by_question,
        invalid_questions=invalid_questions,
    )
    decisions_by_claim = collect_decisions(auto_eval_dir)
    skip_counts: Counter = Counter()
    games: List[Tuple[str, str, int]] = []

    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        benchmarks = load_benchmark_entries(bench_path)
        latest_by_run = latest_by_question.get(q_slug, {})
        q_name = resolve_model(registry, q_slug) or q_slug
        answers_root = answers_dir / q_slug
        if not answers_root.exists():
            continue
        for answer_file in answers_root.glob("*.json"):
            a_slug = answer_file.stem
            answers = load_answer_entries(answer_file)
            answers_by_key: Dict[Tuple[Optional[str], Optional[str], Optional[str]], Tuple[int, AnswerEntry]] = {}
            for idx, answer_entry in enumerate(answers):
                if not answer_entry:
                    continue
                outer_attempt = normalize_outer_attempt(answer_entry.outer_attempt)
                if answer_entry.run_id is not None and latest_by_run:
                    if not is_latest_outer_attempt(answer_entry.run_id, outer_attempt, latest_by_run):
                        continue
                answer_key_value = question_key(
                    answer_entry.question_model or q_slug,
                    answer_entry.run_id,
                    outer_attempt,
                )
                if not answer_key_value:
                    skip_counts["answer_missing_key"] += 1
                    continue
                if answer_key_value in invalid_questions:
                    skip_counts["question_invalid_self_answer"] += 1
                    continue
                prior = answers_by_key.get(answer_key_value)
                if not prior or (
                    prior[1].status != STATUS_SUCCEEDED
                    and answer_entry.status == STATUS_SUCCEEDED
                ):
                    answers_by_key[answer_key_value] = (idx, answer_entry)

            for idx, bench_entry in enumerate(benchmarks):
                if not bench_entry or bench_entry.status != STATUS_SUCCEEDED:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                outer_attempt = normalize_outer_attempt(bench_entry.outer_attempt)
                if bench_entry.run_id is not None and latest_by_run:
                    if not is_latest_outer_attempt(bench_entry.run_id, outer_attempt, latest_by_run):
                        continue
                question_text = final_question(bench_entry)
                if not question_text:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                bench_key = question_key(q_slug, bench_entry.run_id, outer_attempt)
                if not bench_key:
                    skip_counts["question_missing_key"] += 1
                    continue
                if bench_key in invalid_questions:
                    skip_counts["question_invalid_self_answer"] += 1
                    continue
                answer_match = answers_by_key.get(bench_key)
                if not answer_match:
                    skip_counts["answer_missing"] += 1
                    continue
                answer_idx, answer_entry = answer_match

                answer_name = (
                    resolve_model(registry, answer_entry.answer_model)
                    or resolve_model(registry, a_slug)
                    or a_slug
                )

                self_mode, self_info = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    a_slug,
                    q_slug,
                    answer_idx,
                    question_key(
                        answer_entry.question_model or q_slug,
                        answer_entry.run_id,
                        outer_attempt,
                    ),
                    self_answer_critique_mode,
                    fallback_any_mode,
                )
                if self_info:
                    self_verdict = self_info.get("verdict")
                    if self_verdict == CRITIQUE_VERDICT_CORRECT:
                        pass
                    elif self_verdict == CRITIQUE_VERDICT_UNKNOWN:
                        skip_counts["self_answer_unknown"] += 1
                        continue
                    else:
                        q_key = question_key(q_slug, self_info.get("run_id"), self_info.get("outer_attempt"))
                        outcome = self_answer_outcomes.get(q_key)
                        if outcome == VictorySide.ALICE:
                            skip_counts["self_answer_invalid"] += 1
                            continue
                        if outcome in {None, VictorySide.DROP}:
                            skip_counts["self_answer_no_majority"] += 1
                            continue

                if answer_entry.status == STATUS_FAILED:
                    games.append((answer_name, q_name, 0))
                    continue

                if answer_entry.status == STATUS_ILL_POSED:
                    prefix = f"illposed/{q_slug}/{a_slug}"
                    claim_key = _task_key(
                        prefix,
                        answer_entry.run_id,
                        answer_entry.outer_attempt,
                        answer_entry.topic_slug,
                        answer_entry.question,
                    )
                    outcome = resolve_automated_victory(
                        "illposed",
                        decisions_by_claim.get(claim_key, []),
                        context=format_key(claim_key or ()),
                        log_automated_disagreements=log_automated_disagreements,
                    )
                    if outcome == VictorySide.ALICE:
                        skip_counts["illposed_validated"] += 1
                        continue
                    if outcome == VictorySide.BOB:
                        games.append((answer_name, q_name, 0))
                        continue
                    skip_counts["illposed_no_majority"] += 1
                    continue

                if answer_entry.status != STATUS_SUCCEEDED:
                    skip_counts["answer_invalid_status"] += 1
                    continue

                mode, verdict_info = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    q_slug,
                    a_slug,
                    answer_idx,
                    question_key(
                        answer_entry.question_model or q_slug,
                        answer_entry.run_id,
                        outer_attempt,
                    ),
                    answer_critique_mode,
                    fallback_any_mode,
                )
                if not verdict_info:
                    skip_counts["critique_missing"] += 1
                    continue
                verdict = verdict_info.get("verdict")
                if verdict == CRITIQUE_VERDICT_UNKNOWN:
                    skip_counts["critique_unknown"] += 1
                    continue
                if verdict == CRITIQUE_VERDICT_CORRECT:
                    games.append((answer_name, q_name, 1))
                    continue

                prefix = f"critique/{mode}/{q_slug}/{q_slug}__{a_slug}"
                claim_key = _task_key(
                    prefix,
                    verdict_info.get("run_id"),
                    verdict_info.get("outer_attempt"),
                    verdict_info.get("topic_slug"),
                    verdict_info.get("question"),
                )
                outcome = resolve_automated_victory(
                    "critique",
                    decisions_by_claim.get(claim_key, []),
                    context=format_key(claim_key or ()),
                    log_automated_disagreements=log_automated_disagreements,
                )
                if outcome == VictorySide.BOB:
                    games.append((answer_name, q_name, 1))
                    continue
                if outcome == VictorySide.ALICE:
                    games.append((answer_name, q_name, 0))
                    continue
                skip_counts["critique_no_majority"] += 1

    return games, skip_counts


def build_pair_counts(games: List[Tuple[str, str, int]]) -> Dict[Tuple[str, str], List[int]]:
    counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for answerer, questioner, outcome in games:
        wins, losses = counts[(answerer, questioner)]
        if outcome == 1:
            wins += 1
        else:
            losses += 1
        counts[(answerer, questioner)] = [wins, losses]
    return counts


def render_ratings(
    role: str,
    names: List[str],
    logit_scores: Dict[str, float],
    wins: Dict[str, int],
    losses: Dict[str, int],
    base_elo: float,
    min_games: int,
) -> List[Tuple[str, float, float, int, int, int]]:
    scale = 400.0 / math.log(10.0)
    rows = []
    for name in names:
        w = wins.get(name, 0)
        l = losses.get(name, 0)
        total = w + l
        if total < min_games:
            continue
        logit = logit_scores.get(name, 0.0)
        elo = base_elo + scale * logit
        rows.append((name, elo, logit, w, l, total))
    rows.sort(key=lambda item: item[1], reverse=True)
    print(f"\n{role}:")
    for name, elo, _, w, l, total in rows:
        print(f"{elo:8.1f}  {w:4d}-{l:<4d}  ({total:4d})  {name}")
    return rows


def write_csv(path: Path, rows: List[Tuple[str, float, float, int, int, int]], role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["role", "model", "elo", "logit", "wins", "losses", "games"])
        for name, elo, logit, wins, losses, total in rows:
            writer.writerow([role, name, f"{elo:.2f}", f"{logit:.6f}", wins, losses, total])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute bipartite Bradley-Terry Elo ratings from benchmark results."
    )
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--human-evals-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--answer-critique-mode", type=str, default="evaluator")
    parser.add_argument("--self-answer-critique-mode", type=str, default="contradictor")
    parser.add_argument("--fallback-any-mode", action="store_true")
    parser.add_argument("--disable-disagreement-logs", action="store_true")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--base-elo", type=float, default=1000.0)
    parser.add_argument("--min-games", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    games, skips = collect_games(
        args.benchmarks_dir,
        args.answers_dir,
        args.critiques_dir,
        args.automated_dir,
        args.config,
        args.answer_critique_mode,
        args.self_answer_critique_mode,
        args.fallback_any_mode,
        args.human_evals_dir,
        log_automated_disagreements=not args.disable_disagreement_logs,
    )

    if not games:
        print("No valid games found after filtering.")
        if skips:
            print("Skipped:")
            for reason, count in sorted(skips.items()):
                print(f"  {reason}: {count}")
        return 1

    pair_counts = build_pair_counts(games)
    answerers = sorted({pair[0] for pair in pair_counts})
    questioners = sorted({pair[1] for pair in pair_counts})
    answerer_index = {name: idx for idx, name in enumerate(answerers)}
    questioner_index = {name: idx for idx, name in enumerate(questioners)}

    pairs = []
    for (answerer, questioner), (wins, losses) in pair_counts.items():
        pairs.append((answerer_index[answerer], questioner_index[questioner], wins, losses))

    theta, psi = fit_bipartite_bt(
        pairs,
        len(answerers),
        len(questioners),
        args.max_iter,
        args.lr,
        args.tol,
        args.l2,
    )

    answerer_scores = {name: theta[idx] for name, idx in answerer_index.items()}
    questioner_scores = {name: psi[idx] for name, idx in questioner_index.items()}

    answerer_wins = Counter()
    answerer_losses = Counter()
    questioner_wins = Counter()
    questioner_losses = Counter()
    for (answerer, questioner), (wins, losses) in pair_counts.items():
        answerer_wins[answerer] += wins
        answerer_losses[answerer] += losses
        questioner_wins[questioner] += losses
        questioner_losses[questioner] += wins

    total_games = sum(w + l for w, l in pair_counts.values())
    print(f"Games: {total_games}")
    if skips:
        print("Skipped:")
        for reason, count in sorted(skips.items()):
            print(f"  {reason}: {count}")

    answerer_rows = render_ratings(
        "Answerer Elo (Bob)",
        answerers,
        answerer_scores,
        answerer_wins,
        answerer_losses,
        args.base_elo,
        args.min_games,
    )
    questioner_rows = render_ratings(
        "Questioner Elo (Alice)",
        questioners,
        questioner_scores,
        questioner_wins,
        questioner_losses,
        args.base_elo,
        args.min_games,
    )

    if args.output_csv:
        write_csv(args.output_csv, answerer_rows, "answerer")
        write_csv(args.output_csv, questioner_rows, "questioner")
        print(f"\nWrote CSV to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
