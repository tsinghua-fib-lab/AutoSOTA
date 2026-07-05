import argparse
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from model_config import load_registry
from constants import CRITIQUE_VERDICT_CORRECT
from data_models import (
    AnswerEntry,
    HumanEvaluation,
    HumanEvaluationFile,
    JudgingTask,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_human_evaluation_entries,
    save_human_evaluation_entries,
)
from utils import (
    answer_key,
    answer_key_from_entry,
    benchmark_answers_from_entries,
    collect_invalid_questions,
    format_key,
    human_evaluation_key_from_entry,
    is_latest_outer_attempt,
    judging_task_key,
    latest_outer_attempt_by_run,
    normalize_outer_attempt,
    question_key,
)


def final_answer(entry: AnswerEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].answer


def build_illposed_claim(entry: Optional[AnswerEntry]) -> str:
    if not entry:
        return ""
    attempts = entry.attempts or []
    if not attempts:
        return ""
    evaluation = attempts[-1].evaluation or {}
    issues = evaluation.get("issues")
    if not issues:
        return ""
    if isinstance(issues, str):
        issue_list = [issues]
    else:
        try:
            issue_list = [str(issue) for issue in issues if issue]
        except TypeError:
            issue_list = [str(issues)]
    return "\n".join(f"- {issue}" for issue in issue_list)


def benchmark_answer_map(benchmark_dir: Path, q_slug: str) -> Dict:
    bench_path = benchmark_dir / f"{q_slug}.json"
    entries = load_benchmark_entries(bench_path)
    answers = benchmark_answers_from_entries(q_slug, entries)
    mapping: Dict = {}
    for entry in answers:
        if not entry:
            continue
        key = question_key(q_slug, entry.run_id, entry.outer_attempt)
        if key and key not in mapping:
            mapping[key] = final_answer(entry) or ""
    return mapping


def gather_illposed(
    debates_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
    invalid_questions: Optional[set] = None,
) -> List[JudgingTask]:
    items = []
    for debate_file in debates_dir.glob("illposed/*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        question_model = q_slug
        answer_model = a_slug
        debates = load_debate_entries(debate_file)
        answers = load_answer_entries(answers_dir / q_slug / f"{a_slug}.json")
        benchmark_answers = benchmark_answer_map(benchmark_dir, q_slug)
        debate_map: Dict = {}
        for debate in debates:
            if not debate:
                continue
            key = answer_key(q_slug, a_slug, debate.run_id, debate.outer_attempt)
            if key and key not in debate_map:
                debate_map[key] = debate
        answer_map: Dict = {}
        for answer in answers:
            if not answer:
                continue
            key = answer_key_from_entry(answer)
            if key and key not in answer_map:
                answer_map[key] = answer
        keys = set(debate_map) | set(answer_map)
        latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else {}
        for key in keys:
            run_id, _q_model, _a_model, outer_attempt = key
            attempt = normalize_outer_attempt(outer_attempt)
            if run_id is not None and latest_by_run:
                if not is_latest_outer_attempt(run_id, attempt, latest_by_run):
                    continue
            q_key = question_key(q_slug, run_id, outer_attempt)
            if invalid_questions and q_key in invalid_questions:
                continue
            debate = debate_map.get(key)
            question = debate.question if debate else ""
            answer_record = answer_map.get(key)
            answer_text = final_answer(answer_record) if answer_record else ""
            if not answer_text:
            answer_text = benchmark_answers.get(question_key(q_slug, key[0], key[3]), "")
            run_id = (debate.run_id if debate else None) or (answer_record.run_id if answer_record else None)
            outer_attempt = (debate.outer_attempt if debate else None) or (
                answer_record.outer_attempt if answer_record else None
            )
            topic_slug = (debate.topic_slug if debate else None) or (answer_record.topic_slug if answer_record else None)
            claim_text = build_illposed_claim(answer_record)
            items.append(
                JudgingTask(
                    type="illposed",
                    question_model=question_model,
                    answer_model=answer_model,
                    critic_model=None,
                    run_id=run_id,
                    outer_attempt=outer_attempt,
                    topic_slug=topic_slug,
                    question=question or (answer_record.question if answer_record else ""),
                    answer=answer_text,
                    critique=claim_text,
                    debate_history=debate.history if debate else [],
                )
            )
    return items


def gather_critiques(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    latest_by_question: Optional[Dict[str, Dict[str, int]]] = None,
    invalid_questions: Optional[set] = None,
) -> List[JudgingTask]:
    items = []
    for crit_mode_dir in critiques_dir.glob("*"):
        mode = crit_mode_dir.name
        for q_dir in crit_mode_dir.glob("*"):
            q_slug = q_dir.name
            latest_by_run = latest_by_question.get(q_slug, {}) if latest_by_question else {}
            question_model = q_slug
            for crit_file in q_dir.glob("*.json"):
                critic_slug, answer_slug = crit_file.stem.split("__")
                critic_model = critic_slug
                answer_model = answer_slug

                critiques = load_critique_entries(crit_file)
                answers = load_answer_entries(answers_dir / q_slug / f"{answer_slug}.json")
                debates = load_debate_entries(debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json")
                benchmark_answers = benchmark_answer_map(benchmark_dir, q_slug)
                debate_map: Dict = {}
                for debate in debates:
                    if not debate:
                        continue
                    key = answer_key(q_slug, answer_slug, debate.run_id, debate.outer_attempt)
                    if key and key not in debate_map:
                        debate_map[key] = debate
                answer_map: Dict = {}
                for answer in answers:
                    if not answer:
                        continue
                    key = answer_key_from_entry(answer)
                    if key and key not in answer_map:
                        answer_map[key] = answer

                for critique_entry in critiques:
                    if not critique_entry:
                        continue
                    outer_attempt = normalize_outer_attempt(critique_entry.outer_attempt)
                    if critique_entry.run_id is not None and latest_by_run:
                        if not is_latest_outer_attempt(critique_entry.run_id, outer_attempt, latest_by_run):
                            continue
                    q_key = question_key(q_slug, critique_entry.run_id, outer_attempt)
                    if invalid_questions and q_key in invalid_questions:
                        continue
                    critique_attempts = critique_entry.attempts or []
                    if critique_attempts and critique_attempts[-1].verdict == CRITIQUE_VERDICT_CORRECT:
                        continue
                    key = answer_key(
                        critique_entry.question_author,
                        critique_entry.answer_author,
                        critique_entry.run_id,
                        critique_entry.outer_attempt,
                    )
                    debate = debate_map.get(key)
                    question = (debate.question if debate else "") or critique_entry.question
                    answer_record = answer_map.get(key)
                    answer_text = final_answer(answer_record) if answer_record else ""
                    if not answer_text:
                        answer_text = benchmark_answers.get(question_key(q_slug, key[0], key[3]), "")
                    critique_text = critique_attempts[-1].raw_critique if critique_attempts else ""
                    run_id = (debate.run_id if debate else None) or critique_entry.run_id
                    outer_attempt = (debate.outer_attempt if debate else None) or critique_entry.outer_attempt
                    topic_slug = (debate.topic_slug if debate else None) or critique_entry.topic_slug
                    items.append(
                        JudgingTask(
                            type="critique",
                            mode=mode,
                            question_model=question_model,
                            answer_model=answer_model,
                            critic_model=critic_model,
                            run_id=run_id,
                            outer_attempt=outer_attempt,
                            topic_slug=topic_slug,
                            question=question,
                            answer=answer_text,
                            critique=critique_text,
                            debate_history=debate.history if debate else [],
                        )
                    )
    return items


def load_evaluations(path: Path) -> Dict:
    payload = load_human_evaluation_entries(path)
    decisions: Dict = {}
    for decision in payload.decisions:
        key = human_evaluation_key_from_entry(decision)
        if key:
            decisions[key] = decision
    return decisions


def save_evaluation(path: Path, decisions: Dict[str, HumanEvaluation]):
    save_human_evaluation_entries(path, HumanEvaluationFile(decisions=list(decisions.values())))


def render_task(task: JudgingTask, existing: Optional[HumanEvaluation], save_cb):
    task_key = judging_task_key(task) or ()
    task_key_str = format_key(task_key)
    st.markdown(f"### {task_key_str}")
    st.write(f"**Question model:** {task.question_model}  |  **Answer model:** {task.answer_model}  |  **Critic:** {task.critic_model or ''}")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Question")
        st.write(task.question or "")
        st.subheader("Answer")
        st.write(task.answer or "")
    with cols[1]:
        if task.type in ("critique", "illposed"):
            st.subheader("Critique")
            st.write(task.critique or "")
        st.subheader("Debate")
        for msg in task.debate_history or []:
            st.markdown(f"- **{msg.speaker}**: {msg.message}")

    st.divider()
    if task.type == "illposed":
        options = [
            "ill-posed",
            "not ill-posed",
            "ill-posed but wrong reason",
            "unknown",
            "invalid",
        ]
    else:
        options = [
            "incorrect",
            "correct",
            "incorrect but wrong reason",
            "unknown",
            "invalid",
        ]

    existing_choice = existing.verdict if existing else None
    choice = st.radio(
        "Verdict",
        options,
        index=options.index(existing_choice) if existing_choice in options else 0,
        key=f"verdict-{task_key_str}",
    )
    confidence = st.slider(
        "Confidence (1-5)",
        1,
        5,
        int(existing.confidence if existing and existing.confidence is not None else 3),
        1,
        key=f"conf-{task_key_str}",
    )
    comment = st.text_area("Comments", value=existing.comment if existing else "", key=f"comment-{task_key_str}")
    if st.button("Save", key=f"save-{task_key_str}"):
        save_cb(
            HumanEvaluation(
                run_id=task.run_id,
                outer_attempt=task.outer_attempt,
                type=task.type,
                mode=task.mode,
                question_model=task.question_model,
                answer_model=task.answer_model,
                critic_model=task.critic_model,
                verdict=choice,
                confidence=confidence,
                comment=comment,
            )
        )
        st.success("Saved")


def main():
    parser = argparse.ArgumentParser(description="Interactive web labeller")
    parser.add_argument("--username", required=True)
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--automated-evals-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    args, _ = parser.parse_known_args()

    registry = load_registry(str(args.config))

    latest_by_question = {}
    for bench_path in args.benchmark_dir.glob("*.json"):
        q_slug = bench_path.stem
        entries = load_benchmark_entries(bench_path)
        latest_by_question[q_slug] = latest_outer_attempt_by_run(entries)
    invalid_questions = collect_invalid_questions(
        args.critiques_dir,
        args.answers_dir,
        args.automated_evals_dir,
        args.evaluations_dir,
        registry,
        log_automated_disagreements=False,
    )

    illposed = gather_illposed(
        args.debates_dir,
        args.answers_dir,
        args.benchmark_dir,
        registry,
        latest_by_question,
        invalid_questions,
    )
    critiques = gather_critiques(
        args.debates_dir,
        args.critiques_dir,
        args.answers_dir,
        args.benchmark_dir,
        registry,
        latest_by_question,
        invalid_questions,
    )
    tasks = illposed + critiques

    eval_path = args.evaluations_dir / f"{args.username}.json"
    decisions = load_evaluations(eval_path)

    st.title(f"Labeller - {args.username}")

    claim_filter = st.multiselect(
        "Claim type",
        ["illposed", "contradictor", "evaluator"],
        default=["illposed", "contradictor", "evaluator"],
    )
    questioners = sorted({t.question_model for t in tasks if t.question_model})
    answerers = sorted({t.answer_model for t in tasks if t.answer_model})
    critics = sorted({t.critic_model for t in tasks if t.critic_model})
    selected_q = st.multiselect("Question model", questioners, default=questioners)
    selected_a = st.multiselect("Answer model", answerers, default=answerers)
    selected_c = st.multiselect("Critic", critics, default=critics)
    only_unlabeled = st.checkbox("Only not yet labelled by me", value=True)

    filtered = []
    for task in tasks:
        task_key = judging_task_key(task)
        if not task_key:
            continue
        task_key_str = format_key(task_key)
        task_kind = task.type if task.type == "illposed" else task.mode or "critique"
        if task_kind not in claim_filter:
            continue
        if task.question_model not in selected_q:
            continue
        if task.answer_model not in selected_a:
            continue
        if task.critic_model and selected_c and task.critic_model not in selected_c:
            continue
        already = decisions.get(task_key)
        if only_unlabeled and already:
            continue
        filtered.append((task, already, task_key, task_key_str))

    st.subheader(f"Tasks ({len(filtered)})")

    def save_decision(decision: HumanEvaluation):
        key = human_evaluation_key_from_entry(decision)
        if key:
            decisions[key] = decision
        save_evaluation(eval_path, decisions)

    selected_task_key = st.session_state.get("selected_task_key")

    if not selected_task_key:
        for task, existing, task_key, task_key_str in filtered:
            claimant = task.critic_model or task.answer_model
            adversary = task.answer_model if task.type == "critique" else task.question_model
            topic = (task.question or "")[:80] + ("..." if task.question and len(task.question) > 80 else "")
            cols = st.columns([4, 2])
            with cols[0]:
                st.markdown(f"**{task.type}** | claimant: {claimant} | adversary: {adversary} | topic: {topic}")
            with cols[1]:
                if st.button("Open", key=f"open-{task_key_str}"):
                    st.session_state["selected_task_key"] = task_key_str
                    st.rerun()
        if not filtered:
            st.info("No tasks match the current filters.")
    else:
        task, existing = next(((t, e) for t, e, _k, key_str in filtered if key_str == selected_task_key), (None, None))
        if not task:
            st.warning("Selected task not found in current filter. Clear selection.")
        else:
            if st.button("Back to list"):
                st.session_state["selected_task_key"] = None
                st.rerun()
            st.divider()
            render_task(task, existing, save_decision)


if __name__ == "__main__":
    main()
