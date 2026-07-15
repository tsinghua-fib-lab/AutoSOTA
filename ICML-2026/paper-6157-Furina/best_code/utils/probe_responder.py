#!/usr/bin/env python3
"""
Probe Responder — Probe Question Answering Engine
Answers the generated probe questions.

Input: `phase_probe_results/xxx_probes.json` from `phase_probe_generator.py`
Process: send each task's `final_probe_questions` to the target LLM in order
Output:
  - `probe_responses/{base}_all_responses.json` as the single aggregated output file

Features:
  - Sequential task processing to avoid API concurrency issues
  - Resume support from the aggregated output file
  - `--task` support for processing a single task
  - `--view` support for inspecting saved results

For AI Safety Research Only.
"""

import os
import json
import time
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

try:
    from utils.api_client import get_client_for_agent, get_model_for_agent
except ModuleNotFoundError:
    from api_client import get_client_for_agent, get_model_for_agent, backoff_sleep

AGENT_NAME = "probe_responder"

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────

def get_client(model: str):
    return get_client_for_agent(AGENT_NAME, model_override=model)


def llm_call(user_prompt: str,
             model: str = get_model_for_agent(AGENT_NAME),
             temperature: float = 0.3,
    max_retries: int = 10,
    max_tokens: int = 4096) -> str:
    """
    Call the LLM and return the raw text answer without a system prompt.
    """
    client = get_client(model)
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw = response.choices[0].message.content
            if raw is None:
                raise AttributeError("Empty response")
            return raw

        except Exception as e:
            last_exc = e
            print(f"    [Retry {attempt}/{max_retries}] {e}")
            backoff_sleep(attempt)

    raise last_exc



# ─────────────────────────────────────────────────────────────────────────────
# Single-question answering
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(question: str,
                   model: str = get_model_for_agent(AGENT_NAME),
                   question_idx: int = 0) -> dict:
    """
    Submit one question to the LLM and return the answer payload.
    """
    answer = llm_call(
        user_prompt=question,
        model=model,
        temperature=0.3,
        max_tokens=4096
    )

    return {
        "question_index": question_idx,
        "question": question,
        "answer": answer,
        "answer_timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-task processing
# ─────────────────────────────────────────────────────────────────────────────

def process_task(task_id: str,
                 task_data: dict,
                 all_existing: dict,
                 output_path: Path,
                 model: str = get_model_for_agent(AGENT_NAME),
                 verbose: bool = True) -> dict:
    """
    Process one task by answering all `final_probe_questions` in sequence.
    Resume behavior is driven by `all_existing`, and the aggregated file is
    updated atomically after the task finishes.
    """
    original_request = task_data.get("original_request", "")
    questions = task_data.get("final_probe_questions", [])
    verdict   = task_data.get("verdict", "?")
    opt_summary = task_data.get("optimization_summary", {})

    total = len(questions)
    if total == 0:
        if verbose:
            print(f"  ⚠ Task {task_id}: no questions found, skipping")
        return {
            "task_id": str(task_id),
            "original_request": original_request,
            "verdict": verdict,
            "optimization_summary": opt_summary,
            "questions_answered": [],
            "skipped_reason": "no_questions"
        }

    # Resume by restoring previously answered questions from the aggregate file.
    existing_record = all_existing.get(str(task_id), {})
    existing_answers = existing_record.get("questions_answered", [])
    answered_indices = {
        item["question_index"]: item
        for item in existing_answers
    }

    answered_count = len(answered_indices)
    remaining = total - answered_count

    if verbose:
        print(f"\n[Task {task_id}] {original_request[:60]}...")
        print(f"  → {total} questions, answered {answered_count}/{total}, remaining {remaining}")

    # Answer the remaining questions in order.
    questions_answered = list(answered_indices.values())

    for i, question in enumerate(questions):
        if i in answered_indices:
            if verbose:
                print(f"    [{i+1}/{total}] already answered, skipping")
            continue

        if verbose:
            q_preview = question[:65].replace("\n", " ")
            print(f"    [{i+1}/{total}] answering: {q_preview}...")

        try:
            result = answer_question(
                question=question,
                model=model,
                question_idx=i
            )
            questions_answered.append(result)

            if verbose:
                ans_preview = result["answer"][:60].replace("\n", " ")
                print(f"      → {ans_preview}...")

            time.sleep(0.5)

        except Exception as e:
            if verbose:
                print(f"      ✗ failed: {e}")
            continue

    # Keep the stored answers sorted by question index.
    questions_answered.sort(key=lambda x: x["question_index"])

    task_record = {
        "task_id": str(task_id),
        "original_request": original_request,
        "verdict": verdict,
        "optimization_summary": opt_summary,
        "questions_answered": questions_answered,
        "last_updated": datetime.now().isoformat(),
    }

    if verbose:
        print(f"  ✓ Task {task_id} completed: {len(questions_answered)}/{total} answers")

    return task_record


# ─────────────────────────────────────────────────────────────────────────────
# Batch processing and aggregate writes
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(input_path: str,
                  output_dir: str = "Results/probe_responses",
                  model: str = get_model_for_agent(AGENT_NAME),
                  verbose: bool = True):
    """
    Process all tasks and answer all probe questions.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    base = Path(input_path).stem
    all_responses_path = Path(output_dir) / f"{base}_all_responses.json"

    # Load the existing aggregate output for resume support.
    all_existing = {}
    if all_responses_path.exists():
        try:
            with open(all_responses_path, "r", encoding="utf-8") as f:
                all_existing = json.load(f)
        except Exception:
            all_existing = {}

    task_ids = [str(k) for k in tasks.keys() if k != "_meta"]
    total = len(task_ids)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Probe Responder — Probe Question Answering Engine")
        print(f"  Input: {input_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Total tasks: {total} | Model: {model}")
        print(f"{'='*60}")

    completed = 0

    for i, task_id in enumerate(task_ids, 1):
        task_data = tasks[task_id]
        target_count = len(task_data.get("final_probe_questions", []))
        existing_record = all_existing.get(str(task_id), {})
        existing_count = len(existing_record.get("questions_answered", []))
        is_done = (existing_count >= target_count and target_count > 0)

        if is_done:
            if verbose:
                print(f"\n[{i}/{total}] Task {task_id} already completed, skipping")
            continue

        print(f"\n[{i}/{total}] Processing Task {task_id}...")

        result = process_task(
            task_id=str(task_id),
            task_data=task_data,
            all_existing=all_existing,
            output_path=all_responses_path,
            model=model,
            verbose=verbose
        )

        all_existing[str(task_id)] = result

        # Atomically persist the aggregate file after each completed task.
        tmp = all_responses_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_existing, f, ensure_ascii=False, indent=2)
        tmp.replace(all_responses_path)

        completed += 1
        print(f"  Aggregate output updated → {all_responses_path}")

    # Final metadata
    all_existing["_meta"] = {
        "input_path": input_path,
        "output_dir": output_dir,
        "total_tasks": total,
        "completed_tasks": completed,
        "model": model,
        "timestamp": datetime.now().isoformat(),
    }

    with open(all_responses_path, "w", encoding="utf-8") as f:
        json.dump(all_existing, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  All tasks finished")
        print(f"  Aggregate file: {all_responses_path}")
        print(f"  Completed this run: {completed}/{total}")
        print(f"{'='*60}")
    return str(all_responses_path)


# ─────────────────────────────────────────────────────────────────────────────
# View saved results
# ─────────────────────────────────────────────────────────────────────────────

def print_task_summary(result: dict, max_question_len: int = 80,
                       max_answer_len: int = 120):
    """Print a short summary of one task's saved Q&A results."""
    task_id = result.get("task_id", "?")
    req = result.get("original_request", "")
    verdict = result.get("verdict", "?")
    qa_list = result.get("questions_answered", [])

    print(f"\n{'─'*70}")
    print(f"  Task {task_id} | verdict={verdict}")
    print(f"  Request: {req[:70]}...")
    print(f"{'─'*70}")

    if not qa_list:
        print("  (no saved answers)")
        return

    for item in qa_list:
        idx = item.get("question_index", "?")
        q = item.get("question", "")
        a = item.get("answer", "")

        q_disp = q[:max_question_len].replace("\n", " ")
        if len(q) > max_question_len:
            q_disp += "..."

        a_disp = a[:max_answer_len].replace("\n", " ")
        if len(a) > max_answer_len:
            a_disp += "..."

        print(f"\n  Q{idx+1}: {q_disp}")
        print(f"  A{idx+1}: {a_disp}")

    print(f"\n  Total questions: {len(qa_list)}")


def view_results(input_path: str, task_id: str = None):
    """Inspect saved answer results."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if task_id:
        if task_id in data:
            print_task_summary(data[task_id])
        else:
            print(f"Task {task_id} does not exist")
    else:
        for k, v in data.items():
            if k.startswith("_"):
                continue
            print_task_summary(v)
